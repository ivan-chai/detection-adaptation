import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

import pytorch_lightning as pl

from collections import OrderedDict
from contextlib import nullcontext

from ..gradient_reversal_layer import GradientReversalLayer
from ..autoclip import AutoClip
from ...config import prepare_config
from ..detection import Detector, FacesAsPointsLoss
from .discriminator import PixelwiseDiscriminator
from ...metrics import AveragePrecisionCalculator
from .adversarial_background_score_regularization import AdversarialBackgroundScoreRegularizationLoss
from .weak_pseudolabeling import WeakPseudolabeling


DISCRIMINATORS = {
    "pixelwise": PixelwiseDiscriminator
}


class do_nothing:

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class flip_labels:

    def __init__(self, y_true, module_config):
        self.y_true = y_true
        self.module_config = module_config

    def _flip(self):
        for item in self.y_true:
            item["domain_label"] = 1 - item["domain_label"]
        self.module_config["damain_balance"] = 1/self.module_config["domain_balance"]

    def __enter__(self):
        self._flip()

    def __exit__(self, *args):
        self._flip()


class no_grad:
    def __init__(self, module):
        self.module = module
        self.p_states = {}
    
    def __enter__(self):
        for name, p in self.module.named_parameters():
            self.p_states[name] = p.requires_grad
            p.requires_grad = False
            
    def __exit__(self, *args):
        for name, p in self.module.named_parameters():
            p.requires_grad = self.p_states.get(name, p.requires_grad)


class DetectionAdaptationGANModule(pl.LightningModule):

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("detector", None),
            ("detector_loss", None),
            ("discriminator", {
                "type": "pixelwise",
                "config": {}
            }),
            ("discriminator_loss", "mse"),
            ("split_discriminator_batch", True),
            ("detector_lr", 3e-4),
            ("discriminator_lr", 3e-4),
            ("adaptation_factor", 1.0),
            ("discriminator_pretraining_steps", 100),
            ("grad_clip_percentile", 80),
            ("grad_clip_history_size", 100),
            ("source_domain_label", 0),
            ("domain_balance", 1.),
            ("freeze_predictor", False),
            ("detector_batchnorm_momentum", 0.1),
            ("minmax_mode", "labels_flip")
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        self.config = config

        self.detector = Detector(config["detector"])
        self.detector_clipper = AutoClip(config["grad_clip_percentile"])
        self.detection_loss_fn = FacesAsPointsLoss(config["detector_loss"])

        self.discriminator = DISCRIMINATORS[
            config["discriminator"]["type"]
        ](config["discriminator"]["config"])
        self.discriminator_clipper = AutoClip(config["grad_clip_percentile"])

        self.grad_reversal_layer = GradientReversalLayer(1.0)

        self.detector.apply(
            lambda x: self._set_batchnorm_momentum(x, config["detector_batchnorm_momentum"])
        )


    def _freeze_predictor(self):
        self.detector.predictor.eval()
        for p in self.detector.predictor.parameters():
            p.requires_grad = False

    @staticmethod
    def _set_batchnorm_momentum(module, momentum):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                child.momentum = momentum

    def _apply_discriminator(self, y_pred, y_true):
        embedding_t = self.grad_reversal_layer(y_pred["embedding_t"])\
            if self.config["minmax_mode"] == "grad_reversal"\
            else y_pred["embedding_t"]

        if self.config["split_discriminator_batch"]:
            src_domain_label = int(self.config["source_domain_label"])
            is_src_domain = torch.tensor([
                item["domain_label"] == src_domain_label\
                for item in y_true
            ])
            is_tar_domain = torch.logical_not(is_src_domain)

            src_domain_logits = self.discriminator(embedding_t[is_src_domain]).squeeze(1)
            tar_domain_logits = self.discriminator(embedding_t[is_tar_domain]).squeeze(1)

            domain_logits = torch.ones(len(y_true), *src_domain_logits.shape[1:]).to(src_domain_logits.device)
            domain_logits[is_src_domain] = src_domain_logits
            domain_logits[is_tar_domain] = tar_domain_logits
        else:
            domain_logits = self.discriminator(embedding_t).squeeze(1)
        return domain_logits

    def forward(self, X, y_true):
        if self.config["freeze_predictor"]:
            self._freeze_predictor()

        y_pred = self.detector(X)
        ###
        y_pred["embedding_t"] = torch.cat([
            y_pred["scores_t"][:,None,...],
            y_pred["deltas_t"],
            y_pred["sizes_t"]
        ], dim=1)
        ###
        domain_logits = self._apply_discriminator(y_pred, y_true)
        y_pred["domain_logits"] = domain_logits
        return y_pred

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.detector.train()
        self.discriminator.train()

        X, y_true = batch
        if self._is_discriminator_step(batch_idx):
            with no_grad(self.detector):
                y_pred = self(X, y_true)
            loss = self._discriminator_loss(y_pred, y_true).mean()
            self.log("discriminator_train_loss", loss)
            _, optimizer = self.optimizers()
            clip_grad = self._clip_discriminator_grad
        else:
            with no_grad(self.discriminator):
                y_pred = self(X, y_true)
            detection_loss, adaptation_loss =\
                self._detection_and_adaptation_losses(y_pred, y_true)
            detection_loss = detection_loss.mean()
            adaptation_loss = adaptation_loss.mean()
            self.log("detector_train_loss", detection_loss)
            self.log("adaptation_train_loss", adaptation_loss)
            loss = detection_loss + adaptation_loss*self.config["adaptation_factor"]
            optimizer, _ = self.optimizers()
            clip_grad = self._clip_detector_grad

        discriminator_accuracy = self._domain_accuracy(y_pred, y_true)
        self.log("discriminator_train_accuracy", discriminator_accuracy)

        
        optimizer.zero_grad()
        self.manual_backward(loss, optimizer)
        clip_grad()
        optimizer.step()

        return loss

    def _clip_detector_grad(self):
        grad_norm, clipped_grad_norm = self.detector_clipper(self.detector.parameters())
        self.log("detector_grad_norm", grad_norm)
        self.log("detector_clipped_grad_norm", clipped_grad_norm)

    def _clip_discriminator_grad(self):
        grad_norm, clipped_grad_norm = self.discriminator_clipper(self.discriminator.parameters())
        self.log("discriminator_grad_norm", grad_norm)
        self.log("discriminator_clipped_grad_norm", clipped_grad_norm)

    def _is_discriminator_step(self, batch_idx):
        is_pretraining = self.trainer.global_step < int(self.config["discriminator_pretraining_steps"])
        is_even_batch = (batch_idx % 2 == 0)
        return is_pretraining or is_even_batch

    def _domain_accuracy(self, y_pred, y_true):
        domain_logits = y_pred["domain_logits"]
        domain_labels = torch.tensor([
            item["domain_label"] for item in y_true
        ]).to(domain_logits.device)

        domain_logits = domain_logits.flatten(1, -1).mean(dim=-1)
        if self.config["discriminator_loss"] == "mse":
            pred_domain_labels = (domain_logits > 0.5).long()
        if self.config["discriminator_loss"] == "cross_entropy":
            pred_domain_labels = (domain_logits > .0).long()

        accuracy = (pred_domain_labels == domain_labels).float().mean()
        return accuracy

    def _domain_loss(self, y_pred, y_true):
        domain_logits = y_pred["domain_logits"]
        domain_labels = torch.tensor([
            item["domain_label"] for item in y_true
        ]).to(domain_logits.device)

        src_domain_label = int(self.config["source_domain_label"])
        balance = 1/self.config["domain_balance"]
        balancing_weights = torch.ones_like(domain_labels).float()
        balancing_weights[domain_labels == src_domain_label] = balance
        balancing_weights /= (1. + balance)

        domain_labels = domain_labels[:, None, None]

        if self.config["discriminator_loss"] == "mse":
            loss = F.mse_loss(domain_logits, domain_labels.float(), reduction="none")
        if self.config["discriminator_loss"] == "cross_entropy":
            loss = - domain_labels*F.logsigmoid(domain_logits)\
                   - (1 - domain_labels)*F.logsigmoid(-domain_logits)

        loss = loss.flatten(1, -1).mean(dim=-1)
        loss *= balancing_weights

        return loss

    def _discriminator_loss(self, y_pred, y_true):
        loss = self._domain_loss(y_pred, y_true)
        return loss

    def _keep_source(self, y_pred, y_true):
        src_domain_label = int(self.config["source_domain_label"])
        is_src = [
            item["domain_label"] == src_domain_label\
            for item in y_true
        ]
        src_y_true = [
            item for item, item_is_src in zip(y_true, is_src)\
            if item_is_src
        ]
        is_src = torch.tensor(is_src)
        src_y_pred = {}
        for key, value in y_pred.items():
            try:
                new_value = value[is_src]
            except:
                new_value = value
            src_y_pred[key] = new_value

        return src_y_pred, src_y_true

    def _detection_and_adaptation_losses(self, y_pred, y_true):
        detection_loss = self.detection_loss_fn(
            *self._keep_source(y_pred, y_true)
        )

        is_flipping = self.config["minmax_mode"] == "labels_flip"
        context = flip_labels if is_flipping else do_nothing
        with context(y_true, self.config):
            adaptation_loss = self._domain_loss(y_pred, y_true)

        return detection_loss, adaptation_loss

    def validation_step(self, batch, batch_idx):
        self.detector.eval()
        if self.config["split_discriminator_batch"]:
            self.discriminator.train()
        else:
            self.discriminator.eval()

        X, y_true = batch
        y_pred = self(X, y_true)

        detector_loss = self.detection_loss_fn(y_pred, y_true)
        self.log("detector_val_loss", detector_loss)

        discriminator_loss = self._discriminator_loss(y_pred, y_true)
        self.log("discriminator_val_loss", discriminator_loss)

        discriminator_accuracy = self._domain_accuracy(y_pred, y_true)
        self.log("discriminator_val_accuracy", discriminator_accuracy)

        detections = self.detector.postprocessor(y_pred, score_threshold=.01)
        return detections, y_true

    def validation_epoch_end(self, validation_step_outputs):
        detections, y_true = zip(*validation_step_outputs)
        detections = np.concatenate(detections).tolist()
        y_true = np.concatenate(y_true).tolist()
        data = [{
            "scores": _det["scores"],
            "bboxes_pr": _det["bboxes"],
            "bboxes_gt": _y["bboxes"],
            "domain_label": _y["domain_label"],
            } for _y, _det in zip(y_true, detections)]
        ap = AveragePrecisionCalculator()

        AP = ap(data)["all"]
        self.log("val_AP@50_all", AP)

        data = [
            data_point for data_point in data
            if data_point["domain_label"] != int(self.config["source_domain_label"])
        ]
        AP = ap(data)["all"]
        self.log("val_AP@50_target", AP)

    def configure_optimizers(self):
        detector_optimizer = torch.optim.Adam(
            self.detector.parameters(),
            lr=self.config["detector_lr"]
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config["discriminator_lr"]
        )
        return detector_optimizer, discriminator_optimizer

