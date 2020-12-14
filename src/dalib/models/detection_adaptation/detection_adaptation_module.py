import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

import pytorch_lightning as pl

from collections import OrderedDict
from contextlib import nullcontext

from ..autoclip import AutoClip
from ...config import prepare_config
from ..detection import Detector, FacesAsPointsLoss
from .discriminator import PixelwiseDiscriminator
from ...metrics import AveragePrecisionCalculator
from itertools import chain


DISCRIMINATORS = {
    "pixelwise": PixelwiseDiscriminator
}


class DetectionAdaptationModule(pl.LightningModule):
    """A lightning module wrapper for detection training.
    Config:
        loss: config of FacesAsPointsLoss. Default: None.
        detector: config of Detector. Default: None.
        optimizer: "SGD" or "Adam". Default: "SGD".
        start_lr: if None, is equal to max_lr. Default: 1e-6.
        max_lr: Default: 4e-3.
        end_lr: if None, is equal to max_lr. Default: 1e-6.
        total_steps: Default: 25000.
        anneal_strategy: "linear" or "cos". Default: "linear".
        pct_start: fraction of training spent increasing lr. Default: 0.5.
        base_momentum: Default: 0.85.
        max_momentum: Default: 0.95.
        weight_decay: Default: 1e-6.
        grad_clip_percentile: a percentile for clipping grad norm. Default: 80 (no clipping).
        grad_clip_history_size: size of history for tracking grad norm statistics. Default: 100.
    :attr:`detector` is a Detector pytorch module, and is self-contained.
    :attr:`loss_fn` is the FacesAsPointsLoss instance used for training.
    :attr:`detector.embedding_channels` gives the number of channels of the embedding tensor.
    """

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
            ("normalize_discriminator_loss_by_scores", False),
            ("split_discriminator_batch", True),
            ("detector_lr", 3e-4),
            ("discriminator_lr", 3e-4),
            ("adaptation_factor", 1.0),
            ("discriminator_pretraining_steps", 100),
            ("grad_clip_percentile", 80),
            ("grad_clip_history_size", 100),
            ("source_domain_label", 0),
            ("target_domain_loss_weight", 1.),
            ("freeze_predictor", False),
            ("detector_batchnorm_momentum", 0.1),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        self.config = config

        self.detector = Detector(config["detector"])
        self.detector_clipper = AutoClip(config["grad_clip_percentile"])
        self.detector_loss_fn = FacesAsPointsLoss(config["detector_loss"])

        self.discriminator = DISCRIMINATORS[config["discriminator"]["type"]](config["discriminator"]["config"])
        self.discriminator_clipper = AutoClip(config["grad_clip_percentile"])

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

    def forward(self, x, domain_labels=None, step=None):
        if self.config["freeze_predictor"]:
            self._freeze_predictor()

        detector_context = torch.no_grad if step=="discriminator" else nullcontext
        #discriminator_context = torch.no_grad if step=="detector" else nullcontext 
        discriminator_context = nullcontext
        with detector_context():
            y_pred = self.detector(x)
        with discriminator_context():
            if domain_labels is not None:
                is_src_domain = domain_labels == int(self.config["source_domain_label"])
                is_tar_domain = torch.logical_not(is_src_domain)
                src_embeddings_t = y_pred["embedding_t"][is_src_domain]
                tar_embeddings_t = y_pred["embedding_t"][is_tar_domain]
                
                src_domain_logits = self.discriminator(src_embeddings_t)
                tar_domain_logits = self.discriminator(tar_embeddings_t)

                domain_logits = torch.empty(len(domain_labels),1)\
                                .to(src_domain_logits.device)\
                                .type(src_domain_logits.type())
                if src_domain_logits.ndim > 1:
                    domain_logits = domain_logits[...,None,None].repeat(1, *src_domain_logits.shape[1:])

                domain_logits[is_src_domain] = src_domain_logits
                domain_logits[is_tar_domain] = tar_domain_logits
            else:
                domain_logits = self.discriminator(y_pred["embedding_t"])
            y_pred["domain_logits"] = domain_logits.squeeze(1)
        return y_pred

    def _compute_discriminator_loss(self, y_pred, y_true, flip=False):
        domain_logits = y_pred["domain_logits"]
        domain_labels = torch.tensor([_y["domain_label"] for _y in y_true])

        norm_w = self.config["target_domain_loss_weight"]
        norm_w = norm_w / (1 + norm_w)
        normalized_weights = torch.tensor([
            1-norm_w if label==self.config["source_domain_label"] else norm_w\
            for label in domain_labels
        ])
        normalized_weights = normalized_weights.to(domain_logits.device)

        if flip:
            domain_labels = 1 - domain_labels

        domain_labels = domain_labels.to(domain_logits.device)
        if domain_logits.ndim > 1:
            domain_labels = domain_labels[:,None,None].expand(domain_logits.shape)
            normalized_weights = normalized_weights[:,None,None].expand(domain_logits.shape)
        if self.config["discriminator_loss"] == "mse":
            discriminator_loss = F.mse_loss(domain_logits, domain_labels.type(domain_logits.type()), reduction="none")
        if self.config["discriminator_loss"] == "cross_entropy":
            discriminator_loss = F.cross_entropy(domain_logits, domain_labels.long(), reduction="none") 

        discriminator_loss = discriminator_loss*normalized_weights
        if self.config["normalize_discriminator_loss_by_scores"] and domain_logits.ndim > 1:
            scores_t = y_pred["scores_t"].detach()
            scores_t = scores_t / scores_t.flatten(1,-1).mean(dim=-1)[:,None,None]
            discriminator_loss = discriminator_loss*scores_t
        discriminator_loss = discriminator_loss.mean()
        return discriminator_loss

    def _compute_detector_loss(self, y_pred, y_true):
        source_domain_inds = [
            ind for ind, _y in enumerate(y_true)\
            if _y["domain_label"] == int(self.config["source_domain_label"])
        ]
        y_true_src = [y_true[ind] for ind in source_domain_inds]

        source_domain_inds = torch.tensor(source_domain_inds)
        y_pred_src = {}
        for key, value in y_pred.items():
            try:
                new_value = value[source_domain_inds]
            except:
                new_value = value
            y_pred_src[key] = new_value

        loss = self.detector_loss_fn(y_pred_src, y_true_src)

        return loss

    def _freeze_detector(self):
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad = False

    def _on_after_backward(self, optimizer_idx):
        if optimizer_idx == 0:
            grad_norm, clipped_grad_norm = self.detector_clipper(self.detector.parameters())
            self.log("detector_grad_norm", grad_norm)
            self.log("detector_clipped_grad_norm", clipped_grad_norm)
        if optimizer_idx == 1:
            grad_norm, clipped_grad_norm = self.discriminator_clipper(self.discriminator.parameters())
            self.log("discriminator_grad_norm", grad_norm)
            self.log("discriminator_clipped_grad_norm", clipped_grad_norm)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.detector.train()
        self.discriminator.train()

        X, y_true = batch
        domain_labels = torch.tensor([_y["domain_label"] for _y in y_true])\
                if self.config["split_discriminator_batch"] else None
        detector_optimizer, discriminator_optimizer = self.optimizers()

        if (self.trainer.global_step < int(self.config["discriminator_pretraining_steps"])) or (batch_idx % 2 == 1):
            opt = discriminator_optimizer
            optimizer_idx = 1
            y_pred = self(X, domain_labels=domain_labels, step="discriminator")
            loss  = self._compute_discriminator_loss(y_pred, y_true)
            self.log("discriminator_train_loss", loss)
        else:
            opt = detector_optimizer
            optimizer_idx = 0
            y_pred = self(X, domain_labels=domain_labels, step="detector")
            detector_loss = self._compute_detector_loss(y_pred, y_true)
            self.log("detector_train_loss", detector_loss)
            adaptation_loss = self._compute_discriminator_loss(y_pred, y_true, flip=True)
            self.log("adaptation_train_loss", adaptation_loss)
            loss = detector_loss + self.config["adaptation_factor"]*adaptation_loss

        self.manual_backward(loss, opt)
        self._on_after_backward(optimizer_idx)
        opt.step()

        self.detector.zero_grad()
        self.discriminator.zero_grad()

        return loss

    def validation_step(self, batch, batch_idx):
        self.detector.eval()
        self.discriminator.eval()

        X, y_true = batch
        y_pred = self(X)

        detector_loss = self.detector_loss_fn(y_pred, y_true)
        self.log("detector_val_loss", detector_loss)

        discriminator_loss = self._compute_discriminator_loss(y_pred, y_true)
        self.log("discriminator_val_loss", discriminator_loss)

        true_domain_labels = torch.tensor([_y["domain_label"] for _y in y_true])
        domain_logits = y_pred["domain_logits"]
        if domain_logits.ndim > 1:
            domain_logits = domain_logits.flatten(1,-1).mean(dim=-1)
        if self.config["discriminator_loss"] == "mse":
            pred_domain_labels = (domain_logits > 0.5).long()
        if self.config["discriminator_loss"] == "cross_entropy":
            pred_domain_labels = (domain_logits > .0).long()
        pred_domain_labels = pred_domain_labels.cpu()

        accuracy = (true_domain_labels == pred_domain_labels).float().mean()
        self.log("discriminator_val_acc", accuracy)

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
        self.log("val_AP@50", AP)
        
        data = [
            data_point for data_point in data
            if data_point["domain_label"] != int(self.config["source_domain_label"])
        ]
        AP = ap(data)["all"]
        self.log("target_val_AP@50", AP)

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

