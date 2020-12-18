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


class ReversePredictorGrad:

    def __init__(self, predictor, src_domain_label):
        self.predictor = predictor
        self.src_domain_label = src_domain_label

    def __call__(self, y_pred, y_true):
        return reverse_predictor_grad(
            y_pred, y_true, self.predictor, self.src_domain_label
        )


class stop_running_stats_track:
    def __init__(self, module):
        self.module = module

    @staticmethod
    def _stop_or_restore(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                if hasattr(child, "_old_track_running_stats"):
                    child.track_running_stats = child._old_track_running_stats
                    del child._old_track_running_stats
                else:
                    child._old_track_running_stats = child.track_running_stats
                    child.track_running_stats = False

    def __enter__(self):
        self.module.apply(self._stop_or_restore)

    def __exit__(self, *args):
        self.module.apply(self._stop_or_restore)


class reverse_predictor_grad:

    def __init__(self, y_pred, y_true, predictor, src_domain_label):
        self.y_pred = y_pred
        self.predictor = predictor
        self.reverse_layer = GradientReversalLayer(1.0)

    def __enter__(self):
        self.scores_t_orig = self.y_pred["scores_t"]
        self.deltas_t_orig = self.y_pred["deltas_t"]
        self.sizes_t_orig = self.y_pred["sizes_t"]
        self.embedding_t_orig = self.y_pred["embedding_t"]

        self.y_pred["embedding_t"] = self.reverse_layer(self.y_pred["embedding_t"])
        with stop_running_stats_track(self.predictor):
            y_pred_reversed = self.predictor(self.y_pred)

        self.y_pred["scores_t"] = y_pred_reversed["scores_t"]
        self.y_pred["deltas_t"] = y_pred_reversed["deltas_t"]
        self.y_pred["sizes_t"] = y_pred_reversed["sizes_t"]

    def __exit__(self, *args):
        self.y_pred["scores_t"] = self.scores_t_orig
        self.y_pred["deltas_t"] = self.deltas_t_orig
        self.y_pred["sizes_t"] = self.sizes_t_orig
        self.y_pred["embedding_t"] = self.embedding_t_orig


class DetectionAdaptationPseudolabelingModule(pl.LightningModule):

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("detector", None),
            ("detector_loss", None),
            ("lr", 3e-4),
            ("grad_clip_percentile", 80),
            ("grad_clip_history_size", 100),
            ("source_domain_label", 0),
            ("batchnorm_momentum", 0.1),
            ("adversarial_loss_weight", 1.),
            ("pseudolabel_loss_weight", 1.),
            ("weak_pseudolabeling_config", None),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        self.config = config

        self.detector = Detector(config["detector"])
        self.clipper = AutoClip(config["grad_clip_percentile"])
        self.detection_loss_fn = FacesAsPointsLoss(config["detector_loss"])

        self.detector.apply(
            lambda x: self._set_batchnorm_momentum(x, config["batchnorm_momentum"])
        )
        self.adversarial_loss_fn = AdversarialBackgroundScoreRegularizationLoss()

        self.weak_pseudolabeling = WeakPseudolabeling(self.detector)
        self.reverse_predictor_grad = ReversePredictorGrad(
            self.detector.predictor,
            self.config["source_domain_label"]
        )

        self.best_val_score = None

    @staticmethod
    def _set_batchnorm_momentum(module, momentum):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                child.momentum = momentum

    def forward(self, X):
        y_pred = self.detector(X)
        return y_pred

    def training_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)

        with self.weak_pseudolabeling(y_pred=y_pred, y_true=y_true):
            detection_loss = self.detection_loss_fn(y_pred, y_true, reduction="none")
            weights = torch.ones_like(detection_loss)
            tar_domain_idx = [
                idx for idx, item in enumerate(y_true)\
                if item["domain_label"] != int(self.config["source_domain_label"])
            ]
            tar_domain_idx_t = torch.tensor(tar_domain_idx)
            weights[tar_domain_idx_t] = self.config["pseudolabel_loss_weight"]
            detection_loss = (detection_loss * weights).sum()/max(len(detection_loss) - len(tar_domain_idx), 1)
        self.log("detector_train_loss", detection_loss)

        with self.reverse_predictor_grad(y_pred, y_true):
            adversarial_loss = self.adversarial_loss_fn(y_pred, y_true).mean()
        self.log("adversarial_train_loss", adversarial_loss)

        loss = detection_loss + self.config["adversarial_loss_weight"]*adversarial_loss
        return loss

    def validation_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)

        detector_loss = self.detection_loss_fn(y_pred, y_true)
        self.log("detector_val_loss", detector_loss)

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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["lr"]
        )
        return optimizer

    def on_after_backward(self):
        grad_norm, clipped_grad_norm = self.clipper(self.parameters())
        self.log("grad_norm", grad_norm)
        self.log("clipped_grad_norm", clipped_grad_norm)
