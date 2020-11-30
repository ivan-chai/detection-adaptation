import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

import pytorch_lightning as pl

from collections import OrderedDict


from ..autoclip import AutoClip
from ...config import prepare_config
from .faces_as_points_loss import FacesAsPointsLoss
from .detector import Detector
from ...metrics import AveragePrecisionCalculator
from dalib.models import GradientReversalLayer
from itertools import chain
from pytorch_lightning.metrics.functional import accuracy

class DetectionModule(pl.LightningModule):
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
            ("loss", None),
            ("detector", None),
            ("optimizer", "SGD"),
            ("start_lr", 1e-6),
            ("max_lr", 4e-3),
            ("end_lr", 1e-6),
            ("total_steps", 25000),
            ("anneal_strategy", "linear"),
            ("pct_start", 0.5),
            ("base_momentum", 0.85),
            ("max_momentum", 0.95),
            ("weight_decay", 1e-6),
            ("grad_clip_percentile", 100),
            ("grad_clip_history_size", 80),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        self.config = config

        self.detector = Detector(config["detector"])
        self.clipper = AutoClip(config["grad_clip_percentile"])
        self.loss_fn = FacesAsPointsLoss(config["loss"])

    def forward(self, x):
        x = self.detector(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.detector(X)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.detector(X)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss)
        y_pred = self.detector.postprocessor(y_pred, score_threshold=.01)
        return y_pred, y

    def validation_epoch_end(self, validation_step_outputs):
        y_pred, y = zip(*validation_step_outputs)
        y_pred = np.concatenate(y_pred).tolist()
        y = np.concatenate(y).tolist()
        data = [{
            "scores": _y_pred["scores"],
            "bboxes_pr": _y_pred["bboxes"],
            "bboxes_gt": _y["bboxes"]
            } for _y, _y_pred in zip(y, y_pred)]

        ap = AveragePrecisionCalculator()
        AP = ap(data)["all"]
        self.log("val_AP@50", AP)

    def configure_optimizers(self):
        config = self.config
        opts = {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam}
        opt = opts[config["optimizer"]](self.parameters(), lr=4e-3, weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr = config["max_lr"],
                total_steps = config["total_steps"],
                pct_start = config["pct_start"],
                anneal_strategy = config["anneal_strategy"],
                base_momentum = config["base_momentum"],
                max_momentum = config["max_momentum"],
                div_factor = 1 if config["start_lr"] is None else config["max_lr"]/config["start_lr"],
                final_div_factor = 1 if config["end_lr"] is None else config["max_lr"]/config["end_lr"]
        )
        return [opt], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def on_after_backward(self):
        grad_norm, clipped_grad_norm = self.clipper(self.parameters())
        self.log("grad_norm", grad_norm)
        self.log("clipped_grad_norm", clipped_grad_norm)



class DetectionDomainAdaptation(pl.LightningModule):
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
            ("loss", None),
            ("detector", None),
            ("optimizer", "SGD"),
            ("start_lr", 1e-6),
            ("max_lr", 4e-3),
            ("end_lr", 1e-6),
            ("total_steps", 25000),
            ("anneal_strategy", "linear"),
            ("pct_start", 0.5),
            ("base_momentum", 0.85),
            ("max_momentum", 0.95),
            ("weight_decay", 1e-6),
            ("grad_clip_percentile", 100),
            ("grad_clip_history_size", 80),
            ("domain_adaptation", True),
            ("adaptation_factor", 1.0),
            ("domain_classifier_loss", "mse_loss"),
            ("domain_classifier_pool_size", 1),
            ("use_only_y_labels_from_source_domain", True),
            ("target_domain_label", 1),
            ("gan_style_training", False),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        self.config = config

        self.detector = Detector(config["detector"])
        self.clipper = AutoClip(config["grad_clip_percentile"])
        self.loss_fn = FacesAsPointsLoss(config["loss"])

        if self.config["domain_adaptation"]:
            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(lambda_=self.config["adaptation_factor"]),
                nn.Conv2d(self.detector.embedding_channels, 512, kernel_size=1, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 1, kernel_size=1, stride=1)
            )
            self.domain_classifier_loss = getattr(F, config["domain_classifier_loss"])

    def forward(self, x):
        x = self.detector(x)
        if self.config["domain_adaptation"]:
            domain_logits = self.domain_classifier(x["embedding_t"]).squeeze(1)
            return x, domain_logits
        else:
            return x, None

    def _configure_single_optimizer(self, parameters):
        config = self.config
        opts = {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam}
        opt = opts[config["optimizer"]](parameters, lr=4e-3, weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr = config["max_lr"],
                total_steps = config["total_steps"],
                pct_start = config["pct_start"],
                anneal_strategy = config["anneal_strategy"],
                base_momentum = config["base_momentum"],
                max_momentum = config["max_momentum"],
                div_factor = 1 if config["start_lr"] is None else config["max_lr"]/config["start_lr"],
                final_div_factor = 1 if config["end_lr"] is None else config["max_lr"]/config["end_lr"]
        )
        scheduler_dict = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return opt, scheduler_dict

    def _compute_output_loss(self, y_pred, y, is_source_domain=None, mode="train"):
        if self.config["domain_adaptation"] and is_source_domain is not None:
            output_loss = self.loss_fn(y_pred, y, reduction="none")[is_source_domain].mean()
            if not is_source_domain.any():
                output_loss = 0
        else:
            output_loss = self.loss_fn(y_pred, y)
        self.log(f"{mode}_output_loss", output_loss, prog_bar=True)
        return output_loss

    def _compute_discriminator_loss(self, domain_logits, domain_label, mode="train"):
        discriminator_loss = self.domain_classifier_loss(domain_logits, domain_label.float())
        self.log(f"{mode}_discriminator_loss", discriminator_loss, prog_bar=True)
        return discriminator_loss

    def _freeze_detector(self):
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        X, y = batch
        y_pred, domain_logits_img = self(X)

        if self.config["domain_adaptation"]:
            domain_label = torch.from_numpy(np.array([sample["domain_label"] for sample in y])).type_as(domain_logits_img).long()
            domain_label_img = torch.ones_like(domain_logits_img, dtype=torch.long)
            select = domain_label != self.config["target_domain_label"]
            domain_label_img[select] = 0

            if self.config["use_only_y_labels_from_source_domain"]:
                is_source_domain = select
            else:
                is_source_domain = None

            if optimizer_idx is None:
                loss = self._compute_output_loss(y_pred, y, is_source_domain) + \
                    self._compute_discriminator_loss(domain_logits_img, domain_label_img)
                self.log("train_loss", loss)
            elif optimizer_idx == 0:
                loss = self._compute_output_loss(y_pred, y, is_source_domain)
            elif optimizer_idx == 1:
                loss = self._compute_discriminator_loss(domain_logits_img, domain_label_img)
        else:
            loss = self._compute_output_loss(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred, domain_logits_img = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss)

        if domain_logits_img is not None:
            domain_label = torch.from_numpy(np.array([sample["domain_label"] for sample in y])).type_as(domain_logits_img).long()
            domain_label_img = torch.ones_like(domain_logits_img, dtype=torch.long)
            select = domain_label != self.config["target_domain_label"]
            domain_label_img[select] = 0

            self._compute_discriminator_loss(domain_logits_img, domain_label_img, mode="val")

            d_preds = (domain_logits_img > 0.5).long()
            acc = accuracy(d_preds, domain_label_img)
            self.log("val_d_acc", acc)

        y_pred = self.detector.postprocessor(y_pred, score_threshold=.01)
        return y_pred, y

    def validation_epoch_end(self, validation_step_outputs):
        y_pred, y = zip(*validation_step_outputs)
        y_pred = np.concatenate(y_pred).tolist()
        y = np.concatenate(y).tolist()
        data = [{
            "scores": _y_pred["scores"],
            "bboxes_pr": _y_pred["bboxes"],
            "bboxes_gt": _y["bboxes"]
            } for _y, _y_pred in zip(y, y_pred)]

        ap = AveragePrecisionCalculator()
        AP = ap(data)["all"]
        self.log("val_AP@50", AP)

    def configure_optimizers(self):
        if self.config["gan_style_training"]:
            optimizer_y, scheduler_y = self._configure_single_optimizer(
                self.detector.parameters()
            )
            optimizer_d, scheduler_d = self._configure_single_optimizer(
                chain(self.detector.parameters(), self.domain_classifier.parameters())
            )
            return [optimizer_y, optimizer_d], [scheduler_y, scheduler_d]
        else:
            optimizer, scheduler = self._configure_single_optimizer(self.parameters())
            return [optimizer], [scheduler]

    def on_after_backward(self):
        grad_norm, clipped_grad_norm = self.clipper(self.parameters())
        self.log("grad_norm", grad_norm)
        self.log("clipped_grad_norm", clipped_grad_norm)
