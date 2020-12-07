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
        loss = self.loss_fn(y_pred, y, normalization="image")
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

    def state_dict(self):
        return self.detector.state_dict()

    def load_state_dict(self, state_dict):
        self.detector.load_state_dict(state_dict)
