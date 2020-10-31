import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

import pytorch_lightning as pl

from dalib.config import prepare_config
from collections import OrderedDict

from utils import FacesAsPointsLoss
from utils import calculate_PR, calculate_AP

class EvalHead(nn.Module):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("kernel_size", 3),
            ("stride", None),
            ("offsets", None),
            ("threshold", 0.5),
        ])
    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        for k, v in config.items():
            self.__dict__[k] = v
        assert(self.kernel_size%2 == 1)

    def forward(self, x, threshold=None):
        assert(x.dim() == 4)
        threshold = threshold if threshold is not None else self.threshold
        scores = x[:,0,...]
        id_b, id_h, id_w = torch.where(torch.logical_and(scores > threshold,
                    scores == F.max_pool2d(scores, kernel_size=self.kernel_size,
                                               stride=1, padding=self.kernel_size//2)))

        batch_num, counts = torch.unique(id_b.cpu(), sorted=True, return_counts=True)
        splits = torch.zeros(x.shape[0]).long()
        splits[batch_num] = counts

        scores = x[id_b, 0, id_h, id_w]
        deltas = x[id_b, 1:3, id_h, id_w]
        sizes  = x[id_b, 3:5, id_h, id_w]
        y_coord = id_h*self.stride + self.offsets[0]
        x_coord = id_w*self.stride + self.offsets[1]

        pivots = torch.stack([x_coord, y_coord], dim=1)

        centers = pivots + deltas

        bboxes = torch.cat([centers - sizes/2, centers + sizes/2], dim=1).cpu()
        scores = scores.cpu()

        bboxes = torch.split(bboxes, splits.tolist())
        scores = torch.split(scores, splits.tolist())


        has_landmarks = (x.shape[1] > 5) and ((x.shape[1] - 5)%2 == 0)

        if not has_landmarks:
            return [{"bboxes": _bboxes, "scores": _scores} for _bboxes, _scores in zip(bboxes, scores)]

        landmarks = x[id_b, 5:, id_h, id_w].reshape(-1, (x.shape[1]-5)//2, 2)
        landmarks = (landmarks + pivots[:,None,:]).cpu()
        landmarks = torch.split(landmarks, splits.tolist())

        return [{"bboxes": _bboxes, "scores": _scores, "landmarks": _landmarks}\
                    for _bboxes, _scores, _landmarks in zip(bboxes, scores, landmarks)]


class LightningDetectorModule(pl.LightningModule):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("kernel_size", 3),
            ("stride", None),
            ("offsets", None),
            ("threshold", 0.5),
            ("loss_config", None),
        ])
    def __init__(self, model, config=None):
        super().__init__()
        config = prepare_config(self, config)
        self.model = model

        if config["stride"] is None: config["stride"] = self.model.stride
        if config["offsets"] is None: config["offsets"] = self.model.offsets
        if config["loss_config"] is None: config["loss_config"] = {}
        config["loss_config"]["stride"] = config["stride"]
        config["loss_config"]["offsets"] = config["offsets"]

        self.eval_head = EvalHead({k: config[k] for k in ["kernel_size", "stride", "offsets", "threshold"]})

        self.loss_fn = FacesAsPointsLoss(config["loss_config"])

        self.scheduler = None

    def forward(self, x, threshold=None):
        x = self.model(x)
        x = self.eval_head(x, threshold)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss)
        y_pred = self.eval_head(y_pred, threshold=.01)
        return y_pred, y

    def validation_epoch_end(self, validation_step_outputs):
        y_pred, y = [], []
        for _y_pred, _y in validation_step_outputs:
            y_pred += _y_pred
            y += _y

        PR_table = []
        for _y_pred, _y in zip(_y_pred, _y):
            scores_pr = _y_pred["scores"]
            bboxes_pr = _y_pred["bboxes"]
            bboxes_gt = _y["bboxes"]
            _PR_table = calculate_PR(scores_pr, bboxes_pr, bboxes_gt)
            PR_table.append(_PR_table)
        PR_table = np.stack(PR_table, axis=0).mean(axis=0)
        AP = calculate_AP(PR_table)
        self.log("val_AP@50", AP)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=getattr(self, "learning_rate", 4e-3), momentum=0.9)
        if self.scheduler is not None:
            scheduler = {
                    'scheduler': self.scheduler(opt),
                    'interval': 'step',
                    'frequency': 1,
            }
            return [opt], [scheduler]
        else:
            return opt
