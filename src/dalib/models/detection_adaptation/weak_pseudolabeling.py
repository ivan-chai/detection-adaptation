import torch
from torch import nn
from torch.nn import functional as F

from ..bbox_utils import BboxConverter, iou
from ...config import prepare_config

from copy import deepcopy
from collections import OrderedDict


class WeakPseudolabeling(nn.Module):

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("source_domain_label", 0),
            ("detection_score_threshold", 0.7),
            ("foreground_score_threshold", 0.7),
            ("background_score_threshold", 0.2),
            ("srrs_iou_threshold", 0.9),
            ("use_srrs_filtering", False),
        ])

    def __init__(self, detector, config=None):
        super().__init__()
        self.config = prepare_config(self, config)
        self.set_detector(detector)

    def set_detector(self, detector):
        self.detector = deepcopy(detector)
        for p in self.detector.parameters():
            p.requires_grad = False

    def __call__(self, y_true, X=None, y_pred=None):
        assert (X is None) != (y_pred is None)
        weak_pseudolabel_context = weak_pseudolabeling(
            X, y_pred, y_true, self.detector, self.config
        )
        return weak_pseudolabel_context


class weak_pseudolabeling:
    
    def __init__(self, X, y_pred, y_true, detector, config):
        self.config = config
        self.y_pred = y_pred
        self.y_true = y_true
        self.X = X
        self.detector = detector
        self.tar_domain_idx = [
            idx for idx, item in enumerate(y_true)
            if item["domain_label"] != self.config["source_domain_label"]
        ]
        self.tar_domain_idx_t = torch.tensor(self.tar_domain_idx)
        self.original_tar_items = [deepcopy(y_true[idx]) for idx in self.tar_domain_idx]

    def __enter__(self):
        with torch.no_grad():
            self._pseudolabel()
            if self.config["use_srrs_filtering"]:
                self._filter_by_srrs()
            self._add_loss_mask()

    def __exit__(self, *args):
        self._restore_labels()
        if self.X is not None:
            self.y_pred = None

    def _pseudolabel(self):
        if len(self.tar_domain_idx) == 0:
            return

        if self.X is not None:
            self.detector.eval()
            tar_y_pred = self.detector(self.X[self.tar_domain_idx_t])
            self.y_pred = tar_y_pred
        else:
            tar_y_pred = {}
            for key, value in self.y_pred.items():
                try:
                    new_value = value[self.tar_domain_idx_t]
                except:
                    new_value = value
                tar_y_pred[key] = new_value

        y_pseudo = self.detector.postprocessor(
            tar_y_pred, score_threshold=self.config["detection_score_threshold"]
        )
        for idx, bboxes in zip(self.tar_domain_idx, [item["bboxes"] for item in y_pseudo]):
            self.y_true[idx]["bboxes"] = bboxes

    def _filter_by_srrs(self):
        if len(self.tar_domain_idx) == 0:
            return

        bbox_converter = BboxConverter(self.y_pred["offsets"], self.y_pred["stride"])

        scores_t = self.y_pred["scores_t"][self.tar_domain_idx]
        deltas_t = self.y_pred["deltas_t"][self.tar_domain_idx].permute(0,2,3,1)
        sizes_t = self.y_pred["sizes_t"][self.tar_domain_idx].permute(0,2,3,1)
        bboxes_t = bbox_converter.make_bboxes(deltas_t, sizes_t)

        tar_y_true = [self.y_true[idx] for idx in self.tar_domain_idx]

        for _scores_t, _bboxes_t, item in zip(scores_t, bboxes_t, tar_y_true):
            _pseudotrue_bboxes = item["bboxes"] if isinstance(item["bboxes"], torch.Tensor)\
                    else torch.tensor(item["bboxes"])
            _pseudotrue_bboxes = _pseudotrue_bboxes.to(_bboxes_t.device)
            _bboxes = _bboxes_t.flatten(0, 1)
            iou_matrix = iou(_pseudotrue_bboxes, _bboxes)
            iou_matrix = torch.where(
                iou_matrix > self.config["srrs_iou_threshold"],
                torch.ones_like(iou_matrix),
                torch.zeros_like(iou_matrix)
            )
            srrs_scores = iou_matrix @ _scores_t.flatten() / iou_matrix.sum(dim=-1)
            item["bboxes"] = item["bboxes"][
                srrs_scores.cpu() > self.config["detection_score_threshold"]
            ]

    def _add_loss_mask(self):
        tar_scores_t = self.y_pred["scores_t"]
        if self.X is None:
            tar_scores_t = tar_scores_t[self.tar_domain_idx_t]
        loss_mask = torch.logical_or(
            tar_scores_t > self.config["foreground_score_threshold"],
            tar_scores_t < self.config["background_score_threshold"]
        )
        for idx, _loss_mask in zip(self.tar_domain_idx, loss_mask):
            self.y_true[idx]["loss_mask"] = _loss_mask

    def _restore_labels(self):
        for idx, item in zip(self.tar_domain_idx, self.original_tar_items):
            self.y_true[idx] = item
