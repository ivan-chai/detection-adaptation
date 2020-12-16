import torch
from torch import nn
from torch.nn import functional as F

from ..box_utils import BboxConverter, iou
from ...config import prepare_config

from copy import deepcopy
from collections import OrderedDict


class WeakPseudolabel:

    @staticmethod
    def get_default_config():
        return OrderedDict([
        ])

    def __init__(self, postprocessor, config=None):
        self.config = prepare_config(self, config)
        self.postprocessor = postprocessor

    def __call__(self, y_pred, y_true):
        weak_pseudolabel_context = weak_pseudolabel(y_pred, y_true, self.postprocessor, self.config)
        return weak_pseudolabel_context


class weak_pseudolabel:
    
    def __init__(self, y_pred, y_true, postprocessor, config):
        self.config = config
        self.y_pred = y_pred
        self.y_true = y_true
        self.postprocessor = postprocessor
        self.tar_domain_idx = [
            item["domain_label"] != self.config["sorce_domain_label"] for item in y_true
        ]
        self.tar_domain_idx_t = torch.tensor(self.tar_domain_idx)
        self.original_tar_items = [deepcopy(y_true[idx]) for idx in self.tar_domain_idx]

    def __enter__(self):
        self._pseudolabel()
        self._filter_by_srrs()
        self._add_loss_mask()

    def __exit__(self, *args):
        self._restore_labels()
        self._remove_loss_mask()

    def _pseudolabel(self):
        tar_y_pred = {}
        for key, value in self.y_pred.items():
            try:
                new_value = value[self.tar_domain_idx_t]
            except:
                new_value = value
            tar_y_pred[key] = new_value

        y_pseudo = self.postprocessor(
            tar_y_pred, score_threshold=self.config["detection_score_threshold"]
        )
        for idx, bboxes in zip(self.tar_domain_idx, y_pseudo["bboxes"]):
            self.y_true[idx]["bboxes"] = bboxes
            self.y_true[idx]["domain_label"] = self.config["source_domain_label"]

    def _filter_by_srrs(self):
        bbox_converter = BboxConverter(self.y_pred["offsets"], self.y_pred["stride"])

        tar_domain_idx = [
            item["domain_label"] == self.config["source_domain_label"] for item in self.y_true
        ]
        tar_domain_idx = torch.tensor(tar_domain_idx)

        scores_t = self.y_pred["scores_t"][tar_domain_idx]
        deltas_t = self.y_pred["deltas_t"][tar_domain_idx]
        sizes_t = self.y_pred["sizes_t"][tar_domain_idx]
        bboxes_t = bbox_converter.make_bboxes(deltas_t, sizes_t)

        for _scores_t, _bboxes_t, item in zip(scores_t, bboxes_t, self.y_true):
            _pseudotrue_bboxes = torch.tensor(item["bboxes"]).to(_bboxes_t.device)
            iou_matrix = iou(_pseudotrue_bboxes, _bboxes_t)
            iou_matrix = torch.where(iou_matrix > self.config["srrs_iou_threshold"], 1, 0)
            srrs_scores = (iou_matrix @ _scores_t).mean(dim=-1)
            item["bboxes"] = item["bboxes"][
                srrs_scores.cpu().numpy() > self.config["detection_score_threshold"]
            ]

    def _add_loss_mask(self):
        weak_mask = torch.ones_like(self.y_pred["scores_t"]).bool()
        tar_scores_t = self.y_pred["scores_t"][self.tar_domain_idx_t]
        weak_mask[self.tar_domain_idx_t] = torch.logical_or(
            tar_scores_t > self.config["foreground_score_threshold"],
            tar_scores_t < self.config["background_score_threshold"]
        )
        self.y_pred["weak_mask"] = weak_mask

    def _restore_labels(self):
        for idx, item in zip(self.tar_domain_idx, self.original_tar_items):
            self.y_true[idx] = item

    def _remove_loss_mask(self):
        self.y_pred.pop("weak_mask")
