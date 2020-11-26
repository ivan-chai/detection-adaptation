import torch
from torch import nn
from torch.nn import functional as F

from torchvision.ops import nms

from collections import OrderedDict

from ...config import prepare_config

class LocMaxNMSPostprocessor(nn.Module):
    """A score local maxima + nms postprocessor. Keeps predictions that correspond
    to local maxima of scores. Then applies standard nms.

    Config:
        kernel_size: max pooling kernel size for local maxima search. Default: 3
        score_threshold: minimal score needed to keep prediction. Default: 0.5
        nms_iou_threshold: parameter for nms. Default: 0.5

    Shape:
        Input: {
            "scores_t": :math:`(B,H,W)`,
            "deltas_t": :math:`(B,2,H,W)`,
            "sizes_t": :math:`(B,2,H,W)`,
            "landmarks_t": :math:`(B,n_landmarks,2,H,W)`, (optional),
            "offsets": (int, int),
            "stride": int,
        }

        Output: [
            {
              "scores": :math:`(N_{i},)`,
              "bboxes": :math:`(N_{i},4)`,
              "landmarks": :math:`(N_{i},n_landmarks,2)`}, (optional)
            },
            for i in range(B),
        ]
    """
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("kernel_size", 3),
            ("score_threshold", 0.5),
            ("nms_iou_threshold", 0.5),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        for k, v in config.items():
            self.__dict__[k] = v

    def forward(self, x, score_threshold=None, nms_iou_threshold=None):
        offsets = x["offsets"]
        stride = x["stride"]
        score_threshold = self.score_threshold if score_threshold is None else score_threshold
        nms_iou_threshold = self.nms_iou_threshold if nms_iou_threshold is None else nms_iou_threshold

        scores = x["scores_t"]
        deltas = x["deltas_t"]
        sizes = x["sizes_t"]

        id_b, id_h, id_w = torch.where(
            torch.logical_and(
                scores > score_threshold,
                scores == F.max_pool2d(scores, self.kernel_size, 1, self.kernel_size//2)
            )
        )

        batch_idx, results_per_batch_idx = torch.unique(id_b.cpu(),
                                    sorted=True, return_counts=True)
        batch_size = scores.shape[0]
        splits = torch.zeros(batch_size).long()
        splits[batch_idx] = results_per_batch_idx
        splits = splits.tolist()

        scores = scores[id_b, id_h, id_w]
        deltas = deltas[id_b, :, id_h, id_w]
        sizes = sizes[id_b, :, id_h, id_w]

        y_coord = id_h*stride + offsets[0]
        x_coord = id_w*stride + offsets[1]

        pivots = torch.stack([x_coord, y_coord], dim=1)

        centers = pivots + deltas

        scores = scores.float().cpu()
        scores = torch.split(scores, splits)
        bboxes = torch.cat([centers - sizes/2, centers + sizes/2], dim=1).float().cpu()
        bboxes = torch.split(bboxes, splits)

        keep = [nms(_bboxes, _scores, nms_iou_threshold)\
                for _bboxes, _scores in zip(bboxes, scores)]

        scores = [_scores[_keep] for _scores, _keep in zip(scores, keep)]
        bboxes = [_bboxes[_keep] for _bboxes, _keep in zip(bboxes, keep)]

        if not "landmarks_t" in x.keys():
            result = [{"bboxes": _bboxes, "scores": _scores}\
                    for _bboxes, _scores in zip(bboxes, scores)]
            return result
        else:
            landmarks = x["landmarks"]
            landmarks = landmarks[id_b, ..., id_h, id_w]
            landmarks = landmarks + pivots[:,None,:]
            landmarks = landmarks.float().cpu()
            landmarks = torch.split(landmarks, splits)

            landmarks = [_landmarks[_keep] for _landmarks, _keep in zip(landmarks, keep)]

            result = [{"bboxes": _bboxes, "scores": _scores, "landmarks": _landmarks}\
                    for _bboxes, _scores, _landmarks in zip(bboxes, scores, landmarks)]
            return result
