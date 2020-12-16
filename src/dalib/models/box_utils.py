import torch
from torch import nn
from torch.nn import functional as F


def iou(boxes_a, boxes_b):
    boxes_a = boxes_a[:, None, :]
    boxes_b = boxes_b[None, :, :]

    intersections =\
        F.relu(
            torch.min(boxes_a[..., 2], boxes_b[..., 2])\
            -torch.max(boxes_a[..., 0], boxes_b[..., 0])
        )\
        *F.relu(
            torch.min(boxes_a[..., 3], boxes_b[..., 3])\
            -torch.max(boxes_a[..., 1], boxes_b[..., 1])
        )

    areas_a = F.relu((boxes_a[..., 2:] - boxes_a[..., :2]).prod(axis=-1))
    areas_b = F.relu((boxes_b[..., 2:] - boxes_b[..., :2]).prod(axis=-1))

    union = areas_a + areas_b - intersections

    iou_matrix = intersections / (union + 1e-10)

    return iou_matrix


class BoxConverter:
    def __init__(self, offsets, stride):
        self.offsets = offsets
        self.stride = stride

    def make_boxes(self, deltas, sizes, pivots=None):
        if pivots is None:
            assert deltas.ndim in [3.4]
            pivots = self.make_pivots_from_shape(deltas.shape[-2:])
            pivots = pivots[None, ...]
        centers = pivots.to(deltas.device) + deltas
        half_sizes = (sizes/2).to(deltas.device)
        bboxes = torch.cat([
            centers - half_sizes,
            centers + half_sizes
        ], dim=1)
        return bboxes

    def make_pivots_from_indices(self, id_h, id_w):
        y_coord = id_h*self.stride + self.offsets[0]
        x_coord = id_w*self.stride + self.offsets[1]
        pivots = torch.stack([x_coord, y_coord], dim=1)
        return pivots

    def make_pivots_from_shape(self, height, width):
        y_coord = torch.arange(height)*self.stride + self.offsets[0]
        x_coord = torch.arange(width)*self.stride + self.offsets[1]
        pivots = torch.stack([
            x_coord[None, :].expand(height, width),
            y_coord[:, None].expand(height, width)
        ])
        return pivots
