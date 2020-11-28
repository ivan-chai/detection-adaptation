import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from collections import OrderedDict
from ...config import prepare_config


def _make_pivots_and_indices(bboxes, offsets, stride):
    bbox_centers = 0.5*(bboxes[:,:2] + bboxes[:,2:])
    pivots = bbox_centers - torch.tensor(offsets[::-1])
    pivots = torch.round(pivots/stride)
    ind_h, ind_w = pivots.long().split(1, dim=-1)
    ind_h, ind_w = ind_h.squeeze(-1), ind_w.squeeze(-1)
    pivots = pivots*stride + torch.tensor(offsets[::-1])
    return pivots, ind_h, ind_w


def make_target_scores(scores_t, bboxes, offsets, stride):
    """Take predicted scores tensor and target bboxes, and
    return predicted and target scores tensors.

    Args:
        scores_t: :math:`(H, W)`.
        bboxes: :math:`(N, 4)`. (XYXY format)
        offsets: (int, int).
        stride: int.

    Returns:
        (:math:`(H,W)`, :math:`(H,W)`).

    """
    n_h, n_w = scores_t.shape
    if len(bboxes) == 0:
        return scores_t, torch.zeros(n_h, n_w)
    off_h, off_w = offsets
    # switch to <y, x> order
    if isinstance(bboxes, np.ndarray): bboxes = torch.tensor(bboxes)
    bboxes = bboxes[:,[1,0,3,2]]
    bbox_centers = 0.5*(bboxes[:,:2] + bboxes[:,2:])
    bbox_sides = bboxes[:,2:] - bboxes[:,:2]

    h_coords = torch.arange(n_h)
    w_coords = torch.arange(n_w)

    pivots, ind_h, ind_w = _make_pivots_and_indices(bboxes, offsets, stride)

    sigmas = 0.5*bbox_sides/stride
    sig_h, sig_w = sigmas[:,0], sigmas[:,1]

    target_scores_h = (h_coords[None,:] - ind_h[:,None])**2/(2*sig_h[:,None]**2)
    target_scores_h = torch.exp(-target_scores_h)
    target_scores_w = (w_coords[None,:] - ind_w[:,None])**2/(2*sig_w[:,None]**2)
    target_scores_w = torch.exp(-target_scores_w)


    target_scores_t = (target_scores_h[:,:,None]*target_scores_w[:,None,:]).max(dim=0).values

    return scores_t, target_scores_t

def make_target_sizes(sizes_t, bboxes, offsets, stride):
    """Take predicted bbox sizes tensor and target bboxes, 
    and return predicted and target bbox sizes.

    Args:
        sizes_t: :math:`(H, W)`.
        bboxes: :math:`(N, 4)`. (XYXY format)
        offsets: (int, int).
        stride: int.

    Returns:
        (:math:`(N', 2)`, :math:`(N', 2)`)
    """
    n_h, n_w = sizes_t.shape[:2]
    off_h, off_w = offsets
    #switch to <y, x> order
    if isinstance(bboxes, np.ndarray): bboxes = torch.tensor(bboxes)
    bboxes = bboxes[:,[1,0,3,2]]
    bbox_centers = 0.5*(bboxes[:,:2] + bboxes[:,2:])
    bbox_sides = bboxes[:,2:] - bboxes[:,:2]

    pivots, ind_h, ind_w = _make_pivots_and_indices(bboxes, offsets, stride)

    # keep pivots that lie inside the picture
    mask = torch.logical_and(
        torch.logical_and(ind_h >= 0, ind_h < n_h),
        torch.logical_and(ind_w >= 0, ind_w < n_w))

    ind_h, ind_w = ind_h[mask], ind_w[mask]

    target_sizes = bbox_sides[mask]
    #switch to <x, y> order
    target_sizes = target_sizes[...,[1,0]]
    sizes = sizes_t[ind_h, ind_w]

    return sizes, target_sizes

def make_target_deltas(deltas_t, bboxes, offsets, stride):
    """Take predicted bbox deltas tensor and target bboxes,
    and return predicted and target deltas.

    Args:
        deltas_t: :math:`(H,  W, 2)`.
        bboxes: :math:`(N, 4)`. (XYXY format)
        offsets: (int, int).
        stride: int.

    Returns:
        (:math:`(N', 2)`, :math:`(N', 2)`)
    """
    n_h, n_w = deltas_t.shape[:2]
    off_h, off_w = offsets
    # switch to <y, x> order
    if isinstance(bboxes, np.ndarray): bboxes = torch.tensor(bboxes)
    bboxes = bboxes[:,[1,0,3,2]]
    bbox_centers = 0.5*(bboxes[:,:2] + bboxes[:,2:])

    pivots, ind_h, ind_w = _make_pivots_and_indices(bboxes, offsets, stride)

    target_deltas = bbox_centers - pivots

    # keep keypoints that lie inside the picture
    mask = torch.logical_and(
        torch.logical_and(ind_h >= 0, ind_h < n_h),
        torch.logical_and(ind_w >= 0, ind_w < n_w))

    ind_h, ind_w = ind_h[mask], ind_w[mask]
    target_deltas = target_deltas[mask]
    #switch to <x, y> order
    target_deltas = target_deltas[..., [1,0]]
    deltas = deltas_t[ind_h, ind_w]

    return deltas, target_deltas

def make_target_keypoints(keypoints_t, bboxes, keypoints, offsets, stride):
    """Take predicted keypoints tensor, target bboxes and keypoints,
    and return predicted and target keypoints.

    Args:
        keypoints_t: :math:`(H, W, n, 2)`.
        bboxes: :math:`(N, 4)`. (XYXY format)
        keypoints: :math:`(N, n, 2)`.
        offsets: (int, int).
        stride: int.

    Returns:
        (:math:`(N', n, 2)`, :math:`(N', n, 2)`)
    """
    n_h, n_w = keypoints_t.shape[:2]
    off_h, off_w = offsets
    # switch to <y, x> order
    if isinstance(bboxes, np.ndarray): bboxes = torch.tensor(bboxes)
    bboxes = bboxes[:,[1,0,3,2]]

    pivots, ind_h, ind_w = _make_pivots_and_indices(bboxes, offsets, stride)

    # keep keypoints that lie inside the picture
    mask = torch.logical_and(
        torch.logical_and(ind_h >= 0, ind_h < n_h),
        torch.logical_and(ind_w >= 0, ind_w < n_w))

    keypoints = keypoints[mask]
    ind_h, ind_w = ind_h[mask], ind_w[mask]
    if isinstance(keypoints, np.ndarray): keypoints = torch.tensor(keypoints)

    # switch to <x, y> order
    keypoints = keypoints[:,[1,0]]
    target_keypoints = keypoints[mask] - keypoints[:,None,:]
    keypoints = keypoints_t[ind_h, ind_w]

    return keypoints, target_keypoints

def pixelwise_focal(scores_t, target_scores_t, a, b):
    """Pixeslwise focal loss as in arXiv:1904.07850.

    Args:
        scores_t: :math:`(H, W)`.
        target_scores_t: :math:`(H, W)`.
        a: alpha parameter.
        b: beta parameter.

    Returns:
        :math:`(,)`
    """
    eps = 1e-10
    loss = - (1 - target_scores_t + eps)**b * scores_t**a * torch.log(1 - scores_t + eps)
    loss = loss.sum()

    mask = (target_scores_t >= 1 - eps)
    scores_t_masked = scores_t[mask]
    loss += - ((1 - scores_t_masked + eps)**a * torch.log(scores_t_masked + eps)).sum()

    return loss

class FacesAsPointsLoss():
    """Faces as points loss based on arxiv:1904.07850.
    On call returns the value of batch loss, and saves a dict of
    losses by task to :attr:`loss_dict` (detached from graph).

    Config:
        a: alpha parameter of pixelwise focal loss. Default: 2.
        b: beta parameter of pixelwise focal loss. Default: 4.
        weights:
            cls_loss: weight of classification loss. Default: 1.
            delta_loss: weight of bbox center position regression loss. Default: 1.
            size_loss: weight of bbox size regression loss. Default: 1.
            keypts_loss: weight of keypoints regression loss. Default: 1.

    Inputs:
        prediction: {
            "scores_t": :math:`(B,H,W)`,
            "deltas_t": :math:`(B,2,H,W)`,
            "sizes_t": :math:`(B,2,H,W)`,
            "keypoints_t": :math:`(B,n_keypoints,2,H,W)`, (optional)
            "offsets": (int, int),
            "stride": int,
        }

        target: [
            {
              "bboxes": :math:`(N_{i},4)`,
              "keypoints": :math:`(N_{i},n_keypoints,2)`, (optional)
            }
            for i in range(B)
        ]

        reduction: "mean", "sum" or "none". Default: "mean"

    Outputs:
        :math:`(,)`
    """
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("a", 2),
            ("b", 4),
            ("weights", {
                    "cls_loss": 1.,
                    "delta_loss": 1.,
                    "size_loss": 1.,
                    "keypts_loss": 1.,
                }),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        for key, value in config.items():
            self.__dict__[key] = value

    @staticmethod
    def _reduce(tensor, reduction):
        if reduction=="mean":
            return tensor.mean()
        elif reduction=="sum":
            return tensor.sum()
        elif reduction=="none":
            return tensor
        raise Exception("reduction should be 'mean', 'sum' or 'none'")

    def __call__(self, prediction, target, reduction="mean"):
        offsets = prediction["offsets"]
        stride = prediction["stride"]

        bboxes_batch = [t["bboxes"] for t in target]

        cls_losses = []
        scores_batch = prediction["scores_t"]
        for scores_t, bboxes in zip(scores_batch, bboxes_batch):
            scores_t, target_scores_t = make_target_scores(scores_t, bboxes, offsets, stride)
            target_scores_t = target_scores_t.to(scores_t.device)
            loss = pixelwise_focal(scores_t, target_scores_t, self.a, self.b)
            loss /= max(1, len(bboxes))
            cls_losses.append(loss)
        cls_losses = torch.stack(cls_losses)

        delta_losses = []
        size_losses = []
        deltas_batch = prediction["deltas_t"].permute(0,2,3,1)
        sizes_batch = prediction["sizes_t"].permute(0,2,3,1)
        for deltas_t, sizes_t, bboxes in zip(deltas_batch, sizes_batch, bboxes_batch):
            deltas, target_deltas = make_target_deltas(deltas_t, bboxes, offsets, stride)
            sizes, target_sizes = make_target_sizes(sizes_t, bboxes, offsets, stride)
            target_deltas = target_deltas.to(deltas.device)
            target_sizes = target_sizes.to(sizes.device)

            delta_loss = torch.abs((target_deltas - deltas)/target_sizes).sum()
            delta_loss /= max(1, len(target_deltas))
            size_loss = torch.abs(torch.log(sizes/target_sizes)).sum()
            size_loss /= max(1, len(target_sizes))

            delta_losses.append(delta_loss)
            size_losses.append(size_loss)
        delta_losses = torch.stack(delta_losses)
        size_losses = torch.stack(size_losses)

        for sizes_t, bboxes in zip(sizes_batch, bboxes_batch):
            loss = torch.abs(torch.log(sizes/target_sizes)).sum()
            loss /= max(1, len(target_sizes))

        if "keypoints" in prediction:
            keypoints_batch = prediction["keypoints_t"].permute(0,3,4,1,2)
            target_keypoints_batch, target_has_keypoints_batch = [], []
            for t in target:
                if "keypoints" not in t.keys():
                    target_keypoints_batch.append(torch.empty((0,keypoints_batch.shape[-2],2)))
                    target_has_keypoints_batch.append(torch.zeros(len(t["bboxes"])).bool())
                else:
                    target_keypoints_batch.append(t["keypoints"][t["has_keypoints"]])
                    target_has_keypoints_batch.append(t["has_keypoints"])
            keypts_losses = []
            for keypoints_t, bboxes, target_keypoints, target_has_keypoints\
                    in zip(keypoints_batch, bboxes_batch, target_keypoints_batch, target_has_keypoints_batch):
                if isinstance(target_has_keypoints, np.ndarray):
                    target_has_keypoints = torch.tensor(target_has_keypoints)
                if isinstance(bboxes, np.ndarray): bboxes = torch.tensor(bboxes)
                bboxes = bboxes[has_keypoints.bool()]
                keypoints, target_keypoints = make_target_keypoints(keypoints, bboxes, target_keypoints, offsets, stride)
                target_keypoints = target_keypoints.to(keypoints.device)
                if len(keypoints) == 0:
                    loss = torch.tensor(0.).to(keypoints.device)
                else:
                    scale = (target_keypoints[:,None,:,:] - target_keypoints[:,:,None,:])**2
                    scale = scale.sum(dim=-1).max(dim=-1).values.max(dim=-1).values
                    scale = torch.sqrt(scale)
                    loss = (torch.abs(keypoints - target_keypoints)/scale[:,None,None]).sum()
                    loss /= max(1, len(keypoints))
                keypts_losses.append(loss)
            keypts_losses = torch.stack(keypts_losses)

        loss_dict = {
            "cls_loss": cls_losses,
            "delta_loss": delta_losses,
            "size_loss": size_losses,
        }
        if "keypoints" in prediction:
            loss_dict["keypts_loss"] = keypts_losses

        self.loss_dict = {k: v.detach().cpu() for k, v in loss_dict.items()}

        loss = sum([loss_value * self.weights[loss_type] for loss_type, loss_value in loss_dict.items()])
        loss = self._reduce(loss, reduction)

        return loss
