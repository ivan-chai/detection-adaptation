import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from collections import OrderedDict
from dalib.config import prepare_config

def make_target_heatmap(heatmap, bboxes, offsets, stride):
    """
    heatmap: tensor<n_h, n_w>,
    bboxes: tensor<N_boxes, 4>,
    offsets: (int, int),
    stride: int,

    returns: tensor<n_h, n_w>, tensor<n_h,n_w>

    Take predicted heatmap, bboxes and return predicted and target heatmaps
    """
    n_h, n_w = heatmap.shape
    if len(bboxes) == 0:
        return heatmap, torch.zeros(n_h, n_w)
    off_h, off_w = offsets
    # switch to <y, x> order
    if isinstance(bboxes, np.ndarray): bboxes = torch.tensor(bboxes)
    bboxes = bboxes[:,[1,0,3,2]]
    bbox_centers = 0.5*(bboxes[:,:2] + bboxes[:,2:])
    bbox_sides = bboxes[:,2:] - bboxes[:,:2]

    h_coords = off_h + stride*torch.arange(n_h)
    w_coords = off_w + stride*torch.arange(n_w)

    keypoints = bbox_centers - torch.tensor([off_h, off_w])
    keypoints = torch.round(keypoints/stride)
    keypoints = keypoints*stride + torch.tensor([off_h, off_w])

    key_h, key_w = keypoints[:,0], keypoints[:,1]

    sigmas = 0.5*bbox_sides

    sig_h, sig_w = sigmas[:,0], sigmas[:,1]

    heatmap_h = (h_coords[None,:] - key_h[:,None])**2/(2*sig_h[:,None]**2)
    heatmap_h = torch.exp(-heatmap_h)
    heatmap_w = (w_coords[None,:] - key_w[:,None])**2/(2*sig_w[:,None]**2)
    heatmap_w = torch.exp(-heatmap_w)


    target_heatmap = (heatmap_h[:,:,None]*heatmap_w[:,None,:]).max(dim=0).values

    return heatmap, target_heatmap

def make_target_sizes(sizes, bboxes, offsets, stride):
    """
    sizes: tensor<n_h, n_w, 2>,
    bboxes: tensor<N_boxes, 4>,
    offsets: (int, int),
    stride: int,

    return tensor<N_boxes*, 2>, tensor<N_boxes*, 2>
    """
    n_h, n_w = sizes.shape[:2]
    off_h, off_w = offsets
    #switch to <y, x> order
    if isinstance(bboxes, np.ndarray): bboxes = torch.tensor(bboxes)
    bboxes = bboxes[:,[1,0,3,2]]
    bbox_centers = 0.5*(bboxes[:,:2] + bboxes[:,2:])
    bbox_sides = bboxes[:,2:] - bboxes[:,:2]

    inds = bbox_centers - torch.tensor([off_h, off_w])
    inds = torch.round(inds/stride)
    ind_h, ind_w = inds.long().split(1, dim=-1)
    ind_h, ind_w = ind_h.squeeze(-1), ind_w.squeeze(-1)

    # keep keypoints that lie inside the picture
    mask = torch.logical_and(
        torch.logical_and(ind_h >= 0, ind_h < n_h),
        torch.logical_and(ind_w >= 0, ind_w < n_w))

    ind_h, ind_w = ind_h[mask], ind_w[mask]

    target_sizes = bbox_sides[mask]
    #switch to <x, y> order
    target_sizes = target_sizes[...,[1,0]]
    sizes = sizes[ind_h, ind_w]

    return sizes, target_sizes

def make_target_deltas(deltas, bboxes, offsets, stride):
    """
    deltas: tensor<n_h, n_w, 2>,
    bboxes: tensor<N_boxes, 4>,
    offsets: (int, int),
    stride: int,

    returns: tensor<N_boxes*, 2>, tensor<N_boxes*, 2>

    Take predicted deltas, bboxes and return relevant predicted and target deltas
    """
    n_h, n_w = deltas.shape[:2]
    off_h, off_w = offsets
    # switch to <y, x> order
    if isinstance(bboxes, np.ndarray): bboxes = torch.tensor(bboxes)
    bboxes = bboxes[:,[1,0,3,2]]
    bbox_centers = 0.5*(bboxes[:,:2] + bboxes[:,2:])

    keypoints = bbox_centers - torch.tensor([off_h, off_w])
    keypoints = torch.round(keypoints/stride)
    ind_h, ind_w = keypoints.long().split(1, dim=-1)
    ind_h, ind_w = ind_h.squeeze(-1), ind_w.squeeze(-1)
    keypoints = keypoints*stride + torch.tensor([off_h, off_w])

    target_deltas = bbox_centers - keypoints


    # keep keypoints that lie inside the picture
    mask = torch.logical_and(
        torch.logical_and(ind_h >= 0, ind_h < n_h),
        torch.logical_and(ind_w >= 0, ind_w < n_w))

    ind_h, ind_w = ind_h[mask], ind_w[mask]
    target_deltas = target_deltas[mask]
    #switch to <x, y> order
    target_deltas = target_deltas[..., [1,0]]
    deltas = deltas[ind_h, ind_w]

    return deltas, target_deltas

def make_target_landmarks(lm_tensor, bboxes, landmarks, offsets, stride):
    """
    lm_tensor: tensor<n_h, n_w, n_lm, 2>,
    bboxes: tensor<N_boxes, 4>,
    landmarks: tensor<N_boxes, n_lm, 2>,
    offsets: (int, int),
    stride: int,

    returns: tensor<N_boxes*, n_lm, 2>, tensor<N_boxes*, n_lm, 2>

    Take predicted landmarks (relative to predictor"s center), bboxes and landmarks,
    and return relevent predicted and target landmarks
    """
    n_h, n_w = lm_tensor.shape[:2]
    off_h, off_w = offsets
    # switch to <y, x> order
    if isinstance(bboxes, np.ndarray): bboxes = torch.tensor(bboxes)
    bboxes = bboxes[:,[1,0,3,2]]
    bbox_centers = 0.5*(bboxes[:,:2] + bboxes[:,2:])

    keypoints = bbox_centers - torch.tensor([off_h, off_w])
    keypoints = torch.round(keypoints/stride)
    ind_h, ind_w = keypoints.long().split(1, dim=-1)
    ind_h, ind_w = ind_h.squeeze(-1), ind_w.squeeze(-1)
    keypoints = keypoints*stride + torch.tensor([off_h, off_w])

    # keep keypoints that lie inside the picture
    mask = torch.logical_and(
        torch.logical_and(ind_h >= 0, ind_h < n_h),
        torch.logical_and(ind_w >= 0, ind_w < n_w))

    keypoints = keypoints[mask]
    ind_h, ind_w = ind_h[mask], ind_w[mask]
    if isinstance(landmarks, np.ndarray): landmarks = torch.tensor(landmarks)

    # switch to <x, y> order
    keypoints = keypoints[:,[1,0]]
    target_landmarks = landmarks[mask] - keypoints[:,None,:]
    landmarks = lm_tensor[ind_h, ind_w]

    return landmarks, target_landmarks

def pixelwise_focal(y_pred, y_tar, a, b):
    eps = 1e-10
    loss = - (1 - y_tar + eps)**b * y_pred**a * torch.log(1 - y_pred + eps)
    loss = loss.sum()

    mask = (y_tar >= 1 - eps)
    y_pred_masked = y_pred[mask]
    loss += - ((1 - y_pred_masked)**a * torch.log(y_pred_masked + eps)).sum()

    return loss

class FacesAsPointsLoss():
    """
    Parameters:
        offsets: image location that corresponds to upper-left superpixel of a prediction tensor
        stride: stride of the prediction tensor
        a, b: focal loss parameters
        use_landmarks: True if landmarks are available
        weights: the weights with which sub-task losses are weighted

    Bounding box and landmarks coordinates should be in (H, W) frame

    Network output (prediction tensor) channels should have the following structure:
        0: classification score heatmap [0..1]
        1-2: bbox center delta
        3-4: bbox size
        5-: landmarks prediction (if available)

    Target should be a list of dicts: [{"bboxes": tensor<N, 4>, "landmarks": tensor<N, n, 2>, "landmarks_mask": tensor<N>}]
    If use_landmarks flag is False, landmarks and landmarks_mask are ignored

    self.loss_dict holds separate values of each sub-task loss last evaluated
    """
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("offsets", None),
            ("stride", None),
            ("a", 2),
            ("b", 4),
            ("use_landmarks", True),
            ("weights", {
                    "cls_loss": 1.,
                    "delta_loss": 1.,
                    "size_loss": 1.,
                    "lms_loss": 1.,
                }),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        for key, value in config.items():
            self.__dict__[key] = value

    def __call__(self, prediction, target, use_landmarks=None):
        bboxes_batch = [t["bboxes"] for t in target]
        use_landmarks = use_landmarks if use_landmarks is not None else self.use_landmarks
        if use_landmarks:
            landmarks_batch = [t["landmarks"] for t in target]
            landmarks_mask_batch = [t["landmarks_mask"] for t in target]

        offsets = self.offsets
        stride = self.stride

        cls_losses = []
        heatmap_batch = prediction[:,0,...]
        for heatmap, bboxes in zip(heatmap_batch, bboxes_batch):
            heatmap, target_heatmap = make_target_heatmap(heatmap, bboxes, offsets, stride)
            target_heatmap = target_heatmap.to(heatmap.device)
            loss = pixelwise_focal(heatmap, target_heatmap, self.a, self.b)
            loss /= max(1, len(bboxes))
            cls_losses.append(loss)
        cls_losses = torch.stack(cls_losses)

        delta_losses = []
        deltas_batch = prediction[:,1:3,...].permute(0,2,3,1)
        for deltas, bboxes in zip(deltas_batch, bboxes_batch):
            deltas, target_deltas = make_target_deltas(deltas, bboxes, offsets, stride)
            target_deltas = target_deltas.to(deltas.device)
            loss = torch.abs(target_deltas - deltas).sum()
            loss /= max(1, len(target_deltas))
            delta_losses.append(loss)
        delta_losses = torch.stack(delta_losses)

        size_losses = []
        sizes_batch = prediction[:,3:5,...].permute(0,2,3,1)
        for sizes, bboxes in zip(sizes_batch, bboxes_batch):
            sizes, target_sizes = make_target_sizes(sizes, bboxes, offsets, stride)
            target_sizes = target_sizes.to(sizes.device)
            loss = torch.abs(torch.log(sizes/target_sizes)).sum()
            loss /= max(1, len(target_sizes))
            size_losses.append(loss)
        size_losses = torch.stack(size_losses)

        if use_landmarks:
            lm_batch = prediction[:,5:,...].permute(0,2,3,1)
            lm_batch = lm_batch.reshape(*lm_batch.shape[:-1], -1, 2)
            lms_losses = []
            for lm, bboxes, landmarks, landmarks_mask in zip(lm_batch, bboxes_batch, landmarks_batch, landmarks_mask_batch):
                if isinstance(landmarks_mask, np.ndarray): landmarks_mask = torch.tensor(landmarks_mask)
                if isinstance(bboxes, np.ndarray): bboxes = torch.tensor(bboxes)
                bboxes = bboxes[landmarks_mask.bool()]
                landmarks, landmarks_target = make_target_landmarks(lm, bboxes, landmarks, offsets, stride)
                landmarks_target = landmarks_target.to(landmarks.device)
                if len(landmarks) == 0:
                    loss = torch.tensor(0.).to(prediction.device)
                else:
                    scale = (landmarks_target[:,None,:,:] - landmarks_target[:,:,None,:])**2
                    scale = scale.sum(dim=-1).max(dim=-1).values.max(dim=-1).values
                    scale = torch.sqrt(scale)
                    loss = (torch.abs(landmarks - landmarks_target)/scale[:,None,None]).sum()
                    loss /= max(1, len(landmarks))
                lms_losses.append(loss)
            lms_losses = torch.stack(lms_losses)

        loss_dict = {
            "cls_loss": cls_losses.mean(),
            "delta_loss": delta_losses.mean(),
            "size_loss": size_losses.mean(),
        }
        if use_landmarks:
            loss_dict["lms_loss"] = lms_losses.mean()

        self.loss_dict = {k: v.item() for k, v in loss_dict.items()}

        loss = sum([loss_value * self.weights[loss_type] for loss_type, loss_value in loss_dict.items()])

        return loss
