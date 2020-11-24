import torch

import numpy as np
from PIL import Image

from collections import OrderedDict
from copy import deepcopy

from ..config import prepare_config


def intersection(boxes, box):
    """Compute the intersection area of a numpy array of boxes and a single box.

    Args:
        boxes: Numpy array of boxes with shape :math:`(N, 4)`.
        box: Numpy array that represents a single box, with shape :math:`(,4)`.

    Returns:
        Intersection area, a float.
    """
    ix = np.maximum(0, np.minimum(box[2], boxes[:,2]) - np.maximum(box[0], boxes[:,0]))
    iy = np.maximum(0, np.minimum(box[3], boxes[:,3]) - np.maximum(box[1], boxes[:,1]))
    return ix*iy


def crop(image, target, crop_area, in_crop_threshold):
    """Crop an image to a given area and transform target accordingly.

    Args:
        image: PIL.Image.
        target: {
            "bboxes": Numpy array :math:`(N, 4)`,
            "landmarks": Numpy array :math:`(N, n, 2)`, (optional)
            ...
        }
        crop_area: An array or list of four numbers (coordinates of the crop box).
        in_crop_threshold: Float, a threshold for dropping detection targets that
            intersect too little with the crop_area.

    Returns:
        A tuple of image crop (PIL.Image) and transformed targets.
    """
    target = deepcopy(target)
    crop_area = np.array(crop_area)

    bboxes = target["bboxes"]

    intersections = intersection(bboxes, crop_area)
    bbox_areas = (bboxes[:,2:] - bboxes[:,:2]).prod(axis=1)
    in_crop = (intersections/bbox_areas > in_crop_threshold)

    bboxes = bboxes[in_crop] - np.tile(crop_area[:2], 2)
    target["bboxes"] = bboxes

    if "landmarks" in target.keys():
        landmarks = target["landmarks"]
        has_landmarks = target["has_landmarks"]
        landmarks = landmarks[in_crop[has_landmarks]] - crop_area[:2]
        has_landmarks = has_landmarks[in_crop]
        target["landmarks"] = landmarks
        target["has_landmarks"] = has_landmarks

    for key in target.keys():
        if key in ["bboxes", "landmarks", "has_landmarks"]:
            continue
        try:
            target[key] = target[key][in_crop]
        except:
            pass

    image = image.crop(crop_area)

    return image, target


def resize(image, target, target_image_size, bbox_size_threshold):
    """Resize an image to given dimensions and transform the target accordingly.

    Args:
        image: PIL.Image.
        target: {
            "bboxes": Numpy array :math:`(N, 4)`,
            "landmarks": Numpy array :math:`(N, n, 2)`, (optional)
            ...
        }
        target_image_size: list or array of two int, the new image size.
        bbox_size_threshold: Float, transformed detection targets are dropped
            if their diagonal is smaller than this number.

    Returns:
        A tuple of image crop (PIL.Image) and transformed targets.
    """
    target = deepcopy(target)
    scale_factors = np.array(target_image_size)/np.array(image.size)

    bboxes = target["bboxes"]
    bboxes = bboxes * np.tile(scale_factors, 2)
    not_too_small = np.linalg.norm(bboxes[:,2:] - bboxes[:,:2], axis=1) > bbox_size_threshold
    target["bboxes"] = bboxes[not_too_small]

    if "landmarks" in target.keys():
        landmarks = target["landmarks"]
        has_landmarks = target["has_landmarks"]
        landmarks = landmarks * scale_factors
        target["landmarks"] = landmarks[not_too_small[has_landmarks]]
        target["has_landmarks"] = has_landmarks[not_too_small]

    for key in target.keys():
        if key in ["bboxes", "landmarks", "has_landmarks"]:
            continue
        try:
            target[key] = target[key][not_too_small]
        except:
            pass

    image = image.resize(target_image_size)

    return image, target


def generate_semirandom_crop(image, target,
                             target_image_size, p_random, max_magnify,
                             min_bbox_diag, bbox_diag_threshold, in_crop_threshold):
    """Generate a crop of given size and random scale, with accordingly transformed target.
    The crop area is chosen randomly around a randomly chosen detection target (pivot). The rescaling
    is also chosen randomly (with some limitations). With some probability the crop will be
    completely random.

    Args:
        image: PIL.Image.
        target: {
            "bboxes": Numpy array :math:`(N, 4)`,
            "landmarks": Numpy array :math:`(N, n, 2)`, (optional)
            ...
        }
        target_image_size: List or array of two Ints, the output image size.
        p_random: Float, the probability of producing a completely random crop.
        max_magnify: Float, a limiting upscaling factor for the pivot target.
        min_bbox_diag: Float, the smallest pivot diagonal that can be produced
            by rescaling.
        bbox_size_threshold: Float, transformed detection targets are dropped
            if their diagonal is smaller than this number.
        in_crop_threshold: Float, a threshold for dropping detection targets that
            intersect too little with the crop_area.

    Returns:
        A tuple of image crop (PIL.Image) and transformed targets.
    """
    target_image_size = np.array(target_image_size)

    bboxes = target["bboxes"]
    if torch.rand(1).item() > p_random and len(bboxes) > 0:
        bboxes = target["bboxes"]
        bbox = bboxes[torch.randint(len(bboxes), size=(1,)).item()]
        bbox_diag = np.linalg.norm(bbox[2:] - bbox[:2])
        target_bbox_diag = min_bbox_diag + torch.rand(1).item()*min(np.linalg.norm(target_image_size) - min_bbox_diag, max_magnify*bbox_diag - min_bbox_diag)
        rescaling_factor = target_bbox_diag / bbox_diag
        pivot = 0.5*(bbox[:2] + bbox[2:]) + 0.1*(2*torch.rand(2).numpy() - 1)*bbox_diag
    else:
        pivot = np.array(image.size)*(0.2 + 0.6*torch.rand(1).item())
        rescaling_factor = 0.5 + 1.5*torch.rand(1).item()

    if rescaling_factor > 1:
        image, target = crop(image, target, np.concatenate([pivot - 0.5*target_image_size/rescaling_factor, pivot + 0.5*target_image_size/rescaling_factor]), in_crop_threshold)
        image, target = resize(image, target, target_image_size, bbox_diag_threshold)
    else:
        image, target = resize(image, target, (np.array(image.size)*rescaling_factor).astype(np.int), bbox_diag_threshold)
        upper_left = (pivot*rescaling_factor - 0.5*target_image_size).astype(np.int)
        image, target = crop(image, target, np.concatenate([upper_left, upper_left + target_image_size]), in_crop_threshold)

    return image, target


class OnBboxCropper:
    """Generate a crop of given size and random scale, with accordingly transformed target.
    The crop area is chosen randomly around a randomly chosen detection target (pivot). The rescaling
    is also chosen randomly (with some limitations). With some probability the crop will be
    completely random.

    Config:
        target_image_size: List or array of two Ints, the output image size.
        p_random_crop: Float, the probability of producing a completely random crop.
        max_magnify: Float, a limiting upscaling factor for the pivot target.
        min_bbox_diag: Float, the smallest pivot diagonal that can be produced
            by rescaling.
        bbox_size_threshold: Float, transformed detection targets are dropped
            if their diagonal is smaller than this number.
        in_crop_threshold: Float, a threshold for dropping detection targets that
            intersect too little with the crop_area.

    """
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("target_image_size", (256,256)),
            ("p_random_crop", 0.1),
            ("max_magnify", 2),
            ("min_bbox_diag", 10),
            ("bbox_diag_threshold", 5),
            ("in_crop_threshold", 0.3),
        ])

    def __init__(self, config=None):
        config = prepare_config(self, config)
        for k, v in config.items():
            self.__dict__[k] = v

    def __call__(self, image, target):
        """Args:
            image: PIL.Image.
            target: {
                "bboxes": Numpy array :math:`(N, 4)`,
                "landmarks": Numpy array :math:`(N, n, 2)`, (optional)
                ...
            }

        Returns:
            Cropped image and accordingly transformed target of the same shape.
        """
        image, target = generate_semirandom_crop(image, target,
                            self.target_image_size, self.p_random_crop,
                            self.max_magnify, self.min_bbox_diag, self.bbox_diag_threshold,
                            self.in_crop_threshold)
        return image, target
