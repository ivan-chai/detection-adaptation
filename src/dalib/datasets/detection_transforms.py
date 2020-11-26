import torch
from torchvision import transforms

import numpy as np
import cv2
import albumentations as A

from collections import OrderedDict
from copy import deepcopy

from ..config import prepare_config


def intersection(boxes, box):
    """Compute the intersection area of a numpy array of boxes and a single box.

    Args:
        boxes: Numpy array of boxes with shape :math:`(N, 4)` (XYXY format).
        box: Numpy array that represents a single box, with shape :math:`(,4)` (XYXY format).

    Returns:
        Intersection area, a float.
    """
    ix = np.maximum(0, np.minimum(box[2], boxes[:,2]) - np.maximum(box[0], boxes[:,0]))
    iy = np.maximum(0, np.minimum(box[3], boxes[:,3]) - np.maximum(box[1], boxes[:,1]))
    return ix*iy

def _size(image):
    return image.shape[:2][::-1]

def _crop_image(image, crop_area):
    crop_area = np.array(crop_area).astype(np.int)

    top_pad = max(0, -crop_area[1])
    bottom_pad = max(0, crop_area[3] - image.shape[0])
    left_pad = max(0, -crop_area[0])
    right_pad = max(0, crop_area[2] - image.shape[1])

    crop_area[:2] = np.maximum(0, crop_area[:2])
    crop_area[2:] = np.minimum(_size(image), crop_area[2:])

    image = A.functional.crop(image, *crop_area)
    image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT)

    return image


def crop(sample, crop_area, in_crop_threshold):
    """Crop an image to a given area and transform target accordingly.

    Args:
        sample: {
            "image": PIL.Image,
            "bboxes": Numpy array :math:`(N, 4)` (XYXY format),
            "keypoints": Numpy array :math:`(N, n, 2)`, (optional)
            ...
        }
        crop_area: An array or list of four numbers (coordinates of the crop box).
        in_crop_threshold: Float, a threshold for dropping detection targets that
            intersect too little with the crop_area.

    Returns:
        A tuple of image crop (PIL.Image) and transformed targets.
    """
    transformed_sample = {}

    crop_area = np.array(crop_area)

    bboxes = sample["bboxes"]
    intersections = intersection(bboxes, crop_area)
    bbox_areas = (bboxes[:,2:] - bboxes[:,:2]).prod(axis=1)
    in_crop = (intersections/bbox_areas > in_crop_threshold)

    bboxes = bboxes[in_crop] - np.tile(crop_area[:2], 2)
    transformed_sample["bboxes"] = bboxes

    if "keypoints" in sample.keys():
        keypoints = sample["keypoints"]
        keypoints = keypoints[in_crop] - crop_area[:2]
        transformed_sample["keypoints"] = keypoints

    image = sample["image"]
    image = _crop_image(image, crop_area)
    transformed_sample["image"] = image

    for key in sample.keys():
        if key in ["image", "bboxes", "keypoints"]:
            continue
        try:
            transformed_sample[key] = np.array(sample[key])[in_crop]
        except:
            transformed_sample[key] = deepcopy(sample[key])

    return transformed_sample


def resize(sample, target_image_size, bbox_diag_threshold):
    """Resize an image to given dimensions and transform the target accordingly.

    Args:
        sample: {
            "image": PIL.Image,
            "bboxes": Numpy array :math:`(N, 4)` (XYXY format),
            "keypoints": Numpy array :math:`(N, n, 2)`, (optional)
            ...
        }
        target_image_size: list or array of two int, the new image size.
        bbox_diag_threshold: Float, transformed detection targets are dropped
            if their diagonal is smaller than this number.

    Returns:
        A tuple of image crop (PIL.Image) and transformed targets.
    """
    transformed_sample = {}

    image = sample["image"]
    scale_factors = np.array(target_image_size)/np.array(_size(image))

    bboxes = sample["bboxes"]
    bboxes = bboxes * np.tile(scale_factors, 2)
    not_too_small = np.linalg.norm(bboxes[:,2:] - bboxes[:,:2], axis=1) > bbox_diag_threshold
    transformed_sample["bboxes"] = bboxes[not_too_small]

    target_image_size = np.array(target_image_size).astype(np.int)
    image = A.functional.resize(image, *target_image_size[::-1])
    transformed_sample["image"] = image

    if "keypoints" in sample.keys():
        keypoints = sample["keypoints"][not_too_small]
        keypoints = keypoints * scale_factors
        transformed_sample["keypoints"] = keypoints

    for key in sample.keys():
        if key in ["image", "bboxes", "keypoints"]:
            continue
        try:
            transformed_sample[key] = np.array(sample[key])[not_too_small]
        except:
            transformed_sample[key] = deepcopy(sample[key])

    return transformed_sample


class RandomCropOnBboxAndResize:
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
        self.config = config

    @staticmethod
    def _rand():
        return torch.rand(1).item()

    @staticmethod
    def _randint(low, high):
        return torch.randint(low, high, (1,)).item() 

    def _choose_pivot_point_and_rescaling_factor(self, bboxes, image_size):
        p_random_crop = self.config["p_random_crop"]
        if self._rand() < p_random_crop or len(bboxes) == 0:
            pivot_point = np.array(image_size)*(0.2 + 0.6*self._rand())
            rescaling_factor = 0.5 + 1.5*self._rand()
        else:
            min_bbox_diag = self.config["min_bbox_diag"]
            max_magnify = self.config["max_magnify"]
            target_image_size = np.array(self.config["target_image_size"])

            chosen_bbox = bboxes[self._randint(0, len(bboxes))]
            chosen_bbox_diag = np.linalg.norm(chosen_bbox[2:] - chosen_bbox[:2])
            image_diag = np.linalg.norm(np.array(image_size))
            target_image_diag = np.linalg.norm(target_image_size)
            target_bbox_diag = min_bbox_diag\
                + self._rand()*(min(target_image_diag, max_magnify*chosen_bbox_diag) - min_bbox_diag)
            rescaling_factor = target_bbox_diag / chosen_bbox_diag
            pivot_point = (chosen_bbox[:2] + chosen_bbox[2:])/2
            pivot_point += 0.1*(2*np.array([self._rand() for _ in range(2)]) -  1)\
                    *(target_image_size/rescaling_factor - (chosen_bbox[2:] - chosen_bbox[:2]))

        return pivot_point, rescaling_factor


    def __call__(self, **sample):
        """Args:
            target: {
                "image": PIL.Image,
                "bboxes": Numpy array :math:`(N, 4)` (XYXY format),
                "keypoints": Numpy array :math:`(N, n, 2)`, (optional)
                ...
            }

        Returns:
            Dictonary with cropped image and accordingly transformed bboxes, keypoints and labels.
        """
        pivot_point, rescaling_factor\
                = self._choose_pivot_point_and_rescaling_factor(sample["bboxes"], _size(sample["image"]))

        target_image_size = np.array(self.config["target_image_size"])
        crop_area = target_image_size/rescaling_factor 
        crop_area = np.concatenate([pivot_point - crop_area/2, pivot_point + crop_area/2])

        if rescaling_factor < 1:
            sample = resize(sample, np.array(_size(sample["image"]))*rescaling_factor, self.config["bbox_diag_threshold"])
            crop_area = crop_area[:2]*rescaling_factor
            crop_area = crop_area[:2].astype(np.int)
            crop_area = np.concatenate([crop_area, crop_area + target_image_size])
            sample = crop(sample, crop_area, self.config["in_crop_threshold"])
        else:
            sample = crop(sample, crop_area, self.config["in_crop_threshold"])
            sample = resize(sample, target_image_size, self.config["in_crop_threshold"])

        return sample

class DatasetNameToDomainLabel:
    def __init__(self, dataset_names):
        self.dataset_names = dataset_names
    def __call__(self, **sample):
        if "dataset" in sample:
            sample["domain_label"] = self.dataset_names.index(sample["dataset"])
        return sample
