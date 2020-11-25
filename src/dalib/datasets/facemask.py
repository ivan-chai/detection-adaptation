import torch
from torch.utils import data

import numpy as np

from PIL import Image

import xml.etree.ElementTree as ET
import os

from copy import deepcopy


class FaceMaskDataset(data.Dataset):
    """See https://www.kaggle.com/andrewmvd/face-mask-detection.

    Folder structure:
        root
        ---annotations
        ------maksssksksss0.xml
        ------...
        ---images
        ------maksssksksss0.png
        ------...

    Args:
        root: path to root folder of the dataset
        split: "train" or "val"
        transform: transformations callback
        stretch_bboxes: whether to stretch bboxes for more consistency with WIDERFACE labels
    """
    def __init__(self, root, split="train", transform=None, stretch_bboxes=True):
        assert(split in ["train", "val"])
        self.root = root
        self.transform = transform
        self.stretch_bboxes = stretch_bboxes
        ann_files = os.listdir(os.path.join(root, "annotations"))
        self.annotations = []
        for ann_file in sorted(ann_files):
            ann = ET.parse(os.path.join(root, "annotations", ann_file)).getroot()
            filename = ann.find("filename").text

            bboxes = []
            with_mask = []
            is_difficult = []
            for obj in ann.findall('object'):
                bboxes.append([
                    int(obj.find(f'bndbox/{side}').text) for side in ['xmin', 'ymin', 'xmax', 'ymax']
                ])
                with_mask.append(0 if obj.find('name').text == 'without_mask' else 1)
                is_difficult.append(int(obj.find('difficult').text))

            ann = {
                "filename": filename,
                "bboxes": np.array(bboxes),
                "with_mask": np.array(with_mask).astype(np.bool),
                "is_difficult": np.array(is_difficult).astype(np.bool),
            }

            self.annotations.append(ann)

        if split == "train":
            self.annotations = self.annotations[:int(0.5*len(self.annotations))]
        if split == "val":
            self.annotations = self.annotations[int(0.5*len(self.annotations)):]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        target = deepcopy(self.annotations[idx])

        image = Image.open(os.path.join(self.root, "images", target["filename"])).convert("RGB")

        if self.stretch_bboxes:
            bboxes = target["bboxes"]
            heights = bboxes[...,3] - bboxes[...,1]
            bboxes[...,1] = bboxes[...,3] - 1.2*heights
            target["bboxes"] = bboxes

        if self.name:
            target["dataset"] = self.name

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
