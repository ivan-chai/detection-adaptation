import torch
from torch.utils import data

import numpy as np

import os

from PIL import Image


def read_annotations(filename):
    annotations = []
    with open(filename, "r") as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            image_path = line.strip()
            num_faces = fp.readline().strip()
            num_faces = int(num_faces)
            ellipses = []
            for _ in range(num_faces):
                line = fp.readline()
                numbers = line.strip().split()
                numbers = list(map(float, numbers))
                ellipse = {
                    "major_radius": numbers[0],
                    "minor_radius": numbers[1],
                    "angle": numbers[2],
                    "center_x": numbers[3],
                    "center_y": numbers[4],
                }
                ellipses.append(ellipse)
            annotations.append({
                "path": image_path + ".jpg",
                "ellipses": ellipses,
            })
    return annotations


def ellipse_to_bbox(ellipse):
    a = ellipse["minor_radius"]
    b = ellipse["major_radius"]
    phi = ellipse["angle"]
    c_x = ellipse["center_x"]
    c_y = ellipse["center_y"]

    half_width = np.sqrt(a**2 * np.sin(phi)**2 + b**2 * np.cos(phi)**2)
    half_height = np.sqrt(b**2 * np.sin(phi)**2 + a**2 * np.cos(phi)**2)

    return [c_x - half_width, c_y - half_height, c_x + half_width, c_y + half_height]


class FDDBDataset(data.Dataset):
    """Folder structure:
        root
        ---folds
        ------FDDB-fold-01-ellipseList.txt
        ------...
        ------FDDB-fold-10-ellipseList.txt
        ---images
        ------2002/07/19/big/img130.jpg

    Args:
        root: path to root folder of the dataset
        split: "train" or "val"
        transform: transformations callback
    """
    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val"]

        annotations = []
        for i in range(1,9) if split=="train" else [9, 10]:
            path = os.path.join(root, "folds", f"FDDB-fold-{i:02d}-ellipseList.txt")
            annotations.extend(read_annotations(path))
        self.annotations = annotations
        self.transform = None
        self.root = root

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(self.root, "images", ann["path"])

        image = Image.open(image_path).convert("RGB")

        ellipses = ann["ellipses"]
        bboxes = list(map(ellipse_to_bbox, ellipses))

        bboxes = np.array(bboxes)
        target = {"bboxes": bboxes}
        if self.name:
            target["dataset"] = self.name

        if self.transform is not None:
            image, target = self.transform(image, target) 

        return image, target
