import torch
from torchvision.transforms.functional import to_tensor

import numpy as np
from PIL import Image, ImageDraw
import os

from collections import OrderedDict
from dalib.config import prepare_config

def read_annotations(root, split):
    assert(split in ["train","val","test"])
    annotations = []
    with open(os.path.join(root, split, "label.txt"), "r") as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if line[0] == "#":
                path = line[2:]
                annotations.append({"path": os.path.join(split, "images", path), "labels": []})
                line = fp.readline()
                continue
            if "--" in line:
                path = line
                annotations.append({"path": os.path.join(split, "images", path), "labels": []})
                line = fp.readline()
                continue

            numbers = list(map(lambda x: float(x), line.split()))

            if len(numbers) < 4:
                line = fp.readline()
                continue

            box = np.array(numbers[0:4])
            box = np.concatenate([box[:2], box[:2]+box[2:]])

            try:
                landmarks = np.array(numbers[4:19]).reshape(5,3)[:,:2]
                if (landmarks == -1.).all():
                    landmarks = None
            except:
                landmarks = None

            if (box[2]-box[0])*(box[3]-box[1]) > 0:
                annotations[-1]["labels"].append({"box": box, "landmarks": landmarks})
            line = fp.readline()

    return annotations

class WIDERFACEDataset(torch.utils.data.Dataset):
    """
    Expected root dir structure:
        train:
            ---images
            ---label.txt
        test:
            ---images
            ---label.txt
        val:
            ---images
            ---label.txt
    """
    def __init__(self, root, split="train", transform=None):
        assert split in os.listdir(root), f"Folder {split} not found"
        assert "images" in os.listdir(os.path.join(root, split)), f"Folder {split}/images not found"
        assert "label.txt" in os.listdir(os.path.join(root, split)), f"File {split}/label.txt not found"
        self.annotations = read_annotations(root, split)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        a = self.annotations[idx]
        img = Image.open(os.path.join(self.root, a["path"]))
        labels = a["labels"]

        boxes = []
        landmarks = []
        lms_mask = []
        for lb in labels:
            boxes.append(lb["box"])
            pts = lb["landmarks"]
            if pts is not None:
                landmarks.append(pts)
                lms_mask.append(1)
            else:
                lms_mask.append(0)
        boxes = np.array(boxes) if len(boxes)>0 else np.zeros((0,4))
        landmarks = np.array(landmarks) if len(landmarks)>0 else np.zeros((0,5,2))
        lms_mask = np.array(lms_mask) if len(lms_mask)>0 else np.zeros((0,))

        event_name, file_name = a["path"].split("/")[-2:]
        event_name = '--'.join(event_name.split('--')[1:])
        file_name = ".".join(file_name.split(".")[:-1])
        ann = {"event": event_name, "file": file_name, "bboxes": boxes, "landmarks": landmarks, "landmarks_mask": lms_mask}

        if self.transform is not None:
            img, ann = self.transform(img, ann)

        return img, ann
