import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

import numpy as np

from dalib.config import prepare_config
from collections import OrderedDict

import pytorch_lightning as pl

from tqdm import tqdm

from .widerface import WIDERFACEDataset


class WIDERFACEDataModule(pl.LightningDataModule):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("root", "WIDERFACE"),
            ("num_workers", 8),
            ("batch_size", 32),
            ("crop_config", None),
            ("collate_config", None),
            ("shuffle", True),
            ("drop_last", True),
            ("val_samples", 512),
        ])
    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        if config["collate_config"] is not None:
            grid_h = config["collate_config"].get("grid_h", 1)
            grid_w = config["collate_config"].get("grid_w", 1)
            assert(config["batch_size"] % (grid_h * grid_w) == 0)
        self.config = config

    def _prepare_val_dataloader(self):
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)

        X, y = [], []
        for idx in tqdm(range(self.config["val_samples"]), desc="Preparing minival dataset"):
            img, ann = self.val_ds[idx]
            X.append(to_tensor(img))
            y.append(ann)

        class ListDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __len__(self):
                return min(len(self.X), len(self.y))

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        torch.set_rng_state(rng_state)
        return DataLoader(ListDataset(X, y),
                          batch_size=3*self.config["batch_size"],
                          collate_fn = WIDERBatch,
                          drop_last=False,
                          shuffle=False)


    def setup(self, stage=None):
        transform_fn = OnFaceCropper(self.config["crop_config"])
        self.train_ds = WIDERFACEDataset(self.config["root"], split="train",
                                         transform=transform_fn)
        self.val_ds   = WIDERFACEDataset(self.config["root"], split="val",
                                         transform=transform_fn)

        self.val_loader = self._prepare_val_dataloader()

    def train_dataloader(self):
        collate_fn = Collator(self.config["collate_config"])
        return DataLoader(self.train_ds,
                          batch_size=self.config["batch_size"],
                          num_workers=self.config["num_workers"],
                          collate_fn=collate_fn,
                          shuffle=self.config["shuffle"],
                          drop_last=self.config["drop_last"])

    def val_dataloader(self):
        return self.val_loader


class WIDERBatch:
    def __init__(self, samples):
        X, y = zip(*samples)
        self.X = torch.stack(X)
        self.y = y

    def pin_memory(self):
        self.X = self.X.pin_memory()
        return self

    def to(self, device):
        self.X = self.X.to(device)
        return self

    def __len__(self):
        return len(self.X)

    def __iter__(self):
        return iter((self.X, self.y))


#######################
#######################

def intersection(boxes, box):
    ix = np.maximum(0, np.minimum(box[2], boxes[:,2]) - np.maximum(box[0], boxes[:,0]))
    iy = np.maximum(0, np.minimum(box[3], boxes[:,3]) - np.maximum(box[1], boxes[:,1]))
    return ix*iy

def crop(img, ann, crop_area, in_crop_threshold):
    crop_area = np.array(crop_area)
    bboxes = ann["bboxes"]
    landmarks = ann["landmarks"]
    landmarks_mask = ann["landmarks_mask"]

    intersections = intersection(bboxes, crop_area)
    bbox_areas = (bboxes[:,2:] - bboxes[:,:2]).prod(axis=1)
    in_crop = (intersections/bbox_areas > in_crop_threshold)

    bboxes = bboxes[in_crop] - np.tile(crop_area[:2], 2)

    landmarks = landmarks[in_crop[landmarks_mask.astype(np.bool)]] - crop_area[:2]
    landmarks_mask = landmarks_mask[in_crop]

    img = img.crop(crop_area)

    return img, {"bboxes": bboxes, "landmarks": landmarks, "landmarks_mask": landmarks_mask}

def resize(img, ann, target_size, bbox_size_threshold):
    scale_factors = np.array(target_size)/np.array(img.size)
    img = img.resize(target_size)

    bboxes = ann["bboxes"]*np.tile(scale_factors, 2)
    landmarks = ann["landmarks"]*scale_factors
    landmarks_mask = ann["landmarks_mask"]

    not_too_small = np.linalg.norm(bboxes[:,2:] - bboxes[:,:2], axis=1) > bbox_size_threshold

    bboxes = bboxes[not_too_small]
    landmarks = landmarks[not_too_small[landmarks_mask.astype(np.bool)]]
    landmarks_mask = landmarks_mask[not_too_small]

    return img, {"bboxes": bboxes, "landmarks": landmarks, "landmarks_mask": landmarks_mask}

def generate_semirandom_crop(img, ann, target_size, p_random, max_magnify, min_bbox_size, bbox_size_threshold, in_crop_threshold):
    target_size = np.array(target_size)

    bboxes = ann["bboxes"]
    if torch.rand(1).item() > p_random and len(bboxes) > 0:
        bboxes = ann["bboxes"]
        bbox = bboxes[torch.randint(len(bboxes), size=(1,)).item()]
        bbox_size = np.linalg.norm(bbox[2:] - bbox[:2])
        target_bbox_size = min_bbox_size + torch.rand(1).item()*min(np.linalg.norm(target_size) - min_bbox_size, max_magnify*bbox_size - min_bbox_size)
        rescaling_factor = target_bbox_size / bbox_size
        pivot = 0.5*(bbox[:2] + bbox[2:]) + 0.1*(2*torch.rand(2).numpy() - 1)*bbox_size
    else:
        pivot = np.array(img.size)*(0.2 + 0.6*torch.rand(1).item())
        rescaling_factor = 0.5 + 1.5*torch.rand(1).item()

    if rescaling_factor > 1:
        img, ann = crop(img, ann, np.concatenate([pivot - 0.5*target_size/rescaling_factor, pivot + 0.5*target_size/rescaling_factor]), in_crop_threshold)
        img, ann = resize(img, ann, target_size, bbox_size_threshold)
    else:
        img, ann = resize(img, ann, (np.array(img.size)*rescaling_factor).astype(np.int), bbox_size_threshold)
        upper_left = (pivot*rescaling_factor - 0.5*target_size).astype(np.int)
        img, ann = crop(img, ann, np.concatenate([upper_left, upper_left + target_size]), in_crop_threshold)

    return img, ann

class OnFaceCropper:
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("target_size", (256,)*2),
            ("p_random_crop", 0.1),
            ("max_magnify", 2),
            ("min_bbox_size", 10),
            ("bbox_size_threshold", 5),
            ("in_crop_threshold", 0.3),
        ])

    def __init__(self, config=None):
        config = prepare_config(self, config)
        for k, v in config.items():
            self.__dict__[k] = v

    def __call__(self, img, ann):
        img, ann = generate_semirandom_crop(img, ann, self.target_size, self.p_random_crop,
                    self.max_magnify, self.min_bbox_size, self.bbox_size_threshold, self.in_crop_threshold)
        return img, ann


class Collator:
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("grid_h", 1),
            ("grid_w", 1),
        ])

    def __init__(self, config=None):
        config = prepare_config(self, config)
        self.grid_h = config["grid_h"]
        self.grid_w = config["grid_w"]

    def __call__(self, samples):
        images, annotations = zip(*samples)
        bboxes = [a["bboxes"] for a in annotations]
        landmarks = [a["landmarks"] for a in annotations]
        landmarks_mask = [a["landmarks_mask"] for a in annotations]

        grid_h, grid_w = self.grid_h, self.grid_w

        assert (len(samples) % (grid_h * grid_w) == 0)

        imsize = images[0].size

        images = list(map(to_tensor, images))
        images = [torch.cat(images[i:i+grid_w], dim=2) for i in range(0, len(images), grid_w)]
        images = [torch.cat(images[i:i+grid_h], dim=1) for i in range(0, len(images), grid_h)]

        bboxes = list(map(lambda x: x[1] + np.array([imsize[0] * (x[0]%grid_w), imsize[1] * ((x[0]//grid_w)%grid_h)]*2), enumerate(bboxes)))
        bboxes = [np.concatenate(bboxes[i:i+grid_w], axis=0) for i in range(0, len(bboxes), grid_w)]
        bboxes = [np.concatenate(bboxes[i:i+grid_h], axis=0) for i in range(0, len(bboxes), grid_h)]
        bboxes = list(map(torch.tensor, bboxes))

        landmarks = list(map(lambda x: x[1] + np.array([imsize[0] * (x[0]%grid_w), imsize[1] * ((x[0]//grid_w)%grid_h)]), enumerate(landmarks)))
        landmarks = [np.concatenate(landmarks[i:i+grid_w], axis=0) for i in range(0, len(landmarks), grid_w)]
        landmarks = [np.concatenate(landmarks[i:i+grid_h], axis=0) for i in range(0, len(landmarks), grid_h)]
        landmarks = list(map(torch.tensor, landmarks))

        landmarks_mask = [np.concatenate(landmarks_mask[i:i+grid_w], axis=0) for i in range(0, len(landmarks_mask), grid_w)]
        landmarks_mask = [np.concatenate(landmarks_mask[i:i+grid_h], axis=0) for i in range(0, len(landmarks_mask), grid_h)]
        landmarks_mask = list(map(lambda x: torch.tensor(x).bool(), landmarks_mask))

        annotations = [{"bboxes": _bboxes, "landmarks": _landmarks, "landmarks_mask": _landmarks_mask}\
                for _bboxes, _landmarks, _landmarks_mask in zip(bboxes, landmarks, landmarks_mask)]

        return WIDERBatch(list(zip(images, annotations)))
