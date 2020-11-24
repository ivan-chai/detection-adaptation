import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import to_tensor

import numpy as np

from ..config import prepare_config
from .detection_transforms import OnBboxCropper
from .collection import Collection

from collections import OrderedDict

import pytorch_lightning as pl

from tqdm import tqdm


class CustomBatch:
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


class ImageGridCollator:
    """Collate images into a grid of images.
    If the action is non-trivial, dataset-specific labels are removed.

    Config:
        grid_h: vertical dimension of the grid. Default: 1.
        grid_w: horizontal dimension of the grid. Default: 1.
    """
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
        images, targets = zip(*samples)

        if self.grid_h == 1 and self.grid_w == 1:
            images = torch.stack([to_tensor(image) for image in images])
            targets = list(targets)
            return images, targets

        bboxes, landmarks, has_landmarks = [], [], []
        for t in targets:
            bboxes.append(t["bboxes"])
            landmarks.append(t.get("landmarks", np.empty((0,5,2))))
            has_landmarks.append(t.get("has_landmarks",
                                       np.zeros(len(t["bboxes"])).astype(np.bool)))

        grid_h, grid_w = self.grid_h, self.grid_w

        assert (len(samples) % (grid_h * grid_w) == 0)

        image_size = images[0].size

        images = list(map(to_tensor, images))
        images = [torch.cat(images[i:i+grid_w], dim=2) for i in range(0, len(images), grid_w)]
        images = [torch.cat(images[i:i+grid_h], dim=1) for i in range(0, len(images), grid_h)]

        bboxes = list(map(lambda x: x[1] + np.array([image_size[0] * (x[0]%grid_w), image_size[1] * ((x[0]//grid_w)%grid_h)]*2), enumerate(bboxes)))
        bboxes = [np.concatenate(bboxes[i:i+grid_w], axis=0) for i in range(0, len(bboxes), grid_w)]
        bboxes = [np.concatenate(bboxes[i:i+grid_h], axis=0) for i in range(0, len(bboxes), grid_h)]
        bboxes = list(map(torch.tensor, bboxes))

        landmarks = list(map(lambda x: x[1] + np.array([image_size[0] * (x[0]%grid_w), image_size[1] * ((x[0]//grid_w)%grid_h)]), enumerate(landmarks)))
        landmarks = [np.concatenate(landmarks[i:i+grid_w], axis=0) for i in range(0, len(landmarks), grid_w)]
        landmarks = [np.concatenate(landmarks[i:i+grid_h], axis=0) for i in range(0, len(landmarks), grid_h)]
        landmarks = list(map(torch.tensor, landmarks))

        has_landmarks = [np.concatenate(has_landmarks[i:i+grid_w], axis=0) for i in range(0, len(has_landmarks), grid_w)]
        has_landmarks = [np.concatenate(has_landmarks[i:i+grid_h], axis=0) for i in range(0, len(has_landmarks), grid_h)]
        has_landmarks = list(map(lambda x: torch.tensor(x).bool(), has_landmarks))

        annotations = [{"bboxes": _bboxes, "landmarks": _landmarks, "has_landmarks": _has_landmarks}\
                for _bboxes, _landmarks, _has_landmarks in zip(bboxes, landmarks, has_landmarks)]

        return CustomBatch(list(zip(images, annotations)))


class WeightedConcatDataset(ConcatDataset):
    """Construct a single dataset from a list of datasets.
    The method :meth:`get_sample_weights` provides the array of weights such that
    if dataset is sampled using these weights, the probability to pick a specific dataset
    is proportional to dataset weight passed in argument.

    Args:
        weights: weights of corresponding datasets. If None, this list will be
            proportional to the list of dataset sizes. Default: None.
    """
    def __init__(self, datasets, weights=None):
        super().__init__(datasets)
        ds_sizes = [len(ds) for ds in datasets]
        if weights is None: weights = ds_sizes
        assert(len(weights) == len(datasets))

        sample_weights = np.array([w/s for w, s in zip(weights, ds_sizes)])
        sample_weights = np.repeat(sample_weights, ds_sizes)
        sample_weights = sample_weights/sample_weights.sum()

        self.sample_weights = sample_weights

    def get_sample_weights(self):
        return self.sample_weights


class DetectionDataModule(pl.LightningDataModule):
    """A lightning datamodule wrapper for detection datasets.

    Args:
        - path_dict: a dict of dataset paths. If path for a dataset
          is not provided here, it is inferred from environment variables:
          WIDERFACE_DIR, FDDB_DIR, FACEMASK_DIR. Default: {}

    Config:
        - datasets: list of datasets and weights to use for training.
          Default: [{"name": "WIDERFACE", "weight": 1.0}]
        - transform: config for OnBboxCropper. Default: None
        - collate: config for ImageGridCollator. Default: {"grid_h": 1, "grid_w": 1}
        - num_workers: numbers of workers for data prep. Default: 8
        - batch_size: Default: 32
        - drop_last: whether to drop last (incomplete) batch. Default: True
        - val_samples: number of samples to use for mini-validation during
          training. Default: 512
    """
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("datasets", [
                {"name": "WIDERFACE", "weight": 1.0}
            ]),
            ("transform", None),
            ("collate", {"grid_h": 1, "grid_w": 1}),
            ("num_workers", 8),
            ("batch_size", 32),
            ("drop_last", True),
            ("val_samples", 512),
        ])

    def __init__(self, path_dict={}, config=None):
        super().__init__()
        config = prepare_config(self, config)
        if config["collate"] is not None:
            grid_h = config["collate"].get("grid_h", 1)
            grid_w = config["collate"].get("grid_w", 1)
            assert(config["batch_size"] % (grid_h * grid_w) == 0)
        self.config = config

        ds_names, ds_weights = [], []
        for item in config["datasets"]:
            ds_names.append(item["name"])
            ds_weights.append(item["weight"])

        ds_collection = Collection(path_dict)

        self.train_dataset = []
        self.val_dataset = []
        for name in ds_names:
            train_ds = ds_collection.get_dataset(name, split="train")
            train_ds.transform = OnBboxCropper(config["transform"])
            val_ds = ds_collection.get_dataset(name, split="val")
            val_ds.transform = OnBboxCropper(config["transform"])
            self.train_dataset.append(train_ds)
            self.val_dataset.append(val_ds)

        assert len(self.train_dataset) > 0, "No datasets provided"

        self.train_dataset = WeightedConcatDataset(self.train_dataset, ds_weights)
        self.val_dataset = WeightedConcatDataset(self.val_dataset, ds_weights)

    def _prepare_val_dataloader(self):
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)

        X, y = [], []
        sample_weights = self.val_dataset.get_sample_weights()
        sample_indices = list(WeightedRandomSampler(sample_weights, self.config["val_samples"]))
        for idx in tqdm(sample_indices, desc="Preparing validation mini-dataset"):
            image, target = self.val_dataset[idx]
            X.append(to_tensor(image))
            y.append(target)

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
                          collate_fn = CustomBatch,
                          drop_last=False,
                          shuffle=False)


    def setup(self, stage=None):
        self.val_loader = self._prepare_val_dataloader()

    def train_dataloader(self):
        collate_fn = ImageGridCollator(self.config["collate"])
        sample_weights = self.train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return DataLoader(self.train_dataset,
                          batch_size=self.config["batch_size"],
                          num_workers=self.config["num_workers"],
                          collate_fn=collate_fn,
                          sampler=sampler,
                          drop_last=self.config["drop_last"])

    def val_dataloader(self):
        return self.val_loader
