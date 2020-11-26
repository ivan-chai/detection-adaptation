"""All bboxes are in XYXY format."""

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import to_tensor

import numpy as np

from ..config import prepare_config
from .detection_transforms import RandomCropOnBboxAndResize, DatasetNameToDomainLabel
from .collection import DetectionDatasetsCollection

from collections import OrderedDict

import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import transforms as A_pt

from tqdm import tqdm


class DetectionBatch:
    """Utility class for batching samples from detection datasets.

    A batch can be expanded as X, y = batch.
    
    :attr:`X` is a tensor of shape :math:`(B,3,H,W)`.
    :attr:`y` is a list of dictionaries: [
        {
            "bboxes": Numpy array :math:`(N, 4)` (XYXY format),
            ...
        },
        ...
    ]
    """
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
    """Transform a batch of images (and targets) into a batch
    of grids of images (and targets). Images should be in form of
    pytorch tensors.

    The size of the batch should be divisible by grid_h*grid_w.

    If this action is non-trivial (either grid height or grid
    width is not equal to one), dataset-specific labels are dropped,
    and only "bboxes", "keypoints", "has_keypoints" arrays
    are kept.

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
        grid_h, grid_w = self.grid_h, self.grid_w
        assert (len(samples) % (grid_h * grid_w) == 0)

        images = [s.pop("image") for s in samples]
        targets = samples


        if grid_h == 1 and grid_w == 1:
            return DetectionBatch(list(zip(images, targets)))


        image_size = images[0].shape[1:][::-1]

        images = [torch.cat(images[i:i+grid_w], dim=2) for i in range(0, len(images), grid_w)]
        images = [torch.cat(images[i:i+grid_h], dim=1) for i in range(0, len(images), grid_h)]

        bboxes = [t["bboxes"] for t in targets]
        bboxes = list(map(lambda x: x[1] + np.array([image_size[0] * (x[0]%grid_w), image_size[1] * ((x[0]//grid_w)%grid_h)]*2), enumerate(bboxes)))
        bboxes = [np.concatenate(bboxes[i:i+grid_w], axis=0) for i in range(0, len(bboxes), grid_w)]
        bboxes = [np.concatenate(bboxes[i:i+grid_h], axis=0) for i in range(0, len(bboxes), grid_h)]
        bboxes = list(map(torch.tensor, bboxes))

        keypoints = [t["keypoints"] for t in targets if "keypoints" in t.keys()]
        if len(keypoints) > 0:
            n_keypoints = np.array([kp.shape[1] for kp in keypoints])
            n_keypoints = np.unique(n_keypoints)
            assert len(n_keypoints) == 1, "Number of keypoints per detection target should be constant."
            n_keypoints, = n_keypoints

            keypoints = [
                    t.get(
                        "keypoints",
                        np.tile((t["bboxes"][:,2:] + t["bboxes"][:,:2])[:,None,:]/2, (1,n_keypoints,1))
                    ) for t in targets
            ]

            keypoints = list(map(lambda x: x[1] + np.array([image_size[0] * (x[0]%grid_w), image_size[1] * ((x[0]//grid_w)%grid_h)]), enumerate(keypoints)))
            keypoints = [np.concatenate(keypoints[i:i+grid_w], axis=0) for i in range(0, len(keypoints), grid_w)]
            keypoints = [np.concatenate(keypoints[i:i+grid_h], axis=0) for i in range(0, len(keypoints), grid_h)]
            keypoints = list(map(torch.tensor, keypoints))

            has_keypoints = [t.get("has_keypoints", np.zeros(len(t["bboxes"])).astype(np.bool)) for t in targets]
            has_keypoints = [np.concatenate(has_keypoints[i:i+grid_w], axis=0) for i in range(0, len(has_keypoints), grid_w)]
            has_keypoints = [np.concatenate(has_keypoints[i:i+grid_h], axis=0) for i in range(0, len(has_keypoints), grid_h)]
            has_keypoints = list(map(lambda x: torch.tensor(x).bool(), has_keypoints))

            targets = [{"bboxes": _bboxes, "keypoints": _keypoints, "has_keypoints": _has_keypoints}\
                    for _bboxes, _keypoints, _has_keypoints in zip(bboxes, keypoints, has_keypoints)]

        else:
            targets = [{"bboxes": _bboxes} for _bboxes in bboxes]

        return DetectionBatch(list(zip(images, targets)))


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
        data_dir: path to the directory with datasets.

    Config:
        datasets: list of datasets and weights to use for training.
            Default: [{"name": "widerface", "weight": 1.0}].
        crop: config for RandomCropOnBboxAndResize. Default: None.
        augmentations: parameters for certain albumentations augumentations.
            to_gray_prob: probability of applying ToGray. Default: 0.
            random_brightness_prob: probability of applying RandomBrightness.
                Default:None
            random_brightness_limit: limit of RandomBrightness. Default: 0.2.
            motion_blur_prob: probability of applying MotionBlur. Default: 0.
            motion_blur_limit: limit of MotionBlur. Default: 7.
        apply_aug_to_val: whether to apply augmentations when creating
            the mini-validation dataset. Default: False
        collate: config for ImageGridCollator. Default: {"grid_h": 1, "grid_w": 1}.
        num_workers: numbers of workers for data prep. Default: 8.
        batch_size: Default: 32.
        drop_last: whether to drop last (incomplete) batch. Default: True.
        val_samples: number of samples to use for mini-validation during
            training. Default: 512.

    :attr:dataset_names contains a list of names of currently used datasets.
    """

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("datasets", [
                {"name": "widerface", "weight": 1.0}
            ]),
            ("crop", None),
            ("augmentations", None),
            ("apply_aug_to_val", False),
            ("collate", {"grid_h": 1, "grid_w": 1}),
            ("num_workers", 8),
            ("batch_size", 32),
            ("drop_last", True),
            ("val_samples", 512),
        ])

    def _get_transform(self):
        transform = [RandomCropOnBboxAndResize(self.config["crop"])]
        if self.dataset_names:
            transform.append(DatasetNameToDomainLabel(self.dataset_names))
        return transform

    def _get_augmentations(self):
        default_config = {
            "to_gray_prob": 0.0,
            "random_brightness_prob": 0.0,
            "random_brightness_limit": 0.2,
            "motion_blur_prob": 0.0,
            "motion_blur_limit": 7,
        }
        config = prepare_config(default_config, self.config["augmentations"])
        return A.Compose([
            A.ToGray(
                p=config["to_gray_prob"]
            ),
            A.RandomBrightness(
                p=config["random_brightness_prob"],
                limit=config["random_brightness_limit"]
            ),
            A.MotionBlur(
                p=config["motion_blur_prob"],
                blur_limit=config["motion_blur_limit"]
            )
        ])

    def __init__(self, data_dir, config=None):
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
        self.dataset_names = ds_names

        ds_collection = DetectionDatasetsCollection(data_dir)

        self.train_dataset = []
        self.val_dataset = []
        for name in ds_names:
            train_ds = ds_collection.get_dataset(name, split="train")
            train_ds.transform = self._get_transform() + [self._get_augmentations()] + [A_pt.ToTensor()]

            val_ds = ds_collection.get_dataset(name, split="val")
            val_ds.transform = self._get_transform()\
                + ([self._get_augmentations()] if self.config["apply_aug_to_val"] else [])\
                + [A_pt.ToTensor()]

            self.train_dataset.append(train_ds)
            self.val_dataset.append(val_ds)

        assert len(self.train_dataset) > 0, "No datasets provided."

        self.train_dataset = WeightedConcatDataset(self.train_dataset, ds_weights)
        self.val_dataset = WeightedConcatDataset(self.val_dataset, ds_weights)

    def _prepare_val_dataloader(self):
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)

        X, y = [], []
        sample_weights = self.val_dataset.get_sample_weights()
        sample_indices = list(WeightedRandomSampler(sample_weights, self.config["val_samples"]))
        for idx in tqdm(sample_indices, desc="Preparing validation mini-dataset"):
            sample = self.val_dataset[idx]
            image = sample.pop("image")
            target = sample
            X.append(image)
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
                          collate_fn = DetectionBatch,
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
