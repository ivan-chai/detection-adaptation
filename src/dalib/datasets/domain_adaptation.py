import bisect

import pytorch_lightning as pl

from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.svhn import SVHN

from dalib.config import prepare_config
from .transforms import mnist_transform, svhn_transform


class DomainAdaptationDataset(ConcatDataset):
    """Dataset as a concatenation of multiple datasets with dataset idx as domain label.

    This class is useful to assemble datasets from different domains.

    If :meth:`__getitem__` of original datasets returns a tuple
    :class:`dalib.utils.datasets.DomainAdaptationDataset` will return a tuple
    concateneted with domain label based on dataset index in list of datasets.

    If :meth:`__getitem__` of original datasets returns a dict
    :class:`dalib.utils.datasets.DomainAdaptationDataset` will return a dict
    with "domain_label" key.

    Args:
        datasets (list or tuple): List of datasets to be concatenated.
    """

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        item = self.__add_domain_label(
            self.datasets[dataset_idx][sample_idx],
            dataset_idx
        )
        return item

    def __add_domain_label(self, item, domain_label):
        if isinstance(item, tuple):
            item = item + (domain_label,)
        elif isinstance(item, list):
            item = item.append(domain_label)
        elif isinstance(item, dict):
            item["domain_label"] = domain_label
        else:
            raise TypeError(f"unsupported type {type(item)}")
        return item


class SVHNToMNISTDataModule(pl.LightningDataModule):
    """Datamodule for domain adaptation from SVHN to MNIST dataset.

    Config:
        - batch_size: How many samples per batch to load.
        - domain_adaptation: If true, domain label is added.
        - num_workers: How many subprocesses to use for data loading.
        - train_domains: List of training domains. Only "svhn" and "mnist" is valid values.
    """

    @staticmethod
    def get_default_config():
        return {
            "batch_size": 128,
            "domain_adaptation": True,
            "num_workers": 0,
            "train_domains": ["svhn", "mnist"]
        }

    def __init__(self, data_dir, config=None):
        super().__init__()
        self.data_dir = data_dir
        self.config = prepare_config(self, config)
        if self.config["domain_adaptation"] and len(self.config["train_domains"]) != 2:
            raise ValueError("Domain adaptation is only possible with 2 training domains")

        for domain in self.config["train_domains"]:
            assert domain in ["svhn", "mnist"]

    def prepare_data(self):
        if "svhn" in self.config["train_domains"]:
            svhn_train = SVHN(
                self.data_dir,
                split="train",
                download=True
            )
        if "mnist" in self.config["train_domains"]:
            mnist_train = MNIST(
                self.data_dir,
                train=True,
                download=True
            )
        mnist_valid = MNIST(
            self.data_dir,
            train=False,
            download=True
        )

    def setup(self, stage=None):
        datasets = []
        if "svhn" in self.config["train_domains"]:
            datasets.append(SVHN(
                self.data_dir,
                split="train",
                transform=svhn_transform
            ))
        if "mnist" in self.config["train_domains"]:
            datasets.append(MNIST(
                self.data_dir,
                train=True,
                transform=mnist_transform
            ))
        if self.config["domain_adaptation"]:
            self.train_dataset = DomainAdaptationDataset(datasets)
        else:
            self.train_dataset = ConcatDataset(datasets)
        self.valid_dataset = MNIST(
            self.data_dir,
            train=False,
            transform=mnist_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"]
        )
