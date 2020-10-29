import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.svhn import SVHN

from dalib.config import prepare_config
from dalib.models import GradientReversalLayer
from dalib.datasets import DomainAdaptationDataset, mnist_transform, svhn_transform


class SVHNToMNISTDataModule(pl.LightningDataModule):
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

        if (
            self.config["domain_adaptation"] and
            len(self.config["train_domains"]) != 2
        ):
            raise ValueError(
        "Domain adaptation is only possible "
        "with 2 training domains"
        )

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


class SVHNToMNISTModel(pl.LightningModule):
    @staticmethod
    def get_default_config():
        return {
            "adaptation_factor": 0.1,
            "domain_adaptation": True,
            "lr": 1e-3,
            "use_only_y_labels_from_source_domain": True
        }

    def __init__(self, config=None):
        super().__init__()
        self.config = prepare_config(self, config)

        self.feature_extractor = models.resnet18()

        fe_output_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()

        self.label_predictor = nn.Sequential(
            nn.Linear(fe_output_dim, 10),
        )

        if self.config["domain_adaptation"]:
            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(lambda_=self.config["adaptation_factor"]),
                nn.Linear(fe_output_dim, 1)
            )

    def forward(self, x):
        x = self.feature_extractor(x)
        y_logits = self.label_predictor(x)

        if self.config["domain_adaptation"]:
            domain_logits = self.domain_classifier(x).squeeze(1)
            return y_logits, domain_logits
        else:
            return y_logits, None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])

        steps_per_epoch = (
            len(self.trainer.datamodule.train_dataset) // self.trainer.datamodule.config["batch_size"]
        ) // self.trainer.accumulate_grad_batches

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.config["lr"],
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch
            ),
            'interval': 'step',
            'frequency': 1,
            'reduce_on_plateau': False,
            'monitor': 'val_loss'
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if self.config["domain_adaptation"]:
            if self.config["use_only_y_labels_from_source_domain"]:
                x, y, domain_label = batch

                is_source_domain = domain_label == 0

                y_logits, domain_logits = self(x)
                loss_y = F.cross_entropy(y_logits, y, reduction='none')[is_source_domain].mean()

                if not is_source_domain.any():
                    loss_y = 0
            else:
                x, y, domain_label = batch
                y_logits, domain_logits = self(x)
                loss_y = F.cross_entropy(y_logits, y)

            loss_d = F.binary_cross_entropy_with_logits(domain_logits, domain_label.float())
            loss = loss_y + loss_d
            self.log('loss_y', loss_y, prog_bar=True)
            self.log('loss_d', loss_d, prog_bar=True)

        else:
            x, y = batch
            y_logits, _ = self(x)
            loss = F.cross_entropy(y_logits, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits, _ = self(x)
        loss = F.cross_entropy(y_logits, y)
        preds = torch.argmax(y_logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
