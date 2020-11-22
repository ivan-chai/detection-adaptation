import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain

from pytorch_lightning.metrics.functional import accuracy
from torchvision import models

from dalib.config import prepare_config
from dalib.models import GradientReversalLayer


class SVHNToMNISTModel(pl.LightningModule):
    """Digit classification model with domain adaptation.

    Config:
        - adaptation_factor: Multiplier in gradient reversal layer.
        - domain_adaptation: If true, domain adaptation is used.
        - domain_classifier_loss: Only "binary_cross_entropy_with_logits" and "mse_loss" is valid values.
        - gan_style_training: If true, alternating training is used for label predictor and domain classifier.
        - lr: Learning rate.
        - target_domain_label: Label of target domain. Ignored if domain_adaptation is false.
        - use_only_y_labels_from_source_domain: If true, only class labels
        from the source domain are used for training. Ignored if domain_adaptation is false.
    """

    @staticmethod
    def get_default_config():
        return {
            "adaptation_factor": 0.1,
            "domain_adaptation": True,
            "domain_classifier_loss": "binary_cross_entropy_with_logits",
            "gan_style_training": False,
            "lr": 1e-3,
            "target_domain_label": 1,
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
        self.domain_classifier_loss = getattr(F, self.config["domain_classifier_loss"])

    def forward(self, x):
        x = self.feature_extractor(x)
        y_logits = self.label_predictor(x)
        if self.config["domain_adaptation"]:
            domain_logits = self.domain_classifier(x).squeeze(1)
            return y_logits, domain_logits
        else:
            return y_logits, None

    def _configure_single_optimizer(self, parameters):
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config["lr"]
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.config["lr"],
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.steps_per_epoch
            ),
            "interval": "step",
            "frequency": 1,
            "reduce_on_plateau": False,
            "monitor": "val_loss"
        }
        return optimizer, scheduler

    def configure_optimizers(self):
        self.steps_per_epoch = (
            len(self.trainer.datamodule.train_dataset) // self.trainer.datamodule.config["batch_size"]
        ) // self.trainer.accumulate_grad_batches
        if self.config["gan_style_training"]:
            optimizer_y, scheduler_y = self._configure_single_optimizer(
                chain(self.feature_extractor.parameters(), self.label_predictor.parameters())
            )
            optimizer_d, scheduler_d = self._configure_single_optimizer(
                chain(self.feature_extractor.parameters(), self.domain_classifier.parameters())
            )
            return [optimizer_y, optimizer_d], [scheduler_y, scheduler_d]
        else:
            optimizer, scheduler = self._configure_single_optimizer(self.parameters())
            return [optimizer], [scheduler]

    def _compute_output_loss(self, y_logits, y, is_source_domain=None, mode="train"):
        if self.config["domain_adaptation"] and is_source_domain is not None:
            output_loss = F.cross_entropy(y_logits, y, reduction="none")[is_source_domain].mean()
            if not is_source_domain.any():
                output_loss = 0
        else:
            output_loss = F.cross_entropy(y_logits, y)
        self.log(f"{mode}_output_loss", output_loss, prog_bar=True)
        return output_loss

    def _compute_discriminator_loss(self, domain_logits, domain_label, mode="train"):
        discriminator_loss = self.domain_classifier_loss(domain_logits, domain_label.float())
        self.log(f"{mode}_discriminator_loss", discriminator_loss, prog_bar=True)
        return discriminator_loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch[:2]
        y_logits, domain_logits = self(x)
        if self.config["domain_adaptation"]:
            domain_label = batch[2]
            if self.config["use_only_y_labels_from_source_domain"]:
                is_source_domain = domain_label != self.config["target_domain_label"]
            else:
                is_source_domain = None

            if optimizer_idx is None:
                loss = self._compute_output_loss(y_logits, y, is_source_domain) + \
                    self._compute_discriminator_loss(domain_logits, domain_label)
                self.log("train_loss", loss)
            elif optimizer_idx == 0:
                loss = self._compute_output_loss(y_logits, y, is_source_domain)
            elif optimizer_idx == 1:
                loss = self._compute_discriminator_loss(domain_logits, domain_label)
        else:
            loss = self._compute_output_loss(y_logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits, domain_logits = self(x)
        output_loss = self._compute_output_loss(y_logits, y, mode="val")
        if domain_logits is not None:
            domain_label = torch.ones_like(y) * self.config["target_domain_label"]
            self._compute_discriminator_loss(domain_logits, domain_label, mode="val")
        preds = torch.argmax(y_logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_acc", acc, prog_bar=True)
        return output_loss
