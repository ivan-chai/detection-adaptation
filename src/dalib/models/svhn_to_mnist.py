import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.metrics.functional import accuracy
from torchvision import models

from dalib.config import prepare_config
from dalib.models import GradientReversalLayer


class SVHNToMNISTModel(pl.LightningModule):
    """Digit classification model with domain adaptation.

    Config:
        - adaptation_factor: Multiplier in gradient reversal layer.
        - domain_adaptation: If true, domain adaptation is used.
        - lr: Learning rate.
        - use_only_y_labels_from_source_domain: If true, only class labels
        from the source domain are used for training. Ignored if domain_adaptation is false.
    """

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
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.config["lr"],
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch
            ),
            "interval": "step",
            "frequency": 1,
            "reduce_on_plateau": False,
            "monitor": "val_loss"
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if self.config["domain_adaptation"]:
            if self.config["use_only_y_labels_from_source_domain"]:
                x, y, domain_label = batch
                is_source_domain = domain_label == 0
                y_logits, domain_logits = self(x)

                loss_y = F.cross_entropy(y_logits, y, reduction="none")[is_source_domain].mean()
                if not is_source_domain.any():
                    loss_y = 0
            else:
                x, y, domain_label = batch
                y_logits, domain_logits = self(x)

                loss_y = F.cross_entropy(y_logits, y)
            loss_d = F.binary_cross_entropy_with_logits(domain_logits, domain_label.float())
            loss = loss_y + loss_d

            self.log("loss_y", loss_y, prog_bar=True)
            self.log("loss_d", loss_d, prog_bar=True)
        else:
            x, y = batch
            y_logits, _ = self(x)

            loss = F.cross_entropy(y_logits, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits, _ = self(x)

        loss = F.cross_entropy(y_logits, y)
        preds = torch.argmax(y_logits, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
