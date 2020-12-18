import torch
from torch import nn
from torch.nn import functional as F

from ...config import prepare_config

from collections import OrderedDict


class AdversarialBackgroundScoreRegularizationLoss:
    
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("gamma", 2.0),
            #("t", 0.5),
            #("eps", 0.1),
            ("source_domain_label", 0),
        ])

    def __init__(self, config=None):
        self.config = prepare_config(self, config)

    def __call__(self, y_pred_grad_reversed, y_true):
        tar_domain_idx = [
            idx for idx, item in enumerate(y_true)\
            if item["domain_label"] != self.config["source_domain_label"]
        ]

        tar_domain_idx_t = torch.tensor(tar_domain_idx)

        #t = self.config["t"]
        gamma = self.config["gamma"]
        #eps = self.config["eps"]

        tar_scores_t = y_pred_grad_reversed["scores_t"][tar_domain_idx_t]

        #adversarial_loss = t*torch.log(1 - tar_scores_t + eps) + (1 - t)*torch.log(tar_scores_t + eps)
        #adversarial_loss = adversarial_loss*(t - tar_scores_t)**gamma
        adversarial_loss = tar_scores_t**2 * (1 - tar_scores_t)**2

        return adversarial_loss
