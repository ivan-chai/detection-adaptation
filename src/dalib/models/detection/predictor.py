import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

from ...config import prepare_config

from .building_blocks import ConvBlock

class BaselinePredictor(nn.Module):
    """A simple prediction head. Uses 3 consecutive 3x3 conv blocks and one
        linear 1x1 conv block for final predictions.

    Config:
        in_channels: number if input channels. Default: 128
        predict_landmarks: whether to predict landmarks. Default: False
        n_landmarks: number of landmarks to predict. Default: 5

    Shape:
        Input: {
            "embedding_t": :math:`(B,C_{in},H_{in},W_{in})`,
            ...
        }

        Output: {
            "scores_t": :math:`(B,H_{in},W_{in})`,
            "deltas_t": :math:`(B, 2, H_{in}, W_{in})`,
            "sizes_t": :math:`(B, 2, H_{in}, W_{in})`,
            "landmarks_t": :math:`(B, n_landmarks, 2, H_{in}, W_{in})`, (optional)
            ...
        }
    """
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("in_channels", 128),
            ("predict_landmarks", False),
            ("n_landmarks", 5),
        ])
    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        for k, v in config.items():
            self.__dict__[k] = v

        in_channels = self.in_channels

        self.net = nn.Sequential()
        self.net.add_module("conv_1", ConvBlock(in_channels, in_channels, kernel_size=3))
        self.net.add_module("conv_2", ConvBlock(in_channels, in_channels, kernel_size=3))
        self.net.add_module("conv_3", ConvBlock(in_channels, in_channels, kernel_size=3))
        self.conv = nn.Conv2d(in_channels, 5 + 2*(self.n_landmarks*self.predict_landmarks), kernel_size=1)

    def forward(self, x):
        e = x["embedding_t"]
        e = self.net(e)
        e = self.conv(e)
        scores = torch.sigmoid(e[:,0,...])
        deltas = e[:,1:3,...]
        sizes = torch.exp(e[:,3:5,...])

        result = {"scores_t": scores, "deltas_t": deltas, "sizes_t": sizes}

        if self.predict_landmarks:
            landmarks = e[:,5:,...].reshape(e.shape[0], self.n_landmarks, 2, *e.shape[-2:])
            result["landmarks_t"] = landmarks

        result.update(x)

        return result
