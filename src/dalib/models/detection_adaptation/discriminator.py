import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

from ...config import prepare_config

from ..building_blocks import ConvBlock


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU
}


class PixelwiseDiscriminator(nn.Sequential):
    
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("in_channels", 5),
            ("hidden_channels", [64,32,16]),
            ("activation", "leaky_relu"),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        self.config = config
        
        activation = ACTIVATIONS[config["activation"]]

        channels_list = [config["in_channels"]] + config["hidden_channels"]
        for ind, (in_c, out_c) in enumerate(zip(channels_list[:-1], channels_list[1:])):
            self.add_module(f"convblock_{ind}", ConvBlock(in_c, out_c, kernel_size=3, activation=activation))
        self.add_module(f"final_conv", nn.Conv2d(channels_list[-1], 1, kernel_size=1)) 
