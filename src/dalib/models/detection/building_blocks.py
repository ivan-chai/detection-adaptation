import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, activation=nn.ReLU,
                 zero_bn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()

        if zero_bn:
            self.bn.weight.data = 1e-5*torch.ones_like(self.bn.weight.data)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, size=(2*x.shape[-2] - 1, 2*x.shape[-1] - 1),
                mode="bilinear", align_corners=True)
        return x

class FPN(nn.Module):
    """
    Channels are in order from highest to lowest stride.
    """
    def __init__(self, in_channels_list, out_channels_list=None):
        super().__init__()
        self.in_channels_list = in_channels_list
        if out_channels_list is None:
            out_channels_list = in_channels_list
        else:
            assert(len(out_channels_list) == len(in_channels_list))
            out_channels_list = [in_c if out_c is None else out_c\
                    for in_c, out_c in zip(in_channels_list, out_channels_list)]
        self.out_channels_list = out_channels_list

        self.lateral = nn.ModuleList([ConvBlock(in_c, out_c, kernel_size=1)\
                for in_c, out_c in zip(in_channels_list, out_channels_list)])

        self.merge = nn.ModuleList([ConvBlock(out_c, out_c, kernel_size=3)\
                for out_c in out_channels_list[1:]])

        self.upsample = nn.ModuleList([Upsample(out_c_high, out_c_low)\
                for out_c_high, out_c_low\
                in zip(out_channels_list[:-1], out_channels_list[1:])])

    def forward(self, x):
        y = [_lateral(_x) for _lateral, _x in zip(self.lateral, x)]
        for i in range(1, len(y)):
            _y_high = y[i-1]
            _y_low = y[i]
            _y_up = self.upsample[i-1](_y_high)
            _y_merged = self.merge[i-1](_y_low[...,:_y_up.shape[-2],:_y_up.shape[-1]] + _y_up)
            y[i] = _y_merged
        return y
