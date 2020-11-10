import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

from torchvision.models import resnet18, resnet34, resnet50, resnet101

from dalib.config import prepare_config

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=nn.ReLU, zero_bn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()

        if zero_bn:
            self.bn.weight.data = torch.zeros_like(self.bn.weight.data)

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
        x = F.interpolate(x, size=(2*x.shape[-2]-1, 2*x.shape[-1]-1), mode='bilinear', align_corners=True)
        return x

class FPN(nn.Module):
    """
    Channels are in order from highest to lowest stride
    """
    def __init__(self, in_channels_list, out_channels_list=None):
        super().__init__()
        self.in_channels_list = in_channels_list
        out_channels_list = out_channels_list if out_channels_list is not None else in_channels_list
        self.out_channels_list = out_channels_list

        self.lateral  = nn.ModuleList([ConvBlock(in_channels, out_channels, kernel_size=1)\
                        for in_channels, out_channels in zip(in_channels_list, out_channels_list)])
        self.merge    = nn.ModuleList([ConvBlock(out_channels, out_channels, kernel_size=3)\
                        for out_channels in out_channels_list[1:]])
        self.upsample = nn.ModuleList([Upsample(out_channels_higher, out_channels_lower)\
                        for out_channels_higher, out_channels_lower in zip(out_channels_list[:-1], out_channels_list[1:])])

    def forward(self, x_list):
        y_list = [lateral(x) for lateral, x in zip(self.lateral, x_list)]
        for idx in range(1, len(y_list)):
            y_high = y_list[idx-1]
            y_low  = y_list[idx]
            y_upsampled = self.upsample[idx-1](y_high)
            y_merged = self.merge[idx-1](y_low[...,:y_upsampled.shape[-2],:y_upsampled.shape[-1]] + y_upsampled)
            y_list[idx] = y_merged
        return y_list

class ResnetBackbone(nn.Module):
    def __init__(self, resnet, expose_layers=[1,2,3,4]):
        super().__init__()
        self.expose_layers = expose_layers
        self.strides = [2*2**i for i in expose_layers]
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        exposed = []
        for i in [1,2,3,4]:
            if max(self.expose_layers) < i:
                break
            x = getattr(self, f'layer{i}')(x)
            if i in self.expose_layers:
                exposed.insert(0, x)

        return exposed


class Head(nn.Module):
    def __init__(self, in_channels, n_points=0):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('conv_1', ConvBlock(in_channels, in_channels, kernel_size=3))
        self.net.add_module('conv_2', ConvBlock(in_channels, in_channels, kernel_size=3))
        self.net.add_module('conv_3', ConvBlock(in_channels, in_channels, kernel_size=3, zero_bn=True))
        self.conv = nn.Conv2d(in_channels, 5 + n_points*2, kernel_size=1)

    def forward(self, x):
        x = x + self.net(x)
        x = self.conv(x)
        scores = torch.sigmoid(x[:,0:1,...])
        deltas = x[:,1:3,...]
        sizes = torch.exp(x[:,3:5,...])
        points = x[:,5:,...]
        return torch.cat([scores,deltas,sizes,points], dim=1)


class ResnetFPN(nn.Module):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ('depth', 18),
            ('backbone_layers', [1,2,3,4]),
            ('fpn_channels_list', None),
            ('pad_inputs', True),
            ('n_points', 0),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        resnets = {18: resnet18, 34: resnet34, 50: resnet50, 101: resnet101}
        self.backbone = ResnetBackbone(resnets[config["depth"]](pretrained=True), config['backbone_layers'])

        self.backbone = self.backbone.eval()
        fpn_in_channels = [y.shape[1] for y in self.backbone(torch.rand(1,3,33,33))]
        self.backbone = self.backbone.train()

        self.fpn = FPN(fpn_in_channels, config['fpn_channels_list'])

        self.stride = self.backbone.strides[0]
        self.offsets = (0,0)

        self.head = Head(fpn_in_channels[-1], n_points=config['n_points'])

        self.pad_inputs = config['pad_inputs']
        ### prefered pad lambda
        self.pad_lambda = lambda side: (1-(side%32))%32

    def forward(self, x):
        if self.pad_inputs:
            x = F.pad(x, (0, self.pad_lambda(x.shape[-1]), 0, self.pad_lambda(x.shape[-2])))
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x[-1])
        return x
