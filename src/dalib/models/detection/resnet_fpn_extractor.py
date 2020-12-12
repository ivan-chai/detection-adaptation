import torch
from torch import nn
from torch.nn import functional as F

from torchvision.models import resnet18, resnet34, resnet50, resnet101

from collections import OrderedDict

from ...config import prepare_config

from ..building_blocks import FPN


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU
}


class ResnetBackbone(nn.Module):

    def __init__(self, resnet, expose_layers=[1,2,3,4], activation=nn.ReLU):
        super().__init__()
        self.expose_layers = expose_layers
        self.strides = [2*2**i for i in expose_layers]
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.act = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.max_stride = 32

        def relu_to_act(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, activation())
        self.apply(relu_to_act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        exposed = []
        for i in [1,2,3,4]:
            if max(self.expose_layers) < i:
                break
            x = getattr(self, f"layer{i}")(x)
            if i in self.expose_layers:
                exposed.insert(0, x)

        return exposed


class ResnetFPNExtractor(nn.Module):
    """A simple resnet+fpn extractor.

    Config:
        depth: the depth of the resnet, should be in [18, 34, 50, 101]. Default: 18
        backbone_layers: set of resnet layers to expose (from 1 to 4). Default: {1,2,3,4}
        backbone_activation: type of activation function, should be "relu" or "leaky_relu".
            Default: "relu".
        fpn_out_channels: list of fpn output channels (length should be equal to
            len(backbone_layers). If element is None, it becomes equal to number of
            channels of the corresponding resnet layer. Default: [None,None,None,128]
        fpn_activation: type of activation function, should be "relu" or "leaky_relu".
            Default: "relu".
        pad_inputs: whether to pad inputs so that strided layers elements are distributed
            evenly (dimension % max_stride == 1). Default: True

    Shape:
        Input: :math:`(B,3,H_{in},W_{in})`

        Output: {
            "embedding_t": :math:`(B,C_{out},H_{out},W_{out})`,
            "offsets": (int, int),
            "stride": int,
        }

        "offsets" determine how upper-leftmost element of embedding tensor projects
        to input image
    """

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("depth", 18),
            ("backbone_layers", {1,2,3,4}),
            ("backbone_activation", "relu"),
            ("fpn_out_channels", [None,None,None,128]),
            ("fpn_activation", "relu"),
            ("pad_inputs", True),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)
        resnets = {18: resnet18, 34: resnet34, 50: resnet50, 101: resnet101}
        self.backbone = ResnetBackbone(
            resnets[config["depth"]](pretrained=True),
            config["backbone_layers"],
            activation=ACTIVATIONS[config["backbone_activation"]]     
        )

        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(torch.rand(1,3,33,33))
        fpn_in_channels = [_x.shape[1] for _x in x]
        self.backbone.train()

        self.fpn = FPN(
            fpn_in_channels,
            config["fpn_out_channels"],
            activation=ACTIVATIONS[config["fpn_activation"]]
        )

        self.pad_inputs = config["pad_inputs"]
        self.pad_lambda = lambda side:\
                (1 - (side % self.backbone.max_stride)) % self.backbone.max_stride

    def forward(self, x):
        if self.pad_inputs:
            w_pad = (1 - (x.shape[-1] % self.backbone.max_stride)) % self.backbone.max_stride
            h_pad = (1 - (x.shape[-2] % self.backbone.max_stride)) % self.backbone.max_stride
            x = F.pad(x, (0, w_pad, 0, h_pad))

        x = self.backbone(x)
        x = self.fpn(x)[-1]
        return {"embedding_t": x, "offsets": (0, 0), "stride": self.backbone.strides[0]}
