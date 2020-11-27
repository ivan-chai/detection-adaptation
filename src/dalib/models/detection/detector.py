import torch
from torch import nn

from ...config import prepare_config

from collections import OrderedDict

from .resnet_fpn_extractor import ResnetFPNExtractor
from .predictor import BaselinePredictor
from .postprocessor import LocMaxNMSPostprocessor

EXTRACTORS = {
    "resnet_fpn": ResnetFPNExtractor,        
}

PREDICTORS = {
    "baseline": BaselinePredictor,        
}

POSTPROCESSORS = {
    "loc_max_nms": LocMaxNMSPostprocessor,
}


class Detector(nn.Module):
    """Detector wrapper module.
    Can be constructed from config file via :meth:`get_default_config`.
    Call :classmeth:get_descriptions or :classmeth:get_descriptions_string
    for available components descriptions.

    :meth:`forward` sequentially applies extractor and predictor
    :meth:`predict` sequentially applies extractor, predictor and postprocessor


    If constructed from config:
        Config:
            extractor:
                type: type of extractor. Default: resnet_fpn.
                config: config of extractor.
            predictor:
                type: type of predictor. Default: baseline.
                config: config of predictor.
            postprocessor:
                type: type of postprocessor. Default: loc_max_nms.
                config: config of postprocessor.

        The number if input channels of predictor is determined dynamically ("in_channels" config is set).
        This number is also the number of channels in the embedding tensor, and is saved
        in :attr:`embedding_channels`.

    """

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("extractor", {
                "type": "resnet_fpn",
                "config": {},
            }),
            ("predictor", {
                "type": "baseline",
                "config": {},
            }),
            ("postprocessor", {
                "type": "loc_max_nms",
                "config": {},
            }),
        ])

    def __init__(self, config=None):
        super().__init__()
        config = prepare_config(self, config)

        self.extractor = EXTRACTORS[config["extractor"]["type"]](config["extractor"]["config"])

        self.extractor.eval()
        with torch.no_grad():
            embedding_channels = self.extractor(torch.rand(1,3,129,129))["embedding_t"].shape[1]
            self.embedding_channels = embedding_channels
        config["predictor"]["config"]["in_channels"] = embedding_channels
        self.extractor.train()

        self.predictor = PREDICTORS[config["predictor"]["type"]](config["predictor"]["config"])
        self.postprocessor = POSTPROCESSORS[config["postprocessor"]["type"]](config["postprocessor"]["config"])

    def forward(self, x):
        x = self.extractor(x)
        x = self.predictor(x)
        return x

    def predict(self, x, **postprocessor_kwargs):
        x = self.forward(x)
        x = self.postprocessor(x, **postprocessor_kwargs)
        return x

    @staticmethod
    def get_descriptions():
        desc = {}
        for category, collection\
                in zip(["extractors", "predictors", "postprocessors"], [EXTRACTORS, PREDICTORS, POSTPROCESSORS]):
            desc[category] = [{"name": name, "description": module.__doc__} for name, module in collection.items()]
        return desc

    @classmethod
    def get_descriptions_string(cls):
        desc = cls.get_descriptions()
        desc_str = ""
        for category, collection in desc.items():
            desc_str += f"{category}:\n"
            for item in collection:
                name = item["name"]
                description = item["description"]
                desc_str += f"\t{name}:\n"
                description = str(description)
                for line in description.split("\n"):
                    desc_str += f"\t\t{line}\n"
        return desc_str
