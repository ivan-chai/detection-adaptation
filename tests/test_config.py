#!/usr/bin/env python3
import os
import tempfile
from collections import OrderedDict
from unittest import TestCase, main

from dalib.config import *


class TestModel(object):
    @staticmethod
    def get_default_config(model=None, model_config=None):
        return OrderedDict([
            ("model", model),
            ("model_config", model_config)
        ])

    def __init__(self, config=None):
        self.config = prepare_config(self, config)


class TestConfig(TestCase):
    def test_parser(self):
        config_orig = {
            "model": "some-model",
            "model_config <TestModel>": {"arg1": 5, "arg2": None}
        }
        config_gt = {
            "model": "some-model",
            "model_config": {CONFIG_CLASS: "TestModel", "arg1": 5, "arg2": None}
        }
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "config.yaml")
            write_config(config_orig, path)
            config = read_config(path)
        self.assertEqual(config, config_gt)

    def test_types(self):
        config = {
            "model": "some-model",
            "model_config <TestModel>": {"arg1": 5, "arg2": None}
        }
        model = TestModel(config)
        self.assertEqual(model.config["model"], config["model"])
        self.assertEqual(model.config["model_config"]["arg1"], config["model_config <TestModel>"]["arg1"])
        self.assertEqual(model.config["model_config"]["arg2"], config["model_config <TestModel>"]["arg2"])


if __name__ == "__main__":
    main()
