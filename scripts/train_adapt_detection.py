import torch
import pytorch_lightning as pl

import argparse
import os

from collections import OrderedDict

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dalib.models import DetectionDomainAdaptation
from dalib.datasets import DetectionDataModule
from dalib.config import prepare_config, read_config

def get_args():
    parser = argparse.ArgumentParser(
        description="Train face detection model."
    )
    parser.add_argument(
        "-c", "--config-path",
        type=str, required=True,
        help="Path to config."
    )
    parser.add_argument(
        "--checkpoint",
        type=str, default=None,
        help="Checkpoint to resume training from."
    )
    parser.add_argument(
        "-w", "--weights",
        type=str, default=None,
        help="Weights to start training from."
    )
    parser.add_argument(
        "-m", "--model-configs-dir",
        type=str, default=None,
        help="Path to model config directory. If not specified, is deduced from config path."
    )
    parser.add_argument(
        "-n", "--name",
        type=str, default=None,
        help="Experiment name. If not specified then no per-experiment log subdirectory is used."
    )
    parser.add_argument(
        "-g", "--gpus",
        nargs="+", type=int, default=1,
        help="Which GPUs to use."
    )
    parser.add_argument(
        "-l", "--log-dir",
        type=str, default=os.path.join(os.getcwd(), "lightning_logs"),
        help="Directory where to save logs. If not specified then ./lightning_logs is used."
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=str, required=True,
        help="Root directory with datasets."
    )
    return parser.parse_args()


def get_default_config():
    return {
        "checkpoint_callback": {},
        "datamodule": None,
        "detector_config": None,
        "optimization": None,
        "seed": 0,
        "trainer": {},
    }

def main(args):
    config = prepare_config(get_default_config(), args.config_path)

    pl.seed_everything(config["seed"])

    if args.model_configs_dir is None:
        args.model_configs_dir = os.path.dirname(args.config_path)

    if config["detector_config"]:
        detector_config = read_config(os.path.join(args.model_configs_dir, config["detector_config"]))
    else:
        detector_config = None

    try:
        module_config = config["optimization"].copy()
    except:
        module_config = {}
    if detector_config:
        module_config["detector"] = detector_config

    pl_module = DetectionDomainAdaptation(module_config)

    if args.weights is not None:
        weights = torch.load(args.weights)
        try:
            pl_module.detector.load_state_dict(weights)
        except:
            pl_module.load_state_dict(weights['state_dict'], strict=False)

    pl_datamodule = DetectionDataModule(args.data_dir, config["datamodule"])

    tb_logger = TensorBoardLogger(args.log_dir, name=args.name)
    checkpoint_callback = ModelCheckpoint(**config["checkpoint_callback"])
    trainer = pl.Trainer(
            gpus=args.gpus,
            max_steps=config["optimization"]["total_steps"] - 1,
            logger=tb_logger,
            checkpoint_callback=checkpoint_callback,
            val_check_interval=0.1,
            callbacks=[pl.callbacks.LearningRateMonitor("step")],
            resume_from_checkpoint=args.checkpoint,
            **config["trainer"],
    )

    trainer.fit(pl_module, datamodule=pl_datamodule)

if __name__=="__main__":
    main(get_args())
