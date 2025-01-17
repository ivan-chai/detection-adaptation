import torch
import pytorch_lightning as pl

import argparse
import os

from collections import OrderedDict

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dalib.models import DetectionModule
from dalib.datasets import DetectionDataModule
from dalib.config import prepare_config, read_config, write_config


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
        help="Checkpoint to resume training from. Can be in .ckpt or .pth format."
    )
    parser.add_argument(
        "--no-init-optimizer",
        action="store_true",
        help="If set, optimizer state in checkpoint is ignored."
    )
    parser.add_argument(
        "-m", "--model-dir",
        type=str, default=None,
        help="""Directory where to save detector configuration and final weights files.
                If not specified, models/$(experiment_name + "_" + config_name) is used."""
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
        "optimization": None,
        "seed": 0,
        "trainer": {},
    }


def main(args):
    config = prepare_config(get_default_config(), args.config_path)

    pl.seed_everything(config["seed"])

    try:
        module_config = config["optimization"].copy()
    except:
        module_config = {}
    module_config["detector"] = config.get("detector")

    pl_module = DetectionModule(module_config)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        if args.no_init_optimizer or "state_dict" not in checkpoint.keys():
            try:
                pl_module.detector.load_state_dict(checkpoint)
            except:
                pl_module.load_state_dict(checkpoint["state_dict"])
            checkpoint_path = None
        else:
            checkpoint_path = args.checkpoint
    else:
        checkpoint_path = None



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
        resume_from_checkpoint=checkpoint_path,
        **config["trainer"],
    )

    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = args.config_path
        model_dir = os.path.basename(model_dir)
        model_dir = model_dir.split(".")
        model_dir = ".".join(model_dir[:-1]) if len(model_dir) > 1 else model_dir[0]
        if args.name:
            model_dir = args.name + "_" + model_dir
        model_dir = os.path.join(os.getcwd(), "models", model_dir)

    try:
        os.mkdir(model_dir)
    except FileExistsError:
        pass

    trainer.fit(pl_module, datamodule=pl_datamodule)

    detector_config = getattr(pl_module.detector, "config", None)
    write_config(detector_config, os.path.join(model_dir, "config.yaml"))
    torch.save(pl_module.state_dict(), os.path.join(model_dir, "weights.pth"))


if __name__=="__main__":
    main(get_args())
