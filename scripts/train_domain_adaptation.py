import argparse
import os

import pytorch_lightning as pl

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dalib.config import read_config
from dalib import datasets, models


def get_model_class(config):
    return getattr(models, config["model"]["_type"])


def get_datamodule_class(config):
    return getattr(datasets, config["datamodule"]["_type"])


def main(args):
    config = read_config(args.config_path)
    pl.seed_everything(config["seed"])
    dm = get_datamodule_class(config)(
        args.data_dir,
        config["datamodule"]
    )
    model = get_model_class(config)(
        config["model"]
    )
    tb_logger = TensorBoardLogger(args.log_dir, name=args.name)
    checkpoint_callback = ModelCheckpoint(**config["checkpoint_callback"])
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=args.gpus,
        checkpoint_callback=checkpoint_callback,
        **config["trainer"]
    )
    trainer.logger.log_hyperparams(config)
    trainer.fit(model, datamodule=dm)


def get_args():
    parser = argparse.ArgumentParser(
        description="Train domain adaptation model."
    )
    parser.add_argument(
        "-d", "--data_dir",
        type=str, required=True,
        help="Root directory where to download the data."
    )
    parser.add_argument(
        "-c", "--config_path",
        type=str, required=True,
        help="Path to config."
    )
    parser.add_argument(
        "-l", "--log_dir",
        type=str, default=os.path.join(os.getcwd(), "lightning_logs"),
        help="Directory where to save logs. If not specified then ./lightning_logs is used."
    )
    parser.add_argument(
        "-n", "--name",
        type=str, default=None,
        help="Experiment name. If not specified then no per-experiment log subdirectory is used."
    )
    parser.add_argument(
        "-g", "--gpus",
        nargs='+', type=int, default=None,
        help="Which GPUs to use."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
