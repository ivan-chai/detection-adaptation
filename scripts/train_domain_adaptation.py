import argparse

import pytorch_lightning as pl

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dalib.config import read_config
from dalib import models


def get_model_class(config):
    return getattr(models, config["_type"])


def main(args: argparse.Namespace) -> None:
    config = read_config(args.config_path)

    pl.seed_everything(config["seed"])

    dm = get_model_class(config["datamodule"])(
        args.data_dir,
        config["datamodule"]
    )

    model = get_model_class(config["model"])(
        config["model"]
    )

    tb_logger = TensorBoardLogger(args.log_dir)

    checkpoint_callback = ModelCheckpoint(**config["checkpoint_callback"])

    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=args.gpus,
        checkpoint_callback=checkpoint_callback,
        **config["trainer"]
    )

    trainer.logger.log_hyperparams(config)
    trainer.fit(model, datamodule=dm)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train domain adaptation model.'
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
        type=str, required=True,
        help="Directory where to save logs."
    )
    parser.add_argument(
        "-g", "--gpus",
        type=int, default=0,
        help='Number of gpus to use.'
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
