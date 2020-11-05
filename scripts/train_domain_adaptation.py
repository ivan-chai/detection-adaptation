import argparse
import os

import pytorch_lightning as pl

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dalib.config import prepare_config, read_config
from dalib import datasets, models


def get_model_class(config):
    return getattr(models, config["model"]["_type"])


def get_datamodule_class(config):
    return getattr(datasets, config["datamodule"]["_type"])


def get_default_config():
    default_config = {
        "checkpoint_callback": {},
        "datamodule": {
            "_type": "SVHNToMNISTDataModule"
        },
        "model": {
            "_type": "SVHNToMNISTModel"
        },
        "seed": 0,
        "trainer": {}
    }
    return default_config


def main(args):
    config = read_config(args.config_path)
    config = prepare_config(get_default_config(), config)
    pl.seed_everything(config["seed"])
    dm = get_datamodule_class(config)(
        args.data_dir,
        config["datamodule"]
    )
    model = get_model_class(config)(
        config["model"]
    )

    if dm.config["domain_adaptation"] != model.config["domain_adaptation"]:
        raise ValueError("domain_adaptation parameter of datamodule and model must be the same")

    tb_logger = TensorBoardLogger(args.log_dir, name=args.name)
    checkpoint_callback = ModelCheckpoint(**config["checkpoint_callback"])
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=args.gpus,
        checkpoint_callback=checkpoint_callback,
        deterministic=True,
        **config["trainer"]
    )
    trainer.logger.log_hyperparams(config)
    trainer.fit(model, datamodule=dm)


def get_args():
    parser = argparse.ArgumentParser(
        description="Train domain adaptation model."
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=str, required=True,
        help="Root directory where to download the data."
    )
    parser.add_argument(
        "-c", "--config-path",
        type=str, required=True,
        help="Path to config."
    )
    parser.add_argument(
        "-l", "--log-dir",
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
        nargs="+", type=int, default=None,
        help="Which GPUs to use."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
