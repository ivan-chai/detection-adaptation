"""Script for hyperparameter optimization using optuna.

All optimized hyperparameters in the config must be a dict
with `parameter_space` and `parameter_space_arguments` keys.
Only "categorical", "float" and "int" are valid values for `parameter_space` key.

`parameter_space_arguments` is a dict with arguments,
which passed to utility functions provided by :class:`optuna.trial.Trial`
such as :meth:`optuna.trial.Trial.suggest_int`.

https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html

**Example**::

    system:
        subsystem:
            parameter:
                parameter_space: float
                parameter_space_arguments:
                    name: parameter_name
                    low: 0
                    high: 1
                    step: 0.01
"""

import argparse
import os

import optuna
import pytorch_lightning as pl

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dalib.config import prepare_config, read_config
from dalib import datasets, hopt, models


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
        "trainer": {},
        "hyperparameter_optimization": {}
    }
    return default_config


def objective(trial, args, config):
    config = hopt.get_config_with_trial_hyperparameters(config, trial)
    config = prepare_config(get_default_config(), config)
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
        deterministic=True,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=config["checkpoint_callback"]["monitor"])],
        **config["trainer"]
    )
    trainer.fit(model, datamodule=dm)
    return checkpoint_callback.best_model_score


def run_hyperparameter_study(config, args, objective):
    pruner = optuna.pruners.MedianPruner() if config["hyperparameter_optimization"]["pruninig"] else optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction="maximize" if config["checkpoint_callback"]["mode"] == "max" else "minimize",
        pruner=pruner
    )
    study.optimize(
        lambda trial: objective(trial, args, config),
        n_trials=config["hyperparameter_optimization"]["n_trials"]
    )
    hopt.print_summary(study)
    study_path = args.log_dir if args.name is None else os.path.join(args.log_dir, args.name)
    study.trials_dataframe().to_csv(
        os.path.join(study_path, "hyperparameters_study.csv")
    )


def main(args):
    config = read_config(args.config_path)
    run_hyperparameter_study(config, args, objective)


def get_args():
    parser = argparse.ArgumentParser(
        description="Optimize model hyperparameters."
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
