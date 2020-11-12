import argparse
import os

import optuna
import pytorch_lightning as pl

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dalib.config import prepare_config, read_config
from dalib import datasets, models


OPTIMIZE_KEY = "parameter_space"
PARAMETER_SPACE_TO_TRIAL_METHOD = {
    "cat": "suggest_categorical",
    "int": "suggest_int",
    "uniform": "suggest_uniform",
    "loguniform": "suggest_loguniform",
    "discrete_uniform": "suggest_discrete_uniform"
}


def get_config_with_trial_hyperparameters(config, trial):
    new_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            if OPTIMIZE_KEY in value:
                new_config[key] = getattr(
                    trial, PARAMETER_SPACE_TO_TRIAL_METHOD[value[OPTIMIZE_KEY]]
                )(**value["parameter_space_arguments"])
            else:
                new_config[key] = get_config_with_trial_hyperparameters(value, trial)
        else:
            new_config[key] = value
    return new_config


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
    config = get_config_with_trial_hyperparameters(config, trial)
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

    checkpoint_callback = ModelCheckpoint(
        os.path.join(
            args.log_dir, args.name, "trial_{}".format(trial.number), "{epoch}"
        ) if args.name is not None else os.path.join(
            args.log_dir, "trial_{}".format(trial.number), "{epoch}"
        ), **config["checkpoint_callback"]
    )
    trainer = pl.Trainer(
        logger=False,
        gpus=args.gpus,
        checkpoint_callback=checkpoint_callback,
        deterministic=True,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=config["checkpoint_callback"]["monitor"])],
        **config["trainer"]
    )
    trainer.fit(model, datamodule=dm)
    return checkpoint_callback.best_model_score


def main(args):
    config = read_config(args.config_path)
    pruner = optuna.pruners.MedianPruner() if config["hyperparameter_optimization"]["pruninig"] else optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction="maximize" if config["checkpoint_callback"]["mode"] == "max" else "minimize",
        pruner=pruner
    )
    study.optimize(
        lambda trial: objective(trial, args, config),
        n_trials=config["hyperparameter_optimization"]["n_trials"]
    )
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


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
