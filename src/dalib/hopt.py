"""Tools for hyperparameter optimization using optuna.

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
OPTIMIZE_KEY = "parameter_space"
PARAMETER_SPACE_TO_TRIAL_METHOD = {
    "categorical": "suggest_categorical",
    "float": "suggest_float",
    "int": "suggest_int"
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


def print_summary(study):
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
