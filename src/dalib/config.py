import re
from collections import OrderedDict

import yaml


CONFIG_TYPE = "_type"


class ConfigError(Exception):
    """Exception class for errors in config."""
    pass


def read_config(filename):
    with open(filename) as fp:
        return yaml.safe_load(fp)


def write_config(config, filename):
    with open(filename, "w") as fp:
        yaml.dump(config, fp)


def prepare_config(cls_or_default, config=None):
    """Set defaults and check fields.

    Config is a dictionary of values. Method creates new config using
    default class config. Result config keys are the same as default config keys.

    Args:
        cls_or_default: Class with get_default_config method or default config dictionary.
        config: User-provided config.

    Returns:
        Config dictionary with defaults set.
    """
    if isinstance(cls_or_default, dict):
        default_config = cls_or_default
        cls_name = None
    else:
        default_config = cls_or_default.get_default_config()
        cls_name = type(cls_or_default).__name__
    if isinstance(config, str):
        config = read_config(config)
    elif config is None:
        return default_config
    elif not isinstance(config, dict):
        raise ConfigError("Config dictionary expected, got {}".format(type(config)))

    # Check type.
    if CONFIG_TYPE in config:
        if (cls_name is not None) and (cls_name != config[CONFIG_TYPE]):
            raise ConfigError("Type mismatch: expected {}, got {}".format(
                config[CONFIG_TYPE], cls_name))
        del config[CONFIG_TYPE]

    # Merge configs.
    for key in config:
        if key not in default_config:
            raise ConfigError("Unknown parameter {}".format(key))
    new_config = OrderedDict()
    for key, value in default_config.items():
        new_config[key] = config.get(key, value)
    return new_config
