import re
from collections import OrderedDict

import yaml


CONFIG_CLASS = "__class__"


class ConfigError(Exception):
    """Exception class for errors in config."""
    pass


def _parse_types(config):
    """Parse types from config keys."""
    typed_param = re.compile(r"^([^<> \t]*)[ \t]*<([^<>]*)>$")
    if isinstance(config, dict):
        result = OrderedDict()
        for k, v in config.items():
            cls = None
            match = typed_param.match(k)
            if match:
                k = match.group(1)
                cls = match.group(2)
            v = _parse_types(v)
            if cls is not None:
                if not isinstance(v, dict):
                    raise ConfigError("Class config must be dict, got: {}".format(type(v)))
                v[CONFIG_CLASS] = cls
            result[k] = v
        return result
    return config


def read_config(filename):
    with open(filename) as fp:
        config = yaml.safe_load(fp)
    return _parse_types(config)


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
    config = _parse_types(config)

    # Check type.
    if CONFIG_CLASS in config:
        if (cls_name is not None) and (cls_name != config[CONFIG_CLASS]):
            raise ConfigError("Type mismatch: expected {}, got {}".format(
                config[CONFIG_CLASS], cls_name))
        del config[CONFIG_CLASS]

    # Merge configs.
    for key in config:
        if key not in default_config:
            raise ConfigError("Unknown parameter {}".format(key))
    new_config = OrderedDict()
    for key, value in default_config.items():
        new_config[key] = config.get(key, value)
    return new_config
