"""Configuration loading utilities for NNUE-Vision training."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict


class ConfigError(Exception):
    """Exception raised when configuration loading fails."""

    pass


def load_config(config_path: str) -> ModuleType:
    """Load a Python configuration file as a module.

    Args:
        config_path: Path to the Python configuration file.

    Returns:
        The loaded configuration module.

    Raises:
        ConfigError: If the configuration file cannot be loaded.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    if not config_path.suffix == ".py":
        raise ConfigError(
            f"Configuration file must be a Python file (.py): {config_path}"
        )

    try:
        # Load the config file as a module
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None:
            raise ConfigError(f"Failed to create module spec for: {config_path}")

        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        return config_module

    except Exception as e:
        raise ConfigError(f"Failed to load configuration from {config_path}: {e}")


def get_config_dict(config_module: ModuleType) -> Dict[str, Any]:
    """Extract configuration parameters from a module as a dictionary.

    Args:
        config_module: The loaded configuration module.

    Returns:
        Dictionary of configuration parameters (excluding private/special attributes).
    """
    config_dict = {}

    for attr_name in dir(config_module):
        # Skip private/special attributes and imported modules
        if not attr_name.startswith("_") and not isinstance(
            getattr(config_module, attr_name), ModuleType
        ):
            config_dict[attr_name] = getattr(config_module, attr_name)

    return config_dict
