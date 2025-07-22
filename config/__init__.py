"""Configuration package for NNUE-Vision training."""

from .config_loader import ConfigError, load_config

__all__ = ["load_config", "ConfigError"]
