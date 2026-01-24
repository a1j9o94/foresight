"""Foresight Training - GLP trainer, data loaders, and loss functions."""

__version__ = "0.1.0"

# Re-export data module for convenient access
from foresight_training import data
from foresight_training import models

__all__ = ["data", "models"]
