"""Foresight Core - Shared utilities and configuration."""

__version__ = "0.1.0"

from foresight_core.wandb_utils import (
    ExperimentTracker,
    finish_run,
    get_experiment_config,
    init_experiment,
    load_wandb_config,
    log_artifact,
    log_metrics,
)

__all__ = [
    "ExperimentTracker",
    "finish_run",
    "get_experiment_config",
    "init_experiment",
    "load_wandb_config",
    "log_artifact",
    "log_metrics",
]
