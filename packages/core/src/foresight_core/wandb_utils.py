"""W&B utilities for Foresight experiments."""

import os
from pathlib import Path
from typing import Any

import yaml

try:
    import wandb
except ImportError:
    wandb = None


def load_wandb_config(config_name: str = "default") -> dict[str, Any]:
    """Load W&B configuration from configs/wandb/.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        Configuration dictionary
    """
    # Find config file
    config_paths = [
        Path(__file__).parents[4] / "configs" / "wandb" / f"{config_name}.yaml",
        Path.cwd() / "configs" / "wandb" / f"{config_name}.yaml",
    ]

    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(f"W&B config '{config_name}' not found")


def get_experiment_config(experiment_id: str) -> dict[str, Any]:
    """Get W&B config for a specific experiment.

    Args:
        experiment_id: Experiment ID (e.g., 'c1-vlm-latent-sufficiency')

    Returns:
        Configuration dictionary for wandb.init()
    """
    configs = load_wandb_config("experiment_configs")

    # Convert experiment_id to config key
    key = experiment_id.replace("-", "_")

    if key not in configs:
        # Return default config with experiment info
        default = load_wandb_config("default")
        default["config"] = {"experiment_id": experiment_id}
        default["tags"] = [experiment_id.split("-")[0]]  # e.g., 'c1'
        return default

    return configs[key]


def init_experiment(
    experiment_id: str,
    run_name: str | None = None,
    extra_config: dict[str, Any] | None = None,
    **kwargs,
) -> "wandb.sdk.wandb_run.Run | None":
    """Initialize W&B run for an experiment.

    Args:
        experiment_id: Experiment ID (e.g., 'c1-vlm-latent-sufficiency')
        run_name: Optional run name (auto-generated if None)
        extra_config: Additional config to merge
        **kwargs: Additional arguments to wandb.init()

    Returns:
        W&B Run object, or None if wandb unavailable
    """
    if wandb is None:
        print("Warning: wandb not installed, skipping initialization")
        return None

    # Get experiment config
    config = get_experiment_config(experiment_id)

    # Override with environment variables
    if os.environ.get("WANDB_PROJECT"):
        config["project"] = os.environ["WANDB_PROJECT"]
    if os.environ.get("WANDB_ENTITY"):
        config["entity"] = os.environ["WANDB_ENTITY"]

    # Merge extra config
    if extra_config:
        config.setdefault("config", {}).update(extra_config)

    # Override with kwargs
    config.update(kwargs)

    # Set run name
    if run_name:
        config["name"] = run_name

    return wandb.init(**config)


def log_metrics(metrics: dict[str, float], step: int | None = None):
    """Log metrics to W&B (no-op if not initialized)."""
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def log_artifact(
    name: str,
    artifact_type: str,
    path: str | Path,
    description: str | None = None,
):
    """Log an artifact to W&B.

    Args:
        name: Artifact name
        artifact_type: Type (e.g., 'checkpoint', 'plot', 'dataset')
        path: Path to file or directory
        description: Optional description
    """
    if wandb is None or wandb.run is None:
        return

    artifact = wandb.Artifact(name, type=artifact_type, description=description)

    path = Path(path)
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))

    wandb.log_artifact(artifact)


def finish_run(exit_code: int = 0):
    """Finish W&B run."""
    if wandb is not None and wandb.run is not None:
        wandb.finish(exit_code=exit_code)


# Convenience class for experiment tracking
class ExperimentTracker:
    """Context manager for experiment tracking."""

    def __init__(
        self,
        experiment_id: str,
        run_name: str | None = None,
        **kwargs,
    ):
        self.experiment_id = experiment_id
        self.run_name = run_name
        self.kwargs = kwargs
        self.run = None

    def __enter__(self):
        self.run = init_experiment(
            self.experiment_id,
            self.run_name,
            **self.kwargs,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        exit_code = 1 if exc_type is not None else 0
        finish_run(exit_code)
        return False

    def log(self, metrics: dict[str, float], step: int | None = None):
        log_metrics(metrics, step)

    def save_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str | Path,
        description: str | None = None,
    ):
        log_artifact(name, artifact_type, path, description)
