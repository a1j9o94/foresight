"""Main experiment runner for Foresight experiments."""

import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from .config import EXPERIMENT_REGISTRY, get_experiment_config
from .results import ResultsWriter


class ExperimentRunner:
    """Orchestrates experiment execution with tracking and results."""

    def __init__(
        self,
        experiment_id: str,
        results_dir: str | Path = "/results",
        wandb_project: str = "foresight",
        instance_id: str | None = None,
    ):
        """Initialize the experiment runner.

        Args:
            experiment_id: ID of the experiment to run (e.g., 'c1-vlm-latent-sufficiency')
            results_dir: Base directory for results storage
            wandb_project: W&B project name
            instance_id: Unique identifier for this runner instance
        """
        self.experiment_id = experiment_id
        self.results_dir = Path(results_dir) / experiment_id
        self.wandb_project = wandb_project
        self.instance_id = instance_id or f"modal-{uuid.uuid4().hex[:8]}"

        # Get experiment config
        try:
            self.config = get_experiment_config(experiment_id)
        except ValueError as e:
            raise ValueError(f"Unknown experiment: {experiment_id}") from e

        # Initialize results writer
        self.results = ResultsWriter(self.results_dir, experiment_id)
        self.results.set_executed_by(self.instance_id)

        # Set success criteria from config
        if "success_metrics" in self.config:
            criteria = {
                f"{metric}_threshold": vals.get("acceptable", vals.get("target", 0))
                for metric, vals in self.config["success_metrics"].items()
            }
            self.results.set_success_criteria(criteria)

        # W&B run handle (set when initialized)
        self._wandb_run = None

        # Sub-experiment handlers registry
        self._handlers: dict[str, Callable] = {}

    def register_handler(self, sub_exp_id: str, handler: Callable):
        """Register a handler function for a sub-experiment.

        Args:
            sub_exp_id: Sub-experiment ID (e.g., 'e1_1')
            handler: Function to run the sub-experiment.
                     Should accept (runner: ExperimentRunner) and return dict with:
                     - finding: str
                     - metrics: dict[str, float]
                     - artifacts: list[str] (optional)
        """
        self._handlers[sub_exp_id] = handler

    def init_wandb(self, **kwargs):
        """Initialize W&B for this experiment."""
        import wandb

        tags = [
            self.experiment_id.split("-")[0],  # e.g., 'c1'
            self.experiment_id,
            self.instance_id,
        ]

        self._wandb_run = wandb.init(
            project=self.wandb_project,
            name=f"{self.experiment_id}",
            tags=tags,
            config={
                "experiment_id": self.experiment_id,
                "instance_id": self.instance_id,
                "success_metrics": self.config.get("success_metrics", {}),
            },
            **kwargs,
        )
        return self._wandb_run

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log metrics to W&B."""
        if self._wandb_run:
            import wandb

            wandb.log(metrics, step=step)

    def log_artifact(self, name: str, path: str | Path, type: str = "result"):
        """Log an artifact to W&B."""
        if self._wandb_run:
            import wandb

            artifact = wandb.Artifact(name, type=type)
            artifact.add_file(str(path))
            self._wandb_run.log_artifact(artifact)

    def run_sub_experiment(
        self,
        sub_exp_id: str,
        handler: Callable | None = None,
    ) -> dict[str, Any]:
        """Run a single sub-experiment.

        Args:
            sub_exp_id: Sub-experiment ID
            handler: Handler function (optional, uses registered handler if not provided)

        Returns:
            dict with finding, metrics, and artifacts
        """
        # Get handler
        if handler is None:
            handler = self._handlers.get(sub_exp_id)
        if handler is None:
            raise ValueError(f"No handler registered for sub-experiment: {sub_exp_id}")

        print(f"\n{'='*60}")
        print(f"Running sub-experiment: {sub_exp_id}")
        print(f"{'='*60}\n")

        # Mark as in progress
        self.results.start_sub_experiment(sub_exp_id)

        try:
            # Run the handler
            result = handler(self)

            # Validate result structure
            if not isinstance(result, dict):
                raise ValueError(f"Handler must return a dict, got {type(result)}")
            if "finding" not in result:
                result["finding"] = "No finding recorded"
            if "metrics" not in result:
                result["metrics"] = {}
            if "artifacts" not in result:
                result["artifacts"] = []

            # Record results - handler completed successfully
            self.results.complete_sub_experiment(
                sub_exp_id,
                finding=result["finding"],
                metrics=result["metrics"],
                artifacts=result["artifacts"],
            )

            # Log to W&B
            prefixed_metrics = {f"{sub_exp_id}/{k}": v for k, v in result["metrics"].items()}
            self.log_metrics(prefixed_metrics)

            print(f"\n[SUCCESS] Sub-experiment {sub_exp_id} completed")
            print(f"  Finding: {result['finding']}")
            print(f"  Metrics: {result['metrics']}")

            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.results.fail_sub_experiment(sub_exp_id, error_msg)

            print(f"\n[FAILED] Sub-experiment {sub_exp_id} failed")
            print(f"  Error: {error_msg}")

            # Re-raise to stop execution if needed
            raise

    def run_all(self, sub_experiments: list[str] | None = None) -> dict[str, Any]:
        """Run all sub-experiments in sequence.

        Args:
            sub_experiments: List of sub-experiment IDs to run.
                            If None, runs all from config.

        Returns:
            dict with overall results
        """
        if sub_experiments is None:
            sub_experiments = self.config.get("sub_experiments", [])

        print(f"\n{'#'*60}")
        print(f"# Starting experiment: {self.experiment_id}")
        print(f"# Instance: {self.instance_id}")
        print(f"# Sub-experiments: {len(sub_experiments)}")
        print(f"{'#'*60}\n")

        self.results.set_status("in_progress")

        results_summary = {"completed": [], "failed": [], "skipped": []}
        all_metrics = {}

        for sub_exp_id in sub_experiments:
            if sub_exp_id not in self._handlers:
                print(f"\n[SKIP] No handler for {sub_exp_id}, skipping")
                self.results.skip_sub_experiment(sub_exp_id, "No handler registered")
                results_summary["skipped"].append(sub_exp_id)
                continue

            try:
                result = self.run_sub_experiment(sub_exp_id)
                results_summary["completed"].append(sub_exp_id)
                all_metrics.update(result.get("metrics", {}))
            except Exception as e:
                results_summary["failed"].append(sub_exp_id)
                print(f"Error in {sub_exp_id}: {e}")
                # Continue to next sub-experiment

        # Finalize status and compute assessment
        if results_summary["failed"]:
            # Code didn't work - no hypothesis assessment possible
            self.results.set_status("failed")
            self.results.set_recommendation(
                "investigate",
                f"{len(results_summary['failed'])} sub-experiments failed. Fix the code and re-run before assessing hypothesis."
            )
        else:
            # Code worked - assess hypothesis based on metrics
            self.results.complete()
            self._compute_assessment(all_metrics)

        # Print summary
        print(f"\n{'#'*60}")
        print(f"# Experiment Complete: {self.experiment_id}")
        print(f"# Completed: {len(results_summary['completed'])}")
        print(f"# Failed: {len(results_summary['failed'])}")
        print(f"# Skipped: {len(results_summary['skipped'])}")
        print(f"# Status: {self.results.results.status}")
        print(f"# Recommendation: {self.results.results.recommendation}")
        if results_summary["failed"]:
            print(f"#")
            print(f"# CODE ERRORS: {len(results_summary['failed'])} sub-experiments threw exceptions!")
            print(f"# Fix and re-run: {', '.join(results_summary['failed'])}")
        print(f"{'#'*60}\n")

        return results_summary

    def _compute_assessment(self, metrics: dict[str, float]):
        """Compute hypothesis assessment based on collected metrics.

        Only called when all sub-experiments completed successfully (code worked).
        Determines whether hypothesis is supported (proceed) or not (pivot).
        """
        success_metrics = self.config.get("success_metrics", {})
        if not success_metrics:
            # No criteria defined, can't assess
            self.results.set_assessment(
                success_criteria_met=False,
                achieved_metrics=metrics,
                confidence="low",
                confidence_notes="No success criteria defined for this experiment.",
            )
            self.results.set_recommendation("investigate", "No success criteria defined.")
            return

        achieved = {}
        all_met = True
        for metric_name, thresholds in success_metrics.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                achieved[f"{metric_name}_achieved"] = value

                # Check against acceptable threshold
                acceptable = thresholds.get("acceptable", thresholds.get("target"))
                # Determine if lower or higher is better based on target vs failure
                target = thresholds.get("target", 0)
                failure = thresholds.get("failure", 1)

                if target < failure:
                    # Lower is better (e.g., LPIPS)
                    met = value <= acceptable
                else:
                    # Higher is better (e.g., accuracy)
                    met = value >= acceptable

                if not met:
                    all_met = False

        # Determine confidence based on completeness
        n_measured = len(achieved)
        n_required = len(success_metrics)

        if n_measured == n_required:
            confidence = "high"
            confidence_notes = f"Measured {n_measured}/{n_required} required metrics."
        elif n_measured >= n_required / 2:
            confidence = "medium"
            confidence_notes = f"Measured {n_measured}/{n_required} required metrics."
        else:
            confidence = "low"
            confidence_notes = f"Measured {n_measured}/{n_required} required metrics."

        self.results.set_assessment(
            success_criteria_met=all_met,
            achieved_metrics=achieved,
            confidence=confidence,
            confidence_notes=confidence_notes,
        )

        # Set recommendation based on hypothesis assessment
        if all_met:
            self.results.set_recommendation("proceed", "Hypothesis supported - all success criteria met.")
        elif confidence == "low":
            self.results.set_recommendation(
                "investigate", "Insufficient data to assess hypothesis."
            )
        else:
            self.results.set_recommendation("pivot", "Hypothesis not supported - some success criteria not met.")

    def finish(self):
        """Finish the experiment and cleanup."""
        if self._wandb_run:
            import wandb

            # Log final results as artifact
            self.log_artifact(
                f"{self.experiment_id}-results",
                self.results.results_file,
                type="results",
            )
            wandb.finish()

        print(f"\nResults saved to: {self.results.results_file}")


def create_stub_handlers(experiment_id: str) -> dict[str, Callable]:
    """Create stub handlers that report 'not implemented' for all sub-experiments.

    This is useful for testing the harness without actual experiment implementations.
    """
    config = get_experiment_config(experiment_id)
    handlers = {}

    for sub_exp_id in config.get("sub_experiments", []):

        def make_stub(exp_id):
            def stub_handler(runner: ExperimentRunner) -> dict:
                print(f"[STUB] Sub-experiment {exp_id} not implemented")
                return {
                    "finding": f"Sub-experiment {exp_id} not yet implemented",
                    "metrics": {},
                    "artifacts": [],
                }

            return stub_handler

        handlers[sub_exp_id] = make_stub(sub_exp_id)

    return handlers


def run_experiment_with_stubs(experiment_id: str, results_dir: str = "/results") -> dict:
    """Run an experiment with stub handlers (for testing harness)."""
    runner = ExperimentRunner(experiment_id, results_dir=results_dir)
    runner.init_wandb()

    # Register stub handlers
    for sub_exp_id, handler in create_stub_handlers(experiment_id).items():
        runner.register_handler(sub_exp_id, handler)

    # Run all
    results = runner.run_all()

    runner.finish()
    return results
