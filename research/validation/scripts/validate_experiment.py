#!/usr/bin/env python3
"""
Validate an experiment's results.yaml file.

Checks:
1. Schema compliance
2. Artifact existence
3. Metric plausibility
4. Success criteria consistency

Usage:
    python validate_experiment.py <experiment_id>
    python validate_experiment.py c1-vlm-latent-sufficiency
"""

import argparse
import sys
from pathlib import Path

import yaml


# Plausible bounds for common metrics
METRIC_BOUNDS = {
    "lpips": (0.0, 1.0),
    "ssim": (0.0, 1.0),
    "psnr": (0.0, 60.0),
    "fvd": (0.0, 2000.0),
    "spatial_iou": (0.0, 1.0),
    "cosine_similarity": (-1.0, 1.0),
    "r_squared": (0.0, 1.0),
    "accuracy": (0.0, 1.0),
    "correlation": (-1.0, 1.0),
}


def load_results(experiment_dir: Path) -> dict | None:
    """Load results.yaml from experiment directory."""
    results_file = experiment_dir / "results.yaml"
    if not results_file.exists():
        return None
    with open(results_file) as f:
        return yaml.safe_load(f)


def validate_schema(results: dict) -> list[str]:
    """Check required fields are present."""
    errors = []
    required = [
        "experiment_id",
        "claim",
        "status",
        "executed_by",
        "started_at",
        "success_criteria",
        "results",
        "assessment",
        "recommendation",
    ]
    for field in required:
        if field not in results:
            errors.append(f"Missing required field: {field}")

    # Check status is valid
    valid_statuses = ["not_started", "in_progress", "completed", "blocked", "failed", "pivoted"]
    if results.get("status") not in valid_statuses:
        errors.append(f"Invalid status: {results.get('status')}")

    # Check recommendation is valid
    valid_recs = ["proceed", "pivot", "investigate", "block"]
    if results.get("recommendation") not in valid_recs:
        errors.append(f"Invalid recommendation: {results.get('recommendation')}")

    return errors


def validate_artifacts(results: dict, experiment_dir: Path) -> list[str]:
    """Check all referenced artifacts exist."""
    errors = []
    experiments = results.get("results", {}).get("experiments", {})

    for exp_name, exp_data in experiments.items():
        artifacts = exp_data.get("artifacts", [])
        for artifact in artifacts:
            artifact_path = experiment_dir / artifact
            if not artifact_path.exists():
                errors.append(f"Missing artifact: {artifact} (referenced in {exp_name})")

    return errors


def validate_metrics(results: dict) -> list[str]:
    """Check metrics are within plausible bounds."""
    errors = []
    experiments = results.get("results", {}).get("experiments", {})

    for exp_name, exp_data in experiments.items():
        metrics = exp_data.get("metrics", {})
        for metric_name, value in metrics.items():
            # Normalize metric name for lookup
            normalized = metric_name.lower().replace("-", "_")
            if normalized in METRIC_BOUNDS:
                low, high = METRIC_BOUNDS[normalized]
                if not (low <= value <= high):
                    errors.append(
                        f"Implausible metric in {exp_name}: "
                        f"{metric_name}={value} (expected {low}-{high})"
                    )

    return errors


def validate_success_criteria(results: dict) -> list[str]:
    """Check that success_criteria_met matches actual metrics vs thresholds."""
    errors = []
    warnings = []

    criteria = results.get("success_criteria", {})
    assessment = results.get("assessment", {})
    experiments = results.get("results", {}).get("experiments", {})

    # Gather all metrics from all experiments
    all_metrics = {}
    for exp_data in experiments.values():
        all_metrics.update(exp_data.get("metrics", {}))

    # Also check assessment-level metrics
    for key, value in assessment.items():
        if key.endswith("_achieved") and isinstance(value, (int, float)):
            metric_name = key.replace("_achieved", "")
            all_metrics[metric_name] = value

    # Check each criterion
    claimed_success = assessment.get("success_criteria_met", False)
    actual_successes = []

    for criterion_name, threshold in criteria.items():
        # Find corresponding metric
        metric_name = criterion_name.replace("_threshold", "")
        if metric_name in all_metrics:
            actual_value = all_metrics[metric_name]
            # Determine if this is a "lower is better" or "higher is better" metric
            lower_is_better = metric_name.lower() in ["lpips", "fvd", "mse", "loss"]

            if lower_is_better:
                passed = actual_value <= threshold
            else:
                passed = actual_value >= threshold

            actual_successes.append(passed)

            if not passed:
                warnings.append(
                    f"Criterion not met: {metric_name}={actual_value} "
                    f"(threshold: {'<=' if lower_is_better else '>='}{threshold})"
                )

    # Check consistency
    if actual_successes:
        all_passed = all(actual_successes)
        if claimed_success != all_passed:
            errors.append(
                f"Inconsistent success_criteria_met: claimed {claimed_success}, "
                f"but criteria evaluation shows {all_passed}"
            )

    return errors + warnings


def validate_experiment(experiment_id: str, experiments_dir: Path) -> dict:
    """Run all validations on an experiment."""
    experiment_dir = experiments_dir / experiment_id

    result = {
        "experiment_id": experiment_id,
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    # Check directory exists
    if not experiment_dir.exists():
        result["valid"] = False
        result["errors"].append(f"Experiment directory not found: {experiment_dir}")
        return result

    # Load results
    results = load_results(experiment_dir)
    if results is None:
        result["valid"] = False
        result["errors"].append("results.yaml not found")
        return result

    # Run validations
    schema_errors = validate_schema(results)
    artifact_errors = validate_artifacts(results, experiment_dir)
    metric_errors = validate_metrics(results)
    criteria_issues = validate_success_criteria(results)

    # Categorize issues
    all_errors = schema_errors + artifact_errors
    all_warnings = metric_errors + criteria_issues

    result["errors"] = all_errors
    result["warnings"] = all_warnings
    result["valid"] = len(all_errors) == 0

    # Add summary info
    result["status"] = results.get("status", "unknown")
    result["recommendation"] = results.get("recommendation", "unknown")

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate experiment results")
    parser.add_argument("experiment_id", help="Experiment ID (e.g., c1-vlm-latent-sufficiency)")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "experiments",
        help="Path to experiments directory",
    )
    args = parser.parse_args()

    result = validate_experiment(args.experiment_id, args.experiments_dir)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Validation: {args.experiment_id}")
    print(f"{'=' * 60}")

    if result["valid"]:
        print("Status: VALID")
    else:
        print("Status: INVALID")

    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for error in result["errors"]:
            print(f"  - {error}")

    if result["warnings"]:
        print(f"\nWarnings ({len(result['warnings'])}):")
        for warning in result["warnings"]:
            print(f"  - {warning}")

    if not result["errors"] and not result["warnings"]:
        print("\nNo issues found.")

    print(f"\nExperiment status: {result.get('status', 'unknown')}")
    print(f"Recommendation: {result.get('recommendation', 'unknown')}")

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
