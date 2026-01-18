"""Example handler demonstrating all required patterns.

This is a minimal working example that shows how to:
1. Log metrics incrementally
2. Save artifacts
3. Return the required format

Copy this file as a starting point for new handlers.
"""

import json
import sys

# Add runner to path (needed when running in Modal)
sys.path.insert(0, "/root")

from runner import ExperimentRunner


def example_handler(runner: ExperimentRunner) -> dict:
    """Minimal working example showing all required patterns.

    This handler demonstrates the complete workflow without doing
    any actual ML work. Use it to verify the infrastructure works.

    Args:
        runner: ExperimentRunner instance for logging and artifact storage

    Returns:
        Dict with 'finding', 'metrics', and 'artifacts' keys
    """
    print("Example handler starting...")

    # 1. Log metrics incrementally (shows in W&B during run)
    for step in range(5):
        runner.log_metrics(
            {
                "example/progress": (step + 1) / 5,
                "example/step": step,
            },
            step=step,
        )
        print(f"  Step {step + 1}/5 complete")

    # 2. Create some example data
    example_data = {
        "experiment": "c1-vlm-latent-sufficiency",
        "handler": "example",
        "message": "This is a test artifact",
        "values": [1, 2, 3, 4, 5],
    }

    # 3. Save artifacts using the runner's results writer
    # JSON artifact (for structured data)
    json_path = runner.results.save_json_artifact("example_output.json", example_data)
    print(f"  Saved JSON artifact: {json_path}")

    # Text artifact (for logs, reports)
    text_content = "Example text artifact\n\nThis shows how to save text files."
    text_path = runner.results.save_artifact("example_log.txt", text_content)
    print(f"  Saved text artifact: {text_path}")

    # 4. Return the required format
    print("Example handler complete!")

    return {
        "finding": "Example handler executed successfully. Infrastructure is working.",
        "metrics": {
            "example_metric_1": 1.0,
            "example_metric_2": 0.95,
            "steps_completed": 5,
        },
        "artifacts": [json_path, text_path],
    }
