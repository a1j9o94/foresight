#!/usr/bin/env python3
"""
Roll up status from all experiments into a unified view.

Generates:
- status/claims.yaml - Status of all claims and questions
- status/gates.yaml - Go/no-go decision status

Usage:
    python rollup_status.py
    python rollup_status.py --output-dir /path/to/status
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml


# Experiment metadata
CLAIMS = {
    "c1-vlm-latent-sufficiency": {
        "name": "c1_vlm_latent_sufficiency",
        "claim": "VLM latents contain sufficient information",
        "depends_on": [],
        "success_metrics": ["lpips", "spatial_iou"],
    },
    "c2-adapter-bridging": {
        "name": "c2_adapter_bridging",
        "claim": "Small adapter can bridge latent spaces",
        "depends_on": ["c1-vlm-latent-sufficiency"],
        "success_metrics": ["lpips", "param_efficiency"],
    },
    "c3-future-prediction": {
        "name": "c3_future_prediction",
        "claim": "VLM can predict future states",
        "depends_on": ["c2-adapter-bridging"],
        "success_metrics": ["cosine_similarity", "action_accuracy"],
    },
    "c4-pixel-verification": {
        "name": "c4_pixel_verification",
        "claim": "Pixel verification improves accuracy",
        "depends_on": ["c3-future-prediction"],
        "success_metrics": ["accuracy_improvement", "correlation"],
    },
}

QUESTIONS = {
    "q1-latent-alignment": {
        "name": "q1_latent_alignment",
        "question": "How hard is latent space alignment?",
        "depends_on": [],
    },
    "q2-information-preservation": {
        "name": "q2_information_preservation",
        "question": "Does VLM preserve spatial information?",
        "depends_on": [],
    },
    "q3-temporal-coherence": {
        "name": "q3_temporal_coherence",
        "question": "Can we maintain temporal coherence?",
        "depends_on": ["c2-adapter-bridging"],
    },
    "q4-training-data": {
        "name": "q4_training_data",
        "question": "How much training data do we need?",
        "depends_on": ["c2-adapter-bridging"],
    },
    "q5-prediction-horizon": {
        "name": "q5_prediction_horizon",
        "question": "What prediction horizon is optimal?",
        "depends_on": ["c3-future-prediction"],
    },
}

GATES = {
    "gate_1_reconstruction": {
        "description": "Can we reconstruct video from VLM latents?",
        "depends_on": ["c1-vlm-latent-sufficiency", "q1-latent-alignment", "q2-information-preservation"],
        "required_for_phase": 2,
    },
    "gate_2_bridging": {
        "description": "Can small adapter bridge latent spaces?",
        "depends_on": ["c2-adapter-bridging", "q3-temporal-coherence"],
        "required_for_phase": 3,
    },
    "gate_3_prediction": {
        "description": "Can VLM predict future states?",
        "depends_on": ["c3-future-prediction", "q4-training-data", "q5-prediction-horizon"],
        "required_for_phase": 4,
    },
    "gate_4_verification": {
        "description": "Does pixel verification improve accuracy?",
        "depends_on": ["c4-pixel-verification"],
        "required_for_phase": 5,  # Final
    },
}


def load_experiment_results(experiment_id: str, experiments_dir: Path) -> dict | None:
    """Load results.yaml for an experiment."""
    results_file = experiments_dir / experiment_id / "results.yaml"
    if not results_file.exists():
        return None
    with open(results_file) as f:
        return yaml.safe_load(f)


def get_experiment_status(experiment_id: str, experiments_dir: Path, all_results: dict) -> dict:
    """Get status for a single experiment."""
    results = all_results.get(experiment_id)

    if results is None:
        # Check if blocked by dependencies
        meta = CLAIMS.get(experiment_id) or QUESTIONS.get(experiment_id)
        if meta:
            for dep in meta.get("depends_on", []):
                dep_results = all_results.get(dep)
                if dep_results is None or dep_results.get("status") != "completed":
                    return {
                        "status": "blocked",
                        "blocked_by": dep,
                    }
        return {"status": "not_started"}

    status = {
        "status": results.get("status", "unknown"),
        "last_updated": results.get("completed_at") or results.get("started_at"),
    }

    if results.get("status") == "completed":
        assessment = results.get("assessment", {})
        status["success"] = assessment.get("success_criteria_met", False)

        # Extract key metrics
        metrics = {}
        for exp_data in results.get("results", {}).get("experiments", {}).values():
            metrics.update(exp_data.get("metrics", {}))
        if metrics:
            status["metrics"] = metrics

    elif results.get("status") == "in_progress":
        # Calculate progress
        experiments = results.get("results", {}).get("experiments", {})
        if experiments:
            completed = sum(1 for e in experiments.values() if e.get("status") == "completed")
            total = len(experiments)
            status["progress"] = f"{int(100 * completed / total)}%"
            # Find current experiment
            for name, data in experiments.items():
                if data.get("status") == "in_progress":
                    status["current_experiment"] = name
                    break

    if results.get("recommendation"):
        status["recommendation"] = results["recommendation"]

    if results.get("issues"):
        status["notes"] = "; ".join(results["issues"][:2])  # First 2 issues

    return status


def evaluate_gate(gate_id: str, gate_meta: dict, all_statuses: dict) -> dict:
    """Evaluate a go/no-go gate."""
    depends_on = gate_meta["depends_on"]

    # Check each dependency
    dep_statuses = []
    evidence = []
    blocking = []

    for dep in depends_on:
        status = all_statuses.get(dep, {})
        dep_statuses.append(status.get("status"))

        if status.get("status") == "completed":
            if status.get("success"):
                metrics = status.get("metrics", {})
                metric_str = ", ".join(f"{k}={v:.2f}" for k, v in list(metrics.items())[:2])
                evidence.append(f"{dep}: SUCCESS ({metric_str})")
            else:
                evidence.append(f"{dep}: FAILED")
                blocking.append(dep)
        elif status.get("status") == "in_progress":
            blocking.append(f"{dep} ({status.get('progress', 'in progress')})")
        elif status.get("status") == "blocked":
            blocking.append(f"{dep} (blocked by {status.get('blocked_by')})")
        else:
            blocking.append(f"{dep} (not started)")

    # Determine gate status
    if all(s == "completed" for s in dep_statuses):
        # All complete - check if all succeeded
        all_success = all(
            all_statuses.get(dep, {}).get("success", False) for dep in depends_on
        )
        if all_success:
            return {
                "status": "passed",
                "decision": f"proceed_to_phase_{gate_meta['required_for_phase']}",
                "evidence": evidence,
            }
        else:
            return {
                "status": "failed",
                "decision": "pivot_or_investigate",
                "evidence": evidence,
                "failed_experiments": [
                    dep for dep in depends_on if not all_statuses.get(dep, {}).get("success", False)
                ],
            }
    else:
        return {
            "status": "pending",
            "blocking_experiments": blocking,
        }


def rollup_status(experiments_dir: Path, output_dir: Path):
    """Generate rolled-up status files."""
    # Load all results
    all_results = {}
    for exp_id in list(CLAIMS.keys()) + list(QUESTIONS.keys()):
        results = load_experiment_results(exp_id, experiments_dir)
        if results:
            all_results[exp_id] = results

    # Get status for each claim and question
    claims_status = {}
    for exp_id, meta in CLAIMS.items():
        claims_status[meta["name"]] = get_experiment_status(exp_id, experiments_dir, all_results)

    questions_status = {}
    for exp_id, meta in QUESTIONS.items():
        questions_status[meta["name"]] = get_experiment_status(exp_id, experiments_dir, all_results)

    # Calculate overall progress
    claims_validated = sum(
        1 for s in claims_status.values() if s.get("status") == "completed" and s.get("success")
    )
    questions_answered = sum(1 for s in questions_status.values() if s.get("status") == "completed")

    # Determine current phase
    current_phase = 1
    for gate_id, gate_meta in GATES.items():
        gate_status = evaluate_gate(gate_id, gate_meta, {**claims_status, **questions_status})
        if gate_status["status"] == "passed":
            current_phase = gate_meta["required_for_phase"]

    # Find items requiring human review
    requires_review = []
    for exp_id in list(CLAIMS.keys()) + list(QUESTIONS.keys()):
        results = all_results.get(exp_id)
        if results:
            if results.get("assessment", {}).get("confidence") == "low":
                requires_review.append({
                    "experiment": exp_id,
                    "reason": "Low confidence assessment",
                })
            if results.get("issues"):
                requires_review.append({
                    "experiment": exp_id,
                    "reason": results["issues"][0],
                })

    # Build claims.yaml
    claims_yaml = {
        "claims": claims_status,
        "questions": questions_status,
        "overall": {
            "claims_validated": f"{claims_validated}/4",
            "questions_answered": f"{questions_answered}/5",
            "current_phase": current_phase,
            "requires_human_review": requires_review if requires_review else None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Remove None values
    if claims_yaml["overall"]["requires_human_review"] is None:
        del claims_yaml["overall"]["requires_human_review"]

    # Build gates.yaml
    all_statuses = {**claims_status, **questions_status}
    # Also add by experiment_id for gate evaluation
    for exp_id, meta in CLAIMS.items():
        all_statuses[exp_id] = claims_status[meta["name"]]
    for exp_id, meta in QUESTIONS.items():
        all_statuses[exp_id] = questions_status[meta["name"]]

    gates_yaml = {
        "gates": {
            gate_id: {
                "description": meta["description"],
                **evaluate_gate(gate_id, meta, all_statuses),
            }
            for gate_id, meta in GATES.items()
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "claims.yaml", "w") as f:
        yaml.dump(claims_yaml, f, default_flow_style=False, sort_keys=False)

    with open(output_dir / "gates.yaml", "w") as f:
        yaml.dump(gates_yaml, f, default_flow_style=False, sort_keys=False)

    return claims_yaml, gates_yaml


def main():
    parser = argparse.ArgumentParser(description="Roll up experiment status")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "experiments",
        help="Path to experiments directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "status",
        help="Path to output directory",
    )
    args = parser.parse_args()

    claims, gates = rollup_status(args.experiments_dir, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("FORESIGHT RESEARCH STATUS")
    print("=" * 60)

    print(f"\nClaims validated: {claims['overall']['claims_validated']}")
    print(f"Questions answered: {claims['overall']['questions_answered']}")
    print(f"Current phase: {claims['overall']['current_phase']}")

    print("\n--- Claims ---")
    for name, status in claims["claims"].items():
        status_str = status["status"]
        if status.get("success") is not None:
            status_str += " (SUCCESS)" if status["success"] else " (FAILED)"
        print(f"  {name}: {status_str}")

    print("\n--- Questions ---")
    for name, status in claims["questions"].items():
        print(f"  {name}: {status['status']}")

    print("\n--- Gates ---")
    for name, gate in gates["gates"].items():
        print(f"  {name}: {gate['status']}")
        if gate.get("blocking_experiments"):
            print(f"    Blocking: {', '.join(gate['blocking_experiments'][:3])}")

    if claims["overall"].get("requires_human_review"):
        print("\n--- Requires Human Review ---")
        for item in claims["overall"]["requires_human_review"]:
            print(f"  {item['experiment']}: {item['reason']}")

    print(f"\nStatus written to: {args.output_dir}")


if __name__ == "__main__":
    main()
