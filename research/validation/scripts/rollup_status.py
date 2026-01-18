#!/usr/bin/env python3
"""
Roll up status from all experiments into a unified view.

Generates:
- status/claims.yaml - Status of all claims and questions
- status/gates.yaml - Go/no-go decision status

NOTE: Experiment/gate definitions are loaded from research/research_plan.yaml
(single source of truth for all experiment configuration)

Usage:
    python rollup_status.py
    python rollup_status.py --output-dir /path/to/status
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _load_research_plan() -> dict:
    """Load experiment and gate definitions from research_plan.yaml."""
    # This file is at: research/validation/scripts/rollup_status.py
    # YAML is at: research/research_plan.yaml
    config_path = Path(__file__).parent.parent.parent / "research_plan.yaml"

    if not config_path.exists():
        print(f"Warning: research_plan.yaml not found at {config_path}")
        return {"experiments": {}, "gates": {}}

    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_experiment_metadata(research_plan: dict) -> tuple[dict, dict]:
    """Build CLAIMS and QUESTIONS dicts from research_plan.yaml."""
    claims = {}
    questions = {}

    for exp_id, exp_config in research_plan.get("experiments", {}).items():
        exp_type = exp_config.get("type", "claim")

        # Build success metrics list from success_criteria keys
        success_metrics = list(exp_config.get("success_criteria", {}).keys())

        entry = {
            "name": exp_id.replace("-", "_"),
            "depends_on": exp_config.get("dependencies", []),
            "success_metrics": success_metrics,
        }

        if exp_type == "claim":
            entry["claim"] = exp_config.get("description", "")
            claims[exp_id] = entry
        elif exp_type == "question":
            entry["question"] = exp_config.get("description", "")
            questions[exp_id] = entry
        elif exp_type == "pivot":
            # Pivots are treated as claims for status tracking
            entry["claim"] = exp_config.get("description", "")
            entry["replaces"] = exp_config.get("replaces", [])
            claims[exp_id] = entry

    return claims, questions


def _build_gates_metadata(research_plan: dict) -> dict:
    """Build GATES dict from research_plan.yaml."""
    gates = {}

    for gate_id, gate_config in research_plan.get("gates", {}).items():
        # Extract phase number from unlocks string (e.g., "Phase 2" -> 2)
        unlocks = gate_config.get("unlocks", "")
        phase_num = 2  # default
        if "Phase " in unlocks:
            try:
                phase_num = int(unlocks.split("Phase ")[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif "Final" in unlocks:
            phase_num = 5

        gates[gate_id] = {
            "description": gate_config.get("description", gate_config.get("name", "")),
            "depends_on": gate_config.get("experiments", []),
            "required_for_phase": phase_num,
        }

    return gates


# Load from research_plan.yaml (single source of truth)
_research_plan = _load_research_plan()
CLAIMS, QUESTIONS = _build_experiment_metadata(_research_plan)
GATES = _build_gates_metadata(_research_plan)


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
