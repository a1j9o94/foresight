#!/usr/bin/env python3
"""
Check go/no-go decision gates.

Usage:
    python check_gates.py                    # Check all gates
    python check_gates.py gate_1_reconstruction  # Check specific gate
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_gates(status_dir: Path) -> dict:
    """Load gates.yaml."""
    gates_file = status_dir / "gates.yaml"
    if not gates_file.exists():
        print(f"Error: {gates_file} not found. Run rollup_status.py first.")
        sys.exit(1)
    with open(gates_file) as f:
        return yaml.safe_load(f)


def check_gate(gate_id: str, gates_data: dict) -> dict:
    """Check a specific gate and return detailed status."""
    gates = gates_data.get("gates", {})

    if gate_id not in gates:
        return {"error": f"Unknown gate: {gate_id}"}

    gate = gates[gate_id]
    result = {
        "gate_id": gate_id,
        "description": gate["description"],
        "status": gate["status"],
    }

    if gate["status"] == "passed":
        result["decision"] = gate.get("decision", "proceed")
        result["evidence"] = gate.get("evidence", [])
        result["can_proceed"] = True
    elif gate["status"] == "failed":
        result["decision"] = gate.get("decision", "pivot")
        result["evidence"] = gate.get("evidence", [])
        result["failed_experiments"] = gate.get("failed_experiments", [])
        result["can_proceed"] = False
    else:  # pending
        result["blocking_experiments"] = gate.get("blocking_experiments", [])
        result["can_proceed"] = False

    return result


def main():
    parser = argparse.ArgumentParser(description="Check decision gates")
    parser.add_argument("gate_id", nargs="?", help="Specific gate to check (optional)")
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=Path(__file__).parent.parent / "status",
        help="Path to status directory",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    gates_data = load_gates(args.status_dir)

    if args.gate_id:
        # Check specific gate
        result = check_gate(args.gate_id, gates_data)

        if args.json:
            import json

            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'=' * 60}")
            print(f"Gate: {result['gate_id']}")
            print(f"Description: {result['description']}")
            print(f"{'=' * 60}")
            print(f"\nStatus: {result['status'].upper()}")

            if result.get("can_proceed"):
                print(f"Decision: {result.get('decision', 'proceed')}")
                print("\nEvidence:")
                for ev in result.get("evidence", []):
                    print(f"  - {ev}")
            elif result["status"] == "failed":
                print(f"Decision: {result.get('decision', 'pivot')}")
                print("\nFailed experiments:")
                for exp in result.get("failed_experiments", []):
                    print(f"  - {exp}")
            else:
                print("\nBlocking experiments:")
                for exp in result.get("blocking_experiments", []):
                    print(f"  - {exp}")

        sys.exit(0 if result.get("can_proceed") else 1)

    else:
        # Check all gates
        print(f"\n{'=' * 60}")
        print("DECISION GATES STATUS")
        print(f"{'=' * 60}\n")

        all_passed = True
        for gate_id in gates_data.get("gates", {}).keys():
            result = check_gate(gate_id, gates_data)

            status_icon = {
                "passed": "",
                "failed": "",
                "pending": "",
            }.get(result["status"], "?")

            print(f"{status_icon} {gate_id}: {result['status'].upper()}")
            print(f"   {result['description']}")

            if result["status"] == "pending":
                blocking = result.get("blocking_experiments", [])[:3]
                print(f"   Waiting on: {', '.join(blocking)}")
                all_passed = False
            elif result["status"] == "failed":
                all_passed = False

            print()

        # Summary
        print("-" * 60)
        if all_passed:
            print("All gates PASSED - ready for final evaluation!")
        else:
            # Find next gate to focus on
            for gate_id, gate in gates_data.get("gates", {}).items():
                if gate["status"] == "pending":
                    print(f"Next focus: {gate_id}")
                    break

        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
