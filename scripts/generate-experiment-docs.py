#!/usr/bin/env python3
"""
Generate experiment documentation from research_plan.yaml.

This script reads the single source of truth (research/research_plan.yaml)
and generates/updates experiment markdown files in research/experiments/.

Auto-generated sections:
- Header metadata (ID, phase, status, type)
- Success criteria table

Hand-written sections (preserved on update):
- Objective
- Sub-experiments
- Protocol details
- Any other custom content

Usage:
    python scripts/generate-experiment-docs.py
    python scripts/generate-experiment-docs.py --experiment p2-hybrid-encoder
    python scripts/generate-experiment-docs.py --dry-run
"""

import argparse
import re
from pathlib import Path

import yaml

# Markers for auto-generated sections
AUTO_START = "<!-- AUTO-GENERATED FROM research_plan.yaml - DO NOT EDIT THIS SECTION -->"
AUTO_END = "<!-- END AUTO-GENERATED -->"


def load_research_plan(project_root: Path) -> dict:
    """Load research_plan.yaml."""
    config_path = project_root / "research" / "research_plan.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"research_plan.yaml not found at {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_header(exp_id: str, exp_config: dict) -> str:
    """Generate the auto-generated header section."""
    lines = [
        f"# Experiment: {exp_config.get('name', exp_id)}",
        "",
        AUTO_START,
        f"**ID:** `{exp_id}`  ",
        f"**Phase:** {exp_config.get('phase', 'N/A')}  ",
        f"**Status:** {exp_config.get('status', 'not_started')}  ",
        f"**Type:** {exp_config.get('type', 'experiment')}  ",
    ]

    # Add replaces field for pivots
    if exp_config.get("replaces"):
        replaces = ", ".join(f"`{r}`" for r in exp_config["replaces"])
        lines.append(f"**Replaces:** {replaces}  ")

    # Add dependencies if any
    if exp_config.get("dependencies"):
        deps = ", ".join(f"`{d}`" for d in exp_config["dependencies"])
        lines.append(f"**Dependencies:** {deps}  ")

    lines.append("")  # Blank line before description
    lines.append(f"**Description:** {exp_config.get('description', '')}")
    lines.append("")

    return "\n".join(lines)


def generate_success_criteria_table(exp_config: dict) -> str:
    """Generate the success criteria table."""
    criteria = exp_config.get("success_criteria", {})
    if not criteria:
        return ""

    lines = [
        "## Success Criteria",
        "",
        "| Metric | Target | Acceptable | Failure | Direction |",
        "|--------|--------|------------|---------|-----------|",
    ]

    for metric_name, values in criteria.items():
        target = values.get("target", "N/A")
        acceptable = values.get("acceptable", "N/A")
        failure = values.get("failure", "N/A")
        direction = values.get("direction", "N/A")
        lines.append(f"| {metric_name} | {target} | {acceptable} | {failure} | {direction} |")

    lines.append("")
    lines.append(AUTO_END)
    lines.append("")

    return "\n".join(lines)


def generate_sub_experiments_section(exp_config: dict) -> str:
    """Generate placeholder for sub-experiments section."""
    sub_exps = exp_config.get("sub_experiments", [])
    if not sub_exps:
        return ""

    lines = [
        "## Sub-experiments",
        "",
        "| ID | Description | Status |",
        "|----|-------------|--------|",
    ]

    for sub_id in sub_exps:
        lines.append(f"| {sub_id} | *Add description* | pending |")

    lines.append("")
    lines.append("*Describe each sub-experiment's objective and protocol below.*")
    lines.append("")

    return "\n".join(lines)


def generate_new_doc(exp_id: str, exp_config: dict) -> str:
    """Generate a complete new experiment document."""
    header = generate_header(exp_id, exp_config)
    criteria_table = generate_success_criteria_table(exp_config)
    sub_exps = generate_sub_experiments_section(exp_config)

    sections = [
        header,
        criteria_table,
        "## Objective",
        "",
        "*Describe the objective of this experiment.*",
        "",
        sub_exps,
        "## Protocol",
        "",
        "*Describe the experimental protocol.*",
        "",
        "## Expected Outcomes",
        "",
        "*Describe what success and failure look like.*",
        "",
        "## Notes",
        "",
        "*Add any additional notes or observations.*",
        "",
    ]

    return "\n".join(sections)


def update_existing_doc(existing_content: str, exp_id: str, exp_config: dict) -> str:
    """Update an existing document, preserving hand-written sections."""
    header = generate_header(exp_id, exp_config)
    criteria_table = generate_success_criteria_table(exp_config)

    # Pattern to find auto-generated section
    auto_pattern = re.compile(
        r"^# Experiment:.*?" + re.escape(AUTO_END),
        re.MULTILINE | re.DOTALL,
    )

    # Check if document has auto-generated markers
    if AUTO_START in existing_content:
        # Replace auto-generated section
        new_auto_section = header + criteria_table.rstrip()
        updated = auto_pattern.sub(new_auto_section, existing_content)
        return updated
    else:
        # Document exists but has no markers - prepend header and add markers
        # Find the first ## section to preserve
        first_section_match = re.search(r"^## ", existing_content, re.MULTILINE)

        if first_section_match:
            # Insert auto-generated header before first section
            preserved = existing_content[first_section_match.start() :]
            return header + criteria_table + preserved
        else:
            # No sections found - prepend header
            return header + criteria_table + existing_content


def process_experiment(
    exp_id: str, exp_config: dict, experiments_dir: Path, dry_run: bool = False
) -> tuple[str, str]:
    """Process a single experiment, generating or updating its doc."""
    doc_path = experiments_dir / f"{exp_id}.md"

    if doc_path.exists():
        existing_content = doc_path.read_text()
        new_content = update_existing_doc(existing_content, exp_id, exp_config)
        action = "updated" if new_content != existing_content else "unchanged"
    else:
        new_content = generate_new_doc(exp_id, exp_config)
        action = "created"

    if not dry_run and action != "unchanged":
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(new_content)

    return action, str(doc_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment docs from research_plan.yaml"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        help="Process only this experiment ID",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Path to project root",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    experiments_dir = project_root / "research" / "experiments"

    print(f"Loading research_plan.yaml from {project_root}")
    research_plan = load_research_plan(project_root)

    experiments = research_plan.get("experiments", {})
    if args.experiment:
        if args.experiment not in experiments:
            print(f"Error: Experiment '{args.experiment}' not found in research_plan.yaml")
            print(f"Available: {', '.join(experiments.keys())}")
            return 1
        experiments = {args.experiment: experiments[args.experiment]}

    if args.dry_run:
        print("\n[DRY RUN] No files will be modified\n")

    print(f"Processing {len(experiments)} experiments...\n")

    results = {"created": [], "updated": [], "unchanged": []}
    for exp_id, exp_config in experiments.items():
        action, path = process_experiment(
            exp_id, exp_config, experiments_dir, dry_run=args.dry_run
        )
        results[action].append(exp_id)
        status_icon = {"created": "+", "updated": "~", "unchanged": "="}[action]
        print(f"  [{status_icon}] {exp_id}")

    print(f"\nSummary:")
    print(f"  Created:   {len(results['created'])}")
    print(f"  Updated:   {len(results['updated'])}")
    print(f"  Unchanged: {len(results['unchanged'])}")

    return 0


if __name__ == "__main__":
    exit(main())
