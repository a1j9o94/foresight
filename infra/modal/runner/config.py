"""Experiment configuration parsing.

NOTE: Experiment registry is loaded from research/research_plan.yaml
(single source of truth for all experiment configuration)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SubExperiment:
    """A single sub-experiment within a larger experiment."""

    id: str  # e.g., "e1_1" or "e1_2"
    name: str  # Human-readable name
    objective: str
    protocol: str
    metrics: dict[str, float | None] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)


@dataclass
class SuccessCriteria:
    """Success/failure thresholds for an experiment."""

    metric_name: str
    target: float
    acceptable: float
    failure: float


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment."""

    experiment_id: str
    claim: str
    status: str
    priority: str
    objective: str
    sub_experiments: list[SubExperiment] = field(default_factory=list)
    success_criteria: list[SuccessCriteria] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    @classmethod
    def from_markdown(cls, markdown_path: Path) -> "ExperimentConfig":
        """Parse an experiment config from a markdown file."""
        content = markdown_path.read_text()

        # Extract experiment ID from filename
        experiment_id = markdown_path.stem

        # Parse header section
        claim = _extract_section(content, r"\*\*Claim:\*\*\s*(.+?)(?:\n|$)")
        status = _extract_section(content, r"\*\*Status:\*\*\s*(.+?)(?:\n|$)") or "not_started"
        priority = _extract_section(content, r"\*\*Priority:\*\*\s*(.+?)(?:\n|$)") or "normal"

        # Parse objective (section 1)
        objective = _extract_section(content, r"## 1\. Objective\n\n(.+?)(?:\n\n##|\Z)", re.DOTALL)

        # Parse sub-experiments (section 4)
        sub_experiments = _parse_sub_experiments(content)

        # Parse success criteria (section 5)
        success_criteria = _parse_success_criteria(content)

        # Parse dependencies (section 9)
        dependencies = _parse_dependencies(content)

        return cls(
            experiment_id=experiment_id,
            claim=claim or "",
            status=status,
            priority=priority,
            objective=objective or "",
            sub_experiments=sub_experiments,
            success_criteria=success_criteria,
            dependencies=dependencies,
        )


def _extract_section(content: str, pattern: str, flags: int = 0) -> str | None:
    """Extract a section using regex."""
    match = re.search(pattern, content, flags)
    return match.group(1).strip() if match else None


def _parse_sub_experiments(content: str) -> list[SubExperiment]:
    """Parse sub-experiment definitions from markdown."""
    sub_experiments = []

    # Match experiment headers like "### E1.1: Latent Space Visualization"
    pattern = r"### (E\d+\.\d+): (.+?)\n\n\*\*Objective:\*\*\s*(.+?)(?:\n\n\*\*Protocol:\*\*(.+?))?(?=\n###|\n## |\Z)"
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        exp_id = match[0].lower().replace(".", "_")  # e1.1 -> e1_1
        name = match[1].strip()
        objective = match[2].strip()
        protocol = match[3].strip() if len(match) > 3 and match[3] else ""

        sub_experiments.append(
            SubExperiment(
                id=exp_id,
                name=name,
                objective=objective,
                protocol=protocol,
            )
        )

    return sub_experiments


def _parse_success_criteria(content: str) -> list[SuccessCriteria]:
    """Parse success criteria table from markdown."""
    criteria = []

    # Look for the success metrics table
    # Pattern: | LPIPS | < 0.25 | < 0.35 | > 0.45 |
    pattern = r"\|\s*(\w+(?:\s+\([^)]+\))?)\s*\|\s*([<>]?\s*[\d.]+)\s*\|\s*([<>]?\s*[\d.]+)\s*\|\s*([<>]?\s*[\d.]+)\s*\|"
    matches = re.findall(pattern, content)

    for match in matches:
        metric_name = match[0].strip()
        # Skip header row
        if metric_name.lower() == "metric":
            continue

        try:
            target = _parse_threshold(match[1])
            acceptable = _parse_threshold(match[2])
            failure = _parse_threshold(match[3])

            criteria.append(
                SuccessCriteria(
                    metric_name=metric_name,
                    target=target,
                    acceptable=acceptable,
                    failure=failure,
                )
            )
        except ValueError:
            continue

    return criteria


def _parse_threshold(value: str) -> float:
    """Parse a threshold value like '< 0.25' or '> 0.85'."""
    # Remove comparison operators and whitespace
    cleaned = re.sub(r"[<>=\s]", "", value)
    return float(cleaned)


def _parse_dependencies(content: str) -> list[str]:
    """Parse dependency list from markdown."""
    dependencies = []

    # Look for dependencies section
    dep_section = _extract_section(
        content, r"### 9\.2 Blocks Downstream Work\n\n(.+?)(?:\n###|\n## |\Z)", re.DOTALL
    )

    if dep_section:
        # Extract experiment IDs mentioned
        pattern = r"\*\*(C\d|Q\d)[^*]*\*\*"
        matches = re.findall(pattern, dep_section)
        dependencies = list(set(matches))

    return dependencies


# Load experiment registry from research_plan.yaml (single source of truth)
def _load_experiment_registry() -> dict:
    """Load experiment configurations from research_plan.yaml."""
    # Find the research_plan.yaml - check multiple locations:
    # 1. Modal environment: /research/research_plan.yaml
    # 2. Local development: relative to this file
    possible_paths = [
        Path("/research/research_plan.yaml"),  # Modal mount
        Path(__file__).parent.parent.parent.parent / "research" / "research_plan.yaml",  # Local
    ]

    config_path = None
    for path in possible_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        print(f"Warning: research_plan.yaml not found at {possible_paths}")
        return {}

    with open(config_path) as f:
        plan = yaml.safe_load(f)

    registry = {}
    for exp_id, exp_config in plan.get("experiments", {}).items():
        # Transform success_criteria from YAML format to registry format
        success_metrics = {}
        for metric_name, values in exp_config.get("success_criteria", {}).items():
            success_metrics[metric_name] = {
                "target": values.get("target"),
                "acceptable": values.get("acceptable"),
                "failure": values.get("failure"),
            }

        registry[exp_id] = {
            "sub_experiments": exp_config.get("sub_experiments", []),
            "success_metrics": success_metrics,
        }

    return registry


# Load once at module import
EXPERIMENT_REGISTRY = _load_experiment_registry()


def get_experiment_config(experiment_id: str) -> dict:
    """Get configuration for an experiment from the registry."""
    if experiment_id not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown experiment: {experiment_id}. Available: {list(EXPERIMENT_REGISTRY.keys())}")
    return EXPERIMENT_REGISTRY[experiment_id]
