"""Experiment configuration parsing."""

import re
from dataclasses import dataclass, field
from pathlib import Path


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


# Standard experiment registry
EXPERIMENT_REGISTRY = {
    "c1-vlm-latent-sufficiency": {
        "sub_experiments": ["e1_1", "e1_2", "e1_3", "e1_4", "e1_5", "e1_6"],
        "success_metrics": {
            "lpips": {"target": 0.25, "acceptable": 0.35, "failure": 0.45},
            "ssim": {"target": 0.85, "acceptable": 0.75, "failure": 0.65},
            "spatial_iou": {"target": 0.75, "acceptable": 0.6, "failure": 0.5},
        },
    },
    "c2-adapter-bridging": {
        "sub_experiments": ["e2_1", "e2_2", "e2_3", "e2_4"],
        "success_metrics": {
            "param_efficiency": {"target": 0.9, "acceptable": 0.8, "failure": 0.6},
        },
    },
    "c3-future-prediction": {
        "sub_experiments": ["e3_1", "e3_2", "e3_3"],
        "success_metrics": {
            "cosine_sim_t5": {"target": 0.75, "acceptable": 0.65, "failure": 0.5},
        },
    },
    "c4-pixel-verification": {
        "sub_experiments": ["e4_1", "e4_2", "e4_3"],
        "success_metrics": {
            "accuracy_improvement": {"target": 0.15, "acceptable": 0.1, "failure": 0.05},
        },
    },
    "q1-latent-alignment": {
        "sub_experiments": ["eq1_1", "eq1_2", "eq1_3", "eq1_4", "eq1_5", "eq1_6", "eq1_7"],
        "success_metrics": {
            "linear_probe_r2": {"target": 0.7, "acceptable": 0.5, "failure": 0.3},
            "spearman_correlation": {"target": 0.8, "acceptable": 0.6, "failure": 0.4},
            "neighborhood_recall_at_10": {"target": 0.4, "acceptable": 0.2, "failure": 0.1},
            "cka": {"target": 0.5, "acceptable": 0.4, "failure": 0.2},
        },
    },
    "q2-information-preservation": {
        "sub_experiments": ["e_q2_1", "e_q2_2", "e_q2_3", "e_q2_4", "e_q2_5", "e_q2_6"],
        "success_metrics": {
            "bbox_iou": {"target": 0.8, "acceptable": 0.7, "failure": 0.5},
            "lpips": {"target": 0.25, "acceptable": 0.3, "failure": 0.5},
            "edge_f1": {"target": 0.7, "acceptable": 0.6, "failure": 0.4},
            "mAP": {"target": 0.5, "acceptable": 0.4, "failure": 0.3},
        },
    },
    "q3-temporal-coherence": {
        "sub_experiments": ["e_q3_1", "e_q3_2"],
        "success_metrics": {},
    },
    "q4-training-data": {
        "sub_experiments": ["e_q4_1", "e_q4_2"],
        "success_metrics": {},
    },
    "q5-prediction-horizon": {
        "sub_experiments": ["e_q5_1", "e_q5_2"],
        "success_metrics": {},
    },
}


def get_experiment_config(experiment_id: str) -> dict:
    """Get configuration for an experiment from the registry."""
    if experiment_id not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown experiment: {experiment_id}")
    return EXPERIMENT_REGISTRY[experiment_id]
