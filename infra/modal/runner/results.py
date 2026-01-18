"""Results writing and validation for experiments."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SubExperimentResult:
    """Result from a single sub-experiment."""

    # Status values:
    # - not_started: Not yet run
    # - in_progress: Currently running
    # - completed: Finished successfully (code worked)
    # - failed: Crashed/threw exception (code broken - fix and re-run)
    # - skipped: Intentionally skipped
    status: str = "not_started"
    finding: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None  # Only set if status is 'failed'


@dataclass
class Assessment:
    """Overall assessment of experiment results."""

    success_criteria_met: bool = False
    achieved_metrics: dict[str, float] = field(default_factory=dict)
    confidence: str = "low"  # high | medium | low
    confidence_notes: str = ""


@dataclass
class ExperimentResults:
    """Complete results for an experiment."""

    experiment_id: str
    claim: str
    status: str = "not_started"  # not_started | in_progress | completed | blocked | failed | pivoted

    # Metadata
    executed_by: str = ""
    started_at: str | None = None
    completed_at: str | None = None

    # Success criteria from plan
    success_criteria: dict[str, float] = field(default_factory=dict)

    # Sub-experiment results
    sub_experiments: dict[str, SubExperimentResult] = field(default_factory=dict)

    # Assessment
    assessment: Assessment | None = None

    # Recommendation
    recommendation: str = "investigate"  # proceed | pivot | investigate | block
    recommendation_notes: str = ""

    # Issues
    blockers: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    # Analysis link
    detailed_analysis: str | None = None


class ResultsWriter:
    """Writes experiment results in the standardized YAML format."""

    def __init__(self, results_dir: Path, experiment_id: str):
        self.results_dir = Path(results_dir)
        self.experiment_id = experiment_id
        self.results_file = self.results_dir / "results.yaml"
        self.artifacts_dir = self.results_dir / "artifacts"

        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)

        # Load or create results
        self._results = self._load_or_create()

    def _load_or_create(self) -> ExperimentResults:
        """Load existing results or create new."""
        if self.results_file.exists():
            with open(self.results_file) as f:
                data = yaml.safe_load(f)
                return self._dict_to_results(data)
        else:
            return ExperimentResults(
                experiment_id=self.experiment_id,
                claim="",
                started_at=datetime.utcnow().isoformat() + "Z",
            )

    def _dict_to_results(self, data: dict) -> ExperimentResults:
        """Convert dict to ExperimentResults."""
        sub_exp_results = {}
        if "results" in data and "experiments" in data["results"]:
            for name, exp_data in data["results"]["experiments"].items():
                sub_exp_results[name] = SubExperimentResult(
                    status=exp_data.get("status", "not_started"),
                    finding=exp_data.get("finding"),
                    metrics=exp_data.get("metrics", {}),
                    artifacts=exp_data.get("artifacts", []),
                    error=exp_data.get("error"),
                )

        assessment = None
        if "assessment" in data and data["assessment"]:
            assessment = Assessment(
                success_criteria_met=data["assessment"].get("success_criteria_met", False),
                achieved_metrics={
                    k: v
                    for k, v in data["assessment"].items()
                    if k not in ("success_criteria_met", "confidence", "confidence_notes")
                },
                confidence=data["assessment"].get("confidence", "low"),
                confidence_notes=data["assessment"].get("confidence_notes", ""),
            )

        return ExperimentResults(
            experiment_id=data.get("experiment_id", self.experiment_id),
            claim=data.get("claim", ""),
            status=data.get("status", "not_started"),
            executed_by=data.get("executed_by", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            success_criteria=data.get("success_criteria", {}),
            sub_experiments=sub_exp_results,
            assessment=assessment,
            recommendation=data.get("recommendation", "investigate"),
            recommendation_notes=data.get("recommendation_notes", ""),
            blockers=data.get("blockers", []),
            issues=data.get("issues", []),
            detailed_analysis=data.get("detailed_analysis"),
        )

    def _results_to_dict(self) -> dict:
        """Convert results to YAML-friendly dict."""
        sub_exp_dict = {}
        for name, result in self._results.sub_experiments.items():
            sub_exp_dict[name] = {
                "status": result.status,
                "finding": result.finding,
                "metrics": result.metrics,
                "artifacts": result.artifacts,
            }
            if result.error:
                sub_exp_dict[name]["error"] = result.error

        assessment_dict: dict[str, Any] = {}
        if self._results.assessment:
            assessment_dict = {
                "success_criteria_met": self._results.assessment.success_criteria_met,
                **self._results.assessment.achieved_metrics,
                "confidence": self._results.assessment.confidence,
                "confidence_notes": self._results.assessment.confidence_notes,
            }

        return {
            "experiment_id": self._results.experiment_id,
            "claim": self._results.claim,
            "status": self._results.status,
            "executed_by": self._results.executed_by,
            "started_at": self._results.started_at,
            "completed_at": self._results.completed_at,
            "success_criteria": self._results.success_criteria,
            "results": {"experiments": sub_exp_dict},
            "assessment": assessment_dict if assessment_dict else None,
            "recommendation": self._results.recommendation,
            "recommendation_notes": self._results.recommendation_notes,
            "blockers": self._results.blockers,
            "issues": self._results.issues,
            "detailed_analysis": self._results.detailed_analysis,
        }

    def save(self):
        """Save results to YAML file."""
        data = self._results_to_dict()
        with open(self.results_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def set_status(self, status: str):
        """Set overall experiment status."""
        self._results.status = status
        self.save()

    def set_claim(self, claim: str):
        """Set the claim being tested."""
        self._results.claim = claim
        self.save()

    def set_executed_by(self, instance_id: str):
        """Set the instance ID running this experiment."""
        self._results.executed_by = instance_id
        self.save()

    def set_success_criteria(self, criteria: dict[str, float]):
        """Set the success criteria from the experiment plan."""
        self._results.success_criteria = criteria
        self.save()

    def start_sub_experiment(self, sub_exp_id: str):
        """Mark a sub-experiment as in progress."""
        if sub_exp_id not in self._results.sub_experiments:
            self._results.sub_experiments[sub_exp_id] = SubExperimentResult()
        self._results.sub_experiments[sub_exp_id].status = "in_progress"
        self._results.status = "in_progress"
        self.save()

    def complete_sub_experiment(
        self,
        sub_exp_id: str,
        finding: str,
        metrics: dict[str, float],
        artifacts: list[str] | None = None,
    ):
        """Mark a sub-experiment as completed with results.

        Only call this if the handler ran successfully without exceptions.
        If the handler threw an exception, use fail_sub_experiment instead.
        """
        if sub_exp_id not in self._results.sub_experiments:
            self._results.sub_experiments[sub_exp_id] = SubExperimentResult()

        self._results.sub_experiments[sub_exp_id].status = "completed"
        self._results.sub_experiments[sub_exp_id].error = None  # Clear any previous error
        self._results.sub_experiments[sub_exp_id].finding = finding
        self._results.sub_experiments[sub_exp_id].metrics = metrics
        self._results.sub_experiments[sub_exp_id].artifacts = artifacts or []
        self.save()

    def fail_sub_experiment(self, sub_exp_id: str, error: str):
        """Mark a sub-experiment as failed."""
        if sub_exp_id not in self._results.sub_experiments:
            self._results.sub_experiments[sub_exp_id] = SubExperimentResult()

        self._results.sub_experiments[sub_exp_id].status = "failed"
        self._results.sub_experiments[sub_exp_id].error = error
        self.save()

    def skip_sub_experiment(self, sub_exp_id: str, reason: str):
        """Mark a sub-experiment as skipped."""
        if sub_exp_id not in self._results.sub_experiments:
            self._results.sub_experiments[sub_exp_id] = SubExperimentResult()

        self._results.sub_experiments[sub_exp_id].status = "skipped"
        self._results.sub_experiments[sub_exp_id].finding = f"Skipped: {reason}"
        self.save()

    def log_metrics(self, sub_exp_id: str, metrics: dict[str, float]):
        """Log metrics for a sub-experiment (incremental update)."""
        if sub_exp_id not in self._results.sub_experiments:
            self._results.sub_experiments[sub_exp_id] = SubExperimentResult()

        self._results.sub_experiments[sub_exp_id].metrics.update(metrics)
        self.save()

    def add_artifact(self, sub_exp_id: str, artifact_path: str):
        """Add an artifact path to a sub-experiment."""
        if sub_exp_id not in self._results.sub_experiments:
            self._results.sub_experiments[sub_exp_id] = SubExperimentResult()

        self._results.sub_experiments[sub_exp_id].artifacts.append(artifact_path)
        self.save()

    def set_assessment(
        self,
        success_criteria_met: bool,
        achieved_metrics: dict[str, float],
        confidence: str = "medium",
        confidence_notes: str = "",
    ):
        """Set the overall assessment."""
        self._results.assessment = Assessment(
            success_criteria_met=success_criteria_met,
            achieved_metrics=achieved_metrics,
            confidence=confidence,
            confidence_notes=confidence_notes,
        )
        self.save()

    def set_recommendation(self, recommendation: str, notes: str = ""):
        """Set the final recommendation."""
        self._results.recommendation = recommendation
        self._results.recommendation_notes = notes
        self.save()

    def add_blocker(self, blocker: str):
        """Add a blocker."""
        self._results.blockers.append(blocker)
        self._results.status = "blocked"
        self.save()

    def add_issue(self, issue: str):
        """Add an issue (non-blocking)."""
        self._results.issues.append(issue)
        self.save()

    def complete(self):
        """Mark the experiment as complete."""
        self._results.completed_at = datetime.utcnow().isoformat() + "Z"
        self._results.status = "completed"
        self.save()

    def get_artifact_path(self, filename: str) -> Path:
        """Get the full path for an artifact."""
        return self.artifacts_dir / filename

    def save_artifact(self, filename: str, content: bytes | str) -> str:
        """Save an artifact and return its relative path."""
        artifact_path = self.artifacts_dir / filename
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(artifact_path, mode) as f:
            f.write(content)
        return f"artifacts/{filename}"

    def save_json_artifact(self, filename: str, data: dict) -> str:
        """Save a JSON artifact and return its relative path."""
        artifact_path = self.artifacts_dir / filename
        with open(artifact_path, "w") as f:
            json.dump(data, f, indent=2)
        return f"artifacts/{filename}"

    @property
    def results(self) -> ExperimentResults:
        """Get the current results."""
        return self._results
