# Research Validation System

How we track, validate, and roll up results from parallel research experiments.

## The Problem

Software tests answer: "Does the code work?"
Research validation answers: "Did we learn what we needed to learn?"

For each experiment, we need to:
1. **Track progress** - What's done, what's blocked, what's in progress
2. **Capture results** - Metrics, artifacts, observations in a structured way
3. **Validate claims** - Check that reported numbers match actual artifacts
4. **Make decisions** - Go/no-go gates based on pre-defined criteria

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXPERIMENT EXECUTION                        │
│  (Multiple Claude Code instances working in parallel)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STRUCTURED OUTPUTS                            │
│  Each experiment produces:                                      │
│  - results.yaml (metrics, status, evidence paths)               │
│  - artifacts/ (checkpoints, plots, logs)                        │
│  - notebook.ipynb (analysis, visualizations)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VALIDATION LAYER                              │
│  - Schema validation (are all required fields present?)         │
│  - Artifact verification (do claimed files exist?)              │
│  - Metric bounds checking (are numbers plausible?)              │
│  - Cross-reference checks (do related experiments align?)       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ROLLUP DASHBOARD                              │
│  - Overall status: claims.yaml                                  │
│  - Decision gates: gates.yaml                                   │
│  - Evidence index: artifacts.yaml                               │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
research/
├── validation/
│   ├── README.md              # This file
│   ├── schemas/               # YAML schemas for result validation
│   │   ├── experiment-result.schema.yaml
│   │   └── claim-status.schema.yaml
│   ├── scripts/               # Validation and rollup scripts
│   │   ├── validate_experiment.py
│   │   ├── rollup_status.py
│   │   └── check_gates.py
│   └── status/                # Current status (auto-generated)
│       ├── claims.yaml        # Status of all 4 claims
│       ├── questions.yaml     # Status of all 5 questions
│       └── gates.yaml         # Go/no-go decision status
│
├── experiments/
│   ├── c1-vlm-latent-sufficiency/
│   │   ├── results.yaml       # Structured results
│   │   ├── artifacts/         # Checkpoints, plots, logs
│   │   └── analysis.ipynb     # Detailed analysis
│   └── ... (same for each experiment)
```

## Result File Format

Each experiment produces a `results.yaml`:

```yaml
# research/experiments/c1-vlm-latent-sufficiency/results.yaml
experiment_id: c1-vlm-latent-sufficiency
claim: "VLM latents contain sufficient information"
status: completed | in_progress | blocked | failed

# Who ran this and when
executed_by: "claude-instance-1"
started_at: "2025-01-20T10:00:00Z"
completed_at: "2025-01-25T15:30:00Z"

# Pre-registered success criteria (from experiment plan)
success_criteria:
  lpips_threshold: 0.35
  spatial_iou_threshold: 0.6

# Actual results
results:
  experiments:
    e1_1_latent_visualization:
      status: completed
      finding: "Latents show semantic clustering by action type"
      artifacts:
        - artifacts/e1_1_tsne_plot.png
        - artifacts/e1_1_umap_plot.png

    e1_2_reconstruction_probe:
      status: completed
      metrics:
        lpips: 0.31
        ssim: 0.78
        spatial_iou: 0.67
      artifacts:
        - artifacts/e1_2_reconstruction_samples.png
        - artifacts/e1_2_metrics.json

    # ... more experiments

# Overall assessment
assessment:
  success_criteria_met: true
  lpips_achieved: 0.31  # vs threshold 0.35 ✓
  spatial_iou_achieved: 0.67  # vs threshold 0.6 ✓

  confidence: high | medium | low
  confidence_notes: |
    Results are consistent across multiple runs.
    One edge case with fast motion needs investigation.

# Recommendation
recommendation: proceed | pivot | investigate
recommendation_notes: |
  Claim 1 validated. Recommend proceeding to C2.
  Note: Use pre-merge latents for best spatial accuracy.

# Blockers or issues discovered
blockers: []
issues:
  - "Fast motion (>30fps) shows degraded reconstruction"

# Links to detailed analysis
detailed_analysis: analysis.ipynb
```

## Validation Scripts

### 1. Validate Individual Experiment

```bash
python research/validation/scripts/validate_experiment.py c1-vlm-latent-sufficiency
```

Checks:
- [ ] results.yaml exists and matches schema
- [ ] All claimed artifacts exist
- [ ] Metrics are within plausible ranges
- [ ] Success criteria assessment matches actual metrics

### 2. Rollup Status

```bash
python research/validation/scripts/rollup_status.py
```

Generates `status/claims.yaml`:
```yaml
claims:
  c1_vlm_latent_sufficiency:
    status: completed
    success: true
    metrics:
      lpips: 0.31
      spatial_iou: 0.67
    last_updated: "2025-01-25T15:30:00Z"

  c2_adapter_bridging:
    status: in_progress
    progress: 60%
    current_experiment: e2_3_cross_attention
    last_updated: "2025-01-26T09:00:00Z"

  c3_future_prediction:
    status: blocked
    blocked_by: c2_adapter_bridging

  c4_pixel_verification:
    status: not_started
    blocked_by: c3_future_prediction

overall:
  claims_validated: 1/4
  questions_answered: 2/5
  current_phase: 2
  next_gate: "C2 completion"
```

### 3. Check Decision Gates

```bash
python research/validation/scripts/check_gates.py
```

Evaluates go/no-go decisions:
```yaml
gates:
  gate_1_reconstruction:
    description: "Can we reconstruct video from VLM latents?"
    depends_on: [c1, q1, q2]
    status: passed
    decision: proceed_to_phase_2
    evidence:
      - c1: "LPIPS 0.31 < 0.35 threshold"
      - q1: "Linear probe R² = 0.45, suggests MLP adapter needed"
      - q2: "Pre-merge extraction recommended"

  gate_2_bridging:
    description: "Can small adapter bridge latent spaces?"
    depends_on: [c2, q3]
    status: pending
    blocking_experiments:
      - c2_e2_4_parameter_scaling
```

## How Claude Instances Coordinate

### Option A: File-Based Coordination (Simple)

Each instance:
1. Claims an experiment by creating `{experiment}/CLAIMED_BY_{instance_id}`
2. Works on the experiment
3. Writes `results.yaml` when done
4. Deletes claim file

Coordination script checks for conflicts and rolls up status.

### Option B: GitHub Issues (Better Tracking)

Each experiment = GitHub issue with:
- Checklist of sub-experiments
- Assignee (instance ID)
- Labels: `in-progress`, `blocked`, `completed`, `failed`
- Comments for progress updates
- Linked PRs for artifacts

### Option C: Structured Logs + Central State (Most Robust)

Each instance writes to a shared log:
```yaml
# research/validation/activity_log.yaml
- timestamp: "2025-01-26T10:00:00Z"
  instance: claude-1
  action: started
  experiment: c2-adapter-bridging

- timestamp: "2025-01-26T10:30:00Z"
  instance: claude-1
  action: completed_subexperiment
  experiment: c2-adapter-bridging
  subexperiment: e2_1_linear_probe
  result:
    r_squared: 0.23
    status: below_threshold
```

Central state file is updated atomically by validation scripts.

## What This Doesn't Automate

Some things require human judgment:
- **Interpretation**: What do unexpected results mean?
- **Pivots**: Which pivot option to pursue?
- **Quality**: Is the analysis rigorous enough?
- **Novelty**: Are we learning something new?

The system flags these for review:
```yaml
requires_human_review:
  - experiment: c1-vlm-latent-sufficiency
    reason: "Edge case with fast motion needs interpretation"
  - experiment: q1-latent-alignment
    reason: "Results suggest architectural change - need decision"
```

## Artifact Requirements

Each experiment must produce:

| Artifact | Purpose | Validation |
|----------|---------|------------|
| `results.yaml` | Structured results | Schema validation |
| `metrics.json` | Raw metric values | Bounds checking |
| `*.png` plots | Visual evidence | File exists |
| `*.pt` checkpoints | Reproducibility | File exists, loadable |
| `analysis.ipynb` | Detailed analysis | Notebook runs without error |

## Usage

```bash
# Check status of all experiments
python research/validation/scripts/rollup_status.py

# Validate a specific experiment's results
python research/validation/scripts/validate_experiment.py c1-vlm-latent-sufficiency

# Check if we can proceed past a gate
python research/validation/scripts/check_gates.py gate_1_reconstruction

# Generate summary report
python research/validation/scripts/generate_report.py --output status_report.md
```
