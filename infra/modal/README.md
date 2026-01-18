# Modal GPU Infrastructure

Run Foresight experiments on cloud GPUs using [Modal](https://modal.com).

## Quick Start

### 1. Set Up Secrets

First, add your W&B API key to Modal:

```bash
# Get your key from https://wandb.ai/authorize
modal secret create wandb-api-key WANDB_API_KEY=<your-key>
```

### 2. Run Smoke Test

Verify everything works:

```bash
modal run infra/modal/app.py::smoke_test
```

Expected output:
```
FORESIGHT SMOKE TEST
PyTorch version: 2.x.x
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
VRAM: 40.0 GB
...
SMOKE TEST PASSED
```

### 3. Download Models (First Time)

Pre-cache models to avoid re-downloading:

```bash
modal run infra/modal/app.py::download_models
```

This downloads:
- Qwen2.5-VL-7B-Instruct (~15GB)
- LTX-Video (~4GB)

### 4. Run an Experiment

```bash
# Test the harness with stub handlers (no real GPU work)
modal run infra/modal/app.py::run_experiment --experiment-id c1-vlm-latent-sufficiency --stub-mode

# Run full experiment (once handlers are implemented)
modal run infra/modal/app.py::run_experiment --experiment-id c1-vlm-latent-sufficiency

# Run specific sub-experiment
modal run infra/modal/app.py::run_experiment --experiment-id c1-vlm-latent-sufficiency --sub-experiment e1_2
```

### 5. Check Results

```bash
modal run infra/modal/app.py::list_results
```

## Experiment Runner Architecture

The experiment runner (`infra/modal/runner/`) provides a structured framework for running experiments:

### Components

- **`config.py`**: Experiment configuration and registry
- **`results.py`**: `ResultsWriter` class for standardized YAML output
- **`runner.py`**: `ExperimentRunner` orchestrator with W&B integration

### Implementing a New Experiment Handler

Each sub-experiment needs a handler function:

```python
def e1_1_latent_visualization(runner: ExperimentRunner) -> dict:
    """Run latent space visualization sub-experiment."""

    # Your experiment code here
    # Use runner.log_metrics() for incremental logging
    # Use runner.results.get_artifact_path() for saving files

    return {
        "finding": "Latents show clear semantic clustering",
        "metrics": {
            "silhouette_score": 0.72,
        },
        "artifacts": [
            runner.results.save_artifact("tsne_plot.png", plot_bytes),
        ],
    }
```

Register handlers in the experiment runner:

```python
runner = ExperimentRunner("c1-vlm-latent-sufficiency")
runner.register_handler("e1_1", e1_1_latent_visualization)
runner.run_all()
```

### Results Format

Results are saved to `/results/<experiment-id>/results.yaml`:

```yaml
experiment_id: c1-vlm-latent-sufficiency
status: completed
success_criteria:
  lpips_threshold: 0.35
  ssim_threshold: 0.75
results:
  experiments:
    e1_1:
      status: completed
      finding: "Latents show clear semantic clustering"
      metrics:
        silhouette_score: 0.72
      artifacts:
        - artifacts/tsne_plot.png
assessment:
  success_criteria_met: true
  lpips_achieved: 0.31
  confidence: high
recommendation: proceed
```

## GPU Options

Edit `app.py` to change GPU type:

| GPU | VRAM | Cost | Use For |
|-----|------|------|---------|
| `A10G` | 24GB | $ | Q1, Q2 (exploration) |
| `A100` | 40GB | $$ | C1, C2, C3 (main experiments) |
| `H100` | 80GB | $$$ | Large batch training |

## Volumes

Two persistent volumes are used:

- `foresight-results`: Experiment outputs, results.yaml, artifacts
- `foresight-model-cache`: Downloaded HuggingFace models

## Monitoring

- **Modal Dashboard**: https://modal.com/apps
- **W&B Dashboard**: https://wandb.ai/<your-entity>/foresight

## Costs

Rough estimates (Modal pricing as of 2024):

| GPU | Per Hour | 4hr Experiment |
|-----|----------|----------------|
| A10G | ~$1.10 | ~$4.40 |
| A100 | ~$3.30 | ~$13.20 |
| H100 | ~$5.50 | ~$22.00 |

Phase 1 experiments (C1, Q1, Q2) estimate: ~$50-100 total

## Troubleshooting

### "Secret not found"
```bash
modal secret create wandb-api-key WANDB_API_KEY=<your-key>
```

### "Out of memory"
- Switch to larger GPU
- Reduce batch size in experiment config
- Use gradient checkpointing

### "Volume not found"
Volumes are created automatically on first run.

### Timeout
Default is 4-8 hours. For longer experiments:
```python
@app.function(timeout=3600 * 12)  # 12 hours
```
