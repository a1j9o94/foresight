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
# Run full experiment
modal run infra/modal/app.py::run_experiment --experiment-id c1-vlm-latent-sufficiency

# Run specific sub-experiment
modal run infra/modal/app.py::run_experiment --experiment-id c1-vlm-latent-sufficiency --sub-experiment e1_2
```

### 5. Check Results

```bash
modal run infra/modal/app.py::list_results
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
