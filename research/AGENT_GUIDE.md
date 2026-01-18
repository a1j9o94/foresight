# Agent Guide: Running Foresight Experiments

**Read this first** if you're a Claude Code agent assigned to run an experiment.

---

## Quick Start (5 minutes)

### 1. Check Your Assignment

You should have an experiment ID like `c1-vlm-latent-sufficiency`. Find your experiment plan:

```
research/experiments/<experiment-id>.md
```

### 2. Verify Setup is Complete

Check `research/SETUP_CHECKLIST.md` - all items should be checked. If not, resolve blockers first.

### 3. Run in Stub Mode (Test Harness)

Verify the infrastructure works:

```bash
uv run modal run infra/modal/app.py::run_experiment \
  --experiment-id c1-vlm-latent-sufficiency --stub-mode
```

You should see all sub-experiments complete with stub results.

### 4. Implement Your Handlers

Create handlers in `infra/modal/handlers/<experiment>/`. See [Handler Implementation](#handler-implementation) below.

### 5. Run Your Experiment

```bash
uv run modal run infra/modal/app.py::run_experiment \
  --experiment-id c1-vlm-latent-sufficiency
```

### 6. Check Results

- **W&B**: https://wandb.ai/a1j9o94/foresight
- **Results file**: `/results/<experiment-id>/results.yaml` (on Modal volume)

### 7. Document Your Findings

**After achieving `completed` status, you MUST update these files:**

1. **Sync results locally:**
   ```bash
   bash scripts/sync-results.sh
   ```

2. **Update experiment FINDINGS.md** (`research/experiments/<experiment-id>/FINDINGS.md`):
   - Plain-language summary of what you discovered
   - Key metrics and whether thresholds were met
   - Implications for the hypothesis
   - Any surprising or unexpected results

3. **Update project FINDINGS.md** (`research/FINDINGS.md`):
   - Add a summary entry for your experiment
   - Note the recommendation (proceed/pivot)
   - Highlight key metrics

Example experiment findings:
```markdown
# C1: VLM Latent Sufficiency - Findings

## Status: COMPLETED | Recommendation: PIVOT

## Summary
VLM latents achieve good perceptual quality (LPIPS=0.264) but fail to preserve
spatial information (IoU=0.559 < 0.6 threshold).

## Key Metrics
| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| LPIPS | 0.264 | < 0.35 | ✅ Pass |
| SSIM | 0.943 | > 0.75 | ✅ Pass |
| Spatial IoU | 0.559 | > 0.6 | ❌ Fail |

## Implications
The 2x2 token merger destroys positional information. Alternative approaches
needed for spatial tasks.
```

---

## Handler Implementation

### File Location

Put your handlers in:

```
infra/modal/handlers/<experiment-id-prefix>/
```

For C1, that's:

```
infra/modal/handlers/c1/
├── __init__.py      # Exports get_handlers()
├── example.py       # Reference implementation
├── e1_1.py          # Latent visualization
├── e1_2.py          # Reconstruction probe
└── ...
```

### Handler Function Signature

Every handler must:
1. Accept an `ExperimentRunner` instance
2. Return a dict with `finding`, `metrics`, and `artifacts`

```python
from runner import ExperimentRunner

def e1_1_latent_visualization(runner: ExperimentRunner) -> dict:
    """
    Sub-experiment E1.1: Latent Space Visualization

    Objective: Understand the structure of Qwen2.5-VL's visual latent space.
    """

    # Your experiment code here...

    # Log metrics incrementally (shows in W&B during run)
    runner.log_metrics({"progress": 0.5}, step=50)

    # Save artifacts
    plot_path = runner.results.save_artifact("tsne_plot.png", png_bytes)
    data_path = runner.results.save_json_artifact("latent_stats.json", stats_dict)

    # Return required format
    return {
        "finding": "Latents show clear semantic clustering by object category",
        "metrics": {
            "silhouette_score": 0.72,
            "n_clusters": 5,
        },
        "artifacts": [plot_path, data_path],
    }
```

### Registering Handlers

In `infra/modal/handlers/c1/__init__.py`:

```python
from .e1_1 import e1_1_latent_visualization
from .e1_2 import e1_2_reconstruction_probe
# ... import other handlers

def get_handlers() -> dict:
    """Return all handlers for C1 experiment."""
    return {
        "e1_1": e1_1_latent_visualization,
        "e1_2": e1_2_reconstruction_probe,
        # ... register other handlers
    }
```

---

## Available Resources

### Models (Pre-cached on Modal)

```python
# Qwen2.5-VL-7B (8.3B params)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# LTX-Video
from diffusers import LTXPipeline

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16,
)
```

### Datasets (On Modal Volume)

```python
# Synthetic test data (for quick debugging)
SYNTHETIC_DATA = "/datasets/synthetic/"

# Something-Something v2 subset (for real experiments)
SSV2_DATA = "/datasets/something-something-v2/"
```

### Results Storage

```python
# Results directory for your experiment
results_dir = f"/results/{experiment_id}/"

# Artifacts subdirectory
artifacts_dir = f"/results/{experiment_id}/artifacts/"
```

---

## Runner API Reference

### ExperimentRunner

```python
class ExperimentRunner:
    # Log metrics to W&B (incremental, during experiment)
    def log_metrics(self, metrics: dict[str, float], step: int | None = None): ...

    # Access results writer
    @property
    def results(self) -> ResultsWriter: ...

    # Get experiment config
    @property
    def config(self) -> dict: ...
```

### ResultsWriter

```python
class ResultsWriter:
    # Save binary artifact (images, checkpoints)
    def save_artifact(self, filename: str, content: bytes) -> str: ...

    # Save JSON artifact (data, configs)
    def save_json_artifact(self, filename: str, data: dict) -> str: ...

    # Get full path for an artifact
    def get_artifact_path(self, filename: str) -> Path: ...
```

---

## Local Testing (Before Modal)

To iterate faster, test your handler logic locally first:

### 1. Test Imports

```bash
uv run python -c "
import sys
sys.path.insert(0, 'infra/modal')
from handlers.c1.e1_1 import e1_1_latent_visualization
print('Import OK')
"
```

### 2. Mock Runner Test

```python
# test_handler.py
import sys
sys.path.insert(0, 'infra/modal')

from unittest.mock import MagicMock
from handlers.c1.e1_1 import e1_1_latent_visualization

# Create mock runner
runner = MagicMock()
runner.log_metrics = MagicMock()
runner.results.save_artifact = MagicMock(return_value="artifacts/test.png")
runner.results.save_json_artifact = MagicMock(return_value="artifacts/test.json")

# Test handler (without GPU)
result = e1_1_latent_visualization(runner)

# Verify return format
assert "finding" in result
assert "metrics" in result
assert "artifacts" in result
print("Handler structure OK")
```

### 3. Run on Modal (Stub Mode)

```bash
uv run modal run infra/modal/app.py::run_experiment \
  --experiment-id c1-vlm-latent-sufficiency \
  --sub-experiment e1_1
```

---

## Error Handling: Your Code Must Work

### Two possible outcomes

There are only two outcomes for an experiment:

| Status | Meaning | What happens |
|--------|---------|--------------|
| `completed` | Code worked | Hypothesis assessed → `proceed` or `pivot` |
| `failed` | Code broke | Fix code and re-run (no hypothesis assessment possible) |

**There is no middle ground.** If any sub-experiment throws an exception, the whole experiment is `failed`.

### Your responsibility as an agent

**You must run experiments until they achieve `completed` status.**

1. **Run the experiment** to discover failures
2. **Read the error tracebacks** in console output or `results.yaml`
3. **Fix the code** in the handlers
4. **Re-run until status is `completed`** - zero failed sub-experiments

### Example: Failed experiment

```
# Experiment Complete: q1-latent-alignment
# Completed: 3
# Failed: 4         <-- THIS MUST BE 0
# Skipped: 0
# Status: failed    <-- FIX AND RE-RUN
# Recommendation: investigate
#
# CODE ERRORS: 4 sub-experiments threw exceptions!
# Fix and re-run: eq1_1, eq1_2, eq1_4, eq1_5
```

### Example: Successful experiment

```
# Experiment Complete: q1-latent-alignment
# Completed: 7
# Failed: 0         <-- GOOD
# Skipped: 0
# Status: completed <-- CODE WORKED
# Recommendation: proceed  <-- or 'pivot' if hypothesis not supported
```

### How to fix failures

1. Look at the error traceback in:
   - Console output during run
   - `results.yaml` → `results.experiments.<id>.error`

2. Common fixes:
   - **API changes**: Check latest library documentation
   - **Shape mismatches**: Print tensor shapes, verify data flow
   - **Missing dependencies**: Add to Modal image

3. Re-run:
   ```bash
   uv run modal run infra/modal/app.py::run_experiment \
     --experiment-id <your-experiment>
   ```

### When to escalate vs fix yourself

**Fix yourself:**
- Import errors
- Type errors
- Shape mismatches
- Missing parameters
- API deprecations

**Escalate to human:**
- Fundamental algorithm issues
- Missing datasets
- Infrastructure failures
- Unclear requirements

### Important: "Pivot" is NOT a failure

A `pivot` recommendation means:
- ✅ Your code worked correctly
- ❌ The hypothesis was not supported by the results

This is a valid scientific outcome! The experiment ran correctly and produced reliable metrics - they just didn't meet the success criteria.

---

## Pre-flight Checklist

Before running your full experiment:

- [ ] Read the experiment plan in `research/experiments/<id>.md`
- [ ] Understand success/failure criteria from the plan
- [ ] Handlers implemented for all sub-experiments
- [ ] Handlers registered in `__init__.py`
- [ ] Local import test passes
- [ ] Stub mode runs successfully
- [ ] Single sub-experiment test passes on Modal
- [ ] **All sub-experiments complete without errors**

---

## Troubleshooting

### "No handler registered for sub-experiment"

Your handler isn't in the `get_handlers()` return dict. Check:
1. Handler function is defined and exported
2. Handler is registered in `__init__.py`
3. Handler key matches sub-experiment ID (e.g., `"e1_1"`)

### "ImportError: cannot import name..."

Check your handler file for:
1. Syntax errors
2. Missing dependencies (add to Modal image if needed)
3. Circular imports

### "CUDA out of memory"

- Use `torch.bfloat16` instead of `float32`
- Clear cache between models: `torch.cuda.empty_cache()`
- Reduce batch size
- Use gradient checkpointing for training

### "Results not appearing in W&B"

- Check `runner.log_metrics()` was called
- Ensure handler returns proper format
- Check W&B dashboard for the run: https://wandb.ai/a1j9o94/foresight

---

## Success Criteria Reference

From `infra/modal/runner/config.py`:

| Experiment | Metric | Target | Acceptable | Failure |
|------------|--------|--------|------------|---------|
| C1 | LPIPS | < 0.25 | < 0.35 | > 0.45 |
| C1 | SSIM | > 0.85 | > 0.75 | < 0.65 |
| C1 | Spatial IoU | > 0.75 | > 0.6 | < 0.5 |

Your handler should report these metrics. The runner will automatically assess whether criteria are met.

---

## Getting Help

- **Infrastructure issues**: Check `infra/modal/README.md`
- **Experiment details**: Check `research/experiments/<id>.md`
- **Validation system**: Check `research/validation/README.md`
- **W&B setup**: Check `research/SETUP_CHECKLIST.md`
