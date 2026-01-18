# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## For Experiment Agents

**If you're assigned to run an experiment, start here:**

1. Read the **[Agent Guide](research/AGENT_GUIDE.md)** - explains how to implement and run experiments
2. Find your experiment plan in `research/experiments/<experiment-id>.md`
3. Implement handlers in `infra/modal/handlers/<experiment>/`
4. **After completing:** Update these files:
   - `research/experiments/<experiment-id>/results.yaml` - Detailed metrics
   - `research/experiments/<experiment-id>/FINDINGS.md` - Plain-language findings for your experiment
   - `research/FINDINGS.md` - Update the summary with your key results

Quick commands:
```bash
# Test infrastructure with stub mode
uv run modal run infra/modal/app.py::run_experiment --experiment-id c1-vlm-latent-sufficiency --stub-mode

# Run your experiment
uv run modal run infra/modal/app.py::run_experiment --experiment-id c1-vlm-latent-sufficiency
```

---

## Project Overview

**Visualize** is a research prototype exploring whether AI systems can benefit from generating pixel-level video predictions as part of their reasoning process. The core hypothesis: an AI that can "see" predicted outcomes will make better decisions than one reasoning purely in text.

## Architecture

The system uses a **Generative Latent Prediction (GLP)** architecture with four modules:

1. **Vision Encoder** (Qwen2.5-VL vision tower, frozen) - Encodes images/video frames into latent patches
2. **Reasoning Backbone** (Qwen2.5-VL-7B-Instruct, frozen) - Processes conversation + visual context, predicts next world state via learned query tokens
3. **Video Decoder** (LTX-Video or HunyuanVideo-1.5, LoRA fine-tuned) - Decodes predicted latents into video frames
4. **Verification Module** (optional) - Compares predicted video to actual outcomes using LPIPS/VLM comparison

**Trainable components** (~10-50M params total):
- Learned query tokens (~32-64 tokens, ~1M params)
- Conditioning adapter (projects VLM latents → video decoder space)
- Video decoder LoRA weights

## Commands

```bash
# Environment setup
pip install torch torchvision torchaudio
pip install transformers diffusers accelerate
pip install flash-attn --no-build-isolation
pip install gradio wandb bitsandbytes

# Download models
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
huggingface-cli download Lightricks/LTX-Video

# Run demo UI
python demo/app.py

# Training
python scripts/train.py --config configs/training_config.yaml

# Evaluation
python scripts/evaluate.py --checkpoint <path>
```

## Key Design Decisions

- **Frozen backbones**: We don't train new foundation models. We train a small "glue" layer connecting two pretrained models (VLM + video decoder).
- **Pixel grounding matters**: Unlike pure latent prediction (JEPA), we generate actual pixels that can be compared against reality for verification.
- **GLP training objective**: Combines reconstruction loss (LPIPS between predicted and actual video) with latent alignment (cosine similarity in embedding space).

## Model Options

| Video Decoder | Use Case |
|--------------|----------|
| LTX-Video | Prototyping, fast iteration (real-time 30fps) |
| HunyuanVideo-1.5 | Production quality (75s on 4090) |

## Training Data

Primary datasets for video-action pairs:
- COIN dataset (procedural activities)
- CrossTask
- Something-Something v2

## Performance Targets

- Video generation: <2 seconds per 5-second clip
- VLM reasoning: <1 second per response
- Total reasoning step: <3 seconds
- VRAM requirement: ~40GB total (both models loaded)

## Research Workflow

### Adding Paper Reviews

1. Copy `research/papers/_template.md` to `research/papers/<paper-slug>.md`
2. Fill in all sections of the template
3. Add entry to `research/papers/index.md` with review status
4. Link relevant insights to hypothesis documents in `research/hypotheses/`

### Hypothesis Development

Working hypothesis documents are in `research/hypotheses/`. Update these as literature review progresses.

## Experiment Execution & Validation

This project uses a structured validation system to track research progress across parallel experiments. See `research/validation/README.md` for full details.

### Running an Experiment

1. **Find your experiment plan** in `research/experiments/` (e.g., `c1-vlm-latent-sufficiency.md`)
2. **Create results directory**: `research/experiments/<experiment-id>/`
3. **Copy the template**: `cp research/validation/templates/results.yaml.template research/experiments/<experiment-id>/results.yaml`
4. **Execute sub-experiments** as defined in the plan
5. **Update results.yaml** as you complete each sub-experiment
6. **Store artifacts** in `research/experiments/<experiment-id>/artifacts/`

### Results File Structure

Each experiment produces a `results.yaml` with:
- **Metrics**: Actual measured values (LPIPS, accuracy, etc.)
- **Artifacts**: Paths to plots, checkpoints, logs
- **Assessment**: Whether success criteria were met
- **Recommendation**: proceed / pivot / investigate / block

Example:
```yaml
experiment_id: c1-vlm-latent-sufficiency
status: completed
success_criteria:
  lpips_threshold: 0.35
results:
  experiments:
    e1_2_reconstruction_probe:
      status: completed
      metrics:
        lpips: 0.31
      artifacts:
        - artifacts/reconstruction_samples.png
assessment:
  success_criteria_met: true
  lpips_achieved: 0.31
  confidence: high
recommendation: proceed
```

### Validation Commands

```bash
# Validate your experiment results
python research/validation/scripts/validate_experiment.py <experiment-id>

# Check overall project status
python research/validation/scripts/rollup_status.py

# Check if a decision gate can be passed
python research/validation/scripts/check_gates.py <gate-id>
```

### Decision Gates

Progress through phases requires passing gates:

| Gate | Experiments Required | Unlocks |
|------|---------------------|---------|
| `gate_1_reconstruction` | Q1, **P2** | Phase 2 (Bridging) |
| `gate_2_bridging` | C2, Q3 | Phase 3 (Prediction) |
| `gate_3_prediction` | C3, Q4, Q5 | Phase 4 (Verification) |
| `gate_4_verification` | C4 | Final evaluation |

> **Note:** Gate 1 updated after C1/Q2 pivoted. P2 (Hybrid Encoder) replaces spatial validation. See `research/FINDINGS.md` for details.

### Success Criteria Quick Reference

| Experiment | Key Metric | Threshold |
|------------|-----------|-----------|
| **P2** | Spatial IoU | > 0.6 |
| **P2** | LPIPS | < 0.35 |
| **P2** | mAP@0.5 | > 0.4 |
| C2 | Param efficiency | 10M achieves >80% of 100M |
| C3 | Cosine similarity | > 0.65 at t+5 |
| C4 | Accuracy improvement | > 10% from verification |
| ~~C1~~ | ~~Spatial IoU~~ | *Pivoted - replaced by P2* |

### When to Flag for Human Review

Add to `requires_human_review` in your results if:
- Confidence is "low"
- Results are unexpected or contradictory
- Pivot decision needed
- Edge cases discovered that affect interpretation

## W&B Configuration

- **Project name**: `foresight`
- **Entity**: (configure in environment or wandb login)
- **Config location**: `configs/wandb/`

## Package Development

This is a uv workspace with packages in `packages/`:

```bash
# Install all packages in development mode
uv sync

# Run from root
python -c "import foresight_core"

# Add dependencies to a package
cd packages/core && uv add <package>
```

### Package Dependencies

```
foresight-core         # Base utilities, W&B config, types
├── foresight-models   # Vision encoder, backbone, decoder
├── foresight-training # Trainer, data loaders (depends on models)
├── foresight-inference # Pipeline, optimization (depends on models)
└── foresight-eval     # Metrics, verification (depends on models)
```
