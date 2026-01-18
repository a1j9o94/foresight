# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
