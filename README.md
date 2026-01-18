# Foresight

A research prototype exploring whether AI systems can benefit from generating pixel-level video predictions as part of their reasoning process.

## Core Hypothesis

An AI that can "see" predicted outcomes will make better decisions than one reasoning purely in text.

## Architecture

The system uses a **Generative Latent Prediction (GLP)** architecture:

1. **Vision Encoder** - Encodes images/video frames into latent patches (Qwen2.5-VL vision tower, frozen)
2. **Reasoning Backbone** - Processes conversation + visual context (Qwen2.5-VL-7B-Instruct, frozen)
3. **Video Decoder** - Decodes predicted latents into video frames (LTX-Video or HunyuanVideo-1.5)
4. **Verification Module** - Compares predicted video to actual outcomes

## Project Structure

```
foresight/
├── research/           # Literature review & research docs
│   ├── papers/         # Paper summaries
│   └── hypotheses/     # Hypothesis development
├── packages/           # Modular code packages
│   ├── core/           # Shared utilities, W&B config
│   ├── models/         # Vision encoder, backbone, decoder
│   ├── training/       # GLP trainer, data loaders
│   ├── inference/      # Pipeline, optimization
│   └── evaluation/     # Metrics, verification
├── notebooks/          # Jupyter exploration
├── scripts/            # CLI entry points
├── demo/               # Gradio UI
└── configs/            # Model/training configs
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run demo UI
python demo/app.py

# Training
python scripts/train.py --config configs/training_config.yaml

# Evaluation
python scripts/evaluate.py --checkpoint <path>
```

## Documentation

- [Product Requirements (PRD.md)](./PRD.md)
- [Research Overview](./research/README.md)
- [Development Guide (CLAUDE.md)](./CLAUDE.md)

## License

Research prototype - not for production use.
