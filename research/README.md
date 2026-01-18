# Research Overview

This directory contains literature review, hypothesis development, and research documentation for the Foresight project.

## Research Methodology

1. **Literature Review** - Systematic review of related work in video prediction, world models, and multimodal reasoning
2. **Hypothesis Development** - Iterative refinement of core hypothesis based on findings
3. **Experiment Design** - Controlled experiments to test hypotheses
4. **Analysis** - Quantitative and qualitative evaluation of results

## Directory Structure

```
research/
├── papers/              # Paper summaries and analysis
│   ├── _template.md     # Template for paper reviews
│   └── index.md         # Master list of reviewed papers
└── hypotheses/          # Hypothesis development
    └── core-hypothesis.md
```

## Key Research Areas

### Video Prediction Models
- Latent diffusion for video (SVD, LTX-Video, HunyuanVideo)
- Autoregressive video models
- Flow-based generation

### World Models
- JEPA and latent prediction
- Dreamer and model-based RL
- IRIS and discrete world models

### Multimodal Reasoning
- Vision-language models (GPT-4V, Qwen-VL, LLaVA)
- Video understanding (VideoLLM, VideoChat)
- Embodied reasoning

### Verification and Grounding
- Visual question answering
- Video-text alignment
- Temporal reasoning

## Adding a Paper Review

1. Copy `papers/_template.md` to `papers/<paper-slug>.md`
2. Fill in all sections of the template
3. Add entry to `papers/index.md` with status
4. Link relevant insights to hypothesis documents

## Paper Review Status Legend

| Status | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Complete |
| `[!]` | High priority |

## Related Documents

- [Core Hypothesis](./hypotheses/core-hypothesis.md)
- [Product Requirements](../PRD.md)
- [Project README](../README.md)
