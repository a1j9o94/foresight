# Paper Index

Master list of papers reviewed for the Foresight project.

## Status Legend

| Status | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Complete |
| `[!]` | High priority |

## Video Generation Models

| Status | Paper | Year | Notes |
|--------|-------|------|-------|
| `[x]` | [LTX-Video](ltx-video.md) | 2024 | Primary video decoder - real-time DiT, 1.9B params |
| `[x]` | [HunyuanVideo](hunyuan-video.md) | 2024 | High-quality decoder - 13B params, cinematic quality |
| `[ ]` | Stable Video Diffusion | 2023 | Image-to-video baseline |
| `[ ]` | Sora Technical Report | 2024 | Large-scale video generation |

## World Models

| Status | Paper | Year | Notes |
|--------|-------|------|-------|
| `[x]` | [PlaNet / Latent Space Dynamics](latent-space-dynamics.md) | 2019 | Foundational latent dynamics for planning |
| `[x]` | [V-JEPA](v-jepa.md) | 2024 | Latent video prediction - contrast with pixel grounding |
| `[x]` | [Dreamer v3](dreamer-v3.md) | 2023 | World model RL - chain-of-images reasoning proof |
| `[ ]` | IRIS | 2023 | Discrete world model for Atari |
| `[ ]` | Genie | 2024 | World model from video |

## Vision-Language Models

| Status | Paper | Year | Notes |
|--------|-------|------|-------|
| `[x]` | [Qwen2-VL](qwen2-vl.md) | 2024 | Our VLM backbone - 1536-dim visual embeddings |
| `[ ]` | LLaVA-NeXT | 2024 | Alternative VLM |
| `[ ]` | InternVL 2 | 2024 | Strong video understanding |
| `[ ]` | Video-LLaVA | 2024 | Video-specific VLM |

## Video Understanding

| Status | Paper | Year | Notes |
|--------|-------|------|-------|
| `[ ]` | VideoChat | 2023 | Chat with videos |
| `[ ]` | Video-ChatGPT | 2023 | Video QA |
| `[ ]` | LLaVA-Video | 2024 | Long video understanding |

## Verification and Grounding

| Status | Paper | Year | Notes |
|--------|-------|------|-------|
| `[ ]` | LPIPS | 2018 | Perceptual similarity metric |
| `[ ]` | FVD | 2019 | Video quality metric |
| `[ ]` | VideoScore | 2024 | VLM-based video evaluation |

## Datasets

| Status | Paper | Year | Notes |
|--------|-------|------|-------|
| `[ ]` | COIN | 2019 | Procedural activities |
| `[ ]` | CrossTask | 2019 | Instructional videos |
| `[ ]` | Something-Something v2 | 2017 | Object interactions |

## Training Techniques

| Status | Paper | Year | Notes |
|--------|-------|------|-------|
| `[ ]` | LoRA | 2021 | Parameter-efficient fine-tuning |
| `[ ]` | QLoRA | 2023 | Quantized LoRA |
| `[ ]` | Flash Attention 2 | 2023 | Efficient attention |

---

## Recently Added

*Papers added in the last update*

- [LTX-Video](ltx-video.md) - Real-time video generation, primary decoder candidate
- [HunyuanVideo](hunyuan-video.md) - High-quality video generation for production
- [Qwen2-VL](qwen2-vl.md) - Vision-language backbone with extractable latents
- [V-JEPA](v-jepa.md) - Latent prediction approach (contrast with our pixel grounding)
- [Dreamer v3](dreamer-v3.md) - World model proving chain-of-images reasoning works
- [PlaNet / Latent Space Dynamics](latent-space-dynamics.md) - Foundational latent dynamics

## To Review Next

*Priority queue for paper review*

1. IRIS - Discrete world model, different approach to imagination
2. Genie - World model learned from video (no actions)
3. COIN dataset - Training data source
4. LPIPS - Verification metric we'll use
