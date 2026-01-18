# Product Requirements Document: Foresight

## Executive Summary

Foresight is a research project investigating whether AI systems can improve their decision-making by generating and verifying pixel-level video predictions of future states.

## Problem Statement

Current AI reasoning systems operate purely in text/token space. When asked to predict outcomes or plan actions, they:
- Cannot "visualize" what will happen
- Have no mechanism to verify predictions against reality
- Rely entirely on learned text patterns rather than grounded understanding

## Hypothesis

An AI system that can:
1. Generate explicit visual predictions of future states
2. Compare those predictions against actual outcomes
3. Use the comparison for verification and self-correction

...will make more accurate predictions and better decisions than text-only systems.

## Solution: Generative Latent Prediction (GLP)

### Architecture Overview

```
Input Video → Vision Encoder → VLM Backbone → Query Tokens → Adapter → Video Decoder → Predicted Video
                                   ↓
                              Text Response
```

### Components

| Component | Model | Training |
|-----------|-------|----------|
| Vision Encoder | Qwen2.5-VL vision tower | Frozen |
| Reasoning Backbone | Qwen2.5-VL-7B-Instruct | Frozen |
| Query Tokens | 32-64 learned tokens | Trained |
| Conditioning Adapter | MLP projection | Trained |
| Video Decoder | LTX-Video / HunyuanVideo | LoRA fine-tuned |

### Trainable Parameters

- Learned query tokens: ~1M params
- Conditioning adapter: ~5-10M params
- Video decoder LoRA: ~10-30M params
- **Total: ~10-50M parameters**

## Training Data

Primary datasets for video-action pairs:
- **COIN dataset** - Procedural activities
- **CrossTask** - Instructional videos
- **Something-Something v2** - Object interactions

## Training Objective

The GLP loss combines:
1. **Reconstruction loss**: LPIPS between predicted and actual video
2. **Latent alignment**: Cosine similarity in embedding space

## Success Metrics

### Performance Targets
- Video generation: <2 seconds per 5-second clip
- VLM reasoning: <1 second per response
- Total reasoning step: <3 seconds
- VRAM requirement: ~40GB total

### Quality Metrics
- FVD (Frechet Video Distance) on held-out clips
- Action prediction accuracy improvement vs text-only baseline
- Human evaluation of prediction plausibility

## Model Options

| Video Decoder | Use Case | Speed |
|--------------|----------|-------|
| LTX-Video | Prototyping, fast iteration | Real-time 30fps |
| HunyuanVideo-1.5 | Production quality | 75s on 4090 |

## Research Questions

1. Does pixel-level grounding improve action prediction accuracy?
2. How much does verification (comparing predicted vs actual) help?
3. What's the minimum video decoder quality needed for benefits?
4. Does the system generalize to novel scenarios?

## Milestones

### Phase 1: Foundation
- [ ] Literature review of related work
- [ ] Dataset preparation pipeline
- [ ] Basic integration of VLM + video decoder

### Phase 2: Training
- [ ] GLP training implementation
- [ ] Ablation studies on components
- [ ] Hyperparameter optimization

### Phase 3: Evaluation
- [ ] Benchmark on standard datasets
- [ ] Comparison to text-only baselines
- [ ] Analysis of failure modes

### Phase 4: Applications
- [ ] Demo UI for interactive exploration
- [ ] Real-world task evaluation
- [ ] Documentation and paper

## Related Work

- JEPA (Joint Embedding Predictive Architecture)
- World models (Dreamer, IRIS)
- Video prediction models (SVD, Sora)
- Multimodal LLMs (GPT-4V, Qwen-VL, LLaVA)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Video decoder too slow | Use LTX-Video for prototyping |
| Training instability | Freeze backbones, train only adapter |
| Evaluation difficulty | Use standard video prediction benchmarks |
| Scope creep | Focus on core hypothesis first |
