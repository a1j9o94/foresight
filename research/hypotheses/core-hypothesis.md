# Core Hypothesis

## Primary Hypothesis

> An AI system that generates explicit pixel-level predictions of future states and can compare those predictions against actual outcomes will make more accurate decisions than systems reasoning purely in text/token space.

## Hypothesis Breakdown

### H1: Pixel Grounding Improves Prediction

**Claim:** Generating actual pixels (rather than just latent predictions) forces the model to commit to specific, verifiable details about the future.

**Reasoning:**
- Text predictions can be vague ("the cup might fall")
- Pixel predictions must be specific (exact position, orientation, timing)
- This specificity may improve prediction accuracy through forced commitment

**Test:** Compare action prediction accuracy between:
- Text-only VLM
- VLM + latent prediction (JEPA-style)
- VLM + pixel prediction (our approach)

### H2: Verification Enables Self-Correction

**Claim:** Comparing predicted video to actual outcomes provides a learning signal that enables self-correction.

**Reasoning:**
- Current LLMs cannot verify their predictions against reality
- Visual comparison (LPIPS, VLM-based) provides quantifiable error
- This error signal could drive iterative refinement

**Test:** Evaluate prediction accuracy improvement after verification loop:
- Single-shot prediction
- Prediction + verification + reprediction

### H3: Minimal Training Suffices

**Claim:** Training only a small adapter layer (~10-50M params) is sufficient to bridge VLM and video decoder.

**Reasoning:**
- Both foundation models already understand visual concepts
- The gap is primarily in "format translation" between representations
- Similar to how LoRA enables task adaptation with few parameters

**Test:** Compare performance across adapter sizes and training regimes.

## Alternative Hypotheses

### A1: Latent Space Sufficient

JEPA-style latent prediction might be sufficient without pixel generation.

**Counter-argument:** Latent predictions are hard to verify and may miss details that become obvious in pixel space.

### A2: Text Sufficient with Better Prompting

Better chain-of-thought prompting might achieve similar results without video generation.

**Counter-argument:** Text is fundamentally limited in representing spatial/temporal relationships that are natural in video.

### A3: Scale is the Answer

Larger models might achieve these benefits without architectural changes.

**Counter-argument:** Even very large models (GPT-4V) cannot currently generate or verify visual predictions.

## Key Assumptions

1. **VLM latents contain sufficient information** for video reconstruction
2. **Video decoder can be conditioned** on VLM latent space
3. **LPIPS/perceptual metrics** correlate with prediction quality
4. **Dataset variety** (COIN, CrossTask) covers relevant scenarios

## Falsification Criteria

The hypothesis would be falsified if:

1. Pixel prediction does not improve over text-only baseline
2. Verification loop does not improve predictions
3. Training cannot converge to coherent video generation
4. The approach requires prohibitive compute (>>40GB VRAM)

## Evolution of Hypothesis

### v0.1 (Initial)
- "Video generation helps reasoning" (too vague)

### v0.2 (Current)
- Specific architecture (GLP)
- Measurable claims
- Clear falsification criteria

### Future Refinements
- After literature review, may adjust based on:
  - What prior work has already shown
  - What gaps remain unexplored
  - What approaches have been tried and failed

## Related Documents

- [Paper Index](../papers/index.md)
- [Product Requirements](../../PRD.md)
- [Research Overview](../README.md)

## Changelog

| Date | Change |
|------|--------|
| 2024-XX-XX | Initial hypothesis formulation |
