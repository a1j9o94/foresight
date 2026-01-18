# Research Findings

High-level summary of Foresight experiment results. For detailed findings, see individual experiment FINDINGS.md files.

---

## Project Status

**Current Phase:** Phase 1 - Foundation
**Overall Progress:** Gate 1 BLOCKED (1/3 proceed, 2/3 pivot)

**Critical Finding:** VLM latent spaces lose spatial information through the token merger and LLM layers. Architectural pivot required.

---

## Phase 1: Can VLM latents support video reconstruction?

### [C1: VLM Latent Sufficiency](experiments/c1-vlm-latent-sufficiency/FINDINGS.md) - PIVOT

**Question:** Can Qwen2.5-VL's internal representation be decoded back into video?

**Answer:** Partially. Good perceptual quality, but spatial precision fails.

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| LPIPS | 0.264 | < 0.35 | ✅ Pass |
| SSIM | 0.943 | > 0.75 | ✅ Pass |
| Spatial IoU | 0.559 | > 0.6 | ❌ Fail |

**Key Finding:** VLM latents preserve semantic/perceptual information but lose precise spatial positioning. The 2x2 token merger destroys positional information needed for accurate object localization.

→ [Full details](experiments/c1-vlm-latent-sufficiency/FINDINGS.md)

---

### [Q1: Latent Space Alignment](experiments/q1-latent-alignment/FINDINGS.md) - PROCEED ✅

**Question:** Do VLM and video decoder latent spaces have compatible structure?

**Answer:** Yes. All alignment criteria met.

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Linear Probe R² | 0.510 | > 0.5 | ✅ Pass |
| Spearman Correlation | 0.684 | > 0.6 | ✅ Pass |
| Neighborhood Recall@10 | 0.258 | > 0.2 | ✅ Pass |
| CKA | 0.687 | > 0.4 | ✅ Pass |

**Key Finding:** VLM and LTX latent spaces CAN be aligned via an MLP adapter. Best layer for alignment is `layer_final` (CKA=0.687). Dimensionality gap exists (VLM ~10 intrinsic dims vs LTX ~65), requiring dimension-expanding adapter.

→ [Full details](experiments/q1-latent-alignment/FINDINGS.md)

---

### [Q2: Information Preservation](experiments/q2-information-preservation/FINDINGS.md) - PIVOT

**Question:** Where does spatial information get lost in the VLM pipeline?

**Answer:** Everywhere. Spatial info is degraded pre-merge, destroyed post-merge.

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Bounding Box IoU | 0.104 | > 0.7 | ❌ Fail |
| LPIPS | 0.089 | < 0.3 | ✅ Pass |
| Edge F1 | 0.351 | > 0.6 | ❌ Fail |
| mAP@0.5 | 0.001 | > 0.4 | ❌ Fail |

**Key Finding:** Spatial information is severely degraded in VLM embeddings - both before AND after the 2x2 token merger. Object detection from VLM embeddings is nearly impossible (mAP=0.001). The merger is a bottleneck, but spatial info is already limited pre-merge.

→ [Full details](experiments/q2-information-preservation/FINDINGS.md)

---

## Decision Gates

### Gate 1: Reconstruction
**Status:** ⛔ BLOCKED

| Experiment | Status | Recommendation |
|------------|--------|----------------|
| C1 - VLM Latent Sufficiency | ✅ Complete | **PIVOT** |
| Q1 - Latent Alignment | ✅ Complete | **PROCEED** |
| Q2 - Information Preservation | ✅ Complete | **PIVOT** |

**Gate Progress:** 1/3 proceed

**Blocker:** Spatial information loss in VLM embeddings (C1, Q2 both failed spatial metrics)

**Next Steps:** Evaluate architectural pivot options:
1. Pre-merge ViT features directly
2. Hybrid encoder (VLM semantics + separate spatial encoder)
3. Spatial enhancement modules
4. Alternative VLM architecture

---

## Key Learnings

### What's Working

1. **Latent alignment is feasible** - Q1 confirmed VLM and video decoder spaces can be bridged
2. **Perceptual quality preserved** - LPIPS/SSIM thresholds met consistently
3. **Semantic clustering strong** - VLM clearly encodes category/semantic information

### What's NOT Working

1. **Spatial precision** - IoU consistently below thresholds across all experiments
2. **Object detection** - mAP@0.5 = 0.001 (essentially non-functional)
3. **Token merger** - 2x2 compression destroys positional information
4. **LLM layers** - Spatial info further degraded through transformer layers

### Root Cause Analysis

The Qwen2.5-VL architecture is optimized for **language understanding**, not **spatial preservation**:

1. The 2x2 token merger aggressively compresses spatial tokens (4:1)
2. The LLM attention layers mix information across positions
3. No explicit positional encoding preservation through the pipeline

This is a **fundamental architectural mismatch** for the video generation task, not a training or adaptation issue.

---

## Recommended Pivot Direction

Based on Gate 1 findings, the most promising pivot options are:

1. **Hybrid Encoder** (Recommended) - Keep VLM for semantics, add DINOv2/SAM for spatial
2. **Pre-merge ViT** - Use ViT features before the 2x2 merger
3. **Alternative VLM** - Switch to LLaVA-NeXT or InternVL2 with better spatial preservation

See `/research/proposals/` for detailed analysis of each option.

---

## Changelog

| Date | Update |
|------|--------|
| 2026-01-18 | Gate 1 complete - BLOCKED due to spatial information loss |
| 2026-01-18 | C1 completed - PIVOT (spatial IoU=0.559 < 0.6) |
| 2026-01-18 | Q1 completed - PROCEED (all criteria met) |
| 2026-01-18 | Q2 completed - PIVOT (spatial info severely degraded) |
| 2026-01-18 | Pivot proposals initiated |
