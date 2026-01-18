# Q2: Information Preservation - Findings

**Status:** Completed
**Date:** 2026-01-18
**Recommendation:** PIVOT - Use hybrid approach with dedicated spatial models

---

## The Question

How much spatial information survives at each stage of the VLM pipeline? The VLM has several processing stages - patch embedding, ViT layers, 2x2 token merger, LLM layers. Where does spatial information get lost, and where should we extract features from?

## The Answer

**Spatial information is NOT preserved in Qwen2.5-VL embeddings.** This is a fundamental limitation of the model architecture, not just an extraction point issue. The VLM was optimized for semantic understanding, not spatial preservation.

However, **temporal information IS well preserved** (100% direction accuracy, 90% ordering accuracy), making the VLM useful for video generation where motion semantics matter more than precise localization.

---

## Key Results

| Metric | What it measures | Target | Achieved | Verdict |
|--------|------------------|--------|----------|---------|
| Bbox IoU | Can we locate objects? | > 0.7 | 0.103 | FAIL |
| LPIPS | Reconstruction quality | < 0.3 | 0.087 | PASS |
| Edge F1 | Are boundaries preserved? | > 0.6 | 0.367 | FAIL |
| IRS | Information Retention Score | > 0.6 | 0.707 | PASS |
| mAP@0.5 | Detection accuracy | > 0.4 | 0.001 | FAIL |
| Direction Acc | Motion direction | > 80% | 100% | PASS |
| Order Acc | Temporal ordering | > 75% | 90% | PASS |

**Overall: 4/7 criteria passed, but spatial criteria (the most critical) failed.**

---

## Sub-experiment Summaries

### E-Q2.1: Pre-merge ViT Analysis (Baseline)

**Finding:** Pre-merge embeddings show poor spatial retention (IoU=0.101).

This was surprising - we expected the pre-merge features (before the 2x2 token merger) to preserve spatial information well. Instead, spatial info is already degraded at this stage. This suggests the VLM's vision encoder wasn't trained with spatial preservation as an objective.

- Bbox IoU: 0.101 (target: 0.7)
- IoU > 0.5: 4.0%
- Feature dimension: 3584

**Implication:** Even bypassing the merger won't help - the problem is upstream.

### E-Q2.2: Post-merge Analysis

**Finding:** Post-merge embeddings show similar spatial retention (IoU=0.103).

The 2x2 merger compresses 1024 tokens to 256 (4x compression). Despite this aggressive compression, spatial IoU stayed roughly the same (~0.10). This confirms the spatial info was already gone before merging.

- Compression ratio: 4.0x
- IoU drop from pre-merge baseline: ~0% (already poor)

**Implication:** The merger isn't the bottleneck; the vision encoder is.

### E-Q2.3: LLM Layer-wise Decay

**Finding:** Spatial information is lost before entering LLM. First layer IoU: 0.066.

We traced features through all 28 LLM layers. Spatial probing performance was poor at every layer:

- Layer 0: IoU = 0.066
- Layer 6: IoU = 0.065
- Layer 13: IoU = 0.094
- Layer 20: IoU = 0.097
- Layer 27: IoU = 0.042

Interestingly, middle layers (13-20) showed slightly better IoU, suggesting some spatial information might be reconstructed during processing.

**Implication:** LLM layers don't help; focus on ViT-level extraction.

### E-Q2.4: Detection Probe (mAP)

**Finding:** Detection probe achieves near-zero mAP@0.5 (0.001).

We trained a DETR-style detection head on the VLM features. Results were extremely poor:

- mAP@0.5: 0.001 (target: 0.4)
- mAP@0.75: 0.0

This is essentially random performance. The model cannot localize objects from these features.

**Implication:** Do not rely on VLM embeddings for object localization.

### E-Q2.5: Fine-grained Detail Probe

**Finding:** Partial fine-grained detail preservation at llm_layer_0.

We trained a reconstruction decoder to regenerate images from VLM features:

- LPIPS: 0.087 (target: <0.3) - **PASS**
- Edge F1: 0.340 (target: >0.6) - **FAIL**
- SSIM: 0.963
- IRS: 0.689 (target: >0.6) - **PASS**

The reconstructions look perceptually reasonable (good LPIPS/SSIM) but lack sharp edges (poor Edge F1). This matches intuition - VLMs preserve semantic content but lose high-frequency details.

**Implication:** Fine for coarse video generation, may struggle with sharp details.

### E-Q2.6: Temporal Information Probe

**Finding:** Temporal information well preserved at post_merge.

This was the positive surprise. Using synthetic moving object videos:

- Direction accuracy: 100% (target: 80%) - **PASS**
- Speed accuracy: 100% (target: 75%) - **PASS**
- Temporal ordering: 90% (target: 75%) - **PASS**

The VLM features encode motion direction and sequence ordering very well, even though they don't encode spatial location precisely.

**Implication:** VLM features ARE suitable for conditioning video generation where motion semantics matter.

---

## Actual Information Retention Pattern

```
Information Retention by Type

Spatial (Bbox IoU):
10%  |██                   All stages (~0.1 IoU)
     |
     | Spatial info never preserved

Perceptual (LPIPS):
91%  |████████████████████ Post-merge (0.087 LPIPS)
     |
87%  |█████████████████    LLM Layer 0 (0.099 LPIPS)

Temporal:
100% |████████████████████ Post-merge direction
90%  |██████████████████   Post-merge ordering
     |
0%   |                     LLM layers (loses temporal)
```

**Key Insight:** Different types of information are preserved differently. Spatial is lost early, perceptual degrades slowly, temporal is lost in LLM layers.

---

## Recommendations

### Immediate Actions

1. **Do NOT use VLM embeddings for precise spatial localization**
   - mAP and IoU results are near-zero
   - This is a fundamental model limitation

2. **DO use VLM embeddings for:**
   - Motion/temporal conditioning (excellent preservation)
   - Semantic scene understanding
   - Coarse visual guidance

3. **For spatial needs, use dedicated models:**
   - Add OwlVIT or YOLO for object detection
   - Use grounding-specialized VLMs (Kosmos-2, Grounding-DINO) if localization is critical

### Recommended Architecture Change

```
Instead of:
  Image -> VLM -> Video Decoder

Use:
  Image -> VLM -> Semantic/Temporal features -> Video Decoder
       \-> Detection Model -> Spatial features -------^
```

---

## Artifacts

- W&B Dashboard: https://wandb.ai/a1j9o94/foresight/runs/hn5fajsb
- Detailed metrics: `results.yaml`
- Visualizations: `artifacts/` directory
