# C1: VLM Latent Sufficiency - Findings

**Status:** ✅ Completed
**Date:** 2026-01-18
**Recommendation:** Proceed to Phase 2

---

## The Question

Can Qwen2.5-VL's internal representation be decoded back into video? Or does it throw away too much visual information when processing images?

## The Answer

**Yes, it works.** The VLM's latent space contains enough information to reconstruct images with high perceptual quality.

---

## Key Results

| Metric | What it measures | Target | Achieved | Verdict |
|--------|------------------|--------|----------|---------|
| LPIPS | Perceptual similarity (lower = better) | < 0.35 | **0.236** | ✅ Exceeds |
| SSIM | Structural similarity (higher = better) | > 0.75 | **0.946** | ✅ Exceeds |
| Spatial IoU | Object location accuracy | > 0.60 | 0.567 | ⚠️ Marginal |

---

## Sub-experiment Summaries

### E1.1: Latent Space Visualization
**Finding:** Strong semantic clustering (silhouette score = 0.808)

The VLM naturally organizes similar images together. Red circles cluster near other red circles, far from blue squares. This means the VLM isn't just hashing images randomly - it understands visual similarity.

### E1.2: Linear Reconstruction Probe
**Finding:** Information is accessible (LPIPS = 0.331)

Even a simple linear decoder can produce recognizable reconstructions. This proves the information exists in the latent space and isn't deeply entangled.

### E1.3: Pre-merge vs Post-merge Comparison
**Finding:** Post-merge is surprisingly BETTER

| Extraction Point | LPIPS | SSIM |
|------------------|-------|------|
| Pre-merge (before 2x2 compression) | 1.542 | 0.011 |
| Post-merge (after 2x2 compression) | 0.342 | 0.910 |

We expected the token merger to lose information. Instead, it seems to aggregate information helpfully. This simplifies the architecture - we can use the standard output.

### E1.4: Spatial Information Test
**Finding:** Approximate but not precise positioning (IoU = 0.567)

The VLM knows roughly where objects are, but loses some precision. 68% of test images achieved IoU > 0.5, but only 8% achieved IoU > 0.75.

This is the one area below target - may need attention for tasks requiring precise spatial verification.

### E1.5: Full Reconstruction Pipeline
**Finding:** End-to-end works! (LPIPS = 0.236)

A ~20M parameter adapter successfully bridges VLM latents to a VAE decoder. This is the core validation - the full pipeline produces high-quality reconstructions.

### E1.6: Ablation Studies
**Finding:** Layer 14 is optimal, ~20M params is sufficient

| Factor | Best Value | Notes |
|--------|------------|-------|
| Layer depth | Layer 14 | Middle layers beat early or late |
| Adapter size | 2048 hidden dim (~20M params) | Larger helps but diminishing returns |
| Training data | 100+ samples | Quality improves up to ~100 samples |

---

## Surprising Findings

### 1. Post-merge beats pre-merge
The 2x2 token merger doesn't destroy information - it aggregates it helpfully. This was counterintuitive.

**Implication:** Use the simpler post-merge representation.

### 2. Middle layers are best
Layer 14 (middle of 28 layers) provides optimal reconstruction. Early layers lack semantic context; late layers are too abstract for pixel-level reconstruction.

**Implication:** Extract from middle layers, not the final layer.

### 3. Small adapter is sufficient
20M parameters is enough. The mapping between spaces isn't as complex as feared.

**Implication:** Start small, scale up only if needed.

---

## Caveats & Limitations

1. **Synthetic data only** - Tested on colored shapes (circles, squares, triangles). Real video validation needed.

2. **Single-object scenes** - Multi-object reconstruction not validated. Occlusion, overlapping objects may be harder.

3. **Spatial precision marginal** - IoU of 0.567 is below the 0.6 target. Fine-grained verification tasks may struggle.

4. **No temporal testing** - These were static images. Video temporal coherence not tested here.

---

## Recommendations for Phase 2

| Decision | Recommendation | Rationale |
|----------|----------------|-----------|
| Which latents? | Post-merge | Simpler, works better |
| Which layer? | Layer 14 | Best reconstruction quality |
| Adapter size? | ~20M params | Sufficient for the task |
| Training data? | 100+ samples | Quality plateaus around 100 |

### Next Steps
1. Validate on real video (Something-Something v2)
2. Test multi-object scenes
3. If spatial precision becomes an issue, investigate layer 7-14 blend

---

## Artifacts

- W&B Dashboard: https://wandb.ai/a1j9o94/foresight
- Detailed metrics: `results.yaml`
- Visualizations: `artifacts/`
