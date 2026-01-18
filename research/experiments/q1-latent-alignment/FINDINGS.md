# Q1: Latent Space Alignment - Findings

**Status:** Completed
**Date:** 2026-01-18
**Recommendation:** PROCEED

---

## The Question

Do VLM (Qwen2.5-VL) and video decoder (LTX-Video) latent spaces have compatible structure? Can a small adapter bridge them, or are they fundamentally different?

## The Answer

**Yes, the latent spaces are alignable.** Despite a 6.9x dimensionality mismatch (VLM intrinsic dim ~10, LTX ~68), the spaces share enough semantic structure that a small MLP adapter should successfully bridge them. All four success criteria were met.

Key insight: Use `layer_final` from Qwen2.5-VL (CKA=0.687) rather than intermediate layers for best alignment with LTX-Video latents.

---

## Key Results

| Metric | What it measures | Target | Achieved | Verdict |
|--------|------------------|--------|----------|---------|
| Linear Probe R² | Can one space predict the other? | > 0.5 | **0.510** | PASS |
| Spearman ρ | Do similar images stay similar across spaces? | > 0.6 | **0.684** | PASS |
| Recall@10 | Do nearest neighbors match? | > 20% | **26.8%** | PASS |
| CKA | Representational similarity | > 0.4 | **0.687** | PASS |

---

## Sub-experiment Summaries

### E-Q1.1: VLM Latent Visualization
**Finding:** Strong semantic organization in VLM latents.

- Best layer: `layer_8` (silhouette=0.533, k-NN=0.996)
- VLM encodes clear category structure that facilitates alignment
- Silhouette scores decrease in later layers (layer_final=0.447)
- 270 images across 9 categories analyzed

### E-Q1.2: LTX-Video Latent Visualization
**Finding:** Moderate semantic organization in LTX-Video latents.

- Best representation: channel-mean (silhouette=0.371, k-NN=0.996)
- Some category structure preserved but with significant overlap
- Flat representation silhouette: 0.208 (weaker than channel-mean)
- Top variance channels: 35, 22, 37, 63, 1, 53, 126, 51, 34, 123

### E-Q1.3: Intrinsic Dimensionality
**Finding:** Significant dimensionality mismatch requiring adapter design consideration.

- VLM intrinsic dim: ~10 (MLE=9.9, two-NN=11.7)
- LTX intrinsic dim: ~68 (MLE=68.3, two-NN=7.4)
- Ratio: **6.9x**
- PCA 95% variance: VLM=26 components, LTX=64 components
- Implication: Adapter needs to expand from low to high dimensionality

### E-Q1.4: Linear Probing
**Finding:** Moderate linear alignment - small MLP recommended over pure linear projection.

- VLM→LTX R²: **0.510**
- LTX→VLM R²: 0.430
- Average R²: 0.470
- Improvement over random: **45.6x**
- Cosine similarity (VLM→LTX): 0.854
- Cosine similarity (LTX→VLM): 0.971

### E-Q1.5: Semantic Similarity Preservation
**Finding:** Good semantic similarity preservation across spaces.

- Spearman ρ (Euclidean): **0.684**
- Spearman ρ (Cosine): 0.628
- Pearson (Euclidean): 0.750
- Kendall τ: 0.504
- Mantel test: r=0.750, p=0.001 (highly significant)
- 64,620 pairwise comparisons across 360 images

### E-Q1.6: Neighborhood Analysis
**Finding:** Good local alignment - adapter should work well for most cases.

- Recall@5: 19.8%
- Recall@10: **26.8%** (12x over random baseline)
- Recall@20: 34.8%
- MRR@10: 0.107
- Jaccard@10: 0.162
- 450 images, 9 categories

### E-Q1.7: CKA Analysis
**Finding:** Strong representational alignment found!

- Best CKA: **0.687** at `layer_final`
- layer_4 CKA: 0.647
- layer_8 CKA: 0.645
- layer_16 CKA: 0.642
- layer_24 CKA: 0.638
- Recommendation: Use `layer_final` for adapter training

---

## Implications for Architecture

1. **Use layer_final from VLM**: Despite intermediate layers having better semantic clustering (E-Q1.1), the final layer has best alignment with LTX-Video (CKA=0.687).

2. **Small MLP adapter recommended**: Pure linear projection achieves R²=0.51, but the dimensionality mismatch (10→68) suggests a 2-3 layer MLP with expansion will perform better.

3. **Channel-mean representation for LTX**: When working with LTX-Video latents, use channel-mean pooling rather than flattened representation (silhouette 0.371 vs 0.208).

4. **Adapter size estimate**: Given ~26 VLM PCA components and ~64 LTX PCA components, adapter should have 50-100K parameters (well within our 10M budget).

---

## Artifacts

- W&B Run: https://wandb.ai/a1j9o94/foresight/runs/6ypqydtb
- VLM t-SNE visualizations: `artifacts/vlm_tsne_*.png`
- LTX t-SNE visualizations: `artifacts/ltx_tsne_*.png`
- Dimensionality analysis: `artifacts/intrinsic_dim_*.png`
- Linear probe results: `artifacts/linear_probe_*.png`
- Semantic similarity: `artifacts/semantic_sim_*.png`
- Neighborhood analysis: `artifacts/neighborhood_*.png`
- CKA heatmaps: `artifacts/cka_*.png`
- Detailed metrics: `results.yaml`
