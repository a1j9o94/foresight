# E-Q1: Latent Space Alignment Research Plan

**Status:** Not Started
**Risk Level:** HIGH - Core Technical Uncertainty
**Estimated Duration:** 10-14 days
**Dependencies:** None (can run in parallel with Claims testing)

---

## 1. Objective

Understand the geometric structure and semantic organization of two latent spaces that were trained with fundamentally different objectives, and determine the feasibility of bridging them with a small adapter network.

**Primary Questions:**
1. How are VLM latent spaces structured (Qwen2.5-VL)?
2. How are diffusion model latent spaces structured (LTX-Video)?
3. Are there shared structures that a linear or near-linear mapping can exploit?
4. What does this tell us about the adapter architecture we need?

**Null Hypothesis to Falsify:**
"The VLM and video decoder latent spaces have fundamentally incompatible structures that cannot be bridged without learning an essentially arbitrary (memorized) mapping."

---

## 2. Background

### 2.1 How VLM Latent Spaces Are Structured

Vision-Language Models like Qwen2.5-VL learn representations optimized for next-token prediction on text. Key structural properties:

**Semantic Clustering:**
- Representations cluster by semantic category (e.g., "dogs" near "cats", "cars" near "trucks")
- Fine-grained visual attributes (pose, lighting, background) are often entangled or discarded
- Temporal information in video frames may be compressed or abstracted

**Hierarchical Organization:**
- Early layers capture low-level visual features (edges, textures)
- Middle layers capture objects and spatial relationships
- Late layers capture high-level semantics for language generation

**Geometric Properties:**
- Often exhibit cone-like structure (all embeddings have positive dot product with a "mean" direction)
- Semantic similarity roughly correlates with cosine similarity
- Anisotropic - certain directions encode much more variance than others

**Qwen2.5-VL Specifics:**
- ViT output: 1536-dim embeddings per patch (pre-merge)
- After 2x2 merge: compressed but semantically rich tokens
- LLM hidden states: 3584-dim (7B model) with multimodal fusion

### 2.2 How Diffusion Model Latent Spaces Are Structured

Video diffusion models like LTX-Video learn representations optimized for denoising. Key structural properties:

**Noise-Conditioned Structure:**
- Latent space is structured around the denoising process
- Clean latents (t=0) cluster by visual appearance
- Noisy latents (t=1) approach isotropic Gaussian

**Frequency-Based Organization:**
- Different channels often correspond to different frequency bands
- Low-frequency (coarse structure) vs high-frequency (fine details)
- LTX-Video uses 128 channels specifically to preserve information

**Spatial Coherence:**
- Strong local correlations (nearby pixels encoded similarly)
- Global structure often captured in low-frequency channels
- Temporal coherence built into 3D convolution patterns

**LTX-Video Specifics:**
- 128-channel latents at 32x32x8 spatiotemporal compression
- Trained with MSE, perceptual (LPIPS), and GAN losses
- Denoising decoder expects latents at specific noise levels (t~0.05)

### 2.3 Prior Work on Latent Space Alignment

**CLIP (Radford et al., 2021):**
- Aligned vision and language representations via contrastive learning
- Demonstrated that modality-specific representations can be mapped to shared space
- Key insight: alignment doesn't require identical training objectives

**CKA (Kornblith et al., 2019):**
- Centered Kernel Alignment measures representational similarity
- Works across different dimensionalities
- Standard tool for comparing neural network representations

**Cross-Modal Retrieval:**
- Image-text retrieval systems demonstrate cross-space mapping feasibility
- Often use simple linear projections as first baseline
- Non-linear mappings (MLPs) typically provide modest improvements

**Latent Diffusion Model Adapters:**
- ControlNet demonstrates conditioning diffusion with external representations
- IP-Adapter maps CLIP image features to diffusion conditioning space
- These work with ~10-100M parameter adapters

**Relevant Failures:**
- Some latent spaces are truly incompatible (different manifold structures)
- High-dimensional spaces can appear "closer" than they are (curse of dimensionality)
- Linear probes failing suggests need for non-linear mapping or fundamental incompatibility

---

## 3. Experimental Setup

### 3.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 24GB (4090) | 40GB+ (A100) |
| System RAM | 32GB | 64GB |
| Storage | 100GB | 500GB |

**Rationale:**
- Qwen2.5-VL-7B: ~14GB in bfloat16
- LTX-Video VAE: ~2GB
- Headroom for batched inference and visualization

### 3.2 Software Dependencies

```bash
# Core ML
pip install torch torchvision
pip install transformers diffusers accelerate
pip install flash-attn --no-build-isolation

# Visualization
pip install umap-learn scikit-learn
pip install matplotlib seaborn plotly

# Analysis
pip install scipy
pip install pytorch-fid lpips
pip install cka  # or implement from scratch

# Optional but recommended
pip install wandb  # experiment tracking
pip install jupyterlab  # interactive analysis
```

### 3.3 Sample Dataset

**Primary Dataset: Something-Something v2 (subset)**
- 500-1000 video clips, ~5 seconds each
- Diverse actions and scenes
- Already annotated with actions (useful for semantic analysis)

**Selection Criteria:**
- Diverse visual content (avoid mode collapse in analysis)
- Clear actions (for semantic alignment testing)
- Variable scene complexity

**Data Preparation:**
```python
# Extract frames at consistent intervals
# Target: 5 frames per video, 256x256 resolution
# Store as: video_id/frame_{0-4}.png
```

### 3.4 Latent Extraction Pipeline

**VLM Latents (Qwen2.5-VL):**
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def extract_vlm_latents(image_path):
    """Extract latents at multiple extraction points."""
    inputs = processor(images=image, return_tensors="pt")

    # Point 1: ViT output (pre-merge)
    vit_output = model.visual(inputs["pixel_values"], inputs["image_grid_thw"])

    # Point 2: After token merging
    # (Access via model internals)

    # Point 3: LLM hidden states at various layers
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # Tuple of layer outputs

    return {
        "vit_pre_merge": vit_output,
        "llm_hidden_states": hidden_states,
    }
```

**LTX-Video Latents:**
```python
from diffusers import LTXPipeline

def extract_ltx_latents(image_path):
    """Extract VAE encoder latents."""
    vae = pipe.vae

    image_tensor = preprocess(image)  # Normalize, resize
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()

    return latents  # Shape: [1, 128, H//32, W//32]
```

---

## 4. Experiments

### E-Q1.1: Visualize VLM Latent Space Structure

**Objective:** Understand how Qwen2.5-VL organizes visual information internally.

**Method:**
1. Extract VLM latents for 1000 images (diverse categories)
2. Apply dimensionality reduction (PCA, t-SNE, UMAP)
3. Color by semantic category (action type, object class)
4. Analyze clustering patterns

**Extraction Points:**
- ViT layer 12 (mid-depth)
- ViT layer 24 (late)
- ViT output (post-merge)
- LLM layer 8, 16, 24 (visual token positions only)

**Visualizations:**
- 2D projections colored by category
- Cluster purity scores per category
- Nearest-neighbor accuracy (does k-NN classify correctly?)

**Analysis Questions:**
- At which layer do semantic clusters emerge?
- How tight are clusters (intra-class variance)?
- Are visual details (pose, lighting) preserved or discarded?

**Deliverable:** Report with visualizations showing semantic organization by layer.

**Duration:** 2 days

---

### E-Q1.2: Visualize LTX-Video Latent Space Structure

**Objective:** Understand how LTX-Video organizes visual information in its VAE latent space.

**Method:**
1. Extract LTX-Video VAE latents for same 1000 images
2. Flatten spatial dimensions, analyze per-channel statistics
3. Apply dimensionality reduction
4. Color by same semantic categories as E-Q1.1

**Analysis:**
- Channel-wise variance (which channels carry most information?)
- Spatial coherence (are nearby latent positions correlated?)
- Frequency analysis (do channels correspond to frequency bands?)

**Visualizations:**
- 2D projections colored by category (compare to E-Q1.1)
- Per-channel activation distributions
- Spatial autocorrelation maps

**Analysis Questions:**
- Does LTX latent space cluster semantically?
- Which channels encode which aspects (content vs style vs texture)?
- How does spatial structure differ from VLM's token structure?

**Deliverable:** Report with visualizations showing LTX latent organization.

**Duration:** 2 days

---

### E-Q1.3: Measure Intrinsic Dimensionality of Both Spaces

**Objective:** Determine the effective dimensionality of both latent spaces.

**Rationale:** If VLM latents live on a low-dimensional manifold within the high-dimensional space, alignment may be easier than the nominal dimensionality suggests.

**Method:**
1. **PCA Analysis:**
   - Compute explained variance ratio
   - Find k where 95% of variance is explained

2. **Intrinsic Dimensionality Estimators:**
   - Maximum Likelihood Estimation (Levina & Bickel, 2004)
   - Two-NN estimator (Facco et al., 2017)
   - Correlation dimension

3. **Manifold Analysis:**
   - Local linear embedding dimensionality
   - Isomap geodesic distances

**Measurements:**
| Metric | VLM (ViT) | VLM (LLM) | LTX-Video |
|--------|-----------|-----------|-----------|
| Nominal dim | 1536 | 3584 | 128 |
| PCA 95% | ? | ? | ? |
| MLE intrinsic dim | ? | ? | ? |
| Two-NN intrinsic dim | ? | ? | ? |

**Significance:**
- If intrinsic dimensions are similar, linear alignment more likely
- Large difference suggests need for dimension-reducing or expanding adapter
- Very low intrinsic dim suggests representations are heavily constrained

**Deliverable:** Table of intrinsic dimensionality estimates with interpretation.

**Duration:** 1 day

---

### E-Q1.4: Linear Probing (Predict One Space from Other)

**Objective:** Test whether a simple linear transformation can map between spaces.

**Method:**
1. **VLM -> LTX Probe:**
   - Train linear layer: VLM latent -> LTX latent (flattened)
   - Loss: MSE on LTX latents
   - Evaluate: R^2, cosine similarity

2. **LTX -> VLM Probe:**
   - Train linear layer: LTX latent (flattened) -> VLM latent
   - Same metrics

3. **Comparison Baselines:**
   - Random projection (sanity check)
   - PCA-reduced random projection
   - Identity mapping (if dimensions match after projection)

**Training Details:**
- Dataset: 800 train / 200 test images
- Optimizer: Adam, lr=1e-3
- Epochs: 100 (with early stopping)
- Regularization: L2 weight decay

**Code Sketch:**
```python
class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

# Train
for epoch in range(100):
    pred = probe(vlm_latents)
    loss = F.mse_loss(pred, ltx_latents)
    loss.backward()
    optimizer.step()
```

**Metrics:**
| Probe Direction | R^2 | Cosine Sim | Reconstruction LPIPS |
|-----------------|-----|------------|---------------------|
| VLM -> LTX | ? | ? | ? |
| LTX -> VLM | ? | ? | ? |
| Random baseline | ? | ? | ? |

**Success Criteria:**
- R^2 > 0.5: Linear alignment promising
- R^2 > 0.7: Linear alignment likely sufficient
- R^2 < 0.3: Need non-linear adapter

**Deliverable:** Linear probe results with visualizations of prediction quality.

**Duration:** 2 days

---

### E-Q1.5: Semantic Similarity Preservation Test

**Objective:** Test whether semantic relationships are preserved across spaces.

**Rationale:** Even if the spaces aren't linearly alignable, they might preserve the same similarity structure (which is sufficient for learning an adapter).

**Method:**
1. Compute pairwise distances in VLM space (1000x1000 matrix)
2. Compute pairwise distances in LTX space (same images)
3. Measure correlation between distance matrices
4. Analyze which pairs are similar in one space but not the other

**Metrics:**
- **Spearman correlation:** Rank correlation of pairwise distances
- **Procrustes analysis:** Optimal rotation alignment of spaces
- **Mantel test:** Statistical significance of correlation

**Semantic Categories for Analysis:**
- Same action, different scene
- Same scene, different action
- Similar objects (e.g., all "pouring" actions)
- Visually similar but semantically different

**Visualization:**
- Scatter plot: VLM distance vs LTX distance (per pair)
- Highlighted outliers (pairs where spaces disagree)
- Category-specific correlation analysis

**Success Criteria:**
- Spearman rho > 0.6: Good structural alignment
- Spearman rho > 0.8: Excellent structural alignment
- Spearman rho < 0.4: Concerning structural mismatch

**Deliverable:** Correlation analysis with breakdown by semantic category.

**Duration:** 1 day

---

### E-Q1.6: Neighborhood Analysis (Cross-Space Retrieval)

**Objective:** Test whether nearest neighbors are consistent across spaces.

**Rationale:** For a small adapter to work, similar items should be similar in both spaces.

**Method:**
1. For each image, find k=10 nearest neighbors in VLM space
2. Check how many of those neighbors are also near in LTX space
3. Compute Recall@k for various k values
4. Analyze systematic differences

**Metrics:**
- **Recall@k:** Fraction of VLM neighbors that are also LTX neighbors
- **Mean Reciprocal Rank:** Average rank of VLM neighbors in LTX space
- **Neighborhood overlap:** Jaccard similarity of k-NN sets

**Code Sketch:**
```python
def neighborhood_overlap(vlm_embeds, ltx_embeds, k=10):
    # Compute k-NN in both spaces
    vlm_nn = faiss.IndexFlatIP(vlm_embeds.shape[1])
    vlm_nn.add(vlm_embeds)
    vlm_neighbors = vlm_nn.search(vlm_embeds, k+1)[1][:, 1:]  # Exclude self

    ltx_nn = faiss.IndexFlatIP(ltx_embeds.shape[1])
    ltx_nn.add(ltx_embeds)
    ltx_neighbors = ltx_nn.search(ltx_embeds, k+1)[1][:, 1:]

    # Compute overlap
    overlaps = []
    for i in range(len(vlm_embeds)):
        overlap = len(set(vlm_neighbors[i]) & set(ltx_neighbors[i])) / k
        overlaps.append(overlap)

    return np.mean(overlaps)
```

**Results Table:**
| k | Recall@k | Expected (Random) | Significance |
|---|----------|-------------------|--------------|
| 5 | ? | 0.5% | ? |
| 10 | ? | 1% | ? |
| 50 | ? | 5% | ? |
| 100 | ? | 10% | ? |

**Success Criteria:**
- Recall@10 > 20%: Spaces share local structure
- Recall@10 > 40%: Strong local alignment
- Recall@10 < 10%: Local structure very different

**Deliverable:** Neighborhood analysis report with retrieval examples.

**Duration:** 1 day

---

### E-Q1.7 (Optional): Cross-Space CKA Analysis

**Objective:** Use Centered Kernel Alignment to measure representational similarity.

**Method:**
1. Compute CKA between VLM layers and LTX latent space
2. Create CKA heatmap showing which layers align best
3. Identify optimal extraction point for VLM latents

**Code Sketch:**
```python
def cka(X, Y):
    """Compute CKA between two representation matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    XtX = X @ X.T
    YtY = Y @ Y.T

    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)
```

**Expected Output:**
- CKA heatmap: VLM layers (rows) vs LTX channels (columns)
- Identification of VLM layer with highest CKA to LTX

**Duration:** 1 day (if needed)

---

## 5. Success Metrics

### Primary Metrics

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Linear Probe R^2 | > 0.5 | Linear alignment feasible |
| Semantic Correlation (Spearman) | > 0.6 | Structural similarity preserved |
| Neighborhood Recall@10 | > 20% | Local structure shared |
| CKA (best layer pair) | > 0.4 | Meaningful representational overlap |

### Composite Score

```
Alignment Score = 0.3 * R^2 + 0.3 * Spearman + 0.2 * Recall@10 + 0.2 * CKA
```

| Score Range | Interpretation |
|-------------|----------------|
| > 0.6 | Excellent - linear adapter likely sufficient |
| 0.4 - 0.6 | Good - small MLP adapter should work |
| 0.2 - 0.4 | Challenging - need significant adapter capacity |
| < 0.2 | Poor - fundamental incompatibility likely |

---

## 6. Failure Criteria

The spaces are **fundamentally incompatible** if ANY of:

1. **Linear Probe Complete Failure:**
   - R^2 < 0.1 AND doesn't improve with regularization
   - Predictions are essentially random

2. **Anti-Correlated Structure:**
   - Semantic Spearman rho < 0.2 or negative
   - Images similar in VLM space are distant in LTX space

3. **No Shared Local Structure:**
   - Neighborhood Recall@10 < 5% (barely above random)
   - No improvement at any k

4. **Fundamentally Different Manifolds:**
   - Intrinsic dimensionality differs by >10x
   - Manifold topology differs (one is locally Euclidean, other is not)

5. **Channel-Semantic Mismatch:**
   - LTX channels encode pure noise/frequency, no semantic signal
   - VLM discards all spatial information needed for reconstruction

### What "Incompatible" Means for the Project

If spaces are incompatible:
- Adapter cannot learn a generalizable mapping
- Would need to memorize mappings for training data
- Won't generalize to novel scenes/actions
- Core architecture hypothesis is falsified

**Mitigation Options (if failure detected):**
1. Extract VLM latents from earlier layers (pre-language mixing)
2. Use different video decoder with more compatible latent space
3. Train joint embedding space (CLIP-style) - adds significant complexity
4. Pivot to text-conditioned generation (abandon visual latent prediction)

---

## 7. Implications for Architecture

### If Alignment is Good (Score > 0.5)

**Adapter Design:**
- Start with linear projection
- Add 1-2 MLP layers only if needed
- Target: <10M parameters

```python
class SimpleAdapter(nn.Module):
    def __init__(self, vlm_dim=1536, ltx_dim=128*H*W):
        super().__init__()
        self.proj = nn.Linear(vlm_dim, ltx_dim)

    def forward(self, vlm_latent):
        return self.proj(vlm_latent).reshape(-1, 128, H, W)
```

### If Alignment is Moderate (Score 0.3-0.5)

**Adapter Design:**
- MLP with residual connections
- Potentially add cross-attention
- Target: 10-50M parameters

```python
class MLPAdapter(nn.Module):
    def __init__(self, vlm_dim=1536, ltx_dim=128*H*W, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vlm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ltx_dim),
        )

    def forward(self, vlm_latent):
        return self.net(vlm_latent).reshape(-1, 128, H, W)
```

### If Alignment is Poor (Score 0.2-0.3)

**Adapter Design:**
- Need significant capacity
- Consider attention-based adapter
- May need to train with reconstruction loss
- Target: 50-100M parameters

```python
class CrossAttentionAdapter(nn.Module):
    def __init__(self, vlm_dim=1536, ltx_dim=128, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, H*W, ltx_dim))
        self.cross_attn = nn.MultiheadAttention(ltx_dim, num_heads)
        self.proj_k = nn.Linear(vlm_dim, ltx_dim)
        self.proj_v = nn.Linear(vlm_dim, ltx_dim)

    def forward(self, vlm_tokens):  # [B, N_tokens, vlm_dim]
        k = self.proj_k(vlm_tokens)
        v = self.proj_v(vlm_tokens)
        q = self.query.expand(vlm_tokens.size(0), -1, -1)
        out, _ = self.cross_attn(q, k, v)
        return out.reshape(-1, 128, H, W)
```

### Extraction Point Recommendations

Based on CKA analysis results, recommend optimal VLM extraction point:

| If Best CKA at... | Recommendation |
|-------------------|----------------|
| ViT output (pre-merge) | Use spatial features, may need spatial downsampling in adapter |
| ViT output (post-merge) | Good default - balance of spatial and semantic |
| LLM early layers | Use before too much language mixing |
| LLM late layers | Semantic but may lose spatial detail |

---

## 8. Timeline

| Day | Activity | Deliverable |
|-----|----------|-------------|
| 1 | Setup: install dependencies, download models, prepare dataset | Working extraction pipeline |
| 2-3 | E-Q1.1: VLM latent visualization | VLM structure report |
| 4-5 | E-Q1.2: LTX latent visualization | LTX structure report |
| 6 | E-Q1.3: Intrinsic dimensionality | Dimensionality comparison |
| 7-8 | E-Q1.4: Linear probing | Probe results + visualizations |
| 9 | E-Q1.5: Semantic similarity preservation | Correlation analysis |
| 10 | E-Q1.6: Neighborhood analysis | Retrieval results |
| 11 | E-Q1.7: CKA analysis (if needed) | Layer alignment heatmap |
| 12-14 | Analysis, write-up, architecture recommendations | Final report |

**Total: 10-14 days**

### Parallelization Opportunities

- E-Q1.1 and E-Q1.2 can run in parallel (different models)
- E-Q1.4, E-Q1.5, E-Q1.6 can run in parallel (once latents extracted)
- E-Q1.3 and E-Q1.7 can run alongside other analysis

**With parallelization: 7-10 days**

---

## 9. Dependencies

**No blocking dependencies.** This experiment sequence can start immediately.

**Enables:**
- Claim 2 testing (adapter design informed by alignment results)
- Training configuration (learning rate, capacity based on alignment difficulty)
- Go/no-go decision on overall architecture feasibility

**Data Dependencies:**
- Need access to Something-Something v2 or equivalent video dataset
- Can use proxy dataset (COCO, ImageNet) for initial testing if needed

---

## 10. Deliverables

### D1: Visualization Report (Day 5)
- t-SNE/UMAP plots for both spaces
- Semantic clustering analysis
- Per-layer/channel breakdowns
- Format: Jupyter notebook + PDF export

### D2: Alignment Analysis (Day 10)
- Linear probe results with confidence intervals
- Correlation and neighborhood metrics
- CKA heatmaps
- Statistical significance tests
- Format: Technical report (Markdown + figures)

### D3: Architecture Recommendations (Day 14)
- Recommended VLM extraction point
- Recommended adapter architecture
- Parameter budget estimate
- Training strategy suggestions
- Risk assessment with mitigation options
- Format: Design document

### D4: Reusable Code Artifacts
- Latent extraction scripts for both models
- Visualization utilities
- Linear probe training code
- Metric computation functions
- Format: Python modules in `src/analysis/`

---

## Appendix A: Relevant Literature

### Latent Space Analysis
- Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited" (CKA)
- Raghu et al. (2021). "Do Vision Transformers See Like CNNs?" (representational analysis)

### Cross-Modal Alignment
- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
- Girdhar et al. (2023). "ImageBind: One Embedding Space To Bind Them All"

### Adapters for Diffusion Models
- Zhang et al. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet)
- Ye et al. (2023). "IP-Adapter: Text Compatible Image Prompt Adapter" (IP-Adapter)

### Intrinsic Dimensionality
- Levina & Bickel (2004). "Maximum Likelihood Estimation of Intrinsic Dimension"
- Facco et al. (2017). "Estimating the intrinsic dimension of datasets by a minimal neighborhood information"

---

## Appendix B: Quick Start Commands

```bash
# Setup
cd /Users/adrianobleton/foresight
pip install -r requirements.txt

# Download models (one-time)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
huggingface-cli download Lightricks/LTX-Video

# Run experiments
python scripts/experiments/q1_extract_latents.py --dataset sthv2_sample
python scripts/experiments/q1_visualize_vlm.py
python scripts/experiments/q1_visualize_ltx.py
python scripts/experiments/q1_linear_probe.py
python scripts/experiments/q1_semantic_correlation.py
python scripts/experiments/q1_neighborhood_analysis.py
python scripts/experiments/q1_cka_analysis.py

# Generate report
python scripts/experiments/q1_generate_report.py
```

---

## Appendix C: Risk Mitigation Strategies

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| VLM loses spatial info in late layers | Medium | High | Extract from earlier layers, test multiple extraction points |
| LTX latent space is pure frequency, no semantics | Low | High | Use different video decoder (HunyuanVideo) |
| Linear probe overfits to training images | Medium | Medium | Cross-validation, held-out test set |
| Intrinsic dimensionality too different | Medium | High | Dimension-matching adapter (autoencoder-style) |
| Results inconclusive | Low | Medium | Extend experiments with larger dataset |

---

**Document Version:** 1.0
**Created:** 2025-01-18
**Last Updated:** 2025-01-18
**Author:** Research Team
