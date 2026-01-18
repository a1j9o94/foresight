# Experiment Plan: C1 - VLM Latent Sufficiency

**Claim:** Qwen2-VL's internal representations contain enough information to reconstruct the input video at reasonable fidelity.

**Status:** Not Started
**Priority:** Critical (foundational - blocks all downstream work)
**Owner:** TBD
**Created:** 2025-01-18

---

## 1. Objective

Determine whether latent representations from Qwen2.5-VL's vision encoder retain sufficient spatial, temporal, and semantic information to support high-fidelity video reconstruction via a downstream decoder.

**Core Question:** If we extract latents from Qwen2.5-VL's vision tower and pass them through an adapter to a video decoder, can we reconstruct the original video with enough detail to preserve task-relevant information (objects, positions, actions)?

**Why This Matters:** This claim is foundational. If the VLM discards spatial information during processing (especially after 2x2 token merging), we cannot generate what it predicts, and the entire GLP architecture fails.

---

## 2. Background

### 2.1 Qwen2.5-VL Vision Encoder Architecture

From the [Qwen2-VL paper](https://arxiv.org/abs/2409.12191) and [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923):

| Property | Value |
|----------|-------|
| Vision encoder parameters | ~675M |
| Patch size | 14x14 pixels |
| Temporal patch size | 2 frames |
| Output embedding dimension | **1536** |
| Token merging | **2x2 spatial merge via MLP** |
| Position encoding | 2D-RoPE (M-RoPE in full model) |
| Attention (Qwen2.5-VL) | 4 full attention + window attention (8x8 max) |
| Normalization (Qwen2.5-VL) | RMSNorm |
| Activation (Qwen2.5-VL) | SwiGLU |

### 2.2 Token Flow

```
Input Image (e.g., 448x448)
    ↓
Patch Embedding (14x14 patches → 32x32 = 1024 patches)
    ↓
ViT Processing (675M params, 1536-dim output per patch)
    ↓
[EXTRACTION POINT A: Pre-merge latents - 1024 tokens @ 1536-dim]
    ↓
2x2 Token Merger (MLP compresses 4 adjacent tokens → 1)
    ↓
[EXTRACTION POINT B: Post-merge latents - 256 tokens @ 1536-dim]
    ↓
LLM Integration (<vision_start>...<vision_end>)
```

### 2.3 Known Risks

1. **Information loss during merge:** The 2x2 MLP merger compresses 4 tokens into 1. This 4x reduction may discard fine-grained spatial information needed for pixel reconstruction.

2. **Semantic vs spatial optimization:** VLMs are optimized for semantic understanding (answering questions), not pixel reconstruction. The latent space may encode "what" but not "where precisely."

3. **Window attention locality:** Qwen2.5-VL uses window attention for efficiency. This may limit global spatial coherence in extracted features.

### 2.4 Related Work

- **CLIP latent reconstruction:** Prior work shows CLIP image embeddings can be inverted to produce recognizable images, but with significant detail loss.
- **Diffusion inversion:** Techniques like DDIM inversion can recover images from diffusion latents, suggesting dense spatial information persists in diffusion models.
- **VQ-VAE reconstruction:** Discrete latent spaces (VQ-VAE) achieve high reconstruction quality when designed for it.

---

## 3. Experimental Setup

### 3.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x A100 40GB | 1x A100 80GB |
| CPU RAM | 64GB | 128GB |
| Storage | 200GB SSD | 500GB NVMe |
| GPU alternatives | 2x RTX 4090 (48GB total) | 1x H100 80GB |

**VRAM breakdown:**
- Qwen2.5-VL-7B (bf16): ~15GB
- LTX-Video (bf16): ~8GB
- Adapter + activations: ~5-10GB
- **Total: ~30GB minimum**

### 3.2 Software Dependencies

```bash
# Core dependencies
pip install torch>=2.1.0 torchvision torchaudio
pip install transformers>=4.40.0 accelerate>=0.27.0
pip install diffusers>=0.27.0
pip install flash-attn --no-build-isolation

# Evaluation
pip install lpips                    # Perceptual similarity
pip install pytorch-fid              # FID computation
pip install scikit-learn             # t-SNE, clustering
pip install umap-learn               # UMAP visualization

# Utilities
pip install wandb                    # Experiment tracking
pip install qwen-vl-utils[decord]    # Video preprocessing
pip install einops                   # Tensor operations
pip install matplotlib seaborn       # Visualization
```

### 3.3 Model Checkpoints

```bash
# Download required models
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
huggingface-cli download Lightricks/LTX-Video

# Optional: for higher quality validation
huggingface-cli download Tencent/HunyuanVideo-1.5
```

### 3.4 Test Datasets

**Phase 1: Synthetic/Controlled (for debugging)**

| Dataset | N samples | Resolution | Why |
|---------|-----------|------------|-----|
| MNIST video | 100 | 64x64 | Simple, known structure |
| Moving MNIST | 100 | 64x64 | Tests temporal consistency |
| Bouncing balls | 100 | 128x128 | Simple physics, clear objects |

**Phase 2: Real-world (for validation)**

| Dataset | N samples | Resolution | Why |
|---------|-----------|------------|-----|
| Something-Something v2 (subset) | 500 | 224x224 | Object manipulation, action diversity |
| DAVIS 2017 | 90 | 480p | High quality, diverse scenes |
| UCF101 (subset) | 500 | 320x240 | Action diversity |

**Initial test clips (manually selected):**

```
# 10 hand-picked clips for debugging pipeline
1. Static scene with single object (baseline)
2. Single object, simple motion (translation)
3. Single object, complex motion (rotation + translation)
4. Multiple objects, static
5. Multiple objects, simple interaction
6. Hand manipulating object (egocentric)
7. Person walking (full body)
8. Scene with text/signage (tests detail preservation)
9. Scene with fine textures (grass, fabric)
10. Rapid motion / motion blur
```

---

## 4. Experiments

### E1.1: Latent Space Visualization

**Objective:** Understand the structure of Qwen2.5-VL's visual latent space before attempting reconstruction.

**Protocol:**

1. Extract pre-merge and post-merge latents from 1000 diverse images
2. Apply dimensionality reduction (t-SNE, UMAP) to visualize
3. Color-code by:
   - Image category (object type, scene type)
   - Spatial position within image (do nearby patches cluster?)
   - Video frame index (does temporal structure exist?)

**Implementation:**

```python
# Pseudocode for E1.1
def extract_latents(model, images):
    """Extract both pre-merge and post-merge latents."""
    with torch.no_grad():
        # Process through vision encoder
        inputs = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        grid_thw = inputs["image_grid_thw"].to(device)

        # Method 1: Full ViT output (pre-merge might need model modification)
        vision_outputs = model.visual(pixel_values, grid_thw)
        post_merge_latents = vision_outputs  # [N_tokens, 1536]

        # Method 2: Hook into intermediate layers for pre-merge
        # (requires registering forward hooks on the model)

    return {"post_merge": post_merge_latents, "pre_merge": pre_merge_latents}

def visualize_latent_space(latents, labels):
    """t-SNE and UMAP visualization."""
    from sklearn.manifold import TSNE
    import umap

    # Flatten to 2D: [N_images * N_tokens, 1536]
    flat_latents = latents.reshape(-1, 1536)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30)
    tsne_result = tsne.fit_transform(flat_latents[:10000])  # Subsample for speed

    # UMAP
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(flat_latents[:10000])

    # Plot with color coding
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, alpha=0.5)
```

**Analysis questions:**

- Do semantically similar images cluster together?
- Do spatially adjacent patches have similar embeddings?
- Is there clear separation between object vs background tokens?
- Does the post-merge space lose spatial structure visible in pre-merge?

**Deliverables:**
- Visualization plots (saved to `results/e1.1/`)
- Quantitative clustering metrics (silhouette score by category)
- Written analysis of latent space structure

**Time estimate:** 1 day

---

### E1.2: Reconstruction Probe (Linear)

**Objective:** Measure upper bound on information content with a simple linear probe.

**Protocol:**

1. Train a linear decoder: `latents → pixels`
2. Evaluate reconstruction quality on held-out images
3. Compare pre-merge vs post-merge latents

**Implementation:**

```python
class LinearReconstructionProbe(nn.Module):
    """Simple linear projection from latents to pixel space."""
    def __init__(self, latent_dim=1536, n_tokens=256, output_size=(224, 224, 3)):
        super().__init__()
        self.output_size = output_size
        flat_output = output_size[0] * output_size[1] * output_size[2]

        # Option 1: Per-token projection + spatial unflatten
        self.per_token_proj = nn.Linear(latent_dim, 14 * 14 * 3)  # Each token -> 14x14 patch

        # Option 2: Global projection (for comparison)
        self.global_proj = nn.Linear(latent_dim * n_tokens, flat_output)

    def forward(self, latents, mode="per_token"):
        if mode == "per_token":
            # [B, N_tokens, 1536] -> [B, N_tokens, 14*14*3]
            patches = self.per_token_proj(latents)
            # Reshape to image
            # ...
        else:
            # Global projection
            flat = latents.flatten(1)
            return self.global_proj(flat).reshape(-1, *self.output_size)
```

**Training:**
- Loss: MSE (L2) pixel loss + LPIPS perceptual loss
- Optimizer: Adam, lr=1e-3
- Epochs: 100
- Batch size: 32
- Train/val split: 80/20

**Metrics:**

| Metric | Formula | What it measures |
|--------|---------|------------------|
| PSNR | -10 log10(MSE) | Pixel-level accuracy |
| SSIM | structural similarity | Structural preservation |
| LPIPS | perceptual distance | High-level visual similarity |

**Expected outcomes:**

- If linear probe achieves LPIPS < 0.5: Information is accessible
- If linear probe achieves LPIPS > 0.7: Information may be entangled or lost
- Pre-merge should outperform post-merge (more spatial info)

**Deliverables:**
- Trained probe checkpoints
- Reconstruction examples (grid of original vs reconstructed)
- Quantitative metrics table
- Learning curves

**Time estimate:** 2 days

---

### E1.3: Pre-merge vs Post-merge Comparison

**Objective:** Quantify information loss from the 2x2 token merging operation.

**Protocol:**

1. Extract latents from both extraction points for same images
2. Train identical reconstruction probes on each
3. Compare reconstruction quality
4. Analyze what information is lost

**Extracting pre-merge latents:**

```python
# Need to hook into the model before the merger MLP
class PreMergeExtractor:
    def __init__(self, model):
        self.model = model
        self.pre_merge_features = None

        # Register hook on the layer before merger
        # The exact layer depends on model architecture
        # For Qwen2.5-VL, the merger is: model.visual.merger
        def hook_fn(module, input, output):
            self.pre_merge_features = input[0]  # or output of previous layer

        # Find and hook the right layer
        # This requires inspecting model.visual structure
        self.hook = model.visual.blocks[-1].register_forward_hook(hook_fn)

    def get_pre_merge(self, pixel_values, grid_thw):
        with torch.no_grad():
            _ = self.model.visual(pixel_values, grid_thw)
        return self.pre_merge_features
```

**Comparison metrics:**

| Extraction Point | Tokens (224x224) | Spatial Resolution | Expected LPIPS |
|-----------------|------------------|-------------------|----------------|
| Pre-merge | 1024 (32x32) | 7x7 per token | Lower (better) |
| Post-merge | 256 (16x16) | 14x14 per token | Higher (worse) |

**Analysis:**

1. **Quantitative gap:** How much does LPIPS degrade from pre-merge to post-merge?
2. **What's lost:** Visualize difference images (original - reconstructed)
3. **Error distribution:** Are errors uniform or concentrated in specific regions?
4. **Object boundary preservation:** Are object edges preserved or blurred?

**Decision point:**

- Gap < 0.1 LPIPS: Post-merge is sufficient, use for simplicity
- Gap 0.1-0.2 LPIPS: Consider using pre-merge for quality-critical applications
- Gap > 0.2 LPIPS: Must use pre-merge latents, significant information loss

**Deliverables:**
- Quantitative comparison table
- Side-by-side reconstruction visualizations
- Error heatmaps showing where information is lost
- Recommendation on which extraction point to use

**Time estimate:** 2 days

---

### E1.4: Spatial Information Test

**Objective:** Directly measure whether object positions can be recovered from latents.

**Protocol:**

1. Create test set with known object positions (bounding boxes)
2. Train a position regression head on latents
3. Measure localization accuracy

**Test set construction:**

```python
# Use COCO or similar with bounding box annotations
# Or synthetically generate images with objects at known positions

def create_position_test_set(n_samples=1000):
    """Generate images with single objects at various positions."""
    images = []
    positions = []  # [(x_center, y_center, width, height), ...]

    for _ in range(n_samples):
        # Random position
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        size = random.uniform(0.1, 0.3)

        # Generate image with object at (x, y)
        img = create_image_with_object(x, y, size)

        images.append(img)
        positions.append((x, y, size, size))

    return images, positions
```

**Position regression head:**

```python
class PositionHead(nn.Module):
    """Predict bounding box from latents."""
    def __init__(self, latent_dim=1536, n_tokens=256):
        super().__init__()
        # Global pooling + MLP
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # x, y, w, h
        )

    def forward(self, latents):
        # [B, N, D] -> [B, D]
        pooled = latents.mean(dim=1)
        return self.mlp(pooled)
```

**Metrics:**

| Metric | Definition | Target |
|--------|------------|--------|
| IoU | Intersection over Union | > 0.7 |
| Center error | Euclidean distance to true center | < 0.05 (5% of image) |
| Size error | Relative size difference | < 0.1 |

**Extended test - Multiple objects:**

1. Test with 2, 3, 5 objects
2. Measure per-object localization (with Hungarian matching)
3. Track degradation as scene complexity increases

**Deliverables:**
- Position regression accuracy plots
- Visualization of predicted vs actual bounding boxes
- Analysis of failure cases
- Recommendations on spatial precision limits

**Time estimate:** 2 days

---

### E1.5: Full Reconstruction via Video Decoder (End-to-End)

**Objective:** Test the complete pipeline with a real video decoder.

**Protocol:**

1. Train a small adapter: VLM latents -> LTX-Video conditioning space
2. Generate reconstructions through LTX-Video
3. Compare to oracle (original video re-encoded through LTX-Video VAE)

**Adapter architecture:**

```python
class LatentAdapter(nn.Module):
    """Project VLM latents to video decoder conditioning space."""
    def __init__(
        self,
        vlm_dim=1536,
        decoder_dim=4096,  # LTX-Video hidden dim (check actual)
        n_vlm_tokens=256,
        n_decoder_tokens=77,  # CLIP-like conditioning length
        hidden_dim=2048
    ):
        super().__init__()

        # Option 1: Cross-attention projection
        self.query = nn.Parameter(torch.randn(n_decoder_tokens, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.proj_in = nn.Linear(vlm_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, decoder_dim)

        # Option 2: MLP projection (simpler baseline)
        self.mlp = nn.Sequential(
            nn.Linear(vlm_dim * n_vlm_tokens, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, decoder_dim * n_decoder_tokens)
        )

    def forward(self, vlm_latents, mode="cross_attn"):
        if mode == "cross_attn":
            # [B, N_vlm, D] -> [B, N_decoder, D_decoder]
            kv = self.proj_in(vlm_latents)
            q = self.query.unsqueeze(0).expand(vlm_latents.size(0), -1, -1)
            attn_out, _ = self.cross_attn(q, kv, kv)
            return self.proj_out(attn_out)
        else:
            flat = vlm_latents.flatten(1)
            return self.mlp(flat).reshape(-1, self.n_decoder_tokens, self.decoder_dim)
```

**Training:**
- Loss: LPIPS(generated, original) + L2(generated, original)
- Optimizer: AdamW, lr=1e-4
- Batch size: 4-8 (limited by decoder memory)
- Training samples: 10K video clips
- Epochs: 50

**Oracle baseline:**
```python
def compute_oracle_quality(video, ltx_video):
    """Measure LTX-Video's own reconstruction quality."""
    # Encode through VAE
    latents = ltx_video.encode(video)
    # Decode back
    reconstructed = ltx_video.decode(latents)
    # This is the best possible quality
    return compute_lpips(video, reconstructed)
```

**Quality gap analysis:**
- Oracle LPIPS: ~0.05-0.1 (video codec quality)
- Our target: < 0.3 (good reconstruction)
- Acceptable: < 0.4 (recognizable, task-relevant details preserved)

**Deliverables:**
- Trained adapter checkpoint
- Side-by-side video comparisons (original / oracle / ours)
- Quantitative metrics table
- Per-frame LPIPS curves
- Analysis of temporal consistency

**Time estimate:** 5 days

---

### E1.6: Ablation Studies

**Objective:** Understand which factors most affect reconstruction quality.

**Ablations to run:**

| Ablation | Variations | Expected insight |
|----------|------------|------------------|
| Layer depth | Extract from layers 1, 6, 12, 18, 24 | Which layer has best info? |
| Token merging | No merge, 2x2, 4x4 | Quantify merge impact |
| Adapter capacity | 1M, 5M, 10M, 50M params | Where does quality saturate? |
| Training data size | 1K, 5K, 10K, 50K samples | Data efficiency |
| Image resolution | 224, 336, 448 input | Resolution vs quality |

**Analysis:**
- Plot quality vs each factor
- Find optimal operating point
- Identify limiting factors

**Deliverables:**
- Ablation results table
- Quality vs factor plots
- Recommendations for production configuration

**Time estimate:** 3 days

---

## 5. Success Metrics

### 5.1 Primary Metrics

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| LPIPS (reconstruction) | < 0.25 | < 0.35 | > 0.45 |
| SSIM | > 0.85 | > 0.75 | < 0.65 |
| PSNR | > 25 dB | > 22 dB | < 18 dB |
| Spatial IoU (objects) | > 0.75 | > 0.6 | < 0.5 |

### 5.2 Secondary Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| FVD (video quality) | < 200 | Compare to LTX-Video baseline |
| Temporal consistency | > 0.9 | Frame-to-frame SSIM |
| Human eval (recognizable) | > 90% | "Is this the same scene?" |
| Object count accuracy | > 80% | Can we count objects in reconstruction? |

### 5.3 Qualitative Criteria

- [ ] Reconstructed videos are recognizably the same scene
- [ ] Object boundaries are preserved (not blurred into background)
- [ ] Motion trajectories match original
- [ ] Text/fine details are at least partially legible
- [ ] Colors are approximately correct

---

## 6. Failure Criteria

### 6.1 Hard Failures (stop and pivot immediately)

1. **LPIPS > 0.5 with 50M+ adapter:** Latents fundamentally lack information
2. **Spatial IoU < 0.4:** Cannot recover object positions
3. **Pre-merge provides no improvement:** Merger is not the bottleneck
4. **Training diverges/unstable:** Architecture incompatibility

### 6.2 Soft Failures (investigate before pivoting)

1. **LPIPS 0.35-0.45:** May work for coarse reasoning, not fine detail
2. **Temporal inconsistency:** May need temporal-specific adapter
3. **High variance across scenes:** May need scene-type conditioning

---

## 7. Pivot Options

If Claim 1 fails, consider these alternatives:

### 7.1 Use Pre-ViT Features

**Approach:** Extract features before the full ViT processing, directly from patch embeddings.

**Pros:** Maximum spatial information preserved
**Cons:** Loses semantic processing, may require larger adapter

**Effort:** +3 days to test

### 7.2 Fine-tune Vision Encoder

**Approach:** Unfreeze and fine-tune Qwen2.5-VL's vision encoder for reconstruction.

**Pros:** Can learn to preserve spatial info
**Cons:** May degrade VLM capabilities, expensive training

**Effort:** +2 weeks

### 7.3 Dual Encoder

**Approach:** Use a separate image encoder (CLIP, DINOv2) alongside VLM.

**Pros:** Dedicated spatial features
**Cons:** Added complexity, memory, latency

**Effort:** +1 week

### 7.4 Modify Architecture

**Approach:** Replace 2x2 merger with spatial-preserving alternative.

**Pros:** Direct fix for identified problem
**Cons:** Requires model modification, potential instability

**Effort:** +2 weeks

### 7.5 Accept Coarse Reconstruction

**Approach:** If reconstruction is "good enough" for coarse reasoning, proceed anyway.

**Decision criteria:**
- Can we distinguish actions from reconstructions?
- Can we verify physical plausibility?
- Is the task-relevant information preserved?

**Effort:** 0 (proceed with caveats)

---

## 8. Timeline

| Phase | Days | Experiments | Deliverables |
|-------|------|-------------|--------------|
| Setup | 1 | Environment, data download | Working pipeline |
| E1.1 | 1 | Latent visualization | Latent space analysis |
| E1.2 | 2 | Linear probe | Baseline reconstruction |
| E1.3 | 2 | Pre vs post merge | Extraction point decision |
| E1.4 | 2 | Spatial accuracy | Position recovery analysis |
| E1.5 | 5 | Full pipeline | End-to-end reconstruction |
| E1.6 | 3 | Ablations | Optimization recommendations |
| Analysis | 2 | Final report | Go/no-go decision |
| **Total** | **18 days** | | |

**Parallelization opportunities:**
- E1.1 and E1.2 can run in parallel after setup
- E1.4 can run in parallel with E1.3
- E1.6 ablations can be parallelized across GPUs

**Optimistic timeline:** 12 days (with parallelization)
**Pessimistic timeline:** 25 days (with debugging, pivots)

---

## 9. Dependencies

### 9.1 Must Have Before Starting

- [ ] Access to GPU with >= 40GB VRAM
- [ ] Qwen2.5-VL-7B model downloaded
- [ ] LTX-Video model downloaded
- [ ] Test dataset prepared (Something-Something v2 subset)
- [ ] Evaluation metrics implemented (LPIPS, SSIM, etc.)
- [ ] W&B project set up for experiment tracking

### 9.2 Blocks Downstream Work

This experiment blocks:
- **C2 (Adapter Training):** Need to know which latents to use
- **C3 (Future Prediction):** Need working reconstruction first
- **C4 (Verification):** Need generated videos to compare

### 9.3 External Dependencies

- Qwen team may release model updates (monitor releases)
- LTX-Video API may change (pin diffusers version)
- Flash attention compatibility (test before production)

---

## 10. Deliverables

### 10.1 Code Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Latent extractor | `src/foresight/extraction/` | Extract pre/post-merge latents |
| Reconstruction probe | `src/foresight/probes/` | Linear reconstruction baseline |
| Adapter model | `src/foresight/adapter/` | VLM -> video decoder bridge |
| Evaluation suite | `src/foresight/eval/` | LPIPS, SSIM, spatial metrics |
| Visualization tools | `src/foresight/viz/` | Latent space plots, reconstructions |

### 10.2 Checkpoints

| Checkpoint | Size | Description |
|------------|------|-------------|
| `linear_probe_premerg.pt` | ~50MB | Linear probe on pre-merge |
| `linear_probe_postmerge.pt` | ~10MB | Linear probe on post-merge |
| `adapter_best.pt` | ~100MB | Best adapter checkpoint |

### 10.3 Reports

| Report | Format | Audience |
|--------|--------|----------|
| Technical report | Markdown | Researchers |
| Summary slides | PDF | Stakeholders |
| Data tables | CSV | Analysis |

### 10.4 Datasets

| Dataset | Location | Description |
|---------|----------|-------------|
| `test_clips_v1/` | Local/S3 | 500 curated test clips |
| `latent_cache/` | Local | Cached latent extractions |

### 10.5 Decision Document

Final go/no-go decision with:
- Summary of findings
- Quantitative results
- Recommendation (proceed / pivot / abort)
- Next steps

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| VRAM OOM | Medium | High | Use gradient checkpointing, reduce batch |
| Model loading issues | Low | Medium | Pin transformers version, test early |
| Poor reconstruction | Medium | High | Pivot options ready |
| Slow iteration | Medium | Medium | Pre-compute latent cache |
| Qwen2.5-VL API changes | Low | Medium | Pin versions |
| Dataset access issues | Low | High | Download early, check licenses |

---

## 12. Open Questions

To be resolved during experiments:

1. **Exact ViT architecture:** How many layers? Where exactly is the merger?
2. **Pre-merge extraction:** Can we hook into the model cleanly?
3. **LTX-Video conditioning:** What format does it expect? CLIP-like?
4. **Video vs image:** Does video input provide better latents than frames?
5. **Batch effects:** Does reconstruction quality depend on what else is in batch?

---

## 13. Appendix

### A. Relevant Code Snippets

**Loading Qwen2.5-VL:**
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
```

**Extracting visual features:**
```python
# Direct vision encoder access
inputs = processor.image_processor(images=img, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)
grid_thw = inputs["image_grid_thw"].to(device)

with torch.no_grad():
    visual_embeds = model.visual(pixel_values, grid_thw)
    # visual_embeds: [num_tokens, 1536]
```

### B. Related Documents

- [Core Hypothesis](../hypotheses/core-hypothesis.md)
- [Qwen2-VL Paper Summary](../papers/qwen2-vl.md)
- [LTX-Video Paper Summary](../papers/ltx-video.md)
- [Project PRD](../../PRD.md)

### C. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-01-18 | TBD | Initial draft |
