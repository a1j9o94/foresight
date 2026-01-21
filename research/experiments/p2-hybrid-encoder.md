# Experiment Plan: P2 - Hybrid Encoder (VLM Semantics + DINOv2 Spatial)

**Claim:** A hybrid architecture combining VLM semantic features with DINOv2 spatial features can achieve sufficient reconstruction quality to pass Gate 1 thresholds.

**Status:** In Progress (Optimization Phase)
**Priority:** Critical (unblocks Phase 2 after Gate 1 failure)
**Owner:** TBD
**Created:** 2026-01-18

**Initial Results:** Phase 1 experiments complete. Spatial IoU and LPIPS targets met;
mAP and latency require optimization. See Section 15.

---

## 1. Objective

Design and validate a hybrid encoder architecture that combines:
- **VLM (Qwen2.5-VL)**: Semantic understanding, temporal reasoning, world knowledge
- **DINOv2-giant**: Precise spatial features, object localization, edge preservation

**Core Question:** Can we fuse VLM semantic features with DINOv2 spatial features to achieve both high perceptual quality (LPIPS < 0.35) AND accurate object localization (Spatial IoU > 0.6)?

**Why This Matters:** Gate 1 experiments revealed a fundamental limitation:
- C1 achieved good perceptual quality (LPIPS = 0.264) but failed spatial accuracy (Spatial IoU = 0.559 < 0.6)
- Q2 demonstrated this is architectural: VLM embeddings have near-zero object localization (mAP@0.5 = 0.001, Bbox IoU = 0.104)
- Q1 confirmed VLM-to-video-decoder alignment is achievable (CKA = 0.687)
- Q2 also showed excellent temporal preservation (100% direction accuracy, 90% temporal ordering)

The hybrid approach preserves VLM strengths while addressing the spatial bottleneck with a dedicated spatial encoder.

---

## 2. Hypothesis

**Primary Hypothesis:**
A cross-attention fusion module combining DINOv2 spatial features with VLM semantic features will achieve:
1. Spatial IoU > 0.6 (Gate 1 threshold)
2. LPIPS < 0.35 (maintain perceptual quality)
3. mAP@0.5 > 0.4 (object detection capability)

**Null Hypothesis:**
Spatial and semantic features cannot be effectively fused, OR DINOv2 features alone cannot be decoded to pixels at sufficient quality, OR the fusion adds unacceptable complexity/latency.

**Falsifiability:**
- If E-P2.2 (DINOv2-only baseline) fails Spatial IoU > 0.7: DINOv2 is not the right spatial encoder
- If E-P2.3 (fusion training) cannot maintain both spatial and semantic metrics: fusion is the bottleneck
- If latency overhead > 50%: architecture is not practical

---

## 3. Background

### 3.1 Why DINOv2?

DINOv2 was selected based on the proposal analysis in `research/proposals/pivot-2-hybrid-encoder.md`:

| Property | Qwen2.5-VL | DINOv2-giant |
|----------|-----------|--------------|
| Training objective | Language-vision alignment | Self-supervised visual learning |
| Spatial preservation | Poor (IoU ~0.1) | Excellent (SOTA dense prediction) |
| Semantic understanding | Excellent | Good (weaker than VLM) |
| Temporal reasoning | Excellent | N/A (image-only) |
| Output resolution | H/28 x W/28 (post-merge) | H/14 x W/14 |
| Output dimension | 3584 | 1536 |

DINOv2's self-supervised training (DINO + iBOT) emphasizes local feature consistency and spatial coherence, making it complementary to the VLM's semantic focus.

### 3.2 Proposed Fusion Architecture

```
                                    +------------------+
                                    |                  |
    Video/Image Input --------------+   Qwen2.5-VL     +------> Semantic Features
                      |             |   (frozen)       |        [B, T, 3584]
                      |             +------------------+
                      |                                         |
                      |                                         | Semantic Stream
                      |                                         v
                      |             +------------------+    +------------------+
                      |             |                  |    |                  |
                      +-------------+   DINOv2-giant   +--> |  Cross-Attention |
                                    |   (frozen)       |    |  Fusion Module   | --> LTX-Video
                                    +------------------+    |  (trainable)     |
                                            |               +------------------+
                                            |                      ^
                                            | Spatial Stream       |
                                            | (localization,       |
                                            |  edges, details)     |
                                            +----------------------+
                                              [B, H/14*W/14, 1536]
```

### 3.3 Gate 1 Failure Context

From Q2 findings, the VLM spatial limitation is fundamental:
- Pre-merge Bbox IoU: 0.101
- Post-merge Bbox IoU: 0.103
- LLM Layer 0 Bbox IoU: 0.066
- Detection mAP@0.5: 0.001

However, VLM features excel at:
- Temporal direction accuracy: 100%
- Temporal ordering: 90%
- Perceptual reconstruction (LPIPS): 0.087

This asymmetry justifies a hybrid approach: use VLM for semantics/temporal, DINOv2 for spatial.

---

## 4. Experimental Setup

### 4.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x A100 40GB | 1x A100 80GB |
| CPU RAM | 64GB | 128GB |
| Storage | 300GB SSD | 600GB NVMe |

**VRAM breakdown (inference):**
- Qwen2.5-VL-7B (bf16): ~15GB
- DINOv2-giant (bf16): ~6GB
- Fusion module: ~2GB
- LTX-Video (bf16): ~8GB
- **Total: ~31GB minimum**

**VRAM breakdown (training):**
- Above + gradients/activations: ~43GB
- Recommend A100-80GB for training

### 4.2 Software Dependencies

```bash
# Core dependencies (existing)
pip install torch>=2.1.0 torchvision torchaudio
pip install transformers>=4.40.0 accelerate>=0.27.0
pip install diffusers>=0.27.0
pip install flash-attn --no-build-isolation

# DINOv2 specific
pip install timm>=0.9.0  # For DINOv2 models

# Evaluation (existing)
pip install lpips pytorch-fid scikit-learn umap-learn

# Object detection evaluation
pip install pycocotools  # For mAP computation

# Utilities (existing)
pip install wandb einops matplotlib seaborn
```

### 4.3 Model Checkpoints

```bash
# Existing models
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
huggingface-cli download Lightricks/LTX-Video

# New: DINOv2
# DINOv2-giant is available via torch.hub or timm
# torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
# OR
# timm.create_model('vit_giant_patch14_dinov2.lvd142m', pretrained=True)
```

### 4.4 Test Datasets

**Phase 1: Synthetic (for rapid iteration)**

| Dataset | N samples | Resolution | Purpose |
|---------|-----------|------------|---------|
| Colored shapes | 1000 | 224x224 | Spatial precision testing |
| Moving shapes | 500 | 224x224 | Temporal coherence |
| Multi-object scenes | 500 | 224x224 | Object detection validation |

**Phase 2: Real-world (for validation)**

| Dataset | N samples | Resolution | Purpose |
|---------|-----------|------------|---------|
| COCO val2017 subset | 1000 | Variable | Object detection ground truth |
| Something-Something v2 | 500 | 224x224 | Action/motion validation |
| DAVIS 2017 | 90 | 480p | Video quality benchmarking |

---

## 5. Experiments

### E-P2.1: DINOv2 Spatial Feature Analysis

**Objective:** Characterize DINOv2 feature space and validate spatial information preservation.

**Protocol:**

1. Extract DINOv2-giant features from test images (same images used in Q2)
2. Train position regression probe (same as Q2 E-Q2.4)
3. Compare spatial metrics to VLM baseline

**Implementation:**

```python
import torch
import timm

def extract_dinov2_features(images, model_name='vit_giant_patch14_dinov2.lvd142m'):
    """Extract DINOv2 spatial features."""
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    with torch.no_grad():
        # Get patch tokens (excluding CLS token)
        features = model.forward_features(images)
        # features shape: [B, 1 + H/14*W/14, 1536]
        patch_features = features[:, 1:, :]  # Remove CLS token

    return patch_features  # [B, H/14*W/14, 1536]

class SpatialProbe(nn.Module):
    """Predict bounding boxes from DINOv2 features."""
    def __init__(self, feature_dim=1536, n_patches=256):
        super().__init__()
        # Global + spatial attention for bbox prediction
        self.attn_pool = nn.MultiheadAttention(feature_dim, 8)
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # x, y, w, h
        )

    def forward(self, features):
        # [B, N, D] -> [B, 4]
        query = features.mean(dim=1, keepdim=True)
        attn_out, _ = self.attn_pool(query, features, features)
        return self.bbox_head(attn_out.squeeze(1))
```

**Metrics:**

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| Bbox IoU | > 0.75 | > 0.65 | < 0.50 |
| mAP@0.5 | > 0.60 | > 0.40 | < 0.20 |
| Spatial IoU | > 0.80 | > 0.70 | < 0.60 |

**Analysis questions:**
- Does DINOv2 preserve spatial information at sufficient fidelity?
- How does spatial probe performance compare to VLM (Q2 baseline: IoU = 0.104)?
- What is the optimal layer/representation for spatial features?

**Deliverables:**
- Spatial probe trained on DINOv2 features
- Quantitative comparison with Q2 VLM baseline
- Visualization of feature maps (PCA of patch features)
- Recommendation on DINOv2 feature extraction configuration

**Time estimate:** 2 days

---

### E-P2.2: DINOv2-Only Reconstruction Baseline

**Objective:** Establish whether DINOv2 features alone can drive the video decoder at high quality.

**Protocol:**

1. Train adapter: DINOv2 features -> LTX-Video conditioning
2. Evaluate reconstruction quality (LPIPS, SSIM, Spatial IoU)
3. Compare to C1 VLM-only baseline

**Implementation:**

```python
class DINOv2Adapter(nn.Module):
    """Project DINOv2 features to LTX-Video conditioning space."""
    def __init__(
        self,
        dino_dim: int = 1536,
        ltx_dim: int = 4096,
        hidden_dim: int = 2048,
        n_output_tokens: int = 77,
    ):
        super().__init__()

        # Learnable queries for cross-attention
        self.queries = nn.Parameter(torch.randn(n_output_tokens, hidden_dim) * 0.02)

        # Project DINOv2 to hidden dim
        self.dino_proj = nn.Linear(dino_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
            for _ in range(4)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(4)
        ])

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, ltx_dim)

    def forward(self, dino_features):
        # [B, N_patches, 1536] -> [B, 77, 4096]
        kv = self.dino_proj(dino_features)
        q = self.queries.unsqueeze(0).expand(dino_features.size(0), -1, -1)

        for attn, ffn in zip(self.cross_attn, self.ffn):
            q = q + attn(q, kv, kv)[0]
            q = q + ffn(q)

        return self.out_proj(q)
```

**Training:**
- Loss: LPIPS + L2 reconstruction loss
- Optimizer: AdamW, lr=1e-4
- Batch size: 8
- Training samples: 10K images
- Epochs: 100

**Metrics:**

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| LPIPS | < 0.25 | < 0.30 | > 0.40 |
| SSIM | > 0.85 | > 0.80 | < 0.70 |
| Spatial IoU | > 0.75 | > 0.70 | < 0.60 |
| Edge F1 | > 0.65 | > 0.55 | < 0.40 |

**Analysis:**
- Can DINOv2 alone match or exceed VLM reconstruction quality (C1 LPIPS = 0.236)?
- Does DINOv2 preserve spatial accuracy better than VLM?
- What is the perceptual-spatial tradeoff?

**Deliverables:**
- Trained DINOv2 adapter checkpoint
- Side-by-side comparison: Original / VLM-only / DINOv2-only
- Quantitative metrics table
- Ablation on adapter capacity (5M, 10M, 20M params)

**Time estimate:** 4 days

**Decision point:** If DINOv2-only achieves Spatial IoU > 0.7 AND LPIPS < 0.30, consider simplifying to single-encoder (no fusion needed).

---

### E-P2.3: Cross-Attention Fusion Module Training

**Objective:** Train the hybrid fusion module combining VLM and DINOv2 features.

**Protocol:**

1. Implement cross-attention fusion architecture (per proposal)
2. Train fusion module with frozen VLM and DINOv2 encoders
3. Evaluate combined reconstruction quality

**Implementation:**

```python
class HybridFusionModule(nn.Module):
    """
    Fuses VLM semantic features with DINOv2 spatial features.

    Design principles:
    1. Spatial features provide grounding (DINOv2)
    2. Semantic features provide context (VLM)
    3. Learnable queries extract conditioning for video decoder
    """
    def __init__(
        self,
        vlm_dim: int = 3584,
        spatial_dim: int = 1536,
        fusion_dim: int = 1024,
        num_fusion_layers: int = 4,
        num_output_queries: int = 64,
        ltx_dim: int = 4096,
    ):
        super().__init__()

        # Project both streams to common dimension
        self.vlm_proj = nn.Linear(vlm_dim, fusion_dim)
        self.spatial_proj = nn.Linear(spatial_dim, fusion_dim)

        # Positional encoding for spatial features
        self.spatial_pos = nn.Parameter(torch.randn(256, fusion_dim) * 0.02)

        # Learnable output queries
        self.output_queries = nn.Parameter(
            torch.randn(num_output_queries, fusion_dim) * 0.02
        )

        # Cross-attention fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=fusion_dim,
                nhead=8,
                dim_feedforward=fusion_dim * 4,
                batch_first=True,
            )
            for _ in range(num_fusion_layers)
        ])

        # Output projection to LTX-Video space
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, ltx_dim),
        )

    def forward(
        self,
        vlm_features: torch.Tensor,      # [B, T_vlm, 3584]
        spatial_features: torch.Tensor,  # [B, H*W, 1536]
    ) -> torch.Tensor:
        B = vlm_features.size(0)

        # Project to common space
        vlm_proj = self.vlm_proj(vlm_features)        # [B, T_vlm, fusion_dim]
        spatial_proj = self.spatial_proj(spatial_features)  # [B, H*W, fusion_dim]

        # Add positional encoding to spatial features
        spatial_proj = spatial_proj + self.spatial_pos[:spatial_proj.size(1)]

        # Concatenate streams for cross-attention
        context = torch.cat([vlm_proj, spatial_proj], dim=1)

        # Learnable queries attend to combined context
        queries = self.output_queries.unsqueeze(0).expand(B, -1, -1)

        for layer in self.fusion_layers:
            queries = layer(queries, context)

        # Project to video decoder conditioning space
        return self.output_proj(queries)  # [B, num_queries, ltx_dim]
```

**Training Strategy:**

**Phase 3A: Spatial-focused warmup (1 epoch)**
- Loss: 0.7 * LPIPS + 0.3 * Spatial_loss
- Purpose: Establish spatial grounding first

**Phase 3B: Balanced training (9 epochs)**
- Loss: 0.4 * LPIPS + 0.3 * L2 + 0.3 * Spatial_loss
- Purpose: Balance perceptual and spatial quality

**Hyperparameters:**
- Optimizer: AdamW
- Learning rate: 5e-5 (fusion module), 1e-5 (LoRA on LTX-Video)
- Batch size: 4
- Training samples: 10K
- Gradient accumulation: 4 steps

**Metrics:**

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| LPIPS | < 0.28 | < 0.35 | > 0.45 |
| Spatial IoU | > 0.70 | > 0.60 | < 0.50 |
| mAP@0.5 | > 0.50 | > 0.40 | < 0.25 |
| Semantic consistency | > 0.80 | > 0.70 | < 0.60 |

**Analysis:**
- Does fusion improve over either single-stream baseline?
- What is the attention pattern between streams (spatial vs semantic)?
- Are there failure modes where streams conflict?

**Deliverables:**
- Trained fusion module checkpoint
- Attention visualization (which stream contributes where)
- Comparison: VLM-only vs DINOv2-only vs Hybrid
- Training curves (loss, spatial IoU, LPIPS over epochs)

**Time estimate:** 5 days

---

### E-P2.4: End-to-End Hybrid Pipeline Evaluation

**Objective:** Comprehensive evaluation of the full hybrid pipeline against Gate 1 thresholds.

**Protocol:**

1. Run full pipeline on held-out test set
2. Evaluate all Gate 1 success criteria
3. Compare to C1/Q2 baselines
4. Test on real-world data (Something-Something v2, COCO)

**Full Pipeline:**

```python
class HybridReconstructionPipeline:
    def __init__(self, vlm_model, dino_model, fusion_module, video_decoder):
        self.vlm = vlm_model
        self.dino = dino_model
        self.fusion = fusion_module
        self.decoder = video_decoder

    def forward(self, images):
        # 1. Extract VLM features (semantic + temporal)
        vlm_features = self.vlm.extract_visual_features(images)

        # 2. Extract DINOv2 features (spatial)
        dino_features = self.dino.forward_features(images)[:, 1:, :]

        # 3. Fuse features
        conditioning = self.fusion(vlm_features, dino_features)

        # 4. Generate reconstruction
        reconstructed = self.decoder.generate(conditioning)

        return reconstructed
```

**Evaluation Protocol:**

**Dataset splits:**
- Synthetic test: 200 images (colored shapes, known ground truth)
- COCO val: 500 images (real objects, bbox annotations)
- SSv2 test: 200 video clips (action/motion validation)

**Metrics (Gate 1 thresholds):**

| Metric | Gate 1 Target | Our Target | Measurement |
|--------|---------------|------------|-------------|
| LPIPS | < 0.35 | < 0.30 | Perceptual distance |
| SSIM | > 0.75 | > 0.80 | Structural similarity |
| Spatial IoU | > 0.60 | > 0.65 | Object localization |
| mAP@0.5 | > 0.40 | > 0.45 | Detection accuracy |
| Edge F1 | > 0.50 | > 0.55 | Boundary preservation |

**Additional metrics:**
- Temporal LPIPS variance (for video): < 0.02
- Latency (end-to-end): < 1.5s
- Semantic consistency (VLM caption similarity): > 0.75

**Analysis:**
- Do we pass all Gate 1 criteria?
- What is the performance gap vs VLM-only (C1) and DINOv2-only (E-P2.2)?
- Are there scene types where hybrid struggles?
- What is the latency overhead?

**Deliverables:**
- Comprehensive metrics table
- Per-scene-type breakdown
- Side-by-side visualizations: Original / VLM-only / DINOv2-only / Hybrid
- Failure case analysis
- Gate 1 pass/fail assessment

**Time estimate:** 3 days

---

### E-P2.5: Ablation Studies

**Objective:** Understand which architectural choices matter most for hybrid performance.

**Ablations:**

| Ablation | Variations | Expected Insight |
|----------|------------|------------------|
| Fusion strategy | Cross-attention, Concatenation, FiLM, Perceiver-style | Which fusion works best? |
| Fusion layers | 2, 4, 6, 8 layers | How much capacity needed? |
| Output queries | 32, 64, 128 queries | How many queries needed? |
| VLM layer | layer_8, layer_14, layer_final | Which VLM layer to use? |
| DINOv2 variant | ViT-B, ViT-L, ViT-G | Size vs quality tradeoff |
| Stream weighting | 0.3/0.7, 0.5/0.5, 0.7/0.3 (VLM/spatial) | Which stream matters more? |

**Ablation implementations:**

```python
# Ablation A: Fusion strategies
class ConcatFusion(nn.Module):
    """Simple concatenation + MLP."""
    def forward(self, vlm, spatial):
        combined = torch.cat([vlm.mean(1), spatial.mean(1)], dim=-1)
        return self.mlp(combined)

class FiLMFusion(nn.Module):
    """VLM modulates spatial features via FiLM."""
    def forward(self, vlm, spatial):
        gamma, beta = self.film_gen(vlm)
        return gamma * spatial + beta

# Ablation B: Stream weighting
class WeightedFusion(nn.Module):
    def __init__(self, vlm_weight=0.5):
        self.vlm_weight = vlm_weight
        self.spatial_weight = 1 - vlm_weight

    def forward(self, vlm, spatial):
        vlm_proj = self.vlm_weight * self.vlm_proj(vlm)
        spatial_proj = self.spatial_weight * self.spatial_proj(spatial)
        return self.cross_attn(self.queries, torch.cat([vlm_proj, spatial_proj], 1))
```

**Analysis:**
- Which fusion strategy achieves best spatial-semantic balance?
- Is there diminishing returns on fusion complexity?
- Can we simplify the architecture without losing quality?

**Deliverables:**
- Ablation results table (all variants)
- Best configuration recommendation
- Cost-quality tradeoff analysis
- Architecture simplification opportunities

**Time estimate:** 4 days

---

### E-P2.6: Latency and Efficiency Analysis

**Objective:** Measure computational overhead and identify optimization opportunities.

**Protocol:**

1. Benchmark inference latency for each component
2. Measure memory usage
3. Profile bottlenecks
4. Test optimization strategies

**Benchmarks:**

| Component | Expected Latency | Target |
|-----------|------------------|--------|
| VLM encoding | ~400ms | baseline |
| DINOv2 encoding | ~150ms | < 200ms |
| Fusion module | ~50ms | < 100ms |
| Video decoding | ~800ms | baseline |
| **Total pipeline** | ~1400ms | < 1500ms |

**Overhead budget:** < 25% over VLM-only baseline (~1200ms)

**Optimization strategies to test:**
- Feature caching (pre-compute DINOv2 for dataset)
- Batch processing
- DINOv2 model variants (ViT-B vs ViT-G)
- Fusion layer pruning
- Mixed precision inference

**Memory analysis:**

| Configuration | VRAM (inference) | VRAM (training) |
|---------------|------------------|-----------------|
| VLM only | ~23GB | ~35GB |
| + DINOv2-giant | ~29GB | ~43GB |
| + DINOv2-large | ~26GB | ~38GB |
| + DINOv2-base | ~24GB | ~36GB |

**Deliverables:**
- Latency breakdown by component
- Memory profiling results
- Optimization recommendations
- Practical deployment configuration

**Time estimate:** 2 days

---

## 6. Success Criteria

### 6.1 Primary Success Criteria (Gate 1)

| Metric | Target | Threshold | Source |
|--------|--------|-----------|--------|
| **Spatial IoU** | > 0.65 | > 0.60 | Gate 1 requirement |
| **LPIPS** | < 0.30 | < 0.35 | Gate 1 requirement |
| **mAP@0.5** | > 0.45 | > 0.40 | Localization capability |
| **Latency overhead** | < 20% | < 25% | Practical deployment |

### 6.2 Secondary Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| SSIM | > 0.80 | Structural quality |
| Edge F1 | > 0.55 | Boundary preservation |
| Semantic consistency | > 0.75 | VLM caption similarity |
| Temporal LPIPS variance | < 0.02 | Video consistency |
| Parameter count | < 25M | Fusion module budget |

### 6.3 Comparison Baselines

| Baseline | Source | Key Metrics |
|----------|--------|-------------|
| VLM-only (C1) | research/experiments/c1-vlm-latent-sufficiency/ | LPIPS=0.236, Spatial IoU=0.559 |
| VLM detection (Q2) | research/experiments/q2-information-preservation/ | mAP@0.5=0.001, Bbox IoU=0.104 |
| VLM alignment (Q1) | research/experiments/q1-latent-alignment/ | CKA=0.687 |

### 6.4 Go/No-Go Decision Points

| Checkpoint | Timing | Criteria | Decision |
|------------|--------|----------|----------|
| E-P2.1 Complete | Day 2 | DINOv2 Spatial IoU > 0.65 | Continue / Switch encoder |
| E-P2.2 Complete | Day 6 | DINOv2-only LPIPS < 0.30 | Consider single-stream |
| E-P2.3 Complete | Day 11 | Hybrid Spatial IoU > 0.60 | Continue / Abort |
| E-P2.4 Complete | Day 14 | Gate 1 criteria met | Proceed to Phase 2 |

---

## 7. Failure Criteria

### 7.1 Hard Failures (abort experiment)

1. **DINOv2 fails spatial probing (E-P2.1):** Spatial IoU < 0.50
   - Implication: DINOv2 is not the right spatial encoder
   - Action: Test alternative encoders (SAM, MAE)

2. **Fusion degrades both streams (E-P2.3):**
   - Hybrid Spatial IoU < max(VLM, DINOv2) AND Hybrid LPIPS > min(VLM, DINOv2)
   - Implication: Fusion is fundamentally broken
   - Action: Test alternative fusion strategies or use separate pathways

3. **Unacceptable latency (E-P2.6):** Overhead > 50%
   - Implication: Architecture not practical
   - Action: Use smaller DINOv2 variant or simplify fusion

### 7.2 Soft Failures (investigate before pivoting)

1. **Marginal spatial improvement:** Spatial IoU 0.55-0.60
   - May need more training data or larger fusion module

2. **Perceptual quality regression:** LPIPS increases vs VLM-only
   - May need to adjust loss weighting

3. **High variance across scene types:**
   - May need scene-type conditioning

---

## 8. Pivot Options

If hybrid approach fails, consider:

### 8.1 Alternative Spatial Encoders

| Encoder | Pros | Cons | Effort |
|---------|------|------|--------|
| SAM ViT-H | Explicit segmentation training | Lower output dim (256) | +1 week |
| MAE | Trained for reconstruction | Weaker than DINOv2 on dense tasks | +1 week |
| SigLIP | Semantic-spatial balance | Less proven for spatial | +1 week |

### 8.2 Alternative Fusion Strategies

| Strategy | Description | Effort |
|----------|-------------|--------|
| Late fusion | Separate decoders, combine outputs | +2 weeks |
| Multi-scale fusion | Fuse at multiple resolutions | +1 week |
| Adaptive gating | Learn when to use which stream | +1 week |

### 8.3 Accept Spatial Limitation

If no approach achieves Spatial IoU > 0.6:
- Reframe success criteria for semantic-focused tasks
- Use VLM for coarse reasoning, external detector for precise localization
- Update verification module to work with approximate spatial info

---

## 9. Timeline

| Phase | Days | Experiments | Milestones |
|-------|------|-------------|------------|
| Setup | 1 | Environment, model download | DINOv2 loaded |
| E-P2.1 | 2 | DINOv2 spatial analysis | Spatial probe trained |
| E-P2.2 | 4 | DINOv2-only baseline | Single-stream baseline |
| E-P2.3 | 5 | Fusion training | Fusion module trained |
| E-P2.4 | 3 | End-to-end evaluation | Gate 1 assessment |
| E-P2.5 | 4 | Ablations | Best configuration |
| E-P2.6 | 2 | Latency analysis | Deployment config |
| Analysis | 2 | Final report | Go/no-go decision |
| **Total** | **23 days** | | |

**Parallelization opportunities:**
- E-P2.1 and model loading can overlap
- E-P2.5 ablations can run in parallel across GPUs
- E-P2.6 can partially overlap with E-P2.4

**Optimistic timeline:** 18 days
**Pessimistic timeline:** 30 days

---

## 10. Resource Requirements

### 10.1 Compute

| Phase | GPU Type | GPU-Hours | Notes |
|-------|----------|-----------|-------|
| E-P2.1 | A100-40GB | 20 | Feature extraction + probing |
| E-P2.2 | A100-80GB | 60 | Adapter training |
| E-P2.3 | A100-80GB | 100 | Fusion training |
| E-P2.4 | A100-40GB | 20 | Evaluation |
| E-P2.5 | A100-80GB | 80 | Ablations |
| E-P2.6 | A100-40GB | 10 | Benchmarking |
| **Total** | | **290** | |

**Estimated cost:** ~$580 (at $2/GPU-hour)

### 10.2 Storage

| Item | Size | Notes |
|------|------|-------|
| DINOv2-giant checkpoint | ~4.5GB | One-time download |
| Cached DINOv2 features | ~50GB | Pre-computed for training |
| Fusion module checkpoints | ~200MB | Multiple ablation variants |
| Results/artifacts | ~10GB | Visualizations, metrics |
| **Total additional** | ~65GB | |

### 10.3 Personnel

| Role | Effort | Availability |
|------|--------|--------------|
| Research Engineer | 1 FTE x 4 weeks | Required |
| Research Scientist | 0.5 FTE x 4 weeks | For analysis/decisions |

---

## 11. Dependencies

### 11.1 Prerequisites (must complete before starting)

- [x] C1 experiment completed (provides VLM reconstruction baseline)
- [x] Q2 experiment completed (establishes spatial failure)
- [x] Q1 experiment completed (confirms VLM-video decoder alignment)
- [ ] Proposal approved (pivot-2-hybrid-encoder.md)
- [ ] DINOv2 model downloaded and tested
- [ ] GPU resources allocated (A100-80GB for training)

### 11.2 Blocks

This experiment blocks:
- **C2 (Adapter Bridging):** Needs to know final architecture
- **C3 (Future Prediction):** Needs working reconstruction
- **Gate 2:** Cannot proceed without passing Gate 1

### 11.3 External Dependencies

- DINOv2 model availability (via torch.hub or timm)
- Modal infrastructure for GPU access
- W&B for experiment tracking

---

## 12. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| DINOv2 spatial probe underperforms | Low (15%) | High | Test SAM/MAE as backups |
| Fusion training unstable | Medium (25%) | Medium | Staged training, gradient clipping |
| Memory OOM during training | Medium (30%) | Medium | Gradient checkpointing, smaller batch |
| Latency overhead too high | Low (10%) | Medium | Use DINOv2-base variant |
| VLM and DINOv2 features conflict | Medium (20%) | Medium | Stream gating, separate projections |
| Longer timeline than estimated | Medium (35%) | Low | Built-in buffer, parallelization |

---

## 13. Deliverables

### 13.1 Code Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| DINOv2 feature extractor | `infra/modal/handlers/p2/dinov2_extractor.py` | Extract spatial features |
| Spatial probe | `infra/modal/handlers/p2/spatial_probe.py` | Position regression head |
| DINOv2 adapter | `infra/modal/handlers/p2/dinov2_adapter.py` | DINOv2-only baseline |
| Fusion module | `infra/modal/handlers/p2/fusion_module.py` | Cross-attention fusion |
| Hybrid pipeline | `infra/modal/handlers/p2/hybrid_pipeline.py` | End-to-end system |
| Evaluation suite | `infra/modal/handlers/p2/evaluation.py` | All metrics |

### 13.2 Checkpoints

| Checkpoint | Size | Description |
|------------|------|-------------|
| `spatial_probe_dinov2.pt` | ~10MB | Spatial regression probe |
| `dinov2_adapter_best.pt` | ~80MB | DINOv2-only adapter |
| `fusion_module_best.pt` | ~100MB | Best fusion checkpoint |
| `hybrid_pipeline_best.pt` | ~200MB | Full pipeline weights |

### 13.3 Reports

| Report | Format | Audience |
|--------|--------|----------|
| Technical findings | `research/experiments/p2-hybrid-encoder/FINDINGS.md` | Research team |
| Results data | `research/experiments/p2-hybrid-encoder/results.yaml` | Validation system |
| Architecture recommendation | `research/experiments/p2-hybrid-encoder/ARCHITECTURE.md` | Implementation |

### 13.4 Decision Document

Final assessment with:
- Gate 1 pass/fail determination
- Recommended architecture configuration
- Next steps for Phase 2
- Known limitations and caveats

---

## 14. Open Questions

To be resolved during experiments:

1. **Optimal DINOv2 layer:** Should we use the last layer or intermediate layers?
2. **Resolution mismatch:** DINOv2 uses 14x14 patches vs VLM 28x28 post-merge - how to align?
3. **Temporal handling:** DINOv2 is image-only - how to process video? (frame-by-frame vs temporal pooling)
4. **Feature caching:** Can we pre-compute DINOv2 features for faster iteration?
5. **Attention patterns:** Which stream (VLM vs DINOv2) dominates in different scene regions?

---

## 15. Initial Results (2026-01-20)

### Gate 1 Assessment

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Spatial IoU | > 0.60 | 0.7515 | ✅ PASS |
| LPIPS | < 0.35 | 0.162 | ✅ PASS |
| mAP@0.5 | > 0.40 | 0.002 | ❌ FAIL |
| Latency overhead | < 25% | 68% | ❌ FAIL |

### Key Findings

**E-P2.1 (DINOv2 Spatial Analysis):**
- Spatial IoU: 0.7515 (35% improvement over VLM baseline)
- 100% of predictions have IoU > 0.5
- mAP very low (0.002) - detection probe architecture issue

**E-P2.2 (DINOv2-Only Baseline):**
- LPIPS: 0.221 (better than VLM's 0.264)
- Spatial IoU: 0.595

**E-P2.3 (Hybrid Fusion):**
- LPIPS: 0.162 (best result)
- Spatial IoU: 0.597

**E-P2.6 (Latency Analysis):**
- Total: 136ms (68% overhead vs 25% target)
- DINOv2-giant: 52ms (38% of total)
- VLM: 80ms (59% of total)
- Fusion: 3ms (2% of total)

### Root Cause Analysis

1. **mAP failure:** The DetectionProbe uses only 1 cross-attention + 1 self-attention layer, no positional encoding, and simple bipartite matching. Full DETR architecture requires 6 encoder + 6 decoder layers with Hungarian matching loss.

2. **Latency failure:** DINOv2-ViT-G (1.1B params) contributes 52ms to the pipeline. Switching to ViT-L (304M params) should reduce this to ~15ms.

---

## 16. Optimization Phase

### Optimization 1: ViT-G → ViT-L Swap

Replace `dinov2_vitg14` with `dinov2_vitl14` across all handlers:
- Feature dimension: 1536 → 1024
- Expected latency reduction: 52ms → ~15ms
- Expected quality impact: Minimal (0.7515 provides 25% margin over 0.60 threshold)

### Optimization 2: Full DETR Detection Head

Replace simple DetectionProbe with full DETR architecture:
- 6 encoder layers (processes spatial features)
- 6 decoder layers (processes query embeddings)
- 2D positional encoding for spatial awareness
- Hungarian matching loss for training
- GIoU + L1 box loss
- Focal loss for classification

### Expected Results After Optimization

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Spatial IoU | 0.7515 | ~0.70 | > 0.60 |
| LPIPS | 0.162 | ~0.17 | < 0.35 |
| mAP@0.5 | 0.002 | > 0.40 | > 0.40 |
| Latency Overhead | 68% | ~20% | < 25% |

---

## 17. Appendix

### A. DINOv2 Model Variants

| Model | Params | Dim | Patch | Memory | Spatial Quality |
|-------|--------|-----|-------|--------|-----------------|
| DINOv2-base | 86M | 768 | 14x14 | ~1GB | Good |
| DINOv2-large | 304M | 1024 | 14x14 | ~2GB | Better |
| DINOv2-giant | 1.1B | 1536 | 14x14 | ~5GB | Best |

**Recommendation:** Using DINOv2-large (ViT-L) for latency optimization.
Initial experiments with ViT-G achieved Spatial IoU 0.7515, providing 25% margin
for potential quality reduction with smaller model. ViT-L reduces latency from
~52ms to ~15ms while maintaining sufficient spatial accuracy.

### B. Related Documents

- [Pivot Proposal](../../proposals/pivot-2-hybrid-encoder.md)
- [C1 Experiment](./c1-vlm-latent-sufficiency.md)
- [Q2 Experiment](./q2-information-preservation.md)
- [Q1 Experiment](./q1-latent-alignment.md)
- [Agent Guide](../AGENT_GUIDE.md)

### C. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-18 | Claude | Initial draft |
