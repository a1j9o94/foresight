# Claim 2: Small Adapter Can Bridge Latent Spaces

**Experiment ID:** C2
**Status:** Planned
**Priority:** High (Critical Path)
**Depends On:** C1 (partial - needs latent extraction validated)
**Last Updated:** 2025-01-18

---

## 1. Objective

**Primary Goal:** Determine whether a small adapter network (~5-10M parameters) can effectively translate Qwen2.5-VL's 1536-dimensional visual latents to LTX-Video's 128-channel latent conditioning space, achieving reconstruction quality sufficient for downstream prediction tasks.

**Specific Questions:**
1. Is the mapping between VLM and video decoder latent spaces learnable with limited capacity?
2. What is the minimum adapter complexity required for acceptable reconstruction?
3. Does adapter performance scale predictably with parameter count, or is there a capacity cliff?
4. Which architectural pattern (linear, MLP, cross-attention) best suits this bridging task?

**Success Definition:** A 10M parameter adapter achieves >80% of the reconstruction quality (measured by LPIPS) of a 100M parameter adapter, demonstrating that a compact transform can bridge these spaces without memorization.

---

## 2. Background

### 2.1 Qwen2.5-VL Latent Space

| Property | Value | Notes |
|----------|-------|-------|
| Embedding dimension | 1536 | Per visual token after ViT |
| Patch size | 14x14 pixels | Before 2x2 merge |
| Token merge ratio | 4:1 | 2x2 spatial patches -> 1 token |
| Position encoding | M-RoPE | Decomposed temporal/height/width |
| Tokens per frame (448x672) | ~768 | After merge, varies by resolution |

**Key Characteristics:**
- Trained for language-visual alignment (next token prediction)
- Semantic compression prioritized over spatial fidelity
- Window attention in later layers (Qwen2.5-VL) may limit global context
- Rich semantic content but potentially lossy spatial information

### 2.2 LTX-Video Latent Space

| Property | Value | Notes |
|----------|-------|-------|
| Latent channels | 128 | Unusually high vs. typical 16 |
| Spatial compression | 32x | Per dimension |
| Temporal compression | 8x | Per dimension |
| Total compression | 1:192 | 8192 pixels per token |
| Expected input noise | t=0.05 | Denoising decoder expects near-clean latents |

**Key Characteristics:**
- Trained for reconstruction (VAE objective + GAN)
- High channel count compensates for aggressive spatial compression
- Tightly coupled with denoising decoder for high-frequency recovery
- Strong spatial fidelity, learned through reconstruction objective

### 2.3 The Bridging Challenge

The core difficulty: these latent spaces were trained with fundamentally different objectives.

| Aspect | Qwen2.5-VL | LTX-Video |
|--------|------------|-----------|
| Training objective | Language modeling | Visual reconstruction |
| Optimization pressure | Semantic fidelity | Pixel fidelity |
| Spatial structure | Token sequence | 3D spatial grid |
| Information bottleneck | Language alignment | Compression ratio |

**Hypothesis:** Despite different objectives, both spaces encode similar visual semantics (objects, positions, relationships). A learned transform should find correspondences without requiring the adapter to "memorize" mappings.

**Counter-hypothesis (failure mode):** The representations are organized so differently that bridging requires the adapter to essentially learn a decoder-encoder pair, requiring >100M parameters.

### 2.4 Prior Work on Latent Alignment

- **CLIP-to-Diffusion adapters** (IP-Adapter): Successfully bridge CLIP image embeddings to Stable Diffusion. Uses ~30M parameter cross-attention adapter.
- **LLaVA visual projector**: Simple 2-layer MLP (4M params) projects CLIP ViT to LLM space.
- **IC-LoRA (LTX)**: Demonstrates conditioning LTX with external signals via LoRA adapters.

These suggest 5-50M parameter adapters are viable for similar bridging tasks.

---

## 3. Experimental Setup

### 3.1 Hardware Requirements

| Component | VRAM | Notes |
|-----------|------|-------|
| Qwen2.5-VL-7B (frozen, bf16) | ~15GB | Encoder only, no generation |
| LTX-Video VAE (frozen, bf16) | ~2GB | Encoder + decoder |
| LTX-Video Transformer (frozen, bf16) | ~4GB | For diffusion steps |
| Adapter (training) | ~1-2GB | Varies by architecture |
| Batch data + activations | ~8-12GB | Gradient checkpointing reduces this |
| **Total (single GPU)** | **~30-35GB** | A100-40GB or RTX 4090 |

**Recommended Setup:**
- **Development:** Single RTX 4090 (24GB) with gradient checkpointing and reduced batch size
- **Full training:** Single A100-40GB or A100-80GB for larger batch sizes
- **Scaling experiments:** 2-4x A100-40GB for 50M+ parameter adapters

### 3.2 Software Dependencies

```bash
# Core dependencies
pip install torch>=2.1.0 torchvision torchaudio
pip install transformers>=4.37.0 diffusers>=0.25.0 accelerate>=0.25.0
pip install flash-attn --no-build-isolation  # Required for Qwen2.5-VL efficiency

# Evaluation
pip install lpips pytorch-fid  # Perceptual metrics
pip install scikit-learn umap-learn  # Latent space visualization

# Training infrastructure
pip install wandb  # Experiment tracking
pip install bitsandbytes  # Optional: 8-bit optimizers for memory savings

# Video processing
pip install decord opencv-python  # Video loading
pip install qwen-vl-utils[decord]==0.0.8  # Qwen video utils
```

### 3.3 Training Data Requirements

**Primary Dataset:** Something-Something v2 (subset)
- 220,847 videos of hand-object interactions
- Clear object visibility, simple backgrounds
- Good for measuring spatial reconstruction accuracy

**Minimum Viable Dataset:**
- **E2.1-E2.3:** 10,000 video clips (quick iteration)
- **E2.4 (scaling study):** 50,000 video clips (robust conclusions)

**Data Pipeline:**
```
Video -> Sample 8 frames (1 fps) -> Qwen2.5-VL encode -> Extract latents
                                 -> LTX-VAE encode -> Target latents

Training pairs: (VLM_latents, LTX_latents) for each video
```

**Storage:** ~50GB for preprocessed latent pairs (50K videos)

### 3.4 Common Training Configuration

```yaml
# Base configuration for all experiments
training:
  optimizer: AdamW
  weight_decay: 0.01
  warmup_steps: 1000
  max_steps: 50000
  gradient_checkpointing: true
  mixed_precision: bf16

data:
  batch_size: 8  # Per GPU, adjust based on VRAM
  num_workers: 4
  frames_per_clip: 8
  resolution: [512, 768]  # H x W

evaluation:
  eval_every: 1000
  metrics: [lpips, mse, cosine_sim]
  num_eval_samples: 500
```

---

## 4. Experiments

### E2.1: Linear Probe (Baseline)

**Objective:** Establish baseline - can a linear transform bridge the spaces at all?

**Architecture:**
```
VLM Latents [B, T, 1536] -> Linear(1536, 128*H*W) -> Reshape -> LTX Latents [B, 128, T/8, H/32, W/32]
```

**Parameters:** ~200K (just the linear layer weight matrix)

**Training Details:**
- Loss: MSE(adapter_output, target_LTX_latents)
- Learning rate: 1e-4
- Steps: 10,000
- Expected time: 2-4 hours on single GPU

**Evaluation:**
1. Latent space MSE (direct measure of alignment)
2. Decoded video LPIPS (perceptual quality after LTX decoding)
3. t-SNE/UMAP visualization of projected vs. target latents

**Success Criteria:**
- LPIPS < 0.5: Linear mapping captures coarse structure
- Clustering: Projected latents cluster by video content, not randomly

**Expected Outcome:** Poor reconstruction (LPIPS > 0.4) but demonstrates some structure preservation. This establishes the difficulty floor.

---

### E2.2: MLP Adapter (2-3 Layers)

**Objective:** Determine if non-linear transform significantly improves bridging.

**Architecture Options:**

**E2.2a: Shallow MLP (2 layers)**
```python
class ShallowMLPAdapter(nn.Module):
    def __init__(self, vlm_dim=1536, ltx_channels=128, hidden_dim=2048):
        self.proj = nn.Sequential(
            nn.Linear(vlm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ltx_channels)
        )
    # ~6M params with hidden_dim=2048
```

**E2.2b: Deeper MLP (3 layers)**
```python
class DeepMLPAdapter(nn.Module):
    def __init__(self, vlm_dim=1536, ltx_channels=128, hidden_dims=[2048, 1024]):
        self.proj = nn.Sequential(
            nn.Linear(vlm_dim, hidden_dims[0]),
            nn.GELU(),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.LayerNorm(hidden_dims[1]),
            nn.Linear(hidden_dims[1], ltx_channels)
        )
    # ~9M params
```

**E2.2c: Residual MLP**
```python
class ResidualMLPAdapter(nn.Module):
    def __init__(self, vlm_dim=1536, ltx_channels=128, hidden_dim=1536):
        self.input_proj = nn.Linear(vlm_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(4)
        ])
        self.output_proj = nn.Linear(hidden_dim, ltx_channels)
    # ~12M params
```

**Training Details:**
- Loss: MSE + 0.1 * LPIPS (perceptual regularization)
- Learning rate: 5e-5
- Steps: 20,000
- Expected time: 4-8 hours per variant

**Ablations:**
- With vs. without LayerNorm
- GELU vs. SiLU activation
- Dropout (0.0, 0.1) for regularization

**Success Criteria:**
- LPIPS < 0.35: Significant improvement over linear
- E2.2b or E2.2c within 10% of E2.2a despite 50% more params: Diminishing returns suggests learnable transform, not memorization

---

### E2.3: Cross-Attention Adapter

**Objective:** Test if attention-based bridging better preserves spatial relationships.

**Rationale:** VLM latents are a token sequence; cross-attention may better align these with the spatial structure LTX expects.

**Architecture:**
```python
class CrossAttentionAdapter(nn.Module):
    def __init__(self, vlm_dim=1536, ltx_channels=128, num_heads=8, num_layers=4):
        # Learnable spatial queries for LTX latent positions
        self.spatial_queries = nn.Parameter(torch.randn(1, 128, ltx_channels))

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=ltx_channels,
                nhead=num_heads,
                dim_feedforward=ltx_channels * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.vlm_proj = nn.Linear(vlm_dim, ltx_channels)

    def forward(self, vlm_latents):
        # vlm_latents: [B, T*H*W, 1536]
        kv = self.vlm_proj(vlm_latents)  # [B, N, 128]
        queries = self.spatial_queries.expand(B, -1, -1)  # [B, 128, 128]

        for layer in self.layers:
            queries = layer(queries, kv)

        return queries.reshape(B, 128, T//8, H//32, W//32)

    # ~5M params with 4 layers, 8 heads
```

**Training Details:**
- Loss: MSE + 0.1 * LPIPS + 0.05 * cosine_similarity_loss
- Learning rate: 1e-4 (transformers often need higher LR)
- Steps: 30,000
- Expected time: 8-12 hours

**Variants:**
- **E2.3a:** 2 layers (~2.5M params)
- **E2.3b:** 4 layers (~5M params)
- **E2.3c:** 6 layers with LoRA-style low-rank attention (~4M params)

**Analysis:**
- Attention pattern visualization: Do queries attend to spatially corresponding VLM tokens?
- Compare reconstruction quality for different spatial positions (center vs. edges)

**Success Criteria:**
- LPIPS < 0.30: Cross-attention provides meaningful improvement
- Attention patterns show spatial correspondence (not uniform or random)

---

### E2.4: Parameter Scaling Study

**Objective:** Characterize the adapter size vs. quality tradeoff to determine minimum viable size.

**Experimental Matrix:**

| Adapter Type | 1M | 5M | 10M | 50M | 100M |
|--------------|-----|-----|-----|-----|------|
| MLP | E2.4-M1 | E2.4-M5 | E2.4-M10 | E2.4-M50 | E2.4-M100 |
| CrossAttn | E2.4-C1 | E2.4-C5 | E2.4-C10 | - | - |

**Architecture Scaling:**

For MLP:
```python
# Scale by hidden dimension and depth
1M:   [1536, 512, 128] -> 2 layers
5M:   [1536, 2048, 128] -> 2 layers
10M:  [1536, 2048, 1024, 128] -> 3 layers
50M:  [1536, 4096, 4096, 2048, 128] -> 4 layers
100M: [1536, 8192, 8192, 4096, 2048, 128] -> 5 layers
```

For Cross-Attention:
```python
# Scale by number of layers and heads
1M:  2 layers, 4 heads
5M:  4 layers, 8 heads
10M: 6 layers, 16 heads
```

**Training Details:**
- Consistent training budget: 50,000 steps for all variants
- Same data, same evaluation protocol
- Learning rate sweep for each scale: [1e-5, 5e-5, 1e-4, 5e-4]

**Expected Time:**
- 1M-10M: 6-12 hours each
- 50M-100M: 24-48 hours each
- Total: ~1 week for full matrix

**Analysis:**
1. Plot LPIPS vs. parameter count (log scale x-axis)
2. Compute "quality per parameter" metric
3. Identify knee point in scaling curve
4. Compare MLP vs. CrossAttn at each scale

**Success Criteria:**
- 10M adapter achieves >80% quality of 100M adapter
- Clear diminishing returns visible in scaling curve
- No evidence of memorization (test set performance matches train)

---

## 5. Success Metrics

### 5.1 Primary Metrics

| Metric | Target (Success) | Acceptable | Failure |
|--------|------------------|------------|---------|
| LPIPS (10M adapter) | < 0.25 | < 0.35 | > 0.45 |
| LPIPS 10M/100M ratio | > 0.80 | > 0.70 | < 0.60 |
| Training convergence | Stable, decreasing | Noisy but improving | Divergent or flat |
| Test/train gap | < 10% | < 20% | > 30% (memorization) |

### 5.2 Secondary Metrics

| Metric | Purpose |
|--------|---------|
| MSE (latent space) | Direct alignment measure |
| Cosine similarity | Semantic alignment |
| FVD | Video quality (temporal coherence) |
| PSNR | Pixel-level reconstruction |
| Human eval | Subjective quality assessment |

### 5.3 Training Convergence Criteria

**Healthy Training:**
- Loss decreases monotonically (smoothed over 100 steps)
- Gradient norm stable (not exploding or vanishing)
- Validation loss tracks training loss

**Early Stopping:**
- Validation loss increases for 5 consecutive evaluations
- Or maximum steps reached

**Convergence Definition:**
- Training loss change < 1% over 5000 steps
- Validation LPIPS stabilized

---

## 6. Failure Criteria

The adapter bridging approach should be considered **failed** if:

### 6.1 Absolute Failures

1. **No convergence:** Training loss does not decrease below initial value after 10,000 steps
2. **Quality floor:** Best adapter (any size) achieves LPIPS > 0.45 (worse than blurry baseline)
3. **Memorization:** Test LPIPS > 1.3x train LPIPS consistently across scales

### 6.2 Practical Failures

1. **Scale requirement:** Achieving LPIPS < 0.35 requires >50M parameters
2. **Training instability:** Cannot achieve stable training without extensive hyperparameter search
3. **Computational cost:** Training time >1 week for 10M adapter on A100

### 6.3 Diagnostic Signals of Failure

| Signal | Interpretation |
|--------|----------------|
| Linear scaling curve (no knee) | Spaces require brute-force alignment, not learnable transform |
| Attention patterns uniform/random | No spatial correspondence found |
| Reconstruction collapses to mean | Adapter ignores VLM input |
| Fine details lost, coarse OK | Adapter learns compression, not alignment |

---

## 7. Pivot Options

If small adapter approach fails, consider these alternatives:

### 7.1 Extract Earlier VLM Representations

**Rationale:** Pre-merge patch embeddings may contain more spatial information.

**Implementation:**
```python
# Instead of post-merge tokens, extract raw ViT patches
vision_outputs = model.visual.get_intermediate_layers(pixel_values, n=[6, 12, 18])
early_latents = vision_outputs[0]  # Earlier layers, more spatial
```

**Pros:** Preserves spatial resolution (4x more tokens)
**Cons:** 4x computational cost, may lose semantic richness

### 7.2 Use Intermediate LTX Representations

**Rationale:** Target an intermediate space in LTX that's closer to semantic content.

**Implementation:**
- Align VLM latents to LTX DiT intermediate layers (e.g., layer 14/28)
- Let LTX diffusion process handle the final alignment

**Pros:** May find closer latent space alignment
**Cons:** Requires understanding LTX internal representations

### 7.3 Hybrid Adapter with Auxiliary Objectives

**Rationale:** Add supervision signals that guide alignment.

**Implementation:**
```python
loss = mse_loss + 0.1 * lpips_loss + 0.1 * clip_similarity_loss + 0.05 * object_detection_loss
```

**Pros:** Multiple signals may find better alignment
**Cons:** More complex training, potential loss balancing issues

### 7.4 Fine-tune LTX Decoder Jointly

**Rationale:** If adapter alone can't bridge, allow decoder to meet halfway.

**Implementation:**
- LoRA on LTX transformer (4-8M additional params)
- Joint training: adapter + decoder LoRA

**Pros:** More capacity to find alignment
**Cons:** Loses modularity, more expensive training

### 7.5 Alternative Video Decoder

**Rationale:** LTX's 128-channel space may be unusually hard to target.

**Candidates:**
- **CogVideoX:** 16 channels, more standard architecture
- **Open-Sora:** May have closer alignment to VLM-style representations

**Pros:** May find easier alignment
**Cons:** Loses LTX speed advantage (critical for real-time prediction)

---

## 8. Timeline

### Phase 1: Setup and Baselines (Days 1-3)
| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Environment setup, model loading validation | Working inference pipeline |
| 1 | Data preprocessing pipeline | Preprocessed latent pairs (10K videos) |
| 2 | Implement latent extraction from Qwen2.5-VL | Verified VLM latent quality |
| 2 | Implement LTX latent extraction | Verified target latent quality |
| 3 | E2.1: Train and evaluate linear probe | Baseline results, initial analysis |

### Phase 2: Architecture Exploration (Days 4-8)
| Day | Task | Deliverable |
|-----|------|-------------|
| 4 | E2.2a: Shallow MLP training | MLP baseline |
| 5 | E2.2b, E2.2c: Deeper MLP variants | MLP comparison |
| 6-7 | E2.3: Cross-attention adapters (2-6 layers) | Attention analysis |
| 8 | Analysis and comparison | Best architecture selected |

### Phase 3: Scaling Study (Days 9-16)
| Day | Task | Deliverable |
|-----|------|-------------|
| 9-10 | E2.4-M1, M5, M10 (small MLPs) | Small-scale results |
| 11-13 | E2.4-M50, M100 (large MLPs) | Full MLP scaling curve |
| 14-15 | E2.4-C1, C5, C10 (cross-attention scaling) | Cross-attention scaling |
| 16 | Final analysis, curve fitting | Scaling report |

### Phase 4: Documentation and Handoff (Days 17-18)
| Day | Task | Deliverable |
|-----|------|-------------|
| 17 | Write results document | Experimental report |
| 18 | Prepare trained adapters, reproduce key results | Artifacts + code |

**Total Estimated Time:** 18 days (~3 weeks)

**Critical Path Risks:**
- Model loading issues: +2 days
- Training instability requiring hyperparameter search: +3 days
- Inconclusive results requiring additional experiments: +5 days

---

## 9. Dependencies

### 9.1 Dependencies on Other Claims

| Claim | Dependency Type | Notes |
|-------|-----------------|-------|
| C1 (VLM contains spatial info) | Soft | Can proceed in parallel; if C1 fails, C2 results may be moot |
| C3 (VLM predicts futures) | None | C2 tests reconstruction, not prediction |
| C4 (Pixel verification) | None | Independent |

**Recommendation:** Start C2 experiments immediately. Results from C1 (especially pre-merge vs. post-merge comparison) may inform which VLM representations to use in C2.

### 9.2 Infrastructure Dependencies

| Dependency | Status | Blocker? |
|------------|--------|----------|
| GPU access (A100-40GB) | Required | Yes |
| Something-Something v2 access | Required | Yes (license required) |
| Qwen2.5-VL-7B weights | Available | No |
| LTX-Video weights | Available | No |
| W&B project setup | Recommended | No |

### 9.3 Code Dependencies

```python
# Required modules to implement
from foresight.models.vlm import QwenLatentExtractor
from foresight.models.video import LTXLatentExtractor
from foresight.adapters import LinearAdapter, MLPAdapter, CrossAttentionAdapter
from foresight.training import AdapterTrainer
from foresight.evaluation import ReconstructionMetrics
```

---

## 10. Deliverables

### 10.1 Trained Artifacts

| Artifact | Description | Location |
|----------|-------------|----------|
| `adapter_linear_v1.pt` | Linear baseline | `checkpoints/c2/linear/` |
| `adapter_mlp_10m_v1.pt` | Best MLP (10M) | `checkpoints/c2/mlp/` |
| `adapter_crossattn_5m_v1.pt` | Best cross-attention | `checkpoints/c2/crossattn/` |
| `adapter_mlp_100m_v1.pt` | Large MLP reference | `checkpoints/c2/mlp/` |

### 10.2 Benchmark Results

| Benchmark | Format | Location |
|-----------|--------|----------|
| Reconstruction quality (all adapters) | CSV + plots | `results/c2/reconstruction.csv` |
| Scaling curves | Plots (PNG, PDF) | `results/c2/scaling/` |
| Attention visualizations | Images | `results/c2/attention/` |
| Latent space t-SNE | Interactive HTML | `results/c2/tsne/` |

### 10.3 Documentation

| Document | Content | Location |
|----------|---------|----------|
| Experimental log | Daily notes, hyperparams, observations | `research/experiments/c2-log.md` |
| Results summary | Key findings, go/no-go decision | `research/experiments/c2-results.md` |
| Technical report | Full analysis for writeup | `research/reports/c2-adapter-bridging.md` |

### 10.4 Code

| Module | Purpose | Location |
|--------|---------|----------|
| Latent extraction | VLM/LTX latent pipelines | `src/foresight/extraction/` |
| Adapter architectures | All tested architectures | `src/foresight/adapters/` |
| Training scripts | Reproducible training | `scripts/train_adapter.py` |
| Evaluation scripts | Metrics computation | `scripts/evaluate_adapter.py` |
| Notebooks | Analysis and visualization | `notebooks/c2_analysis.ipynb` |

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| VLM latents lack spatial info | Medium | High | Pivot to pre-merge extraction (7.1) |
| LTX latent space too different | Medium | High | Try alternative decoders (7.5) |
| Training instability | Low | Medium | LR sweep, gradient clipping |
| Memorization | Low | High | Strong train/test split, data augmentation |
| Compute availability | Medium | Medium | Cloud burst compute (Lambda, RunPod) |

---

## 12. Go/No-Go Decision Framework

After completing experiments, use this framework:

### GREEN (Proceed to C3)
- LPIPS < 0.30 with 10M adapter
- Clear diminishing returns in scaling curve
- No memorization signals
- Training stable and reproducible

### YELLOW (Proceed with caution)
- LPIPS 0.30-0.40 with 10M adapter
- Scaling curve shows continued improvement (may need more capacity)
- Consider pivots 7.1 or 7.4 before proceeding

### RED (Major pivot required)
- LPIPS > 0.40 with 10M adapter
- Linear scaling (no diminishing returns)
- Memorization detected
- Requires >50M params for acceptable quality

**Decision Point:** End of Phase 3 (Day 16)

---

## Appendix A: Latent Space Visualization Protocol

To understand latent alignment, create these visualizations:

1. **t-SNE/UMAP of VLM latents** (colored by video category)
2. **t-SNE/UMAP of LTX latents** (same videos, same coloring)
3. **t-SNE/UMAP of projected VLM latents** (after adapter, compare to target)
4. **Cosine similarity heatmap** (VLM token i vs. LTX position j)
5. **Attention patterns** (for cross-attention adapter, what attends to what?)

## Appendix B: Reconstruction Quality Examples

Create a gallery showing:
- Input frames (ground truth)
- VLM-encoded -> adapter -> LTX-decoded (reconstructed)
- Difference maps (highlight where reconstruction fails)

Categories:
- Simple scenes (single object, static background)
- Complex scenes (multiple objects, occlusion)
- High-motion scenes (fast movement)
- Fine detail scenes (text, small objects)

## Appendix C: Related Experiments

| Experiment | Relationship | Notes |
|------------|--------------|-------|
| C1: VLM Information Content | Upstream | C2 depends on VLM having sufficient info |
| C3: Future Prediction | Downstream | Uses C2 adapter for prediction |
| HunyuanVideo adapter | Alternative | If LTX fails, try HunyuanVideo bridging |
