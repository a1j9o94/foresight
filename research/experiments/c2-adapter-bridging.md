# Experiment Plan: C2 - Adapter Bridging

**Claim:** A small adapter (~10-50M parameters) can effectively bridge the hybrid encoder (DINOv2 + VLM) to the LTX-Video decoder, achieving >90% of the reconstruction quality of a much larger adapter.

**Status:** Planning
**Priority:** Critical (Phase 2 - required for Gate 2)
**Owner:** TBD
**Created:** 2026-01-20 (Updated from 2025-01-18 draft)

**Dependencies:** P2-Hybrid-Encoder (PASSED - spatial_iou=0.837, lpips=0.162)

---

## 1. Objective

Design and validate an efficient adapter architecture that:
- Bridges the hybrid encoder output (DINOv2 spatial + VLM semantic features) to LTX-Video conditioning
- Achieves high parameter efficiency: 10M params should achieve >90% of 100M params quality
- Maintains or improves upon P2 reconstruction quality (LPIPS < 0.35, Spatial IoU > 0.60)
- Keeps inference latency overhead minimal (<25% over baseline)

**Core Question:** What is the minimal adapter architecture that can effectively translate hybrid encoder features into video decoder conditioning?

**Why This Matters:**
- P2 validated that DINOv2 + VLM fusion achieves excellent reconstruction (LPIPS=0.162, Spatial IoU=0.837)
- However, P2's fusion module was 78M params - can we achieve similar quality with 10-20M params?
- Efficient adapters are critical for practical deployment and eventual real-time inference
- Phase 3 (future prediction) will train on top of this adapter architecture

---

## 2. Hypothesis

**Primary Hypothesis:**
A 10M parameter adapter using cross-attention with learned query tokens can achieve >90% of the reconstruction quality (measured by LPIPS) of a 100M parameter adapter, when bridging the hybrid encoder to LTX-Video.

**Quantitative Predictions:**

| Adapter Size | Expected LPIPS | Expected Spatial IoU | Params |
|--------------|----------------|---------------------|--------|
| 100M (reference) | 0.16 | 0.84 | 100M |
| 50M | 0.17 | 0.82 | 50M |
| 20M | 0.18 | 0.80 | 20M |
| 10M | 0.19 | 0.78 | 10M |

**Null Hypothesis:**
Adapter capacity has a strong linear relationship with reconstruction quality, and 10M params cannot achieve >80% of 100M performance.

**Falsifiability:**
- If 10M adapter LPIPS > 0.20 (i.e., >25% worse than 100M): Small adapters are insufficient
- If 10M adapter requires >2x training time: Efficiency benefits are negated
- If 10M adapter spatial IoU < 0.70: Spatial information is lost in compression

---

## 3. Background

### 3.1 What P2 Established

From P2 results (`research/experiments/p2-hybrid-encoder/results.yaml`):

| Component | Achievement | Notes |
|-----------|-------------|-------|
| Spatial IoU | 0.837 | DINOv2-ViT-L preserves spatial info |
| LPIPS | 0.162 | Hybrid fusion achieves excellent perceptual quality |
| Fusion module | 78M params | Cross-attention fusion (4 layers) |
| Latency overhead | 31.9% | With ViT-L (down from 68% with ViT-G) |

**Key insight:** The hybrid encoder works. Now we need to optimize the adapter connecting it to the video decoder.

### 3.2 Adapter Design Space

The adapter must transform hybrid encoder output to LTX-Video conditioning:

```
Input: Hybrid Features
  - DINOv2-ViT-L: [B, 256, 1024] (16x16 patches, 1024 dim)
  - VLM (Qwen2.5-VL): [B, T_vlm, 3584] (variable length, 3584 dim)

Output: LTX-Video Conditioning
  - [B, 77, 4096] (fixed 77 tokens, 4096 dim)
```

**Design choices to explore:**
1. **Query-based adapter:** Learned queries attend to hybrid features (DETR-style)
2. **Bottleneck adapter:** Compress features through bottleneck, then expand
3. **LoRA-style adapter:** Low-rank projections from each stream
4. **Mixture of experts:** Sparse routing to specialized sub-adapters

### 3.3 Related Work

- **Adapter layers (Houlsby et al.):** Add small bottleneck layers between transformer blocks
- **LoRA (Hu et al.):** Low-rank decomposition for efficient fine-tuning
- **Q-Former (BLIP-2):** Learned queries bridge vision encoder to LLM
- **Perceiver (Jaegle et al.):** Fixed number of latent queries process variable-length inputs

### 3.4 Lessons from P2

From P2 ablation study (E-P2.5):
- Cross-attention outperformed FiLM and concatenation for fusion
- 2 layers achieved best balance of quality vs training stability
- VLM weight 0.3 (vs DINOv2 0.7) showed best results
- Deeper models (4-6 layers) showed overfitting on limited data

---

## 4. Experimental Setup

### 4.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x A100 40GB | 1x A100 80GB |
| CPU RAM | 64GB | 128GB |
| Storage | 200GB SSD | 400GB NVMe |

**VRAM breakdown (inference):**
- Qwen2.5-VL-7B (bf16): ~15GB
- DINOv2-ViT-L (bf16): ~2GB
- Adapter (variable): 0.1-0.4GB
- LTX-Video (bf16): ~8GB
- **Total: ~26GB minimum**

### 4.2 Software Dependencies

```bash
# Existing from P2
pip install torch>=2.1.0 torchvision torchaudio
pip install transformers>=4.40.0 accelerate>=0.27.0
pip install diffusers>=0.27.0
pip install flash-attn --no-build-isolation
pip install timm>=0.9.0  # DINOv2

# Evaluation
pip install lpips pytorch-fid scikit-learn

# Utilities
pip install wandb einops matplotlib seaborn
```

### 4.3 Models (Pre-cached on Modal)

From P2 infrastructure:
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `dinov2_vitl14` (via torch.hub)
- `Lightricks/LTX-Video`

### 4.4 Datasets

**Training:**

| Dataset | N samples | Purpose |
|---------|-----------|---------|
| Synthetic shapes | 10,000 | Controlled spatial testing |
| COCO val2017 subset | 5,000 | Real-world objects |
| Something-Something v2 | 5,000 | Action/motion |

**Validation:**

| Dataset | N samples | Purpose |
|---------|-----------|---------|
| Synthetic shapes | 1,000 | Spatial IoU testing |
| COCO val2017 (held-out) | 500 | Real-world evaluation |

---

## 5. Experiments

### E2.1: Baseline Adapter Scaling Study

**Objective:** Establish the scaling relationship between adapter size and reconstruction quality.

**Protocol:**
1. Train adapters at 5M, 10M, 20M, 50M, 100M parameter scales
2. Use identical training procedure (loss, optimizer, data)
3. Evaluate reconstruction quality at each scale
4. Compute efficiency curve (quality vs params)

**Implementation:**

```python
class ScalableQueryAdapter(nn.Module):
    """Adapter with configurable capacity via width and depth."""

    def __init__(
        self,
        vlm_dim: int = 3584,
        dino_dim: int = 1024,
        ltx_dim: int = 4096,
        hidden_dim: int = 512,      # Controls width
        n_layers: int = 2,           # Controls depth
        n_queries: int = 77,         # Output tokens
    ):
        super().__init__()

        # Input projections
        self.vlm_proj = nn.Linear(vlm_dim, hidden_dim)
        self.dino_proj = nn.Linear(dino_dim, hidden_dim)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 64),
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, ltx_dim)

    def forward(self, vlm_features, dino_features):
        B = vlm_features.size(0)

        # Project and concatenate
        vlm = self.vlm_proj(vlm_features)
        dino = self.dino_proj(dino_features)
        context = torch.cat([vlm, dino], dim=1)

        # Cross-attention
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, context)

        return self.out_proj(queries)
```

**Adapter configurations:**

| Config | hidden_dim | n_layers | Approx Params |
|--------|------------|----------|---------------|
| 5M | 256 | 2 | ~5M |
| 10M | 384 | 2 | ~10M |
| 20M | 512 | 3 | ~20M |
| 50M | 768 | 4 | ~50M |
| 100M | 1024 | 4 | ~100M |

**Training:**
- Loss: LPIPS + 0.1 * L2
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Batch size: 8
- Epochs: 50
- Gradient accumulation: 4 steps

**Metrics:**

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| LPIPS (10M) | < 0.20 | < 0.22 | > 0.25 |
| Param efficiency | > 0.90 | > 0.85 | < 0.80 |
| Training time ratio | < 1.2x | < 1.5x | > 2x |

**Deliverables:**
- Scaling curve: LPIPS vs params
- Efficiency metric: (Quality_10M / Quality_100M)
- Training time comparison
- Recommendation for optimal adapter size

**Time estimate:** 5 days

---

### E2.2: Architecture Comparison

**Objective:** Compare different adapter architectures at fixed parameter budget (~10M params).

**Architectures to test:**

#### A. Query-based Adapter (baseline from E2.1)
- Learned queries attend to concatenated features
- Simple, proven architecture

#### B. Bottleneck Adapter
```python
class BottleneckAdapter(nn.Module):
    """Compress features through bottleneck, then expand."""

    def __init__(self, vlm_dim=3584, dino_dim=1024, ltx_dim=4096,
                 bottleneck_dim=128, n_tokens=77):
        super().__init__()

        # Compress each stream
        self.vlm_compress = nn.Sequential(
            nn.Linear(vlm_dim, 512),
            nn.GELU(),
            nn.Linear(512, bottleneck_dim),
        )
        self.dino_compress = nn.Sequential(
            nn.Linear(dino_dim, 512),
            nn.GELU(),
            nn.Linear(512, bottleneck_dim),
        )

        # Bottleneck fusion
        self.fusion = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

        # Expand to output
        self.expand = nn.Sequential(
            nn.Linear(512, n_tokens * ltx_dim // 4),
            nn.GELU(),
            nn.Linear(n_tokens * ltx_dim // 4, n_tokens * ltx_dim),
        )

    def forward(self, vlm_features, dino_features):
        vlm = self.vlm_compress(vlm_features.mean(1))
        dino = self.dino_compress(dino_features.mean(1))
        fused = self.fusion(torch.cat([vlm, dino], dim=-1))
        out = self.expand(fused)
        return out.view(out.size(0), 77, -1)
```

#### C. LoRA-style Adapter
```python
class LoRAAdapter(nn.Module):
    """Low-rank projections for efficient bridging."""

    def __init__(self, vlm_dim=3584, dino_dim=1024, ltx_dim=4096,
                 rank=64, n_tokens=77):
        super().__init__()

        # Low-rank projections for VLM
        self.vlm_down = nn.Linear(vlm_dim, rank, bias=False)
        self.vlm_up = nn.Linear(rank, ltx_dim, bias=False)

        # Low-rank projections for DINOv2
        self.dino_down = nn.Linear(dino_dim, rank, bias=False)
        self.dino_up = nn.Linear(rank, ltx_dim, bias=False)

        # Query generation
        self.query_gen = nn.Linear(rank * 2, n_tokens * ltx_dim)

    def forward(self, vlm_features, dino_features):
        vlm_low = self.vlm_down(vlm_features.mean(1))
        dino_low = self.dino_down(dino_features.mean(1))
        combined = torch.cat([vlm_low, dino_low], dim=-1)
        out = self.query_gen(combined)
        return out.view(out.size(0), 77, -1)
```

#### D. Perceiver-style Adapter
```python
class PerceiverAdapter(nn.Module):
    """Iterative cross-attention with fixed latent array."""

    def __init__(self, vlm_dim=3584, dino_dim=1024, ltx_dim=4096,
                 latent_dim=256, n_latents=77, n_iterations=3):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(n_latents, latent_dim) * 0.02)

        self.vlm_proj = nn.Linear(vlm_dim, latent_dim)
        self.dino_proj = nn.Linear(dino_dim, latent_dim)

        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(latent_dim, 4, batch_first=True)
            for _ in range(n_iterations)
        ])
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(latent_dim, 4, batch_first=True)
            for _ in range(n_iterations)
        ])

        self.out_proj = nn.Linear(latent_dim, ltx_dim)

    def forward(self, vlm_features, dino_features):
        B = vlm_features.size(0)
        context = torch.cat([
            self.vlm_proj(vlm_features),
            self.dino_proj(dino_features)
        ], dim=1)

        x = self.latents.unsqueeze(0).expand(B, -1, -1)
        for cross_attn, self_attn in zip(self.cross_attns, self.self_attns):
            x = x + cross_attn(x, context, context)[0]
            x = x + self_attn(x, x, x)[0]

        return self.out_proj(x)
```

**Protocol:**
1. Implement each architecture at ~10M params
2. Train with identical procedure
3. Compare reconstruction quality and inference speed

**Metrics:**

| Metric | Per Architecture |
|--------|------------------|
| LPIPS | Lower is better |
| Spatial IoU | Higher is better |
| Inference latency | Lower is better |
| Training stability | Loss variance |
| Memory usage | Peak VRAM |

**Deliverables:**
- Architecture comparison table
- Best architecture recommendation
- Trade-off analysis (quality vs speed)

**Time estimate:** 4 days

---

### E2.3: Training Strategy Optimization

**Objective:** Find optimal training strategy for the chosen adapter architecture.

**Strategies to test:**

#### A. Loss Functions

| Loss | Description |
|------|-------------|
| LPIPS only | Perceptual loss |
| LPIPS + L2 | Perceptual + pixel |
| LPIPS + SSIM | Perceptual + structural |
| Multi-scale LPIPS | Multiple resolutions |

#### B. Learning Rate Schedules

| Schedule | Description |
|----------|-------------|
| Constant | lr=1e-4 |
| Cosine decay | Start 1e-4, decay to 1e-6 |
| Warmup + decay | 1k warmup, cosine decay |
| Cyclic | Oscillate between 1e-5 and 1e-4 |

#### C. Data Augmentation

| Augmentation | Description |
|--------------|-------------|
| None | Baseline |
| Color jitter | Random brightness/contrast |
| Random crop | Scale 0.8-1.0 |
| MixUp | Blend image pairs |

#### D. Regularization

| Technique | Description |
|-----------|-------------|
| Dropout | 0.1, 0.2 on adapter layers |
| Weight decay | 1e-4, 1e-3 |
| Label smoothing | 0.1 (for auxiliary tasks) |

**Protocol:**
1. Grid search over loss functions
2. Grid search over learning rate schedules
3. Ablate augmentation strategies
4. Combine best settings

**Metrics:**

| Metric | Target |
|--------|--------|
| Final LPIPS | Minimum achievable |
| Training stability | Low variance |
| Convergence speed | Epochs to plateau |

**Deliverables:**
- Training recipe with best hyperparameters
- Ablation results table
- Learning curves for each strategy

**Time estimate:** 4 days

---

### E2.4: Final Efficiency Validation

**Objective:** Comprehensive validation of the optimized adapter against success criteria.

**Protocol:**
1. Train final adapter with best architecture + training strategy
2. Evaluate on held-out test sets
3. Compute parameter efficiency metric
4. Measure inference latency
5. Compare to P2 baseline

**Evaluation datasets:**
- Synthetic shapes (500 samples) - Spatial precision
- COCO val (500 samples) - Real-world objects
- Something-Something v2 (200 clips) - Temporal coherence

**Metrics (success criteria from research_plan.yaml):**

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| param_efficiency | > 0.90 | > 0.85 | < 0.80 |
| LPIPS (10M adapter) | < 0.18 | < 0.20 | > 0.25 |
| Spatial IoU | > 0.75 | > 0.70 | < 0.65 |
| Inference latency | < 150ms | < 200ms | > 300ms |

**param_efficiency calculation:**
```
param_efficiency = 1 - (LPIPS(10M) - LPIPS(100M)) / LPIPS(100M)
```
- Target: 10M achieves >90% of 100M quality
- If 100M LPIPS = 0.16, then 10M must achieve LPIPS < 0.178

**Additional metrics:**
- Memory reduction: 100M VRAM vs 10M VRAM
- Training time ratio: 10M epochs vs 100M epochs
- Generalization gap: train vs val LPIPS

**Deliverables:**
- Final trained adapter checkpoint
- Comprehensive metrics table
- Side-by-side visualizations
- Gate 2 readiness assessment
- Recommended configuration for Phase 3

**Time estimate:** 3 days

---

## 6. Success Criteria

### 6.1 Primary Success Criteria (from research_plan.yaml)

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| **param_efficiency** | > 0.90 | > 0.85 | < 0.80 |

**Definition:** 10M parameter adapter achieves X% of 100M parameter adapter reconstruction quality, where:
- param_efficiency = 1 - (LPIPS(10M) - LPIPS(100M)) / LPIPS(100M)

### 6.2 Secondary Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| LPIPS (10M adapter) | < 0.18 | Must be <12.5% worse than 100M |
| Spatial IoU | > 0.75 | Maintain P2 spatial quality |
| Inference latency | < 150ms | Adapter contribution only |
| Training time | < 24 GPU-hours | For 10M adapter |
| Memory (10M) | < 0.5GB | Adapter weights only |

### 6.3 Comparison Baselines

| Baseline | Source | Metrics |
|----------|--------|---------|
| P2 Fusion Module | P2 results | LPIPS=0.162, params=78M |
| VLM-only (C1) | C1 results | LPIPS=0.264 |

### 6.4 Go/No-Go Decision Points

| Checkpoint | Timing | Criteria | Decision |
|------------|--------|----------|----------|
| E2.1 Complete | Day 5 | 10M efficiency > 0.85 | Continue / Increase min size |
| E2.2 Complete | Day 9 | Best arch identified | Continue / Iterate |
| E2.3 Complete | Day 13 | Training recipe found | Continue / Extend |
| E2.4 Complete | Day 16 | param_efficiency > 0.90 | Pass Gate / Investigate |

---

## 7. Failure Criteria

### 7.1 Hard Failures (abort experiment)

1. **Scaling shows no efficiency gains (E2.1):**
   - 10M params < 70% of 100M quality
   - Implication: Small adapters fundamentally insufficient
   - Action: Investigate architectural bottlenecks, consider hybrid training

2. **All architectures fail quality threshold (E2.2):**
   - Best architecture LPIPS > 0.25 at 10M params
   - Implication: Architectural search space too limited
   - Action: Expand search to include transformer variants, MoE

3. **Training is unstable across all strategies (E2.3):**
   - Loss diverges or oscillates wildly
   - Implication: Feature distribution mismatch
   - Action: Add feature normalization, investigate encoder consistency

### 7.2 Soft Failures (investigate before pivoting)

1. **Marginal efficiency gains:** param_efficiency 0.80-0.85
   - May need 20M params as minimum viable size

2. **Latency regression:** Adapter adds >200ms
   - May need architectural optimization or quantization

3. **Generalization gap:** Train LPIPS 0.18, val LPIPS 0.25
   - May need more diverse training data or regularization

---

## 8. Pivot Options

If C2 approach fails, consider these alternatives:

### 8.1 Increase Minimum Adapter Size

**Rationale:** Accept 20-30M params if 10M is insufficient.

**Trade-off:** Higher memory, slower inference, but within acceptable limits.

### 8.2 Joint Adapter + Decoder LoRA Training

**Rationale:** Allow video decoder to "meet halfway" via LoRA fine-tuning.

**Implementation:**
- LoRA on LTX transformer (4-8M additional params)
- Joint training: adapter + decoder LoRA

**Pros:** More capacity for alignment
**Cons:** Loses pure modularity

### 8.3 Distillation from Large Adapter

**Rationale:** Train large adapter first, distill to small.

**Implementation:**
- Train 100M adapter to convergence
- Use teacher-student distillation to 10M

**Pros:** May capture key mappings efficiently
**Cons:** Requires 2x training time

### 8.4 Mixture of Experts Adapter

**Rationale:** Sparse computation allows larger effective capacity.

**Implementation:**
- 8 expert adapters, each 2M params
- Router selects top-2 per sample
- Effective 4M active, 16M total

**Pros:** Larger capacity with sparse compute
**Cons:** More complex training dynamics

---

## 9. Implementation Plan

### 9.1 Handler Structure

```
infra/modal/handlers/c2/
├── __init__.py           # Exports get_handlers()
├── e2_1.py               # Baseline adapter scaling study
├── e2_2.py               # Architecture comparison
├── e2_3.py               # Training strategy optimization
├── e2_4.py               # Final efficiency validation
├── adapters/
│   ├── query_adapter.py  # Query-based adapter
│   ├── bottleneck.py     # Bottleneck adapter
│   ├── lora_adapter.py   # LoRA-style adapter
│   └── perceiver.py      # Perceiver-style adapter
└── utils/
    ├── feature_extraction.py  # Reuse from P2
    └── evaluation.py          # Metrics computation
```

### 9.2 Reuse from P2

The following can be directly reused:
- `handlers/p2/e_p2_1.py`: DINOv2 feature extraction (`extract_dinov2_features`)
- `handlers/p2/e_p2_1.py`: Dataset generation (`generate_position_dataset`, `generate_detection_dataset`)
- VLM feature extraction patterns
- Evaluation metrics (LPIPS, IoU computation)

### 9.3 New Components Required

1. **Scalable adapter implementations** (4 architectures)
2. **Training harness** with configurable loss/optimizer
3. **Efficiency metric computation**
4. **Adapter checkpointing** for comparison

---

## 10. Timeline

| Phase | Days | Experiments | Milestones |
|-------|------|-------------|------------|
| Setup | 1 | Codebase prep | Adapter templates ready |
| E2.1 | 5 | Scaling study | Efficiency curve |
| E2.2 | 4 | Architecture comparison | Best architecture |
| E2.3 | 4 | Training optimization | Best training recipe |
| E2.4 | 3 | Final validation | Gate 2 assessment |
| Analysis | 2 | Final report | Documentation |
| **Total** | **19 days** | | |

**Parallelization opportunities:**
- E2.1 adapter training can run in parallel across scales
- E2.2 architectures can train in parallel
- E2.3 hyperparameter search can use parallel trials

**Optimistic timeline:** 14 days
**Pessimistic timeline:** 25 days

---

## 11. Resource Requirements

### 11.1 Compute

| Phase | GPU Type | GPU-Hours | Notes |
|-------|----------|-----------|-------|
| E2.1 | A100-80GB | 80 | 5 adapter scales x 16h each |
| E2.2 | A100-80GB | 60 | 4 architectures x 15h each |
| E2.3 | A100-80GB | 50 | ~20 hyperparameter configs |
| E2.4 | A100-80GB | 30 | Final training + evaluation |
| **Total** | | **220** | |

**Estimated cost:** ~$440 (at $2/GPU-hour)

### 11.2 Storage

| Item | Size | Notes |
|------|------|-------|
| Cached features (from P2) | ~50GB | Reuse |
| Adapter checkpoints | ~5GB | 10-100M params x 5 scales |
| Results/artifacts | ~5GB | Visualizations, metrics |
| **Total additional** | ~10GB | |

---

## 12. Dependencies

### 12.1 Prerequisites (must complete before starting)

- [x] P2 experiment completed (provides hybrid encoder baseline)
- [x] Gate 1 passed (validates reconstruction approach)
- [x] DINOv2-ViT-L validated (spatial preservation confirmed)
- [ ] GPU resources allocated (A100-80GB for training)

### 12.2 Blocks

This experiment blocks:
- **C3 (Future Prediction):** Needs efficient adapter for prediction training
- **Q3 (Temporal Coherence):** Needs stable adapter for temporal analysis
- **Gate 2:** Cannot proceed without C2 completion

### 12.3 External Dependencies

- Modal infrastructure for GPU access
- W&B for experiment tracking
- Pre-cached models from P2

---

## 13. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 10M adapter quality too low | Medium (30%) | High | Test 20M as fallback target |
| Training instability | Low (15%) | Medium | Gradient clipping, warmup |
| Feature distribution shift | Low (10%) | Medium | Add layer normalization |
| Latency overhead too high | Low (10%) | Medium | Test FP16 inference |
| Longer timeline | Medium (25%) | Low | Built-in buffer, parallelization |

---

## 14. Deliverables

### 14.1 Code Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Query adapter | `handlers/c2/adapters/query_adapter.py` | Baseline adapter |
| Bottleneck adapter | `handlers/c2/adapters/bottleneck.py` | Compressed adapter |
| LoRA adapter | `handlers/c2/adapters/lora_adapter.py` | Low-rank adapter |
| Perceiver adapter | `handlers/c2/adapters/perceiver.py` | Iterative adapter |
| Training harness | `handlers/c2/utils/training.py` | Configurable training |

### 14.2 Checkpoints

| Checkpoint | Size | Description |
|------------|------|-------------|
| `adapter_5m_best.pt` | ~20MB | 5M adapter |
| `adapter_10m_best.pt` | ~40MB | 10M adapter |
| `adapter_20m_best.pt` | ~80MB | 20M adapter |
| `adapter_50m_best.pt` | ~200MB | 50M adapter |
| `adapter_100m_best.pt` | ~400MB | 100M reference |

### 14.3 Reports

| Report | Format | Audience |
|--------|--------|----------|
| Technical findings | `research/experiments/c2-adapter-bridging/FINDINGS.md` | Research team |
| Results data | `research/experiments/c2-adapter-bridging/results.yaml` | Validation system |
| Scaling curve | `artifacts/scaling_curve.png` | Visualization |

---

## 15. Open Questions

To be resolved during experiments:

1. **Optimal bottleneck dimension:** What is the minimum information-preserving bottleneck?
2. **Query count:** Do we need 77 output tokens or can we use fewer?
3. **Positional encoding:** Should the adapter include spatial positional info?
4. **Stream weighting:** Should VLM and DINOv2 features be weighted differently?
5. **Temporal handling:** How does the adapter handle video (frame-by-frame vs temporal pooling)?

---

## 16. Relation to Research Plan

From `research/research_plan.yaml`:

```yaml
c2-adapter-bridging:
  name: "C2: Adapter Bridging"
  type: claim
  phase: 2
  status: not_started
  description: "A small adapter (~10-50M params) can effectively bridge VLM to video decoder"
  sub_experiments: [e2_1, e2_2, e2_3, e2_4]
  success_criteria:
    param_efficiency:
      target: 0.9
      acceptable: 0.8
      failure: 0.6
      direction: higher
      note: "10M params achieves X% of 100M performance"
  dependencies: [p2-hybrid-encoder]
```

This experiment is required to pass **Gate 2: Bridging** which also requires Q3 (Temporal Coherence).

---

## 17. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-01-18 | Claude | Initial draft (pre-P2) |
| 1.0 | 2026-01-20 | Claude | Complete rewrite incorporating P2 results, updated architecture designs, detailed sub-experiments |
