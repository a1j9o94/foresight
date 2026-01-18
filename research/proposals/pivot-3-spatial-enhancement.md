# Pivot Option 3: Spatial Enhancement Modules

**Research Proposal for Department Chair Review**

**Project:** Foresight - Generative Latent Prediction for Visual Reasoning
**Proposal Date:** 2026-01-18
**Author:** Research Team
**Status:** Pivot Candidate (Gate 1 Partial Failure)

---

## Executive Summary

Gate 1 experiments have revealed a fundamental limitation in the Foresight architecture: Qwen2.5-VL's latent representations lose critical spatial information that cannot be recovered by our current adapter design. While the VLM achieves excellent perceptual reconstruction (LPIPS=0.236, well below the 0.35 target), spatial localization fails catastrophically: bounding box IoU=0.104 (target >0.7) and detection mAP@0.5=0.001 (target >0.4). This is not a training or extraction point issue--the 2x2 token merger and LLM layers fundamentally destroy positional information that was already degraded in the vision encoder itself.

This proposal presents **Pivot Option 3: Spatial Enhancement Modules**--a set of learned components designed to recover or reconstruct spatial information from VLM embeddings. The approach adds 5-20M trainable parameters in the form of spatial attention mechanisms, coordinate encoding injection, and learned upsampling layers positioned between the VLM and the video decoder's conditioning interface.

The core technical question is whether spatial information that has been compressed and entangled can be "hallucinated" back into a form useful for video generation, or whether the information loss is truly irrecoverable. Our analysis suggests this is a high-risk, moderate-reward pivot: if it works, we preserve the architectural elegance of using a single VLM for both reasoning and visual generation; if it fails, we have strong evidence that a hybrid approach (separate spatial and semantic pathways) is necessary. We estimate 6-8 weeks of experimental work and approximately $15K-25K in compute costs to reach a definitive conclusion.

---

## 1. Technical Background

### 1.1 The Spatial Information Loss Problem

Our Q2 experiments systematically traced where spatial information degrades in the Qwen2.5-VL pipeline:

| Stage | Spatial IoU | Notes |
|-------|-------------|-------|
| Raw image input | 1.0 | Perfect by definition |
| ViT patch embedding | ~0.10 | **Already degraded before merger** |
| Post-2x2 token merger | ~0.10 | No additional loss |
| LLM Layer 0 | 0.066 | Slight further degradation |
| LLM Layer 14 (optimal) | 0.094 | Marginal recovery |
| LLM Layer 27 (final) | 0.042 | Worst performance |

**Critical Finding:** The spatial information is not "compressed" in a recoverable way--it is entangled with semantic features in a manner that destroys the original positional structure. The VLM's training objective (language-visual alignment via next-token prediction) actively optimizes away spatial precision in favor of semantic compression.

### 1.2 What Spatial Enhancement Must Accomplish

A spatial enhancement module would need to:

1. **Reconstruct positional relationships** from features that encode "what" but not "where"
2. **Maintain semantic consistency** while adding spatial structure
3. **Generate plausible spatial layouts** when true positions cannot be recovered
4. **Interface cleanly** with LTX-Video's 128-channel latent space

The fundamental challenge: we cannot recover information that has been destroyed. What we can potentially do is **learn to predict plausible spatial arrangements** based on semantic content and learned priors about object layouts.

### 1.3 Prior Work on Spatial Recovery

| Approach | Domain | Key Mechanism | Relevance |
|----------|--------|---------------|-----------|
| **Super-resolution** | Image enhancement | Learned upsampling + perceptual loss | High - similar "hallucination" of detail |
| **Depth estimation from monocular** | 3D vision | Learned geometric priors | Medium - reconstructs structure from 2D |
| **Grounding DINO** | Open-vocabulary detection | Fused text-image features + spatial heads | High - adds spatial outputs to semantic features |
| **SAM (Segment Anything)** | Segmentation | Prompt-conditioned spatial decoder | Medium - spatial structure from prompts |
| **IP-Adapter** | Image generation | Cross-attention injection of CLIP features | High - similar conditioning pathway |

**Key insight from literature:** Systems that successfully add spatial information to semantic features typically do so through dedicated spatial heads with strong inductive biases (convolutions, deformable attention, coordinate encoding).

---

## 2. Technical Approach

### 2.1 Proposed Architecture

We propose a **Spatial Enhancement Module (SEM)** inserted between VLM latent extraction and the adapter that conditions LTX-Video:

```
Current Pipeline:
  Image -> Qwen2.5-VL -> [latents] -> Adapter -> LTX-Video Conditioning

Proposed Pipeline:
  Image -> Qwen2.5-VL -> [latents] -> SEM -> Adapter -> LTX-Video Conditioning
                                       ^
                                       |
                              Spatial Enhancement Module
```

The SEM consists of three sub-modules that can be evaluated independently or jointly:

#### 2.1.1 Module A: Coordinate Encoding Injection

**Concept:** Re-inject explicit positional information that was lost during VLM processing.

**Implementation:**
```python
class CoordinateEncodingInjection(nn.Module):
    def __init__(self, hidden_dim=3584, num_tokens=256):
        super().__init__()
        # Learnable spatial position embeddings
        self.spatial_embed = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)

        # Sinusoidal coordinate encoding (2D Fourier features)
        self.coord_encoder = FourierFeatures(d_model=hidden_dim, max_freq=10.0)

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, vlm_latents, original_grid_shape):
        # vlm_latents: [B, N, D] - semantic features without spatial structure
        # original_grid_shape: (H, W) - the spatial grid these tokens came from

        # Generate coordinate encodings for each position
        h, w = original_grid_shape
        coords = self.generate_grid_coords(h, w)  # [H*W, 2]
        coord_features = self.coord_encoder(coords)  # [H*W, D]

        # Combine semantic and coordinate features
        combined = torch.cat([vlm_latents, coord_features.expand(B, -1, -1)], dim=-1)
        gate_weights = self.gate(combined)
        output = gate_weights * vlm_latents + (1 - gate_weights) * self.fusion(combined)

        return output  # [B, N, D] - now with spatial structure

# Parameters: ~7M (for hidden_dim=3584)
```

**Rationale:** The VLM tokens originated from a spatial grid (before the 2x2 merger: 1024 tokens from a 32x32 grid; after: 256 tokens from a 16x16 grid). By re-injecting coordinates based on token position, we provide the model with spatial grounding that can be learned to align with semantic content.

**Limitation:** This assumes tokens still maintain rough spatial correspondence to their original positions--an assumption that may not hold after attention-based mixing.

#### 2.1.2 Module B: Deformable Spatial Attention

**Concept:** Learn to attend to spatially-relevant features using deformable attention, which can adapt its receptive field based on content.

**Implementation:**
```python
class DeformableSpatialAttention(nn.Module):
    def __init__(self, hidden_dim=3584, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.deform_attn = DeformableAttention(
            d_model=hidden_dim,
            n_heads=num_heads,
            n_levels=num_levels,
            n_points=num_points
        )

        # Spatial queries - learnable spatial prototypes
        self.spatial_queries = nn.Parameter(torch.randn(256, hidden_dim) * 0.02)

        # Reference points generator
        self.reference_points = nn.Linear(hidden_dim, 2)  # Predict (x, y)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vlm_latents):
        # vlm_latents: [B, N, D]
        B = vlm_latents.shape[0]

        # Generate reference points for spatial queries
        ref_points = self.reference_points(self.spatial_queries)  # [256, 2]
        ref_points = ref_points.sigmoid()  # Normalize to [0, 1]

        # Deformable attention: queries attend to vlm_latents with learned offsets
        spatial_features = self.deform_attn(
            query=self.spatial_queries.expand(B, -1, -1),
            key=vlm_latents,
            value=vlm_latents,
            reference_points=ref_points.expand(B, -1, -1)
        )

        return self.output_proj(spatial_features)  # [B, 256, D]

# Parameters: ~15M
```

**Rationale:** Deformable attention has proven highly effective in detection tasks (DETR, Deformable DETR) for extracting spatially-precise features from semantic representations. The learned reference points and sampling offsets can adapt to where objects actually are, rather than assuming fixed spatial relationships.

**Key advantage:** Does not assume tokens maintain spatial correspondence--learns to find spatial information wherever it may be encoded.

#### 2.1.3 Module C: Learned Spatial Upsampling

**Concept:** Upsample the compressed token representation back to a higher spatial resolution, learning to "fill in" spatial detail.

**Implementation:**
```python
class LearnedSpatialUpsampling(nn.Module):
    def __init__(self, hidden_dim=3584, output_dim=128, scale_factor=4):
        super().__init__()

        # Reshape tokens to spatial grid
        self.pre_reshape = nn.Linear(hidden_dim, 512)

        # Pixel shuffle upsampling (efficient learned upsampling)
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512 // (4**i), 512 // (4**(i+1)) * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GroupNorm(32, 512 // (4**(i+1))),
                nn.GELU()
            ) for i in range(int(np.log2(scale_factor)))
        ])

        # Spatial refinement with residual blocks
        self.refine = nn.Sequential(
            ResBlock(512 // (scale_factor**2), 256),
            ResBlock(256, 256),
            nn.Conv2d(256, output_dim, kernel_size=1)
        )

    def forward(self, vlm_latents, spatial_shape=(16, 16)):
        # vlm_latents: [B, N, D] where N should match H*W
        B, N, D = vlm_latents.shape
        H, W = spatial_shape

        # Project and reshape to spatial grid
        x = self.pre_reshape(vlm_latents)  # [B, N, 512]
        x = x.transpose(1, 2).reshape(B, 512, H, W)  # [B, 512, H, W]

        # Learned upsampling
        for block in self.upsample_blocks:
            x = block(x)

        # Spatial refinement
        x = self.refine(x)  # [B, 128, H*scale, W*scale]

        return x

# Parameters: ~8M
```

**Rationale:** This approach treats the problem as analogous to super-resolution: given a low-resolution (spatially compressed) representation, learn to generate plausible high-resolution spatial structure. The key difference from standard super-resolution is that we're upsampling in feature space, not pixel space.

**Connection to LTX-Video:** The output can directly interface with LTX-Video's 128-channel latent conditioning, bypassing the need for a separate adapter.

### 2.2 Combined Architecture

For maximum spatial recovery, we can stack these modules:

```
VLM Latents [B, 256, 3584]
     |
     v
Coordinate Encoding Injection  (+7M params)
     |
     v
Deformable Spatial Attention   (+15M params)
     |
     v
Learned Spatial Upsampling     (+8M params)
     |
     v
Spatially-Enhanced Features [B, 128, 64, 64]
     |
     v
LTX-Video Conditioning
```

**Total additional parameters:** ~30M (within acceptable budget for adapter-scale training)

### 2.3 Training Strategy

**Phase 1: Spatial Reconstruction Pretraining**
- Objective: Reconstruct ground-truth spatial information from VLM latents
- Supervision: Use ground-truth bounding boxes, segmentation masks, or depth maps
- Loss: Combination of IoU loss, spatial MSE, and perceptual loss
- Data: Something-Something v2 with object annotations (can be auto-generated via detection model)

**Phase 2: End-to-End Video Reconstruction**
- Objective: Generate video that matches ground truth through the full pipeline
- Loss: LPIPS + spatial IoU + temporal consistency
- Fine-tune SEM while keeping VLM frozen

**Phase 3: Prediction Training (if Phases 1-2 succeed)**
- Extend to future frame prediction using learned query tokens
- Evaluate whether spatial enhancement enables meaningful prediction

---

## 3. Pros and Cons

### 3.1 Advantages

| Advantage | Impact | Confidence |
|-----------|--------|------------|
| **Preserves architectural simplicity** | High | High |
| Single VLM for both reasoning and visual generation; no need for separate spatial pathway | | |
| **Moderate parameter overhead** | Medium | High |
| 10-30M additional parameters is within our adapter training budget | | |
| **Leverages proven techniques** | Medium | Medium |
| Deformable attention, coordinate encoding, and learned upsampling are well-established | | |
| **Provides diagnostic value** | High | High |
| Even if unsuccessful, results clarify whether spatial information is fundamentally unrecoverable | | |
| **Incremental development** | Medium | High |
| Can evaluate each module independently before full integration | | |
| **Reusable components** | Low | Medium |
| Spatial enhancement modules could benefit other VLM applications requiring localization | | |

### 3.2 Disadvantages

| Disadvantage | Impact | Confidence |
|--------------|--------|------------|
| **May not work (information already destroyed)** | Critical | Medium |
| If spatial information is truly irrecoverable, no amount of learning will help | | |
| **Adds architectural complexity** | Medium | High |
| More components to train, debug, and maintain | | |
| **Increased inference latency** | Low-Medium | High |
| Additional forward passes through SEM (~10-20% latency increase estimated) | | |
| **Training data requirements unclear** | Medium | Medium |
| May need spatial annotations that existing datasets don't provide | | |
| **Risk of hallucinating incorrect spatial relationships** | Medium | Medium |
| Model might generate plausible but wrong spatial layouts | | |
| **Delays investigation of alternative pivots** | Medium | High |
| 6-8 weeks invested here cannot be spent on hybrid approaches | | |

### 3.3 Comparison to Alternative Pivots

| Criterion | Pivot 3 (This Proposal) | Pivot 1 (Hybrid Pathway) | Pivot 2 (Different VLM) |
|-----------|------------------------|--------------------------|-------------------------|
| **Architectural elegance** | High | Low | Medium |
| **Probability of success** | 30-40% | 60-70% | 50-60% |
| **Time to evaluate** | 6-8 weeks | 8-12 weeks | 4-6 weeks |
| **Compute cost** | $15-25K | $25-40K | $10-15K |
| **Diagnostic value** | High | Medium | Low |
| **Preserves original hypothesis** | Yes | Partially | No |

---

## 4. Resource Requirements

### 4.1 Compute Resources

| Phase | GPU Hours | Estimated Cost | Duration |
|-------|-----------|----------------|----------|
| **Phase 1: Module Development & Initial Training** | 200 | $600 | 2 weeks |
| - Coordinate encoding experiments | 50 | | |
| - Deformable attention experiments | 80 | | |
| - Upsampling experiments | 70 | | |
| **Phase 2: Combined Training** | 400 | $1,200 | 2 weeks |
| - Full SEM training on reconstruction | 250 | | |
| - Ablation studies | 150 | | |
| **Phase 3: End-to-End Integration** | 600 | $1,800 | 2 weeks |
| - VLM + SEM + LTX-Video pipeline | 400 | | |
| - Hyperparameter tuning | 200 | | |
| **Phase 4: Evaluation & Analysis** | 200 | $600 | 1 week |
| **Buffer (30%)** | 420 | $1,260 | 1 week |
| **Total** | **1,820** | **$5,460** | **8 weeks** |

*Costs based on $3/GPU-hour for A100-40GB cloud instances*

**Additional costs:**
- Data annotation (if needed): $2,000-5,000
- Human evaluation: $1,000-2,000
- Storage and bandwidth: $500-1,000

**Total estimated budget: $15,000-25,000**

### 4.2 Personnel

| Role | Effort | Responsibility |
|------|--------|----------------|
| **Lead Researcher** | 80% FTE | Architecture design, experiment analysis |
| **ML Engineer** | 100% FTE | Implementation, training infrastructure |
| **Research Assistant** | 50% FTE | Data preparation, evaluation |

### 4.3 Infrastructure

- **Training:** 4x A100-40GB (or equivalent), 8 weeks availability
- **Storage:** 2TB for checkpoints, intermediate features
- **Experiment tracking:** W&B project (existing)

---

## 5. Risk Assessment

### 5.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Spatial information fundamentally unrecoverable** | 40% | Critical | Early checkpoint at Week 2 to assess feasibility |
| **Modules don't compose well** | 25% | High | Test each module independently first |
| **Training instability** | 20% | Medium | Careful learning rate scheduling, gradient clipping |
| **Spatial hallucination (plausible but wrong)** | 35% | Medium | Strong supervision in Phase 1, calibration metrics |
| **Integration with LTX-Video fails** | 15% | High | Test conditioning interface early |
| **Insufficient training data** | 20% | Medium | Plan for synthetic data augmentation |

### 5.2 The Core Question: Can You Recover Destroyed Information?

This is the central scientific question underlying this proposal. Our assessment:

**Arguments that recovery IS possible:**
1. **Semantic content constrains spatial layout:** A cup is likely on a table, a person is likely standing on the ground. The VLM's semantic features implicitly encode spatial constraints.
2. **Redundancy in visual features:** Even after compression, some spatial information may remain in correlations between token features.
3. **Learned priors can fill gaps:** Super-resolution works by learning statistical priors about high-frequency content. Similarly, spatial enhancement can learn priors about object arrangements.

**Arguments that recovery is NOT possible:**
1. **Information-theoretic limits:** If the VLM's training actively discards spatial information, it cannot be reconstructed without additional input.
2. **Q2 results are stark:** mAP@0.5 = 0.001 is essentially random. This suggests spatial information is not merely degraded but eliminated.
3. **Attention mixing destroys correspondence:** After many attention layers, there's no guaranteed relationship between token position and original spatial position.

**Our probability estimate:** 30-40% chance of success sufficient to proceed past Gate 1.

### 5.3 Decision Checkpoints

| Checkpoint | Week | Success Criteria | Decision if Failed |
|------------|------|------------------|-------------------|
| **CP1: Module Feasibility** | 2 | Any module achieves IoU > 0.2 on synthetic test | Abort, pursue Pivot 1 |
| **CP2: Combined Performance** | 4 | Combined SEM achieves IoU > 0.4 on SS-v2 | Reduce scope or abort |
| **CP3: End-to-End Integration** | 6 | Full pipeline LPIPS < 0.35 AND IoU > 0.5 | Proceed with caveats or abort |
| **CP4: Final Evaluation** | 8 | Meets Gate 1 criteria (IoU > 0.6) | Proceed to Phase 2 or Pivot |

---

## 6. Implications for Remaining Experiments

### 6.1 Impact on Phase 2 Experiments

| Experiment | Impact if SEM Succeeds | Impact if SEM Fails |
|------------|------------------------|---------------------|
| **C2: Adapter Bridging** | Minimal - SEM becomes part of adapter | Must redesign for hybrid pathway |
| **Q3: Temporal Coherence** | Positive - better spatial grounding may help | Unchanged - temporal is orthogonal |

### 6.2 Impact on Phase 3 Experiments

| Experiment | Impact if SEM Succeeds | Impact if SEM Fails |
|------------|------------------------|---------------------|
| **C3: Future Prediction** | Enables spatial prediction testing | Cannot evaluate spatial prediction |
| **Q4: Training Data** | May need spatial annotations | Unchanged |
| **Q5: Prediction Horizon** | Full evaluation possible | Limited to semantic evaluation |

### 6.3 Impact on Phase 4 Experiments

| Experiment | Impact if SEM Succeeds | Impact if SEM Fails |
|------------|------------------------|---------------------|
| **C4: Pixel Verification** | Core hypothesis testable | Verification on semantic features only |

### 6.4 Summary: Critical Path Analysis

```
If SEM Succeeds:
  Gate 1 PASS -> C2 (with SEM) -> Gate 2 -> C3 -> Gate 3 -> C4 -> Gate 4
  Timeline: On track (minor delay from pivot investigation)

If SEM Fails:
  Gate 1 BLOCKED -> Pivot to Hybrid Architecture
                 -> Re-scope C2, C3, C4 for semantic-only pathway
                 -> Spatial reasoning via separate model (adds 4-8 weeks)
  Timeline: Significant delay, but project continues with modified hypothesis
```

---

## 7. Success Metrics

### 7.1 Primary Success Metrics (Gate 1 Requirements)

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| **Spatial IoU** | 0.559 (C1) / 0.104 (Q2) | > 0.6 | Bounding box overlap on test set |
| **Detection mAP@0.5** | 0.001 | > 0.4 | COCO-style evaluation |
| **LPIPS** | 0.236 | < 0.35 | Perceptual similarity (maintain) |

### 7.2 Secondary Success Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| **Spatial consistency across frames** | > 0.8 correlation | Objects stay in consistent locations |
| **Edge preservation (F1)** | > 0.5 | Sharp boundaries in generated video |
| **Training efficiency** | < 500 GPU-hours | Cost-effective solution |

### 7.3 Diagnostic Metrics (Even if Primary Fails)

| Metric | Purpose |
|--------|---------|
| **Best achievable IoU** | Quantify the ceiling on spatial recovery |
| **IoU vs. semantic category** | Identify which objects/scenes are recoverable |
| **Module contribution analysis** | Which enhancement mechanism helps most |

---

## 8. Recommendation

### 8.1 Assessment Summary

| Factor | Assessment |
|--------|------------|
| **Technical feasibility** | Uncertain (30-40% success probability) |
| **Strategic value** | High (preserves original architecture if successful) |
| **Diagnostic value** | High (definitive answer on spatial recovery) |
| **Resource efficiency** | Moderate ($15-25K, 6-8 weeks) |
| **Opportunity cost** | Significant (delays hybrid pivot by 6-8 weeks) |

### 8.2 Recommendation: Proceed with Time-Boxed Evaluation

**We recommend proceeding with Pivot 3, subject to the following constraints:**

1. **Strict time-box:** 8 weeks maximum, with hard decision point at Week 4
2. **Clear abort criteria:** If Week 2 checkpoint fails (no module achieves IoU > 0.2), immediately transition to Pivot 1
3. **Parallel preparation:** Begin architecture design for Pivot 1 (hybrid pathway) during Weeks 1-2, so transition is smooth if needed
4. **Budget cap:** $25K maximum; if additional spend required, trigger review

### 8.3 Rationale

The 30-40% success probability may seem low for a research investment, but consider:

1. **Diagnostic clarity:** Even failure provides valuable information that informs the correct path forward
2. **Architectural preference:** Success preserves the elegant single-pathway architecture that motivates Foresight
3. **Reversibility:** The work is not wasted; SEM modules could be useful in a hybrid architecture
4. **Alternative is also uncertain:** Pivot 1 (hybrid) has higher success probability (~60-70%) but also higher complexity and cost

### 8.4 Alternative Recommendations

**If department prioritizes speed to results:**
- Skip Pivot 3, proceed directly to Pivot 1 (hybrid architecture)
- Higher probability of Phase 2 entry, but loses architectural elegance
- Estimated: 8-12 weeks to Gate 1 pass

**If department prioritizes cost minimization:**
- Proceed with Pivot 2 (evaluate alternative VLMs with better spatial preservation)
- Lower cost ($10-15K) but may not find suitable alternative
- Estimated: 4-6 weeks to decision

**If department prioritizes scientific understanding:**
- Proceed with Pivot 3 as recommended, with comprehensive ablations
- Maximum diagnostic value, moderate cost
- Estimated: 8 weeks to definitive answer

---

## 9. Appendices

### Appendix A: Detailed Module Specifications

#### A.1 Coordinate Encoding Details

```python
class FourierFeatures(nn.Module):
    """2D Fourier positional encoding for coordinate injection."""

    def __init__(self, d_model=3584, max_freq=10.0, num_bands=64):
        super().__init__()
        self.d_model = d_model
        self.num_bands = num_bands

        # Frequency bands (log-spaced)
        freqs = torch.exp(torch.linspace(0, np.log(max_freq), num_bands))
        self.register_buffer('freqs', freqs)

        # Linear projection to model dimension
        self.proj = nn.Linear(num_bands * 4, d_model)  # 4 = sin/cos for x and y

    def forward(self, coords):
        # coords: [N, 2] normalized coordinates in [0, 1]
        x, y = coords[:, 0:1], coords[:, 1:2]

        # Apply frequency bands
        x_enc = torch.cat([torch.sin(x * f * 2 * np.pi) for f in self.freqs] +
                         [torch.cos(x * f * 2 * np.pi) for f in self.freqs], dim=-1)
        y_enc = torch.cat([torch.sin(y * f * 2 * np.pi) for f in self.freqs] +
                         [torch.cos(y * f * 2 * np.pi) for f in self.freqs], dim=-1)

        return self.proj(torch.cat([x_enc, y_enc], dim=-1))
```

#### A.2 Deformable Attention Details

We use the implementation from Deformable DETR with the following modifications:
- Single-scale attention (VLM features are single-scale)
- Learned reference points rather than from object queries
- Spatial queries designed for dense prediction rather than sparse detection

#### A.3 Upsampling Architecture Details

```
Input: [B, 256, 3584] (VLM tokens)
  |
Linear(3584 -> 512): [B, 256, 512]
  |
Reshape to spatial: [B, 512, 16, 16]
  |
PixelShuffle 2x: [B, 128, 32, 32]
  |
PixelShuffle 2x: [B, 32, 64, 64]
  |
ResBlock: [B, 32, 64, 64]
  |
ResBlock: [B, 32, 64, 64]
  |
Conv 1x1: [B, 128, 64, 64]
  |
Output: LTX-Video compatible conditioning
```

### Appendix B: Experimental Protocols

#### B.1 Synthetic Spatial Recovery Test

To isolate spatial recovery capability:

1. Generate synthetic images with known object positions
2. Process through Qwen2.5-VL to obtain latents
3. Train SEM to predict object positions from latents
4. Measure recovery accuracy (IoU, mAP)

This provides a clean test of whether spatial recovery is possible before integrating with video generation.

#### B.2 Ablation Study Design

| Ablation | What it Tests |
|----------|---------------|
| SEM without coordinate encoding | Necessity of explicit positional info |
| SEM without deformable attention | Necessity of content-adaptive spatial attention |
| SEM without upsampling | Necessity of spatial resolution increase |
| Frozen SEM + trained adapter | Whether SEM features are compatible with adapter |
| Joint SEM + adapter training | End-to-end optimization benefit |

### Appendix C: Related Literature

1. **Deformable DETR** (Zhu et al., 2020): Deformable attention for efficient detection
2. **NeRF** (Mildenhall et al., 2020): Coordinate-based neural representations
3. **ESRGAN** (Wang et al., 2018): Learned upsampling for super-resolution
4. **IP-Adapter** (Ye et al., 2023): Image conditioning for diffusion models
5. **Grounding DINO** (Liu et al., 2023): Open-vocabulary spatial grounding
6. **SAM** (Kirillov et al., 2023): Prompt-based spatial segmentation

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-18 | Initial proposal |

---

*Prepared for Department Chair Review*
*Foresight Project - Research Team*
