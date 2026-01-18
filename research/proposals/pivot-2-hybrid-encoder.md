# Research Proposal: Hybrid Architecture for Spatial-Semantic Video Generation

**Pivot Option 2: VLM Semantics + Separate Spatial Encoder**

**Proposal ID:** PIVOT-2-HYBRID
**Date:** 2026-01-18
**Status:** Proposed
**Principal Investigators:** Foresight Research Team
**Requested Review:** Department Chair

---

## Executive Summary

Gate 1 experiments have revealed a fundamental limitation in the Foresight architecture: Qwen2.5-VL's latent representations do not preserve spatial information at sufficient fidelity for video generation. Specifically, C1 achieved Spatial IoU of 0.559 (below the 0.6 threshold), and Q2 demonstrated near-zero object localization capability (Bounding Box IoU = 0.104, mAP@0.5 = 0.001). The 2x2 token merger and LLM layers are designed for semantic compression, not spatial preservation.

However, the news is not entirely negative. Q2 also revealed that **temporal information is excellently preserved** (100% direction accuracy, 90% temporal ordering accuracy), and Q1 confirmed that **VLM-to-video-decoder latent alignment is achievable** (CKA = 0.687). This suggests the VLM pathway remains valuable for semantic and temporal conditioning, but cannot serve as the sole source of visual information.

We propose a **hybrid architecture** that combines the VLM's semantic understanding with a dedicated spatial encoder (DINOv2 or SAM encoder). The VLM stream would provide action semantics, temporal reasoning, and world knowledge, while the spatial stream would provide precise object localization, edge information, and fine-grained visual details. A learned fusion module would combine these streams before conditioning the video decoder.

This pivot preserves the core research hypothesis--that AI systems benefit from generating pixel-level predictions--while addressing the empirically-demonstrated spatial bottleneck. We estimate the hybrid approach adds approximately 4-6 weeks to the timeline and 15-25M additional trainable parameters, but significantly increases the probability of achieving project success criteria.

---

## 1. Technical Approach

### 1.1 Architecture Overview

```
                                    +------------------+
                                    |                  |
    Video/Image Input ──────────────┤   Qwen2.5-VL     ├──────> Semantic Features
                      │             │   (frozen)       │        [B, T, 3584]
                      │             +------------------+
                      │                                         │
                      │                                         │ Semantic Stream
                      │                                         │ (temporal, action,
                      │                                         │  world knowledge)
                      │                                         v
                      │             +------------------+    +------------------+
                      │             │                  │    │                  │
                      └─────────────┤   DINOv2-giant   ├──> │  Cross-Attention │ ──> LTX-Video
                                    │   (frozen)       │    │  Fusion Module   │     Conditioning
                                    +------------------+    │  (trainable)     │
                                            │               +------------------+
                                            │                      ^
                                            │ Spatial Stream       │
                                            │ (localization,       │
                                            │  edges, details)     │
                                            └──────────────────────┘
                                              [B, H/14*W/14, 1536]
```

### 1.2 Spatial Encoder Selection

We evaluated three candidates for the spatial encoder:

| Encoder | Dim | Spatial Res | Strengths | Weaknesses |
|---------|-----|-------------|-----------|------------|
| **DINOv2-giant** | 1536 | H/14 x W/14 | Best spatial features, proven for dense tasks, emergent segmentation | Large (1.1B params), 14x14 patches |
| **SAM ViT-H** | 256 | H/16 x W/16 | Explicit segmentation training, strong edges | Designed for segmentation, not reconstruction |
| **SigLIP-SO400M** | 1152 | H/14 x W/14 | Good semantic-spatial balance, efficient | Weaker than DINOv2 on dense prediction |

**Recommendation: DINOv2-giant** is the primary candidate due to:

1. **Empirically validated spatial preservation**: DINOv2 features enable SOTA performance on depth estimation, semantic segmentation, and object detection without fine-tuning--precisely the properties we need.

2. **Emergent object awareness**: PCA visualization of DINOv2 features shows spontaneous object/background separation, suggesting the model learns localization without explicit supervision.

3. **Complementary to VLM**: DINOv2 was trained with self-supervised objectives (iBOT + DINO) that emphasize local feature consistency, whereas Qwen2.5-VL was trained for language-vision alignment. The information is genuinely complementary.

4. **Resolution compatibility**: DINOv2's 14x14 patch size is close to Qwen2.5-VL's pre-merge resolution, simplifying fusion.

### 1.3 Fusion Module Design

We propose a **cross-attention fusion module** that allows the spatial and semantic streams to interact:

```python
class HybridFusionModule(nn.Module):
    """
    Fuses VLM semantic features with DINOv2 spatial features.

    Design principles:
    1. Spatial features are the "query" - we want spatially-grounded output
    2. Semantic features are "key/value" - providing what/when information
    3. Learnable queries extract conditioning for video decoder
    """
    def __init__(
        self,
        vlm_dim: int = 3584,          # Qwen2.5-VL hidden dimension
        spatial_dim: int = 1536,       # DINOv2-giant dimension
        fusion_dim: int = 1024,        # Internal fusion dimension
        num_fusion_layers: int = 4,    # Cross-attention layers
        num_output_queries: int = 64,  # Queries for video decoder
        ltx_channels: int = 128,       # LTX-Video latent channels
    ):
        super().__init__()

        # Project both streams to common dimension
        self.vlm_proj = nn.Linear(vlm_dim, fusion_dim)
        self.spatial_proj = nn.Linear(spatial_dim, fusion_dim)

        # Learnable output queries (similar to DETR, Q-Former)
        self.output_queries = nn.Parameter(
            torch.randn(num_output_queries, fusion_dim) * 0.02
        )

        # Cross-attention: queries attend to both streams
        self.fusion_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=fusion_dim,
                num_heads=8,
                mlp_ratio=4.0,
            ) for _ in range(num_fusion_layers)
        ])

        # Final projection to LTX-Video conditioning space
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, ltx_channels),
        )

    def forward(
        self,
        vlm_features: Tensor,      # [B, T_vlm, 3584]
        spatial_features: Tensor,  # [B, H*W, 1536]
        spatial_positions: Tensor, # [B, H*W, 2] normalized coordinates
    ) -> Tensor:
        # Project to common space
        vlm_proj = self.vlm_proj(vlm_features)        # [B, T_vlm, fusion_dim]
        spatial_proj = self.spatial_proj(spatial_features)  # [B, H*W, fusion_dim]

        # Add positional information to spatial features
        spatial_proj = spatial_proj + self.pos_embed(spatial_positions)

        # Concatenate streams for cross-attention
        context = torch.cat([vlm_proj, spatial_proj], dim=1)  # [B, T_vlm + H*W, fusion_dim]

        # Learnable queries attend to combined context
        queries = self.output_queries.unsqueeze(0).expand(B, -1, -1)

        for layer in self.fusion_layers:
            queries = layer(queries, context)

        # Project to video decoder conditioning space
        output = self.output_proj(queries)  # [B, num_queries, ltx_channels]

        return output
```

**Design rationale:**

1. **Query-based output**: Using learnable queries (similar to DETR and Q-Former) allows the model to extract a fixed-size conditioning signal regardless of input resolution. This provides flexibility and enables attention-based selection of relevant features.

2. **Spatial features as primary**: The spatial stream provides the "backbone" of visual information, while semantic features modulate what aspects are emphasized for the current task/action.

3. **Position encoding**: Explicit positional encoding for spatial features ensures the fusion module knows where information comes from, compensating for the lack of position information in VLM features.

### 1.4 Training Strategy

**Phase 1: Spatial Reconstruction (2 weeks)**
- Train fusion module + spatial encoder pathway only (VLM frozen and bypassed)
- Objective: Reconstruct input frames from DINOv2 features alone
- Purpose: Validate spatial pathway before adding complexity
- Success criteria: LPIPS < 0.25, Spatial IoU > 0.7

**Phase 2: Semantic Integration (2 weeks)**
- Add VLM pathway, train fusion module
- Objective: Combine spatial reconstruction with semantic conditioning
- Purpose: Learn to leverage both streams effectively
- Success criteria: Maintains Phase 1 quality, semantic consistency > 0.8

**Phase 3: Full System Training (2-3 weeks)**
- Fine-tune full system including LoRA on video decoder
- Objective: End-to-end video generation with hybrid conditioning
- Purpose: Optimize for video generation quality
- Success criteria: LPIPS < 0.30, Temporal LPIPS variance < 0.02

**Total additional training time: 6-7 weeks**

### 1.5 Inference Pipeline

```
1. Input video/image arrives
2. [Parallel] Extract VLM features (Qwen2.5-VL forward pass)
3. [Parallel] Extract spatial features (DINOv2 forward pass)
4. Fusion module combines streams -> conditioning signal
5. LTX-Video generates prediction conditioned on fused features
6. [Optional] Verification module compares to actual outcome
```

**Latency estimate:**
- VLM encoding: ~400ms (existing)
- DINOv2 encoding: ~150ms (additional)
- Fusion module: ~50ms (additional)
- Video generation: ~800ms (existing)
- **Total: ~1.4s (vs ~1.2s baseline, +17% latency)**

---

## 2. Pros and Cons

### 2.1 Advantages

| Advantage | Impact | Confidence |
|-----------|--------|------------|
| **Addresses root cause** | Directly solves the spatial information bottleneck identified in Q2 | High |
| **Preserves VLM strengths** | Maintains semantic understanding, temporal reasoning, and world knowledge | High |
| **Modular design** | Can independently validate each pathway before integration | High |
| **Proven components** | DINOv2 is SOTA for dense prediction; fusion architectures are well-understood | Medium-High |
| **Graceful degradation** | If one stream fails, the other may still provide value | Medium |
| **Future-proof** | Architecture can accommodate better spatial encoders as they emerge | Medium |
| **Interpretable** | Attention weights reveal which stream contributes to each output region | Medium |

### 2.2 Disadvantages

| Disadvantage | Impact | Mitigation |
|--------------|--------|------------|
| **Increased complexity** | Two encoders, fusion module, more failure modes | Modular testing; clear interfaces |
| **Higher compute cost** | ~17% latency increase, ~2x VRAM for encoding | DINOv2 is efficient; can share GPU with VLM |
| **More trainable parameters** | 15-25M additional (fusion module) | Still within 50M budget |
| **Extended timeline** | 4-6 weeks additional development | Parallelizable with some existing experiments |
| **Potential information redundancy** | Both encoders see same pixels | Orthogonal training objectives should minimize |
| **Fusion module learning difficulty** | Must learn to combine heterogeneous representations | Well-studied problem; cross-attention is effective |
| **Debugging complexity** | Harder to attribute failures to specific components | Ablation studies; visualization tools |

### 2.3 Risk-Adjusted Assessment

The hybrid approach trades **known architecture complexity** for **reduced research risk**. The current single-encoder approach has a demonstrated fundamental limitation (spatial IoU = 0.559, mAP@0.5 = 0.001). Continuing down this path requires either:

1. Accepting degraded spatial accuracy (undermines verification hypothesis)
2. Hoping VLM spatial representations improve with different extraction points (low probability given Q2 results showing spatial loss is pervasive)
3. Training a new VLM from scratch with spatial preservation objectives (prohibitive cost)

The hybrid approach offers a well-understood solution with proven components. The primary risks are engineering complexity and timeline extension, both of which are manageable.

---

## 3. Resource Requirements

### 3.1 Compute Resources

| Resource | Current Usage | Additional for Hybrid | Total |
|----------|---------------|----------------------|-------|
| **GPU VRAM (inference)** | ~27GB | +6GB (DINOv2) | ~33GB |
| **GPU VRAM (training)** | ~35GB | +8GB | ~43GB |
| **GPU-hours (experiments)** | 400 estimated | +200 | ~600 |
| **Storage** | 500GB | +100GB (DINOv2 features) | ~600GB |

**Hardware recommendation:** A100-80GB for training; A100-40GB or 2x A10-24GB for inference.

### 3.2 Time Estimates

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Architecture implementation | 1 week | None |
| Phase 1: Spatial reconstruction | 2 weeks | Implementation complete |
| Phase 2: Semantic integration | 2 weeks | Phase 1 success |
| Phase 3: Full system training | 2-3 weeks | Phase 2 success |
| Integration with remaining experiments | 1 week | Phase 3 success |
| **Total additional time** | **6-7 weeks** | |

### 3.3 Personnel Requirements

| Role | Effort | Current Availability |
|------|--------|---------------------|
| Research Engineer | 1 FTE for 6 weeks | Available |
| Research Scientist | 0.5 FTE for 6 weeks | Available |
| Infrastructure support | 0.25 FTE | Available |

### 3.4 Financial Estimate

| Item | Cost |
|------|------|
| Compute (600 GPU-hours @ $2/hr) | $1,200 |
| Storage (100GB x 6 months) | $50 |
| Personnel (already budgeted) | $0 marginal |
| **Total incremental cost** | **~$1,250** |

---

## 4. Risk Assessment

### 4.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **DINOv2 features don't improve spatial metrics** | Low (15%) | High | DINOv2 is proven for dense prediction; early validation in Phase 1 |
| **Fusion module fails to combine streams effectively** | Medium (30%) | High | Use proven architectures (Q-Former, DETR); extensive ablations |
| **Increased latency unacceptable** | Low (10%) | Medium | DINOv2-base (smaller variant) achieves similar quality at lower cost |
| **Training instability with multiple encoders** | Medium (25%) | Medium | Staged training; gradient scaling; careful initialization |
| **VLM and spatial features interfere** | Low (20%) | Medium | Gating mechanisms; stream-specific projections |

### 4.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Phase 1 takes longer than expected** | Medium (30%) | Medium | Clear success criteria; quick pivots if metrics don't improve |
| **Integration issues with existing code** | Medium (25%) | Low | Clean interfaces; comprehensive testing |
| **Unforeseen dependencies block progress** | Low (15%) | High | Parallel workstreams; early integration testing |

### 4.3 Worst-Case Scenario

If the hybrid approach fails (estimated 20% probability), we will have:
- Validated that spatial encoding alone is insufficient (Phase 1 failure)
- OR validated that semantic-spatial fusion is the bottleneck (Phase 2 failure)
- Detailed diagnostic data to inform next pivot

**Fallback options:**
1. Use hybrid only for spatial-critical tasks, VLM-only for semantic tasks
2. Investigate alternative video decoders more tolerant of imprecise conditioning
3. Reframe research questions to focus on semantic prediction (de-emphasize spatial accuracy)

### 4.4 Go/No-Go Decision Points

| Checkpoint | Criteria | Timeline | Decision |
|------------|----------|----------|----------|
| **Phase 1 Complete** | Spatial IoU > 0.7 from DINOv2 alone | Week 3 | Continue / Abort |
| **Phase 2 Complete** | Combined metrics exceed single-stream | Week 5 | Continue / Simplify |
| **Phase 3 Complete** | Full success criteria met | Week 7 | Proceed / Partial integration |

---

## 5. Implications for Remaining Experiments

The hybrid architecture affects each remaining experiment differently:

### 5.1 C2: Small Adapter Can Bridge Latent Spaces

**Impact: Significant modification required**

- **Original plan**: Train adapter from VLM latents to LTX-Video
- **New plan**: Train fusion module from (VLM + DINOv2) to LTX-Video
- **Changes needed**:
  - Replace E2.1-E2.4 (single-stream adapters) with fusion module experiments
  - Add ablation: VLM-only vs DINOv2-only vs hybrid
  - Update scaling study for fusion module size

**Timeline impact**: +1-2 weeks (fusion module is more complex than single adapter)

### 5.2 Q3: Temporal Coherence

**Impact: Minimal modification**

- **Original concern**: Conditioning injection disrupts temporal dynamics
- **Hybrid advantage**: Spatial features are per-frame, naturally temporally consistent
- **Changes needed**:
  - Add tests for spatial-temporal interaction
  - Verify DINOv2 features don't introduce flickering
  - Same evaluation metrics apply

**Timeline impact**: Neutral (may actually be easier due to cleaner spatial signal)

### 5.3 C3: VLM Future State Prediction in Latent Space

**Impact: Moderate modification**

- **Original plan**: Query tokens extract future state from VLM
- **New plan**: Query tokens extract semantic future; spatial features inform current state
- **Key question**: Can VLM predict *semantic* changes while spatial encoder tracks current state?
- **Changes needed**:
  - Separate semantic prediction (VLM) from spatial grounding (DINOv2)
  - May simplify the task: VLM predicts "what happens", spatial encoder shows "where"
  - Redefine success metrics for semantic vs spatial prediction

**Timeline impact**: +1 week (reframing and additional ablations)

### 5.4 Q4: Training Data Requirements

**Impact: Moderate increase**

- **Original plan**: ~100K video-action pairs
- **New plan**: Same data, but must pre-extract DINOv2 features
- **Changes needed**:
  - Feature extraction preprocessing step
  - Larger storage requirement
  - Potentially different data efficiency (fusion may need more data)

**Timeline impact**: +0.5 weeks (preprocessing), data requirements TBD

### 5.5 Q5: Prediction Horizon

**Impact: Minimal modification**

- **Original plan**: Find optimal prediction horizon
- **New consideration**: Spatial vs semantic degradation may have different horizon profiles
- **Changes needed**:
  - Evaluate spatial accuracy vs horizon (in addition to semantic)
  - May find spatial accuracy determines practical horizon limit

**Timeline impact**: Neutral

### 5.6 C4: Pixel Verification Improves Accuracy

**Impact: Potentially positive**

- **Original concern**: LPIPS may not correlate with correctness
- **Hybrid advantage**: Better spatial accuracy means clearer verification signal
- **Key insight**: If spatial predictions are accurate, LPIPS becomes more meaningful
- **Changes needed**: None (may actually improve success probability)

**Timeline impact**: Neutral (likely easier with better spatial quality)

### 5.7 Summary Table

| Experiment | Modification Level | Timeline Impact | Risk Change |
|------------|-------------------|-----------------|-------------|
| C2 | High | +1-2 weeks | Reduced (clearer design) |
| Q3 | Low | Neutral | Neutral |
| C3 | Medium | +1 week | Reduced (simpler task) |
| Q4 | Medium | +0.5 weeks | Neutral |
| Q5 | Low | Neutral | Neutral |
| C4 | Low | Neutral | Reduced (better signal) |

---

## 6. Success Metrics

### 6.1 Phase 1: Spatial Reconstruction

| Metric | Target | Failure Threshold | Measurement |
|--------|--------|-------------------|-------------|
| **Spatial IoU** | > 0.75 | < 0.65 | Object localization accuracy |
| **LPIPS (DINOv2 only)** | < 0.25 | > 0.35 | Perceptual reconstruction quality |
| **Bounding Box IoU** | > 0.50 | < 0.30 | Q2-style detection probe |
| **Edge F1** | > 0.65 | < 0.50 | Edge preservation |

### 6.2 Phase 2: Semantic Integration

| Metric | Target | Failure Threshold | Measurement |
|--------|--------|-------------------|-------------|
| **Spatial IoU** | > 0.70 | < 0.60 | Maintain spatial quality |
| **LPIPS (hybrid)** | < 0.28 | > 0.35 | Combined perceptual quality |
| **Temporal ordering accuracy** | > 85% | < 75% | Maintain VLM temporal strength |
| **Semantic consistency** | > 0.80 | < 0.65 | VLM caption similarity |

### 6.3 Phase 3: Full System

| Metric | Target | Failure Threshold | Measurement |
|--------|--------|-------------------|-------------|
| **LPIPS (video)** | < 0.30 | > 0.40 | Video reconstruction quality |
| **FVD** | < 150 | > 250 | Video distribution quality |
| **Spatial IoU** | > 0.65 | < 0.55 | Maintained spatial accuracy |
| **Temporal LPIPS variance** | < 0.02 | > 0.04 | Temporal consistency |
| **Latency** | < 1.5s | > 2.5s | Inference speed |

### 6.4 Overall Project Success

The hybrid approach is successful if:

1. **Gate 1 criteria are met**: Spatial IoU > 0.6, LPIPS < 0.35 (enabling Phase 2)
2. **No significant regression**: VLM temporal/semantic strengths maintained
3. **Practical deployment**: Latency increase < 50%, VRAM < 48GB

---

## 7. Recommendation

### 7.1 Assessment Summary

| Factor | Rating | Notes |
|--------|--------|-------|
| **Technical feasibility** | High | Proven components, well-understood architecture |
| **Addresses core problem** | High | Directly targets spatial information bottleneck |
| **Resource requirements** | Moderate | +$1,250 compute, +6 weeks, existing personnel |
| **Risk level** | Medium | 20% failure probability, clear fallback options |
| **Impact on project timeline** | Moderate | +4-6 weeks net (some parallel work possible) |
| **Alternative options** | Limited | Other pivots (new video decoder, new VLM) are more expensive |

### 7.2 Comparison to Alternatives

| Pivot Option | Timeline | Cost | Technical Risk | Addresses Root Cause |
|--------------|----------|------|----------------|---------------------|
| **Hybrid Encoder (this proposal)** | +6 weeks | +$1.2K | Medium | Yes - directly |
| New Video Decoder | +8 weeks | +$3K | High | Partially |
| Fine-tune VLM for spatial | +12 weeks | +$20K | Very High | Yes - but expensive |
| Accept spatial limitation | +0 weeks | $0 | Low | No |
| Extract pre-merge VLM features | +2 weeks | +$500 | Medium | Unlikely (Q2 showed spatial loss is early) |

### 7.3 Recommendation

**We recommend proceeding with the hybrid encoder architecture.**

The Gate 1 findings are clear: VLM latent spaces fundamentally lack spatial information required for precise video generation. Q2 demonstrated this is not an extraction-point issue--spatial information is lost throughout the VLM pipeline. Continuing with the current architecture has a high probability of failure at Gate 2 and beyond.

The hybrid approach:
- Uses proven components (DINOv2 is SOTA for dense prediction)
- Has a well-understood architecture (cross-attention fusion is established)
- Preserves the VLM's demonstrated strengths (temporal reasoning, semantic understanding)
- Provides clear validation checkpoints (phased implementation)
- Has manageable resource requirements (+6 weeks, +$1.2K)

The 20% failure probability is acceptable given:
- Early detection via Phase 1 checkpoint (Week 3)
- Diagnostic value even in failure case
- No better alternatives at comparable cost

### 7.4 Requested Decision

We request approval to:

1. **Pause C2 adapter experiments** until hybrid architecture is validated
2. **Implement hybrid architecture** per the phased plan (6-7 weeks)
3. **Allocate additional compute budget** ($1,250)
4. **Revise project timeline** to account for pivot (+4-6 weeks net)

If approved, implementation begins immediately with a Phase 1 checkpoint at Week 3 to confirm or abort the approach.

---

## Appendix A: Detailed DINOv2 Analysis

### A.1 Why DINOv2 Preserves Spatial Information

DINOv2 was trained with a combination of:

1. **Self-distillation (DINO)**: Student must predict teacher's representation of augmented views, forcing consistent spatial understanding across crops and transforms.

2. **Masked image modeling (iBOT)**: Model must reconstruct masked patches from context, requiring precise spatial reasoning.

3. **Large-scale curation**: Trained on 142M images curated for visual diversity, not language alignment.

The result: features that spontaneously encode object boundaries, relative positions, and fine-grained details--precisely what Qwen2.5-VL lacks.

### A.2 DINOv2 for Dense Prediction

Published results on standard benchmarks:

| Task | Dataset | DINOv2 Performance | Notes |
|------|---------|-------------------|-------|
| Depth estimation | NYUv2 | RMSE 0.279 | Linear probe only |
| Semantic segmentation | ADE20K | mIoU 49.0 | Linear probe only |
| Object detection | COCO | 47.4 mAP | ViTDet integration |
| Instance segmentation | COCO | 40.3 mAP | ViTDet integration |

These results demonstrate that DINOv2 features contain rich spatial information extractable by simple probes--exactly what we need.

### A.3 Feature Comparison: VLM vs DINOv2

| Property | Qwen2.5-VL | DINOv2-giant |
|----------|-----------|--------------|
| Training objective | Language-vision alignment | Self-supervised visual learning |
| Spatial preservation | Poor (IoU ~0.1) | Excellent (enables SOTA dense prediction) |
| Semantic understanding | Excellent | Good (weaker than VLM) |
| Temporal reasoning | Excellent | N/A (image-only) |
| Resolution | Post-merge: H/28 x W/28 | H/14 x W/14 |
| Dimension | 3584 | 1536 |

---

## Appendix B: Alternative Spatial Encoders

### B.1 SAM ViT-H

**Pros:**
- Explicitly trained for segmentation
- Very strong edge detection
- Efficient (designed for interactive use)

**Cons:**
- Lower resolution output (256-dim)
- Trained only for segmentation, not reconstruction
- May not capture fine texture details

**Verdict:** Strong alternative if DINOv2 proves insufficient. Worth testing in ablations.

### B.2 EVA-CLIP

**Pros:**
- Combined CLIP alignment + strong visual features
- Bridges semantic and spatial

**Cons:**
- Still language-aligned (may share VLM weaknesses)
- Less proven for dense prediction than DINOv2

**Verdict:** Second-tier option. Consider if hybrid needs more semantic grounding.

### B.3 MAE (Masked Autoencoder)

**Pros:**
- Trained for reconstruction (directly relevant)
- Good spatial features

**Cons:**
- Weaker than DINOv2 on dense prediction benchmarks
- Less semantic awareness

**Verdict:** Worth testing if DINOv2 + VLM has redundancy issues.

---

## Appendix C: Fusion Architecture Alternatives

### C.1 Concatenation Baseline

```python
def concat_fusion(vlm_features, spatial_features):
    combined = torch.cat([vlm_features.mean(1), spatial_features.mean(1)], dim=-1)
    return self.projection(combined)
```

**Pros:** Simple, fast
**Cons:** No interaction between streams; loses spatial structure

### C.2 FiLM Conditioning

```python
def film_fusion(vlm_features, spatial_features):
    gamma, beta = self.film_generator(vlm_features)  # Predict scale and shift
    return gamma * spatial_features + beta
```

**Pros:** VLM modulates spatial features
**Cons:** Asymmetric; may underutilize VLM information

### C.3 Perceiver-Style

```python
def perceiver_fusion(vlm_features, spatial_features, latent_queries):
    # Iterative cross-attention
    for _ in range(num_iterations):
        latent_queries = cross_attend(latent_queries, spatial_features)
        latent_queries = cross_attend(latent_queries, vlm_features)
    return latent_queries
```

**Pros:** Flexible; handles arbitrary input lengths
**Cons:** More complex; slower

### C.4 Recommended: Q-Former Style (Primary Proposal)

Selected for balance of expressiveness, efficiency, and proven effectiveness in multimodal fusion (BLIP-2, InstructBLIP).

---

## Appendix D: Computational Cost Breakdown

### D.1 Inference Cost

| Component | Time (ms) | VRAM (GB) | Notes |
|-----------|-----------|-----------|-------|
| Qwen2.5-VL encoding | 400 | 15 | Existing |
| DINOv2-giant encoding | 150 | 6 | New |
| Fusion module | 50 | 2 | New |
| LTX-Video generation | 800 | 8 | Existing |
| **Total** | **1400** | **31** | +17% latency |

### D.2 Training Cost

| Phase | GPU-hours | Storage (GB) | Notes |
|-------|-----------|--------------|-------|
| Phase 1 | 80 | 50 | DINOv2 features pre-extracted |
| Phase 2 | 100 | 50 | VLM features added |
| Phase 3 | 120 | 100 | Full video training |
| **Total** | **300** | **200** | |

---

## References

1. Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193.

2. Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML.

3. Wang, P., et al. (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv.

4. Carion, N., et al. (2020). End-to-End Object Detection with Transformers (DETR). ECCV.

5. HaCohen, Y., et al. (2024). LTX-Video: Realtime Video Latent Diffusion. arXiv:2501.00103.

6. Kirillov, A., et al. (2023). Segment Anything. ICCV.

---

*Document prepared for department chair review. Questions and feedback welcome.*

*Last updated: 2026-01-18*
