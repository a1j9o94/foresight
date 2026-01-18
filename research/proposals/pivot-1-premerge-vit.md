# Research Proposal: Pre-Merge ViT Features for Spatial Preservation

**Proposal ID:** PIVOT-1
**Date:** 2026-01-18
**Author:** Foresight Research Team
**Status:** Draft for Review
**Decision Required By:** [To be determined by department chair]

---

## Executive Summary

Gate 1 experiments have revealed a fundamental architectural constraint in the Foresight project: Qwen2.5-VL's vision processing pipeline destroys spatial information before features reach the point where we extract them for video generation. Specifically, the 2x2 token merger and subsequent LLM layers reduce bounding box IoU from an already-poor 0.101 (pre-merge) to 0.103 (post-merge), with detection mAP@0.5 collapsing to 0.001 -- effectively random performance. This spatial degradation prevents the generation of videos with accurate object positioning, a critical requirement for the verification-based reasoning that differentiates Foresight from latent-only approaches like V-JEPA.

This proposal evaluates a pivot strategy: **bypassing the VLM's token merger entirely and extracting features directly from the Vision Transformer (ViT) backbone** before spatial compression occurs. The approach would use raw ViT patch embeddings (at their native ~14x14 pixel resolution) for spatial tasks while potentially retaining post-merge features for semantic/temporal tasks where spatial precision is less critical.

Our analysis indicates this pivot is **technically feasible but architecturally complex**, requiring careful integration to avoid losing the semantic richness that post-merge features provide. We estimate 6-10 weeks of additional development with moderate risk. We recommend proceeding with a controlled Phase 1 investigation (2-3 weeks) to validate the core assumption that raw ViT features preserve spatial information better than our current probes suggest, before committing to full implementation.

---

## 1. Problem Statement

### 1.1 Current Architecture Limitation

The Foresight architecture extracts visual features from Qwen2.5-VL's intermediate representations to condition video generation. Our Gate 1 experiments (C1, Q1, Q2) have characterized where and how information flows through the VLM:

```
Input Image (448x672)
    |
    v
ViT Patch Embedding (14x14 patches) --> 1024 tokens (32x32 grid equivalent)
    |
    v
[WHERE WE EXPECTED TO EXTRACT PRE-MERGE]
    |
    v
2x2 Token Merger --> 256 tokens (16x16 grid)
    |
    v
LLM Layers (28 transformer blocks)
    |
    v
[WHERE WE CURRENTLY EXTRACT POST-MERGE]
```

### 1.2 Experimental Evidence

**Q2 Experiment Results (Information Preservation):**

| Extraction Point | Bbox IoU | mAP@0.5 | LPIPS | Verdict |
|-----------------|----------|---------|-------|---------|
| Pre-merge ViT | 0.101 | - | 1.542 | FAIL |
| Post-merge | 0.103 | 0.001 | 0.087 | FAIL (spatial), PASS (perceptual) |
| LLM Layer 0 | 0.066 | - | 0.099 | FAIL |
| LLM Layer 13 | 0.094 | - | - | FAIL |
| LLM Layer 27 | 0.042 | - | - | FAIL |

**Key Finding:** Spatial information is severely degraded at ALL measured extraction points. Even pre-merge features show only 0.101 IoU, suggesting the issue may originate earlier than the token merger -- potentially in the ViT training objective itself (semantic understanding prioritized over spatial preservation).

**Positive Findings:**
- Temporal information is excellently preserved (100% direction accuracy, 90% ordering)
- Perceptual reconstruction quality is strong (LPIPS = 0.087 post-merge)
- Latent space alignment between VLM and video decoder is viable (CKA = 0.687 from Q1)

### 1.3 Core Tension

The VLM was trained for **semantic understanding** (visual question answering, image captioning), not **spatial preservation**. The architecture aggressively compresses spatial dimensions to increase the effective context window for language modeling. This design choice is antithetical to our requirement for pixel-accurate video generation.

---

## 2. Technical Approach

### 2.1 Proposed Architecture Change

We propose a **dual-stream extraction** architecture that separates spatial and semantic feature pathways:

```
Input Image/Video
       |
       v
  Qwen2.5-VL ViT Backbone (frozen)
       |
       +---> Raw ViT Patches (1024 tokens) ---> Spatial Stream
       |         |                                    |
       |         v                                    |
       |    [BYPASS MERGER]                           |
       |                                              |
       +---> 2x2 Token Merger --> LLM --> Semantic Stream
                                              |
                                              v
                                    [MERGE FEATURES]
                                              |
                                              v
                                    Conditioning Adapter
                                              |
                                              v
                                    Video Decoder (LTX-Video)
```

### 2.2 Implementation Options

**Option A: Pure Pre-Merge Extraction**
- Extract only raw ViT patch embeddings (before any merging)
- Dimension: [B, 1024, 1536] for a 448x672 image
- 4x more tokens than current approach
- Requires complete adapter redesign

**Option B: Hierarchical Dual-Stream (Recommended)**
- Extract raw ViT patches for spatial positioning
- Extract post-merge/LLM features for semantic context
- Fuse via cross-attention or concatenation
- Leverages both streams' strengths

**Option C: Multi-Scale ViT Extraction**
- Extract from multiple ViT layers (early, middle, late)
- Weight combination learned during adapter training
- Similar to feature pyramid networks in detection

### 2.3 Technical Implementation Details

**Extracting Pre-Merge Features from Qwen2.5-VL:**

```python
# Current extraction (post-merge)
outputs = model(**inputs, output_hidden_states=True)
post_merge_features = outputs.hidden_states[target_layer]

# Proposed extraction (pre-merge)
# Requires modifying forward pass or hooking into vision encoder
def extract_vit_features(model, pixel_values):
    """
    Extract features BEFORE the spatial merger.
    Qwen2.5-VL structure: visual -> merger -> LLM
    """
    # Access the vision encoder directly
    vision_encoder = model.visual  # Qwen2VisionTransformerPretrainedModel

    # Get patch embeddings from ViT
    # This bypasses the merger that normally follows
    patch_embeds = vision_encoder.patch_embed(pixel_values)

    # Run through ViT transformer blocks
    hidden_states = vision_encoder.blocks(patch_embeds)  # [B, num_patches, hidden_dim]

    return hidden_states  # Shape: [B, H*W/14^2, 1536]
```

**Challenges:**
1. Qwen2.5-VL's vision encoder is tightly coupled with the merger
2. May require custom forward pass implementation
3. Flash attention optimizations may not be available for modified path
4. Memory usage increases 4x due to unmerged tokens

### 2.4 Adapter Architecture Modifications

**Current Adapter (for 256 tokens):**
```python
class CurrentAdapter(nn.Module):
    def __init__(self):
        self.proj = nn.Sequential(
            nn.Linear(1536, 2048),
            nn.GELU(),
            nn.Linear(2048, 128)  # LTX-Video channels
        )
        # ~3.5M parameters
```

**Proposed Dual-Stream Adapter:**
```python
class DualStreamAdapter(nn.Module):
    def __init__(self):
        # Spatial stream (1024 tokens)
        self.spatial_proj = nn.Sequential(
            nn.Linear(1536, 512),
            nn.GELU(),
            nn.Linear(512, 64)
        )

        # Semantic stream (256 tokens, from LLM)
        self.semantic_proj = nn.Sequential(
            nn.Linear(3584, 1024),  # LLM hidden dim
            nn.GELU(),
            nn.Linear(1024, 64)
        )

        # Fusion via cross-attention
        self.fusion = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            batch_first=True
        )

        # Final projection to LTX-Video space
        self.output_proj = nn.Linear(64, 128)

        # ~8-12M parameters total
```

---

## 3. Advantages and Disadvantages

### 3.1 Advantages

| Advantage | Impact | Confidence |
|-----------|--------|------------|
| **Preserves spatial resolution** | 4x more spatial tokens available for positioning | Medium - depends on whether ViT itself preserves spatial info |
| **Maintains semantic capability** | Dual-stream retains LLM's reasoning features | High - proven in C1/Q1 |
| **Minimal VLM modification** | Uses existing components, just different extraction point | Medium - requires careful implementation |
| **Backward compatible** | Can fall back to current approach if spatial stream fails | High |
| **Interpretable** | Can visualize which stream contributes to predictions | Medium |
| **Leverages existing infrastructure** | Uses same video decoder, similar training pipeline | High |

### 3.2 Disadvantages

| Disadvantage | Impact | Mitigation |
|--------------|--------|------------|
| **4x memory increase** | 1024 vs 256 tokens per frame | Gradient checkpointing, mixed precision |
| **Increased compute cost** | More tokens through adapter | Efficient attention, token pruning |
| **Architectural complexity** | Dual-stream fusion is non-trivial | Start with simple fusion, iterate |
| **ViT may not preserve spatial info** | Our probe showed 0.101 IoU even pre-merge | Investigate earlier ViT layers |
| **Training instability risk** | Two feature streams with different characteristics | Careful loss weighting, staged training |
| **Inference latency** | Additional forward pass through spatial stream | Can be parallelized with semantic stream |
| **Uncertain benefit** | Pre-merge IoU was only marginally better | Phase 1 validation critical |

### 3.3 Risk-Weighted Assessment

**Probability of Success: 45-60%**

The key uncertainty is whether the ViT backbone itself preserves spatial information at any layer. Our Q2 probe used a linear decoder, which may have been too weak to extract spatial information that a more sophisticated adapter could access. However, the mAP@0.5 of 0.001 with a DETR-style detector is concerning -- this was a capable architecture that still failed.

---

## 4. Resource Requirements

### 4.1 Compute Resources

| Phase | GPU-Hours | GPU Type | Storage | Timeline |
|-------|-----------|----------|---------|----------|
| Phase 1: Validation | 100-150 | A100-40GB | 200GB | 2-3 weeks |
| Phase 2: Implementation | 300-500 | A100-80GB (dual) | 500GB | 3-4 weeks |
| Phase 3: Training & Eval | 500-800 | A100-80GB (4x) | 1TB | 3-4 weeks |
| **Total** | **900-1450** | - | 1TB | **8-11 weeks** |

**Cloud Cost Estimate:** $4,000 - $7,000 (at ~$3/GPU-hour for A100-80GB)

### 4.2 Personnel Requirements

| Role | FTE | Duration | Responsibilities |
|------|-----|----------|------------------|
| ML Research Engineer | 1.0 | 8-11 weeks | Implementation, experimentation |
| Research Scientist | 0.5 | 8-11 weeks | Analysis, pivots, paper writing |
| Infrastructure Engineer | 0.25 | 4 weeks | Training pipeline, monitoring |
| **Total** | **1.75 FTE** | **8-11 weeks** | - |

### 4.3 Data Requirements

No additional data collection required. Existing datasets suffice:
- Something-Something v2 (already in use)
- COIN (for procedural validation)
- Synthetic spatial benchmark (can generate)

---

## 5. Risk Assessment

### 5.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **ViT fundamentally lacks spatial info** | Medium (40%) | Critical | Early validation in Phase 1; pivot to external spatial model if confirmed |
| **Memory constraints block training** | Medium (30%) | High | Gradient checkpointing, model parallelism, token pruning |
| **Fusion degrades semantic quality** | Low (20%) | Medium | Gated fusion allowing model to ignore spatial stream |
| **Training instability** | Medium (35%) | Medium | Staged training, loss weighting tuning |
| **Inference too slow** | Low (15%) | Medium | Parallel extraction, distillation |

### 5.2 Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Phase 1 inconclusive** | Medium (35%) | High | Define clear go/no-go criteria upfront |
| **Qwen2.5-VL internals harder to modify** | Medium (30%) | Medium | Allocate buffer time, engage HuggingFace community |
| **Hyperparameter search takes longer** | High (50%) | Low | Automated sweeps, narrow search space based on prior work |

### 5.3 Opportunity Cost

Pursuing this pivot delays:
- C2 (Adapter Bridging): 6-10 weeks
- Q3 (Temporal Coherence): 6-10 weeks (depends on C2)
- C3 (Future Prediction): 8-12 weeks
- C4 (Pixel Verification): 10-14 weeks
- Full system evaluation: 12-16 weeks

**Alternative use of resources:**
- Could instead pursue hybrid VLM + dedicated spatial model approach
- Could invest in better video decoder instead of better spatial features
- Could reduce scope to semantic-only predictions (abandoning spatial verification)

---

## 6. Implications for Remaining Experiments

### 6.1 Impact on C2 (Adapter Bridging)

**Current Plan:** Train small adapter (5-10M params) to bridge VLM and video decoder latent spaces.

**Impact of Pivot:**
- **Delay:** 6-10 weeks (must wait for new feature extraction)
- **Scope Change:** Adapter architecture significantly more complex (dual-stream)
- **Parameter Budget:** May increase from 10M to 15-25M parameters
- **Success Criteria:** Need to add spatial IoU > 0.6 as explicit criterion

**Recommendation:** Defer C2 until Phase 1 validation complete.

### 6.2 Impact on Q3 (Temporal Coherence)

**Current Plan:** Ensure generated videos maintain temporal consistency.

**Impact of Pivot:**
- **Delay:** Inherits C2 delay
- **New Risk:** Dual-stream features may introduce temporal inconsistency between streams
- **Opportunity:** Pre-merge features may contain better temporal signal (closer to raw video)

**Recommendation:** Add temporal consistency test to Phase 1 validation.

### 6.3 Impact on C3 (Future Prediction)

**Current Plan:** Train query tokens to predict future latent states.

**Impact of Pivot:**
- **Delay:** Significant (8-12 weeks)
- **Scope Change:** Query tokens must predict both spatial and semantic streams
- **Complexity Increase:** Predicting 4x more spatial tokens is harder
- **Potential Simplification:** Could predict only semantic stream, use spatial stream for current frame only

**Recommendation:** Consider asymmetric design where spatial stream is reconstruction-only (not prediction).

### 6.4 Impact on Q4 (Training Data) and Q5 (Prediction Horizon)

**Q4 Impact:** Minimal change to data requirements, but training efficiency may differ.

**Q5 Impact:** Spatial precision may enable longer useful prediction horizons (can verify position drift earlier).

**Recommendation:** Proceed with planning but delay execution.

### 6.5 Impact on C4 (Pixel Verification)

**Current Plan:** Compare predicted video to actual outcomes using perceptual metrics.

**Impact of Pivot:**
- **Benefit:** Better spatial features should improve verification signal
- **New Metric:** Spatial IoU becomes viable verification metric
- **Delay:** Inherits full pipeline delay

**Recommendation:** This pivot is specifically designed to enable C4's spatial verification. If pivot fails, reconsider C4 scope (focus on semantic verification only).

### 6.6 Decision Tree

```
PHASE 1 RESULT
    |
    +---> Spatial IoU > 0.4 (pre-merge ViT) ---> PROCEED with dual-stream
    |
    +---> Spatial IoU < 0.2 (pre-merge ViT) ---> ABORT pivot
    |         |
    |         +---> Consider external spatial model (OwlViT, YOLO)
    |         +---> Or reduce scope to semantic-only predictions
    |
    +---> Spatial IoU 0.2-0.4 ---> INVESTIGATE deeper
              |
              +---> Try earlier ViT layers
              +---> Try stronger probe architecture
              +---> If still fails after 2 weeks, ABORT
```

---

## 7. Success Metrics

### 7.1 Phase 1 Validation Criteria (Go/No-Go)

| Metric | Go Threshold | Marginal | No-Go |
|--------|--------------|----------|-------|
| **Spatial Bbox IoU** (pre-merge ViT) | > 0.4 | 0.2 - 0.4 | < 0.2 |
| **mAP@0.5** (detection probe) | > 0.15 | 0.05 - 0.15 | < 0.05 |
| **LPIPS** (reconstruction) | < 0.35 | 0.35 - 0.45 | > 0.45 |
| **Memory Overhead** | < 2x baseline | 2x - 3x | > 3x |

**Decision Rule:** Proceed to Phase 2 only if Spatial Bbox IoU > 0.4 AND mAP@0.5 > 0.15.

### 7.2 Phase 2 Implementation Criteria

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| **Spatial IoU** (dual-stream adapter) | > 0.6 | > 0.5 | < 0.4 |
| **LPIPS** (full pipeline) | < 0.30 | < 0.35 | > 0.40 |
| **Parameter Efficiency** | 15M achieves target | 25M achieves target | > 50M required |
| **Training Convergence** | Stable, < 50K steps | < 100K steps | Divergent |

### 7.3 Phase 3 System-Level Criteria

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| **Spatial IoU on test set** | > 0.65 | > 0.55 | < 0.50 |
| **Action Prediction Accuracy** (vs baseline) | +15% | +10% | < +5% |
| **Inference Latency** | < 5 seconds | < 8 seconds | > 10 seconds |
| **Verification Correlation** (LPIPS vs correctness) | r > 0.4 | r > 0.3 | r < 0.2 |

---

## 8. Recommendation

### 8.1 Overall Assessment

| Factor | Rating | Notes |
|--------|--------|-------|
| Technical Feasibility | Medium | Core uncertainty around ViT spatial preservation |
| Resource Availability | High | Within budget constraints |
| Strategic Alignment | High | Directly addresses Gate 1 failure |
| Risk Level | Medium-High | 40-55% failure probability |
| Opportunity Cost | Medium | 8-12 week delay to full system |

### 8.2 Recommendation

**Proceed with controlled Phase 1 investigation (2-3 weeks) before committing to full implementation.**

**Rationale:**
1. The Q2 experiment revealed a critical blocker that must be addressed for Foresight's core value proposition (spatial verification).
2. A 2-3 week Phase 1 investment is small compared to the overall project timeline.
3. Clear go/no-go criteria limit downside risk.
4. If Phase 1 fails, we have time to pursue alternatives (external spatial model, scope reduction).

**Alternative Recommendation (if Phase 1 fails):**

Pursue **Hybrid Architecture** using a dedicated spatial model:

```
Input Image
    |
    +---> Qwen2.5-VL ---> Semantic/Temporal features
    |
    +---> OwlViT/YOLO/DETR ---> Spatial features (bboxes, positions)
    |
    v
Fusion Adapter ---> Video Decoder
```

This hybrid approach was already identified as a pivot option in the Q2 findings and may be more robust if VLM spatial features are fundamentally limited.

### 8.3 Proposed Next Steps

**Week 1-2 (Phase 1a):**
1. Implement clean pre-merge ViT feature extraction from Qwen2.5-VL
2. Train spatial probe (linear + 2-layer MLP) on bounding box regression
3. Evaluate on SSv2 validation set

**Week 2-3 (Phase 1b):**
4. If spatial IoU < 0.4: Try earlier ViT layers (layer 6, 12, 18)
5. If still failing: Train DETR-style detection head (stronger probe)
6. Make go/no-go decision

**Week 3-4 (if Go):**
7. Begin dual-stream adapter implementation
8. Set up training infrastructure for larger memory footprint

**Week 3-4 (if No-Go):**
7. Evaluate hybrid architecture with OwlViT
8. Revise project scope and timeline

---

## 9. Appendices

### Appendix A: Relevant Q2 Experiment Data

```yaml
experiment_id: q2-information-preservation
status: completed
date: 2026-01-18

key_findings:
  - Pre-merge ViT features show 0.101 bounding box IoU (below 0.7 target)
  - Post-merge features show similar spatial performance (0.103 IoU)
  - LLM layers do not help (Layer 0: 0.066 IoU)
  - mAP@0.5 detection performance is essentially random (0.001)
  - Temporal information is well-preserved (100% direction accuracy)
  - Perceptual quality is excellent (LPIPS 0.087)

implication: "Spatial information is NOT preserved in Qwen2.5-VL embeddings.
This is a fundamental limitation of the model architecture, not just an
extraction point issue."

recommendation: "PIVOT - Use hybrid approach with dedicated spatial models"
```

### Appendix B: C1 Spatial IoU Discrepancy

The C1 experiment reported Spatial IoU = 0.567 (marginal pass), while Q2 reported IoU = 0.103 (clear fail). This discrepancy requires investigation:

**Possible Explanations:**
1. Different probe architectures (C1 used full reconstruction decoder, Q2 used linear probe)
2. Different evaluation methodology (C1 may have used softer IoU threshold)
3. Different test data distribution

**Resolution:** Phase 1 should include standardized evaluation using identical metrics across experiments.

### Appendix C: Memory Estimation

**Current Pipeline:**
- Post-merge tokens: 256 tokens x 1536 dim = 393K parameters per frame
- Batch of 8 frames: ~3.1M activations
- With gradients: ~12MB per frame

**Proposed Pipeline (Dual-Stream):**
- Pre-merge tokens: 1024 tokens x 1536 dim = 1.57M parameters per frame
- Post-merge tokens: 256 tokens x 3584 dim = 917K parameters per frame
- Combined: ~2.5M parameters per frame
- Batch of 8 frames: ~20M activations
- With gradients: ~80MB per frame

**Mitigation Strategies:**
1. Gradient checkpointing (2-3x memory reduction)
2. Mixed precision (bf16 throughout)
3. Sequential frame processing for spatial stream
4. Token pruning (keep top-k most informative spatial tokens)

### Appendix D: Related Literature

| Paper | Relevance | Key Insight |
|-------|-----------|-------------|
| DINOv2 (Oquab et al. 2023) | ViT spatial features | Self-supervised ViT preserves spatial info for dense prediction |
| SAM (Kirillov et al. 2023) | Spatial preservation | Dedicated encoder preserves fine spatial detail |
| ViTDet (Li et al. 2022) | ViT for detection | Simple feature pyramid from ViT backbone enables detection |
| Grounding DINO | VLM + spatial | Shows VLM features can be combined with spatial grounding |

**Key Takeaway:** Other work has successfully extracted spatial information from ViT backbones, suggesting the issue may be Qwen2.5-VL-specific (training objective or architecture) rather than fundamental to transformers.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-18 | Foresight Research Team | Initial proposal |

---

*This proposal is submitted for review by the department chair. We request feedback on:*
1. *Appropriateness of the 2-3 week Phase 1 investigation*
2. *Resource allocation priorities*
3. *Acceptable risk thresholds for proceeding to Phase 2*
4. *Preference between this pivot and the hybrid architecture alternative*
