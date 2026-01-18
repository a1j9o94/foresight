# Research Proposal: Alternative VLM Architectures for Spatial Information Preservation

**Proposal ID:** PIVOT-4
**Author:** Research Team
**Date:** 2026-01-18
**Status:** Draft for Department Review

---

## 1. Executive Summary

The Foresight project's Gate 1 experiments have revealed a fundamental architectural limitation: **Qwen2.5-VL's vision encoder destroys spatial information before it reaches the language model layers**. Our Q2 experiments demonstrated that bounding box IoU is only 0.104 and mAP@0.5 is 0.001--effectively random performance--for object localization tasks using VLM embeddings. The C1 experiment achieved Spatial IoU of 0.559, narrowly missing the 0.6 threshold. Critically, this spatial degradation occurs *before* the 2x2 token merger, indicating the problem is embedded in the vision encoder's training objective rather than a downstream processing artifact.

This spatial information loss poses a significant obstacle to our core hypothesis: that VLM latents can be decoded into accurate video predictions for verification-based reasoning. While our Q1 experiment confirms that semantic alignment between VLM and video decoder spaces is achievable (CKA=0.687), the lack of spatial precision means generated videos would fail to preserve object positions, making pixel-level verification unreliable.

We propose a systematic evaluation of alternative VLM architectures that may better preserve spatial information while maintaining the semantic reasoning capabilities essential for our approach. This pivot represents a calculated trade-off: abandoning 3-4 weeks of Qwen-specific integration work in exchange for potentially unblocking the entire research program. The decision to pivot should be made within 2 weeks, with full VLM replacement achievable in 6-8 weeks if warranted.

---

## 2. Technical Background

### 2.1 Problem Diagnosis: Where Spatial Information Is Lost

Our Q2 experiments traced spatial information through Qwen2.5-VL's processing pipeline:

```
Input Image (full spatial info)
      |
      v
Patch Embedding (14x14 patches) -----> IoU: ~0.10 (already degraded)
      |
      v
ViT Layers (28 transformer blocks)
      |
      v
2x2 Token Merger (4:1 compression) --> IoU: ~0.10 (no additional loss)
      |
      v
LLM Layers (28 transformer blocks) --> IoU: 0.04-0.10 (slight fluctuation)
```

**Key Finding:** Spatial information is not destroyed by the 2x2 merger or LLM layers--it is never properly encoded in the first place. The vision encoder was trained with a language modeling objective that prioritizes semantic discrimination over spatial preservation.

### 2.2 Architectural Features of Qwen2.5-VL

| Feature | Value | Spatial Impact |
|---------|-------|----------------|
| Vision encoder | ViT-based with M-RoPE | Positional encoding exists but semantic loss dominates |
| Patch size | 14x14 pixels | Standard resolution |
| Token merger | 2x2 averaging | 4x spatial compression (not the bottleneck) |
| Vision-language fusion | Late fusion after merger | No spatial grounding during fusion |
| Training objective | Next token prediction | Optimizes semantic, not spatial fidelity |

### 2.3 Why This Matters for Foresight

Our Generative Latent Prediction (GLP) architecture requires:

1. **Spatial accuracy:** Predicted videos must place objects correctly for verification
2. **Semantic richness:** VLM must understand scene context and action implications
3. **Temporal coherence:** Features must support consistent video generation

Qwen2.5-VL excels at #2 and potentially #3, but fundamentally fails at #1. Without spatial accuracy, our verification module (C4) cannot reliably compare predicted vs. actual outcomes--the core differentiator from latent-only approaches like V-JEPA.

---

## 3. Technical Approach

### 3.1 Candidate VLM Architectures

We propose evaluating four alternative architectures, selected for their differing approaches to vision-language fusion and spatial preservation:

#### 3.1.1 LLaVA-NeXT (LLaVA-1.6)

**Architecture:** Projects CLIP ViT features through a simple MLP into LLaMA/Vicuna, preserving all spatial tokens.

**Why it might preserve spatial info:**
- No token merging: All 576 tokens (24x24 grid for 336px) preserved
- CLIP encoder trained with contrastive loss that preserves some spatial structure
- Simple linear projection minimizes information loss
- Anyres processing maintains high-resolution details

**Variants to test:**
- LLaVA-NeXT-7B (Vicuna-7B backbone)
- LLaVA-NeXT-13B (Vicuna-13B backbone)
- LLaVA-NeXT-34B (Hermes-Yi-34B backbone) [if compute allows]

**Risk factors:**
- CLIP encoder also optimized for semantics, not spatial localization
- May have similar fundamental limitation at lower severity

#### 3.1.2 InternVL2

**Architecture:** Custom vision encoder (InternViT-6B) with progressive fusion.

**Why it might preserve spatial info:**
- 6B-parameter vision encoder (vs. ~300M in CLIP-based models)
- Trained with detection and segmentation auxiliary objectives
- Dynamic high-resolution support (up to 4K)
- Pixel shuffle for spatial upsampling during fusion

**Variants to test:**
- InternVL2-8B
- InternVL2-26B
- InternVL2-40B-MPO [if compute allows]

**Risk factors:**
- Larger model = longer inference time
- Custom architecture = more integration work
- May still lose spatial info in LLM layers

#### 3.1.3 CogVLM2

**Architecture:** Explicit spatial grounding through visual expert modules.

**Why it might preserve spatial info:**
- Visual expert attention in every LLM layer
- Explicit grounding capability (can output bounding boxes)
- Trained with visual grounding datasets
- Maintains separate visual and text processing streams

**Variants to test:**
- CogVLM2-19B
- CogVLM2-Video [for temporal considerations]

**Risk factors:**
- Grounding capability doesn't guarantee latent space preservation
- Larger parameter count
- Less community support than LLaVA

#### 3.1.4 Qwen2.5-VL with Modifications (Baseline++)

Before switching models entirely, we should test architectural modifications to Qwen:

**Modification A: Extract Pre-Merger Features**
- Already tested in Q2; did not help (IoU still ~0.10)
- Confirms problem is upstream of merger

**Modification B: Extract From Vision Encoder Directly**
- Bypass LLM entirely for spatial features
- Use LLM only for semantic/temporal reasoning
- Hybrid architecture with separate spatial pathway

**Modification C: Fine-tune Vision Encoder**
- Add spatial reconstruction objective
- Risk: May degrade semantic capabilities
- Requires significant compute

### 3.2 Evaluation Protocol

For each candidate VLM, we will run a compressed version of our C1 and Q2 experiments:

#### Phase 1: Quick Spatial Assessment (3 days per model)

```yaml
experiments:
  - name: spatial_probe
    description: Linear probe for object detection from VLM features
    metrics:
      - bbox_iou  # Target: > 0.5
      - mAP_0.5   # Target: > 0.2
      - mAP_0.75  # Target: > 0.1
    dataset: synthetic_shapes (same as Q2)

  - name: reconstruction_probe
    description: Can we reconstruct spatial layout from features?
    metrics:
      - spatial_iou  # Target: > 0.6
      - lpips        # Target: < 0.35
    dataset: synthetic_shapes
```

#### Phase 2: Full Assessment (1 week per model)

Only for models passing Phase 1:

```yaml
experiments:
  - name: latent_alignment
    description: CKA alignment with LTX-Video latents
    metrics:
      - cka_similarity  # Target: > 0.5

  - name: adapter_bridging
    description: Small adapter reconstruction test
    metrics:
      - lpips_10m_adapter  # Target: < 0.35

  - name: temporal_preservation
    description: Motion direction/ordering from features
    metrics:
      - direction_accuracy  # Target: > 80%
      - ordering_accuracy   # Target: > 75%
```

### 3.3 Decision Criteria

| Criterion | Weight | Threshold |
|-----------|--------|-----------|
| Spatial IoU (primary) | 40% | > 0.60 |
| Bbox IoU | 20% | > 0.50 |
| CKA alignment | 15% | > 0.50 |
| LPIPS reconstruction | 15% | < 0.35 |
| Inference speed | 10% | < 2x Qwen baseline |

**Decision matrix:**

| Outcome | Action |
|---------|--------|
| One model exceeds all thresholds | Switch to that model |
| Multiple models exceed thresholds | Select based on speed/simplicity |
| No model exceeds spatial threshold but one is >0.5 | Investigate hybrid approach |
| All models fail spatial threshold | Pivot to different architecture (see Section 7) |

---

## 4. Pros and Cons Analysis

### 4.1 Advantages of Pivoting to Alternative VLM

**Technical advantages:**
1. **Potentially unblocks entire research program** - If spatial information is fundamentally unavailable in Qwen, no amount of adapter engineering will fix it
2. **Better foundation for C2-C4** - Spatial accuracy directly impacts verification viability
3. **Reduced adapter complexity** - May need simpler bridging if source features are higher quality
4. **Community learnings available** - LLaVA, InternVL have extensive benchmarks and known characteristics

**Strategic advantages:**
1. **Early pivot is cheaper** - 4 weeks invested vs. potentially 14 weeks wasted
2. **De-risks core hypothesis** - Tests whether spatial preservation is possible with any VLM
3. **Generates valuable comparative data** - Understanding why spatial info is lost advances the field

### 4.2 Disadvantages of Pivoting

**Sunk costs:**
1. **3-4 weeks of Qwen-specific work** - Latent extraction pipelines, adapter prototypes, evaluation scripts
2. **Modal infrastructure built for Qwen** - Handler configurations, model loading code
3. **Familiarity and debugging time** - Team has learned Qwen's quirks

**New risks:**
1. **All alternatives may have same fundamental limitation** - VLMs optimized for QA, not spatial preservation
2. **Integration complexity** - Each model has different APIs, tokenization, output formats
3. **Inference speed degradation** - Larger models (InternVL-40B) may be impractically slow
4. **Unknown unknowns** - New models bring new bugs and edge cases

**Opportunity costs:**
1. **Delays Phase 2 experiments** - C2 cannot start until VLM is selected
2. **Reduces time for C3/C4** - May compress critical hypothesis testing
3. **Team context switching** - Learning curve for new architecture

### 4.3 Honest Assessment

**Probability of success (finding better VLM):** 40-60%

The fundamental challenge is that VLMs are trained for language modeling, not pixel-level reconstruction. While some architectures (CogVLM2, InternVL2) incorporate spatial supervision, this is typically for grounding tasks (output bounding boxes) rather than latent space preservation.

**Recommendation:** Proceed with pivot investigation, but **hedge** by also exploring the hybrid architecture approach (using separate spatial encoder alongside VLM for semantic reasoning).

---

## 5. Resource Requirements

### 5.1 Compute Resources

| Phase | GPU Hours | Estimated Cost |
|-------|-----------|----------------|
| Phase 1: Quick assessment (4 models x 3 days) | 200 | $400 |
| Phase 2: Full assessment (2 models x 7 days) | 350 | $700 |
| Model switching integration | 100 | $200 |
| Buffer/debugging | 150 | $300 |
| **Total** | **800** | **$1,600** |

**GPU requirements:**
- Phase 1: A100-40GB sufficient (all models fit with quantization)
- Phase 2: A100-80GB preferred for full-precision experiments

### 5.2 Personnel

| Role | Allocation | Duration |
|------|------------|----------|
| ML Engineer (primary) | 100% | 6-8 weeks |
| Research Scientist (guidance) | 25% | 6-8 weeks |
| Infrastructure Engineer (Modal) | 10% | 2 weeks |

### 5.3 Timeline

```
Week 1-2:   Phase 1 rapid assessment (all 4 candidates)
            - Day 1-3: LLaVA-NeXT
            - Day 4-6: InternVL2
            - Day 7-9: CogVLM2
            - Day 10: Qwen modifications

Week 3:     Go/no-go decision point
            - Analysis of Phase 1 results
            - Select top 2 candidates

Week 4-5:   Phase 2 full assessment (top 2 candidates)
            - Full C1/Q1/Q2 equivalent experiments
            - Adapter bridging prototypes

Week 6:     Final selection and integration planning
            - Performance analysis
            - Integration requirements document

Week 7-8:   (If pivot approved) Integration and validation
            - Port extraction pipelines
            - Update Modal handlers
            - Validate against C1 metrics
```

### 5.4 External Dependencies

| Dependency | Status | Risk |
|------------|--------|------|
| LLaVA-NeXT weights | Available on HuggingFace | Low |
| InternVL2 weights | Available on HuggingFace | Low |
| CogVLM2 weights | Available (requires license) | Medium |
| A100 GPU access | Available via Modal | Low |
| Something-Something v2 | Requires registration | Low (already have) |

---

## 6. Risk Assessment

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| All VLMs have similar spatial limitations | 40% | High | Parallel-path hybrid architecture exploration |
| Best VLM is too slow for real-time | 30% | Medium | Focus on 7-13B models; quantization research |
| Integration with LTX-Video fails | 20% | High | Early adapter prototyping in Phase 2 |
| New VLM has unexpected failure modes | 50% | Medium | Comprehensive evaluation protocol |
| Temporal coherence degrades with new VLM | 30% | Medium | Include temporal metrics in Phase 2 |

### 6.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Phase 1 takes longer than expected | 40% | Medium | Parallelizable experiments; strict time-boxing |
| No clear winner after Phase 2 | 30% | High | Pre-define decision criteria; accept good-enough |
| Integration uncovers fundamental incompatibility | 20% | High | Early smoke tests in Phase 2 |

### 6.3 Resource Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU availability constraints | 20% | Medium | Reserve capacity; cloud burst option |
| Key personnel unavailable | 15% | High | Documentation; cross-training |
| Budget overrun | 25% | Medium | Weekly spend tracking; early warning system |

### 6.4 Overall Risk Assessment

**Risk Level: MEDIUM-HIGH**

The pivot itself is relatively low-risk (well-understood models, clear evaluation criteria). The high-risk element is the possibility that *no* alternative VLM adequately preserves spatial information, which would require a more fundamental architectural pivot.

---

## 7. Implications for Remaining Experiments

### 7.1 Impact on C2: Adapter Bridging

**If pivot proceeds:**
- C2 experimental design remains valid
- Adapter architectures (MLP, cross-attention) are model-agnostic
- May need to adjust input dimensions (different VLMs have different hidden sizes)
- Timeline: Delayed 6-8 weeks

**If pivot finds no suitable VLM:**
- C2 approach may need hybrid input (VLM semantics + external spatial features)
- Adapter complexity increases significantly
- May require rethinking the fundamental approach

### 7.2 Impact on Q3: Temporal Coherence

**If pivot proceeds:**
- Q3 can proceed with new VLM once adapter is working
- Different VLMs may have different temporal characteristics
- LLaVA models known to work well with video; InternVL2-Video exists

**Key consideration:** We should prefer VLMs with existing video variants (InternVL2-Video, LLaVA-Video) to de-risk Q3.

### 7.3 Impact on C3: Future Prediction

**If pivot proceeds:**
- Query token approach remains valid
- May need different extraction layer (new VLM, new optimal layer)
- Training procedure unchanged

**Key consideration:** Semantic richness must not degrade. New VLM must score >0.6 on standard VQA benchmarks.

### 7.4 Impact on Q4: Training Data Requirements

**Minimal impact:**
- Training data experiments are VLM-agnostic
- Can proceed in parallel once C2 shows feasibility

### 7.5 Impact on Q5: Prediction Horizon

**Minimal impact:**
- Horizon experiments depend on C3 setup
- VLM choice affects but doesn't fundamentally change approach

### 7.6 Impact on C4: Pixel Verification

**If pivot produces spatially-accurate VLM:**
- C4 becomes viable
- LPIPS correlation experiments can proceed as designed

**If spatial accuracy remains marginal (0.5-0.6 IoU):**
- C4 may need to use semantic rather than pixel-level verification
- VLM-based comparison (qualitative) may replace LPIPS (quantitative)
- Weakens core hypothesis but may still produce useful system

### 7.7 Updated Dependency Graph

```
Original:    C1 -> C2 -> C3 -> C4
             Q1 ----^
             Q2 ----^

With Pivot:
                        ┌─────────────────────┐
                        │  VLM Selection      │
                        │  (6-8 weeks)        │
                        └──────────┬──────────┘
                                   │
                                   v
             C1 ────────────────> C2 -> C3 -> C4
             Q1 ──────────────────^
             Q2 ──────────────────^

Net delay: 6-8 weeks to C2 start
           Potential 8-10 weeks to overall timeline
```

---

## 8. Success Metrics

### 8.1 Phase 1 Success (Quick Assessment)

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| At least one VLM with Spatial IoU > 0.5 | > 0.60 | > 0.50 | All < 0.50 |
| At least one VLM with Bbox IoU > 0.3 | > 0.50 | > 0.30 | All < 0.30 |
| Assessment completed on time | 2 weeks | 3 weeks | > 4 weeks |

### 8.2 Phase 2 Success (Full Assessment)

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| Final VLM Spatial IoU | > 0.65 | > 0.60 | < 0.55 |
| Final VLM CKA alignment | > 0.60 | > 0.50 | < 0.40 |
| Final VLM LPIPS reconstruction | < 0.30 | < 0.35 | > 0.40 |
| Inference speed | < 1.5x Qwen | < 2x Qwen | > 3x Qwen |

### 8.3 Overall Pivot Success

The pivot is successful if:

1. **Primary:** Selected VLM achieves Spatial IoU > 0.60 (the threshold C1 failed to meet)
2. **Secondary:** Semantic capabilities maintained (>95% of Qwen scores on VQA)
3. **Tertiary:** Integration completed within 8 weeks of pivot decision

---

## 9. Recommendation

### 9.1 Summary Assessment

| Factor | Assessment |
|--------|------------|
| Technical viability | MEDIUM - Uncertain if alternatives preserve spatial info |
| Risk level | MEDIUM-HIGH - Significant time investment with uncertain outcome |
| Strategic importance | HIGH - Spatial accuracy is critical for C4 verification |
| Opportunity cost | MEDIUM - Delays Phase 2 but may unblock later phases |
| Alternatives | LIMITED - Hybrid approach is main fallback |

### 9.2 Decision Recommendation

**Recommendation: PROCEED WITH PIVOT INVESTIGATION**

**Rationale:**

1. **Blocking issue is real:** Q2 results conclusively show Qwen2.5-VL cannot preserve spatial information. No amount of downstream engineering will fix a fundamental architectural limitation.

2. **Early pivot is cheapest:** We have invested 4 weeks in Qwen. Investing 6-8 weeks now to potentially save 10+ weeks of fruitless work on C2-C4 is a reasonable trade-off.

3. **Generates valuable data:** Even if all VLMs fail, we learn something important about the feasibility of our core hypothesis and can pivot to hybrid approaches or alternative architectures with confidence.

4. **Hedging is possible:** We can explore the hybrid architecture (VLM for semantics, separate encoder for spatial) in parallel, ensuring we have a fallback.

### 9.3 Proposed Decision Points

| Date | Decision | Criteria |
|------|----------|----------|
| Week 2 | Continue to Phase 2? | At least one VLM shows Spatial IoU > 0.50 |
| Week 5 | Commit to new VLM? | Best VLM exceeds all acceptability thresholds |
| Week 6 | Full project pivot? | Integration plan viable; team consensus |

### 9.4 Contingency Plan

**If no VLM meets thresholds by Week 3:**

1. **Immediate:** Begin hybrid architecture exploration (separate spatial encoder)
2. **Week 4-6:** Implement hybrid with OwlVIT/DETR for spatial + Qwen for semantic
3. **Week 7-8:** Evaluate hybrid against original C1/Q2 thresholds

**If pivot is not approved:**

1. **Accept marginal spatial accuracy:** Proceed with Qwen, knowing C4 verification may be limited to semantic rather than pixel-level comparison
2. **Redefine success criteria:** Adjust C4 metrics to emphasize semantic verification
3. **Acknowledge hypothesis limitation:** Document that pixel-level grounding may not be feasible with current VLM architectures

---

## 10. Appendices

### Appendix A: Detailed Architecture Comparison

| Feature | Qwen2.5-VL | LLaVA-NeXT | InternVL2 | CogVLM2 |
|---------|------------|------------|-----------|---------|
| Vision encoder | Custom ViT | CLIP ViT-L | InternViT-6B | EVA-CLIP |
| Vision encoder size | ~300M | ~300M | 6B | ~1B |
| Token merging | 2x2 | None | Dynamic | None |
| High-res support | Natvie | Anyres | Dynamic (4K) | 1344px |
| Spatial grounding | No | Partial | Yes | Yes |
| Video support | Yes | Yes (v1.6) | Yes | Yes |
| HF availability | Yes | Yes | Yes | Yes |
| Quantization support | Good | Good | Good | Good |

### Appendix B: Spatial Preservation Literature

1. **Florence-2 (Microsoft):** Unified representation for various vision tasks including detection. May preserve spatial info but focused on task-specific outputs rather than latent space.

2. **Grounding-DINO:** Explicitly designed for spatial grounding. May have best spatial preservation but lacks language generation capabilities.

3. **Kosmos-2:** Multimodal with grounding capabilities. Worth considering if initial candidates fail.

4. **Cambrian-1:** Recent (2024) multimodal model with focus on visual representation quality. Limited availability.

### Appendix C: References

1. Wang et al. (2024). "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution." arXiv.
2. Liu et al. (2024). "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge." lmms-lab blog.
3. Chen et al. (2024). "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks." CVPR.
4. Wang et al. (2024). "CogVLM: Visual Expert for Pretrained Language Models." NeurIPS.
5. Tong et al. (2024). "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs." arXiv.

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-18 | Research Team | Initial draft |

---

*This proposal was prepared for department review. All estimates are subject to revision based on initial experimental results.*
