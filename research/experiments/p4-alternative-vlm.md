# Experiment Plan: P4 - Alternative VLM Architecture Evaluation

**Pivot Rationale:** Gate 1 failed due to spatial information loss in Qwen2.5-VL (Spatial IoU=0.559, mAP=0.001)
**Status:** Not Started
**Priority:** Critical (blocks all downstream work)
**Owner:** TBD
**Created:** 2026-01-18
**Reference:** [Pivot Proposal](/Users/adrianobleton/foresight/research/proposals/pivot-4-alternative-vlm.md)

---

## 1. Objective

Evaluate alternative VLM architectures to determine if any preserve spatial information sufficiently to enable accurate video prediction for verification-based reasoning.

**Core Question:** Can we find a VLM that achieves Spatial IoU > 0.6 while maintaining semantic reasoning capabilities, thereby unblocking the Generative Latent Prediction (GLP) architecture?

**Why This Matters:** Our Q2 experiments conclusively demonstrated that Qwen2.5-VL cannot preserve spatial information (Bbox IoU=0.104, mAP@0.5=0.001). This is a fundamental architectural limitation, not an extraction point issue. Without spatial accuracy, generated videos cannot place objects correctly for C4 pixel-level verification--the core differentiator of our approach from latent-only methods like V-JEPA.

---

## 2. Background

### 2.1 The Problem with Qwen2.5-VL

From Q2 experiments, we traced spatial information through the processing pipeline:

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

**Key Finding:** Spatial information is not destroyed by the 2x2 merger or LLM layers--it was never properly encoded in the first place. The vision encoder was trained with a language modeling objective that prioritizes semantic discrimination over spatial preservation.

### 2.2 Gate 1 Failure Summary

| Metric | Target | C1 Achieved | Q2 Achieved | Status |
|--------|--------|-------------|-------------|--------|
| LPIPS | < 0.35 | 0.236 | 0.087 | PASS |
| Spatial IoU | > 0.60 | 0.559 | - | FAIL |
| Bbox IoU | > 0.50 | - | 0.104 | FAIL |
| mAP@0.5 | > 0.20 | - | 0.001 | FAIL |
| Direction Accuracy | > 80% | - | 100% | PASS |

**Diagnosis:** VLM preserves semantic and temporal information but not spatial localization.

### 2.3 Candidate VLM Architectures

We will evaluate three alternative architectures with different approaches to spatial preservation:

| VLM | Vision Encoder | Token Merging | Spatial Features | Why It Might Work |
|-----|----------------|---------------|------------------|-------------------|
| **LLaVA-NeXT** | CLIP ViT-L (~300M) | None | Anyres (high-res patches) | Preserves all 576+ spatial tokens, no compression |
| **InternVL2** | InternViT-6B (6B) | Dynamic | Detection/segmentation objectives | Trained with spatial supervision tasks |
| **CogVLM2** | EVA-CLIP (~1B) | None | Visual expert modules | Explicit grounding, separate visual stream |

### 2.4 Hypothesis

**H-P4:** At least one alternative VLM architecture (LLaVA-NeXT, InternVL2, or CogVLM2) will preserve sufficient spatial information (Spatial IoU > 0.6, Bbox IoU > 0.5) to enable the GLP architecture while maintaining semantic reasoning capabilities (>95% of Qwen2.5-VL on VQA tasks).

**Null Hypothesis to Falsify:** All VLMs optimized for vision-language tasks fundamentally discard spatial information unnecessary for their training objective, making pixel-level verification impossible without auxiliary spatial encoders.

---

## 3. Experimental Setup

### 3.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x A100 40GB | 1x A100 80GB |
| CPU RAM | 64GB | 128GB |
| Storage | 300GB SSD | 500GB NVMe |

**VRAM breakdown per model (bf16):**
- LLaVA-NeXT-7B: ~14GB
- LLaVA-NeXT-13B: ~26GB
- InternVL2-8B: ~16GB
- InternVL2-26B: ~52GB (requires A100 80GB)
- CogVLM2-19B: ~38GB
- LTX-Video VAE: ~2GB
- Probes/adapters: ~2-5GB

### 3.2 Software Dependencies

```bash
# Core dependencies
pip install torch>=2.1.0 torchvision torchaudio
pip install transformers>=4.40.0 accelerate>=0.27.0
pip install diffusers>=0.27.0
pip install flash-attn --no-build-isolation

# Model-specific
pip install llava-next  # or clone from GitHub
pip install internvl    # OpenGVLab/InternVL
# CogVLM2 requires manual installation from THUDM/CogVLM2

# Evaluation
pip install lpips pytorch-fid scikit-learn
pip install pycocotools  # For mAP evaluation

# Utilities
pip install wandb einops matplotlib seaborn
```

### 3.3 Model Checkpoints

```bash
# LLaVA-NeXT variants
huggingface-cli download lmms-lab/llava-next-interleave-qwen-7b
huggingface-cli download lmms-lab/llava-next-interleave-qwen-14b

# InternVL2 variants
huggingface-cli download OpenGVLab/InternVL2-8B
huggingface-cli download OpenGVLab/InternVL2-26B  # If compute allows

# CogVLM2
huggingface-cli download THUDM/cogvlm2-llama3-chat-19B

# Video decoder (for final validation)
huggingface-cli download Lightricks/LTX-Video
```

### 3.4 Test Datasets

**Phase 1: Synthetic (Fast Screening)**

| Dataset | N Samples | Resolution | Purpose |
|---------|-----------|------------|---------|
| Synthetic shapes | 500 | 224x224 | Controlled spatial tests (same as Q2) |
| Moving shapes | 200 | 224x224 | Temporal preservation |

**Phase 2: Real-world (Full Validation)**

| Dataset | N Samples | Resolution | Purpose |
|---------|-----------|------------|---------|
| COCO 2017 val | 1000 | 448x448 | Object detection, spatial probing |
| Something-Something v2 | 500 | 224x224 | Video/action understanding |

### 3.5 Extraction Points Per Model

**LLaVA-NeXT:**
| ID | Point | Shape | Description |
|----|-------|-------|-------------|
| L1 | CLIP ViT output | [576+, 1024] | All spatial tokens preserved |
| L2 | After MLP projector | [576+, 4096] | Projected to LLM space |
| L3 | LLM layer 8 | [seq, 4096] | Early LLM |
| L4 | LLM layer 16 | [seq, 4096] | Mid LLM |
| L5 | LLM layer 32 | [seq, 4096] | Final LLM |

**InternVL2:**
| ID | Point | Shape | Description |
|----|-------|-------|-------------|
| I1 | InternViT layer 24 | [N_patches, 3200] | Mid-vision encoder |
| I2 | InternViT output | [N_patches, 3200] | Full vision encoder |
| I3 | After pixel shuffle | [N_tokens, hidden] | After spatial upsampling |
| I4 | LLM layer 8 | [seq, 4096] | Early LLM |
| I5 | LLM layer 16 | [seq, 4096] | Mid LLM |

**CogVLM2:**
| ID | Point | Shape | Description |
|----|-------|-------|-------------|
| C1 | EVA-CLIP output | [N_patches, 1792] | Vision encoder output |
| C2 | Visual expert layer 8 | [N_patches, 4096] | Early visual expert |
| C3 | Visual expert layer 16 | [N_patches, 4096] | Mid visual expert |
| C4 | After visual-text fusion | [seq, 4096] | Fused representation |

---

## 4. Experiments

### E-P4.1: LLaVA-NeXT Spatial Information Analysis

**Objective:** Replicate Q2 spatial probing experiments on LLaVA-NeXT to measure spatial information preservation.

**Protocol:**

1. Extract embeddings from all extraction points (L1-L5) for synthetic shapes dataset
2. Train linear bounding box probe on each extraction point
3. Train detection probe (DETR-style) on each extraction point
4. Measure fine-grained detail preservation via reconstruction decoder

**Implementation:**

```python
# Extraction code sketch for LLaVA-NeXT
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images

def extract_llava_features(model, image, extraction_points):
    """Extract features from LLaVA-NeXT at specified points."""
    results = {}

    # Process image
    image_tensor = process_images([image], image_processor, model.config)

    # Get CLIP ViT features (L1)
    with torch.no_grad():
        vision_tower = model.get_vision_tower()
        image_features = vision_tower(image_tensor)
        results['L1_clip_output'] = image_features

        # After MLP projection (L2)
        projected = model.mm_projector(image_features)
        results['L2_projected'] = projected

        # LLM hidden states (L3-L5)
        outputs = model.llm(
            inputs_embeds=projected,
            output_hidden_states=True
        )
        results['L3_llm_8'] = outputs.hidden_states[8]
        results['L4_llm_16'] = outputs.hidden_states[16]
        results['L5_llm_32'] = outputs.hidden_states[32]

    return results
```

**Metrics:**

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| Bbox IoU (best point) | > 0.60 | > 0.50 | < 0.40 |
| mAP@0.5 | > 0.30 | > 0.20 | < 0.10 |
| LPIPS reconstruction | < 0.30 | < 0.35 | > 0.45 |
| Edge F1 | > 0.60 | > 0.50 | < 0.40 |

**Duration:** 3 days

**Deliverables:**
- Layer-by-layer spatial metrics table
- Comparison to Qwen2.5-VL baseline
- Visualization of spatial probing results
- Go/no-go recommendation for Phase 2

---

### E-P4.2: InternVL2 Spatial Information Analysis

**Objective:** Replicate Q2 spatial probing experiments on InternVL2, focusing on the larger vision encoder trained with spatial objectives.

**Protocol:**

1. Extract embeddings from all extraction points (I1-I5) for synthetic shapes dataset
2. Train same probes as E-P4.1
3. Special attention to InternViT layers (designed for spatial tasks)
4. Test both 8B and 26B variants if resources allow

**Implementation:**

```python
# Extraction code sketch for InternVL2
from transformers import AutoModel, AutoTokenizer

def extract_internvl_features(model, image, extraction_points):
    """Extract features from InternVL2 at specified points."""
    results = {}

    # Register hooks for intermediate layers
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            results[name] = output.detach()
        return hook

    # Hook InternViT layers
    for i, layer in enumerate(model.vision_model.encoder.layers):
        if i in [23, -1]:  # Layer 24 and final
            hooks.append(layer.register_forward_hook(hook_fn(f'I{i+1}')))

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values=image, output_hidden_states=True)

        results['I2_vit_output'] = outputs.image_embeds
        results['I3_pixel_shuffle'] = outputs.vision_features

        # LLM layers
        for i, hidden in enumerate(outputs.hidden_states):
            if i in [8, 16]:
                results[f'I{4 if i==8 else 5}_llm_{i}'] = hidden

    # Remove hooks
    for h in hooks:
        h.remove()

    return results
```

**Metrics:** Same as E-P4.1

**Special Analysis:**
- Compare InternViT-6B to CLIP ViT (LLaVA) and Qwen ViT
- Analyze impact of detection/segmentation pretraining
- Test high-resolution (1K+) input modes

**Duration:** 3 days

**Deliverables:**
- Layer-by-layer spatial metrics table
- InternViT-specific analysis (does larger vision encoder help?)
- High-resolution vs standard resolution comparison

---

### E-P4.3: CogVLM2 Spatial Information Analysis

**Objective:** Replicate Q2 spatial probing experiments on CogVLM2, focusing on the visual expert modules.

**Protocol:**

1. Extract embeddings from all extraction points (C1-C4) for synthetic shapes dataset
2. Train same probes as E-P4.1
3. Special analysis of visual expert pathway
4. Test explicit grounding capability

**Implementation:**

```python
# Extraction code sketch for CogVLM2
# Note: CogVLM2 has unique architecture with visual expert modules

def extract_cogvlm_features(model, image, extraction_points):
    """Extract features from CogVLM2 at specified points."""
    results = {}

    # EVA-CLIP output
    with torch.no_grad():
        vision_outputs = model.vision_model(image)
        results['C1_evaclip'] = vision_outputs.last_hidden_state

        # Visual expert layers (unique to CogVLM)
        # These maintain separate visual processing stream
        visual_expert_states = model.get_visual_expert_states(
            vision_outputs, output_all_layers=True
        )
        results['C2_expert_8'] = visual_expert_states[8]
        results['C3_expert_16'] = visual_expert_states[16]

        # After fusion
        fused = model.vision_language_fusion(vision_outputs)
        results['C4_fused'] = fused

    return results
```

**Grounding Test:**

CogVLM2 has explicit grounding capability (outputs bounding boxes). Test this:

```python
# Direct grounding test
prompt = "Find the location of the red circle"
response = model.generate(image, prompt)
# Parse bounding box from response
# Compare to ground truth
```

**Metrics:** Same as E-P4.1 + grounding-specific metrics

**Special Analysis:**
- Visual expert vs standard pathway comparison
- Does explicit grounding capability help latent space?
- Separate visual stream preservation

**Duration:** 3 days

**Deliverables:**
- Layer-by-layer spatial metrics table
- Visual expert analysis report
- Grounding capability assessment

---

### E-P4.4: Phase 1 Screening Summary and Selection

**Objective:** Compare all three VLMs and select top 2 candidates for full evaluation.

**Protocol:**

1. Compile all E-P4.1, E-P4.2, E-P4.3 results
2. Apply selection criteria
3. Select top 2 candidates for Phase 2
4. Identify fallback options

**Selection Criteria:**

| Criterion | Weight | Threshold |
|-----------|--------|-----------|
| Spatial IoU (best point) | 40% | > 0.50 to proceed |
| Bbox IoU | 20% | > 0.40 to proceed |
| LPIPS reconstruction | 15% | < 0.40 to proceed |
| Inference speed | 15% | < 3x Qwen baseline |
| Integration complexity | 10% | Subjective assessment |

**Decision Matrix:**

| Outcome | Action |
|---------|--------|
| One VLM exceeds all thresholds | Continue with single candidate |
| Multiple VLMs exceed thresholds | Select top 2 based on weighted score |
| One VLM has IoU > 0.50, others fail | Continue with single candidate + hybrid backup |
| All VLMs fail IoU > 0.50 | STOP - pivot to hybrid architecture (see Section 7) |

**Duration:** 1 day

**Deliverables:**
- Comparative analysis report
- Ranked VLM list with scores
- Selection justification
- Phase 2 plan refinement

---

### E-P4.5: Best Candidate - Latent Alignment Study (Q1 Replication)

**Objective:** Replicate Q1 latent alignment experiments on the best candidate VLM(s) to assess bridgeability to video decoder.

**Protocol:**

For each selected candidate:

1. Extract VLM latents for 1000 images (Something-Something v2)
2. Extract LTX-Video VAE latents for same images
3. Measure CKA alignment between spaces
4. Train linear probe for cross-space prediction
5. Analyze structural similarity (Spearman correlation, neighborhood overlap)

**Implementation:**

```python
def cka_alignment_study(vlm_features, ltx_features):
    """Compute CKA and other alignment metrics."""

    # CKA computation
    cka_score = centered_kernel_alignment(vlm_features, ltx_features)

    # Linear probe: VLM -> LTX
    probe = LinearProbe(vlm_dim, ltx_dim)
    train_probe(probe, vlm_features, ltx_features)
    r2_vlm_to_ltx = evaluate_probe(probe, test_vlm, test_ltx)

    # Semantic similarity preservation
    vlm_distances = pairwise_distances(vlm_features)
    ltx_distances = pairwise_distances(ltx_features)
    spearman_rho = spearmanr(vlm_distances.flatten(), ltx_distances.flatten())

    # Neighborhood overlap
    recall_at_10 = neighborhood_overlap(vlm_features, ltx_features, k=10)

    return {
        'cka': cka_score,
        'r2': r2_vlm_to_ltx,
        'spearman': spearman_rho,
        'recall_at_10': recall_at_10
    }
```

**Metrics:**

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| CKA (best layer) | > 0.50 | > 0.40 | < 0.30 |
| Linear R^2 | > 0.50 | > 0.40 | < 0.25 |
| Spearman rho | > 0.60 | > 0.50 | < 0.40 |
| Recall@10 | > 30% | > 20% | < 10% |

**Comparison to Qwen2.5-VL Baseline:**

From Q1 experiments:
- Qwen CKA: 0.687
- Qwen Spearman: 0.72

New VLM must achieve comparable or better alignment while also preserving spatial info.

**Duration:** 5 days (per candidate)

**Deliverables:**
- CKA heatmaps (VLM layers vs LTX channels)
- Linear probe results
- Structural similarity analysis
- Optimal extraction layer recommendation

---

### E-P4.6: Best Candidate - Reconstruction Probe (C1 Replication)

**Objective:** Replicate C1 reconstruction experiments on the best candidate VLM(s) to validate end-to-end feasibility.

**Protocol:**

For each selected candidate:

1. Train adapter network: VLM latents -> LTX-Video conditioning
2. Generate reconstructions through LTX-Video decoder
3. Measure reconstruction quality (LPIPS, SSIM, Spatial IoU)
4. Compare to Qwen2.5-VL C1 results

**Adapter Architectures to Test:**

```python
# Architecture 1: Simple MLP (baseline)
class MLPAdapter(nn.Module):
    def __init__(self, vlm_dim, ltx_dim, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vlm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ltx_dim)
        )

# Architecture 2: Cross-attention (for spatial preservation)
class SpatialAdapter(nn.Module):
    def __init__(self, vlm_dim, ltx_dim, n_vlm_tokens, n_ltx_tokens):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_ltx_tokens, ltx_dim))
        self.cross_attn = nn.MultiheadAttention(ltx_dim, num_heads=8)
        self.proj_k = nn.Linear(vlm_dim, ltx_dim)
        self.proj_v = nn.Linear(vlm_dim, ltx_dim)

    def forward(self, vlm_tokens):
        k = self.proj_k(vlm_tokens)
        v = self.proj_v(vlm_tokens)
        q = self.queries.unsqueeze(0).expand(vlm_tokens.size(0), -1, -1)
        out, _ = self.cross_attn(q.transpose(0,1), k.transpose(0,1), v.transpose(0,1))
        return out.transpose(0,1)
```

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Loss | LPIPS + 0.1 * L2 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Training samples | 5000 |
| Epochs | 50 |

**Metrics:**

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| LPIPS | < 0.30 | < 0.35 | > 0.45 |
| SSIM | > 0.80 | > 0.75 | < 0.65 |
| Spatial IoU | > 0.65 | > 0.60 | < 0.55 |
| mAP@0.5 (from reconstruction) | > 0.30 | > 0.20 | < 0.10 |

**Ablation Studies:**

| Factor | Variations | Purpose |
|--------|------------|---------|
| Adapter size | 5M, 10M, 20M, 50M params | Find minimum viable size |
| Extraction layer | Best 3 from E-P4.5 | Confirm optimal layer |
| Resolution | 224, 336, 448 | Test high-res impact |

**Duration:** 7 days (per candidate)

**Deliverables:**
- Reconstruction quality metrics
- Side-by-side comparisons (original / Qwen / new VLM)
- Adapter architecture recommendation
- Spatial IoU improvement verification

---

### E-P4.7: Comparative Analysis and Final Recommendation

**Objective:** Synthesize all results into a final recommendation for VLM selection.

**Protocol:**

1. Compile all metrics across experiments
2. Compare to Qwen2.5-VL baseline and Gate 1 thresholds
3. Assess integration complexity and inference speed
4. Make final recommendation

**Comprehensive Comparison Table:**

| Metric | Qwen2.5-VL | LLaVA-NeXT | InternVL2 | CogVLM2 | Target |
|--------|------------|------------|-----------|---------|--------|
| Spatial IoU | 0.559 | ? | ? | ? | > 0.60 |
| Bbox IoU | 0.104 | ? | ? | ? | > 0.50 |
| mAP@0.5 | 0.001 | ? | ? | ? | > 0.20 |
| LPIPS recon | 0.236 | ? | ? | ? | < 0.35 |
| CKA alignment | 0.687 | ? | ? | ? | > 0.40 |
| Direction acc | 100% | ? | ? | ? | > 80% |
| Inference time | 1.0x | ? | ? | ? | < 2x |

**Decision Criteria:**

| Priority | Criterion | Weight |
|----------|-----------|--------|
| 1 | Spatial IoU > 0.60 | Must pass |
| 2 | mAP@0.5 > 0.20 | Must pass |
| 3 | LPIPS < 0.35 | Must pass |
| 4 | CKA > 0.40 | Should pass |
| 5 | Inference < 2x | Should pass |
| 6 | Integration complexity | Tie-breaker |

**Recommendation Framework:**

| Outcome | Recommendation |
|---------|----------------|
| One VLM meets all Must Pass | **Proceed** - Switch to new VLM |
| Multiple VLMs meet Must Pass | Select by weighted score + complexity |
| One VLM meets Spatial IoU but fails others | **Investigate** - May need adapter modifications |
| No VLM meets Spatial IoU > 0.55 | **Pivot** - Hybrid architecture required |
| No VLM meets Spatial IoU > 0.50 | **Major Pivot** - Rethink fundamental approach |

**Duration:** 2 days

**Deliverables:**
- Comprehensive comparison report
- Final VLM recommendation with confidence level
- Integration roadmap (if proceeding)
- Contingency plans (if pivoting)

---

## 5. Success Criteria

### 5.1 Primary Success Metrics (Must Pass All)

| Metric | Threshold | Measured In |
|--------|-----------|-------------|
| Spatial IoU | > 0.60 | E-P4.6 |
| Bbox IoU (Phase 1 screen) | > 0.50 | E-P4.1/2/3 |
| mAP@0.5 | > 0.20 | E-P4.1/2/3 |
| LPIPS reconstruction | < 0.35 | E-P4.6 |

### 5.2 Secondary Success Metrics (Should Pass)

| Metric | Threshold | Measured In |
|--------|-----------|-------------|
| CKA alignment | > 0.40 | E-P4.5 |
| Spearman correlation | > 0.50 | E-P4.5 |
| Temporal direction accuracy | > 80% | E-P4.6 extension |
| Inference speed | < 2x Qwen | E-P4.7 |

### 5.3 Success Criteria Summary

**FULL SUCCESS:** At least one VLM achieves:
- Spatial IoU > 0.60 (Gate 1 threshold)
- mAP@0.5 > 0.20
- LPIPS < 0.35
- CKA > 0.40

**PARTIAL SUCCESS:** At least one VLM achieves:
- Spatial IoU > 0.55 (within 10% of threshold)
- Bbox IoU > 0.50 (screening threshold)
- Other metrics acceptable

**FAILURE:** All VLMs achieve:
- Spatial IoU < 0.50
- mAP@0.5 < 0.10

---

## 6. Timeline

### 6.1 Overview

| Phase | Duration | Experiments | Deliverable |
|-------|----------|-------------|-------------|
| Phase 1: Screening | 10 days | E-P4.1, E-P4.2, E-P4.3, E-P4.4 | Top 2 candidates selected |
| Phase 2: Deep Eval | 12 days | E-P4.5, E-P4.6 (x2 candidates) | Full validation results |
| Phase 3: Decision | 2 days | E-P4.7 | Final recommendation |
| **Total** | **24 days** | | |

### 6.2 Detailed Schedule

```
Week 1: Phase 1 Screening
  Day 1-3:   E-P4.1 LLaVA-NeXT spatial analysis
  Day 4-6:   E-P4.2 InternVL2 spatial analysis
  Day 7-9:   E-P4.3 CogVLM2 spatial analysis
  Day 10:    E-P4.4 Screening summary & selection

Week 2-3: Phase 2 Deep Evaluation (Candidate 1)
  Day 11-15: E-P4.5 Latent alignment study
  Day 16-22: E-P4.6 Reconstruction probe

Week 3-4: Phase 2 Deep Evaluation (Candidate 2) [Parallel if resources allow]
  Day 16-20: E-P4.5 Latent alignment study (Candidate 2)
  Day 21-27: E-P4.6 Reconstruction probe (Candidate 2)

Week 4: Final Analysis
  Day 28-29: E-P4.7 Comparative analysis
  Day 30:    Final recommendation document
```

### 6.3 Parallelization Opportunities

With 2 A100 GPUs:
- E-P4.1, E-P4.2, E-P4.3 can partially overlap (different models)
- E-P4.5 for two candidates can run in parallel
- E-P4.6 for two candidates can run in parallel

**With parallelization: 18-20 days**

### 6.4 Critical Path

```
E-P4.1/2/3 (screening) -> E-P4.4 (selection) -> E-P4.5 (alignment) -> E-P4.6 (reconstruction) -> E-P4.7 (decision)
```

Go/no-go decision point at E-P4.4 (Day 10).

---

## 7. Resource Requirements

### 7.1 Compute Budget

| Phase | GPU Hours | Estimated Cost (@$2/hr) |
|-------|-----------|-------------------------|
| Phase 1: Screening (3 VLMs x 3 days) | 216 | $432 |
| Phase 2: Alignment (2 candidates x 5 days) | 240 | $480 |
| Phase 2: Reconstruction (2 candidates x 7 days) | 336 | $672 |
| Buffer/debugging | 150 | $300 |
| **Total** | **942** | **$1,884** |

### 7.2 Personnel

| Role | Allocation | Duration |
|------|------------|----------|
| ML Engineer (primary) | 100% | 4 weeks |
| Research Scientist (guidance) | 25% | 4 weeks |
| Infrastructure Engineer (Modal) | 10% | 1 week |

### 7.3 External Dependencies

| Dependency | Status | Risk |
|------------|--------|------|
| LLaVA-NeXT weights | Available on HuggingFace | Low |
| InternVL2 weights | Available on HuggingFace | Low |
| CogVLM2 weights | Requires license acceptance | Medium |
| A100 GPU access | Available via Modal | Low |
| Something-Something v2 | Already downloaded | Low |
| COCO 2017 | Public dataset | Low |

---

## 8. Dependencies

### 8.1 Prerequisites (Must Complete Before Starting)

- [x] Q2 experiment results (spatial information baseline)
- [x] C1 experiment results (reconstruction baseline)
- [x] Q1 experiment results (alignment baseline)
- [x] Gate 1 failure confirmed
- [ ] GPU resources allocated (Modal)
- [ ] All model weights downloaded

### 8.2 This Experiment Blocks

| Experiment | Why Blocked |
|------------|-------------|
| C2: Adapter Bridging | Need to know which VLM |
| Q3: Temporal Coherence | Need working VLM selection |
| C3: Future Prediction | Depends on C2 |
| C4: Pixel Verification | Depends on C3 |

### 8.3 Unblocking Timeline

If P4 succeeds by Day 24:
- C2 can start Day 25
- Overall project delayed 4 weeks from original timeline
- Gate 2 achievable by Week 12 (vs original Week 8)

---

## 9. Risks and Mitigations

### 9.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| All VLMs have similar spatial limitations | 40% | High | Begin hybrid architecture planning in parallel |
| Best VLM is too slow for real-time | 25% | Medium | Focus on smaller variants; quantization research |
| Integration with LTX-Video fails | 15% | High | Early smoke test in Phase 2 |
| New VLM has unexpected failure modes | 35% | Medium | Comprehensive evaluation protocol |
| InternVL2-26B doesn't fit in memory | 30% | Low | Use 8B variant as fallback |

### 9.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Phase 1 takes longer than expected | 30% | Medium | Strict time-boxing; prioritize by likelihood of success |
| No clear winner after Phase 2 | 25% | High | Pre-defined decision criteria; accept good-enough |
| GPU availability constraints | 15% | Medium | Reserve Modal capacity; cloud burst option |

### 9.3 Strategic Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Sunk cost of Qwen-specific work | 100% | Medium | Most code is reusable; only extraction layer changes |
| Team context switching | 50% | Low | Document differences; create integration guide |
| Community support for new VLM is limited | 30% | Low | Prefer LLaVA-NeXT (largest community) |

### 9.4 Risk Assessment Summary

**Overall Risk Level: MEDIUM-HIGH**

The highest risk is that no alternative VLM adequately preserves spatial information (40% probability). If this occurs, we must pivot to a hybrid architecture approach.

---

## 10. Pivot Options (If P4 Fails)

### 10.1 Hybrid Architecture (Primary Fallback)

If no VLM meets spatial thresholds:

**Approach:** Use separate encoders for spatial and semantic information.

```
Image -> VLM (Qwen2.5-VL) -> Semantic/Temporal features ----\
      \                                                      -> Adapter -> Video Decoder
       -> Spatial Encoder (OwlVIT/DETR/DINOv2) -> Spatial features ----/
```

**Pros:**
- Guaranteed spatial accuracy (detection models are precise)
- Preserves semantic reasoning capability

**Cons:**
- Increased complexity
- Higher inference cost (2 encoders)
- More challenging adapter design

**Effort:** +3-4 weeks

### 10.2 Grounding-Specialized VLMs

**Candidates:**
- Kosmos-2 (Microsoft)
- Grounding-DINO + LLM hybrid
- Florence-2

**When to consider:** If all candidates fail but show promise (IoU > 0.45)

**Effort:** +2 weeks screening

### 10.3 Fine-tune Vision Encoder

**Approach:** Unfreeze and fine-tune best VLM's vision encoder with spatial reconstruction objective.

**Pros:** Direct fix for spatial preservation

**Cons:**
- May degrade semantic capabilities
- Expensive training
- Requires careful balancing

**Effort:** +4-6 weeks

### 10.4 Accept Marginal Performance

**Approach:** If best VLM achieves Spatial IoU 0.55-0.60, proceed anyway.

**Implications:**
- C4 verification may need to be semantic rather than pixel-level
- Redefine success criteria for pixel grounding
- Document limitations

**When to consider:** If marginal performance with high confidence

---

## 11. Deliverables

### 11.1 Per-Experiment Deliverables

| Experiment | Primary Deliverable | Location |
|------------|---------------------|----------|
| E-P4.1 | LLaVA-NeXT spatial metrics | `results/p4/llava-spatial.yaml` |
| E-P4.2 | InternVL2 spatial metrics | `results/p4/internvl-spatial.yaml` |
| E-P4.3 | CogVLM2 spatial metrics | `results/p4/cogvlm-spatial.yaml` |
| E-P4.4 | Candidate selection report | `results/p4/phase1-summary.md` |
| E-P4.5 | Alignment analysis | `results/p4/alignment-study.md` |
| E-P4.6 | Reconstruction results | `results/p4/reconstruction-study.md` |
| E-P4.7 | Final recommendation | `results/p4/final-recommendation.md` |

### 11.2 Code Artifacts

| Artifact | Purpose | Location |
|----------|---------|----------|
| VLM extractors | Feature extraction for all VLMs | `infra/modal/handlers/p4/extractors/` |
| Spatial probes | Bbox, mAP, edge probes | `infra/modal/handlers/p4/probes/` |
| Alignment analysis | CKA, correlation scripts | `infra/modal/handlers/p4/alignment/` |
| Reconstruction pipeline | Adapter training + eval | `infra/modal/handlers/p4/reconstruction/` |

### 11.3 Final Report Structure

```markdown
# P4 Alternative VLM Evaluation - Final Report

## Executive Summary
- Selected VLM: [name]
- Key metrics achieved: [table]
- Confidence level: [High/Medium/Low]

## Screening Results (Phase 1)
- LLaVA-NeXT: [summary]
- InternVL2: [summary]
- CogVLM2: [summary]

## Deep Evaluation Results (Phase 2)
- Alignment study: [findings]
- Reconstruction study: [findings]

## Recommendation
- Primary: [proceed/pivot/investigate]
- Integration requirements: [list]
- Timeline impact: [estimate]

## Next Steps
- If proceeding: [C2 modifications needed]
- If pivoting: [hybrid architecture plan]
```

---

## 12. Open Questions

To be resolved during experiments:

1. **LLaVA-NeXT extraction:** Does the Anyres mechanism provide better spatial features than standard resolution?

2. **InternVL2 vision encoder:** Does the 6B vision encoder genuinely preserve spatial info, or is it also optimized away?

3. **CogVLM2 grounding:** Does explicit grounding capability translate to better latent space structure?

4. **Cross-model adapters:** Can an adapter trained on Qwen be fine-tuned for new VLM, or must we train from scratch?

5. **Inference optimization:** Can we use smaller VLM variants (7B) with acceptable quality loss?

---

## 13. Appendix

### A. Model Architecture Details

**LLaVA-NeXT-7B:**
- Vision: CLIP ViT-L/14 (304M params)
- Language: Vicuna-7B or Qwen-7B
- Projection: 2-layer MLP
- Total: ~7.5B params

**InternVL2-8B:**
- Vision: InternViT-6B (5.5B params)
- Language: InternLM2-Chat-7B
- Projection: MLP with PixelShuffle
- Total: ~8B params

**CogVLM2-19B:**
- Vision: EVA-CLIP-E (4.4B params)
- Language: LLaMA-3-8B with visual expert
- Projection: Visual expert modules in every layer
- Total: ~19B params

### B. Reference Commands

```bash
# Run Phase 1 screening
uv run modal run infra/modal/app.py::run_experiment --experiment-id p4-llava-screening
uv run modal run infra/modal/app.py::run_experiment --experiment-id p4-internvl-screening
uv run modal run infra/modal/app.py::run_experiment --experiment-id p4-cogvlm-screening

# Run Phase 2 for selected candidate
uv run modal run infra/modal/app.py::run_experiment --experiment-id p4-alignment-<vlm>
uv run modal run infra/modal/app.py::run_experiment --experiment-id p4-reconstruction-<vlm>

# Validate results
python research/validation/scripts/validate_experiment.py p4-alternative-vlm
```

### C. References

1. Liu et al. (2024). "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge." lmms-lab blog.
2. Chen et al. (2024). "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks." CVPR.
3. Wang et al. (2024). "CogVLM: Visual Expert for Pretrained Language Models." NeurIPS.
4. Wang et al. (2024). "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution." arXiv.

---

## 14. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-18 | Research Team | Initial draft |

---

*This experiment plan was prepared for Pivot 4: Alternative VLM Architecture Evaluation following Gate 1 failure due to spatial information loss in Qwen2.5-VL.*
