# Experiment Plan: Q3 - Temporal Coherence

**Question:** Does conditioning injection from the hybrid encoder disrupt video decoder's temporal dynamics?

**Status:** Not Started
**Priority:** High (required for Gate 2)
**Owner:** TBD
**Created:** 2026-01-18
**Updated:** 2026-01-20

**Dependencies:** P2 Hybrid Encoder (PASSED - spatial_iou=0.837, lpips=0.162)

---

## 1. Objective

Validate that videos generated through the GLP architecture (hybrid encoder + LTX-Video) maintain temporal coherence across frames. This is critical because:

1. **LTX-Video has learned temporal priors** - Causal 3D convolutions and temporal attention create smooth motion
2. **Our conditioning injection could disrupt these priors** - VLM/DINOv2 features might override learned dynamics
3. **Poor temporal coherence would make generated videos unusable** - Flickering, identity drift, or physics violations

**Core Question:** Can we inject hybrid encoder conditioning while preserving LTX-Video's temporal modeling capabilities?

**Why This Matters:** Gate 2 requires both C2 (adapter efficiency) and Q3 (temporal coherence) to pass. Without temporal coherence, the video prediction system cannot produce useful visualizations for reasoning verification.

---

## 2. Hypothesis

**Primary Hypothesis:**
The hybrid encoder conditioning (DINOv2 spatial + VLM semantic features) can be injected into LTX-Video without significantly degrading temporal coherence, achieving:
- temporal_consistency > 0.7 (acceptable threshold)
- Ideally temporal_consistency > 0.8 (target threshold)

**Null Hypothesis:**
Conditioning injection fundamentally disrupts LTX-Video's temporal modeling, causing unacceptable flickering, identity drift, or motion discontinuities regardless of injection strategy.

**Falsifiability:**
- If temporal_consistency < 0.5: Fundamental incompatibility (FAIL)
- If conditioning strength vs coherence tradeoff has no acceptable operating point: Architecture needs redesign
- If human evaluation shows > 50% "unnatural motion" ratings: Automated metrics are insufficient

---

## 3. Background

### 3.1 How LTX-Video Maintains Temporal Consistency

LTX-Video employs several mechanisms for temporal coherence:

1. **Causal 3D Convolutions in VAE**
   - Encoder uses causal temporal convolutions (8x temporal compression)
   - First frame encoded independently; subsequent frames conditioned on previous
   - Creates autoregressive structure in latent space

2. **Temporal Attention in DiT**
   - 28-block DiT uses spatiotemporal attention
   - Tokens attend across both spatial and temporal dimensions
   - RoPE positional encoding preserves temporal relationships

3. **Joint Denoising**
   - All frames denoised together (not independently)
   - Global noise schedule ensures consistent denoising trajectory

4. **High Channel Latents (128 channels)**
   - More capacity to encode temporal relationships
   - Information flows through latent space

### 3.2 P2 Hybrid Encoder Foundation

From P2 results (passed Gate 1):
- **Spatial IoU:** 0.837 (target > 0.60)
- **LPIPS:** 0.162 (target < 0.35)
- **Architecture:** DINOv2-ViT-L + VLM cross-attention fusion
- **Latency overhead:** 31.9% (acceptable)

The P2 fusion module outputs conditioning vectors that will be injected into LTX-Video. Q3 tests whether this injection disrupts temporal modeling.

### 3.3 How Conditioning Might Break Temporal Coherence

**Risk 1: Per-frame conditioning variance**
If the adapter produces slightly different conditioning for semantically-identical content, this variance propagates to frame-level inconsistencies.

**Risk 2: Override of temporal attention**
Strong conditioning signal might dominate over learned temporal priors, causing the model to generate each frame "independently."

**Risk 3: Latent space misalignment**
Hybrid encoder latents may not respect LTX-Video's temporal structure. Injecting misaligned latents could corrupt causal encoding.

**Risk 4: Conditioning injection point**
Where we inject conditioning (early/late layers, cross-attention) affects which temporal mechanisms are preserved.

### 3.4 Temporal Coherence Metric

The primary metric is **temporal_consistency**, computed as a weighted combination of:

1. **Flow Smoothness (FS)** - Optical flow acceleration variance (lower = smoother)
2. **Temporal LPIPS Variance** - Frame-to-frame perceptual consistency
3. **Identity Preservation** - DINO feature consistency for tracked objects
4. **Warping Error** - How well optical flow predicts next frame

```python
def temporal_consistency(video: torch.Tensor) -> float:
    """
    Compute temporal consistency score (0-1, higher is better).

    Components:
    - flow_smoothness: 1 - normalized_acceleration_variance
    - temporal_lpips: 1 - normalized_lpips_variance
    - identity_score: mean DINO feature correlation across frames
    - warp_accuracy: 1 - normalized_warping_error
    """
    fs = 1.0 - min(flow_acceleration_variance(video) / MAX_ACCEL, 1.0)
    tl = 1.0 - min(temporal_lpips_variance(video) / MAX_LPIPS_VAR, 1.0)
    id_score = dino_identity_correlation(video)
    warp = 1.0 - min(warping_error(video) / MAX_WARP, 1.0)

    return 0.3 * fs + 0.25 * tl + 0.25 * id_score + 0.2 * warp
```

---

## 4. Experimental Setup

### 4.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x A100 40GB | 1x A100 80GB |
| CPU RAM | 64GB | 128GB |
| Storage | 200GB SSD | 500GB NVMe |

**VRAM breakdown (inference):**
- Qwen2.5-VL-7B (bf16): ~15GB
- DINOv2-ViT-L (bf16): ~2GB
- Fusion module: ~100MB
- LTX-Video (bf16): ~8GB
- RAFT optical flow: ~2GB
- **Total: ~28GB minimum**

### 4.2 Software Dependencies

```bash
# Core dependencies
pip install torch>=2.1.0 torchvision torchaudio
pip install transformers>=4.40.0 accelerate>=0.27.0
pip install diffusers>=0.27.0
pip install flash-attn --no-build-isolation
pip install timm>=0.9.0  # DINOv2

# Temporal metrics
pip install raft-stereo  # Optical flow
pip install lpips
pip install pytorch-fid  # For FVD computation

# Utilities
pip install wandb einops matplotlib seaborn opencv-python
```

### 4.3 Model Checkpoints

```bash
# Existing models (from P2)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
huggingface-cli download Lightricks/LTX-Video

# P2 trained checkpoints (required)
# - fusion_module_best.pt (from P2)
# - dinov2_vitl14 (via torch.hub)

# Temporal metrics models
# - RAFT (optical flow) - download via torchvision
```

### 4.4 Test Datasets

**Tier 1: Static Scenes (Flickering Detection)**
- 50 static scene videos from DAVIS 2017 (minimal motion)
- Purpose: Detect conditioning-induced flickering

**Tier 2: Simple Motion**
- 100 clips from Something-Something v2 (single object motion)
- Filter: "Moving X", "Pushing X", "Lifting X"
- Purpose: Test motion smoothness and identity preservation

**Tier 3: Complex Interactions**
- 50 clips from Something-Something v2 (multi-object)
- Filter: "Pouring X into Y", "Putting X on Y"
- Purpose: Test causal consistency and physics

---

## 5. Sub-Experiments

### E-Q3.1: Baseline Temporal Coherence Measurement

**Objective:** Establish baseline temporal_consistency score for LTX-Video without our conditioning, then measure degradation with hybrid encoder conditioning.

**Protocol:**

**Phase 1: Baseline (No Conditioning)**
1. Generate 100 videos using LTX-Video with text prompts only
2. Use prompts spanning all three difficulty tiers
3. Compute temporal_consistency and component metrics
4. Record as baseline ceiling performance

**Phase 2: With Hybrid Encoder Conditioning**
1. Run same prompts through hybrid encoder pipeline
2. Generate videos with P2 fusion module conditioning
3. Compute same metrics
4. Compare to baseline

**Implementation:**

```python
def e_q3_1_baseline_measurement(runner: ExperimentRunner) -> dict:
    """
    E-Q3.1: Baseline Temporal Coherence Measurement

    Measures temporal consistency with and without conditioning.
    """
    from temporal_metrics import TemporalMetrics
    from ltx_video import LTXVideoPipeline
    from hybrid_encoder import HybridEncoderPipeline

    metrics = TemporalMetrics()
    ltx = LTXVideoPipeline()
    hybrid = HybridEncoderPipeline()  # Uses P2 fusion module

    # Test prompts (20 per tier)
    prompts = {
        'static': [...],  # 20 static scene prompts
        'motion': [...],  # 20 simple motion prompts
        'interaction': [...]  # 20 interaction prompts
    }

    results = {
        'baseline': {'static': [], 'motion': [], 'interaction': []},
        'conditioned': {'static': [], 'motion': [], 'interaction': []}
    }

    for tier, tier_prompts in prompts.items():
        for prompt in tier_prompts:
            # Baseline: text-only
            video_baseline = ltx.generate(prompt)
            tc_baseline = metrics.temporal_consistency(video_baseline)
            results['baseline'][tier].append(tc_baseline)

            # Conditioned: hybrid encoder
            video_cond = hybrid.generate(prompt)
            tc_cond = metrics.temporal_consistency(video_cond)
            results['conditioned'][tier].append(tc_cond)

    # Aggregate metrics
    baseline_mean = np.mean([v for tier in results['baseline'].values() for v in tier])
    conditioned_mean = np.mean([v for tier in results['conditioned'].values() for v in tier])
    degradation = (baseline_mean - conditioned_mean) / baseline_mean

    return {
        'finding': f'Temporal consistency: baseline={baseline_mean:.3f}, conditioned={conditioned_mean:.3f}, degradation={degradation:.1%}',
        'metrics': {
            'temporal_consistency_baseline': baseline_mean,
            'temporal_consistency_conditioned': conditioned_mean,
            'degradation_percent': degradation * 100,
            'flow_smoothness_baseline': ...,
            'flow_smoothness_conditioned': ...,
            'temporal_lpips_var_baseline': ...,
            'temporal_lpips_var_conditioned': ...,
            'identity_preservation_baseline': ...,
            'identity_preservation_conditioned': ...,
        },
        'artifacts': [
            'artifacts/baseline_vs_conditioned_comparison.png',
            'artifacts/per_tier_breakdown.json',
            'artifacts/sample_videos/'
        ]
    }
```

**Metrics:**

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| temporal_consistency (conditioned) | > 0.80 | > 0.70 | < 0.50 |
| degradation vs baseline | < 10% | < 20% | > 30% |
| flow_smoothness | > 0.85 | > 0.75 | < 0.60 |
| temporal_lpips_variance | < 0.02 | < 0.03 | > 0.05 |
| identity_preservation | > 0.85 | > 0.75 | < 0.60 |

**Analysis Questions:**
- Does conditioning degrade temporal coherence uniformly or in specific scenarios?
- Which component metric degrades most (flow, LPIPS, identity)?
- Are there prompt types where conditioning helps temporal coherence?

**Deliverables:**
- Baseline vs conditioned comparison table
- Per-tier breakdown (static/motion/interaction)
- Sample video comparisons (10 best, 10 worst)
- Component metric correlation analysis

**Time Estimate:** 3 days

---

### E-Q3.2: Conditioning Strength vs Coherence Tradeoff

**Objective:** Find the optimal conditioning strength that balances semantic control with temporal coherence, and test conditioning injection strategies.

**Protocol:**

**Phase 1: Conditioning Strength Sweep**
1. Vary conditioning strength: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
2. Generate 30 videos per strength setting
3. Measure temporal_consistency AND semantic_accuracy
4. Plot Pareto frontier

**Phase 2: Injection Strategy Ablation**
1. Test injection points: early (layers 1-7), mid (layers 8-14), late (layers 15-28)
2. Test injection methods: cross-attention, addition, concatenation
3. Test temporal spread: all frames, keyframes only, first frame only

**Implementation:**

```python
def e_q3_2_conditioning_tradeoff(runner: ExperimentRunner) -> dict:
    """
    E-Q3.2: Conditioning Strength vs Coherence Tradeoff

    Finds optimal conditioning parameters.
    """
    from temporal_metrics import TemporalMetrics
    from semantic_metrics import SemanticAccuracy
    from hybrid_encoder import HybridEncoderPipeline

    metrics = TemporalMetrics()
    semantic = SemanticAccuracy()  # CLIP similarity to prompt

    # Phase 1: Strength sweep
    strengths = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    prompts = [...]  # 30 diverse prompts

    strength_results = {}
    for strength in strengths:
        hybrid = HybridEncoderPipeline(conditioning_strength=strength)
        tc_scores = []
        sem_scores = []

        for prompt in prompts:
            video = hybrid.generate(prompt)
            tc_scores.append(metrics.temporal_consistency(video))
            sem_scores.append(semantic.score(video, prompt))

        strength_results[strength] = {
            'temporal_consistency': np.mean(tc_scores),
            'semantic_accuracy': np.mean(sem_scores),
            'combined_score': 0.5 * np.mean(tc_scores) + 0.5 * np.mean(sem_scores)
        }

    # Find optimal strength (maximize combined score while tc > 0.7)
    valid_strengths = {s: r for s, r in strength_results.items()
                       if r['temporal_consistency'] > 0.7}
    optimal_strength = max(valid_strengths.keys(),
                           key=lambda s: valid_strengths[s]['combined_score'])

    # Phase 2: Injection strategy ablation (at optimal strength)
    injection_results = {}
    for injection_point in ['early', 'mid', 'late']:
        for injection_method in ['cross_attention', 'addition']:
            hybrid = HybridEncoderPipeline(
                conditioning_strength=optimal_strength,
                injection_point=injection_point,
                injection_method=injection_method
            )
            # ... measure metrics

    return {
        'finding': f'Optimal conditioning strength: {optimal_strength}, achieves tc={strength_results[optimal_strength]["temporal_consistency"]:.3f}',
        'metrics': {
            'temporal_consistency': strength_results[optimal_strength]['temporal_consistency'],
            'optimal_strength': optimal_strength,
            'semantic_accuracy_at_optimal': strength_results[optimal_strength]['semantic_accuracy'],
            'best_injection_point': ...,
            'best_injection_method': ...,
        },
        'artifacts': [
            'artifacts/pareto_frontier.png',
            'artifacts/strength_sweep_results.json',
            'artifacts/injection_ablation_results.json',
            'artifacts/recommended_config.yaml'
        ]
    }
```

**Metrics:**

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| temporal_consistency (at optimal) | > 0.80 | > 0.70 | < 0.50 |
| semantic_accuracy (at optimal) | > 0.75 | > 0.65 | < 0.50 |
| combined_score | > 0.75 | > 0.65 | < 0.55 |

**Ablation Variables:**

| Variable | Options | Purpose |
|----------|---------|---------|
| Conditioning strength | 0.0 - 1.0 | Control semantic influence |
| Injection point | early/mid/late | Where to inject in DiT |
| Injection method | cross_attn/addition | How to combine features |
| Temporal spread | all/keyframes/first | Which frames to condition |

**Analysis Questions:**
- Is there a "sweet spot" where both temporal and semantic quality are high?
- Does injection point matter more than strength?
- Can keyframe-only conditioning preserve coherence better?

**Deliverables:**
- Pareto frontier plot (temporal vs semantic)
- Recommended configuration for production
- Ablation results table
- Guidelines document for conditioning tuning

**Time Estimate:** 4 days

---

## 6. Success Criteria

### 6.1 Primary Success Criteria (Gate 2)

| Metric | Target | Acceptable | Failure | Source |
|--------|--------|------------|---------|--------|
| **temporal_consistency** | > 0.80 | > 0.70 | < 0.50 | research_plan.yaml |

### 6.2 Secondary Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Degradation vs baseline | < 10% | Conditioning shouldn't hurt much |
| Flow smoothness | > 0.85 | No motion artifacts |
| Identity preservation | > 0.85 | Objects stay consistent |
| Semantic accuracy | > 0.70 | At optimal conditioning |

### 6.3 Comparison Baselines

| Baseline | Source | Key Metrics |
|----------|--------|-------------|
| LTX-Video (text-only) | E-Q3.1 baseline | temporal_consistency ceiling |
| P2 Hybrid Encoder | P2 results | lpips=0.162, spatial_iou=0.837 |

### 6.4 Go/No-Go Decision Points

| Checkpoint | Timing | Criteria | Decision |
|------------|--------|----------|----------|
| E-Q3.1 Baseline | Day 3 | Baseline temporal_consistency > 0.85 | Continue / Debug LTX |
| E-Q3.1 Conditioned | Day 3 | Conditioned temporal_consistency > 0.50 | Continue / Major issue |
| E-Q3.2 Sweep | Day 5 | At least one strength achieves > 0.70 | Continue / Pivot |
| E-Q3.2 Complete | Day 7 | Optimal config achieves > 0.70 | Pass Gate 2 / Investigate |

---

## 7. Failure Criteria

### 7.1 Hard Failures (Abandon Approach)

1. **Catastrophic temporal degradation:** Conditioned temporal_consistency < 0.50 at ALL conditioning strengths
   - Implication: Fundamental incompatibility between hybrid encoder and LTX temporal modeling
   - Action: Consider alternative video decoder or completely different conditioning approach

2. **No viable operating point:** Cannot achieve temporal_consistency > 0.70 AND semantic_accuracy > 0.60 simultaneously
   - Implication: Tradeoff is too severe for practical use
   - Action: Pivot to post-processing temporal smoothing or keyframe-only approach

3. **Severe flickering at all settings:** Visual flickering makes videos unwatchable regardless of parameters
   - Implication: Conditioning injection introduces irremovable noise
   - Action: Redesign conditioning pathway

### 7.2 Soft Failures (Investigation Required)

1. **Marginal temporal coherence:** temporal_consistency = 0.60-0.70
   - May need conditioning loss modification or injection refinement

2. **Scene-type sensitivity:** Good coherence on static, poor on interactions
   - May need scene-adaptive conditioning strength

3. **Component metric imbalance:** Flow good but identity poor (or vice versa)
   - May need targeted fixes for specific failure modes

---

## 8. Pivot Options

If Q3 shows unacceptable temporal degradation:

### Pivot A: Reduced Conditioning Strength
**Strategy:** Use minimal conditioning (0.1-0.25) for semantic guidance only
**Tradeoff:** Less semantic control, preserves video decoder's learned priors
**Implementation:** Scale adapter output to 10-25%

### Pivot B: Temporal Smoothing Post-Processing
**Strategy:** Apply temporal smoothing after generation
**Options:**
- Optical flow-based interpolation
- Temporal Gaussian filtering in latent space
- Frame blending
**Tradeoff:** May blur details, adds latency

### Pivot C: Keyframe-Only Conditioning
**Strategy:** Condition only on keyframes (every 8th frame), let video decoder interpolate
**Tradeoff:** Less frame-level control, preserves inter-frame coherence
**Implementation:** Sparse conditioning mask in temporal dimension

### Pivot D: Temporal Consistency Training Loss
**Strategy:** Retrain adapter with explicit temporal consistency loss
**Loss function:**
```python
def temporal_consistency_loss(generated_video):
    flow_loss = flow_smoothness_penalty(generated_video)
    lpips_loss = consecutive_frame_lpips(generated_video)
    return flow_loss + 0.5 * lpips_loss
```
**Tradeoff:** Requires adapter retraining, longer experiment

### Pivot E: Alternative Video Decoder
**Strategy:** Switch to video decoder with stronger temporal priors
**Options:**
- CogVideoX (slower but potentially more robust)
- HunyuanVideo-1.5 (higher quality, 75s/clip)
**Tradeoff:** Different conditioning interface, higher latency

---

## 9. Timeline

| Day | Experiment | Deliverables |
|-----|------------|--------------|
| 1 | Setup, metrics implementation | Temporal metrics working |
| 2-3 | E-Q3.1: Baseline + conditioned | Baseline comparison |
| 4-5 | E-Q3.2: Strength sweep | Pareto frontier |
| 6-7 | E-Q3.2: Injection ablation | Optimal config |
| 8 | Analysis + writeup | Final report |

**Total Estimated Time:** 8 days

**Critical Path:** Setup -> E-Q3.1 -> E-Q3.2 (sequential dependency)

**Parallelization:** Phase 2 of E-Q3.2 can overlap with Phase 1 analysis

---

## 10. Resource Requirements

### 10.1 Compute

| Phase | GPU Type | GPU-Hours | Notes |
|-------|----------|-----------|-------|
| Setup | A100-40GB | 5 | Metrics validation |
| E-Q3.1 Baseline | A100-40GB | 30 | 100 videos, ~18min each |
| E-Q3.1 Conditioned | A100-40GB | 30 | 100 videos with hybrid |
| E-Q3.2 Sweep | A100-40GB | 50 | 6 strengths x 30 videos |
| E-Q3.2 Ablation | A100-40GB | 40 | Injection point variations |
| **Total** | | **155** | |

**Estimated Cost:** ~$310 (at $2/GPU-hour)

### 10.2 Storage

| Item | Size | Notes |
|------|------|-------|
| Generated videos | ~50GB | 500 videos @ ~100MB each |
| Metrics cache | ~5GB | Pre-computed features |
| Checkpoints | ~200MB | P2 fusion module |
| Results/artifacts | ~2GB | Plots, JSON, samples |
| **Total** | ~58GB | |

---

## 11. Dependencies

### 11.1 Prerequisites (Must Complete Before Starting)

- [x] P2 Hybrid Encoder passed (spatial_iou=0.837, lpips=0.162)
- [x] P2 fusion module checkpoint available
- [x] LTX-Video inference working
- [ ] RAFT optical flow model downloaded
- [ ] LPIPS model loaded
- [ ] DINO model for identity tracking

### 11.2 Blocks

This experiment blocks:
- **C3 (Future Prediction):** Needs temporally coherent video generation
- **Gate 2:** Cannot pass without Q3 completing

### 11.3 External Dependencies

- RAFT model availability (via torchvision or hub)
- Something-Something v2 validation set
- DAVIS 2017 dataset (optional, for static scenes)

---

## 12. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Conditioning causes severe flickering | Medium (30%) | High | Test multiple injection points, fall back to keyframes |
| No viable semantic-temporal tradeoff | Low (15%) | High | Accept lower semantic accuracy, focus on temporal |
| Metrics don't correlate with perception | Medium (25%) | Medium | Add human evaluation spot-checks |
| LTX-Video baseline is poor | Low (10%) | High | Debug video decoder first, consider alternatives |
| Compute budget insufficient | Low (10%) | Medium | Prioritize strength sweep, defer ablations |

---

## 13. Deliverables

### 13.1 Code Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Temporal metrics | `infra/modal/handlers/q3/temporal_metrics.py` | Flow, LPIPS, identity metrics |
| E-Q3.1 handler | `infra/modal/handlers/q3/e_q3_1.py` | Baseline measurement |
| E-Q3.2 handler | `infra/modal/handlers/q3/e_q3_2.py` | Strength/injection ablation |
| Evaluation script | `scripts/evaluate_temporal.py` | Standalone evaluation |

### 13.2 Checkpoints

| Checkpoint | Size | Description |
|------------|------|-------------|
| P2 fusion module | ~100MB | Required input from P2 |
| Metrics cache | ~500MB | Pre-computed RAFT/DINO features |

### 13.3 Reports

| Report | Format | Audience |
|--------|--------|----------|
| Technical findings | `research/experiments/q3-temporal-coherence/FINDINGS.md` | Research team |
| Results data | `research/experiments/q3-temporal-coherence/results.yaml` | Validation system |
| Conditioning guidelines | `research/experiments/q3-temporal-coherence/CONDITIONING_GUIDE.md` | Implementation |

### 13.4 Decision Document

Final assessment with:
- Gate 2 pass/fail determination (Q3 component)
- Recommended conditioning configuration
- Known failure modes and mitigations
- Implications for C3 (Future Prediction)

---

## 14. Open Questions

To be resolved during experiments:

1. **Baseline quality:** What is LTX-Video's baseline temporal_consistency? (Need to establish ceiling)
2. **Injection point impact:** Does early vs late injection affect temporal coherence differently?
3. **Scene-type sensitivity:** Do static scenes behave differently than motion scenes?
4. **Metric correlation:** Do automated metrics correlate with human perception of temporal quality?
5. **Semantic-temporal coupling:** Is there a principled way to decouple semantic control from temporal coherence?

---

## 15. Related Documents

- [P2 Hybrid Encoder Results](./p2-hybrid-encoder.md) - Foundation for Q3
- [C2 Adapter Bridging Plan](./c2-adapter-bridging.md) - Parallel Gate 2 experiment
- [Agent Guide](../AGENT_GUIDE.md) - How to run experiments
- [Research Plan](../research_plan.yaml) - Success criteria source

---

## Appendix A: Temporal Metrics Implementation

### A.1 Flow Smoothness Score

```python
def flow_smoothness_score(video: torch.Tensor) -> float:
    """
    Measure temporal smoothness via optical flow acceleration.

    Lower acceleration variance = smoother motion.
    Returns score 0-1 (higher = smoother).

    Args:
        video: (T, C, H, W) tensor

    Returns:
        Smoothness score normalized to [0, 1]
    """
    T = video.shape[0]

    # Compute optical flow for consecutive frames using RAFT
    flows = []
    for t in range(T - 1):
        flow = compute_optical_flow(video[t], video[t + 1])  # (2, H, W)
        flows.append(flow)

    # Compute velocity (flow difference)
    velocities = [flows[t + 1] - flows[t] for t in range(len(flows) - 1)]

    # Compute acceleration (velocity difference)
    accelerations = [velocities[t + 1] - velocities[t]
                     for t in range(len(velocities) - 1)]

    # Score = normalized acceleration magnitude
    acc_magnitudes = [torch.norm(a, dim=0).mean() for a in accelerations]
    mean_acc = torch.stack(acc_magnitudes).mean().item()

    # Normalize: MAX_ACCEL empirically set to 5.0 for typical videos
    MAX_ACCEL = 5.0
    return max(0, 1.0 - mean_acc / MAX_ACCEL)
```

### A.2 Temporal LPIPS Variance

```python
def temporal_lpips_variance(video: torch.Tensor) -> float:
    """
    Measure variance of LPIPS between consecutive frames.

    High variance = inconsistent frame-to-frame appearance.
    Returns variance (lower = more consistent).
    """
    lpips_model = LPIPS(net='alex')

    scores = []
    for t in range(len(video) - 1):
        score = lpips_model(video[t:t+1], video[t+1:t+2])
        scores.append(score.item())

    return np.var(scores)
```

### A.3 Identity Preservation Score

```python
def identity_preservation_score(video: torch.Tensor) -> float:
    """
    Measure object appearance consistency using DINO features.

    High correlation = consistent object appearance.
    Returns mean pairwise correlation (higher = better).
    """
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    features = []
    for t in range(len(video)):
        feat = dino(video[t:t+1])  # (1, D)
        features.append(feat.squeeze())

    features = torch.stack(features)  # (T, D)

    # Compute pairwise correlations
    correlations = []
    for t in range(len(features) - 1):
        corr = F.cosine_similarity(features[t:t+1], features[t+1:t+2])
        correlations.append(corr.item())

    return np.mean(correlations)
```

### A.4 Combined Temporal Consistency

```python
def temporal_consistency(video: torch.Tensor) -> float:
    """
    Combined temporal consistency score.

    Weighted combination of:
    - Flow smoothness (30%)
    - Temporal LPIPS variance (25%)
    - Identity preservation (25%)
    - Warping accuracy (20%)

    Returns score 0-1 (higher = better temporal consistency).
    """
    # Normalization constants (empirically determined)
    MAX_LPIPS_VAR = 0.05
    MAX_WARP_ERROR = 0.15

    fs = flow_smoothness_score(video)
    tl = 1.0 - min(temporal_lpips_variance(video) / MAX_LPIPS_VAR, 1.0)
    id_score = identity_preservation_score(video)
    warp = 1.0 - min(warping_error(video) / MAX_WARP_ERROR, 1.0)

    return 0.30 * fs + 0.25 * tl + 0.25 * id_score + 0.20 * warp
```

---

## Appendix B: Test Video Prompts

### Static Scenes (Tier 1)

```yaml
static_prompts:
  - "A cozy living room with a fireplace, no movement"
  - "A still mountain lake reflecting snow-capped peaks"
  - "A bowl of fresh fruit on a kitchen counter"
  - "An empty classroom with rows of desks"
  - "A garden with flowers on a windless day"
```

### Simple Motion (Tier 2)

```yaml
motion_prompts:
  - "A red ball rolling slowly from left to right"
  - "A person walking forward down a hallway"
  - "A car driving straight on an empty road"
  - "A bird flying across a clear sky"
  - "A pendulum swinging back and forth"
```

### Complex Interactions (Tier 3)

```yaml
interaction_prompts:
  - "A hand pouring water from a pitcher into a glass"
  - "A person stacking wooden blocks"
  - "Two billiard balls colliding on a table"
  - "A cat jumping onto a table"
  - "Dominoes falling in a chain reaction"
```

---

## Appendix C: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-18 | Claude | Initial draft |
| 0.2 | 2026-01-20 | Claude | Updated to align with research_plan.yaml, P2 results, and 2-sub-experiment structure |
