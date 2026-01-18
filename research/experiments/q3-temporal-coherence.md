# Q3: Temporal Coherence Research Plan

**Open Question:** Can we generate temporally coherent video (not just good individual frames)?

**Risk Level:** MEDIUM - Video decoders handle temporal modeling inherently, but our VLM conditioning injection may disrupt learned temporal dynamics.

**Status:** Not Started
**Created:** 2025-01-18
**Dependencies:** Working conditioning adapter (from C2 experiments)

---

## 1. Objective

Ensure that videos generated through our GLP architecture maintain:

1. **Temporal smoothness** - No flickering, jitter, or abrupt frame-to-frame changes
2. **Motion continuity** - Objects follow physically plausible trajectories
3. **Causal consistency** - Effects follow causes (ball hits cup → cup moves)
4. **Identity preservation** - Objects maintain consistent appearance across frames

The key challenge: LTX-Video has sophisticated temporal modeling via its causal 3D convolutions and temporal attention. Injecting VLM conditioning through our adapter risks disrupting these learned temporal priors.

---

## 2. Background

### 2.1 How LTX-Video Maintains Temporal Consistency

LTX-Video employs several mechanisms for temporal coherence:

1. **Causal 3D Convolutions in VAE**
   - Encoder uses causal temporal convolutions (8x temporal compression)
   - First frame encoded independently; subsequent frames conditioned on previous
   - This creates an autoregressive structure in latent space

2. **Temporal Attention in DiT**
   - The 28-block DiT uses spatiotemporal attention
   - Tokens attend across both spatial and temporal dimensions
   - RoPE positional encoding with exponential frequency spacing preserves temporal relationships

3. **Joint Denoising**
   - All frames denoised together (not independently)
   - Global noise schedule ensures consistent denoising trajectory
   - Denoising decoder operates on temporally-coherent latents

4. **High Channel Latents (128 channels)**
   - More channels = more capacity to encode temporal relationships
   - Information flows through latent space rather than being lost

### 2.2 Common Failure Modes

| Failure Mode | Description | Likely Cause |
|-------------|-------------|--------------|
| **Flickering** | High-frequency intensity/color oscillation between frames | Inconsistent conditioning, noise injection artifacts |
| **Identity drift** | Objects gradually change appearance | Weak temporal attention, conditioning override |
| **Discontinuous motion** | Objects "teleport" between frames | Broken temporal consistency in latent space |
| **Physics violations** | Impossible trajectories (objects passing through each other) | Conditioning conflicts with learned physics priors |
| **Temporal aliasing** | Motion appears jerky/stuttered | Temporal downsampling artifacts |
| **Causal violations** | Effects precede causes | Bidirectional attention bleeding |

### 2.3 How Our Conditioning Might Break Temporal Coherence

**Risk 1: Per-frame conditioning variance**
If our adapter produces slightly different conditioning for semantically-identical content, this variance propagates to frame-level inconsistencies.

**Risk 2: Override of temporal attention**
Strong conditioning signal might dominate over learned temporal priors, causing the model to generate each frame "independently" rather than coherently.

**Risk 3: Latent space misalignment**
VLM latents may not respect the temporal structure of LTX-Video's latent space. Injecting misaligned latents could corrupt the causal encoding.

**Risk 4: Conditioning injection point**
Where we inject conditioning (early layers, late layers, cross-attention) affects which temporal mechanisms are preserved or disrupted.

### 2.4 Metrics for Temporal Coherence

| Metric | What It Measures | Implementation |
|--------|-----------------|----------------|
| **Flow Smoothness (FS)** | Consistency of optical flow between frames | Compute optical flow (RAFT), measure flow acceleration variance |
| **Temporal LPIPS** | Perceptual difference between consecutive frames | LPIPS(frame_t, frame_t+1) variance over video |
| **Frechet Video Distance (FVD)** | Overall video quality including temporal aspects | I3D features, compare to reference distribution |
| **Warping Error** | Whether optical flow accurately predicts next frame | Warp frame_t by flow, compare to frame_t+1 |
| **CLIP-Temporal** | Semantic consistency across frames | CLIP embedding variance across frames |
| **Motion Consistency Score (MCS)** | Whether motion follows physical laws | Optical flow trajectory analysis |
| **Identity Preservation (DINO-track)** | Object appearance consistency | DINO features for tracked objects |

---

## 3. Experimental Setup

### 3.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 24GB (A10/3090) | 40GB+ (A100/A6000) |
| System RAM | 32GB | 64GB |
| Storage | 100GB (datasets + models) | 500GB (full experiments) |
| Compute time | 2-3 days | 1 week |

**Note:** Full VLM + video decoder requires ~40GB VRAM. For initial experiments with smaller conditioning adapter, 24GB may suffice.

### 3.2 Temporal Metrics Implementation

Create `src/metrics/temporal_coherence.py`:

```python
# Pseudo-code for core metrics

class TemporalMetrics:
    def __init__(self):
        self.flow_model = RAFT()  # Optical flow
        self.lpips = LPIPS()
        self.i3d = InceptionI3D()  # For FVD
        self.dino = DINOv2()

    def flow_smoothness(self, video: Tensor) -> float:
        """Measure variance in optical flow acceleration."""
        flows = [self.flow_model(video[t], video[t+1])
                 for t in range(len(video)-1)]
        velocities = [flow.mean(dim=(1,2)) for flow in flows]
        accelerations = [v2 - v1 for v1, v2 in zip(velocities[:-1], velocities[1:])]
        return torch.stack(accelerations).var().item()

    def temporal_lpips(self, video: Tensor) -> Tuple[float, float]:
        """Mean and variance of consecutive frame LPIPS."""
        scores = [self.lpips(video[t], video[t+1])
                  for t in range(len(video)-1)]
        return torch.stack(scores).mean(), torch.stack(scores).var()

    def warping_error(self, video: Tensor) -> float:
        """Error when warping frame by optical flow."""
        errors = []
        for t in range(len(video)-1):
            flow = self.flow_model(video[t], video[t+1])
            warped = warp(video[t], flow)
            errors.append(F.l1_loss(warped, video[t+1]))
        return torch.stack(errors).mean().item()

    def fvd(self, generated: List[Tensor], reference: List[Tensor]) -> float:
        """Frechet Video Distance using I3D features."""
        gen_feats = [self.i3d(v) for v in generated]
        ref_feats = [self.i3d(v) for v in reference]
        return compute_frechet_distance(gen_feats, ref_feats)

    def identity_preservation(self, video: Tensor,
                               object_tracks: List[BBox]) -> float:
        """DINO feature consistency for tracked objects."""
        features_per_object = []
        for obj_track in object_tracks:
            obj_features = []
            for t, bbox in enumerate(obj_track):
                crop = video[t, :, bbox.y1:bbox.y2, bbox.x1:bbox.x2]
                obj_features.append(self.dino(crop))
            features_per_object.append(torch.stack(obj_features))

        # Measure feature variance per object
        variances = [f.var(dim=0).mean() for f in features_per_object]
        return torch.tensor(variances).mean().item()
```

### 3.3 Test Scenarios

We evaluate temporal coherence across three difficulty tiers:

**Tier 1: Static Scenes**
- Camera stationary, minimal object motion
- Tests: flickering, color drift, background stability
- Examples: Indoor room, landscape view, still life

**Tier 2: Simple Motion**
- Single object moving, predictable trajectory
- Tests: motion smoothness, identity preservation, trajectory physics
- Examples: Ball rolling, person walking, car driving straight

**Tier 3: Complex Interactions**
- Multiple objects, interactions, occlusions
- Tests: causal consistency, collision physics, identity through occlusion
- Examples: Pouring water, stacking blocks, two people passing

### 3.4 Datasets

| Dataset | Use Case | Temporal Challenge |
|---------|----------|-------------------|
| **DAVIS 2017** | Object tracking benchmark | Identity preservation through motion |
| **Something-Something v2** | Object interactions | Causal temporal relationships |
| **Kinetics-400** | Diverse motion | General motion quality |
| **Custom static videos** | Flickering detection | Minimal motion baseline |
| **Physics simulation renders** | Physics accuracy | Known ground-truth dynamics |

---

## 4. Experiments

### E-Q3.1: Baseline LTX-Video Temporal Coherence

**Objective:** Establish baseline temporal quality of LTX-Video without our conditioning.

**Method:**
1. Generate 100 videos using LTX-Video with text prompts only
2. Use prompts spanning all three difficulty tiers
3. Compute all temporal metrics
4. Human evaluation (n=20 raters, 5-point scale for smoothness)

**Prompts:**
```
# Tier 1 (Static)
"A peaceful living room with sunlight streaming through windows"
"A still life of fruit on a wooden table"

# Tier 2 (Simple Motion)
"A red ball rolling across a wooden floor"
"A person walking down a hallway"

# Tier 3 (Complex)
"A hand pouring water from a pitcher into a glass"
"Two people shaking hands"
```

**Metrics recorded:**
- Flow smoothness: mean, std across videos
- Temporal LPIPS: mean, variance per video
- FVD: against reference videos
- Warping error: per-tier breakdown
- Human smoothness rating: mean, confidence interval

**Expected outcome:** Establish "ceiling" performance for temporal coherence.

**Time estimate:** 1 day (generation + metrics + human eval setup)

---

### E-Q3.2: With Conditioning - Static Scenes

**Objective:** Test if our conditioning disrupts temporal coherence in the easiest case.

**Method:**
1. Select 20 static scene videos from validation set
2. Encode with VLM, generate conditioning through adapter
3. Generate videos using LTX-Video + our conditioning
4. Compare to baseline (E-Q3.1) on same prompts

**Experimental conditions:**
| Condition | Description |
|-----------|-------------|
| A | Baseline (no conditioning) |
| B | Conditioning strength = 0.25 |
| C | Conditioning strength = 0.5 |
| D | Conditioning strength = 1.0 |

**Key metrics:**
- Flickering score (high-frequency intensity variance)
- Background stability (LPIPS of background regions)
- Color drift (mean color shift over time)

**Success criterion:** Condition D (full conditioning) shows <10% degradation vs baseline on flickering score.

**Analysis:**
- If flickering increases with conditioning strength → adapter introduces noise
- If background becomes unstable → conditioning leaks spatial information
- If color drifts → adapter has temporal inconsistency

**Time estimate:** 1 day

---

### E-Q3.3: With Conditioning - Simple Motion

**Objective:** Test motion quality with single moving objects.

**Method:**
1. Use Something-Something v2 validation set (single-object actions)
2. Filter for simple motion: "Moving X from left to right", "Pushing X"
3. Generate conditioned predictions for 50 video clips
4. Measure motion-specific temporal metrics

**Test cases:**
- Linear motion (pushing object across table)
- Curved motion (rolling ball)
- Vertical motion (lifting object)
- Rotational motion (turning object)

**Key metrics:**
- Flow smoothness score
- Trajectory linearity (for linear motion cases)
- Identity preservation (DINO consistency of moving object)
- Motion blur consistency

**Comparison:**
- Our conditioning vs baseline LTX-Video
- Our conditioning vs ground truth video

**Success criterion:**
- Flow smoothness within 20% of baseline
- Identity preservation (DINO variance) < 0.15

**Time estimate:** 2 days

---

### E-Q3.4: With Conditioning - Object Interactions

**Objective:** Test the hardest case: multi-object interactions with causal relationships.

**Method:**
1. Use Something-Something v2 validation set (interactions)
2. Select: "Pouring X into Y", "Putting X on Y", "X colliding with Y"
3. Generate conditioned predictions for 50 clips
4. Measure interaction-specific metrics

**Test cases:**
- Contact interactions (putting object on surface)
- Transfer interactions (pouring liquid)
- Collision interactions (objects hitting each other)
- Occlusion interactions (object passing behind another)

**Key metrics:**
- Causal consistency score (effects follow causes)
- Collision physics score (objects don't pass through each other)
- Post-occlusion identity (object appears correctly after occlusion)

**Causal consistency evaluation:**
```python
def causal_consistency_score(video, event_frames):
    """
    Check if effects happen AFTER causes.
    event_frames: dict mapping event_name -> frame_number
    """
    # Example: pour_start should precede liquid_appears
    violations = 0
    if event_frames['liquid_appears'] < event_frames['pour_start']:
        violations += 1
    # ... more causal checks
    return 1.0 - (violations / total_checks)
```

**Success criterion:**
- Causal consistency > 0.8
- Physics violations < 15% of clips
- Identity through occlusion correlation > 0.85

**Time estimate:** 3 days

---

### E-Q3.5: Ablation - Conditioning Strength vs Coherence Tradeoff

**Objective:** Find the optimal conditioning strength that balances semantic control with temporal coherence.

**Method:**
Systematically vary conditioning strength and measure both:
- Semantic accuracy (does output match intended content?)
- Temporal coherence (is motion smooth and consistent?)

**Experimental design:**
```
Conditioning strengths: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
Videos per strength: 30
Metrics: flow_smoothness, temporal_lpips, semantic_similarity
```

**Analysis:**
```python
# Expected tradeoff curve
# semantic_accuracy ↑ as conditioning ↑
# temporal_coherence ↓ as conditioning ↑
# Goal: find elbow point

def find_optimal_strength(results):
    """
    Find conditioning strength that maximizes combined score.
    """
    scores = []
    for strength, metrics in results.items():
        combined = (
            metrics['semantic_accuracy'] * 0.6 +
            metrics['temporal_coherence'] * 0.4
        )
        scores.append((strength, combined))
    return max(scores, key=lambda x: x[1])[0]
```

**Ablation sub-experiments:**
- E-Q3.5a: Vary conditioning injection layer (early/mid/late)
- E-Q3.5b: Vary conditioning temporal spread (all frames vs keyframes)
- E-Q3.5c: Conditioning interpolation (smooth transition vs sharp)

**Deliverables:**
- Pareto frontier plot: semantic accuracy vs temporal coherence
- Recommended conditioning strength per use case
- Architectural insights on conditioning injection

**Time estimate:** 3 days

---

### E-Q3.6: Human Evaluation of Motion Smoothness

**Objective:** Validate automated metrics with human perception of temporal quality.

**Method:**
1. Select 50 video pairs: (baseline, ours) matched by content
2. Conduct A/B preference study
3. Additional 5-point rating scale for specific attributes

**Study design:**
- Participants: 30 (mix of ML researchers and non-technical)
- Platform: Custom web interface or Prolific
- Compensation: Standard rate for 20-minute study

**Questions:**
1. "Which video has smoother motion?" (forced choice)
2. "Rate the motion smoothness" (1-5 scale)
3. "Rate the physical plausibility" (1-5 scale)
4. "Does anything look wrong?" (free text)

**Controls:**
- Include baseline vs baseline comparisons (attention check)
- Randomize left/right placement
- Include known-bad examples (temporal glitches)

**Analysis:**
- Preference rate: % preferring ours vs baseline
- Correlation: human ratings vs automated metrics
- Failure mode identification: qualitative analysis of "wrong" responses

**Success criterion:**
- Preference rate > 40% (not significantly worse than baseline)
- Correlation with automated metrics > 0.7

**Time estimate:** 1 week (including participant recruitment)

---

## 5. Success Metrics

### Quantitative Thresholds

| Metric | Acceptable | Good | Excellent |
|--------|------------|------|-----------|
| Flow smoothness degradation | <25% vs baseline | <15% | <5% |
| Temporal LPIPS variance | <0.05 | <0.03 | <0.02 |
| FVD | <150 | <100 | <75 |
| Warping error | <0.08 | <0.05 | <0.03 |
| Identity preservation (DINO) | >0.7 | >0.8 | >0.9 |
| Human preference rate | >35% | >45% | >50% |
| Physics violations | <20% | <10% | <5% |

### Pass/Fail Criteria

**PASS:** All metrics in "Acceptable" range, human preference >40%
**CONDITIONAL PASS:** Most metrics acceptable, clear path to improvement
**FAIL:** Multiple metrics in unacceptable range, no clear mitigation

---

## 6. Failure Criteria

The temporal coherence research is considered FAILED if:

### Hard Failures (Abandon approach)

1. **Catastrophic flickering** - Conditioning causes severe flickering (>3x baseline) that cannot be reduced by lowering conditioning strength

2. **Fundamental incompatibility** - VLM latent structure fundamentally incompatible with LTX-Video's temporal modeling, requiring >100M parameter adapter to achieve baseline performance

3. **Physics violations at all strengths** - Even minimal conditioning (0.1) causes physics violations in >50% of interaction videos

### Soft Failures (Pivot required)

1. **Unacceptable tradeoff curve** - Cannot achieve semantic accuracy >0.6 while maintaining temporal coherence >0.7

2. **Human rejection** - Human preference rate <25% across all conditions

3. **Metric divergence** - Automated metrics show "acceptable" but humans consistently rate videos as "bad"

---

## 7. Pivot Options

If temporal coherence experiments show unacceptable degradation:

### Pivot A: Reduce Conditioning Strength

**Strategy:** Use minimal conditioning (0.1-0.25) for temporal guidance only
**Tradeoff:** Less semantic control, but preserves video decoder's learned priors
**Implementation:** Scale adapter output before injection

### Pivot B: Temporal Smoothing Post-Processing

**Strategy:** Apply temporal smoothing after generation
**Options:**
- Optical flow-based interpolation
- Temporal Gaussian filtering in latent space
- Frame blending

**Tradeoff:** May blur details, adds latency
**Implementation:**
```python
def temporal_smooth(video, sigma=1.0):
    """Apply temporal Gaussian filter."""
    kernel = gaussian_kernel_1d(sigma)
    return F.conv1d(video.permute(0,2,3,1), kernel)
```

### Pivot C: Keyframe-Only Conditioning

**Strategy:** Condition only on keyframes (every 8th frame), let video decoder interpolate
**Tradeoff:** Less frame-level control, but preserves inter-frame coherence
**Implementation:** Sparse conditioning mask

### Pivot D: Temporal Consistency Loss

**Strategy:** Add explicit temporal consistency loss during adapter training
**Loss function:**
```python
def temporal_consistency_loss(generated_video):
    """Penalize temporal discontinuities."""
    flow_loss = flow_smoothness_penalty(generated_video)
    lpips_loss = consecutive_frame_lpips(generated_video)
    return flow_loss + 0.5 * lpips_loss
```
**Tradeoff:** Requires adapter retraining, may reduce semantic fidelity

### Pivot E: Alternative Video Decoder

**Strategy:** Switch to video decoder with stronger temporal priors
**Options:**
- CogVideoX (slower but potentially more robust)
- Custom fine-tuned LTX-Video with temporal consistency emphasis
- HunyuanVideo-1.5 (higher quality, 75s/clip)

**Tradeoff:** Slower inference, different conditioning interface

---

## 8. Timeline

| Day | Experiment | Deliverables |
|-----|------------|--------------|
| 1 | E-Q3.1 Baseline measurement | Baseline metrics, human eval setup |
| 2 | E-Q3.2 Static scenes | Static scene coherence report |
| 3-4 | E-Q3.3 Simple motion | Motion quality analysis |
| 5-7 | E-Q3.4 Object interactions | Interaction coherence report |
| 8-10 | E-Q3.5 Conditioning ablation | Optimal strength recommendation |
| 11-14 | E-Q3.6 Human evaluation | Human preference study results |
| 15-16 | Analysis & writeup | Final temporal coherence report |

**Total estimated time:** 2-3 weeks

**Critical path:** E-Q3.1 → E-Q3.2 → E-Q3.3 → E-Q3.4 (sequential, builds complexity)

**Parallelizable:** E-Q3.5 can partially overlap with E-Q3.4; E-Q3.6 can start once E-Q3.3 completes

---

## 9. Dependencies

### Required Before Starting

1. **Working conditioning adapter (C2)** - Must have trained adapter that can inject VLM latents into LTX-Video
2. **LTX-Video inference working** - Baseline generation must be functional
3. **Optical flow model (RAFT)** - For flow-based metrics
4. **LPIPS model** - For perceptual similarity
5. **I3D model** - For FVD computation
6. **DINOv2 model** - For identity tracking

### Software Dependencies

```python
# requirements-temporal.txt
torch>=2.0
diffusers>=0.25
transformers>=4.36
opencv-python  # Video I/O
raft-stereo    # Optical flow
lpips
pytorch-fid    # For FVD computation
timm           # For I3D/DINO models
```

### Data Dependencies

- Something-Something v2 validation set (or subset)
- DAVIS 2017 validation set
- Custom static scene test videos (to be created)

---

## 10. Deliverables

### Primary Deliverables

1. **Temporal Coherence Benchmark**
   - Curated test set (150 videos across 3 difficulty tiers)
   - Automated evaluation pipeline
   - Baseline scores for LTX-Video
   - Scores for our method at various conditioning strengths

2. **Conditioning Guidelines Document**
   - Recommended conditioning strength per use case
   - Architectural recommendations (injection point, temporal spread)
   - Known failure modes and mitigations

3. **Metrics Implementation**
   - `src/metrics/temporal_coherence.py` - All temporal metrics
   - `scripts/evaluate_temporal.py` - Evaluation script
   - Unit tests with known-good/bad video examples

### Secondary Deliverables

4. **Human Evaluation Protocol**
   - Study design document
   - Web interface code
   - Analysis scripts
   - Raw and processed human ratings

5. **Ablation Study Results**
   - Pareto frontier visualization
   - Conditioning strength sweep data
   - Injection point comparison

6. **Failure Mode Catalog**
   - Video examples of each failure mode
   - Automatic detection heuristics
   - Mitigation strategies per failure type

### Documentation

7. **Technical Report**
   - Full methodology description
   - Results with statistical analysis
   - Comparison to prior work
   - Limitations and future work

---

## Appendix A: Detailed Metric Definitions

### A.1 Flow Smoothness Score

```python
def flow_smoothness_score(video: torch.Tensor) -> float:
    """
    Measure temporal smoothness via optical flow acceleration.

    Lower score = smoother motion

    Args:
        video: (T, C, H, W) tensor

    Returns:
        Smoothness score (0-1, lower is smoother)
    """
    T = video.shape[0]

    # Compute optical flow for consecutive frames
    flows = []
    for t in range(T - 1):
        flow = compute_optical_flow(video[t], video[t + 1])  # (2, H, W)
        flows.append(flow)

    # Compute flow differences (velocity)
    velocities = [flows[t + 1] - flows[t] for t in range(len(flows) - 1)]

    # Compute acceleration (change in velocity)
    accelerations = [velocities[t + 1] - velocities[t]
                     for t in range(len(velocities) - 1)]

    # Score = mean magnitude of acceleration
    acc_magnitudes = [torch.norm(a, dim=0).mean() for a in accelerations]

    return torch.stack(acc_magnitudes).mean().item()
```

### A.2 Frechet Video Distance

```python
def compute_fvd(generated_videos: List[torch.Tensor],
                 reference_videos: List[torch.Tensor],
                 i3d_model: nn.Module) -> float:
    """
    Compute Frechet Video Distance using I3D features.

    Args:
        generated_videos: List of (T, C, H, W) tensors
        reference_videos: List of (T, C, H, W) tensors
        i3d_model: Pretrained I3D model

    Returns:
        FVD score (lower is better)
    """
    # Extract I3D features
    gen_features = []
    ref_features = []

    for video in generated_videos:
        # I3D expects (B, C, T, H, W)
        video = video.permute(1, 0, 2, 3).unsqueeze(0)
        features = i3d_model(video)  # (1, feature_dim)
        gen_features.append(features.squeeze())

    for video in reference_videos:
        video = video.permute(1, 0, 2, 3).unsqueeze(0)
        features = i3d_model(video)
        ref_features.append(features.squeeze())

    gen_features = torch.stack(gen_features)  # (N_gen, feature_dim)
    ref_features = torch.stack(ref_features)  # (N_ref, feature_dim)

    # Compute statistics
    mu_gen, sigma_gen = gen_features.mean(0), torch_cov(gen_features)
    mu_ref, sigma_ref = ref_features.mean(0), torch_cov(ref_features)

    # Frechet distance
    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)

    fvd = (diff @ diff +
           torch.trace(sigma_gen + sigma_ref - 2 * covmean))

    return fvd.item()
```

---

## Appendix B: Test Video Prompts

### Static Scenes (Tier 1)

```yaml
static_prompts:
  - "A cozy living room with a fireplace, no people or pets"
  - "A still mountain lake reflecting snow-capped peaks"
  - "A bowl of fresh fruit on a kitchen counter"
  - "An empty classroom with rows of desks"
  - "A garden with flowers, no wind"
  - "A bookshelf filled with colorful books"
  - "A parking lot with parked cars, no movement"
  - "An office desk with computer and lamp"
  - "A bathroom with marble countertops"
  - "A beach at sunset with calm water"
```

### Simple Motion (Tier 2)

```yaml
motion_prompts:
  - "A red ball rolling slowly from left to right"
  - "A person walking forward down a hallway"
  - "A car driving straight on an empty road"
  - "A bird flying across a clear sky"
  - "A balloon floating upward"
  - "A toy train moving on circular tracks"
  - "A leaf falling from a tree"
  - "A pendulum swinging back and forth"
  - "A person waving their hand"
  - "A dog running across a field"
```

### Complex Interactions (Tier 3)

```yaml
interaction_prompts:
  - "A hand pouring water from a pitcher into a glass"
  - "A person stacking wooden blocks"
  - "Two billiard balls colliding on a table"
  - "A cat jumping onto a table and knocking over a cup"
  - "A person opening a door and walking through"
  - "Dominoes falling in a chain reaction"
  - "A hand placing a book on top of a stack"
  - "Two people shaking hands"
  - "A ball bouncing and coming to rest"
  - "A spoon stirring liquid in a cup"
```

---

## Appendix C: Human Evaluation Interface

### Study Protocol

1. **Introduction** (2 min)
   - Explain task: "You will compare video pairs for motion quality"
   - Define criteria: smoothness, physical plausibility, naturalness
   - Practice round with example pairs

2. **Main Evaluation** (15 min)
   - 50 video pair comparisons
   - Each pair shown for 5 seconds, looped
   - Questions after each pair

3. **Debrief** (3 min)
   - Free-form feedback
   - Demographics (optional)

### Interface Mockup

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Quality Study                       │
│                                                              │
│  ┌─────────────────┐       ┌─────────────────┐              │
│  │                 │       │                 │              │
│  │    Video A      │       │    Video B      │              │
│  │                 │       │                 │              │
│  └─────────────────┘       └─────────────────┘              │
│                                                              │
│  Which video has smoother, more natural motion?              │
│                                                              │
│  ○ Video A          ○ Video B          ○ About the same     │
│                                                              │
│  Rate the smoothness of Video A: [1] [2] [3] [4] [5]        │
│  Rate the smoothness of Video B: [1] [2] [3] [4] [5]        │
│                                                              │
│  Notice anything wrong? [____________________________]       │
│                                                              │
│                         [Next Pair]                          │
│                                                              │
│  Progress: ████████░░░░░░░░░░░░ 17/50                       │
└─────────────────────────────────────────────────────────────┘
```

---

## References

1. HaCohen et al. (2024). LTX-Video: Realtime Video Latent Diffusion. arXiv:2501.00103
2. Unterthiner et al. (2018). Towards Accurate Generative Models of Video: A New Metric & Challenges. arXiv:1812.01717 (FVD)
3. Teed & Deng (2020). RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. ECCV.
4. Zhang et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR. (LPIPS)
5. Goyal et al. (2017). The "Something Something" Video Database. ICCV.
