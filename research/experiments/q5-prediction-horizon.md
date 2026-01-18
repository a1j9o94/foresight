# Q5: The Right Prediction Target

## Status: Planned
## Priority: Medium (optimization, not go/no-go)
## Dependencies: Requires working system (C1-C3 validated)

---

## 1. Objective

Determine the optimal prediction horizon that maximizes reasoning utility while maintaining acceptable generation quality. This is fundamentally a tradeoff between:

- **Longer horizons**: More useful for reasoning about consequences and planning
- **Shorter horizons**: Higher generation quality and lower computational cost

The goal is to find the "sweet spot" where predicted videos are both accurate enough to be useful and long enough to capture meaningful state changes.

---

## 2. Background

### 2.1 Prediction Horizon in World Models

**Dreamer series (Hafner et al.)**: Uses 15-step imagination horizon as the default across all experiments. This was found empirically to balance:
- Long enough to propagate value estimates
- Short enough that accumulated prediction errors don't dominate
- The Minecraft diamond achievement required reasoning ~100+ steps ahead, but this was done through iterated 15-step rollouts

**Key insight**: Dreamer operates in latent space where prediction errors compound more gracefully. In pixel space, errors may be more visible but also more verifiable.

### 2.2 Video Generation Quality vs Length

From our model analysis:

| Model | Optimal Duration | Quality Degradation Pattern |
|-------|------------------|----------------------------|
| LTX-Video | 5 seconds (121 frames @ 24fps) | Temporal coherence degrades after ~3-4s |
| HunyuanVideo | 5 seconds (129 frames @ 24fps) | Better long-term coherence, still degrades |
| Typical DiT models | 2-4 seconds | Significant motion drift beyond |

**Observed failure modes at longer horizons**:
- Motion drift (objects gradually shift position)
- Identity loss (subjects change appearance)
- Physics violations (gravity, momentum inconsistencies)
- Semantic drift (scene context changes inappropriately)

### 2.3 Task-Relevant Horizons

Different reasoning tasks require different prediction horizons:

| Task Type | Minimum Useful Horizon | Examples |
|-----------|------------------------|----------|
| Immediate consequence | 0.5-1s | "What happens when I drop this?" |
| Action completion | 2-5s | "How does pouring water fill the glass?" |
| Multi-step procedure | 10-30s | "How do I assemble this furniture?" |
| Long-term planning | 60s+ | "What will this room look like after cleaning?" |

**Critical observation**: Useful horizon depends on the task, not just model capability.

---

## 3. Experimental Setup

### 3.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 24GB (A10) | 40GB (A100) |
| GPU Count | 1 | 2-4 (for parallel experiments) |
| System RAM | 64GB | 128GB |
| Storage | 500GB SSD | 1TB NVMe |

**Estimated compute**: ~200 GPU-hours total across all experiments

### 3.2 Evaluation Datasets

**Primary datasets** (require different prediction horizons):

| Dataset | Task | Relevant Horizon | Size |
|---------|------|------------------|------|
| Something-Something v2 | Object manipulation | 1-3s | 220K videos |
| COIN | Procedural activities | 5-30s | 11K videos |
| Physion | Physical reasoning | 1-5s | 2K scenarios |
| CLEVRER | Causal reasoning | 2-10s | 20K videos |

**Synthetic benchmark** (controlled):
- Generate test scenarios with known outcomes at each timestep
- Enables precise measurement of accuracy vs horizon

### 3.3 Metrics

#### Generation Quality Metrics

| Metric | Measures | Target |
|--------|----------|--------|
| LPIPS | Perceptual similarity to ground truth | < 0.3 |
| FVD | Distribution similarity for videos | < 200 |
| FID (per-frame) | Individual frame quality | < 30 |
| Temporal Consistency Score | Frame-to-frame coherence | > 0.9 |
| SSIM | Structural similarity | > 0.7 |

#### Reasoning Utility Metrics

| Metric | Measures | Computation |
|--------|----------|-------------|
| Action Prediction Accuracy | Can VLM predict correct action from generated video? | % correct on held-out set |
| State Change Detection | Are key state changes visible? | Human evaluation |
| Causal Accuracy | Are cause-effect relationships preserved? | CLEVRER-style evaluation |
| Semantic Preservation | Is meaning maintained throughout? | VLM consistency check |

#### Efficiency Metrics

| Metric | Measures |
|--------|----------|
| Generation Time | Wall-clock time per prediction |
| Memory Usage | Peak VRAM during generation |
| Error Accumulation Rate | How quickly does quality degrade? |

---

## 4. Experiments

### E-Q5.1: Generation Quality vs Horizon Length

**Objective**: Establish baseline quality degradation curve.

**Method**:
1. Select 1000 videos from Something-Something v2 (diverse actions)
2. For each video, generate predictions at horizons: 1, 2, 5, 10, 30 frames
3. Compare predicted frame N to actual frame N using LPIPS, SSIM, FID
4. Plot quality metrics as a function of horizon

**Independent variables**:
- Prediction horizon: {1, 2, 5, 10, 30} frames at 24fps
  - 1 frame = 42ms
  - 2 frames = 83ms
  - 5 frames = 208ms
  - 10 frames = 417ms
  - 30 frames = 1.25s

**Dependent variables**:
- LPIPS score (per frame and average)
- FVD (for video segments)
- Temporal consistency score

**Expected outcome**: Quality curve showing exponential degradation with horizon length.

**Duration**: 2 days

---

### E-Q5.2: Reasoning Accuracy vs Horizon

**Objective**: Measure how prediction horizon affects downstream reasoning accuracy.

**Method**:
1. Use action prediction task: given video frames 1-N, predict which action occurs
2. Compare conditions:
   - Baseline: VLM sees real video only (no prediction)
   - Condition A: VLM generates 1-frame prediction, then reasons
   - Condition B: VLM generates 5-frame prediction, then reasons
   - Condition C: VLM generates 30-frame prediction, then reasons
3. Measure action prediction accuracy for each condition

**Dataset**: Something-Something v2 (174 action classes)

**Hypothesis**: Longer horizons improve reasoning up to a point, then degrade as generation quality hurts more than temporal context helps.

**Expected finding**: Optimal horizon between 5-15 frames for this task.

**Duration**: 3 days

---

### E-Q5.3: Single Frame (Final State) vs Video Trajectory

**Objective**: Determine if full trajectory is necessary or if predicting just the final state suffices.

**Method**:
1. Train two adapter variants:
   - **Final-state model**: Predicts only the last frame of action outcome
   - **Trajectory model**: Predicts full video sequence
2. Evaluate both on:
   - Action prediction (does final state contain enough information?)
   - Physical reasoning (is intermediate motion needed?)
   - Causal reasoning (do we need to see the process?)

**Conditions**:
| Condition | Output | Frames Generated |
|-----------|--------|------------------|
| Single frame | Final state only | 1 |
| Sparse trajectory | Start + end | 2 |
| Full trajectory | All intermediate frames | 10-30 |

**Key questions**:
- For what tasks is final state sufficient?
- When is observing the process (trajectory) essential?
- Is "final state + initial state" a good middle ground?

**Metrics**:
- Task accuracy (per task type)
- Generation cost (latency, compute)
- Information density (bits of task-relevant info per generated frame)

**Duration**: 4 days

---

### E-Q5.4: Keyframe Prediction vs Dense Frames

**Objective**: Explore whether predicting sparse keyframes (then interpolating) is more effective than dense prediction.

**Method**:
1. Implement three prediction strategies:
   - **Dense**: Predict all N frames sequentially
   - **Keyframe + interpolate**: Predict frames {1, N/2, N}, interpolate between
   - **Adaptive keyframe**: Model predicts which frames are "key" then fills in

2. Compare quality and efficiency:
   - Generation quality at equivalent compute budget
   - Semantic accuracy (do keyframes capture important state changes?)
   - Interpolation artifacts

**Rationale**: Video models are trained on continuous motion. Keyframe prediction might:
- Focus model capacity on semantically important moments
- Reduce accumulated drift by "anchoring" at multiple points
- Allow longer effective horizons with same quality budget

**Metrics**:
- Quality per compute (LPIPS / generation time)
- Keyframe selection accuracy (do predicted keyframes align with ground truth state changes?)
- Interpolation quality

**Duration**: 3 days

---

### E-Q5.5: Adaptive Horizon (Model-Decided Prediction Length)

**Objective**: Can the model learn to predict "as far as it can confidently predict"?

**Method**:
1. Train model to output a confidence score alongside predictions
2. Implement adaptive rollout:
   - Generate frames until confidence drops below threshold
   - Or until task-relevant state change detected
3. Compare to fixed-horizon approaches

**Implementation**:
```
while confidence > threshold and frames < max_horizon:
    next_frame, confidence = model.predict_next(current_state)
    frames.append(next_frame)
    current_state = next_frame
```

**Confidence estimation approaches**:
- Auxiliary head predicting generation quality
- VLM self-assessment ("Is this prediction still reliable?")
- Reconstruction consistency (encode predicted frame, compare latents)

**Hypothesis**: Adaptive horizon outperforms fixed horizon because:
- Easy predictions extend further
- Hard predictions stop early (before quality degrades)
- Compute is allocated where it's most useful

**Challenges**:
- Training the confidence estimator
- Defining "confident" vs "uncertain" predictions
- Avoiding degenerate solutions (always predict 1 frame)

**Duration**: 5 days

---

### E-Q5.6: Task-Specific Optimal Horizons

**Objective**: Determine whether different tasks have fundamentally different optimal horizons.

**Method**:
1. Evaluate system on diverse task battery with multiple horizon settings
2. For each task, find the horizon that maximizes accuracy
3. Analyze patterns across task types

**Task battery**:

| Task | Dataset | Horizons to Test | Expected Optimal |
|------|---------|------------------|------------------|
| Object state prediction | Something-Something v2 | 1, 3, 5, 10 frames | 3-5 frames |
| Physical outcome | Physion | 5, 10, 20, 30 frames | 10-20 frames |
| Action recognition | COIN | 10, 30, 60, 120 frames | 30-60 frames |
| Causal reasoning | CLEVRER | 5, 15, 30, 60 frames | 15-30 frames |
| Procedural understanding | CrossTask | 30, 60, 120, 300 frames | 60-120 frames |

**Analysis**:
- Cluster tasks by optimal horizon
- Identify task features that predict optimal horizon
- Build predictor: given task description, estimate optimal horizon

**Deliverable**: Task-to-horizon mapping for production use.

**Duration**: 5 days

---

## 5. Success Metrics

### Primary Success Criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Identify optimal horizon for action prediction | Clear peak in accuracy curve | E-Q5.2 results |
| Quantify quality-utility tradeoff | Pareto frontier established | E-Q5.1 + E-Q5.2 combined |
| Task-specific recommendations | >80% of tasks have clear optimal | E-Q5.6 analysis |

### Secondary Success Criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Adaptive horizon outperforms fixed | >5% accuracy improvement | E-Q5.5 vs E-Q5.2 |
| Keyframe prediction is viable | <10% quality loss at 2x speed | E-Q5.4 results |
| Final-state sufficient for some tasks | Identify task categories | E-Q5.3 results |

### Quality Gates

- **Generation quality floor**: LPIPS < 0.5 at longest tested horizon
- **Reasoning utility floor**: Action accuracy > random baseline at all horizons
- **Reproducibility**: Results consistent across 3 random seeds

---

## 6. Failure Criteria

**N/A - This is optimization, not go/no-go.**

This experiment cannot "fail" in a way that blocks the project. All possible outcomes provide useful information:

| Outcome | Interpretation | Next Step |
|---------|----------------|-----------|
| No clear optimal horizon | Use task-specific horizons | Implement adaptive system |
| Quality degrades too fast | Focus on short-horizon tasks | Improve generation quality first |
| All horizons work equally | Horizon is not a key variable | Focus optimization elsewhere |
| Adaptive horizon doesn't help | Fixed horizons are sufficient | Simpler implementation |

---

## 7. Recommendations Framework

Based on experimental results, we will produce recommendations in this format:

### For System Design

```
DEFAULT_HORIZON = [result from E-Q5.2]  # frames

TASK_HORIZONS = {
    "object_manipulation": [from E-Q5.6],
    "physical_reasoning": [from E-Q5.6],
    "procedural_activity": [from E-Q5.6],
    "causal_reasoning": [from E-Q5.6],
}

USE_ADAPTIVE_HORIZON = [True/False based on E-Q5.5]
USE_KEYFRAME_PREDICTION = [True/False based on E-Q5.4]
```

### For Users/API

```
prediction_horizon:
  - "auto": Model decides based on task (if E-Q5.5 succeeds)
  - "short": 1-5 frames (fast, high quality)
  - "medium": 10-15 frames (balanced)
  - "long": 30+ frames (more context, lower quality)
```

### Decision Tree

```
Is task type known?
├─ Yes: Use task-specific optimal horizon (E-Q5.6)
└─ No: Is adaptive horizon reliable?
       ├─ Yes: Use adaptive (E-Q5.5)
       └─ No: Use default (E-Q5.2 result)
```

---

## 8. Timeline

| Experiment | Duration | Dependencies | Parallelizable |
|------------|----------|--------------|----------------|
| E-Q5.1: Quality vs horizon | 2 days | Working system | Yes |
| E-Q5.2: Reasoning vs horizon | 3 days | Working system | Yes (with E-Q5.1) |
| E-Q5.3: Final state vs trajectory | 4 days | E-Q5.1 complete | Yes (with E-Q5.2) |
| E-Q5.4: Keyframe prediction | 3 days | E-Q5.1 complete | Yes (with E-Q5.3) |
| E-Q5.5: Adaptive horizon | 5 days | E-Q5.2 complete | No |
| E-Q5.6: Task-specific horizons | 5 days | E-Q5.2 complete | Partial |
| Analysis and writeup | 2 days | All complete | No |

**Critical path**: E-Q5.1 -> E-Q5.2 -> E-Q5.5 -> Analysis (12 days)

**Total estimated time**:
- Sequential: 24 days
- With parallelization: 14 days
- With 2 GPUs parallel: ~10 days

### Gantt Chart (simplified)

```
Week 1: [E-Q5.1][E-Q5.2-----]
Week 2: [E-Q5.3---][E-Q5.4--][E-Q5.2]
Week 3: [E-Q5.5--------][E-Q5.6-----]
Week 4: [E-Q5.6][Analysis--]
```

---

## 9. Dependencies

### Required Before Starting

| Dependency | Status | Blocker? |
|------------|--------|----------|
| C1: VLM latents extractable | Must be validated | Yes |
| C2: Adapter trained | Must work at some quality level | Yes |
| C3: End-to-end generation | Must produce coherent output | Yes |

### Required Components

- **Working prediction pipeline**: VLM -> Adapter -> Video Decoder
- **Evaluation scripts**: LPIPS, FVD, temporal consistency
- **Action prediction benchmark**: Something-Something v2 setup
- **Compute allocation**: 200 GPU-hours reserved

### Data Requirements

| Dataset | Download Size | Preprocessing |
|---------|---------------|---------------|
| Something-Something v2 | 20GB (subset) | Extract 1000 evaluation videos |
| Physion | 5GB | Download test split |
| CLEVRER | 10GB | Download validation set |
| COIN | 15GB | Extract relevant subsets |

---

## 10. Deliverables

### Primary Deliverables

1. **Horizon Analysis Report** (`research/results/q5-horizon-analysis.md`)
   - Quality vs horizon curves with confidence intervals
   - Reasoning accuracy vs horizon curves
   - Statistical analysis and significance tests

2. **Task-Horizon Mapping** (`configs/task_horizons.yaml`)
   ```yaml
   tasks:
     object_manipulation:
       optimal_horizon: 5
       acceptable_range: [3, 10]
       confidence: 0.95
     physical_reasoning:
       optimal_horizon: 15
       acceptable_range: [10, 20]
       confidence: 0.87
   ```

3. **Recommendation Document** (`research/results/q5-recommendations.md`)
   - Default horizon for unknown tasks
   - When to use adaptive vs fixed
   - Task-specific recommendations
   - Integration guide for production system

### Secondary Deliverables

4. **Ablation Tables** (in analysis report)
   - Keyframe vs dense prediction comparison
   - Final-state vs trajectory comparison
   - Adaptive vs fixed horizon comparison

5. **Code Artifacts**
   - Evaluation scripts for all metrics
   - Adaptive horizon implementation (if successful)
   - Keyframe prediction module (if beneficial)

6. **Figures for Paper/Documentation**
   - Quality-utility tradeoff curve
   - Task-specific optimal horizon chart
   - Error accumulation visualization

---

## Appendix A: Detailed Metric Definitions

### LPIPS (Learned Perceptual Image Patch Similarity)

```python
import lpips
loss_fn = lpips.LPIPS(net='alex')
distance = loss_fn(predicted_frame, actual_frame)
# Lower is better; < 0.3 is "good", < 0.1 is "excellent"
```

### FVD (Frechet Video Distance)

```python
from frechet_video_distance import FrechetVideoDistance
fvd = FrechetVideoDistance()
score = fvd(predicted_videos, real_videos)
# Lower is better; < 200 is good for short clips
```

### Temporal Consistency Score

```python
def temporal_consistency(video):
    """Measures frame-to-frame stability"""
    scores = []
    for i in range(len(video) - 1):
        flow = compute_optical_flow(video[i], video[i+1])
        consistency = measure_flow_smoothness(flow)
        scores.append(consistency)
    return np.mean(scores)
```

---

## Appendix B: Baseline Comparisons

### Human Prediction Horizon

Cognitive science research suggests humans predict:
- Immediate physical events: ~500ms ahead
- Action outcomes: 1-2 seconds ahead
- Social interactions: 3-5 seconds ahead
- Long-term planning: Minutes to hours (but abstractly, not pixel-level)

Our prediction horizons should likely fall in the 0.5-5 second range for most useful applications.

### Dreamer's 15-Step Horizon

At typical environment step rates:
- Atari (4 frames per step): 15 steps = 60 frames = 1 second
- DMC (frame skip 2): 15 steps = 30 frames = 0.5 seconds
- Minecraft (20 fps): 15 steps = 0.75 seconds

This suggests 0.5-1 second is empirically validated for RL tasks.

---

## Appendix C: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Quality degrades too fast to be useful | Medium | High | Focus on keyframe prediction, shorter horizons |
| Task-specific horizons vary too much | Low | Medium | Build adaptive system, accept complexity |
| Experiments take longer than estimated | Medium | Low | Prioritize E-Q5.1 and E-Q5.2, parallelize |
| Results are inconclusive | Low | Medium | Increase sample sizes, add more horizons |

---

## Appendix D: Related Work Reference

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| Dreamer V3 | 15-step horizon works across tasks | Baseline reference |
| VideoGPT | Quality degrades exponentially with length | Expect similar |
| TECO | Keyframe + interpolation effective | Informs E-Q5.4 |
| FDM | Longer horizon = more diverse but less accurate | Tradeoff confirmation |
| UniSim | Task-conditioned prediction length | Supports E-Q5.6 |

---

*Last updated: 2025-01-18*
*Author: Research Team*
*Status: Ready for execution pending C1-C3 validation*
