# C4: Pixel Verification Improves Accuracy

**Claim:** Comparing predicted video to actual outcomes provides a signal that improves prediction accuracy.

**Status:** Ready to start (dependencies met via E3.8 pivot)
**Priority:** High (key differentiator from V-JEPA)
**Last Updated:** 2026-01-25

---

## Architecture Update: E3.8 Pivot

**Original assumption:** VLM predicts future states in latent space (C3)
**Actual architecture:** Video Predicts → VLM Describes (E3.8 validated)

The verification loop now operates as:
```
1. LTX-Video generates future frames from context + action
2. VLM describes what happens in generated video
3. Compare VLM description to actual outcome
4. Feedback loop: Use discrepancy to improve next prediction
```

**Key metrics from E3.8:**
- LTX-Video temporal coherence: 0.89 (realistic motion)
- VLM retention on generated video: 93% (70% vs 75% action recall)

This means VLM can reliably describe LTX-Video outputs, enabling semantic verification.

---

## 1. Objective

Test whether pixel-level verification (comparing predicted video frames to actual outcomes) provides a meaningful error signal that:

1. **Correlates with prediction correctness** - High perceptual error indicates wrong predictions
2. **Enables self-correction** - Models can detect and fix their own errors
3. **Improves downstream task accuracy** - Verification loop outperforms single-shot prediction

This is the critical test that differentiates Foresight from latent-only approaches (V-JEPA, Dreamer). If pixel verification does not help, the computational cost of video generation is not justified.

### Null Hypothesis (to disprove)

> Perceptual similarity metrics (LPIPS, FVD) between predicted and actual video do not correlate with task-relevant prediction accuracy, and verification loops do not improve performance.

---

## 2. Background

### 2.1 Perceptual Similarity Metrics

#### LPIPS (Learned Perceptual Image Patch Similarity)

- **What it measures:** Perceptual distance between images using deep network feature activations
- **Range:** 0 (identical) to ~1 (completely different)
- **Strengths:** Correlates well with human perception; differentiable; robust to minor pixel shifts
- **Weaknesses:**
  - Trained on human perceptual judgments, not task relevance
  - May miss semantically important but visually subtle differences
  - Frame-by-frame metric; temporal consistency requires aggregation
- **Reference:** Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018)

**Typical values:**
| Comparison | LPIPS |
|------------|-------|
| Same image | 0.0 |
| Minor augmentation | 0.05-0.15 |
| Same scene, different angle | 0.2-0.4 |
| Different scenes | 0.5-0.8 |

#### FVD (Frechet Video Distance)

- **What it measures:** Distribution-level similarity between video sets using I3D features
- **Range:** 0 (identical distributions) to infinity
- **Strengths:** Captures temporal dynamics; standard benchmark metric for video generation
- **Weaknesses:**
  - Requires many samples (not single-video)
  - Expensive to compute
  - Sensitive to video length and resolution
- **Reference:** Unterthiner et al., "FVD: A New Metric for Video Generation" (2019)

#### VLM-Based Comparison

- **What it measures:** Semantic similarity judged by vision-language model
- **Implementation:** Feed both videos to VLM with prompt "Do these videos show the same outcome?"
- **Strengths:** Can capture semantic equivalence despite visual differences
- **Weaknesses:**
  - Expensive (full VLM inference)
  - May be overly lenient or strict depending on prompt
  - Not differentiable

### 2.2 Verification in Other Domains

| Domain | Verification Approach | Improvement |
|--------|----------------------|-------------|
| **Code generation** | Execute and check output | +20-40% accuracy (AlphaCode) |
| **Math reasoning** | Verify answer satisfies constraints | +15-25% accuracy (Minerva) |
| **Robotics** | Compare predicted vs actual sensor readings | Enables closed-loop control |
| **Self-play games** | Play out predicted scenarios | Foundation of AlphaGo/MuZero |
| **Language (CoT)** | Self-consistency voting | +5-15% accuracy |

### 2.3 Why Pixel Verification Might Work

1. **Physics violations are visible:** Predictions that violate physics (objects interpenetrating, floating, disappearing) manifest as visual errors
2. **Semantic errors correlate with perceptual errors:** Wrong action prediction often produces visually distinct outcomes
3. **Calibration signal:** High uncertainty should correlate with high prediction error

### 2.4 Why Pixel Verification Might Fail

1. **Perceptually similar but semantically different:** Two outcomes could look similar but represent different actions (e.g., pushing left vs right on a symmetric object)
2. **Perceptually different but semantically equivalent:** Same action could produce visually different results due to irrelevant variation (lighting, texture)
3. **Error signal too noisy:** LPIPS captures low-level details that swamp task-relevant signal
4. **Verification too slow:** By the time we verify, it's too late to correct

---

## 3. Experimental Setup

### 3.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 24GB (RTX 4090) | 48GB (A6000) or 2x24GB |
| System RAM | 32GB | 64GB |
| Storage | 500GB SSD | 1TB NVMe |
| Expected runtime per experiment | 4-8 hours | 2-4 hours |

**VRAM Breakdown:**
- Qwen2.5-VL-7B (quantized): ~14GB
- LTX-Video: ~8GB
- LPIPS model: ~0.5GB
- Verification overhead: ~4GB
- Total: ~27GB (requires offloading or quantization on single 24GB GPU)

### 3.2 Metrics Implementation

#### LPIPS for Video

```python
# Pseudo-code for video LPIPS
import lpips

loss_fn = lpips.LPIPS(net='alex')  # or 'vgg'

def video_lpips(predicted_video, actual_video):
    """
    Args:
        predicted_video: Tensor [T, C, H, W] or [B, T, C, H, W]
        actual_video: Tensor [T, C, H, W] or [B, T, C, H, W]
    Returns:
        Frame-wise LPIPS scores and aggregates
    """
    frame_scores = []
    for t in range(T):
        score = loss_fn(predicted_video[t], actual_video[t])
        frame_scores.append(score)

    return {
        'per_frame': frame_scores,
        'mean': np.mean(frame_scores),
        'max': np.max(frame_scores),
        'final_frame': frame_scores[-1],  # Often most task-relevant
    }
```

#### FVD Implementation

```python
# Use pytorch-fvd or implement with I3D features
from pytorch_fvd import fvd_score

def compute_fvd(predicted_videos, actual_videos):
    """
    Args:
        predicted_videos: List of videos or Tensor [N, T, C, H, W]
        actual_videos: List of videos or Tensor [N, T, C, H, W]
    Returns:
        FVD score (lower is better)
    """
    # Requires at least ~100 videos for stable estimate
    return fvd_score(predicted_videos, actual_videos)
```

#### VLM-Based Comparison

```python
def vlm_compare(predicted_video, actual_video, vlm_model, task_description):
    """
    Use VLM to judge semantic similarity
    """
    prompt = f"""
    Task: {task_description}

    I will show you two videos:
    Video A: [predicted outcome]
    Video B: [actual outcome]

    Questions:
    1. Do both videos show the same action being performed? (yes/no)
    2. Do both videos show the same end state? (yes/no)
    3. Rate the semantic similarity from 0-10.
    4. What are the key differences?

    Respond in JSON format.
    """

    response = vlm_model.generate(
        prompt=prompt,
        video_a=predicted_video,
        video_b=actual_video
    )
    return parse_json(response)
```

### 3.3 Dataset Requirements

**Primary Dataset: Something-Something v2 (SSv2)**

| Property | Value |
|----------|-------|
| Size | 220,847 videos |
| Actions | 174 fine-grained action classes |
| Duration | 2-6 seconds |
| Why suitable | Clear action-outcome relationships; minimal background variation |

**Subset for experiments:**
- Training: 10,000 videos (balanced across 50 representative classes)
- Validation: 2,000 videos
- Test: 2,000 videos

**Ground truth requirements:**
- Video frames before action (input)
- Action label or description (conditioning)
- Video frames after action (ground truth outcome)
- Task-specific correctness label (does prediction match correct action?)

**Secondary Dataset: COIN**

- 11,827 videos of procedural activities
- Useful for multi-step verification chains

### 3.4 Verification Module Architecture

```
                    Predicted Video
                          |
                          v
+------------------+     +------------------+
|  LPIPS Network   |---->|                  |
+------------------+     |   Verification   |
                         |   Aggregator     |---> Verification Score
+------------------+     |                  |        (0-1)
|  Actual Video    |---->|                  |
+------------------+     +------------------+
                                |
                                v
                    +------------------+
                    |  Correction      |
                    |  Signal          |---> Feedback to model
                    +------------------+
```

---

## 4. Experiments

### E4.1: Correlation Study (LPIPS vs Prediction Correctness)

**Objective:** Determine if LPIPS error predicts whether predictions are correct.

**Protocol:**

1. Generate N=1000 predictions using trained Foresight model
2. For each prediction:
   - Compute LPIPS(predicted, actual)
   - Determine correctness (does predicted video match ground truth action?)
3. Compute correlation between LPIPS and correctness

**Correctness definition:**
- Use VLM to classify: "What action is being performed in this video?"
- Correct = VLM classification of predicted video matches ground truth action

**Analysis:**
- Plot LPIPS distribution for correct vs incorrect predictions
- Compute point-biserial correlation coefficient
- ROC curve: Can LPIPS threshold distinguish correct/incorrect?

**Metrics:**
| Metric | Target | Minimum |
|--------|--------|---------|
| Point-biserial correlation | r > 0.5 | r > 0.3 |
| AUROC (LPIPS as classifier) | > 0.75 | > 0.65 |
| LPIPS gap (incorrect - correct) | > 0.15 | > 0.08 |

**Failure criteria:**
- Correlation not statistically significant (p > 0.05)
- AUROC < 0.6 (barely better than random)
- Large overlap in LPIPS distributions

**Duration:** 2-3 days
- Model inference: 1000 predictions * ~3s = ~1 hour
- LPIPS computation: ~30 minutes
- Analysis and iteration: 1-2 days

---

### E4.2: Calibration Study (Does Model Know When It's Wrong?)

**Objective:** Test if model's uncertainty correlates with prediction error.

**Protocol:**

1. Extract model's internal confidence/uncertainty signals:
   - Query token attention entropy
   - Predicted latent variance (if available)
   - Video decoder sampling variance
2. Compare to actual LPIPS error

**Uncertainty extraction methods:**

```python
# Method 1: Attention entropy
def attention_entropy(attention_weights):
    """Higher entropy = more uncertain"""
    return -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)

# Method 2: Monte Carlo dropout
def mc_dropout_variance(model, input, n_samples=10):
    """Variance across dropout samples"""
    model.train()  # Enable dropout
    predictions = [model(input) for _ in range(n_samples)]
    return torch.var(torch.stack(predictions), dim=0)

# Method 3: Ensemble disagreement
def ensemble_variance(models, input):
    """Variance across ensemble members"""
    predictions = [model(input) for model in models]
    return torch.var(torch.stack(predictions), dim=0)
```

**Analysis:**
- Correlation between uncertainty and LPIPS
- Calibration plot: binned uncertainty vs actual error
- Expected Calibration Error (ECE)

**Metrics:**
| Metric | Target | Minimum |
|--------|--------|---------|
| Uncertainty-LPIPS correlation | r > 0.4 | r > 0.2 |
| Expected Calibration Error | < 0.1 | < 0.15 |
| Reliability diagram R^2 | > 0.8 | > 0.6 |

**Failure criteria:**
- No correlation between uncertainty and error
- Model overconfident on wrong predictions
- Calibration curve is flat (no relationship)

**Duration:** 2-3 days
- Uncertainty extraction implementation: 1 day
- Inference and analysis: 1-2 days

---

### E4.3: Single Verification Loop

**Objective:** Test if verification improves accuracy on second attempt.

**Protocol:**

```
Round 1:
  Input: Video frames + action description
  Output: Predicted video V1
  Score: LPIPS(V1, actual)

Verification:
  Compare V1 to actual outcome
  Generate feedback signal

Round 2:
  Input: Video frames + action description + V1 + feedback
  Output: Predicted video V2
  Score: LPIPS(V2, actual)

Compare: Accuracy(V2) vs Accuracy(V1)
```

**Feedback signal options (test all):**

1. **Binary feedback:** "Your prediction was [correct/incorrect]"
2. **LPIPS score:** "Your prediction had perceptual error of 0.35"
3. **VLM feedback:** "Your prediction showed X but actual showed Y"
4. **Visual feedback:** Concatenate predicted + actual as context

**Conditions:**
| Condition | Round 1 Context | Round 2 Context |
|-----------|-----------------|-----------------|
| Baseline (no verification) | Video + action | Video + action |
| Binary feedback | Video + action | Video + action + "incorrect" |
| LPIPS feedback | Video + action | Video + action + LPIPS score |
| VLM feedback | Video + action | Video + action + VLM description |
| Visual feedback | Video + action | Video + action + V1 + actual |
| Oracle (upper bound) | Video + action | Video + action + actual video |

**Sample size:** N=500 initially incorrect predictions (to measure correction rate)

**Metrics:**
| Metric | Target | Minimum |
|--------|--------|---------|
| Correction rate (V2 correct | V1 incorrect) | > 30% | > 15% |
| Overall accuracy improvement | > 10% | > 5% |
| V2 LPIPS < V1 LPIPS rate | > 60% | > 55% |

**Failure criteria:**
- Correction rate < 10% (verification not helpful)
- V2 worse than V1 (verification causes regression)
- No significant difference between feedback types

**Duration:** 4-5 days
- Implementation of verification loop: 2 days
- Running all conditions: 2 days
- Analysis: 1 day

---

### E4.4: Multiple Verification Iterations

**Objective:** Determine optimal number of verification loops and diminishing returns.

**Protocol:**

```
for k in [1, 2, 3, 4, 5]:
    V_k = predict(context + V_{k-1} + feedback_{k-1})
    score_k = LPIPS(V_k, actual)
    accuracy_k = measure_correctness(V_k)
```

**Analysis:**
- Plot accuracy vs iteration number
- Plot LPIPS vs iteration number
- Find convergence point
- Measure computational cost per iteration

**Questions to answer:**
1. Does accuracy monotonically improve?
2. Where is the diminishing returns threshold?
3. Do some predictions get worse with more iterations?
4. What is the optimal stopping criterion?

**Stopping criteria to test:**
- Fixed iterations (k=2, k=3)
- Confidence threshold (stop when uncertainty below threshold)
- LPIPS improvement threshold (stop when improvement < 0.02)
- Adaptive (let model decide when to stop)

**Metrics:**
| Metric | Target | Minimum |
|--------|--------|---------|
| Accuracy at k=2 vs k=1 | +10% | +5% |
| Accuracy at k=3 vs k=2 | +5% | +2% |
| Optimal k (accuracy plateau) | k <= 3 | k <= 5 |
| Cost per 1% accuracy | < 2x inference cost | < 5x inference cost |

**Failure criteria:**
- No improvement beyond k=1
- Accuracy decreases with more iterations
- Cost prohibitive (>10x for marginal gains)

**Duration:** 3-4 days
- Implementation: 1 day (extension of E4.3)
- Running experiments: 2 days
- Analysis: 1 day

---

### E4.5: Compare Verification Metrics

**Objective:** Determine which verification metric best predicts and enables correction.

**Protocol:**

Test each metric as:
1. **Correlation signal:** Which metric best predicts correctness?
2. **Feedback signal:** Which metric provides best correction guidance?
3. **Stopping criterion:** Which metric best indicates when to stop iterating?

**Metrics to compare:**

| Metric | Compute Cost | Differentiable | Semantic |
|--------|-------------|----------------|----------|
| LPIPS (AlexNet) | Low | Yes | Perceptual |
| LPIPS (VGG) | Medium | Yes | Perceptual |
| FVD | High | No | Distributional |
| MSE/PSNR | Very Low | Yes | Pixel-level |
| SSIM | Low | Yes | Structural |
| VLM binary | High | No | Semantic |
| VLM detailed | Very High | No | Semantic |
| CLIP similarity | Medium | Yes | Semantic |

**Experimental matrix:**

For each metric M:
1. Compute correlation(M, correctness) -- which metric best identifies errors?
2. Use M as feedback in verification loop -- which improves accuracy most?
3. Use M as stopping criterion -- which gives best accuracy/cost tradeoff?

**Hybrid approaches:**
- LPIPS + VLM: Use LPIPS for fast filtering, VLM for high-stakes decisions
- Learned combination: Train small network to combine metrics

**Metrics:**
| Comparison | Winner Criteria |
|------------|-----------------|
| Correlation with correctness | Highest point-biserial r |
| Verification improvement | Largest accuracy gain |
| Cost efficiency | Best accuracy per FLOP |
| Robustness | Smallest variance across tasks |

**Duration:** 5-7 days
- Implement all metrics: 2 days
- Run correlation study for all: 2 days
- Run verification loop with each: 2-3 days
- Analysis: 1 day

---

## 5. Success Metrics

### Primary Success Criteria

| Metric | Threshold | Justification |
|--------|-----------|---------------|
| LPIPS-correctness correlation | r >= 0.35, p < 0.01 | Statistically significant, meaningful effect size |
| Verification loop accuracy gain | >= 10% relative improvement | Justifies computational cost |
| Correction rate | >= 25% of errors corrected | Meaningful error recovery |

### Secondary Success Criteria

| Metric | Threshold | Justification |
|--------|-----------|---------------|
| Calibration ECE | < 0.12 | Model knows when it's wrong |
| Optimal iteration count | k <= 3 | Practical for inference |
| Best metric identified | Clear winner (>5% gap) | Actionable guidance |

### Statistical Requirements

- All comparisons: Two-tailed tests, alpha = 0.05
- Sample size: N >= 500 for per-condition comparisons
- Effect size: Report Cohen's d or equivalent
- Multiple comparisons: Bonferroni correction when testing >3 conditions

---

## 6. Failure Criteria

### Claim 4 is Falsified If

1. **No correlation exists:**
   - LPIPS-correctness correlation r < 0.2 (negligible effect)
   - AUROC < 0.6 for LPIPS as error classifier
   - p-value > 0.05 for correlation test

2. **Verification doesn't help:**
   - Accuracy improvement < 5% with verification loop
   - Correction rate < 10%
   - V2 accuracy significantly worse than V1 in any condition

3. **Cost is prohibitive:**
   - Verification loop > 10x inference cost for < 5% improvement
   - Optimal k > 5 iterations

4. **Fundamental disconnect:**
   - High LPIPS predictions are MORE accurate (inverse correlation)
   - VLM-based verification significantly outperforms pixel-based (pixels are noise)

### Partial Failures (Claim Weakened but Not Falsified)

- LPIPS works but only for specific action types
- Verification helps but plateaus quickly (k=1 optimal)
- Works only with VLM feedback, not raw pixel comparison

---

## 7. Pivot Options

### If LPIPS Doesn't Correlate

**Pivot 1: Task-Specific Perceptual Metrics**
- Train LPIPS-style network on task-relevant differences
- Dataset: Pairs of (correct_prediction, incorrect_prediction) with same input
- Objective: Learn what visual differences matter for this task

**Pivot 2: Object-Centric Metrics**
- Track specific objects rather than full-frame comparison
- Use object detection + tracking
- Compare object positions, states, relationships

**Pivot 3: Structured Prediction**
- Don't generate raw pixels
- Generate scene graphs or object-state descriptions
- Verify structured representations instead

### If Verification Loop Doesn't Help

**Pivot 4: Verification for Filtering, Not Correction**
- Use verification to detect errors, not fix them
- Output confidence score instead of corrected prediction
- "I'm 80% confident this prediction is correct"

**Pivot 5: Verification at Training Time**
- Use verification loss during training, not inference
- Model learns to make verification-friendly predictions
- No inference-time verification loop needed

**Pivot 6: Ensemble Verification**
- Generate multiple predictions
- Use verification to select best, not improve each
- Similar to self-consistency in language models

### If Too Expensive

**Pivot 7: Lightweight Verification**
- Use small, efficient metric (SSIM, MSE)
- Verify only final frame, not full video
- Verify only when uncertainty is high

**Pivot 8: Amortized Verification**
- Train model to predict its own verification score
- Cheap uncertainty estimate during inference
- Only invoke actual verification when predicted score is low

---

## 8. Timeline

### Phase 1: Setup (Days 1-3)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Implement metrics (LPIPS, FVD, VLM) | `metrics/` module |
| 2 | Prepare SSv2 evaluation subset | `data/ssv2_eval/` |
| 3 | Verification module skeleton | `verification/` module |

### Phase 2: Core Experiments (Days 4-15)

| Days | Experiment | Deliverable |
|------|------------|-------------|
| 4-6 | E4.1: Correlation study | Correlation analysis, ROC curves |
| 7-9 | E4.2: Calibration study | Calibration plots, ECE scores |
| 10-14 | E4.3: Single verification loop | Accuracy comparison, best feedback type |
| 15-18 | E4.4: Multiple iterations | Iteration curves, optimal k |

### Phase 3: Metric Comparison (Days 19-24)

| Days | Task | Deliverable |
|------|------|-------------|
| 19-22 | E4.5: Compare all metrics | Metric comparison table |
| 23-24 | Hybrid metric experiments | Best combination |

### Phase 4: Analysis and Reporting (Days 25-28)

| Days | Task | Deliverable |
|------|------|-------------|
| 25-26 | Statistical analysis | Significance tests, effect sizes |
| 27-28 | Write-up and recommendations | Final report |

**Total estimated time:** 28 days (6 weeks with buffer)

### Critical Path

```
E4.1 (correlation) --> E4.3 (verification loop) --> E4.4 (iterations)
                   \                             /
                    --> E4.2 (calibration) ------
```

E4.1 must complete before E4.3 (need to know if correlation exists).
E4.2 can run in parallel with E4.1.
E4.5 can start after E4.1 completes.

---

## 9. Dependencies

### Requires Claims 1-3 (All Satisfied)

| Dependency | Status | How Satisfied |
|------------|--------|---------------|
| C1: VLM latents contain information | ✅ P2 validated | Hybrid encoder (DINOv2 + VLM) |
| C2: Adapter bridges latent spaces | ✅ Passed | 10M query adapter, param_efficiency=1.165 |
| C3: Future prediction capability | ✅ E3.8 validated | LTX-Video generates, VLM describes (93% retention) |

**Architecture for C4 (via E3.8 pivot):**
```
Context frames → LTX-Video → Generated future → VLM describes
                                    ↓
                          Compare to actual outcome
                                    ↓
                          Verification signal
```

**Minimum viable setup for C4 experiments:** ✅ MET
- Working end-to-end prediction pipeline: LTX Image-to-Video + Qwen2.5-VL
- VLM action recall on generated: 70% (> 20% threshold)
- Temporal coherence: 0.89 (realistic motion)

### Alternative: Synthetic Testing

If C1-C3 not ready, can do preliminary tests with:
1. Pretrained video prediction model (VideoGPT, MCVD)
2. Synthetic dataset with perfect predictions + injected errors
3. Focus on E4.1 and E4.5 (metric analysis) without full system

### External Dependencies

| Dependency | Source | Version |
|------------|--------|---------|
| LPIPS | pip install lpips | >= 0.1.4 |
| pytorch-fvd | github/cvpr2022-fvd | latest |
| SSv2 dataset | TwentyBN website | v2 |
| Qwen2-VL | HuggingFace | 7B-Instruct |

---

## 10. Deliverables

### Code Artifacts

1. **Verification Module** (`src/verification/`)
   - `metrics.py` - LPIPS, FVD, VLM comparison implementations
   - `verification_loop.py` - Single and multi-iteration verification
   - `calibration.py` - Uncertainty extraction and calibration analysis

2. **Evaluation Scripts** (`scripts/eval/`)
   - `run_correlation_study.py` - E4.1
   - `run_calibration_study.py` - E4.2
   - `run_verification_loop.py` - E4.3, E4.4
   - `compare_metrics.py` - E4.5

3. **Analysis Notebooks** (`notebooks/`)
   - `c4_correlation_analysis.ipynb`
   - `c4_calibration_plots.ipynb`
   - `c4_verification_results.ipynb`

### Documentation

1. **Experiment Report** (`research/results/c4-pixel-verification-report.md`)
   - All metrics and statistical tests
   - Figures: correlation plots, calibration curves, iteration curves
   - Conclusions and recommendations

2. **Technical Appendix**
   - Hyperparameter settings
   - Compute costs breakdown
   - Failure cases analysis

### Decision Outputs

1. **Go/No-Go Recommendation**
   - Is pixel verification worth the cost?
   - Which metric(s) should Foresight use?
   - How many verification iterations?

2. **Architecture Recommendations**
   - Should verification be at inference time or training time?
   - Standalone module vs integrated into model?

3. **Next Steps**
   - If successful: Integration plan for verification module
   - If failed: Recommended pivot from Section 7

---

## Appendix A: Detailed Metric Definitions

### A.1 LPIPS Computation

```python
import lpips
import torch

class VideoLPIPS:
    def __init__(self, net='alex', device='cuda'):
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.device = device

    def __call__(self, video1, video2, reduction='mean'):
        """
        Compute LPIPS between two videos.

        Args:
            video1: Tensor [T, C, H, W] in range [-1, 1]
            video2: Tensor [T, C, H, W] in range [-1, 1]
            reduction: 'mean', 'max', 'per_frame', or 'weighted'

        Returns:
            LPIPS score (lower is better)
        """
        assert video1.shape == video2.shape
        T = video1.shape[0]

        scores = []
        for t in range(T):
            frame1 = video1[t:t+1].to(self.device)
            frame2 = video2[t:t+1].to(self.device)
            score = self.loss_fn(frame1, frame2)
            scores.append(score.item())

        scores = torch.tensor(scores)

        if reduction == 'mean':
            return scores.mean().item()
        elif reduction == 'max':
            return scores.max().item()
        elif reduction == 'per_frame':
            return scores.tolist()
        elif reduction == 'weighted':
            # Weight later frames more (they reflect action outcome)
            weights = torch.linspace(0.5, 1.5, T)
            return (scores * weights).sum().item() / weights.sum().item()
```

### A.2 Task Correctness Evaluation

```python
def evaluate_correctness(predicted_video, ground_truth_action, vlm_model):
    """
    Determine if predicted video shows correct action.

    Returns:
        correct: bool
        predicted_action: str
        confidence: float
    """
    prompt = f"""
    Watch this video carefully and answer:
    What action is being performed on the object?

    Choose from the following options:
    {ACTION_CLASSES}

    Respond with just the action name.
    """

    predicted_action = vlm_model.generate(
        prompt=prompt,
        video=predicted_video
    ).strip()

    correct = (predicted_action.lower() == ground_truth_action.lower())

    return {
        'correct': correct,
        'predicted_action': predicted_action,
        'ground_truth': ground_truth_action
    }
```

---

## Appendix B: Compute Cost Estimates

| Operation | Time (RTX 4090) | VRAM |
|-----------|-----------------|------|
| Single video prediction | ~3 sec | ~25GB |
| LPIPS (full video) | ~0.1 sec | ~1GB |
| FVD (100 videos) | ~30 sec | ~8GB |
| VLM comparison | ~2 sec | ~14GB |

**Per-experiment estimates:**

| Experiment | Predictions | Total Time |
|------------|-------------|------------|
| E4.1 (1000 samples) | 1000 | ~1 hour |
| E4.2 (1000 samples, 10 MC) | 10000 | ~10 hours |
| E4.3 (500 x 5 conditions x 2 rounds) | 5000 | ~4 hours |
| E4.4 (500 x 5 iterations) | 2500 | ~2 hours |
| E4.5 (1000 x 8 metrics) | 1000 + metric compute | ~3 hours |

**Total compute: ~20 GPU-hours for core experiments**

---

## Appendix C: Related Work References

1. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018) - LPIPS
2. Unterthiner et al., "FVD: A New Metric for Video Generation" (2019)
3. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" (2022) - Verification in LLMs
4. Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination" (2020) - World model verification
5. Guo et al., "On Calibration of Modern Neural Networks" (2017) - Calibration metrics
6. Goyal et al., "Something-Something v2" (2017) - Dataset

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | 2025-01-18 | Initial research plan |
