# Q4: Training Data Requirements

**Open Question:** How much paired (video, action, outcome) data do we need?

**Risk Level:** LOW-MEDIUM - Datasets exist, but scale requirements are unclear.

**Status:** Planning

**Last Updated:** 2025-01-18

---

## 1. Objective

Determine the minimum viable dataset size and composition required to train the GLP (Generative Latent Prediction) adapter to achieve meaningful prediction quality. Specifically:

1. Establish learning curves showing performance vs. dataset size
2. Compare data efficiency across different source datasets
3. Identify diminishing returns thresholds
4. Determine if existing public datasets are sufficient or if custom data collection is needed
5. Quantify quality vs. quantity tradeoffs

**Primary Goal:** Find the smallest dataset that achieves >80% of maximum performance to enable rapid prototyping and iteration.

---

## 2. Background

### 2.1 Available Datasets with Action Labels

| Dataset | Size | Action Types | Video Length | Year |
|---------|------|--------------|--------------|------|
| COIN | 11,827 videos | 180 tasks, 778 steps | 2.4 min avg | 2019 |
| CrossTask | 4,700 videos | 83 tasks, 105 steps | 4.5 min avg | 2019 |
| Something-Something v2 | 220,847 videos | 174 action classes | 2-6 sec | 2017 |
| HowTo100M | 1.2M videos | Weak labels (ASR) | Variable | 2019 |
| Ego4D | 3,670 hours | Diverse egocentric | Variable | 2022 |
| Epic-Kitchens-100 | 100 hours | Kitchen activities | Variable | 2022 |

### 2.2 Dataset Statistics Deep Dive

**COIN (Comprehensive Instructional Videos)**
- **Total clips:** ~47,000 annotated segments
- **Domains:** 12 categories (vehicles, gadgets, food, etc.)
- **Label quality:** Human-annotated action boundaries and descriptions
- **Resolution:** Variable (YouTube sourced, typically 720p+)
- **Pros:** Clean labels, diverse procedural tasks, clear action-outcome pairs
- **Cons:** Instructional bias (people explaining actions), limited object diversity

**CrossTask**
- **Total clips:** ~18,000 annotated segments
- **Domains:** 18 primary task categories
- **Label quality:** Weak supervision from narration, manually verified
- **Resolution:** Variable (YouTube sourced)
- **Pros:** Natural task execution, step-by-step annotations
- **Cons:** Smaller scale, more noise in labels, limited action variety

**Something-Something v2**
- **Total clips:** 220,847 short clips
- **Domains:** Object manipulation (single-object focus)
- **Label quality:** Crowd-sourced, template-based ("Pushing X from left to right")
- **Resolution:** Standardized (12 fps, ~2-6 seconds)
- **Pros:** Large scale, clean labels, consistent format, clear cause-effect
- **Cons:** Limited to tabletop manipulation, artificial scenarios, short clips

### 2.3 Prior Work on Data Efficiency in Video Models

| Work | Finding | Relevance |
|------|---------|-----------|
| **Stable Video Diffusion** | WebVid-10M + proprietary data; shows importance of data curation over raw scale | High - similar architecture |
| **LTX-Video** | Trained on internal dataset; quality degrades significantly with <1M samples | High - our target decoder |
| **VideoLDM** | 10M+ video-text pairs for good generalization | Medium - text conditioning |
| **JEPA/V-JEPA** | Self-supervised; 2M videos for video understanding | Medium - different objective |
| **Dreamer** | <100K interactions sufficient in controlled RL settings | Low - different domain |
| **GR-1 (Robotics)** | ~130K trajectories for generalist robot policy | Medium - action-conditioned |

**Key Insight:** Video generation models typically require 1M+ samples for good quality, but our task is narrower (adapter training with frozen backbones). Prior work on adapters (LoRA, etc.) suggests 10K-100K samples may suffice for domain-specific adaptation.

---

## 3. Dataset Analysis

### 3.1 COIN Dataset

**Statistics:**
- 11,827 videos across 180 tasks
- 778 unique step labels
- Average video length: 2.4 minutes
- Total hours: ~470 hours
- Annotation: Action start/end timestamps + text description

**Action Types:**
- Procedural tasks (cooking, repairs, crafts)
- Clear sequential structure
- Explicit goal states (before/after visible)

**Video Quality:**
- Source: YouTube instructional videos
- Resolution: Variable (mostly 720p-1080p)
- Frame rate: Variable (24-30 fps typical)
- Lighting: Generally good (instructional content)

**Suitability for GLP:**
| Factor | Rating | Notes |
|--------|--------|-------|
| Action clarity | 4/5 | Clear demonstrations |
| Outcome visibility | 4/5 | Goal states usually shown |
| Temporal alignment | 3/5 | Some narration delay |
| Visual diversity | 3/5 | Limited environments |
| Scale | 3/5 | Moderate size |

**Preprocessing Required:**
- Extract action segments using timestamps
- Filter for clear before/after visibility
- Subsample to consistent frame rate
- Quality filtering (blur, occlusion)

### 3.2 CrossTask Dataset

**Statistics:**
- 4,700 videos across 83 tasks
- 105 unique step labels
- Average video length: 4.5 minutes
- Total hours: ~350 hours
- Annotation: Step labels with approximate timestamps

**Action Types:**
- Daily tasks (cooking, DIY, beauty)
- More natural/less staged than COIN
- Variable execution styles

**Video Quality:**
- Source: YouTube
- Resolution: Variable
- More natural (less staged) appearance

**Suitability for GLP:**
| Factor | Rating | Notes |
|--------|--------|-------|
| Action clarity | 3/5 | More natural, less clear |
| Outcome visibility | 3/5 | Variable |
| Temporal alignment | 2/5 | Weaker annotations |
| Visual diversity | 4/5 | More varied settings |
| Scale | 2/5 | Smaller dataset |

**Preprocessing Required:**
- More aggressive filtering needed
- Manual verification of subset recommended
- Alignment correction for temporal labels

### 3.3 Something-Something v2 Dataset

**Statistics:**
- 220,847 video clips
- 174 template-based action classes
- Clip length: 2-6 seconds
- Total hours: ~180 hours
- Annotation: Template labels (verb + object + direction)

**Action Types:**
- Object manipulation primitives
- Examples: "Pushing X from left to right", "Putting X into Y"
- Highly controlled action vocabulary

**Video Quality:**
- Source: Crowd-sourced recordings
- Resolution: Standardized
- Frame rate: 12 fps
- Consistent tabletop setting

**Suitability for GLP:**
| Factor | Rating | Notes |
|--------|--------|-------|
| Action clarity | 5/5 | Designed for action recognition |
| Outcome visibility | 5/5 | Clear before/after |
| Temporal alignment | 5/5 | Short, complete actions |
| Visual diversity | 2/5 | Limited to tabletop |
| Scale | 5/5 | Largest dataset |

**Preprocessing Required:**
- Minimal - already well-formatted
- May need upsampling for video decoder (12fps -> 24fps)
- Object segmentation optional (could improve conditioning)

### 3.4 Dataset Comparison Summary

| Criterion | COIN | CrossTask | SSv2 |
|-----------|------|-----------|------|
| **Recommended for** | Mid-stage training | Transfer experiments | Initial development |
| **Primary strength** | Procedural diversity | Naturalistic | Scale + clarity |
| **Primary weakness** | Instructional bias | Label noise | Domain narrow |
| **Data efficiency** | Medium | Low | High |
| **Preprocessing effort** | Medium | High | Low |

**Recommendation:** Start with Something-Something v2 for rapid iteration due to clean labels and large scale, then validate on COIN for procedural tasks.

---

## 4. Experiments

### E-Q4.1: Learning Curves

**Objective:** Determine how performance scales with dataset size.

**Method:**
1. Use Something-Something v2 as primary dataset
2. Create fixed train/val/test splits (80/10/10)
3. Train adapter on subsets: 1%, 5%, 10%, 25%, 50%, 100% of training data
4. Keep all other hyperparameters constant
5. Evaluate on full test set

**Subset Sizes (SSv2):**
| Subset | Training Samples | Estimated Training Time |
|--------|-----------------|------------------------|
| 1% | ~1,800 | 1-2 hours |
| 5% | ~8,800 | 4-6 hours |
| 10% | ~17,600 | 8-12 hours |
| 25% | ~44,000 | 20-30 hours |
| 50% | ~88,000 | 40-60 hours |
| 100% | ~176,000 | 80-120 hours |

**Metrics:**
- LPIPS (perceptual similarity)
- FVD (Frechet Video Distance)
- Action prediction accuracy (downstream task)
- Latent cosine similarity (predicted vs actual)

**Analysis:**
- Plot learning curves (metric vs log(dataset size))
- Fit power law: performance = a * N^b + c
- Identify diminishing returns threshold (where derivative < 0.1)
- Estimate sample efficiency: how many samples for 80%, 90%, 95% of max performance?

**Expected Output:**
```
Dataset Size | LPIPS | FVD | Action Acc | Training Hours
-------------|-------|-----|------------|---------------
1% (1.8K)    | ?     | ?   | ?          | 2
5% (8.8K)    | ?     | ?   | ?          | 6
10% (17.6K)  | ?     | ?   | ?          | 12
25% (44K)    | ?     | ?   | ?          | 30
50% (88K)    | ?     | ?   | ?          | 60
100% (176K)  | ?     | ?   | ?          | 120
```

**Success Criteria:**
- Clear diminishing returns visible (curve flattens)
- Identify threshold N where performance > 80% of maximum

---

### E-Q4.2: Dataset Source Comparison

**Objective:** Compare data efficiency across different source datasets.

**Method:**
1. Select matched subsets from each dataset:
   - 10,000 training samples from SSv2
   - 10,000 training samples from COIN
   - 4,000 training samples from CrossTask (maximum available)
2. Train separate adapters with identical hyperparameters
3. Evaluate each on:
   - Its own test set (in-distribution)
   - Other datasets' test sets (out-of-distribution)

**Evaluation Matrix:**

| Training Data | Test on SSv2 | Test on COIN | Test on CrossTask |
|---------------|--------------|--------------|-------------------|
| SSv2 10K | In-dist | OOD | OOD |
| COIN 10K | OOD | In-dist | OOD |
| CrossTask 4K | OOD | OOD | In-dist |

**Metrics:**
- Same as E-Q4.1 (LPIPS, FVD, Action Acc, Latent similarity)

**Analysis:**
- Which dataset provides best in-distribution performance?
- Which dataset transfers best to other domains?
- What is the OOD performance drop?

**Success Criteria:**
- Identify which dataset provides best sample efficiency
- Quantify transfer gap between datasets

---

### E-Q4.3: Transfer Learning

**Objective:** Determine if pretraining on one dataset improves learning on another.

**Method:**
1. Pretrain adapter on source dataset A (full dataset)
2. Finetune on target dataset B (varying amounts)
3. Compare to training from scratch on B

**Transfer Pairs to Test:**

| Source (Pretrain) | Target (Finetune) | Rationale |
|-------------------|-------------------|-----------|
| SSv2 (large) | COIN | Scale -> Procedural |
| COIN | SSv2 | Procedural -> Manipulation |
| SSv2 + COIN | CrossTask | Combined -> Natural |
| HowTo100M (subset) | SSv2 | Weak labels -> Strong labels |

**Finetuning Amounts:** 1%, 10%, 50%, 100% of target dataset

**Metrics:**
- Same as E-Q4.1
- Transfer efficiency: (pretrained_performance - scratch_performance) / scratch_performance

**Analysis:**
- Does pretraining help? How much?
- How much target data needed to match scratch training?
- Negative transfer cases?

**Success Criteria:**
- Pretraining reduces required target data by >50%
- OR: Identify when pretraining doesn't help (saves time)

---

### E-Q4.4: Data Augmentation Effectiveness

**Objective:** Measure impact of augmentation on data efficiency.

**Method:**
1. Train on fixed small subset (10% of SSv2)
2. Compare augmentation strategies:

| Augmentation | Description |
|--------------|-------------|
| None (baseline) | Raw video clips |
| Spatial | Random crop, flip, color jitter |
| Temporal | Speed variation (0.8x-1.2x), frame skip |
| Spatial + Temporal | Combined |
| CutMix/MixUp | Video-level mixing |
| Synthetic | Generate variations with video model |

**Implementation:**
```python
# Augmentation configurations
aug_configs = {
    'spatial': {
        'random_crop': 0.8,  # crop ratio
        'horizontal_flip': 0.5,  # probability
        'color_jitter': 0.2  # intensity
    },
    'temporal': {
        'speed_range': (0.8, 1.2),
        'frame_skip': [1, 2]  # skip patterns
    }
}
```

**Metrics:**
- Performance improvement vs baseline
- Effective dataset multiplier (aug performance at N = baseline performance at M*N)

**Analysis:**
- Which augmentations help most?
- Augmentation + small data vs. no augmentation + large data?

**Success Criteria:**
- Identify augmentation that provides >1.5x effective data multiplier

---

### E-Q4.5: Minimum Viable Dataset

**Objective:** Find smallest dataset that achieves proof-of-concept quality.

**Definition of Proof-of-Concept Quality:**
1. Generated video is recognizable as the predicted action (human eval >60%)
2. LPIPS < 0.4 (moderate perceptual similarity)
3. Action prediction accuracy > random baseline + 20%
4. Qualitative: outputs are coherent, not noise

**Method:**
1. Start with 100 samples, double until quality threshold met
2. Use best augmentation strategy from E-Q4.4
3. Use best dataset from E-Q4.2

**Sample Sizes to Test:** 100, 200, 500, 1K, 2K, 5K, 10K

**Evaluation:**
- Automated metrics (LPIPS, FVD)
- Human eval (Amazon MTurk or internal): "Does this video show [action]? Y/N"
- Qualitative inspection of failure modes

**Success Criteria:**
- Identify minimum N where all PoC criteria met
- Target: N < 10,000 samples

---

### E-Q4.6: Data Quality vs Quantity Tradeoff

**Objective:** Determine if curated small data beats uncurated large data.

**Method:**
1. Create quality tiers of SSv2 data:

| Tier | Criteria | Est. Size |
|------|----------|-----------|
| Platinum | Manual review, perfect examples | 5K |
| Gold | Auto-filtered (motion, clarity scores) | 20K |
| Silver | Basic filtering (length, resolution) | 80K |
| Bronze | All data, no filtering | 176K |

**Quality Filtering Pipeline:**
```python
quality_filters = {
    'motion_score': lambda v: optical_flow_magnitude(v) > threshold,
    'clarity_score': lambda v: laplacian_variance(v) > threshold,
    'action_completeness': lambda v: action_detector(v).confidence > 0.8,
    'no_occlusion': lambda v: occlusion_detector(v) < 0.2
}
```

2. Train on each tier and measure performance

**Comparison:**
| Training Data | Samples | Performance |
|---------------|---------|-------------|
| Platinum (curated) | 5K | ? |
| Gold (auto-filtered) | 20K | ? |
| Silver (basic) | 80K | ? |
| Bronze (all) | 176K | ? |

**Analysis:**
- Performance per sample (efficiency)
- Break-even points (where quantity overcomes quality)
- Cost-benefit of manual curation

**Success Criteria:**
- Quantify quality premium: how much is 1 curated sample worth?
- Recommendation: curate or scale?

---

## 5. Success Metrics

### 5.1 Primary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Minimum viable dataset size** | <10K samples | E-Q4.5 |
| **80% performance threshold** | Identify N | E-Q4.1 |
| **Transfer efficiency** | >50% reduction in target data | E-Q4.3 |
| **Augmentation multiplier** | >1.5x | E-Q4.4 |

### 5.2 Learning Curve Deliverables

```
Performance vs Dataset Size (SSv2)
|
|                    xxxxxxxxxx (plateau)
|               xxxxx
|          xxxx
|       xxx
|    xx
|  x
|x
+-------------------------> log(N)
     1K   10K   100K
```

**Required Outputs:**
- Power law fit parameters (a, b, c)
- Inflection point (where second derivative = 0)
- 80%, 90%, 95% performance thresholds

### 5.3 Transfer Efficiency Matrix

| Pretrain \ Finetune | SSv2 | COIN | CrossTask |
|---------------------|------|------|-----------|
| None | baseline | baseline | baseline |
| SSv2 | - | +X% | +Y% |
| COIN | +A% | - | +B% |
| Combined | +C% | +D% | +E% |

---

## 6. Failure Criteria

The data requirements are **problematic** if:

### 6.1 Scale Requirements Too High

- Minimum viable dataset > 100K samples (requires custom collection)
- 80% performance threshold > 500K samples
- No clear diminishing returns (linear scaling required)

**Mitigation if true:**
- Investigate self-supervised pretraining
- Use synthetic data generation
- Simplify task (shorter predictions, fewer action types)

### 6.2 Transfer Doesn't Work

- Pretraining provides <10% improvement
- Negative transfer observed
- OOD performance drops >50%

**Mitigation if true:**
- Domain-specific adapters instead of universal
- Multi-task training from the start
- Investigate domain adaptation techniques

### 6.3 Existing Datasets Insufficient

- All datasets show quality ceiling below acceptable threshold
- Domain mismatch too large for target application
- Label noise prevents convergence

**Mitigation if true:**
- Custom data collection (see Section 7)
- Label cleaning/refinement
- Active learning for efficient annotation

### 6.4 Quality vs Quantity Inconclusive

- No clear winner between curation strategies
- Quality filtering doesn't improve efficiency
- Manual curation cost > benefit

**Mitigation if true:**
- Use simple auto-filtering only
- Focus on scale over curation
- Develop better quality estimation

---

## 7. Data Collection Options

If existing datasets prove insufficient, we have these options:

### 7.1 Option A: Crowdsourced Collection

**Approach:** MTurk/Prolific workers record short videos following templates

**Estimated Costs:**
| Amount | Quality | Cost | Time |
|--------|---------|------|------|
| 10K clips | High (instructions) | $5-10K | 2-4 weeks |
| 50K clips | Medium | $15-25K | 4-8 weeks |
| 100K clips | Variable | $30-50K | 8-12 weeks |

**Template Example:**
```
Record a 5-second video showing:
"Pushing the [RED OBJECT] from the left side of the table to the right"

Requirements:
- Stable camera (use phone stand)
- Good lighting
- Object clearly visible throughout
- Complete action start to finish
```

### 7.2 Option B: Simulation Data

**Approach:** Generate synthetic video-action pairs in simulation

**Options:**
- AI2-THOR (household environments)
- RLBench (robotic manipulation)
- Habitat (navigation)
- Custom Unity/Unreal scenes

**Pros:** Unlimited scale, perfect labels, controllable
**Cons:** Sim-to-real gap, limited visual diversity

**Estimated Effort:**
| Environment | Setup Time | Generation Rate |
|-------------|------------|-----------------|
| AI2-THOR | 1-2 weeks | 10K/day |
| Custom Unity | 4-8 weeks | 50K/day |

### 7.3 Option C: Web Scraping + Auto-labeling

**Approach:** Scrape instructional videos, use VLM for labeling

**Pipeline:**
1. Scrape YouTube tutorials by keyword
2. Run Qwen2-VL to identify action segments
3. Filter by confidence threshold
4. Manual verification of subset

**Estimated Yield:**
- 100K scraped videos -> ~20K usable segments (after filtering)
- Cost: Compute only (~$500-1K)
- Time: 1-2 weeks

### 7.4 Option D: Hybrid Approach

**Recommended if custom collection needed:**

| Component | Source | Amount |
|-----------|--------|--------|
| Foundation | SSv2 + COIN | 200K |
| Domain-specific | Crowdsourced | 10-20K |
| Hard examples | Active learning selection | 5K |

---

## 8. Timeline

### Phase 1: Setup and Baseline (Week 1)
| Day | Task | Hours |
|-----|------|-------|
| 1-2 | Dataset download and preprocessing | 8 |
| 3 | Create data loading pipeline | 4 |
| 4 | Implement evaluation metrics | 4 |
| 5 | Baseline training (100% SSv2) | 8* |

*Training runs overnight

### Phase 2: Learning Curves (Week 2)
| Day | Task | Hours |
|-----|------|-------|
| 1 | Launch E-Q4.1 experiments (1%, 5%, 10%) | 4 |
| 2-3 | Training runs (parallel on multiple GPUs) | 16* |
| 4 | Launch remaining (25%, 50%) | 4 |
| 5 | Analysis and visualization | 6 |

### Phase 3: Dataset Comparison (Week 3)
| Day | Task | Hours |
|-----|------|-------|
| 1 | Preprocess COIN and CrossTask | 8 |
| 2-3 | E-Q4.2 experiments | 12* |
| 4-5 | E-Q4.3 transfer experiments (start) | 8* |

### Phase 4: Augmentation and Quality (Week 4)
| Day | Task | Hours |
|-----|------|-------|
| 1-2 | E-Q4.4 augmentation experiments | 12* |
| 3-4 | E-Q4.6 quality filtering + training | 12* |
| 5 | E-Q4.5 minimum viable dataset | 8* |

### Phase 5: Analysis and Recommendations (Week 5)
| Day | Task | Hours |
|-----|------|-------|
| 1-2 | Complete all experiments, gather results | 8 |
| 3-4 | Statistical analysis, curve fitting | 8 |
| 5 | Write recommendations report | 6 |

**Total Timeline:** ~5 weeks
**Total Active Hours:** ~40-50 hours
**Total GPU Hours:** ~200-300 hours (parallelizable)

---

## 9. Dependencies

### 9.1 Required Before Starting

| Dependency | Status | Blocker Level |
|------------|--------|---------------|
| Working adapter training (C2) | Required | Hard |
| Video decoder integration | Required | Hard |
| Evaluation metrics implemented | Required | Hard |
| Dataset access (SSv2, COIN) | Required | Hard |
| Multi-GPU training setup | Recommended | Soft |

### 9.2 Component Dependencies

```
C1 (Latent extraction) ──┐
                         ├──> This experiment (Q4)
C2 (Adapter training) ───┘
                         │
                         v
            Q4 results inform ──> Full system training
```

### 9.3 Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 24GB | 40GB+ |
| Storage | 500GB | 1TB |
| GPUs for parallel | 1 | 4+ |
| Compute budget | 200 GPU-hours | 500 GPU-hours |

---

## 10. Deliverables

### 10.1 Primary Deliverables

| Deliverable | Format | Description |
|-------------|--------|-------------|
| **Learning curves** | Plots + data | Performance vs dataset size |
| **Dataset recommendations** | Report | Which dataset(s) to use and why |
| **Data pipeline** | Code | Scripts for loading, preprocessing, augmenting |
| **Minimum viable config** | Config file | Smallest effective training setup |

### 10.2 Code Artifacts

```
experiments/q4-training-data/
├── data/
│   ├── prepare_ssv2.py
│   ├── prepare_coin.py
│   ├── prepare_crosstask.py
│   └── augmentation.py
├── training/
│   ├── train_subset.py
│   ├── configs/
│   │   ├── ssv2_1pct.yaml
│   │   ├── ssv2_10pct.yaml
│   │   └── ...
│   └── sweep.sh
├── evaluation/
│   ├── compute_metrics.py
│   ├── generate_curves.py
│   └── human_eval_template.html
├── analysis/
│   ├── fit_power_law.py
│   ├── transfer_matrix.py
│   └── visualize_results.py
└── results/
    ├── learning_curves.json
    ├── transfer_matrix.json
    └── recommendations.md
```

### 10.3 Final Report Contents

1. **Executive Summary**
   - Minimum viable dataset: N samples
   - Recommended dataset: X
   - Key finding: [one sentence]

2. **Learning Curves**
   - Fitted parameters
   - Threshold analysis
   - Extrapolation to larger scales

3. **Dataset Comparison**
   - Efficiency rankings
   - Transfer matrix
   - Domain-specific recommendations

4. **Recommendations**
   - For proof-of-concept: use X with Y augmentation
   - For production: use Z with W samples
   - If insufficient: pursue option A from Section 7

5. **Appendix**
   - All experiment configurations
   - Raw results tables
   - Failure case analysis

---

## Evolution Log

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | 2025-01-18 | Initial research plan |

---

## Related Documents

- [Core Hypothesis](../hypotheses/core-hypothesis.md) - Open Question Q4
- [Claim 2: Adapter Training](TBD) - Training dependency
- [Paper: LTX-Video](../papers/ltx-video.md) - Video decoder data requirements
