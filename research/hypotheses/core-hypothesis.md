# Core Hypothesis

## Primary Hypothesis

> An AI system that generates explicit pixel-level predictions of future states and can compare those predictions against actual outcomes will make more accurate decisions than systems reasoning purely in text/token space.

## Why This Might Be True

The Dreamer line of work proves that "imagining" future states improves decision-making in RL agents. V-JEPA shows latent prediction is efficient but Meta researchers note it may be "sophisticated pattern matching" - there's no way to verify predictions against reality.

Our hypothesis: **pixel grounding provides verification**. By generating actual video frames, we can:
1. Compare predictions to reality (LPIPS, perceptual metrics)
2. Detect hallucinations (when predictions diverge from physics)
3. Enable interpretability (humans can inspect what the model "thinks" will happen)

---

## Testable Claims

We break the primary hypothesis into **four independent claims**, each testable in isolation. See [Experiment Plans](../experiments/README.md) for detailed research protocols.

### Claim 1: VLM Latents Contain Sufficient Information
📋 **[Detailed Experiment Plan](../experiments/c1-vlm-latent-sufficiency.md)**

**Statement:** Qwen2-VL's internal representations contain enough information to reconstruct the input video at reasonable fidelity.

**Why this matters:** If we can't reconstruct what the VLM *sees*, we certainly can't generate what it *predicts*.

**Independent Test:**
```
Input: Video frames
Process: Encode with Qwen2-VL → Extract latents → Adapter → Video decoder
Output: Reconstructed video
Compare: Original vs reconstructed
```

**Metrics:**
- LPIPS (perceptual similarity) < 0.3 indicates good reconstruction
- FVD (Frechet Video Distance) comparable to other video models
- Human eval: "Is this recognizably the same scene?"

**Success criterion:** Reconstruction quality sufficient to preserve task-relevant details (objects, positions, actions).

**Failure mode:** Latents lose spatial information during VLM processing (especially after token merging). May need to extract pre-merge patch embeddings.

---

### Claim 2: Small Adapter Can Bridge Latent Spaces
📋 **[Detailed Experiment Plan](../experiments/c2-adapter-bridging.md)**

**Statement:** A small adapter network (~5-10M params) can translate VLM latent space to video decoder conditioning space.

**Why this matters:** If bridging requires massive retraining, the approach isn't practical.

**Independent Test:**
```
Input: VLM latents from real video (not predictions)
Process: Adapter → Video decoder
Output: Generated video
Compare: Original video vs generated
```

**Metrics:**
- Training convergence (loss decreases, doesn't diverge)
- Reconstruction quality (same as Claim 1)
- Parameter efficiency: quality vs adapter size curve

**Success criterion:** 10M parameter adapter achieves >80% of quality from 100M adapter.

**Failure mode:** Latent spaces are too different; adapter essentially needs to memorize mappings rather than learn a transform.

---

### Claim 3: VLM Can Predict Future States in Latent Space
📋 **[Detailed Experiment Plan](../experiments/c3-future-prediction.md)**

**Statement:** Given current video + action/question, the VLM (with learned query tokens) produces latents that align with actual future frames.

**Why this matters:** This is the "imagination" step - can the VLM reason about what comes next?

**Independent Test:**
```
Input: Video frames 1-10 + action description
Process: VLM encodes → Query tokens extract "predicted future" latent
Ground truth: VLM encoding of actual frames 11-20
Compare: Predicted latent vs actual future latent
```

**Metrics:**
- Cosine similarity between predicted and actual latents
- Clustering: do predictions land near correct futures in latent space?
- Action classification: can we recover the action from predicted latents?

**Success criterion:** Predicted latents are closer to correct futures than to random futures (statistically significant).

**Failure mode:** Query tokens just learn to copy current state, or produce "average" futures that don't reflect the specific action.

---

### Claim 4: Pixel Verification Improves Accuracy
📋 **[Detailed Experiment Plan](../experiments/c4-pixel-verification.md)**

**Statement:** Comparing predicted video to actual outcomes provides a signal that improves prediction accuracy.

**Why this matters:** This is the key differentiator from latent-only approaches like V-JEPA.

**Independent Test:**
```
Condition A (no verification):
  Predict once → measure accuracy

Condition B (with verification):
  Predict → compare to actual → predict again → measure accuracy
```

**Metrics:**
- Accuracy improvement from verification loop
- Correlation: does high LPIPS error predict incorrect predictions?
- Calibration: does the model "know when it's wrong"?

**Success criterion:** Verification loop improves accuracy by >10% on incorrect predictions.

**Failure mode:** LPIPS doesn't correlate with task-relevant errors (model can be perceptually correct but semantically wrong, or vice versa).

---

## Open Questions (What We Don't Know)

See [Experiment Plans](../experiments/README.md) for detailed research protocols.

### Q1: Latent Space Alignment
📋 **[Detailed Experiment Plan](../experiments/q1-latent-alignment.md)**

**Question:** VLM latents and video decoder latents were trained on different objectives. How hard is it to align them?

**Risk level:** High - this is the core technical uncertainty.

**Experiments to resolve:**
1. Visualize both latent spaces (t-SNE/UMAP) - do they have similar structure?
2. Linear probe: can we predict video decoder latents from VLM latents?
3. Try multiple adapter architectures (linear, MLP, cross-attention)

### Q2: Information Preservation Through VLM
📋 **[Detailed Experiment Plan](../experiments/q2-information-preservation.md)**

**Question:** Qwen2-VL merges 4 patches into 1 token. Does this lose spatial information needed for video generation?

**Risk level:** Medium - may require architectural changes.

**Experiments to resolve:**
1. Compare reconstruction from pre-merge vs post-merge latents
2. Try extracting from different VLM layers
3. Measure spatial reconstruction accuracy (object positions, sizes)

### Q3: Temporal Coherence
📋 **[Detailed Experiment Plan](../experiments/q3-temporal-coherence.md)**

**Question:** Can we generate temporally coherent video (not just good individual frames)?

**Risk level:** Medium - video decoders handle this, but our conditioning may break it.

**Experiments to resolve:**
1. Measure frame-to-frame consistency metrics
2. Human eval of motion smoothness
3. Compare to baseline video decoder without our conditioning

### Q4: Training Data Requirements
📋 **[Detailed Experiment Plan](../experiments/q4-training-data.md)**

**Question:** How much paired (video, action, outcome) data do we need?

**Risk level:** Low-medium - datasets exist, but scale is unclear.

**Experiments to resolve:**
1. Learning curves: quality vs dataset size
2. Transfer: does training on COIN help with CrossTask?
3. Minimum viable dataset for proof-of-concept

### Q5: The Right Prediction Target
📋 **[Detailed Experiment Plan](../experiments/q5-prediction-horizon.md)**

**Question:** Should we predict the next 1 second? 5 seconds? Just the final state?

**Risk level:** Low - empirical question.

**Experiments to resolve:**
1. Compare different prediction horizons
2. Measure accuracy vs horizon length
3. Find the sweet spot for reasoning utility vs generation difficulty

---

## End-to-End Evaluation

Once components are validated, test the full system:

### Primary Benchmark: Action Prediction

**Task:** Given video of initial state + multiple choice actions, predict which action was taken based on outcome.

**Dataset:** Something-Something v2 (object interactions with clear outcomes)

**Baselines:**
| Baseline | Description |
|----------|-------------|
| Text-only VLM | Qwen2-VL answers in text, no video generation |
| Random | Chance performance |
| Human | Human accuracy on same task |

**Comparison conditions:**
| Condition | Description |
|-----------|-------------|
| Foresight (ours) | Full system with pixel prediction |
| Latent-only | Predict in latent space, no pixel generation |
| No verification | Pixel prediction but no comparison loop |

### Secondary Benchmarks

1. **Physical reasoning:** CLEVRER, Physion (do predictions obey physics?)
2. **Procedural activities:** COIN, CrossTask (multi-step action sequences)
3. **Generation quality:** FVD, human preference vs baseline video models

---

## Falsification Criteria

The hypothesis is **falsified** if any of:

1. **Component failures that can't be fixed:**
   - Claim 1 false: VLM latents fundamentally lack spatial information
   - Claim 2 false: Adapter can't bridge spaces even with more capacity
   - Claim 3 false: VLM can't predict futures better than chance

2. **System-level failures:**
   - Full system doesn't beat text-only baseline on action prediction
   - Pixel prediction doesn't beat latent-only prediction
   - Verification loop doesn't improve accuracy

3. **Practical failures:**
   - Requires >80GB VRAM (can't run on available hardware)
   - Requires >100M training examples (impractical to collect)
   - Inference takes >10 seconds per prediction (too slow for reasoning)

---

## Experiment Sequence

Ordered to fail fast and isolate issues:

### Phase 1: Validate Components (Weeks 1-4)

| Week | Experiment | Go/No-Go |
|------|------------|----------|
| 1 | Extract Qwen2-VL latents, visualize | Do latents look reasonable? |
| 2 | Train reconstruction adapter (real video → real video) | LPIPS < 0.4? |
| 3 | Train future prediction (video → future latent) | Better than random? |
| 4 | End-to-end generation (video + action → predicted video) | Coherent output? |

### Phase 2: Measure Claims (Weeks 5-8)

| Week | Experiment | Measures |
|------|------------|----------|
| 5-6 | Action prediction benchmark | Claims 1-3 |
| 7 | Add verification loop | Claim 4 |
| 8 | Compare to baselines | Primary hypothesis |

### Phase 3: Ablations & Analysis (Weeks 9-12)

- What components matter most?
- Where do failures come from?
- What's the minimum viable system?

---

## Alternative Hypotheses to Test Against

### A1: Latent Prediction is Sufficient (V-JEPA position)

**Their claim:** Predicting in latent space is more efficient and equally effective.

**Our counter:** Latent predictions can't be verified against reality.

**Test:** Compare latent-only vs pixel prediction on tasks requiring physical accuracy.

### A2: Text Reasoning is Sufficient

**Their claim:** Better prompting/CoT achieves similar results without generation.

**Our counter:** Text can't represent spatial/temporal relationships precisely.

**Test:** Compare our system vs CoT-prompted VLM on spatial reasoning tasks.

### A3: Just Scale the VLM

**Their claim:** Larger VLMs will naturally develop world models.

**Our counter:** Even GPT-4V can't generate or verify visual predictions.

**Test:** Compare our 7B system vs larger text-only VLMs (70B+).

---

## Evolution Log

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | Initial | "Video generation helps reasoning" (vague) |
| v0.2 | Initial | Specific GLP architecture, measurable claims |
| v0.3 | 2025-01-18 | Added component-level tests, open questions, experiment sequence |
| v0.4 | 2025-01-18 | Linked detailed experiment plans for all claims and questions |

## Related Documents

- [Paper Index](../papers/index.md)
- [Product Requirements](../../PRD.md)
- [Research Overview](../README.md)
