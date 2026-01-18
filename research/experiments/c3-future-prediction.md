# Experiment C3: VLM Future State Prediction in Latent Space

**Claim:** Given current video + action/question, the VLM (with learned query tokens) produces latents that align with actual future frames.

**Status:** Not Started
**Dependencies:** Partial (C1 for validation, but can proceed independently for latent-only experiments)
**Estimated Duration:** 3-4 weeks
**Last Updated:** 2025-01-18

---

## 1. Objective

**Primary Goal:** Determine whether a Vision-Language Model (Qwen2.5-VL-7B-Instruct), augmented with learned query tokens, can predict latent representations of future video frames given:
1. Current video context (frames 1-N)
2. An action description or question about what happens next

**Core Question:** Can we train query tokens that extract "imagined future" representations from the VLM that meaningfully correlate with the VLM's encoding of actual future frames?

**Success Definition:** Predicted latents are statistically closer to correct future latents than to:
- Random future latents from other videos
- Future latents from videos with different actions
- A "copy current state" baseline

---

## 2. Background

### 2.1 Query Tokens and Learnable Prompts

Query tokens (also called "learnable queries" or "soft prompts") are trainable embedding vectors appended to the input sequence that learn to extract specific information from a frozen model. This approach was pioneered by:

**Q-Former (BLIP-2, Li et al. 2023):**
- 32 learnable query tokens interact with frozen image encoder via cross-attention
- Queries learn to extract visual features relevant for language grounding
- Key insight: queries can be trained to extract *specific types* of information without modifying the base model

**Perceiver Resampler (Flamingo, Alayrac et al. 2022):**
- Fixed number of latent queries cross-attend to variable-length visual input
- Produces fixed-size output regardless of input resolution
- Demonstrates queries can compress and abstract visual information

**Prefix Tuning (Li & Liang, 2021):**
- Learnable prefix tokens prepended to transformer input
- Shows soft prompts can steer model behavior with minimal parameters

### 2.2 Our Approach: Future Prediction Queries

We extend the query token concept to future prediction:

```
Standard Q-Former:    Image -> Query Tokens -> Extract relevant features
Our approach:         Video + Action -> Query Tokens -> Extract "predicted future" features
```

**Hypothesis:** If the VLM has learned a world model through pretraining, query tokens can learn to "ask" it: "Given this video and this action, what would the next frames look like in your latent space?"

### 2.3 Prior Art in Latent Future Prediction

**V-JEPA (Bardes et al. 2024):**
- Predicts masked spatiotemporal regions in latent space
- Uses predictor network (not query tokens) to fill in missing regions
- Key difference: V-JEPA predicts *masked* regions, we predict *future* regions

**VideoGPT (Yan et al. 2021):**
- Autoregressive prediction in discrete latent space (VQ-VAE codes)
- Demonstrates latent space prediction is learnable for video

**GAIA-1 (Hu et al. 2023):**
- World model predicting future video latents conditioned on actions
- Uses full transformer decoder, not query tokens
- Validates that action-conditioned future prediction is feasible

### 2.4 Why Query Tokens (Not a Predictor Network)?

| Approach | Parameters | Integration | Flexibility |
|----------|------------|-------------|-------------|
| Predictor Network | 10-100M | Requires training | Fixed architecture |
| Query Tokens | ~1M | Plug into frozen VLM | Adjustable count |

Query tokens are preferable because:
1. **Minimal intervention:** We want to test what the VLM *already knows*, not train a new predictor
2. **Leverage VLM's attention:** Queries attend to VLM's full context, leveraging its world knowledge
3. **Efficient:** ~1M parameters vs 10-100M for predictor networks
4. **Interpretable:** We can analyze which input regions queries attend to

---

## 3. Experimental Setup

### 3.1 Hardware Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| GPU | 1x A100 40GB | 1x A100 80GB | For Qwen2.5-VL-7B + training |
| GPU Memory | 24GB | 40GB | 16-bit inference + gradients |
| CPU RAM | 64GB | 128GB | Dataset loading |
| Storage | 500GB | 1TB | SSv2 dataset + checkpoints |
| Training Time | ~2 days/experiment | ~1 day/experiment | With recommended hardware |

**Compute Budget Estimate:**
- Total GPU hours: 200-400 hours
- Cloud cost (A100): ~$400-800

### 3.2 Dataset: Something-Something v2

**Why SSv2:**
- 220,847 videos of object manipulation
- 174 action templates (e.g., "Pushing [something] from left to right")
- Actions have clear visual consequences
- Standard benchmark for temporal reasoning

**Dataset Structure:**
```
something-something-v2/
├── videos/           # 220K MP4 files (30GB compressed)
├── labels/
│   ├── train.json    # 168,913 training videos
│   ├── validation.json # 24,777 validation videos
│   └── test.json     # 27,157 test videos (no labels)
└── something-something-v2-labels.json  # Action templates
```

**Preprocessing Pipeline:**
```python
# Frame extraction settings
FPS = 12                    # Native is ~12fps
CONTEXT_FRAMES = 16         # Input to VLM (frames 1-16)
FUTURE_FRAMES = 8           # Prediction target (frames 17-24)
FRAME_SIZE = (224, 224)     # After resize

# Action label processing
# "Pushing [something] from left to right" -> action embedding
```

**Data Split for Experiments:**
| Split | Videos | Use |
|-------|--------|-----|
| Train | 168,913 | Query token training |
| Val | 24,777 | Hyperparameter tuning |
| Test | 5,000 (subset) | Final evaluation |

### 3.3 Model Configuration

**VLM: Qwen2.5-VL-7B-Instruct**
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
model.eval()  # Frozen during query training
for param in model.parameters():
    param.requires_grad = False
```

**Query Token Architecture:**
```python
class FuturePredictionQueries(nn.Module):
    def __init__(
        self,
        num_queries: int = 32,          # Number of query tokens
        hidden_dim: int = 3584,          # Qwen2.5-VL-7B hidden size
        num_output_tokens: int = 64,     # Output latent size
    ):
        super().__init__()
        # Learnable query embeddings
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # Optional: projection to output space
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vlm_hidden_states, attention_mask=None):
        """
        Args:
            vlm_hidden_states: [batch, seq_len, hidden_dim] from VLM
        Returns:
            predicted_future: [batch, num_queries, hidden_dim]
        """
        # Query tokens attend to VLM hidden states via cross-attention
        # (Implementation depends on integration method - see Section 3.4)
        pass
```

**Parameter Count:**
- 32 queries x 3584 dim = 114,688 parameters
- 64 queries x 3584 dim = 229,376 parameters
- With projection: +12.8M parameters
- **Total: ~1-15M trainable parameters**

### 3.4 Query Token Integration Methods

We will test three integration strategies:

**Method A: Prefix Injection**
```
Input sequence: [QUERY_TOKENS] [VIDEO_TOKENS] [ACTION_TEXT]
                 ^^^^^^^^^^^^
                 32-64 learnable tokens prepended to sequence
```
- Simplest approach
- Queries attend to video/text via self-attention
- Output: Take query token hidden states from final layer

**Method B: Cross-Attention Module**
```
Input sequence: [VIDEO_TOKENS] [ACTION_TEXT]
                      |
                      v (cross-attention)
              [QUERY_TOKENS] -> predicted_future
```
- Add cross-attention layers after VLM
- More flexible but more parameters
- Similar to Q-Former architecture

**Method C: Suffix Queries (Decoder-Style)**
```
Input sequence: [VIDEO_TOKENS] [ACTION_TEXT] [SEP] [QUERY_TOKENS]
                                                    ^^^^^^^^^^^^
                                                    Queries at end, causal attention
```
- Queries can attend to all context
- Natural for autoregressive VLMs
- Output: Query token hidden states

**Default Choice:** Method A (Prefix Injection) for simplicity. Method B for ablation.

### 3.5 Training Procedure

**Objective Function:**
```python
def future_prediction_loss(predicted_latent, actual_future_latent):
    """
    Args:
        predicted_latent: [batch, num_queries, hidden_dim] - from query tokens
        actual_future_latent: [batch, num_future_tokens, hidden_dim] - VLM encoding of future frames
    """
    # Option 1: Cosine similarity (primary)
    pred_pooled = predicted_latent.mean(dim=1)  # [batch, hidden_dim]
    future_pooled = actual_future_latent.mean(dim=1)
    cos_sim = F.cosine_similarity(pred_pooled, future_pooled, dim=-1)
    loss_cos = 1 - cos_sim.mean()

    # Option 2: MSE in normalized space
    pred_norm = F.normalize(pred_pooled, dim=-1)
    future_norm = F.normalize(future_pooled, dim=-1)
    loss_mse = F.mse_loss(pred_norm, future_norm)

    # Option 3: Contrastive loss (prediction should be closer to correct future than random)
    # Implemented in E3.5

    return loss_cos + 0.1 * loss_mse
```

**Training Configuration:**
```yaml
# configs/c3_training.yaml
training:
  optimizer: AdamW
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 500
  max_steps: 50000
  batch_size: 8  # Per GPU
  gradient_accumulation: 4

  # Learning rate schedule
  scheduler: cosine
  min_lr: 1e-6

data:
  dataset: something-something-v2
  context_frames: 16
  future_frames: 8
  frame_size: 224

model:
  num_queries: 32
  query_dim: 3584
  integration: prefix  # or cross_attention, suffix

evaluation:
  eval_every: 1000
  metrics: [cosine_similarity, retrieval_accuracy, action_classification]
```

**Training Loop Pseudocode:**
```python
for batch in dataloader:
    # 1. Encode context frames with VLM
    context_inputs = processor(videos=batch["context_frames"], text=batch["action"])
    with torch.no_grad():
        context_outputs = vlm(**context_inputs, output_hidden_states=True)
    context_hidden = context_outputs.hidden_states[-1]  # [batch, seq, hidden]

    # 2. Get predicted future from query tokens
    predicted_future = query_model(context_hidden)  # [batch, num_queries, hidden]

    # 3. Encode actual future frames with VLM (ground truth)
    future_inputs = processor(videos=batch["future_frames"])
    with torch.no_grad():
        future_outputs = vlm(**future_inputs, output_hidden_states=True)
    actual_future = future_outputs.hidden_states[-1]

    # 4. Compute loss and update
    loss = future_prediction_loss(predicted_future, actual_future)
    loss.backward()
    optimizer.step()
```

---

## 4. Experiments

### E3.1: Sanity Check - Reconstruct Current Frame

**Objective:** Verify that query tokens can learn *anything* from VLM hidden states by training them to reconstruct the current (last context) frame's latent.

**Setup:**
- Input: Video frames 1-16
- Target: VLM encoding of frame 16 (same as last input frame)
- This should be "easy" - the information is directly in the input

**Protocol:**
```python
# Target is the last context frame's encoding, not future
target_latent = context_hidden[:, -num_visual_tokens:, :]  # Last frame's tokens
predicted = query_model(context_hidden)
loss = cosine_distance(predicted.mean(1), target_latent.mean(1))
```

**Success Criteria:**
- Cosine similarity > 0.95 between predicted and actual
- Training converges within 5,000 steps
- Loss decreases monotonically

**Failure Indicators:**
- Cosine similarity < 0.8 after convergence
- Loss oscillates or diverges
- Queries collapse to identical outputs

**Duration:** 0.5 days

---

### E3.2: Single Frame Future Prediction

**Objective:** Train query tokens to predict the latent representation of the *next* frame (frame 17 given frames 1-16).

**Setup:**
- Input: Video frames 1-16 (no action text initially)
- Target: VLM encoding of frame 17
- Baseline: Using frame 16's encoding as prediction (copy baseline)

**Protocol:**
```python
# Phase 1: No action conditioning
context = frames[0:16]
target = vlm_encode(frames[16:17])
predicted = query_model(vlm_encode(context))

# Evaluate
cos_sim_predicted = cosine_similarity(predicted, target)
cos_sim_copy = cosine_similarity(vlm_encode(frames[15:16]), target)

improvement = cos_sim_predicted - cos_sim_copy
```

**Metrics:**
| Metric | Computation | Target |
|--------|-------------|--------|
| Cosine Similarity | cos(predicted, actual) | > 0.7 |
| Copy Baseline Delta | cos(predicted, actual) - cos(copy, actual) | > 0.05 |
| Random Baseline Delta | cos(predicted, actual) - cos(random, actual) | > 0.3 |

**Success Criteria:**
- Predicted future is closer to actual future than copy baseline
- Improvement is statistically significant (p < 0.01, paired t-test)
- Cosine similarity > 0.7 on validation set

**Failure Criteria:**
- Predicted latents indistinguishable from copy baseline
- High variance across samples (model memorizing, not generalizing)

**Duration:** 1 day

---

### E3.3: Action-Conditioned Prediction

**Objective:** Test whether action descriptions improve prediction accuracy. The same initial video can lead to different futures depending on the action.

**Setup:**
- Input: Video frames 1-16 + action text (e.g., "Pushing [cup] from left to right")
- Target: VLM encoding of frames 17-24
- Control: Prediction without action text

**Protocol:**
```python
# With action conditioning
context_with_action = vlm_encode(frames[0:16], text=action_description)
predicted_with_action = query_model(context_with_action)

# Without action conditioning
context_no_action = vlm_encode(frames[0:16], text="")
predicted_no_action = query_model(context_no_action)

# Compare
actual_future = vlm_encode(frames[16:24])
gain = cos_sim(predicted_with_action, actual_future) - cos_sim(predicted_no_action, actual_future)
```

**Action Template Processing:**
```python
# SSv2 action templates
# "Pushing [something] from left to right"
# "Picking [something] up"
# "Putting [something] onto [something]"

def process_action(template, objects):
    """Convert template + detected objects to natural language"""
    return template.replace("[something]", objects[0])
```

**Metrics:**
| Metric | Computation | Target |
|--------|-------------|--------|
| Action Gain | cos_sim(with_action) - cos_sim(no_action) | > 0.05 |
| Action Specificity | cos_sim(correct_action) - cos_sim(wrong_action) | > 0.1 |

**Ablation: Action Representations**
- Template only: "Pushing from left to right"
- Template + object: "Pushing cup from left to right"
- Full description: "A hand pushes a red cup from the left side to the right side of the table"

**Success Criteria:**
- Action conditioning improves prediction (p < 0.01)
- Wrong action conditioning hurts prediction
- Model distinguishes between opposite actions (push left vs push right)

**Failure Criteria:**
- No significant difference with/without action
- Model ignores action, predicts "average" future

**Duration:** 1.5 days

---

### E3.4: Multi-Frame Future Prediction

**Objective:** Evaluate prediction quality at different time horizons. How far ahead can the model "see"?

**Setup:**
- Input: Video frames 1-16 + action
- Targets: Frame 17 (next), Frame 21 (5 ahead), Frame 26 (10 ahead)
- Separate query sets or shared queries with temporal embedding

**Protocol:**
```python
# Option A: Separate query sets per horizon
queries_1 = FuturePredictionQueries(num_queries=32, horizon="next_1")
queries_5 = FuturePredictionQueries(num_queries=32, horizon="next_5")
queries_10 = FuturePredictionQueries(num_queries=32, horizon="next_10")

# Option B: Shared queries with temporal conditioning
queries = FuturePredictionQueries(num_queries=32)
horizon_embedding = nn.Embedding(10, hidden_dim)  # 1-10 frames ahead

predicted = queries(context, horizon_embedding[t])
```

**Evaluation Grid:**
| Horizon | Frames Ahead | Target Frame | Expected Difficulty |
|---------|--------------|--------------|---------------------|
| t+1 | 1 frame | 17 | Easy |
| t+3 | 3 frames | 19 | Medium |
| t+5 | 5 frames | 21 | Medium |
| t+8 | 8 frames | 24 | Hard |
| t+10 | 10 frames | 26 | Very Hard |

**Metrics:**
| Metric | Description |
|--------|-------------|
| Cosine @ t+k | Similarity at each horizon |
| Decay Rate | How fast accuracy drops with horizon |
| Horizon Threshold | Maximum t where cos_sim > 0.6 |

**Success Criteria:**
- Monotonic (or near-monotonic) decay with horizon
- cos_sim > 0.6 for at least t+5
- Action conditioning helps more at longer horizons

**Failure Criteria:**
- Flat accuracy across horizons (not using temporal information)
- Accuracy at t+1 same as t+10 (predicting "average" future)

**Duration:** 2 days

---

### E3.5: Action Discrimination Test

**Objective:** Test whether predicted latents encode action-specific information. Given a prediction, can we recover which action was performed?

**Setup:**
- Train linear classifier on predicted latents to classify actions
- Compare to classifier trained on actual future latents (upper bound)

**Protocol:**
```python
# Generate predictions for all validation videos
predictions = []
actions = []
for video, action_label in val_loader:
    context = vlm_encode(video[:16])
    pred = query_model(context)
    predictions.append(pred.mean(1))  # Pool to single vector
    actions.append(action_label)

# Train action classifier
classifier = nn.Linear(hidden_dim, num_actions)  # 174 SSv2 actions
classifier_loss = cross_entropy(classifier(predictions), actions)

# Evaluate
top1_accuracy = (classifier(predictions).argmax(1) == actions).mean()
top5_accuracy = ...
```

**Comparison Conditions:**
| Condition | Input to Classifier | Expected Accuracy |
|-----------|--------------------|--------------------|
| Random | Random vectors | ~0.6% (1/174) |
| Current Frame | Encoding of frame 16 | ~30% (some actions visible) |
| Predicted Future | Query token output | Target: >40% |
| Actual Future | Encoding of frames 17-24 | Upper bound: ~60% |

**Metrics:**
| Metric | Description | Target |
|--------|-------------|--------|
| Top-1 Accuracy | Correct action predicted | > 40% |
| Top-5 Accuracy | Correct in top 5 | > 70% |
| Confusion Quality | Similar actions confused, not random | Qualitative |

**Analysis:**
- Confusion matrix visualization
- Which action pairs are most confused?
- Do predictions capture action direction (left vs right)?

**Success Criteria:**
- Predicted latents enable action classification significantly above chance
- Accuracy approaches actual future baseline
- Semantically similar actions are confused (push left/right) not random

**Failure Criteria:**
- Accuracy near random chance
- All predictions collapse to single "average" representation

**Duration:** 1 day

---

### E3.6: Contrastive Evaluation (Retrieval Task)

**Objective:** Given a predicted future latent, can we retrieve the correct actual future from a set of candidates?

**Setup:**
- For each video, predict future latent
- Retrieve nearest neighbor from pool of actual future latents
- Measure retrieval accuracy

**Protocol:**
```python
# Build retrieval pool
actual_futures = {}  # video_id -> latent
for video_id, frames in val_videos:
    actual_futures[video_id] = vlm_encode(frames[16:24]).mean(1)

# Retrieval evaluation
retrieval_results = []
for video_id, frames, action in val_loader:
    predicted = query_model(vlm_encode(frames[:16], action))
    predicted_pooled = predicted.mean(1)

    # Find nearest neighbor
    similarities = {
        vid: cosine_similarity(predicted_pooled, actual_futures[vid])
        for vid in actual_futures
    }
    top_k = sorted(similarities.items(), key=lambda x: -x[1])[:10]

    # Check if correct video in top-k
    correct_rank = [i for i, (vid, _) in enumerate(top_k) if vid == video_id]
    retrieval_results.append({
        "correct_in_top1": video_id == top_k[0][0],
        "correct_in_top5": video_id in [v for v, _ in top_k[:5]],
        "correct_in_top10": video_id in [v for v, _ in top_k[:10]],
    })
```

**Pool Sizes:**
| Pool Size | Random Baseline | Target |
|-----------|-----------------|--------|
| 100 | 1% | > 20% |
| 1,000 | 0.1% | > 10% |
| 10,000 | 0.01% | > 5% |

**Metrics:**
| Metric | Description |
|--------|-------------|
| R@1 | Recall at rank 1 |
| R@5 | Recall at rank 5 |
| R@10 | Recall at rank 10 |
| MRR | Mean Reciprocal Rank |

**Success Criteria:**
- R@1 > 10% (100x better than random)
- R@10 > 30%
- MRR > 0.2

**Failure Criteria:**
- Performance near random
- Retrieved videos are not semantically similar to query

**Duration:** 0.5 days

---

## 5. Success Metrics Summary

### Primary Metrics

| Metric | Threshold | Measured In |
|--------|-----------|-------------|
| **Cosine Similarity (t+1)** | > 0.75 | E3.2 |
| **Cosine Similarity (t+5)** | > 0.65 | E3.4 |
| **Action Conditioning Gain** | > 0.05 | E3.3 |
| **Action Classification Top-1** | > 40% | E3.5 |
| **Retrieval R@1 (1K pool)** | > 10% | E3.6 |

### Secondary Metrics

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Copy Baseline Delta | > 0.05 | Not just copying |
| Wrong Action Delta | < -0.05 | Action matters |
| Horizon Decay Rate | Monotonic | Temporal understanding |
| Training Convergence | < 20K steps | Efficiency |

### Statistical Significance

All comparisons require:
- Paired t-test or Wilcoxon signed-rank test
- p < 0.01 for primary claims
- p < 0.05 for secondary claims
- 95% confidence intervals reported

---

## 6. Failure Criteria and Diagnostics

### Failure Mode 1: Copy Collapse

**Symptom:** Predicted latents are identical to current frame encoding
**Diagnostic:** cos_sim(predicted, current) >> cos_sim(predicted, future)
**Cause:** Queries learn to "pass through" current information
**Mitigation:**
- Add explicit regularization: `loss -= 0.1 * cos_sim(predicted, current)`
- Use contrastive loss that pushes away from current

### Failure Mode 2: Mean Collapse

**Symptom:** All predictions converge to dataset mean
**Diagnostic:** Low variance in predicted latents; all predictions highly similar
**Cause:** Model hedges by predicting "average" future
**Mitigation:**
- Action-conditional training (E3.3)
- Contrastive loss with hard negatives
- Increase query count

### Failure Mode 3: Memorization

**Symptom:** High training accuracy, low validation accuracy
**Diagnostic:** Train/val gap > 0.2 in cosine similarity
**Cause:** Queries memorize training videos
**Mitigation:**
- Increase regularization (dropout, weight decay)
- Reduce query count
- Data augmentation

### Failure Mode 4: No Action Sensitivity

**Symptom:** Predictions identical regardless of action text
**Diagnostic:** cos_sim(pred|action_A, pred|action_B) > 0.95
**Cause:** Queries ignore text conditioning
**Mitigation:**
- Verify action text is properly tokenized
- Add action-specific loss term
- Try different integration method (cross-attention)

### When to Declare "No Better Than Random"

Claim 3 is falsified if ALL of the following hold after hyperparameter search:
1. cos_sim(predicted, actual_future) - cos_sim(random, actual_future) < 0.1
2. Action classification accuracy < 5% (vs 0.6% random)
3. Retrieval R@10 < 2% (vs 1% random for pool of 100)
4. These results hold across 3+ random seeds

---

## 7. Pivot Options

If initial experiments fail, try these alternatives in order:

### Pivot 1: Different Query Integration

**If prefix queries fail:**
- Try cross-attention queries (more expressive)
- Try suffix queries (decoder-style attention)
- Try combination (prefix + cross-attention)

**Implementation:**
```python
class CrossAttentionQueries(nn.Module):
    def __init__(self, num_queries, hidden_dim, num_layers=2):
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8)
            for _ in range(num_layers)
        ])

    def forward(self, context_hidden):
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.cross_attn_layers:
            queries = layer(queries, context_hidden, context_hidden)
        return queries
```

### Pivot 2: Different Training Objective

**If cosine similarity doesn't work:**
- Contrastive learning (InfoNCE loss)
- Reconstruction through decoder
- Adversarial training (discriminator on predicted vs actual)

**Contrastive Loss:**
```python
def contrastive_loss(predicted, actual_future, temperature=0.1):
    """
    InfoNCE: predicted should be similar to actual_future,
    dissimilar to other futures in batch
    """
    pred = F.normalize(predicted, dim=-1)
    actual = F.normalize(actual_future, dim=-1)

    # Positive: predicted vs its actual future
    pos_sim = (pred * actual).sum(-1) / temperature

    # Negatives: predicted vs other futures in batch
    neg_sim = torch.mm(pred, actual.T) / temperature  # [batch, batch]

    # InfoNCE
    labels = torch.arange(batch_size).to(device)
    loss = F.cross_entropy(neg_sim, labels)
    return loss
```

### Pivot 3: Different Latent Extraction

**If final layer hidden states don't work:**
- Try intermediate layers (layer 16, 24 of 32)
- Try attention-weighted pooling
- Try vision encoder outputs directly (pre-LLM)

**Layer Ablation:**
```python
# Extract from multiple layers
outputs = vlm(**inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states

# Test which layer works best
for layer_idx in [8, 16, 24, 32]:
    latent = hidden_states[layer_idx]
    # Evaluate prediction quality
```

### Pivot 4: Auxiliary Supervision

**If pure latent alignment doesn't work:**
- Add pixel-level supervision (requires C2 decoder)
- Add action classification auxiliary loss
- Add temporal ordering auxiliary loss

### Pivot 5: Simpler Task First

**If SSv2 is too hard:**
- Start with synthetic dataset (moving MNIST)
- Try simpler actions (static camera, single object)
- Reduce prediction horizon to t+1 only

---

## 8. Timeline

### Week 1: Infrastructure and Baselines (5 days)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Set up SSv2 data pipeline | DataLoader, preprocessing |
| 2 | Implement query token module | Working FuturePredictionQueries class |
| 3 | Implement training loop | train.py with logging |
| 4 | E3.1: Sanity check | Baseline reconstruction results |
| 5 | Debug and iterate | Stable training |

### Week 2: Core Experiments (5 days)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | E3.2: Single frame prediction | Results, analysis |
| 3-4 | E3.3: Action conditioning | Results, ablations |
| 5 | E3.4: Multi-horizon (start) | Initial results |

### Week 3: Advanced Experiments (5 days)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | E3.4: Multi-horizon (complete) | Full horizon analysis |
| 2 | E3.5: Action discrimination | Classification results |
| 3 | E3.6: Retrieval evaluation | Retrieval metrics |
| 4 | Analysis and visualization | Figures, confusion matrices |
| 5 | Buffer / catch-up | Address issues |

### Week 4: Pivots and Documentation (5 days)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-3 | Pivot experiments (if needed) | Alternative approaches |
| 4 | Final analysis | Statistical tests, conclusions |
| 5 | Documentation | This document updated, code cleaned |

**Total: 20 working days (4 weeks)**

---

## 9. Dependencies

### Required from Other Claims

| Dependency | Required For | Blocking? |
|------------|--------------|-----------|
| C1: VLM latent quality | Validating that latents are meaningful | No - can proceed, but C1 informs interpretation |
| C2: Adapter trained | Visualizing predictions as pixels | No - C3 is latent-only |

### C3 Provides to Other Claims

| Output | Used By |
|--------|---------|
| Trained query tokens | C4 (full system) |
| Prediction benchmarks | C4 evaluation |
| Failure analysis | Architecture decisions |

### External Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| Qwen2.5-VL-7B-Instruct | HuggingFace | Available |
| Something-Something v2 | Official download | Requires registration |
| A100 GPU | Cloud provider | Budget approved |

---

## 10. Deliverables

### Code Artifacts

```
src/
├── data/
│   └── ssv2_dataset.py        # SSv2 DataLoader
├── models/
│   └── query_tokens.py        # FuturePredictionQueries
├── training/
│   └── train_c3.py            # Training script
├── evaluation/
│   ├── metrics.py             # Cosine sim, retrieval, classification
│   └── evaluate_c3.py         # Evaluation script
└── configs/
    └── c3_*.yaml              # Experiment configs
```

### Trained Models

| Checkpoint | Description |
|------------|-------------|
| `query_tokens_e3.1_sanity.pt` | Current frame reconstruction |
| `query_tokens_e3.2_t1.pt` | Single frame prediction |
| `query_tokens_e3.3_action.pt` | Action-conditioned |
| `query_tokens_e3.4_multihorizon.pt` | Multi-horizon |
| `query_tokens_final.pt` | Best overall model |

### Results and Analysis

| Document | Contents |
|----------|----------|
| `results/c3_metrics.json` | All quantitative results |
| `results/c3_figures/` | Visualizations, confusion matrices |
| `results/c3_analysis.md` | Interpretation, failure analysis |

### Success Summary Table

To be filled after experiments:

| Experiment | Status | Key Metric | Value | Pass/Fail |
|------------|--------|------------|-------|-----------|
| E3.1 | | cos_sim | | |
| E3.2 | | cos_sim (t+1) | | |
| E3.3 | | action gain | | |
| E3.4 | | cos_sim (t+5) | | |
| E3.5 | | top-1 accuracy | | |
| E3.6 | | R@1 | | |

---

## Appendix A: SSv2 Action Categories

Top 20 most common actions in Something-Something v2:

1. Pushing [something] from left to right
2. Pushing [something] from right to left
3. Picking [something] up
4. Putting [something] down
5. Moving [something] away from the camera
6. Moving [something] towards the camera
7. Dropping [something]
8. Throwing [something]
9. Turning [something] upside down
10. Covering [something] with [something]
...

These action pairs are ideal for testing directional prediction:
- Push left vs Push right
- Move towards vs Move away
- Pick up vs Put down

---

## Appendix B: Qwen2.5-VL Hidden State Details

| Layer | Output Shape | Description |
|-------|--------------|-------------|
| 0 (embed) | [batch, seq, 3584] | Token embeddings |
| 1-32 | [batch, seq, 3584] | Transformer layers |
| 32 (final) | [batch, seq, 3584] | Pre-LM head |

Visual tokens are interleaved with text tokens:
```
[BOS] [IMG_START] [vis_1] [vis_2] ... [vis_N] [IMG_END] [text_tokens] [EOS]
```

To extract visual-only hidden states:
```python
# Find visual token positions
img_start = (input_ids == IMG_START_TOKEN).nonzero()
img_end = (input_ids == IMG_END_TOKEN).nonzero()
visual_hidden = hidden_states[:, img_start:img_end, :]
```

---

## Appendix C: Statistical Analysis Templates

### Paired T-Test for Baseline Comparison

```python
from scipy import stats

# Compare predicted vs copy baseline
predicted_sims = [...]  # N samples
copy_sims = [...]       # N samples

t_stat, p_value = stats.ttest_rel(predicted_sims, copy_sims)
print(f"t={t_stat:.3f}, p={p_value:.4f}")

# Effect size (Cohen's d)
diff = np.array(predicted_sims) - np.array(copy_sims)
cohens_d = diff.mean() / diff.std()
print(f"Cohen's d = {cohens_d:.3f}")
```

### Bootstrap Confidence Intervals

```python
from scipy.stats import bootstrap

# 95% CI for cosine similarity
data = (predicted_sims,)
ci = bootstrap(data, np.mean, confidence_level=0.95, n_resamples=10000)
print(f"95% CI: [{ci.confidence_interval.low:.3f}, {ci.confidence_interval.high:.3f}]")
```

---

## References

1. Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML.
2. Alayrac, J.B., et al. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. NeurIPS.
3. Bardes, A., et al. (2024). V-JEPA: Latent Video Prediction for Visual Representation Learning. ICLR.
4. Goyal, R., et al. (2017). The "Something Something" Video Database for Learning and Evaluating Visual Common Sense. ICCV.
5. Wang, P., et al. (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv.
