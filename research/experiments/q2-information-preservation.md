# Q2: Information Preservation Through VLM

**Status:** Planning
**Risk Level:** MEDIUM
**Dependencies:** None (can run early)
**Estimated Duration:** 5-7 days

---

## 1. Objective

Quantify information loss at each stage of Qwen2.5-VL's visual processing pipeline and determine whether sufficient spatial information is preserved to enable high-quality video generation conditioning.

**Primary Questions:**
1. How much spatial information is lost during the 2x2 token merging step?
2. At which VLM layer does spatial information degrade below useful thresholds?
3. Can we identify extraction points that preserve sufficient detail for video generation?

**Success Definition:** Identify at least one extraction point where spatial reconstruction accuracy (measured by bounding box IoU) remains above 0.7 and fine-grained detail preservation enables LPIPS < 0.3 reconstruction.

---

## 2. Background

### 2.1 Qwen2.5-VL Architecture

The visual processing pipeline in Qwen2.5-VL consists of three stages:

```
Raw Image (H x W x 3)
        |
        v
+-------------------+
| Patch Embedding   |  Divide into 14x14 patches
| (14x14 patches)   |  Each patch -> 1536-dim embedding
+-------------------+
        |
        v (N_patches = (H/14) x (W/14) patches)
+-------------------+
| ViT Encoder       |  675M parameters
| (Pre-merge)       |  Window attention (8x8 max) + 4 full attention layers
|                   |  Outputs: [N_patches, 1536]
+-------------------+
        |
        v
+-------------------+
| 2x2 Token Merger  |  MLP compresses 4 adjacent tokens -> 1 token
| (Post-merge)      |  Outputs: [N_patches/4, 1536]
+-------------------+
        |
        v
+-------------------+
| Qwen2 LLM         |  28 layers (7B model)
| Backbone          |  Hidden dim: 3584
+-------------------+
        |
        v
    Text Output
```

**Key Architectural Parameters:**

| Parameter | Value |
|-----------|-------|
| Patch size | 14 x 14 pixels |
| ViT embedding dim | 1536 |
| ViT layers | ~32 (estimated from 675M params) |
| Merge ratio | 2x2 = 4:1 compression |
| LLM hidden dim | 3584 (7B model) |
| LLM layers | 28 |

**Token Calculation Example:**
```
224x224 image:
- Patches: 16 x 16 = 256 patches (pre-merge)
- After merge: 8 x 8 = 64 tokens (post-merge)
- Compression: 4x spatial reduction

448x448 image:
- Patches: 32 x 32 = 1024 patches (pre-merge)
- After merge: 16 x 16 = 256 tokens (post-merge)
```

### 2.2 Video Generation Requirements

For LTX-Video conditioning, we need to provide information that can reconstruct:

| Requirement | Resolution | Description |
|-------------|------------|-------------|
| Object positions | ~32 pixels | Bounding box accuracy |
| Object boundaries | ~8 pixels | Edge detection |
| Textures | ~4 pixels | Fine-grained patterns |
| Motion vectors | frame-to-frame | Temporal consistency |

**LTX-Video Latent Space:**
- 128 channels
- 32x spatial compression, 8x temporal compression
- Effective resolution: 24x16 latent grid for 768x512 video

**Critical Question:** Does the 2x2 merge (14x14 -> 28x28 effective patch size) lose information needed for 32x pixel positioning?

### 2.3 Prior Work on Information Bottlenecks

**Relevant Findings:**

1. **ViT Probing Studies** (Raghu et al., 2021): Information is increasingly abstracted in deeper layers; spatial relationships preserved better in early layers.

2. **VLM Spatial Reasoning** (various): VLMs often struggle with precise spatial questions ("which object is to the left?"), suggesting spatial information degradation.

3. **Token Merging in ViT** (Bolya et al., 2023 - ToMe): Aggressive token merging can be done without quality loss for classification, but reconstruction tasks suffer.

4. **CLIP Reconstruction** (Patashnik et al.): Reconstructing images from CLIP embeddings possible but loses fine details, suggesting similar challenges for VLMs.

**Key Insight:** Classification-optimized models may discard spatial information unnecessary for their training objective but crucial for generation tasks.

---

## 3. Experimental Setup

### 3.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | RTX 4090 (24GB) | A100 (40GB) |
| VRAM per experiment | 20-24GB | 24-32GB |
| Storage | 50GB | 100GB |
| CPU RAM | 32GB | 64GB |

**Estimated Total GPU Hours:** 20-40 hours

### 3.2 Probing Methodology

We use **linear probes** to measure information content at each extraction point. A linear probe is a single linear layer trained to predict target information from frozen representations.

**Why Linear Probes:**
- If information exists in the representation, a linear probe can extract it
- Non-linear probes might "create" information, giving false positives
- Linear probe accuracy directly measures information content

**Probe Architecture:**
```python
class SpatialProbe(nn.Module):
    def __init__(self, input_dim, num_boxes=10):
        super().__init__()
        # Predict bounding boxes: [x1, y1, x2, y2] for num_boxes objects
        self.fc = nn.Linear(input_dim, num_boxes * 4)

    def forward(self, features):
        # features: [batch, seq_len, dim] -> pool -> [batch, dim]
        pooled = features.mean(dim=1)
        return self.fc(pooled).view(-1, num_boxes, 4)
```

### 3.3 Test Dataset

**Primary Dataset:** COCO 2017 val (5000 images)
- Rich annotations: bounding boxes, segmentation masks, keypoints
- Diverse scenes and object scales
- Standard benchmark for spatial understanding

**Supplementary Datasets:**
- **ADE20K:** Dense semantic segmentation for fine-grained analysis
- **Something-Something v2 frames:** Task-relevant video frames

**Test Image Categories:**

| Category | Purpose | Examples |
|----------|---------|----------|
| Simple geometry | Baseline spatial | Shapes on solid backgrounds |
| Multi-object scenes | Relationship preservation | COCO multi-object images |
| Fine texture | Detail preservation | Fabric, grass, brick walls |
| Edges/boundaries | Boundary preservation | High-contrast edges |
| Video frames | Temporal relevance | Something-Something frames |

### 3.4 Extraction Points

We extract representations from 8 distinct points:

| ID | Extraction Point | Shape | Description |
|----|------------------|-------|-------------|
| E1 | ViT patch embeddings | [N_patches, 1536] | Raw patch features (pre-ViT) |
| E2 | ViT layer 8 | [N_patches, 1536] | Early ViT |
| E3 | ViT layer 16 | [N_patches, 1536] | Mid ViT |
| E4 | ViT final (pre-merge) | [N_patches, 1536] | Full ViT, before compression |
| E5 | Post-merge (ViT output) | [N_tokens, 1536] | After 2x2 merge |
| E6 | LLM layer 1 | [seq_len, 3584] | First LLM layer |
| E7 | LLM layer 14 | [seq_len, 3584] | Mid LLM |
| E8 | LLM layer 28 | [seq_len, 3584] | Final LLM |

**Extraction Code:**
```python
def extract_representations(model, image, extraction_points):
    """Extract representations from multiple points in Qwen2.5-VL."""

    representations = {}

    # Process image through vision encoder
    inputs = processor.image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    grid_thw = inputs["image_grid_thw"].to(device)

    # Hook into ViT layers
    vit_hooks = []
    def make_vit_hook(name):
        def hook(module, input, output):
            representations[name] = output.detach()
        return hook

    # Register hooks on ViT layers
    for i, layer in enumerate(model.visual.encoder.layers):
        if i in [7, 15]:  # Layer 8 and 16 (0-indexed)
            vit_hooks.append(layer.register_forward_hook(make_vit_hook(f'vit_layer_{i+1}')))

    # Get pre-merge output
    with torch.no_grad():
        # This requires modifying forward pass to expose pre-merge features
        vit_output = model.visual.encoder(pixel_values, grid_thw)
        representations['vit_pre_merge'] = vit_output

        # Post-merge (standard ViT output)
        visual_embeds = model.visual(pixel_values, grid_thw)
        representations['vit_post_merge'] = visual_embeds

    # Remove hooks
    for hook in vit_hooks:
        hook.remove()

    # Get LLM hidden states
    # ... (similar hook pattern for LLM layers)

    return representations
```

---

## 4. Experiments

### E-Q2.1: Pre-Merge ViT Embedding Analysis

**Objective:** Establish baseline spatial information content before any compression.

**Method:**
1. Extract patch embeddings from ViT final layer (pre-merge)
2. Train linear probes for:
   - Bounding box prediction (4 coords per object)
   - Patch-level object classification
   - Relative position prediction (object A left/right/above/below object B)

**Implementation:**
```python
# Extract pre-merge embeddings
with torch.no_grad():
    # Access ViT encoder output before merger
    vit_features = model.visual.encoder(pixel_values, grid_thw)
    # Shape: [batch, num_patches, 1536]

# Train bounding box probe
bbox_probe = nn.Linear(1536, num_objects * 4)
optimizer = torch.optim.Adam(bbox_probe.parameters(), lr=1e-3)

for epoch in range(50):
    for batch in dataloader:
        features = extract_vit_premerge(batch['images'])
        pooled = features.mean(dim=1)  # Global average pooling
        pred_boxes = bbox_probe(pooled).view(-1, num_objects, 4)
        loss = F.l1_loss(pred_boxes, batch['boxes'])
        loss.backward()
        optimizer.step()
```

**Metrics:**
- Bounding box IoU (target: >0.8)
- Relative position accuracy (target: >90%)
- Per-patch classification accuracy

**Expected Outcome:** High spatial accuracy (>0.85 IoU), establishing the upper bound.

**Duration:** 4-6 hours

---

### E-Q2.2: Post-Merge Embedding Analysis

**Objective:** Quantify information loss from 2x2 token merging.

**Method:**
1. Extract post-merge embeddings (standard ViT output)
2. Train identical probes as E-Q2.1
3. Compare accuracy drop

**Implementation:**
```python
# Extract post-merge embeddings (standard path)
with torch.no_grad():
    visual_embeds = model.visual(pixel_values, grid_thw)
    # Shape: [batch, num_tokens, 1536] where num_tokens = num_patches / 4

# Train same probes on compressed representations
# ... (same training loop as E-Q2.1)
```

**Key Comparisons:**

| Metric | Pre-merge (E-Q2.1) | Post-merge (E-Q2.2) | Delta |
|--------|-------------------|---------------------|-------|
| Bbox IoU | ? | ? | ? |
| Position accuracy | ? | ? | ? |
| Classification | ? | ? | ? |

**Expected Outcome:** 10-30% accuracy drop, depending on task granularity.

**Duration:** 4-6 hours

---

### E-Q2.3: LLM Layer-wise Analysis

**Objective:** Track spatial information degradation through LLM layers.

**Method:**
1. Extract hidden states from LLM layers 1, 7, 14, 21, 28
2. Train spatial probes at each layer
3. Plot information decay curve

**Implementation:**
```python
# Get hidden states from all layers
outputs = model(
    **inputs,
    output_hidden_states=True,
    return_dict=True
)

# hidden_states is tuple of (batch, seq_len, hidden_dim)
# Index 0 is embedding layer, 1-28 are transformer layers
layer_indices = [1, 7, 14, 21, 28]

for layer_idx in layer_indices:
    hidden = outputs.hidden_states[layer_idx]

    # Extract visual token positions (between <vision_start> and <vision_end>)
    visual_hidden = hidden[:, vision_start:vision_end, :]

    # Train probe on this layer's representations
    probe = train_spatial_probe(visual_hidden, targets)
    results[f'llm_layer_{layer_idx}'] = evaluate_probe(probe, test_set)
```

**Expected Decay Pattern:**
```
Layer 1:  [|||||||||||||||||||] 95% spatial info
Layer 7:  [|||||||||||||||||  ] 85% spatial info
Layer 14: [||||||||||||||     ] 70% spatial info
Layer 21: [|||||||||||        ] 55% spatial info
Layer 28: [||||||||           ] 40% spatial info
```

**Key Question:** At which layer does spatial accuracy drop below 0.7 IoU?

**Duration:** 8-12 hours

---

### E-Q2.4: Spatial Reconstruction Probe (Bounding Box)

**Objective:** Measure precise object localization capability at each extraction point.

**Method:**
1. Use COCO validation set with ground truth boxes
2. Train detection head on frozen features
3. Evaluate mAP at different extraction points

**Probe Architecture:**
```python
class DetectionProbe(nn.Module):
    """DETR-style detection probe for spatial accuracy."""

    def __init__(self, input_dim, num_queries=100):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, 256))
        self.cross_attn = nn.MultiheadAttention(256, 8)
        self.proj_in = nn.Linear(input_dim, 256)
        self.box_head = nn.Linear(256, 4)  # [x_center, y_center, w, h]
        self.class_head = nn.Linear(256, 91)  # COCO classes

    def forward(self, features):
        # features: [batch, seq_len, input_dim]
        feat_proj = self.proj_in(features)  # [batch, seq_len, 256]

        queries = self.queries.unsqueeze(0).expand(features.size(0), -1, -1)
        queries = queries.permute(1, 0, 2)  # [num_queries, batch, 256]
        feat_proj = feat_proj.permute(1, 0, 2)  # [seq_len, batch, 256]

        attended, _ = self.cross_attn(queries, feat_proj, feat_proj)
        attended = attended.permute(1, 0, 2)  # [batch, num_queries, 256]

        boxes = self.box_head(attended).sigmoid()
        classes = self.class_head(attended)

        return boxes, classes
```

**Metrics:**
- mAP@0.5 (standard COCO metric)
- mAP@0.75 (stricter localization)
- mAP@[0.5:0.95] (averaged across thresholds)
- Small/medium/large object breakdown

**Results Table Template:**

| Extraction Point | mAP@0.5 | mAP@0.75 | mAP (avg) | Small | Medium | Large |
|------------------|---------|----------|-----------|-------|--------|-------|
| Pre-merge ViT | | | | | | |
| Post-merge ViT | | | | | | |
| LLM Layer 1 | | | | | | |
| LLM Layer 14 | | | | | | |
| LLM Layer 28 | | | | | | |

**Duration:** 12-16 hours

---

### E-Q2.5: Fine-Grained Detail Probe (Texture/Edges)

**Objective:** Measure preservation of high-frequency visual information.

**Method:**
1. Train decoder to reconstruct image from features
2. Measure edge preservation (Canny edge F1)
3. Measure texture similarity (Gram matrix correlation)

**Decoder Architecture:**
```python
class ReconstructionDecoder(nn.Module):
    """Decode features back to image for quality assessment."""

    def __init__(self, input_dim, output_size=224):
        super().__init__()
        self.output_size = output_size

        # Project to spatial format
        self.proj = nn.Linear(input_dim, 512 * 7 * 7)

        # Upsample to image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 224x224
            nn.Sigmoid()
        )

    def forward(self, features):
        # features: [batch, seq_len, dim] -> pool -> [batch, dim]
        pooled = features.mean(dim=1)
        spatial = self.proj(pooled).view(-1, 512, 7, 7)
        return self.decoder(spatial)
```

**Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| LPIPS | Perceptual similarity | < 0.3 |
| SSIM | Structural similarity | > 0.7 |
| Edge F1 | Canny edge overlap | > 0.6 |
| Gram Loss | Texture correlation | < 0.1 |

**Analysis:**
- Plot LPIPS vs extraction point
- Visualize edge maps: original vs reconstructed
- Identify which frequencies are lost at each stage

**Duration:** 8-12 hours

---

### E-Q2.6: Temporal Information Probe (Video Inputs)

**Objective:** Assess whether temporal/motion information is preserved through the pipeline.

**Method:**
1. Use Something-Something v2 video clips
2. Extract features from video input (multiple frames)
3. Probe for: action direction, speed, temporal ordering

**Test Cases:**

| Test | Input | Target | Description |
|------|-------|--------|-------------|
| Action direction | "pushing left" vs "pushing right" | Binary | Motion direction preserved? |
| Speed estimation | Fast vs slow actions | Regression | Relative speed? |
| Temporal order | Frames 1,2,3 vs 3,2,1 | Binary | Sequence ordering? |
| Frame prediction | Frames 1-4 | Frame 5 latent | Temporal extrapolation? |

**Implementation:**
```python
# Process video through Qwen2.5-VL
video_message = {
    "role": "user",
    "content": [
        {"type": "video", "video": video_path, "max_pixels": 360*420, "fps": 1.0},
        {"type": "text", "text": "Describe the action."}
    ]
}

# Extract video features from different points
# M-RoPE encodes temporal position - does this survive to late layers?

# Probe for temporal relationships
temporal_probe = nn.Linear(hidden_dim, num_temporal_classes)
```

**Metrics:**
- Action classification accuracy (baseline: text-only VLM)
- Temporal ordering accuracy
- Motion direction accuracy
- Frame interpolation quality (if applicable)

**Duration:** 8-12 hours

---

## 5. Success Metrics

### 5.1 Primary Metrics

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| **Bbox IoU** | > 0.7 | Objects can be located accurately |
| **mAP@0.5** | > 0.4 | Detection-quality localization |
| **LPIPS** | < 0.3 | Reconstructions are perceptually similar |
| **Edge F1** | > 0.6 | Boundaries are preserved |

### 5.2 Information Retention Score

We define an **Information Retention Score (IRS)** for each extraction point:

```
IRS = 0.4 * (Bbox_IoU / 0.85) + 0.3 * (LPIPS_inv / 0.7) + 0.3 * (Edge_F1 / 0.8)

where LPIPS_inv = 1 - LPIPS (higher is better)
```

**IRS Interpretation:**
- IRS > 0.8: Excellent - use this extraction point
- IRS 0.6-0.8: Good - usable with quality tradeoffs
- IRS 0.4-0.6: Marginal - may need architectural changes
- IRS < 0.4: Poor - insufficient for video generation

### 5.3 Layer-wise Information Curve

Expected output visualization:

```
Information Retention by Layer
1.0 |*****
    |     ****
0.8 |         ***
    |            **      <- Target threshold (0.7)
0.6 |              **
    |                *
0.4 |                 **
    |                   **
0.2 |                     ***
    +------------------------
      Pre  Post  L1  L7  L14  L21  L28
      merge merge
```

---

## 6. Failure Criteria

### 6.1 Hard Failures (Require Pivot)

| Condition | Implication |
|-----------|-------------|
| Pre-merge IoU < 0.7 | ViT itself loses spatial info; fundamental issue |
| Post-merge IoU < 0.5 | Merger destroys too much; must use pre-merge |
| All LLM layers IoU < 0.4 | LLM incompatible with spatial tasks |
| LPIPS > 0.5 at all points | Features fundamentally unsuitable for generation |

### 6.2 Soft Failures (Require Adaptation)

| Condition | Adaptation |
|-----------|------------|
| Post-merge IoU 0.5-0.7 | May still work with quality tradeoff |
| LLM drops below 0.5 by layer 14 | Extract from earlier layer |
| Edge F1 < 0.5 everywhere | May need auxiliary edge features |

### 6.3 Decision Matrix

```
                          Post-merge IoU
                    |  > 0.7   |  0.5-0.7  |  < 0.5
--------------------|----------|-----------|----------
Pre-merge > 0.8    |  GREEN   |  YELLOW   |   RED
Pre-merge 0.7-0.8  |  YELLOW  |  YELLOW   |   RED
Pre-merge < 0.7    |   RED    |   RED     |   RED
```

- **GREEN:** Proceed with post-merge extraction
- **YELLOW:** Investigate pre-merge or hybrid approach
- **RED:** Fundamental architectural concern; may need different VLM

---

## 7. Pivot Options

### 7.1 If Post-Merge Loses Too Much Information

**Option A: Use Pre-Merge Embeddings**
- Extract from ViT final layer before 2x2 merger
- 4x more tokens, 4x more compute for adapter
- Higher memory requirements but preserves spatial detail

```python
# Modified extraction to get pre-merge features
# Requires patching Qwen2VL visual encoder
class PatchedVisualEncoder(nn.Module):
    def forward_premerge(self, x, grid_thw):
        # Run ViT without final merge layer
        features = self.encoder(x, grid_thw)
        return features  # [batch, num_patches, 1536]
```

**Option B: Hybrid Extraction**
- Pre-merge for spatial tokens
- Post-merge for semantic context
- Combine in adapter

```python
class HybridAdapter(nn.Module):
    def __init__(self):
        self.spatial_proj = nn.Linear(1536, 512)  # Pre-merge
        self.semantic_proj = nn.Linear(1536, 512)  # Post-merge
        self.fusion = nn.Linear(1024, 128 * latent_h * latent_w)

    def forward(self, pre_merge, post_merge):
        spatial = self.spatial_proj(pre_merge)
        semantic = self.semantic_proj(post_merge.unsqueeze(2).expand_as(pre_merge))
        fused = torch.cat([spatial, semantic], dim=-1)
        return self.fusion(fused.mean(1))
```

### 7.2 If LLM Layers Lose Spatial Information

**Option A: Early Layer Extraction**
- Extract from LLM layer 1-7 instead of final layer
- Loses some reasoning capability, preserves spatial info
- May need separate reasoning pass

**Option B: Dual-Path Architecture**
- Path 1: Full VLM for text reasoning
- Path 2: ViT-only for spatial conditioning
- Combine at adapter level

```
                    +-> LLM -> Text Response
Image -> ViT ------+
                    +-> Adapter -> Video Decoder
```

### 7.3 If Pre-Merge Also Insufficient

**Option A: Different VLM**
- Try VLMs without aggressive token merging
- Candidates: LLaVA-NeXT, InternVL (check merge ratios)

**Option B: Multi-Scale Extraction**
- Extract from multiple ViT layers
- Combine early (high-res) and late (semantic) features

**Option C: Auxiliary Encoder**
- Add separate high-resolution encoder (e.g., DINOv2)
- Fuse with VLM features in adapter

---

## 8. Timeline

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| **Day 1** | Setup, data preparation | Extraction pipeline ready |
| **Day 2** | E-Q2.1: Pre-merge analysis | Pre-merge baseline metrics |
| **Day 3** | E-Q2.2: Post-merge analysis | Merge impact quantification |
| **Day 4** | E-Q2.3: LLM layer analysis | Layer-wise decay curve |
| **Day 5** | E-Q2.4: Detection probe | mAP results by layer |
| **Day 6** | E-Q2.5: Fine-grained probe | LPIPS, edge metrics |
| **Day 7** | E-Q2.6: Temporal probe, analysis | Final report, recommendation |

**Total: 5-7 days**

### Detailed Hour Breakdown

| Experiment | Setup | Training | Eval | Analysis | Total |
|------------|-------|----------|------|----------|-------|
| E-Q2.1 | 2h | 2h | 1h | 1h | 6h |
| E-Q2.2 | 1h | 2h | 1h | 1h | 5h |
| E-Q2.3 | 2h | 6h | 2h | 2h | 12h |
| E-Q2.4 | 2h | 8h | 2h | 2h | 14h |
| E-Q2.5 | 2h | 6h | 2h | 2h | 12h |
| E-Q2.6 | 2h | 6h | 2h | 2h | 12h |
| **Total** | **11h** | **30h** | **10h** | **10h** | **61h** |

---

## 9. Dependencies

**External Dependencies:** None - this experiment can run independently.

**Internal Dependencies:**
- Requires Qwen2.5-VL model weights (publicly available)
- Requires COCO dataset (publicly available)
- Does NOT require trained adapter (we're probing frozen features)

**Can Block:**
- Q1 (Latent Space Alignment) - findings here inform adapter design
- Training decisions - extraction point choice affects architecture

**Recommended Order:** Run Q2 early to inform other experiments.

---

## 10. Deliverables

### 10.1 Primary Deliverables

1. **Layer-by-Layer Information Analysis Report**
   - Quantitative metrics for each extraction point
   - Visualizations of information decay
   - Statistical significance tests

2. **Extraction Point Recommendation**
   - Recommended extraction point(s) for video conditioning
   - Trade-off analysis (spatial vs computational cost)
   - Confidence level in recommendation

3. **Code Artifacts**
   - `extract_representations.py` - Feature extraction pipeline
   - `spatial_probes.py` - Probe architectures and training
   - `evaluate_probes.py` - Evaluation and visualization

### 10.2 Report Structure

```markdown
# Q2 Results: Information Preservation Analysis

## Executive Summary
- Best extraction point: [E1/E2/.../E8]
- Information retention at recommendation: [X]%
- Confidence: [High/Medium/Low]

## Quantitative Results
[Tables and figures]

## Key Findings
1. Finding 1
2. Finding 2
3. Finding 3

## Recommendations
- Primary: Use [extraction point] for conditioning
- Fallback: If quality insufficient, use [alternative]

## Next Steps
- Proceed to Q1 with [extraction point]
- Consider [architectural changes] if needed
```

### 10.3 Success Criteria for Deliverables

| Deliverable | Success Criterion |
|-------------|-------------------|
| Analysis Report | All 8 extraction points evaluated with 3+ metrics each |
| Recommendation | Clear guidance with confidence intervals |
| Code | Reproducible, documented, tested |

---

## Appendix A: Code Templates

### A.1 Full Extraction Pipeline

```python
"""
q2_extraction.py - Extract representations from Qwen2.5-VL at multiple points
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Dict, List
import numpy as np

class Qwen2VLExtractor:
    """Extract representations from multiple points in Qwen2.5-VL."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

        self._hooks = []
        self._activations = {}

    def _register_hooks(self, layer_indices: List[int]):
        """Register forward hooks on specified layers."""

        # ViT layer hooks
        for i, layer in enumerate(self.model.visual.encoder.layers):
            if i in layer_indices:
                hook = layer.register_forward_hook(
                    lambda m, inp, out, idx=i: self._activations.update({f'vit_{idx}': out.detach()})
                )
                self._hooks.append(hook)

        # LLM layer hooks
        for i, layer in enumerate(self.model.model.layers):
            if i in layer_indices:
                hook = layer.register_forward_hook(
                    lambda m, inp, out, idx=i: self._activations.update({f'llm_{idx}': out[0].detach()})
                )
                self._hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    @torch.no_grad()
    def extract(self, image, extraction_points: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Extract representations from specified points.

        Args:
            image: PIL Image or path
            extraction_points: List of points like ['vit_pre_merge', 'vit_post_merge', 'llm_14']

        Returns:
            Dictionary mapping extraction point names to tensors
        """
        if extraction_points is None:
            extraction_points = ['vit_pre_merge', 'vit_post_merge', 'llm_0', 'llm_13', 'llm_27']

        self._activations = {}

        # Determine which layers to hook
        layer_indices = []
        for point in extraction_points:
            if point.startswith('vit_') and point not in ['vit_pre_merge', 'vit_post_merge']:
                layer_indices.append(int(point.split('_')[1]))
            elif point.startswith('llm_'):
                layer_indices.append(int(point.split('_')[1]))

        self._register_hooks(layer_indices)

        try:
            # Prepare inputs
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Forward pass with hidden states
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            # Collect results
            results = {}

            for point in extraction_points:
                if point == 'vit_post_merge':
                    # Standard visual output after merger
                    results[point] = self._get_visual_output(inputs)
                elif point == 'vit_pre_merge':
                    # Need to run ViT encoder separately without merge
                    results[point] = self._get_premerge_output(inputs)
                elif point in self._activations:
                    results[point] = self._activations[point]
                elif point.startswith('llm_'):
                    idx = int(point.split('_')[1])
                    results[point] = outputs.hidden_states[idx + 1]  # +1 because index 0 is embeddings

            return results

        finally:
            self._remove_hooks()

    def _get_visual_output(self, inputs):
        """Get post-merge visual embeddings."""
        pixel_values = inputs.get('pixel_values')
        grid_thw = inputs.get('image_grid_thw')
        return self.model.visual(pixel_values, grid_thw)

    def _get_premerge_output(self, inputs):
        """Get pre-merge visual embeddings (requires model modification)."""
        # This requires accessing internal ViT state before merger
        # Implementation depends on specific model version
        # Placeholder for now
        raise NotImplementedError("Pre-merge extraction requires custom forward pass")


# Usage example
if __name__ == "__main__":
    from PIL import Image

    extractor = Qwen2VLExtractor()

    image = Image.open("test_image.jpg")
    features = extractor.extract(image, ['vit_post_merge', 'llm_0', 'llm_13', 'llm_27'])

    for name, tensor in features.items():
        print(f"{name}: {tensor.shape}")
```

### A.2 Spatial Probe Training

```python
"""
q2_probes.py - Linear probes for spatial information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from typing import Tuple, Dict
import numpy as np

class BoundingBoxProbe(nn.Module):
    """Linear probe for bounding box prediction."""

    def __init__(self, input_dim: int, max_objects: int = 10):
        super().__init__()
        self.max_objects = max_objects
        self.fc = nn.Linear(input_dim, max_objects * 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, seq_len, dim] or [batch, dim]
        Returns:
            boxes: [batch, max_objects, 4] in (x1, y1, x2, y2) format normalized to [0,1]
        """
        if features.dim() == 3:
            features = features.mean(dim=1)  # Global average pooling

        boxes = self.fc(features).view(-1, self.max_objects, 4)
        return boxes.sigmoid()  # Normalize to [0, 1]


class RelativePositionProbe(nn.Module):
    """Probe for relative spatial relationships."""

    def __init__(self, input_dim: int):
        super().__init__()
        # Predict: left-of, right-of, above, below for object pairs
        self.fc = nn.Linear(input_dim, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)
        return self.fc(features)


class EdgeReconstructionProbe(nn.Module):
    """Probe for reconstructing edge maps."""

    def __init__(self, input_dim: int, output_size: int = 224):
        super().__init__()
        self.output_size = output_size

        self.proj = nn.Linear(input_dim, 512 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)

        spatial = self.proj(features).view(-1, 512, 7, 7)
        edges = self.decoder(spatial)
        return F.interpolate(edges, size=(self.output_size, self.output_size), mode='bilinear')


def train_probe(
    probe: nn.Module,
    extractor,
    dataloader: DataLoader,
    extraction_point: str,
    loss_fn,
    target_key: str,
    epochs: int = 50,
    lr: float = 1e-3
) -> Dict[str, float]:
    """
    Train a probe on frozen features.

    Returns:
        Dictionary with training metrics
    """
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    probe.train()

    best_loss = float('inf')
    history = []

    for epoch in range(epochs):
        epoch_losses = []

        for batch in dataloader:
            images = batch['image']
            targets = batch[target_key]

            # Extract features (frozen)
            with torch.no_grad():
                features = extractor.extract(images, [extraction_point])[extraction_point]

            # Forward through probe
            predictions = probe(features)
            loss = loss_fn(predictions, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return {
        'final_loss': history[-1],
        'best_loss': best_loss,
        'history': history
    }


def evaluate_bbox_probe(
    probe: nn.Module,
    extractor,
    dataloader: DataLoader,
    extraction_point: str
) -> Dict[str, float]:
    """
    Evaluate bounding box probe.

    Returns:
        Dictionary with IoU metrics
    """
    probe.eval()
    all_ious = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image']
            gt_boxes = batch['boxes']  # [batch, num_objects, 4]

            features = extractor.extract(images, [extraction_point])[extraction_point]
            pred_boxes = probe(features)

            # Calculate IoU for each image
            for i in range(len(images)):
                # Filter valid boxes (non-zero)
                valid_gt = gt_boxes[i][gt_boxes[i].sum(dim=1) > 0]
                valid_pred = pred_boxes[i][:len(valid_gt)]

                if len(valid_gt) > 0:
                    ious = box_iou(valid_pred, valid_gt)
                    best_ious = ious.max(dim=1)[0]
                    all_ious.extend(best_ious.cpu().numpy())

    return {
        'mean_iou': np.mean(all_ious),
        'median_iou': np.median(all_ious),
        'iou_std': np.std(all_ious),
        'iou_above_0.5': np.mean(np.array(all_ious) > 0.5),
        'iou_above_0.7': np.mean(np.array(all_ious) > 0.7)
    }
```

---

## Appendix B: Expected Results Reference

Based on similar probing studies and the Qwen2-VL architecture, we estimate:

| Extraction Point | Expected IoU | Confidence |
|------------------|--------------|------------|
| Pre-merge ViT | 0.80-0.90 | High |
| Post-merge ViT | 0.60-0.75 | Medium |
| LLM Layer 1 | 0.55-0.70 | Medium |
| LLM Layer 14 | 0.40-0.55 | Low |
| LLM Layer 28 | 0.30-0.45 | Low |

**Rationale:**
- ViT layers optimized for visual processing, should retain spatial info
- 2x2 merge causes ~15-25% information loss based on ToMe papers
- LLM progressively abstracts toward language, losing spatial specificity

---

## Appendix C: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pre-merge extraction difficult | Medium | High | Document alternative: use multiple ViT layers |
| Probes don't converge | Low | Medium | Try non-linear probes, more training |
| Results inconclusive | Medium | Medium | Increase dataset size, add metrics |
| GPU OOM during extraction | Medium | Low | Use gradient checkpointing, smaller batches |

---

## References

1. Raghu et al., "Do Vision Transformers See Like Convolutional Neural Networks?" NeurIPS 2021
2. Bolya et al., "Token Merging: Your ViT But Faster" ICLR 2023
3. Wang et al., "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution" 2024
4. Alain & Bengio, "Understanding intermediate layers using linear classifier probes" ICLR Workshop 2017
