"""E-Q2.4: Spatial Reconstruction Probe (Bounding Box Detection)

Objective: Measure precise object localization capability at each extraction point.

Protocol:
1. Use COCO-style validation data with ground truth boxes
2. Train DETR-style detection head on frozen features
3. Evaluate mAP at different extraction points

Probe Architecture: DETR-style with learnable queries and cross-attention.

Results Table Template:
| Extraction Point | mAP@0.5 | mAP@0.75 | mAP (avg) | Small | Medium | Large |
|------------------|---------|----------|-----------|-------|--------|-------|
| Pre-merge ViT    |    ?    |    ?     |     ?     |   ?   |   ?    |   ?   |
| Post-merge ViT   |    ?    |    ?     |     ?     |   ?   |   ?    |   ?   |
| LLM Layer 1      |    ?    |    ?     |     ?     |   ?   |   ?    |   ?   |

Success Criteria: mAP@0.5 > 0.4 at some extraction point
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.ops import box_iou, nms
from collections import defaultdict

from runner import ExperimentRunner


class SyntheticDetectionDataset:
    """Synthetic dataset with multi-object scenes for detection evaluation."""

    def __init__(self, n_samples: int = 300, img_size: int = 448):
        self.n_samples = n_samples
        self.img_size = img_size
        self.shapes = ["circle", "square", "triangle"]
        self.colors = [
            ("red", (255, 0, 0)),
            ("green", (0, 255, 0)),
            ("blue", (0, 0, 255)),
            ("yellow", (255, 255, 0)),
        ]
        # 12 classes: 3 shapes x 4 colors
        self.n_classes = len(self.shapes) * len(self.colors)

        np.random.seed(42)
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        for _ in range(self.n_samples):
            n_objects = np.random.randint(1, 6)  # 1-5 objects
            img = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            boxes = []
            labels = []
            sizes = []  # Track object size category

            for _ in range(n_objects):
                # Variable object sizes
                size_cat = np.random.choice(["small", "medium", "large"])
                if size_cat == "small":
                    size = np.random.randint(20, 40)
                elif size_cat == "medium":
                    size = np.random.randint(40, 70)
                else:
                    size = np.random.randint(70, 100)

                x1 = np.random.randint(0, max(1, self.img_size - size))
                y1 = np.random.randint(0, max(1, self.img_size - size))
                x2 = x1 + size
                y2 = y1 + size

                shape_idx = np.random.randint(len(self.shapes))
                color_idx = np.random.randint(len(self.colors))
                shape = self.shapes[shape_idx]
                color_name, color = self.colors[color_idx]

                # Class label: shape_idx * n_colors + color_idx
                label = shape_idx * len(self.colors) + color_idx

                if shape == "circle":
                    draw.ellipse([x1, y1, x2, y2], fill=color)
                elif shape == "square":
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                else:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    half = size // 2
                    points = [(cx, y1), (x1, y2), (x2, y2)]
                    draw.polygon(points, fill=color)

                boxes.append([
                    x1 / self.img_size, y1 / self.img_size,
                    x2 / self.img_size, y2 / self.img_size,
                ])
                labels.append(label)
                sizes.append(size_cat)

            samples.append({
                "image": img,
                "boxes": np.array(boxes, dtype=np.float32),
                "labels": np.array(labels, dtype=np.int64),
                "sizes": sizes,
                "n_objects": n_objects,
            })

        return samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.samples[idx]


class DetectionProbe(nn.Module):
    """DETR-style detection probe for spatial accuracy measurement.

    Uses learnable object queries with cross-attention to features.
    """

    def __init__(self, input_dim: int, num_queries: int = 20, n_classes: int = 12, hidden_dim: int = 256):
        super().__init__()
        self.num_queries = num_queries
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        # Project input features to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Cross-attention layer (simple single-head for linear probe spirit)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Self-attention for queries
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Output heads
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes + 1),  # +1 for background
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch, seq_len, input_dim]

        Returns:
            boxes: [batch, num_queries, 4] normalized (x1, y1, x2, y2)
            class_logits: [batch, num_queries, n_classes + 1]
        """
        batch_size = features.shape[0]

        # Project features
        features = self.input_proj(features)  # [batch, seq_len, hidden_dim]

        # Get queries
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: queries attend to features
        queries, _ = self.cross_attention(queries, features, features)

        # Self-attention between queries
        queries, _ = self.self_attention(queries, queries, queries)

        # Predict boxes and classes
        boxes = self.box_head(queries).sigmoid()  # [batch, num_queries, 4]
        class_logits = self.class_head(queries)  # [batch, num_queries, n_classes+1]

        return boxes, class_logits


def extract_features_for_detection(
    model, processor, images: list, extraction_point: str, device, runner: ExperimentRunner
) -> torch.Tensor:
    """Extract features from specified extraction point.

    Args:
        extraction_point: One of "post_merge", "llm_layer_0", "llm_layer_13", etc.

    Returns:
        Tensor of shape [n_images, seq_len, dim]
    """
    features_list = []

    with torch.no_grad():
        for i, img in enumerate(images):
            if extraction_point == "post_merge":
                # Use visual embeddings directly (don't need full forward pass)
                image_inputs = processor.image_processor(images=[img], return_tensors="pt")
                pixel_values = image_inputs["pixel_values"].to(device).to(model.dtype)
                image_grid_thw = image_inputs["image_grid_thw"].to(device)
                visual_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                # Handle both possible shapes
                if visual_embeds.dim() == 2:
                    # Shape is [num_tokens, embed_dim], add batch dim
                    feat = visual_embeds.unsqueeze(0)
                else:
                    # Shape is [batch, num_tokens, embed_dim]
                    feat = visual_embeds
            else:
                # For LLM layers, need full forward pass
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Describe."},
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text],
                    images=[img],
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

                if extraction_point.startswith("llm_layer_"):
                    layer_idx = int(extraction_point.split("_")[-1])
                    # +1 because index 0 is embedding layer
                    feat = outputs.hidden_states[layer_idx + 1]
                else:
                    # Default: first hidden state
                    feat = outputs.hidden_states[1]

            # feat should be [1, seq_len, dim]
            features_list.append(feat.float().cpu())

            if (i + 1) % 50 == 0:
                runner.log_metrics({"e_q2_4/extraction_progress": (i + 1) / len(images)})
                print(f"    Extracted {i + 1}/{len(images)} images")

    return torch.cat(features_list, dim=0)


def hungarian_matching(pred_boxes, pred_logits, gt_boxes, gt_labels):
    """Simple matching based on IoU (simplified Hungarian matching)."""
    from scipy.optimize import linear_sum_assignment

    n_pred = pred_boxes.shape[0]
    n_gt = gt_boxes.shape[0]

    if n_gt == 0:
        return [], []

    # Compute cost matrix based on IoU
    iou_matrix = box_iou(pred_boxes, gt_boxes)  # [n_pred, n_gt]
    cost_matrix = 1 - iou_matrix.cpu().numpy()

    # Class cost (simplified: just check if predicted class matches)
    pred_classes = pred_logits.argmax(dim=-1)  # [n_pred]
    class_cost = (pred_classes.unsqueeze(1) != gt_labels.unsqueeze(0)).float() * 2
    cost_matrix += class_cost.cpu().numpy()

    # Hungarian matching
    pred_idx, gt_idx = linear_sum_assignment(cost_matrix)

    return pred_idx.tolist(), gt_idx.tolist()


def compute_map(
    all_pred_boxes: list,
    all_pred_logits: list,
    all_gt_boxes: list,
    all_gt_labels: list,
    all_gt_sizes: list,
    iou_threshold: float = 0.5,
    n_classes: int = 12,
) -> dict:
    """Compute mean Average Precision (mAP) at given IoU threshold."""

    # Collect predictions and ground truths per class
    class_predictions = defaultdict(list)  # class -> list of (score, is_tp, size_cat)
    class_n_gt = defaultdict(int)
    class_n_gt_by_size = {
        "small": defaultdict(int),
        "medium": defaultdict(int),
        "large": defaultdict(int),
    }

    for img_idx in range(len(all_pred_boxes)):
        pred_boxes = all_pred_boxes[img_idx]
        pred_logits = all_pred_logits[img_idx]
        gt_boxes = all_gt_boxes[img_idx]
        gt_labels = all_gt_labels[img_idx]
        gt_sizes = all_gt_sizes[img_idx]

        if len(gt_boxes) == 0:
            continue

        # Convert to tensor if needed
        if not isinstance(pred_boxes, torch.Tensor):
            pred_boxes = torch.tensor(pred_boxes)
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.tensor(gt_boxes)
        if not isinstance(gt_labels, torch.Tensor):
            gt_labels = torch.tensor(gt_labels)

        # Count ground truths
        for i, label in enumerate(gt_labels.tolist()):
            class_n_gt[label] += 1
            if i < len(gt_sizes):
                class_n_gt_by_size[gt_sizes[i]][label] += 1

        # Get predictions with confidence scores
        pred_probs = F.softmax(pred_logits, dim=-1)  # [num_queries, n_classes+1]
        pred_classes = pred_probs[:, :-1].argmax(dim=-1)  # Ignore background
        pred_scores = pred_probs[:, :-1].max(dim=-1)[0]

        # Filter low confidence predictions
        keep = pred_scores > 0.3
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]

        if len(pred_boxes) == 0:
            continue

        # Match predictions to ground truths
        matched_gt = set()
        for pred_idx in range(len(pred_boxes)):
            pred_box = pred_boxes[pred_idx:pred_idx+1]
            pred_class = pred_classes[pred_idx].item()
            pred_score = pred_scores[pred_idx].item()

            # Find best matching GT of same class
            best_iou = 0
            best_gt_idx = -1

            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                if gt_labels[gt_idx].item() != pred_class:
                    continue

                iou = box_iou(pred_box, gt_boxes[gt_idx:gt_idx+1])[0, 0].item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            is_tp = best_iou >= iou_threshold
            if is_tp:
                matched_gt.add(best_gt_idx)
                size_cat = gt_sizes[best_gt_idx] if best_gt_idx < len(gt_sizes) else "unknown"
            else:
                size_cat = "unknown"

            class_predictions[pred_class].append((pred_score, is_tp, size_cat))

    # Compute AP per class
    aps = []
    ap_by_size = {"small": [], "medium": [], "large": []}

    for class_idx in range(n_classes):
        predictions = class_predictions[class_idx]
        n_gt = class_n_gt[class_idx]

        if n_gt == 0:
            continue

        # Sort by confidence
        predictions.sort(key=lambda x: x[0], reverse=True)

        # Compute precision-recall curve
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for score, is_tp, size_cat in predictions:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / n_gt

            precisions.append(precision)
            recalls.append(recall)

        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            ap += max(prec_at_recall) if prec_at_recall else 0

        ap /= 11
        aps.append(ap)

    map_score = np.mean(aps) if aps else 0

    return {
        "mAP": float(map_score),
        "n_classes_evaluated": len(aps),
    }


def train_detection_probe(
    features: torch.Tensor,
    dataset: SyntheticDetectionDataset,
    epochs: int = 100,
    lr: float = 1e-4,
) -> tuple[DetectionProbe, dict]:
    """Train detection probe on frozen features."""
    device = features.device
    input_dim = features.shape[-1]
    n_classes = dataset.n_classes

    probe = DetectionProbe(input_dim, num_queries=20, n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)

    # Split into train/val
    n_train = int(len(features) * 0.8)
    train_features = features[:n_train]
    val_features = features[n_train:]
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(features)))

    history = {"train_loss": [], "val_map": []}

    for epoch in range(epochs):
        probe.train()
        epoch_losses = []

        # Mini-batch training
        batch_size = 16
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_features = train_features[start:end]
            batch_indices = train_indices[start:end]

            pred_boxes, pred_logits = probe(batch_features)

            # Compute loss
            total_loss = 0
            for i, idx in enumerate(batch_indices):
                sample = dataset[idx]
                gt_boxes = torch.tensor(sample["boxes"]).to(device)
                gt_labels = torch.tensor(sample["labels"]).to(device)

                if len(gt_boxes) == 0:
                    continue

                # Simple matching and loss
                pred_b = pred_boxes[i]
                pred_l = pred_logits[i]

                # Find best matching predictions for each GT
                ious = box_iou(pred_b, gt_boxes)
                best_pred_idx = ious.argmax(dim=0)

                # Box loss (L1)
                matched_pred_boxes = pred_b[best_pred_idx]
                box_loss = F.l1_loss(matched_pred_boxes, gt_boxes)

                # Class loss (cross-entropy)
                matched_pred_logits = pred_l[best_pred_idx]
                class_loss = F.cross_entropy(matched_pred_logits, gt_labels)

                total_loss += box_loss + class_loss

            if total_loss > 0:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_losses.append(total_loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        history["train_loss"].append(avg_loss)

        # Validation every 20 epochs
        if (epoch + 1) % 20 == 0:
            probe.eval()
            all_pred_boxes = []
            all_pred_logits = []
            all_gt_boxes = []
            all_gt_labels = []
            all_gt_sizes = []

            with torch.no_grad():
                for i, idx in enumerate(val_indices):
                    sample = dataset[idx]
                    feat = val_features[i:i+1]

                    pred_b, pred_l = probe(feat)
                    all_pred_boxes.append(pred_b[0])
                    all_pred_logits.append(pred_l[0])
                    all_gt_boxes.append(torch.tensor(sample["boxes"]).to(device))
                    all_gt_labels.append(torch.tensor(sample["labels"]).to(device))
                    all_gt_sizes.append(sample["sizes"])

            metrics = compute_map(
                all_pred_boxes, all_pred_logits,
                all_gt_boxes, all_gt_labels, all_gt_sizes,
                iou_threshold=0.5, n_classes=n_classes
            )
            history["val_map"].append(metrics["mAP"])
            print(f"      Epoch {epoch+1}: loss={avg_loss:.4f}, val_mAP@0.5={metrics['mAP']:.4f}")

    return probe, history


def create_detection_visualization(
    dataset: SyntheticDetectionDataset,
    pred_boxes_list: list,
    pred_logits_list: list,
    extraction_point: str,
    map_score: float,
) -> bytes:
    """Create visualization of detection results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx in range(min(8, len(pred_boxes_list))):
        ax = axes[idx]
        sample = dataset[idx]
        img = sample["image"]
        gt_boxes = sample["boxes"]
        gt_labels = sample["labels"]

        pred_boxes = pred_boxes_list[idx]
        pred_logits = pred_logits_list[idx]

        # Show image
        ax.imshow(img)

        # Draw GT boxes (green)
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box * dataset.img_size
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor="green", facecolor="none",
            )
            ax.add_patch(rect)

        # Draw predicted boxes (red)
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_scores = pred_probs[:, :-1].max(dim=-1)[0]
        keep = pred_scores > 0.4

        for i, (box, score) in enumerate(zip(pred_boxes[keep].cpu().numpy(), pred_scores[keep].cpu().numpy())):
            x1, y1, x2, y2 = box * dataset.img_size
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)

        ax.set_title(f"Sample {idx}")
        ax.axis("off")

    plt.suptitle(f"Detection Results ({extraction_point})\nmAP@0.5: {map_score:.3f}")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e_q2_4_detection_probe(runner: ExperimentRunner) -> dict:
    """Run DETR-style detection probe at multiple extraction points.

    This measures precise object localization using mAP metrics,
    providing a more rigorous assessment than simple IoU.

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-Q2.4: Detection Probe (mAP Evaluation)")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"e_q2_4/stage": 0, "e_q2_4/progress": 0.0})

    # =========================================================================
    # Stage 1: Create dataset
    # =========================================================================
    print("\n[Stage 1/4] Creating synthetic detection dataset...")

    dataset = SyntheticDetectionDataset(n_samples=200, img_size=448)
    print(f"  Created {len(dataset)} samples with 1-5 objects each")
    print(f"  Number of classes: {dataset.n_classes}")

    runner.log_metrics({"e_q2_4/stage": 1, "e_q2_4/progress": 0.1})

    # =========================================================================
    # Stage 2: Load model and extract features
    # =========================================================================
    print("\n[Stage 2/4] Loading Qwen2.5-VL and extracting features...")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
    )
    device = model.device
    print(f"  Model loaded on {device}")

    images = [dataset[i]["image"] for i in range(len(dataset))]

    # Extract from multiple points for comparison
    extraction_points = ["post_merge", "llm_layer_0", "llm_layer_13"]
    features_by_point = {}

    for point in extraction_points:
        print(f"\n  Extracting features from {point}...")
        features_by_point[point] = extract_features_for_detection(
            model, processor, images, point, device, runner
        )
        print(f"    Shape: {features_by_point[point].shape}")

    runner.log_metrics({"e_q2_4/stage": 2, "e_q2_4/progress": 0.4})

    # Free GPU memory
    del model
    del processor
    torch.cuda.empty_cache()

    # =========================================================================
    # Stage 3: Train and evaluate detection probes
    # =========================================================================
    print("\n[Stage 3/4] Training detection probes...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_by_point = {}
    best_point = None
    best_map = 0

    for point in extraction_points:
        print(f"\n  Training probe for {point}...")
        features = features_by_point[point].to(device)

        probe, history = train_detection_probe(features, dataset, epochs=100)

        # Final evaluation
        probe.eval()
        all_pred_boxes = []
        all_pred_logits = []
        all_gt_boxes = []
        all_gt_labels = []
        all_gt_sizes = []

        n_val = int(len(features) * 0.2)
        val_indices = list(range(len(features) - n_val, len(features)))

        with torch.no_grad():
            for idx in val_indices:
                sample = dataset[idx]
                feat = features[idx:idx+1]
                pred_b, pred_l = probe(feat)

                all_pred_boxes.append(pred_b[0])
                all_pred_logits.append(pred_l[0])
                all_gt_boxes.append(torch.tensor(sample["boxes"]).to(device))
                all_gt_labels.append(torch.tensor(sample["labels"]).to(device))
                all_gt_sizes.append(sample["sizes"])

        # Compute mAP at different thresholds
        map_05 = compute_map(
            all_pred_boxes, all_pred_logits,
            all_gt_boxes, all_gt_labels, all_gt_sizes,
            iou_threshold=0.5, n_classes=dataset.n_classes
        )["mAP"]

        map_075 = compute_map(
            all_pred_boxes, all_pred_logits,
            all_gt_boxes, all_gt_labels, all_gt_sizes,
            iou_threshold=0.75, n_classes=dataset.n_classes
        )["mAP"]

        results_by_point[point] = {
            "mAP@0.5": map_05,
            "mAP@0.75": map_075,
            "mAP_avg": (map_05 + map_075) / 2,
            "pred_boxes": all_pred_boxes,
            "pred_logits": all_pred_logits,
        }

        print(f"    mAP@0.5: {map_05:.4f}, mAP@0.75: {map_075:.4f}")

        if map_05 > best_map:
            best_map = map_05
            best_point = point

        runner.log_metrics({
            f"e_q2_4/{point}_map_05": map_05,
            f"e_q2_4/{point}_map_075": map_075,
        })

    runner.log_metrics({"e_q2_4/stage": 3, "e_q2_4/progress": 0.8})

    # =========================================================================
    # Stage 4: Create visualizations and save results
    # =========================================================================
    print("\n[Stage 4/4] Creating visualizations...")

    # Create visualization for best point
    vis_bytes = create_detection_visualization(
        dataset,
        results_by_point[best_point]["pred_boxes"],
        results_by_point[best_point]["pred_logits"],
        best_point,
        results_by_point[best_point]["mAP@0.5"],
    )
    vis_path = runner.results.save_artifact("detection_visualization.png", vis_bytes)

    # Create comparison chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(extraction_points))
    width = 0.35

    map_05_vals = [results_by_point[p]["mAP@0.5"] for p in extraction_points]
    map_075_vals = [results_by_point[p]["mAP@0.75"] for p in extraction_points]

    ax.bar(x - width/2, map_05_vals, width, label="mAP@0.5", color="blue", alpha=0.7)
    ax.bar(x + width/2, map_075_vals, width, label="mAP@0.75", color="orange", alpha=0.7)

    ax.set_ylabel("mAP")
    ax.set_xlabel("Extraction Point")
    ax.set_title("Detection Performance by Extraction Point")
    ax.set_xticks(x)
    ax.set_xticklabels(extraction_points)
    ax.legend()
    ax.axhline(y=0.4, color="red", linestyle="--", label="Target (0.4)")
    ax.set_ylim(0, 1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    comparison_path = runner.results.save_artifact("detection_comparison.png", buf.read())

    runner.log_metrics({
        "e_q2_4/stage": 4,
        "e_q2_4/progress": 1.0,
        "e_q2_4/best_point": extraction_points.index(best_point),
        "e_q2_4/best_map_05": best_map,
        "mAP": best_map,  # For overall assessment
    })

    # Save detailed results
    results_data = {
        "extraction_points_evaluated": extraction_points,
        "results_by_point": {
            point: {
                "mAP@0.5": results_by_point[point]["mAP@0.5"],
                "mAP@0.75": results_by_point[point]["mAP@0.75"],
                "mAP_avg": results_by_point[point]["mAP_avg"],
            }
            for point in extraction_points
        },
        "best_point": best_point,
        "best_map_05": best_map,
        "n_classes": dataset.n_classes,
        "n_samples": len(dataset),
    }
    data_path = runner.results.save_json_artifact("detection_analysis.json", results_data)

    # =========================================================================
    # Interpret results
    # =========================================================================
    if best_map >= 0.4:
        finding = (
            f"Detection probe achieves target mAP@0.5 ({best_map:.3f}) at {best_point}. "
            f"Spatial information is sufficient for accurate object localization. "
            f"Recommended extraction point: {best_point}."
        )
    elif best_map >= 0.3:
        finding = (
            f"Detection probe achieves moderate mAP@0.5 ({best_map:.3f}) at {best_point}. "
            f"Spatial information is partially preserved but may need enhancement "
            f"for high-quality video generation."
        )
    else:
        finding = (
            f"Detection probe achieves low mAP@0.5 ({best_map:.3f}) at {best_point}. "
            f"Spatial information is significantly degraded. Consider using pre-merge "
            f"features or architectural modifications."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "best_map_05": float(best_map),
            "best_extraction_point": best_point,
            **{f"{point}_map_05": results_by_point[point]["mAP@0.5"]
               for point in extraction_points},
            **{f"{point}_map_075": results_by_point[point]["mAP@0.75"]
               for point in extraction_points},
        },
        "artifacts": [vis_path, comparison_path, data_path],
    }
