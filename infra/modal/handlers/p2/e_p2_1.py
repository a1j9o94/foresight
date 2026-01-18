"""E-P2.1: DINOv2 Spatial Feature Analysis

Objective: Characterize DINOv2 feature space and validate spatial information preservation.

Protocol:
1. Extract DINOv2-giant features from test images
2. Train position regression probe (same as Q2 E-Q2.4)
3. Compare spatial metrics to VLM baseline

Success Metrics:
- Bbox IoU > 0.65 (target: > 0.75)
- mAP@0.5 > 0.40 (target: > 0.60)
- Spatial IoU > 0.70 (target: > 0.80)

This experiment validates that DINOv2 preserves spatial information at sufficient fidelity.
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
from collections import defaultdict

from runner import ExperimentRunner


class SpatialProbe(nn.Module):
    """Predict bounding boxes from DINOv2 features using cross-attention."""

    def __init__(self, feature_dim: int = 1536, hidden_dim: int = 512):
        super().__init__()
        # Attention pooling
        self.attn_pool = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        # Query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        # Bbox prediction head
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # x_center, y_center, width, height
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N_patches, feature_dim] DINOv2 patch features

        Returns:
            [B, 4] bounding box predictions (x_c, y_c, w, h) normalized
        """
        B = features.shape[0]
        query = self.query.expand(B, -1, -1)
        attn_out, _ = self.attn_pool(query, features, features)
        return self.bbox_head(attn_out.squeeze(1))


class DetectionProbe(nn.Module):
    """DETR-style detection probe for mAP evaluation."""

    def __init__(self, input_dim: int = 1536, num_queries: int = 20, n_classes: int = 12, hidden_dim: int = 256):
        super().__init__()
        self.num_queries = num_queries
        self.n_classes = n_classes

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes + 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        features = self.input_proj(features)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        queries, _ = self.cross_attention(queries, features, features)
        queries, _ = self.self_attention(queries, queries, queries)
        boxes = self.box_head(queries).sigmoid()
        class_logits = self.class_head(queries)
        return boxes, class_logits


def load_dinov2_model(device: torch.device):
    """Load DINOv2-giant model."""
    print("  Loading DINOv2-giant model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    model = model.to(device)
    model.eval()
    print(f"  DINOv2-giant loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    return model


def extract_dinov2_features(
    images: list,
    model: nn.Module,
    device: torch.device,
    runner: ExperimentRunner,
    prefix: str = "",
) -> torch.Tensor:
    """Extract DINOv2 patch features from images.

    Args:
        images: List of PIL Images
        model: DINOv2 model
        device: Device to use
        runner: ExperimentRunner for logging
        prefix: Prefix for logging

    Returns:
        Tensor of shape [N_images, N_patches, 1536]
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_list = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = torch.stack([transform(img) for img in batch]).to(device)

            # Get patch features (excluding CLS token)
            features = model.forward_features(batch_tensors)
            # DINOv2 returns dict with 'x_norm_patchtokens'
            if isinstance(features, dict):
                patch_features = features.get('x_norm_patchtokens', features.get('x_prenorm', None))
                if patch_features is None:
                    # Fallback: get last layer output
                    patch_features = features['x_norm'][:, 1:, :]  # Exclude CLS
            else:
                # features is tensor [B, 1 + N_patches, dim]
                patch_features = features[:, 1:, :]  # Exclude CLS token

            features_list.append(patch_features.cpu())

            if prefix:
                progress = min(1.0, (i + batch_size) / len(images))
                runner.log_metrics({f"e_p2_1/{prefix}_progress": progress})
            if (i + batch_size) % 50 == 0 or i + batch_size >= len(images):
                print(f"    Processed {min(i + batch_size, len(images))}/{len(images)} images")

    return torch.cat(features_list, dim=0)


def generate_position_dataset(n_images: int = 200, img_size: int = 224) -> tuple[list, np.ndarray]:
    """Generate images with objects at known positions."""
    shapes = ["circle", "square", "triangle"]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    images = []
    boxes = []
    np.random.seed(42)

    for i in range(n_images):
        shape = shapes[i % len(shapes)]
        color = colors[i % len(colors)]

        size = np.random.randint(25, 60)
        margin = size + 10
        cx = np.random.randint(margin, img_size - margin)
        cy = np.random.randint(margin, img_size - margin)

        img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        if shape == "circle":
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif shape == "square":
            draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif shape == "triangle":
            points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
            draw.polygon(points, fill=color)

        images.append(img)
        boxes.append([
            cx / img_size,
            cy / img_size,
            (2 * size) / img_size,
            (2 * size) / img_size,
        ])

    return images, np.array(boxes, dtype=np.float32)


def generate_detection_dataset(n_samples: int = 200, img_size: int = 224):
    """Generate multi-object dataset for detection evaluation."""
    shapes = ["circle", "square", "triangle"]
    colors = [
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
        ("yellow", (255, 255, 0)),
    ]
    n_classes = len(shapes) * len(colors)

    np.random.seed(42)
    samples = []

    for _ in range(n_samples):
        n_objects = np.random.randint(1, 5)
        img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        sample_boxes = []
        sample_labels = []

        for _ in range(n_objects):
            size = np.random.randint(20, 50)
            x1 = np.random.randint(0, max(1, img_size - size))
            y1 = np.random.randint(0, max(1, img_size - size))
            x2 = x1 + size
            y2 = y1 + size

            shape_idx = np.random.randint(len(shapes))
            color_idx = np.random.randint(len(colors))
            shape = shapes[shape_idx]
            color_name, color = colors[color_idx]
            label = shape_idx * len(colors) + color_idx

            if shape == "circle":
                draw.ellipse([x1, y1, x2, y2], fill=color)
            elif shape == "square":
                draw.rectangle([x1, y1, x2, y2], fill=color)
            else:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                points = [(cx, y1), (x1, y2), (x2, y2)]
                draw.polygon(points, fill=color)

            sample_boxes.append([x1 / img_size, y1 / img_size, x2 / img_size, y2 / img_size])
            sample_labels.append(label)

        samples.append({
            "image": img,
            "boxes": np.array(sample_boxes, dtype=np.float32),
            "labels": np.array(sample_labels, dtype=np.int64),
        })

    return samples, n_classes


def compute_iou_metrics(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> dict:
    """Compute IoU and localization metrics."""
    ious = []
    center_errors = []
    size_errors = []

    for pred, gt in zip(pred_boxes, gt_boxes):
        # Convert center format to corner format
        pred_x1 = pred[0] - pred[2] / 2
        pred_y1 = pred[1] - pred[3] / 2
        pred_x2 = pred[0] + pred[2] / 2
        pred_y2 = pred[1] + pred[3] / 2

        gt_x1 = gt[0] - gt[2] / 2
        gt_y1 = gt[1] - gt[3] / 2
        gt_x2 = gt[0] + gt[2] / 2
        gt_y2 = gt[1] + gt[3] / 2

        inter_x1 = max(pred_x1, gt_x1)
        inter_y1 = max(pred_y1, gt_y1)
        inter_x2 = min(pred_x2, gt_x2)
        inter_y2 = min(pred_y2, gt_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union_area = pred_area + gt_area - inter_area

        iou = inter_area / (union_area + 1e-6)
        ious.append(iou)

        center_error = np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
        center_errors.append(center_error)

        size_error = (abs(pred[2] - gt[2]) / gt[2] + abs(pred[3] - gt[3]) / gt[3]) / 2
        size_errors.append(size_error)

    return {
        "mean_iou": float(np.mean(ious)),
        "iou_above_0.5": float(np.mean([1 if iou > 0.5 else 0 for iou in ious])),
        "iou_above_0.75": float(np.mean([1 if iou > 0.75 else 0 for iou in ious])),
        "center_error": float(np.mean(center_errors)),
        "size_error": float(np.mean(size_errors)),
    }


def compute_map(
    all_pred_boxes: list,
    all_pred_logits: list,
    all_gt_boxes: list,
    all_gt_labels: list,
    iou_threshold: float = 0.5,
    n_classes: int = 12,
) -> float:
    """Compute mean Average Precision at given IoU threshold."""
    from torchvision.ops import box_iou

    class_predictions = defaultdict(list)
    class_n_gt = defaultdict(int)

    for img_idx in range(len(all_pred_boxes)):
        pred_boxes = all_pred_boxes[img_idx]
        pred_logits = all_pred_logits[img_idx]
        gt_boxes = all_gt_boxes[img_idx]
        gt_labels = all_gt_labels[img_idx]

        if len(gt_boxes) == 0:
            continue

        if not isinstance(pred_boxes, torch.Tensor):
            pred_boxes = torch.tensor(pred_boxes)
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.tensor(gt_boxes)
        if not isinstance(gt_labels, torch.Tensor):
            gt_labels = torch.tensor(gt_labels)

        for label in gt_labels.tolist():
            class_n_gt[label] += 1

        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_classes = pred_probs[:, :-1].argmax(dim=-1)
        pred_scores = pred_probs[:, :-1].max(dim=-1)[0]

        keep = pred_scores > 0.3
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]

        if len(pred_boxes) == 0:
            continue

        matched_gt = set()
        for pred_idx in range(len(pred_boxes)):
            pred_box = pred_boxes[pred_idx:pred_idx+1]
            pred_class = pred_classes[pred_idx].item()
            pred_score = pred_scores[pred_idx].item()

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

            class_predictions[pred_class].append((pred_score, is_tp))

    aps = []
    for class_idx in range(n_classes):
        predictions = class_predictions[class_idx]
        n_gt = class_n_gt[class_idx]

        if n_gt == 0:
            continue

        predictions.sort(key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for score, is_tp in predictions:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / n_gt
            precisions.append(precision)
            recalls.append(recall)

        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            ap += max(prec_at_recall) if prec_at_recall else 0
        ap /= 11
        aps.append(ap)

    return np.mean(aps) if aps else 0


def create_visualization(
    images: list,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    metrics: dict,
) -> bytes:
    """Create visualization of spatial probe results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n_images = min(8, len(images))
    fig, axes = plt.subplots(2, n_images, figsize=(2.5 * n_images, 5))
    img_size = 224

    for i in range(n_images):
        # Ground truth
        axes[0, i].imshow(images[i])
        gt = gt_boxes[i]
        gt_rect = patches.Rectangle(
            ((gt[0] - gt[2] / 2) * img_size, (gt[1] - gt[3] / 2) * img_size),
            gt[2] * img_size, gt[3] * img_size,
            linewidth=2, edgecolor="green", facecolor="none",
        )
        axes[0, i].add_patch(gt_rect)
        if i == 0:
            axes[0, i].set_ylabel("Ground Truth", fontsize=10)
        axes[0, i].axis("off")

        # Prediction
        axes[1, i].imshow(images[i])
        pred = pred_boxes[i]
        pred_rect = patches.Rectangle(
            ((pred[0] - pred[2] / 2) * img_size, (pred[1] - pred[3] / 2) * img_size),
            pred[2] * img_size, pred[3] * img_size,
            linewidth=2, edgecolor="red", facecolor="none",
        )
        axes[1, i].add_patch(pred_rect)
        if i == 0:
            axes[1, i].set_ylabel("DINOv2 Pred", fontsize=10)
        axes[1, i].axis("off")

    plt.suptitle(
        f"DINOv2 Spatial Probe Results\n"
        f"Mean IoU: {metrics['mean_iou']:.3f} | Center Error: {metrics['center_error']:.3f}",
        fontsize=11,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e_p2_1_dinov2_spatial_analysis(runner: ExperimentRunner) -> dict:
    """Run DINOv2 spatial feature analysis.

    This implementation:
    1. Generates test images with known bounding boxes
    2. Extracts DINOv2-giant features
    3. Trains spatial probe for bbox prediction
    4. Trains detection probe for mAP evaluation
    5. Compares to VLM baseline (from Q2)

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-P2.1: DINOv2 Spatial Feature Analysis")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e_p2_1/stage": 0, "e_p2_1/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate datasets
    # =========================================================================
    print("\n[Stage 1/5] Generating test datasets...")

    train_images, train_boxes = generate_position_dataset(n_images=200)
    test_images, test_boxes = generate_position_dataset(n_images=50)

    detection_samples, n_classes = generate_detection_dataset(n_samples=200)

    print(f"  Position dataset: {len(train_images)} train, {len(test_images)} test")
    print(f"  Detection dataset: {len(detection_samples)} samples, {n_classes} classes")

    runner.log_metrics({
        "e_p2_1/stage": 1,
        "e_p2_1/progress": 0.1,
        "e_p2_1/n_train": len(train_images),
        "e_p2_1/n_test": len(test_images),
    })

    # =========================================================================
    # Stage 2: Load DINOv2 and extract features
    # =========================================================================
    print("\n[Stage 2/5] Loading DINOv2 and extracting features...")

    dinov2 = load_dinov2_model(device)

    print("  Extracting features for position dataset...")
    train_features = extract_dinov2_features(train_images, dinov2, device, runner, "train")
    test_features = extract_dinov2_features(test_images, dinov2, device, runner, "test")

    print(f"  Train features shape: {train_features.shape}")
    print(f"  Test features shape: {test_features.shape}")

    print("  Extracting features for detection dataset...")
    detection_images = [s["image"] for s in detection_samples]
    detection_features = extract_dinov2_features(detection_images, dinov2, device, runner, "detection")

    runner.log_metrics({
        "e_p2_1/stage": 2,
        "e_p2_1/progress": 0.35,
        "e_p2_1/feature_dim": train_features.shape[-1],
        "e_p2_1/n_patches": train_features.shape[1],
    })

    # Free model memory
    del dinov2
    torch.cuda.empty_cache()

    # =========================================================================
    # Stage 3: Train and evaluate spatial probe
    # =========================================================================
    print("\n[Stage 3/5] Training spatial probe...")

    train_features_t = train_features.to(device)
    test_features_t = test_features.to(device)
    train_boxes_t = torch.tensor(train_boxes, dtype=torch.float32).to(device)
    test_boxes_t = torch.tensor(test_boxes, dtype=torch.float32).to(device)

    spatial_probe = SpatialProbe(feature_dim=train_features.shape[-1]).to(device)
    optimizer = torch.optim.Adam(spatial_probe.parameters(), lr=1e-3)

    n_epochs = 100
    batch_size = 32

    for epoch in range(n_epochs):
        spatial_probe.train()
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_features_t), batch_size):
            batch_feat = train_features_t[i:i + batch_size]
            batch_boxes = train_boxes_t[i:i + batch_size]

            optimizer.zero_grad()
            pred_boxes = spatial_probe(batch_feat)
            loss = F.smooth_l1_loss(pred_boxes, batch_boxes)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 20 == 0:
            print(f"    Epoch {epoch}/{n_epochs}, Loss: {epoch_loss / n_batches:.4f}")
            runner.log_metrics({"e_p2_1/spatial_loss": epoch_loss / n_batches}, step=epoch)

    # Evaluate
    spatial_probe.eval()
    with torch.no_grad():
        pred_boxes = spatial_probe(test_features_t)

    spatial_metrics = compute_iou_metrics(pred_boxes.cpu().numpy(), test_boxes)

    print(f"  Spatial IoU: {spatial_metrics['mean_iou']:.4f}")
    print(f"  IoU > 0.5: {spatial_metrics['iou_above_0.5']:.1%}")
    print(f"  Center error: {spatial_metrics['center_error']:.4f}")

    runner.log_metrics({
        "e_p2_1/stage": 3,
        "e_p2_1/progress": 0.6,
        "e_p2_1/spatial_iou": spatial_metrics["mean_iou"],
        "e_p2_1/iou_above_0.5": spatial_metrics["iou_above_0.5"],
        "e_p2_1/center_error": spatial_metrics["center_error"],
    })

    # =========================================================================
    # Stage 4: Train and evaluate detection probe
    # =========================================================================
    print("\n[Stage 4/5] Training detection probe...")

    detection_features_t = detection_features.to(device)

    det_probe = DetectionProbe(
        input_dim=detection_features.shape[-1],
        num_queries=20,
        n_classes=n_classes,
    ).to(device)
    det_optimizer = torch.optim.AdamW(det_probe.parameters(), lr=1e-4, weight_decay=0.01)

    n_train = int(len(detection_features) * 0.8)
    train_det_features = detection_features_t[:n_train]
    val_det_features = detection_features_t[n_train:]

    from torchvision.ops import box_iou as torch_box_iou

    for epoch in range(100):
        det_probe.train()
        epoch_losses = []
        batch_size = 16

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_features = train_det_features[start:end]

            pred_boxes, pred_logits = det_probe(batch_features)

            total_loss = 0
            for i in range(len(batch_features)):
                sample = detection_samples[start + i]
                gt_boxes = torch.tensor(sample["boxes"]).to(device)
                gt_labels = torch.tensor(sample["labels"]).to(device)

                if len(gt_boxes) == 0:
                    continue

                ious = torch_box_iou(pred_boxes[i], gt_boxes)
                best_pred_idx = ious.argmax(dim=0)

                matched_pred_boxes = pred_boxes[i][best_pred_idx]
                box_loss = F.l1_loss(matched_pred_boxes, gt_boxes)

                matched_pred_logits = pred_logits[i][best_pred_idx]
                class_loss = F.cross_entropy(matched_pred_logits, gt_labels)

                total_loss += box_loss + class_loss

            if total_loss > 0:
                det_optimizer.zero_grad()
                total_loss.backward()
                det_optimizer.step()
                epoch_losses.append(total_loss.item())

        if (epoch + 1) % 20 == 0:
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            print(f"    Epoch {epoch+1}/100, Loss: {avg_loss:.4f}")

    # Evaluate mAP
    det_probe.eval()
    all_pred_boxes = []
    all_pred_logits = []
    all_gt_boxes = []
    all_gt_labels = []

    with torch.no_grad():
        for i in range(n_train, len(detection_features)):
            sample = detection_samples[i]
            feat = detection_features_t[i:i+1]
            pred_b, pred_l = det_probe(feat)

            all_pred_boxes.append(pred_b[0])
            all_pred_logits.append(pred_l[0])
            all_gt_boxes.append(torch.tensor(sample["boxes"]).to(device))
            all_gt_labels.append(torch.tensor(sample["labels"]).to(device))

    map_05 = compute_map(all_pred_boxes, all_pred_logits, all_gt_boxes, all_gt_labels, 0.5, n_classes)
    map_075 = compute_map(all_pred_boxes, all_pred_logits, all_gt_boxes, all_gt_labels, 0.75, n_classes)

    print(f"  mAP@0.5: {map_05:.4f}")
    print(f"  mAP@0.75: {map_075:.4f}")

    runner.log_metrics({
        "e_p2_1/stage": 4,
        "e_p2_1/progress": 0.85,
        "e_p2_1/map_05": map_05,
        "e_p2_1/map_075": map_075,
    })

    # =========================================================================
    # Stage 5: Save artifacts and create visualizations
    # =========================================================================
    print("\n[Stage 5/5] Creating visualizations and saving results...")

    viz_bytes = create_visualization(
        test_images[:8],
        test_boxes[:8],
        pred_boxes.cpu().numpy()[:8],
        spatial_metrics,
    )
    viz_path = runner.results.save_artifact("dinov2_spatial_probe.png", viz_bytes)

    # Save metrics
    results_data = {
        "spatial_metrics": {
            "mean_iou": spatial_metrics["mean_iou"],
            "iou_above_0.5": spatial_metrics["iou_above_0.5"],
            "iou_above_0.75": spatial_metrics["iou_above_0.75"],
            "center_error": spatial_metrics["center_error"],
            "size_error": spatial_metrics["size_error"],
        },
        "detection_metrics": {
            "mAP@0.5": float(map_05),
            "mAP@0.75": float(map_075),
        },
        "vlm_baseline_comparison": {
            "vlm_bbox_iou": 0.104,  # From Q2 results
            "vlm_map_05": 0.001,  # From Q2 results
            "dinov2_bbox_iou": spatial_metrics["mean_iou"],
            "dinov2_map_05": float(map_05),
            "improvement_factor_iou": spatial_metrics["mean_iou"] / 0.104,
            "improvement_factor_map": float(map_05) / 0.001 if map_05 > 0 else float("inf"),
        },
        "feature_info": {
            "n_patches": int(train_features.shape[1]),
            "feature_dim": int(train_features.shape[-1]),
        },
    }
    data_path = runner.results.save_json_artifact("dinov2_spatial_analysis.json", results_data)

    runner.log_metrics({
        "e_p2_1/stage": 5,
        "e_p2_1/progress": 1.0,
        "spatial_iou": spatial_metrics["mean_iou"],
        "mAP": map_05,
    })

    # =========================================================================
    # Determine finding
    # =========================================================================
    iou_target = 0.70
    map_target = 0.40

    if spatial_metrics["mean_iou"] > iou_target and map_05 > map_target:
        finding = (
            f"DINOv2 achieves excellent spatial preservation "
            f"(Spatial IoU={spatial_metrics['mean_iou']:.3f}, mAP@0.5={map_05:.3f}). "
            f"This is {spatial_metrics['mean_iou'] / 0.104:.1f}x better than VLM baseline (IoU=0.104). "
            f"DINOv2 is suitable as the spatial encoder for hybrid architecture."
        )
    elif spatial_metrics["mean_iou"] > 0.60 or map_05 > 0.30:
        finding = (
            f"DINOv2 achieves good spatial preservation "
            f"(Spatial IoU={spatial_metrics['mean_iou']:.3f}, mAP@0.5={map_05:.3f}). "
            f"Performance is significantly better than VLM baseline. "
            f"Proceeding with DINOv2 as spatial encoder, may need optimization."
        )
    else:
        finding = (
            f"DINOv2 spatial preservation is lower than expected "
            f"(Spatial IoU={spatial_metrics['mean_iou']:.3f}, mAP@0.5={map_05:.3f}). "
            f"Consider using different DINOv2 layers or alternative spatial encoders."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "spatial_iou": spatial_metrics["mean_iou"],
            "iou_above_0.5": spatial_metrics["iou_above_0.5"],
            "center_error": spatial_metrics["center_error"],
            "mAP@0.5": float(map_05),
            "mAP@0.75": float(map_075),
            "vlm_comparison_iou_improvement": spatial_metrics["mean_iou"] / 0.104,
        },
        "artifacts": [viz_path, data_path],
    }
