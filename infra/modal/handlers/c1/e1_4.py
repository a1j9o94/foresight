"""E1.4: Spatial Information Test

Objective: Directly measure whether object positions can be recovered from latents.

Protocol:
1. Create test set with known object positions (bounding boxes)
2. Train a position regression head on latents
3. Measure localization accuracy (IoU, center error)

Success Metrics:
- Spatial IoU > 0.6 (target: > 0.75)
- Center error < 0.05 (5% of image)
- Size error < 0.1

This experiment tests whether VLM latents preserve precise spatial information
needed for video reconstruction and verification.
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

from runner import ExperimentRunner


class PositionRegressionHead(nn.Module):
    """Predict bounding box from pooled latents."""

    def __init__(self, latent_dim: int = 3584, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # x_center, y_center, width, height
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, latent_dim] pooled latent vectors

        Returns:
            [B, 4] bounding box predictions (x_center, y_center, width, height)
                   all normalized to [0, 1]
        """
        return self.mlp(latents)


def e1_4_spatial_information(runner: ExperimentRunner) -> dict:
    """Run spatial information test sub-experiment.

    This implementation:
    1. Generates images with objects at known positions
    2. Extracts VLM latents
    3. Trains a position regression head
    4. Evaluates localization accuracy (IoU, center error, size error)

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E1.4: Spatial Information Test")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e1_4/stage": 0, "e1_4/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate images with known positions
    # =========================================================================
    print("\n[Stage 1/5] Generating images with known bounding boxes...")

    train_images, train_boxes = generate_position_dataset(n_images=200)
    test_images, test_boxes = generate_position_dataset(n_images=50)

    print(f"  Generated {len(train_images)} training images")
    print(f"  Generated {len(test_images)} test images")

    runner.log_metrics({
        "e1_4/stage": 1,
        "e1_4/progress": 0.1,
        "e1_4/n_train": len(train_images),
        "e1_4/n_test": len(test_images),
    })

    # =========================================================================
    # Stage 2: Extract latents
    # =========================================================================
    print("\n[Stage 2/5] Extracting VLM latents...")

    train_latents = extract_latents_batch(train_images, runner, "train")
    test_latents = extract_latents_batch(test_images, runner, "test")

    print(f"  Train latents shape: {train_latents.shape}")
    print(f"  Test latents shape: {test_latents.shape}")

    runner.log_metrics({
        "e1_4/stage": 2,
        "e1_4/progress": 0.4,
        "e1_4/latent_dim": train_latents.shape[-1],
    })

    # =========================================================================
    # Stage 3: Train position regression head
    # =========================================================================
    print("\n[Stage 3/5] Training position regression head...")

    train_latents_t = torch.tensor(train_latents, dtype=torch.float32).to(device)
    train_boxes_t = torch.tensor(train_boxes, dtype=torch.float32).to(device)
    test_latents_t = torch.tensor(test_latents, dtype=torch.float32).to(device)
    test_boxes_t = torch.tensor(test_boxes, dtype=torch.float32).to(device)

    head = PositionRegressionHead(latent_dim=train_latents.shape[-1]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    n_epochs = 100
    batch_size = 32
    best_loss = float("inf")

    for epoch in range(n_epochs):
        head.train()
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_latents_t), batch_size):
            batch_latents = train_latents_t[i : i + batch_size]
            batch_boxes = train_boxes_t[i : i + batch_size]

            optimizer.zero_grad()

            # Forward
            pred_boxes = head(batch_latents)

            # Smooth L1 loss (more robust than MSE for regression)
            loss = F.smooth_l1_loss(pred_boxes, batch_boxes)

            # Backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if epoch % 20 == 0:
            print(f"    Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}")
            runner.log_metrics({
                "e1_4/train_loss": avg_loss,
                "e1_4/epoch": epoch,
            }, step=epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f"  Training complete. Best loss: {best_loss:.4f}")

    runner.log_metrics({"e1_4/stage": 3, "e1_4/progress": 0.7})

    # =========================================================================
    # Stage 4: Evaluate localization accuracy
    # =========================================================================
    print("\n[Stage 4/5] Evaluating localization accuracy...")

    head.eval()
    with torch.no_grad():
        pred_boxes = head(test_latents_t)

    # Compute metrics
    metrics = compute_localization_metrics(
        pred_boxes.cpu().numpy(),
        test_boxes,
    )

    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  IoU > 0.5: {metrics['iou_above_0.5']:.1%}")
    print(f"  IoU > 0.75: {metrics['iou_above_0.75']:.1%}")
    print(f"  Center error: {metrics['center_error']:.4f}")
    print(f"  Size error: {metrics['size_error']:.4f}")

    runner.log_metrics({
        "e1_4/stage": 4,
        "e1_4/progress": 0.9,
        "e1_4/mean_iou": metrics["mean_iou"],
        "e1_4/iou_above_0.5": metrics["iou_above_0.5"],
        "e1_4/iou_above_0.75": metrics["iou_above_0.75"],
        "e1_4/center_error": metrics["center_error"],
        "e1_4/size_error": metrics["size_error"],
        "spatial_iou": metrics["mean_iou"],  # For success criteria
    })

    # =========================================================================
    # Stage 5: Save artifacts
    # =========================================================================
    print("\n[Stage 5/5] Saving artifacts...")

    # Create visualization
    viz_bytes = create_localization_visualization(
        test_images[:8],
        test_boxes[:8],
        pred_boxes.cpu().numpy()[:8],
        metrics,
    )
    viz_path = runner.results.save_artifact("localization_results.png", viz_bytes)

    # Save detailed metrics
    metrics_data = {
        "mean_iou": float(metrics["mean_iou"]),
        "iou_above_0.5": float(metrics["iou_above_0.5"]),
        "iou_above_0.75": float(metrics["iou_above_0.75"]),
        "center_error": float(metrics["center_error"]),
        "size_error": float(metrics["size_error"]),
        "all_ious": [float(x) for x in metrics["all_ious"]],
        "n_test": len(test_images),
        "latent_dim": int(train_latents.shape[-1]),
    }
    metrics_path = runner.results.save_json_artifact("localization_metrics.json", metrics_data)

    runner.log_metrics({"e1_4/stage": 5, "e1_4/progress": 1.0})

    # =========================================================================
    # Determine finding
    # =========================================================================
    iou_target = 0.6
    iou_ideal = 0.75

    if metrics["mean_iou"] > iou_ideal:
        finding = (
            f"Excellent spatial information preservation (IoU={metrics['mean_iou']:.3f} > {iou_ideal}). "
            f"VLM latents accurately encode object positions. "
            f"Center error: {metrics['center_error']:.3f}, Size error: {metrics['size_error']:.3f}."
        )
    elif metrics["mean_iou"] > iou_target:
        finding = (
            f"Good spatial information preservation (IoU={metrics['mean_iou']:.3f} > {iou_target}). "
            f"VLM latents encode object positions with acceptable accuracy. "
            f"Center error: {metrics['center_error']:.3f}, Size error: {metrics['size_error']:.3f}."
        )
    else:
        finding = (
            f"Insufficient spatial information (IoU={metrics['mean_iou']:.3f} < {iou_target}). "
            f"VLM latents may not preserve precise positions needed for reconstruction. "
            f"Center error: {metrics['center_error']:.3f}, Size error: {metrics['size_error']:.3f}. "
            f"Consider alternative approaches or pre-merge latents."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "spatial_iou": float(metrics["mean_iou"]),
            "iou_above_0.5": float(metrics["iou_above_0.5"]),
            "iou_above_0.75": float(metrics["iou_above_0.75"]),
            "center_error": float(metrics["center_error"]),
            "size_error": float(metrics["size_error"]),
        },
        "artifacts": [viz_path, metrics_path],
    }


def generate_position_dataset(n_images: int = 200) -> tuple[list, np.ndarray]:
    """Generate images with objects at known positions.

    Creates images with single colored shapes at random positions.
    Returns both images and their bounding boxes.

    Args:
        n_images: Number of images to generate

    Returns:
        Tuple of (images, boxes) where boxes is [N, 4] array of
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    shapes = ["circle", "square", "triangle"]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    images = []
    boxes = []
    img_size = 224

    np.random.seed(42)

    for i in range(n_images):
        shape = shapes[i % len(shapes)]
        color = colors[i % len(colors)]

        # Random position and size (in pixels)
        size = np.random.randint(25, 60)
        margin = size + 10
        cx = np.random.randint(margin, img_size - margin)
        cy = np.random.randint(margin, img_size - margin)

        # Create image
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

        # Store normalized bounding box
        # (x_center, y_center, width, height) all in [0, 1]
        box = [
            cx / img_size,
            cy / img_size,
            (2 * size) / img_size,
            (2 * size) / img_size,
        ]
        boxes.append(box)

    return images, np.array(boxes, dtype=np.float32)


def extract_latents_batch(images: list, runner: ExperimentRunner, prefix: str) -> np.ndarray:
    """Extract latents from VLM for a batch of images."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"  Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
    )

    print(f"  Model loaded on {model.device}")
    print(f"  Extracting latents from {len(images)} images...")

    latents_list = []
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            for img in batch:
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
                ).to(model.device)

                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

                hidden_states = outputs.hidden_states[-1]
                latent = hidden_states[0].float().cpu().numpy()
                latent_pooled = latent.mean(axis=0, keepdims=True)
                latents_list.append(latent_pooled)

            progress = min(1.0, (i + batch_size) / len(images))
            runner.log_metrics({f"e1_4/{prefix}_extraction_progress": progress})
            print(f"    Processed {min(i + batch_size, len(images))}/{len(images)} images")

    latents = np.concatenate(latents_list, axis=0)

    del model
    del processor
    torch.cuda.empty_cache()

    return latents


def compute_localization_metrics(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> dict:
    """Compute localization metrics.

    Args:
        pred_boxes: Predicted boxes [N, 4] (x_c, y_c, w, h) normalized
        gt_boxes: Ground truth boxes [N, 4] (x_c, y_c, w, h) normalized

    Returns:
        Dict with IoU, center error, size error metrics
    """
    ious = []
    center_errors = []
    size_errors = []

    for pred, gt in zip(pred_boxes, gt_boxes):
        # Convert center format to corner format for IoU
        pred_x1 = pred[0] - pred[2] / 2
        pred_y1 = pred[1] - pred[3] / 2
        pred_x2 = pred[0] + pred[2] / 2
        pred_y2 = pred[1] + pred[3] / 2

        gt_x1 = gt[0] - gt[2] / 2
        gt_y1 = gt[1] - gt[3] / 2
        gt_x2 = gt[0] + gt[2] / 2
        gt_y2 = gt[1] + gt[3] / 2

        # Compute IoU
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

        # Center error (Euclidean distance)
        center_error = np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
        center_errors.append(center_error)

        # Size error (relative)
        size_error = abs(pred[2] - gt[2]) / gt[2] + abs(pred[3] - gt[3]) / gt[3]
        size_errors.append(size_error / 2)  # Average of width and height errors

    return {
        "mean_iou": float(np.mean(ious)),
        "all_ious": ious,
        "iou_above_0.5": float(np.mean([1 if iou > 0.5 else 0 for iou in ious])),
        "iou_above_0.75": float(np.mean([1 if iou > 0.75 else 0 for iou in ious])),
        "center_error": float(np.mean(center_errors)),
        "size_error": float(np.mean(size_errors)),
    }


def create_localization_visualization(
    images: list,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    metrics: dict,
) -> bytes:
    """Create visualization of predicted vs ground truth bounding boxes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n_images = len(images)
    fig, axes = plt.subplots(2, n_images, figsize=(2.5 * n_images, 5))

    img_size = 224

    for i in range(n_images):
        # Show image with GT box
        axes[0, i].imshow(images[i])
        gt = gt_boxes[i]
        gt_rect = patches.Rectangle(
            ((gt[0] - gt[2] / 2) * img_size, (gt[1] - gt[3] / 2) * img_size),
            gt[2] * img_size,
            gt[3] * img_size,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
            label="GT" if i == 0 else "",
        )
        axes[0, i].add_patch(gt_rect)
        if i == 0:
            axes[0, i].set_ylabel("Ground Truth", fontsize=10)
        axes[0, i].axis("off")

        # Show image with predicted box
        axes[1, i].imshow(images[i])
        pred = pred_boxes[i]
        pred_rect = patches.Rectangle(
            ((pred[0] - pred[2] / 2) * img_size, (pred[1] - pred[3] / 2) * img_size),
            pred[2] * img_size,
            pred[3] * img_size,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="Pred" if i == 0 else "",
        )
        axes[1, i].add_patch(pred_rect)

        # Compute IoU for this image
        pred_x1 = pred[0] - pred[2] / 2
        pred_y1 = pred[1] - pred[3] / 2
        pred_x2 = pred[0] + pred[2] / 2
        pred_y2 = pred[1] + pred[3] / 2
        gt_x1 = gt[0] - gt[2] / 2
        gt_y1 = gt[1] - gt[3] / 2
        gt_x2 = gt[0] + gt[2] / 2
        gt_y2 = gt[1] + gt[3] / 2
        inter_x1, inter_y1 = max(pred_x1, gt_x1), max(pred_y1, gt_y1)
        inter_x2, inter_y2 = min(pred_x2, gt_x2), min(pred_y2, gt_y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1) + (gt_x2 - gt_x1) * (gt_y2 - gt_y1) - inter_area
        iou = inter_area / (union_area + 1e-6)

        axes[1, i].set_title(f"IoU: {iou:.2f}", fontsize=9)
        if i == 0:
            axes[1, i].set_ylabel("Predicted", fontsize=10)
        axes[1, i].axis("off")

    plt.suptitle(
        f"Spatial Localization Results\n"
        f"Mean IoU: {metrics['mean_iou']:.3f} | Center Error: {metrics['center_error']:.3f} | "
        f"Size Error: {metrics['size_error']:.3f}",
        fontsize=11,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
