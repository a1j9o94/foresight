"""E-Q2.3: LLM Layer-wise Decay Analysis

Objective: Track spatial information degradation through LLM layers.

Protocol:
1. Extract hidden states from LLM layers 1, 7, 14, 21, 28
2. Train spatial probes at each layer
3. Plot information decay curve

Expected Decay Pattern:
Layer 1:  [|||||||||||||||||||] 95% spatial info
Layer 7:  [|||||||||||||||||  ] 85% spatial info
Layer 14: [||||||||||||||     ] 70% spatial info
Layer 21: [|||||||||||        ] 55% spatial info
Layer 28: [||||||||           ] 40% spatial info

Key Question: At which layer does spatial accuracy drop below 0.7 IoU?
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
from torchvision.ops import box_iou

from runner import ExperimentRunner


class SyntheticBboxDataset:
    """Synthetic dataset with known bounding boxes."""

    def __init__(self, n_samples: int = 200, img_size: int = 448):
        self.n_samples = n_samples
        self.img_size = img_size
        self.shapes = ["circle", "square", "triangle"]
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        np.random.seed(42)
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        for _ in range(self.n_samples):
            n_objects = np.random.randint(1, 4)
            img = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            boxes = []
            for _ in range(n_objects):
                size = np.random.randint(30, 80)
                x1 = np.random.randint(0, self.img_size - size)
                y1 = np.random.randint(0, self.img_size - size)
                x2 = x1 + size
                y2 = y1 + size

                shape = np.random.choice(self.shapes)
                color = self.colors[np.random.randint(len(self.colors))]

                if shape == "circle":
                    draw.ellipse([x1, y1, x2, y2], fill=color)
                elif shape == "square":
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                else:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    points = [(cx, y1), (x1, y2), (x2, y2)]
                    draw.polygon(points, fill=color)

                boxes.append([
                    x1 / self.img_size, y1 / self.img_size,
                    x2 / self.img_size, y2 / self.img_size,
                ])

            while len(boxes) < 3:
                boxes.append([0.0, 0.0, 0.0, 0.0])

            samples.append((img, np.array(boxes[:3], dtype=np.float32), n_objects))

        return samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img, boxes, n_objects = self.samples[idx]
        return {"image": img, "boxes": boxes, "n_objects": n_objects}


class BoundingBoxProbe(nn.Module):
    """Linear probe for bounding box prediction."""

    def __init__(self, input_dim: int, max_objects: int = 3):
        super().__init__()
        self.max_objects = max_objects
        self.fc = nn.Linear(input_dim, max_objects * 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)
        boxes = self.fc(features).view(-1, self.max_objects, 4)
        return boxes.sigmoid()


def extract_layerwise_features(
    model, processor, images: list, layer_indices: list, device, runner: ExperimentRunner
) -> dict[int, torch.Tensor]:
    """Extract features from multiple LLM layers.

    Args:
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        images: List of PIL Images
        layer_indices: Which LLM layers to extract (0-indexed, 0-27 for 28 layers)
        device: Torch device
        runner: ExperimentRunner for logging

    Returns:
        Dictionary mapping layer index to feature tensor [n_images, hidden_dim]
    """
    # Initialize storage
    features_by_layer = {idx: [] for idx in layer_indices}

    with torch.no_grad():
        for i, img in enumerate(images):
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

            # Forward pass with all hidden states
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # outputs.hidden_states is tuple of length n_layers + 1
            # Index 0 is embedding layer, indices 1-28 are transformer layers
            for layer_idx in layer_indices:
                # Add 1 because index 0 is the embedding layer
                hidden_state = outputs.hidden_states[layer_idx + 1]

                # Pool across sequence dimension
                # Shape: [1, seq_len, hidden_dim] -> [1, hidden_dim]
                pooled = hidden_state.mean(dim=1).float().cpu()
                features_by_layer[layer_idx].append(pooled)

            if (i + 1) % 50 == 0:
                progress = (i + 1) / len(images)
                runner.log_metrics({"e_q2_3/extraction_progress": progress})
                print(f"    Extracted {i + 1}/{len(images)} images")

    # Stack features for each layer
    for layer_idx in layer_indices:
        features_by_layer[layer_idx] = torch.cat(features_by_layer[layer_idx], dim=0)

    return features_by_layer


def train_and_evaluate_probe(
    features: torch.Tensor,
    boxes: torch.Tensor,
    n_objects: torch.Tensor,
    epochs: int = 80,
    lr: float = 1e-3,
) -> dict:
    """Train probe and return evaluation metrics."""
    device = features.device
    input_dim = features.shape[-1]

    probe = BoundingBoxProbe(input_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # Train/val split
    n_train = int(len(features) * 0.8)
    train_features, val_features = features[:n_train], features[n_train:]
    train_boxes, val_boxes = boxes[:n_train], boxes[n_train:]
    train_n_obj, val_n_obj = n_objects[:n_train], n_objects[n_train:]

    # Training loop
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        pred_boxes = probe(train_features)

        mask = torch.arange(3).unsqueeze(0).expand(len(train_features), -1).to(device)
        mask = mask < train_n_obj.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand(-1, -1, 4)

        loss = F.smooth_l1_loss(pred_boxes[mask], train_boxes[mask], reduction="mean")
        loss.backward()
        optimizer.step()

    # Final evaluation
    probe.eval()
    with torch.no_grad():
        val_pred = probe(val_features)
        all_ious = compute_batch_iou(val_pred, val_boxes, val_n_obj)

    mean_iou = all_ious.mean().item()
    iou_above_07 = (all_ious > 0.7).float().mean().item()
    iou_above_05 = (all_ious > 0.5).float().mean().item()

    return {
        "mean_iou": mean_iou,
        "iou_above_0.7": iou_above_07,
        "iou_above_0.5": iou_above_05,
    }


def compute_batch_iou(
    pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, n_objects: torch.Tensor
) -> torch.Tensor:
    """Compute IoU for batched predictions."""
    batch_size = pred_boxes.shape[0]
    ious = []

    for i in range(batch_size):
        n = n_objects[i].item()
        if n == 0:
            continue

        pred = pred_boxes[i, :n]
        gt = gt_boxes[i, :n]

        iou_matrix = box_iou(pred, gt)
        best_ious = iou_matrix.max(dim=1)[0]
        ious.extend(best_ious.tolist())

    return torch.tensor(ious) if ious else torch.tensor([0.0])


def create_decay_curve_visualization(
    layer_results: dict, layer_labels: list
) -> bytes:
    """Create visualization of information decay across layers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: IoU decay curve
    ax1 = axes[0]
    layers = list(layer_results.keys())
    ious = [layer_results[l]["mean_iou"] for l in layers]

    ax1.plot(range(len(layers)), ious, "b-o", linewidth=2, markersize=8, label="Mean IoU")
    ax1.axhline(y=0.7, color="r", linestyle="--", label="Target threshold (0.7)")
    ax1.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="Minimum threshold (0.5)")

    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Spatial IoU", fontsize=12)
    ax1.set_title("Spatial Information Decay Through LLM Layers", fontsize=14)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layer_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Right plot: Bar chart with breakdown
    ax2 = axes[1]
    x = np.arange(len(layers))
    width = 0.35

    iou_07 = [layer_results[l]["iou_above_0.7"] for l in layers]
    iou_05 = [layer_results[l]["iou_above_0.5"] for l in layers]

    ax2.bar(x - width/2, iou_07, width, label="IoU > 0.7", color="green", alpha=0.7)
    ax2.bar(x + width/2, iou_05, width, label="IoU > 0.5", color="blue", alpha=0.7)

    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Fraction of Predictions", fontsize=12)
    ax2.set_title("Detection Quality by Layer", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_retention_heatmap(layer_results: dict, layer_labels: list) -> bytes:
    """Create heatmap showing information retention."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))

    # Create data matrix
    metrics = ["Mean IoU", "IoU > 0.7", "IoU > 0.5"]
    layers = list(layer_results.keys())
    data = np.array([
        [layer_results[l]["mean_iou"] for l in layers],
        [layer_results[l]["iou_above_0.7"] for l in layers],
        [layer_results[l]["iou_above_0.5"] for l in layers],
    ])

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layer_labels)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)

    # Add value annotations
    for i in range(len(metrics)):
        for j in range(len(layers)):
            value = data[i, j]
            color = "white" if value < 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)

    ax.set_title("Information Retention Heatmap", fontsize=14)
    plt.colorbar(im, ax=ax, label="Score")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e_q2_3_llm_layer_analysis(runner: ExperimentRunner) -> dict:
    """Run layer-wise information decay analysis.

    This tracks how spatial information degrades through the LLM layers,
    helping identify the optimal extraction point for video conditioning.

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-Q2.3: LLM Layer-wise Information Decay Analysis")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"e_q2_3/stage": 0, "e_q2_3/progress": 0.0})

    # =========================================================================
    # Stage 1: Create dataset
    # =========================================================================
    print("\n[Stage 1/5] Creating synthetic dataset...")

    dataset = SyntheticBboxDataset(n_samples=150, img_size=448)
    print(f"  Created {len(dataset)} samples")

    runner.log_metrics({"e_q2_3/stage": 1, "e_q2_3/progress": 0.1})

    # =========================================================================
    # Stage 2: Load model
    # =========================================================================
    print("\n[Stage 2/5] Loading Qwen2.5-VL...")

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

    # Determine number of layers - Qwen2.5-VL uses model.model which is a Qwen2_5_VLModel
    # The text model layers are in model.model.layers
    try:
        n_layers = len(model.model.model.layers)
    except AttributeError:
        # Fallback: estimate from config
        n_layers = getattr(model.config, 'num_hidden_layers', 28)
    print(f"  Model has {n_layers} transformer layers")

    runner.log_metrics({"e_q2_3/stage": 2, "e_q2_3/progress": 0.2, "e_q2_3/n_layers": n_layers})

    # =========================================================================
    # Stage 3: Extract features from multiple layers
    # =========================================================================
    print("\n[Stage 3/5] Extracting features from multiple LLM layers...")

    # Sample layers evenly (0-indexed)
    # Qwen2.5-VL-7B has 28 layers (0-27)
    layer_indices = [0, 6, 13, 20, 27]  # ~evenly spaced
    layer_labels = ["L1", "L7", "L14", "L21", "L28"]

    images = [dataset[i]["image"] for i in range(len(dataset))]
    boxes = torch.tensor(np.array([dataset[i]["boxes"] for i in range(len(dataset))]))
    n_objects = torch.tensor([dataset[i]["n_objects"] for i in range(len(dataset))])

    features_by_layer = extract_layerwise_features(
        model, processor, images, layer_indices, device, runner
    )

    print(f"  Extracted features from layers: {layer_indices}")
    for idx in layer_indices:
        print(f"    Layer {idx}: shape {features_by_layer[idx].shape}")

    runner.log_metrics({"e_q2_3/stage": 3, "e_q2_3/progress": 0.5})

    # Free GPU memory before training probes
    del model
    del processor
    torch.cuda.empty_cache()

    # =========================================================================
    # Stage 4: Train probes and evaluate each layer
    # =========================================================================
    print("\n[Stage 4/5] Training probes for each layer...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxes = boxes.to(device)
    n_objects = n_objects.to(device)

    layer_results = {}
    for i, layer_idx in enumerate(layer_indices):
        print(f"\n  Training probe for Layer {layer_idx} ({layer_labels[i]})...")
        features = features_by_layer[layer_idx].to(device)

        metrics = train_and_evaluate_probe(features, boxes, n_objects, epochs=80)
        layer_results[layer_idx] = metrics

        print(f"    IoU: {metrics['mean_iou']:.4f}, >0.7: {metrics['iou_above_0.7']:.2f}, >0.5: {metrics['iou_above_0.5']:.2f}")

        # Log metrics for this layer
        runner.log_metrics({
            f"e_q2_3/layer_{layer_idx}_iou": metrics["mean_iou"],
            f"e_q2_3/layer_{layer_idx}_iou_above_07": metrics["iou_above_0.7"],
        })

    runner.log_metrics({"e_q2_3/stage": 4, "e_q2_3/progress": 0.85})

    # =========================================================================
    # Stage 5: Analysis and visualization
    # =========================================================================
    print("\n[Stage 5/5] Creating visualizations and analysis...")

    # Find optimal layer (last layer with IoU > 0.7)
    optimal_layer = None
    threshold_layer = None  # First layer where IoU drops below 0.7

    for layer_idx in layer_indices:
        if layer_results[layer_idx]["mean_iou"] >= 0.7:
            optimal_layer = layer_idx
        elif threshold_layer is None:
            threshold_layer = layer_idx

    # Calculate decay rate
    first_iou = layer_results[layer_indices[0]]["mean_iou"]
    last_iou = layer_results[layer_indices[-1]]["mean_iou"]
    total_decay = first_iou - last_iou
    decay_rate_per_layer = total_decay / (layer_indices[-1] - layer_indices[0])

    print(f"\n  Analysis Results:")
    print(f"    First layer IoU: {first_iou:.4f}")
    print(f"    Last layer IoU: {last_iou:.4f}")
    print(f"    Total decay: {total_decay:.4f}")
    print(f"    Decay rate per layer: {decay_rate_per_layer:.4f}")
    if optimal_layer is not None:
        print(f"    Optimal extraction layer: {optimal_layer} (last with IoU >= 0.7)")
    if threshold_layer is not None:
        print(f"    Threshold crossed at layer: {threshold_layer}")

    # Create visualizations
    decay_vis = create_decay_curve_visualization(layer_results, layer_labels)
    decay_path = runner.results.save_artifact("layer_decay_curve.png", decay_vis)

    heatmap_vis = create_retention_heatmap(layer_results, layer_labels)
    heatmap_path = runner.results.save_artifact("layer_retention_heatmap.png", heatmap_vis)

    runner.log_metrics({
        "e_q2_3/stage": 5,
        "e_q2_3/progress": 1.0,
        "e_q2_3/first_layer_iou": first_iou,
        "e_q2_3/last_layer_iou": last_iou,
        "e_q2_3/total_decay": total_decay,
        "e_q2_3/decay_rate_per_layer": decay_rate_per_layer,
    })

    # Save detailed results
    results_data = {
        "layer_analysis": {
            layer_idx: {
                "label": layer_labels[i],
                **layer_results[layer_idx]
            }
            for i, layer_idx in enumerate(layer_indices)
        },
        "summary": {
            "n_layers_analyzed": len(layer_indices),
            "layer_indices": layer_indices,
            "first_layer_iou": float(first_iou),
            "last_layer_iou": float(last_iou),
            "total_decay": float(total_decay),
            "decay_rate_per_layer": float(decay_rate_per_layer),
            "optimal_layer": optimal_layer,
            "threshold_layer": threshold_layer,
        },
    }
    data_path = runner.results.save_json_artifact("layer_analysis.json", results_data)

    # =========================================================================
    # Interpret results
    # =========================================================================
    if optimal_layer is not None and optimal_layer >= 13:
        finding = (
            f"Spatial information persists deep into the LLM (optimal layer: {optimal_layer}, "
            f"IoU: {layer_results[optimal_layer]['mean_iou']:.3f}). Can extract from mid-to-late "
            f"layers while maintaining spatial accuracy. Decay rate: {decay_rate_per_layer:.4f}/layer."
        )
    elif optimal_layer is not None:
        finding = (
            f"Spatial information degrades relatively quickly (optimal layer: {optimal_layer}, "
            f"IoU: {layer_results[optimal_layer]['mean_iou']:.3f}). Should extract from early LLM "
            f"layers for best spatial accuracy. Decay rate: {decay_rate_per_layer:.4f}/layer."
        )
    else:
        finding = (
            f"Spatial information is lost before entering LLM. First layer IoU: {first_iou:.3f}. "
            f"Must extract from ViT (pre-merge or post-merge) for spatial tasks."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "first_layer_iou": float(first_iou),
            "last_layer_iou": float(last_iou),
            "total_decay": float(total_decay),
            "decay_rate_per_layer": float(decay_rate_per_layer),
            "optimal_layer": optimal_layer if optimal_layer is not None else -1,
            **{f"layer_{layer_idx}_iou": layer_results[layer_idx]["mean_iou"]
               for layer_idx in layer_indices},
        },
        "artifacts": [decay_path, heatmap_path, data_path],
    }
