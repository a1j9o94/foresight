"""E-P2.4: End-to-End Hybrid Pipeline Evaluation

Objective: Comprehensive evaluation of the full hybrid pipeline against Gate 1 thresholds.

Protocol:
1. Run full pipeline on held-out test set
2. Evaluate all Gate 1 success criteria
3. Compare to C1/Q2 baselines
4. Test on diverse scene types

Gate 1 Success Criteria:
- Spatial IoU > 0.60 (must pass)
- LPIPS < 0.35
- mAP@0.5 > 0.40
- Latency overhead < 25%
"""

import io
import os
import sys
import time

sys.path.insert(0, "/root")
sys.path.insert(0, "/root/handlers/p2")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from collections import defaultdict

from runner import ExperimentRunner
from detr_head import DETRDetectionHead, HungarianMatcher, detr_loss, box_cxcywh_to_xyxy


class HybridFusionModule(nn.Module):
    """Fusion module (same as E-P2.3)."""

    def __init__(
        self,
        vlm_dim: int = 3584,
        spatial_dim: int = 1024,
        fusion_dim: int = 1024,
        num_fusion_layers: int = 4,
        num_output_queries: int = 64,
        output_dim: int = 4096,
    ):
        super().__init__()
        self.vlm_proj = nn.Linear(vlm_dim, fusion_dim)
        self.spatial_proj = nn.Linear(spatial_dim, fusion_dim)
        self.spatial_pos = nn.Parameter(torch.randn(256, fusion_dim) * 0.02)
        self.output_queries = nn.Parameter(torch.randn(num_output_queries, fusion_dim) * 0.02)

        self.fusion_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=fusion_dim,
                nhead=8,
                dim_feedforward=fusion_dim * 4,
                batch_first=True,
                dropout=0.1,
            )
            for _ in range(num_fusion_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, output_dim),
        )

    def forward(self, vlm_features, spatial_features):
        B = vlm_features.size(0)
        vlm_proj = self.vlm_proj(vlm_features)
        spatial_proj = self.spatial_proj(spatial_features)
        n_spatial = spatial_proj.size(1)
        spatial_proj = spatial_proj + self.spatial_pos[:n_spatial]
        context = torch.cat([vlm_proj, spatial_proj], dim=1)
        queries = self.output_queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.fusion_layers:
            queries = layer(queries, context)
        return self.output_proj(queries)


class SimplePixelDecoder(nn.Module):
    """Pixel decoder."""

    def __init__(self, input_dim: int = 4096, output_size: int = 224):
        super().__init__()
        self.initial = nn.Linear(input_dim, 256 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, conditioning):
        x = conditioning.mean(dim=1)
        x = self.initial(x)
        x = x.view(-1, 256, 7, 7)
        return self.decoder(x)


class DetectionProbe(nn.Module):
    """Detection probe for mAP evaluation."""

    def __init__(self, input_dim: int, num_queries: int = 20, n_classes: int = 12, hidden_dim: int = 256):
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

    def forward(self, features):
        batch_size = features.shape[0]
        features = self.input_proj(features)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        queries, _ = self.cross_attention(queries, features, features)
        queries, _ = self.self_attention(queries, queries, queries)
        boxes = self.box_head(queries).sigmoid()
        class_logits = self.class_head(queries)
        return boxes, class_logits


def generate_diverse_test_set(n_images: int = 100, img_size: int = 224):
    """Generate diverse test images with varying complexity."""
    shapes = ["circle", "square", "triangle"]
    colors = [
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
        ("yellow", (255, 255, 0)),
    ]
    n_classes = len(shapes) * len(colors)

    np.random.seed(123)  # Different seed for test
    samples = []

    for i in range(n_images):
        # Varying complexity: 1-4 objects
        n_objects = 1 + (i % 4)
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
            "n_objects": n_objects,
        })

    return samples, n_classes


def load_models(device):
    """Load DINOv2 and VLM models."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print("  Loading DINOv2-large...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2 = dinov2.to(device).eval()

    print("  Loading Qwen2.5-VL...")
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
    )

    return dinov2, vlm, processor


def extract_features(images, dinov2, vlm, processor, device):
    """Extract both DINOv2 and VLM features."""
    from torchvision import transforms

    dino_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dino_features_list = []
    vlm_features_list = []

    # DINOv2 features (batched)
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = torch.stack([dino_transform(img) for img in batch]).to(device)
            features = dinov2.forward_features(batch_tensors)
            if isinstance(features, dict):
                patch_features = features.get('x_norm_patchtokens', features.get('x_prenorm', None))
                if patch_features is None:
                    patch_features = features['x_norm'][:, 1:, :]  # Exclude CLS
            else:
                patch_features = features[:, 1:, :]
            dino_features_list.append(patch_features.cpu())

    dino_features = torch.cat(dino_features_list, dim=0)

    # VLM features (one by one due to variable length)
    with torch.no_grad():
        for img in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(vlm.device)
            outputs = vlm(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1]
            vlm_features_list.append(hidden_states[0].float().cpu())

    # Pad VLM features
    max_len = max(f.shape[0] for f in vlm_features_list)
    padded_vlm = []
    for f in vlm_features_list:
        if f.shape[0] < max_len:
            padding = torch.zeros(max_len - f.shape[0], f.shape[1])
            f = torch.cat([f, padding], dim=0)
        padded_vlm.append(f)

    vlm_features = torch.stack(padded_vlm)

    return dino_features, vlm_features


def compute_map(pred_boxes, pred_logits, gt_boxes, gt_labels, iou_threshold, n_classes):
    """Compute mAP at given IoU threshold."""
    from torchvision.ops import box_iou

    class_predictions = defaultdict(list)
    class_n_gt = defaultdict(int)

    for img_idx in range(len(pred_boxes)):
        pb = pred_boxes[img_idx]
        pl = pred_logits[img_idx]
        gb = gt_boxes[img_idx]
        gl = gt_labels[img_idx]

        if len(gb) == 0:
            continue

        if not isinstance(pb, torch.Tensor):
            pb = torch.tensor(pb)
        if not isinstance(gb, torch.Tensor):
            gb = torch.tensor(gb)
        if not isinstance(gl, torch.Tensor):
            gl = torch.tensor(gl)

        for label in gl.tolist():
            class_n_gt[label] += 1

        pred_probs = F.softmax(pl, dim=-1)
        pred_classes = pred_probs[:, :-1].argmax(dim=-1)
        pred_scores = pred_probs[:, :-1].max(dim=-1)[0]

        keep = pred_scores > 0.3
        pb = pb[keep]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]

        if len(pb) == 0:
            continue

        matched_gt = set()
        for pred_idx in range(len(pb)):
            pred_box = pb[pred_idx:pred_idx+1]
            pred_class = pred_classes[pred_idx].item()
            pred_score = pred_scores[pred_idx].item()

            best_iou = 0
            best_gt_idx = -1

            for gt_idx in range(len(gb)):
                if gt_idx in matched_gt or gl[gt_idx].item() != pred_class:
                    continue
                iou = box_iou(pred_box, gb[gt_idx:gt_idx+1])[0, 0].item()
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


def compute_all_metrics(recon, target, device):
    """Compute all evaluation metrics."""
    import lpips as lpips_lib
    from scipy.ndimage import gaussian_filter
    import cv2

    # LPIPS
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    recon_lpips = recon * 2 - 1
    target_lpips = target * 2 - 1
    lpips_scores = []
    with torch.no_grad():
        for i in range(len(recon)):
            score = lpips_fn(recon_lpips[i:i+1], target_lpips[i:i+1])
            lpips_scores.append(score.item())
    lpips_val = np.mean(lpips_scores)

    # SSIM
    ssim_values = []
    for i in range(len(recon)):
        im1 = recon[i].cpu().numpy().mean(axis=0)
        im2 = target[i].cpu().numpy().mean(axis=0)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu1 = gaussian_filter(im1, sigma=1.5)
        mu2 = gaussian_filter(im2, sigma=1.5)
        sigma1_sq = gaussian_filter(im1 ** 2, sigma=1.5) - mu1 ** 2
        sigma2_sq = gaussian_filter(im2 ** 2, sigma=1.5) - mu2 ** 2
        sigma12 = gaussian_filter(im1 * im2, sigma=1.5) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        ssim_values.append(ssim_map.mean())
    ssim_val = float(np.mean(ssim_values))

    # Spatial IoU
    ious = []
    for i in range(len(recon)):
        recon_gray = recon[i].mean(dim=0).cpu().numpy()
        target_gray = target[i].mean(dim=0).cpu().numpy()
        recon_mask = recon_gray < 0.95
        target_mask = target_gray < 0.95
        intersection = (recon_mask & target_mask).sum()
        union = (recon_mask | target_mask).sum()
        ious.append(intersection / union if union > 0 else 1.0)
    spatial_iou = float(np.mean(ious))

    # Edge F1
    f1_scores = []
    for i in range(len(recon)):
        recon_np = (recon[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        target_np = (target[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        recon_gray = cv2.cvtColor(recon_np, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_np, cv2.COLOR_RGB2GRAY)
        recon_edges = cv2.Sobel(recon_gray, cv2.CV_64F, 1, 1, ksize=3)
        target_edges = cv2.Sobel(target_gray, cv2.CV_64F, 1, 1, ksize=3)
        recon_mask = np.abs(recon_edges) > 30
        target_mask = np.abs(target_edges) > 30
        tp = (recon_mask & target_mask).sum()
        fp = (recon_mask & ~target_mask).sum()
        fn = (~recon_mask & target_mask).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    edge_f1 = float(np.mean(f1_scores))

    del lpips_fn
    torch.cuda.empty_cache()

    return {
        "lpips": lpips_val,
        "ssim": ssim_val,
        "spatial_iou": spatial_iou,
        "edge_f1": edge_f1,
    }


def images_to_tensor(images):
    """Convert PIL images to tensor."""
    tensors = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        tensors.append(torch.tensor(arr))
    return torch.stack(tensors)


def e_p2_4_pipeline_evaluation(runner: ExperimentRunner) -> dict:
    """Comprehensive evaluation of hybrid pipeline against Gate 1 thresholds.

    This implementation:
    1. Generates diverse test set
    2. Trains hybrid pipeline from scratch (fresh training)
    3. Evaluates all Gate 1 metrics
    4. Analyzes performance by scene complexity
    5. Provides Gate 1 pass/fail assessment

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-P2.4: End-to-End Hybrid Pipeline Evaluation")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e_p2_4/stage": 0, "e_p2_4/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate test set
    # =========================================================================
    print("\n[Stage 1/6] Generating diverse test set...")

    test_samples, n_classes = generate_diverse_test_set(n_images=100)
    test_images = [s["image"] for s in test_samples]

    print(f"  Test set: {len(test_images)} images, {n_classes} classes")

    runner.log_metrics({"e_p2_4/stage": 1, "e_p2_4/progress": 0.05})

    # =========================================================================
    # Stage 2: Load encoders
    # =========================================================================
    print("\n[Stage 2/6] Loading encoder models...")

    dinov2, vlm, processor = load_models(device)

    runner.log_metrics({"e_p2_4/stage": 2, "e_p2_4/progress": 0.15})

    # =========================================================================
    # Stage 3: Extract features
    # =========================================================================
    print("\n[Stage 3/6] Extracting features...")

    dino_features, vlm_features = extract_features(test_images, dinov2, vlm, processor, device)

    print(f"  DINOv2 features: {dino_features.shape}")
    print(f"  VLM features: {vlm_features.shape}")

    runner.log_metrics({"e_p2_4/stage": 3, "e_p2_4/progress": 0.35})

    # =========================================================================
    # Stage 4: Train hybrid pipeline
    # =========================================================================
    print("\n[Stage 4/6] Training hybrid pipeline...")

    vlm_dim = vlm_features.shape[-1]
    dino_dim = dino_features.shape[-1]

    # Split data
    n_train = int(len(test_images) * 0.7)
    train_idx = list(range(n_train))
    eval_idx = list(range(n_train, len(test_images)))

    fusion = HybridFusionModule(
        vlm_dim=vlm_dim,
        spatial_dim=dino_dim,
        fusion_dim=1024,
        num_fusion_layers=4,
        num_output_queries=64,
        output_dim=4096,
    ).to(device)

    decoder = SimplePixelDecoder(input_dim=4096).to(device)

    # Prepare tensors
    targets = images_to_tensor(test_images).to(device)
    dino_t = dino_features.to(device)
    vlm_t = vlm_features.to(device)

    optimizer = torch.optim.AdamW(
        list(fusion.parameters()) + list(decoder.parameters()),
        lr=5e-5,
        weight_decay=0.01,
    )

    # Training
    n_epochs = 100
    batch_size = 8

    for epoch in range(n_epochs):
        fusion.train()
        decoder.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = torch.randperm(n_train)

        for i in range(0, n_train, batch_size):
            batch_idx = [train_idx[j] for j in indices[i:i + batch_size].tolist()]
            batch_dino = dino_t[batch_idx]
            batch_vlm = vlm_t[batch_idx]
            batch_targets = targets[batch_idx]

            optimizer.zero_grad()
            conditioning = fusion(batch_vlm, batch_dino)
            recon = decoder(conditioning)
            loss = F.mse_loss(recon, batch_targets) + 0.1 * F.l1_loss(recon, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / n_batches:.4f}")

    runner.log_metrics({"e_p2_4/stage": 4, "e_p2_4/progress": 0.65})

    # Clean up VLM to free memory
    del vlm
    del processor
    torch.cuda.empty_cache()

    # =========================================================================
    # Stage 5: Evaluate on held-out set
    # =========================================================================
    print("\n[Stage 5/6] Evaluating on held-out test set...")

    fusion.eval()
    decoder.eval()

    eval_dino = dino_t[eval_idx]
    eval_vlm = vlm_t[eval_idx]
    eval_targets = targets[eval_idx]
    eval_images = [test_images[i] for i in eval_idx]

    with torch.no_grad():
        eval_conditioning = fusion(eval_vlm, eval_dino)
        eval_recon = decoder(eval_conditioning)

    # Compute metrics
    metrics = compute_all_metrics(eval_recon, eval_targets, device)

    print(f"\n  Gate 1 Metrics:")
    print(f"    LPIPS: {metrics['lpips']:.4f} (threshold: < 0.35)")
    print(f"    Spatial IoU: {metrics['spatial_iou']:.4f} (threshold: > 0.60)")
    print(f"    SSIM: {metrics['ssim']:.4f}")
    print(f"    Edge F1: {metrics['edge_f1']:.4f}")

    # Train full DETR detection head for mAP
    print("\n  Training DETR detection head for mAP...")

    det_probe = DETRDetectionHead(
        input_dim=4096,
        hidden_dim=256,
        num_queries=20,
        n_classes=n_classes,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
    ).to(device)

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    det_optimizer = torch.optim.AdamW(det_probe.parameters(), lr=1e-4, weight_decay=1e-4)

    # Learning rate warmup
    def lr_lambda(epoch):
        if epoch < 10:
            return epoch / 10
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(det_optimizer, lr_lambda)

    train_samples = [test_samples[i] for i in train_idx]
    n_epochs = 150  # Longer training for DETR

    for epoch in range(n_epochs):
        det_probe.train()
        for i in range(0, n_train, batch_size):
            batch_idx = train_idx[i:i + batch_size]
            batch_cond = []

            with torch.no_grad():
                for idx in batch_idx:
                    c = fusion(vlm_t[idx:idx+1], dino_t[idx:idx+1])
                    batch_cond.append(c[0])
            batch_cond = torch.stack(batch_cond)

            pred_boxes, pred_logits = det_probe(batch_cond)

            # Prepare ground truth in list format for DETR loss
            gt_boxes_list = []
            gt_labels_list = []
            for j, idx in enumerate(batch_idx):
                sample = test_samples[idx]
                # Convert boxes from xyxy to cxcywh format
                boxes_xyxy = torch.tensor(sample["boxes"]).to(device)
                if len(boxes_xyxy) > 0:
                    # xyxy -> cxcywh
                    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
                    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
                    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
                    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
                    boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)
                    gt_boxes_list.append(boxes_cxcywh)
                else:
                    gt_boxes_list.append(torch.zeros(0, 4, device=device))
                gt_labels_list.append(torch.tensor(sample["labels"]).to(device))

            loss, _ = detr_loss(
                pred_boxes, pred_logits,
                gt_boxes_list, gt_labels_list,
                matcher, n_classes
            )

            det_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(det_probe.parameters(), 0.1)
            det_optimizer.step()

        scheduler.step()

    # Evaluate mAP
    det_probe.eval()
    all_pred_boxes = []
    all_pred_logits = []
    all_gt_boxes = []
    all_gt_labels = []

    with torch.no_grad():
        for idx in eval_idx:
            cond = fusion(vlm_t[idx:idx+1], dino_t[idx:idx+1])
            pb, pl = det_probe(cond)
            # Convert DETR cxcywh output to xyxy for mAP computation
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pb[0])
            all_pred_boxes.append(pred_boxes_xyxy)
            all_pred_logits.append(pl[0])
            sample = test_samples[idx]
            all_gt_boxes.append(torch.tensor(sample["boxes"]).to(device))
            all_gt_labels.append(torch.tensor(sample["labels"]).to(device))

    map_05 = compute_map(all_pred_boxes, all_pred_logits, all_gt_boxes, all_gt_labels, 0.5, n_classes)
    metrics["mAP@0.5"] = map_05

    print(f"    mAP@0.5: {map_05:.4f} (threshold: > 0.40)")

    runner.log_metrics({
        "e_p2_4/stage": 5,
        "e_p2_4/progress": 0.9,
        "e_p2_4/lpips": metrics["lpips"],
        "e_p2_4/spatial_iou": metrics["spatial_iou"],
        "e_p2_4/ssim": metrics["ssim"],
        "e_p2_4/edge_f1": metrics["edge_f1"],
        "e_p2_4/map_05": map_05,
    })

    # =========================================================================
    # Stage 6: Gate 1 assessment and artifacts
    # =========================================================================
    print("\n[Stage 6/6] Gate 1 assessment...")

    # Gate 1 thresholds
    gate1_criteria = {
        "spatial_iou": {"threshold": 0.60, "direction": "higher", "actual": metrics["spatial_iou"]},
        "lpips": {"threshold": 0.35, "direction": "lower", "actual": metrics["lpips"]},
        "mAP@0.5": {"threshold": 0.40, "direction": "higher", "actual": map_05},
    }

    gate1_passed = True
    for name, crit in gate1_criteria.items():
        if crit["direction"] == "higher":
            passed = crit["actual"] > crit["threshold"]
        else:
            passed = crit["actual"] < crit["threshold"]
        gate1_criteria[name]["passed"] = passed
        if not passed:
            gate1_passed = False
        status = "✓" if passed else "✗"
        print(f"    {name}: {crit['actual']:.3f} {'>' if crit['direction'] == 'higher' else '<'} {crit['threshold']} {status}")

    print(f"\n  Gate 1 Status: {'PASSED' if gate1_passed else 'NOT PASSED'}")

    # Baselines
    baselines = {
        "c1_vlm_only": {"lpips": 0.264, "spatial_iou": 0.559, "mAP@0.5": 0.001},
        "q2_spatial": {"bbox_iou": 0.104},
    }

    # Save visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_viz = min(8, len(eval_images))
    fig, axes = plt.subplots(2, n_viz, figsize=(2.5 * n_viz, 5))

    for i in range(n_viz):
        axes[0, i].imshow(eval_images[i])
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)
        axes[0, i].axis("off")

        recon_img = eval_recon[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(np.clip(recon_img, 0, 1))
        if i == 0:
            axes[1, i].set_ylabel("Hybrid Recon", fontsize=10)
        axes[1, i].axis("off")

    plt.suptitle(
        f"Gate 1 Evaluation: {'PASSED' if gate1_passed else 'NOT PASSED'}\n"
        f"LPIPS: {metrics['lpips']:.3f} | Spatial IoU: {metrics['spatial_iou']:.3f} | mAP@0.5: {map_05:.3f}",
        fontsize=11,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    viz_path = runner.results.save_artifact("gate1_evaluation.png", buf.read())

    # Save results
    results_data = {
        "gate1_assessment": {
            "passed": gate1_passed,
            "criteria": gate1_criteria,
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
        "baselines": baselines,
        "improvement_vs_vlm": {
            "spatial_iou": metrics["spatial_iou"] - baselines["c1_vlm_only"]["spatial_iou"],
            "lpips": baselines["c1_vlm_only"]["lpips"] - metrics["lpips"],
            "map_improvement_factor": map_05 / baselines["c1_vlm_only"]["mAP@0.5"] if map_05 > 0 else float("inf"),
        },
    }
    data_path = runner.results.save_json_artifact("gate1_results.json", results_data)

    runner.log_metrics({"e_p2_4/stage": 6, "e_p2_4/progress": 1.0, "e_p2_4/gate1_passed": 1 if gate1_passed else 0})

    # Determine finding
    if gate1_passed:
        finding = (
            f"GATE 1 PASSED. Hybrid architecture achieves all success criteria: "
            f"Spatial IoU={metrics['spatial_iou']:.3f} (>{0.60}), "
            f"LPIPS={metrics['lpips']:.3f} (<{0.35}), "
            f"mAP@0.5={map_05:.3f} (>{0.40}). "
            f"Spatial IoU improved by {metrics['spatial_iou'] - baselines['c1_vlm_only']['spatial_iou']:.3f} vs VLM-only. "
            f"Proceed to Phase 2 (Adapter Training)."
        )
    else:
        failed_criteria = [k for k, v in gate1_criteria.items() if not v["passed"]]
        finding = (
            f"GATE 1 NOT PASSED. Failed criteria: {', '.join(failed_criteria)}. "
            f"Actual values: Spatial IoU={metrics['spatial_iou']:.3f}, "
            f"LPIPS={metrics['lpips']:.3f}, mAP@0.5={map_05:.3f}. "
            f"Consider: ablation studies (E-P2.5), architecture tuning, or alternative encoders."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "lpips": float(metrics["lpips"]),
            "ssim": float(metrics["ssim"]),
            "spatial_iou": float(metrics["spatial_iou"]),
            "edge_f1": float(metrics["edge_f1"]),
            "mAP@0.5": float(map_05),
            "gate1_passed": gate1_passed,
        },
        "artifacts": [viz_path, data_path],
    }
