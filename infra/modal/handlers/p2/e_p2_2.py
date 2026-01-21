"""E-P2.2: DINOv2-Only Reconstruction Baseline

Objective: Establish whether DINOv2 features alone can drive the video decoder at high quality.

Protocol:
1. Train adapter: DINOv2 features -> LTX-Video conditioning
2. Evaluate reconstruction quality (LPIPS, SSIM, Spatial IoU)
3. Compare to C1 VLM-only baseline

Success Metrics:
- LPIPS < 0.30 (target: < 0.25)
- SSIM > 0.80 (target: > 0.85)
- Spatial IoU > 0.70 (target: > 0.75)
- Edge F1 > 0.55 (target: > 0.65)

Decision point: If DINOv2-only achieves Spatial IoU > 0.7 AND LPIPS < 0.30,
consider simplifying to single-encoder (no fusion needed).
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


class DINOv2Adapter(nn.Module):
    """Project DINOv2 features to LTX-Video conditioning space."""

    def __init__(
        self,
        dino_dim: int = 1024,
        ltx_dim: int = 4096,
        hidden_dim: int = 2048,
        n_output_tokens: int = 77,
    ):
        super().__init__()
        self.n_output_tokens = n_output_tokens
        self.ltx_dim = ltx_dim

        # Learnable queries for cross-attention
        self.queries = nn.Parameter(torch.randn(n_output_tokens, hidden_dim) * 0.02)

        # Project DINOv2 to hidden dim
        self.dino_proj = nn.Linear(dino_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
            for _ in range(4)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(4)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(8)
        ])

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, ltx_dim)

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dino_features: [B, N_patches, 1536] DINOv2 patch features

        Returns:
            [B, n_output_tokens, ltx_dim] conditioning for video decoder
        """
        B = dino_features.size(0)
        kv = self.dino_proj(dino_features)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)

        for i, (attn, ffn) in enumerate(zip(self.cross_attn, self.ffn)):
            # Cross-attention
            attn_out, _ = attn(q, kv, kv)
            q = self.layer_norms[2 * i](q + attn_out)
            # FFN
            ffn_out = ffn(q)
            q = self.layer_norms[2 * i + 1](q + ffn_out)

        return self.out_proj(q)


class SimplePixelDecoder(nn.Module):
    """Simple pixel decoder for reconstruction evaluation.

    For this experiment, we use a simple decoder to test if DINOv2 features
    contain enough information for reconstruction. In the full pipeline,
    LTX-Video would be used instead.
    """

    def __init__(self, input_dim: int = 4096, output_size: int = 224):
        super().__init__()
        self.output_size = output_size

        # Progressive upsampling
        self.initial = nn.Linear(input_dim, 256 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 7 -> 14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 14 -> 28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 28 -> 56
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),   # 56 -> 112
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),    # 112 -> 224
            nn.Sigmoid(),
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conditioning: [B, n_tokens, input_dim]

        Returns:
            [B, 3, output_size, output_size] reconstructed images
        """
        # Pool across tokens
        x = conditioning.mean(dim=1)  # [B, input_dim]
        x = self.initial(x)  # [B, 256 * 7 * 7]
        x = x.view(-1, 256, 7, 7)
        return self.decoder(x)


def load_dinov2_model(device: torch.device):
    """Load DINOv2-large model (ViT-L for latency optimization)."""
    print("  Loading DINOv2-large model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = model.to(device)
    model.eval()
    print(f"  DINOv2-large loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    return model


def extract_dinov2_features(
    images: list,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Extract DINOv2 patch features from images."""
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

            features = model.forward_features(batch_tensors)
            if isinstance(features, dict):
                patch_features = features.get('x_norm_patchtokens', features.get('x_prenorm', None))
                if patch_features is None:
                    patch_features = features['x_norm'][:, 1:, :]
            else:
                patch_features = features[:, 1:, :]

            features_list.append(patch_features.cpu())

    return torch.cat(features_list, dim=0)


def generate_training_images(n_images: int = 200, img_size: int = 224):
    """Generate synthetic training images with known spatial structure."""
    shapes = ["circle", "square", "triangle"]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    images = []
    boxes = []  # Ground truth bounding boxes for spatial eval
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
            (cx - size) / img_size,
            (cy - size) / img_size,
            (cx + size) / img_size,
            (cy + size) / img_size,
        ])

    return images, np.array(boxes, dtype=np.float32)


def images_to_tensor(images: list) -> torch.Tensor:
    """Convert list of PIL images to tensor."""
    tensors = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        tensors.append(torch.tensor(arr))
    return torch.stack(tensors)


def compute_spatial_iou(recon: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute spatial IoU between reconstructed and target images.

    Uses a simple thresholding approach to find object regions.
    """
    ious = []

    for i in range(len(recon)):
        # Convert to grayscale and threshold
        recon_gray = recon[i].mean(dim=0).cpu().numpy()
        target_gray = target[i].mean(dim=0).cpu().numpy()

        # Find non-white regions (object pixels)
        recon_mask = recon_gray < 0.95
        target_mask = target_gray < 0.95

        # Compute IoU
        intersection = (recon_mask & target_mask).sum()
        union = (recon_mask | target_mask).sum()

        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(1.0)  # Both empty

    return float(np.mean(ious))


def compute_edge_f1(recon: torch.Tensor, target: torch.Tensor) -> float:
    """Compute edge F1 score using Sobel edge detection."""
    import cv2

    f1_scores = []

    for i in range(len(recon)):
        # Convert to grayscale numpy
        recon_np = (recon[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        target_np = (target[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        recon_gray = cv2.cvtColor(recon_np, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_np, cv2.COLOR_RGB2GRAY)

        # Sobel edge detection
        recon_edges = cv2.Sobel(recon_gray, cv2.CV_64F, 1, 1, ksize=3)
        target_edges = cv2.Sobel(target_gray, cv2.CV_64F, 1, 1, ksize=3)

        # Threshold edges
        recon_edge_mask = np.abs(recon_edges) > 30
        target_edge_mask = np.abs(target_edges) > 30

        # Compute precision, recall, F1
        tp = (recon_edge_mask & target_edge_mask).sum()
        fp = (recon_edge_mask & ~target_edge_mask).sum()
        fn = (~recon_edge_mask & target_edge_mask).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def compute_reconstruction_metrics(
    recon: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute all reconstruction quality metrics."""
    import lpips as lpips_lib

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

    # MSE and PSNR
    mse = F.mse_loss(recon, target).item()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

    # SSIM
    ssim_val = compute_ssim(recon, target)

    # Spatial IoU
    spatial_iou = compute_spatial_iou(recon, target)

    # Edge F1
    edge_f1 = compute_edge_f1(recon, target)

    del lpips_fn
    torch.cuda.empty_cache()

    return {
        "lpips": lpips_val,
        "ssim": ssim_val,
        "psnr": psnr,
        "mse": mse,
        "spatial_iou": spatial_iou,
        "edge_f1": edge_f1,
    }


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute SSIM between two batches of images."""
    from scipy.ndimage import gaussian_filter

    ssim_values = []
    for i in range(len(img1)):
        im1 = img1[i].cpu().numpy().mean(axis=0)
        im2 = img2[i].cpu().numpy().mean(axis=0)

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

    return float(np.mean(ssim_values))


def create_reconstruction_grid(
    original_images: list,
    reconstructed: torch.Tensor,
    metrics: dict,
) -> bytes:
    """Create visualization grid comparing original and reconstructed images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_images = min(8, len(original_images))
    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 5))

    for i in range(n_images):
        # Original
        axes[0, i].imshow(original_images[i])
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)
        axes[0, i].axis("off")

        # Reconstructed
        recon_img = reconstructed[i].permute(1, 2, 0).cpu().numpy()
        recon_img = np.clip(recon_img, 0, 1)
        axes[1, i].imshow(recon_img)
        if i == 0:
            axes[1, i].set_ylabel("DINOv2 Recon", fontsize=10)
        axes[1, i].axis("off")

    plt.suptitle(
        f"DINOv2-Only Reconstruction\n"
        f"LPIPS: {metrics['lpips']:.3f} | SSIM: {metrics['ssim']:.3f} | "
        f"Spatial IoU: {metrics['spatial_iou']:.3f}",
        fontsize=11,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e_p2_2_dinov2_reconstruction(runner: ExperimentRunner) -> dict:
    """Run DINOv2-only reconstruction baseline experiment.

    This implementation:
    1. Generates synthetic training/test images
    2. Extracts DINOv2 features
    3. Trains adapter + pixel decoder for reconstruction
    4. Evaluates quality metrics (LPIPS, SSIM, Spatial IoU, Edge F1)
    5. Compares to C1 VLM-only baseline

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-P2.2: DINOv2-Only Reconstruction Baseline")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e_p2_2/stage": 0, "e_p2_2/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate training and test images
    # =========================================================================
    print("\n[Stage 1/5] Generating training and test images...")

    train_images, train_boxes = generate_training_images(n_images=300)
    test_images, test_boxes = generate_training_images(n_images=50)

    print(f"  Train: {len(train_images)} images")
    print(f"  Test: {len(test_images)} images")

    runner.log_metrics({
        "e_p2_2/stage": 1,
        "e_p2_2/progress": 0.1,
        "e_p2_2/n_train": len(train_images),
        "e_p2_2/n_test": len(test_images),
    })

    # =========================================================================
    # Stage 2: Extract DINOv2 features
    # =========================================================================
    print("\n[Stage 2/5] Extracting DINOv2 features...")

    dinov2 = load_dinov2_model(device)

    train_features = extract_dinov2_features(train_images, dinov2, device)
    test_features = extract_dinov2_features(test_images, dinov2, device)

    print(f"  Train features: {train_features.shape}")
    print(f"  Test features: {test_features.shape}")

    del dinov2
    torch.cuda.empty_cache()

    runner.log_metrics({
        "e_p2_2/stage": 2,
        "e_p2_2/progress": 0.3,
        "e_p2_2/feature_dim": train_features.shape[-1],
        "e_p2_2/n_patches": train_features.shape[1],
    })

    # =========================================================================
    # Stage 3: Train adapter and decoder
    # =========================================================================
    print("\n[Stage 3/5] Training adapter and decoder...")

    feature_dim = train_features.shape[-1]
    adapter = DINOv2Adapter(dino_dim=feature_dim, ltx_dim=4096, n_output_tokens=64).to(device)
    decoder = SimplePixelDecoder(input_dim=4096, output_size=224).to(device)

    adapter_params = sum(p.numel() for p in adapter.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Adapter params: {adapter_params:,}")
    print(f"  Decoder params: {decoder_params:,}")

    # Convert images to tensors
    train_targets = images_to_tensor(train_images).to(device)
    test_targets = images_to_tensor(test_images).to(device)
    train_features_t = train_features.to(device)
    test_features_t = test_features.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(decoder.parameters()),
        lr=1e-4,
        weight_decay=0.01,
    )

    # Training
    n_epochs = 150
    batch_size = 16
    best_lpips = float("inf")

    for epoch in range(n_epochs):
        adapter.train()
        decoder.train()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle indices
        indices = torch.randperm(len(train_features_t))

        for i in range(0, len(train_features_t), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_feat = train_features_t[batch_idx]
            batch_targets = train_targets[batch_idx]

            optimizer.zero_grad()

            # Forward
            conditioning = adapter(batch_feat)
            recon = decoder(conditioning)

            # Loss: MSE + perceptual (simplified)
            mse_loss = F.mse_loss(recon, batch_targets)
            # Add L1 for sharper edges
            l1_loss = F.l1_loss(recon, batch_targets)
            loss = mse_loss + 0.1 * l1_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
            runner.log_metrics({
                "e_p2_2/train_loss": avg_loss,
                "e_p2_2/epoch": epoch + 1,
            }, step=epoch)

    runner.log_metrics({"e_p2_2/stage": 3, "e_p2_2/progress": 0.7})

    # =========================================================================
    # Stage 4: Evaluate reconstruction quality
    # =========================================================================
    print("\n[Stage 4/5] Evaluating reconstruction quality...")

    adapter.eval()
    decoder.eval()

    with torch.no_grad():
        test_conditioning = adapter(test_features_t)
        test_recon = decoder(test_conditioning)

    metrics = compute_reconstruction_metrics(test_recon, test_targets, device)

    print(f"  LPIPS: {metrics['lpips']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  Spatial IoU: {metrics['spatial_iou']:.4f}")
    print(f"  Edge F1: {metrics['edge_f1']:.4f}")

    # C1 VLM-only baseline comparison
    c1_baseline = {
        "lpips": 0.264,
        "ssim": 0.943,
        "spatial_iou": 0.559,
    }

    runner.log_metrics({
        "e_p2_2/stage": 4,
        "e_p2_2/progress": 0.9,
        "e_p2_2/lpips": metrics["lpips"],
        "e_p2_2/ssim": metrics["ssim"],
        "e_p2_2/psnr": metrics["psnr"],
        "e_p2_2/spatial_iou": metrics["spatial_iou"],
        "e_p2_2/edge_f1": metrics["edge_f1"],
    })

    # =========================================================================
    # Stage 5: Save artifacts
    # =========================================================================
    print("\n[Stage 5/5] Saving artifacts...")

    # Visualization
    viz_bytes = create_reconstruction_grid(test_images[:8], test_recon[:8], metrics)
    viz_path = runner.results.save_artifact("dinov2_reconstruction_grid.png", viz_bytes)

    # Metrics data
    results_data = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "c1_baseline": c1_baseline,
        "comparison": {
            "lpips_improvement": c1_baseline["lpips"] - metrics["lpips"],
            "spatial_iou_improvement": metrics["spatial_iou"] - c1_baseline["spatial_iou"],
            "ssim_change": metrics["ssim"] - c1_baseline["ssim"],
        },
        "model_info": {
            "adapter_params": adapter_params,
            "decoder_params": decoder_params,
            "total_params": adapter_params + decoder_params,
            "feature_dim": feature_dim,
        },
    }
    data_path = runner.results.save_json_artifact("dinov2_reconstruction_results.json", results_data)

    runner.log_metrics({"e_p2_2/stage": 5, "e_p2_2/progress": 1.0})

    # =========================================================================
    # Determine finding
    # =========================================================================
    lpips_target = 0.30
    spatial_iou_target = 0.70

    if metrics["lpips"] < lpips_target and metrics["spatial_iou"] > spatial_iou_target:
        finding = (
            f"DINOv2-only reconstruction achieves excellent quality "
            f"(LPIPS={metrics['lpips']:.3f}, Spatial IoU={metrics['spatial_iou']:.3f}). "
            f"Spatial IoU improved by {metrics['spatial_iou'] - c1_baseline['spatial_iou']:.3f} vs VLM baseline. "
            f"Consider if hybrid fusion is necessary or if DINOv2-only suffices."
        )
    elif metrics["spatial_iou"] > 0.60:
        finding = (
            f"DINOv2-only reconstruction shows good spatial preservation "
            f"(Spatial IoU={metrics['spatial_iou']:.3f}) but needs improvement "
            f"(LPIPS={metrics['lpips']:.3f}). "
            f"Hybrid fusion may help balance spatial and perceptual quality."
        )
    else:
        finding = (
            f"DINOv2-only reconstruction needs improvement "
            f"(LPIPS={metrics['lpips']:.3f}, Spatial IoU={metrics['spatial_iou']:.3f}). "
            f"Fusion with VLM semantic features is essential."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "lpips": float(metrics["lpips"]),
            "ssim": float(metrics["ssim"]),
            "psnr": float(metrics["psnr"]),
            "spatial_iou": float(metrics["spatial_iou"]),
            "edge_f1": float(metrics["edge_f1"]),
            "c1_lpips_baseline": c1_baseline["lpips"],
            "c1_spatial_iou_baseline": c1_baseline["spatial_iou"],
        },
        "artifacts": [viz_path, data_path],
    }
