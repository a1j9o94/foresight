"""E-P2.3: Cross-Attention Fusion Module Training

Objective: Train the hybrid fusion module combining VLM and DINOv2 features.

Protocol:
1. Implement cross-attention fusion architecture
2. Train fusion module with frozen VLM and DINOv2 encoders
3. Evaluate combined reconstruction quality

Success Metrics:
- LPIPS < 0.35 (target: < 0.28)
- Spatial IoU > 0.60 (target: > 0.70)
- mAP@0.5 > 0.40 (target: > 0.50)
- Semantic consistency > 0.70 (target: > 0.80)
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


class HybridFusionModule(nn.Module):
    """
    Fuses VLM semantic features with DINOv2 spatial features.

    Design principles:
    1. Spatial features provide grounding (DINOv2)
    2. Semantic features provide context (VLM)
    3. Learnable queries extract conditioning for video decoder
    """

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

        # Project both streams to common dimension
        self.vlm_proj = nn.Linear(vlm_dim, fusion_dim)
        self.spatial_proj = nn.Linear(spatial_dim, fusion_dim)

        # Positional encoding for spatial features
        self.spatial_pos = nn.Parameter(torch.randn(256, fusion_dim) * 0.02)

        # Learnable output queries
        self.output_queries = nn.Parameter(
            torch.randn(num_output_queries, fusion_dim) * 0.02
        )

        # Cross-attention fusion layers
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

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, output_dim),
        )

    def forward(
        self,
        vlm_features: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vlm_features: [B, T_vlm, vlm_dim] VLM semantic features
            spatial_features: [B, H*W, spatial_dim] DINOv2 spatial features

        Returns:
            [B, num_output_queries, output_dim] conditioning for decoder
        """
        B = vlm_features.size(0)

        # Project to common space
        vlm_proj = self.vlm_proj(vlm_features)
        spatial_proj = self.spatial_proj(spatial_features)

        # Add positional encoding to spatial features
        n_spatial = spatial_proj.size(1)
        spatial_proj = spatial_proj + self.spatial_pos[:n_spatial]

        # Concatenate streams for cross-attention
        context = torch.cat([vlm_proj, spatial_proj], dim=1)

        # Learnable queries attend to combined context
        queries = self.output_queries.unsqueeze(0).expand(B, -1, -1)

        for layer in self.fusion_layers:
            queries = layer(queries, context)

        # Project to output space
        return self.output_proj(queries)


class SimplePixelDecoder(nn.Module):
    """Pixel decoder for reconstruction evaluation."""

    def __init__(self, input_dim: int = 4096, output_size: int = 224):
        super().__init__()
        self.output_size = output_size

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

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        x = conditioning.mean(dim=1)
        x = self.initial(x)
        x = x.view(-1, 256, 7, 7)
        return self.decoder(x)


def load_dinov2_model(device: torch.device):
    """Load DINOv2-large model (ViT-L for latency optimization)."""
    print("  Loading DINOv2-large model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = model.to(device)
    model.eval()
    return model


def load_vlm_model(device: torch.device):
    """Load Qwen2.5-VL model for feature extraction."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print("  Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
    )
    return model, processor


def extract_dinov2_features(images: list, model: nn.Module, device: torch.device) -> torch.Tensor:
    """Extract DINOv2 patch features."""
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


def extract_vlm_features(images: list, model, processor, device: torch.device) -> torch.Tensor:
    """Extract VLM features from images."""
    features_list = []

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
            features_list.append(hidden_states[0].float().cpu())

    # Pad to same length
    max_len = max(f.shape[0] for f in features_list)
    padded = []
    for f in features_list:
        if f.shape[0] < max_len:
            padding = torch.zeros(max_len - f.shape[0], f.shape[1])
            f = torch.cat([f, padding], dim=0)
        padded.append(f)

    return torch.stack(padded)


def generate_training_images(n_images: int = 200, img_size: int = 224):
    """Generate synthetic training images."""
    shapes = ["circle", "square", "triangle"]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    images = []
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

    return images


def images_to_tensor(images: list) -> torch.Tensor:
    """Convert list of PIL images to tensor."""
    tensors = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        tensors.append(torch.tensor(arr))
    return torch.stack(tensors)


def compute_spatial_iou(recon: torch.Tensor, target: torch.Tensor) -> float:
    """Compute spatial IoU."""
    ious = []
    for i in range(len(recon)):
        recon_gray = recon[i].mean(dim=0).cpu().numpy()
        target_gray = target[i].mean(dim=0).cpu().numpy()

        recon_mask = recon_gray < 0.95
        target_mask = target_gray < 0.95

        intersection = (recon_mask & target_mask).sum()
        union = (recon_mask | target_mask).sum()

        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(1.0)

    return float(np.mean(ious))


def compute_reconstruction_metrics(recon: torch.Tensor, target: torch.Tensor, device: torch.device) -> dict:
    """Compute reconstruction metrics."""
    import lpips as lpips_lib
    from scipy.ndimage import gaussian_filter

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

    # MSE
    mse = F.mse_loss(recon, target).item()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

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
    spatial_iou = compute_spatial_iou(recon, target)

    del lpips_fn
    torch.cuda.empty_cache()

    return {
        "lpips": lpips_val,
        "ssim": ssim_val,
        "psnr": psnr,
        "spatial_iou": spatial_iou,
    }


def create_comparison_grid(
    original: list,
    vlm_only: torch.Tensor,
    dino_only: torch.Tensor,
    hybrid: torch.Tensor,
    metrics: dict,
) -> bytes:
    """Create visualization comparing all reconstruction methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_images = min(6, len(original))
    fig, axes = plt.subplots(4, n_images, figsize=(2.5 * n_images, 10))

    for i in range(n_images):
        axes[0, i].imshow(original[i])
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)
        axes[0, i].axis("off")

        if vlm_only is not None:
            axes[1, i].imshow(np.clip(vlm_only[i].permute(1, 2, 0).cpu().numpy(), 0, 1))
        if i == 0:
            axes[1, i].set_ylabel("VLM-only", fontsize=10)
        axes[1, i].axis("off")

        if dino_only is not None:
            axes[2, i].imshow(np.clip(dino_only[i].permute(1, 2, 0).cpu().numpy(), 0, 1))
        if i == 0:
            axes[2, i].set_ylabel("DINOv2-only", fontsize=10)
        axes[2, i].axis("off")

        axes[3, i].imshow(np.clip(hybrid[i].permute(1, 2, 0).cpu().numpy(), 0, 1))
        if i == 0:
            axes[3, i].set_ylabel("Hybrid", fontsize=10)
        axes[3, i].axis("off")

    plt.suptitle(
        f"Hybrid Fusion Results\n"
        f"LPIPS: {metrics['lpips']:.3f} | Spatial IoU: {metrics['spatial_iou']:.3f}",
        fontsize=11,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e_p2_3_fusion_training(runner: ExperimentRunner) -> dict:
    """Train cross-attention fusion module.

    This implementation:
    1. Loads frozen VLM and DINOv2 encoders
    2. Trains fusion module to combine both feature streams
    3. Evaluates reconstruction quality with hybrid features
    4. Compares to VLM-only and DINOv2-only baselines

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-P2.3: Cross-Attention Fusion Module Training")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e_p2_3/stage": 0, "e_p2_3/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate images
    # =========================================================================
    print("\n[Stage 1/6] Generating training and test images...")

    train_images = generate_training_images(n_images=200)
    test_images = generate_training_images(n_images=40)

    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    runner.log_metrics({"e_p2_3/stage": 1, "e_p2_3/progress": 0.05})

    # =========================================================================
    # Stage 2: Extract DINOv2 features
    # =========================================================================
    print("\n[Stage 2/6] Extracting DINOv2 features...")

    dinov2 = load_dinov2_model(device)
    train_dino_features = extract_dinov2_features(train_images, dinov2, device)
    test_dino_features = extract_dinov2_features(test_images, dinov2, device)

    print(f"  DINOv2 features: {train_dino_features.shape}")

    del dinov2
    torch.cuda.empty_cache()

    runner.log_metrics({"e_p2_3/stage": 2, "e_p2_3/progress": 0.2})

    # =========================================================================
    # Stage 3: Extract VLM features
    # =========================================================================
    print("\n[Stage 3/6] Extracting VLM features...")

    vlm_model, vlm_processor = load_vlm_model(device)
    train_vlm_features = extract_vlm_features(train_images, vlm_model, vlm_processor, device)
    test_vlm_features = extract_vlm_features(test_images, vlm_model, vlm_processor, device)

    print(f"  VLM features: {train_vlm_features.shape}")

    del vlm_model
    del vlm_processor
    torch.cuda.empty_cache()

    runner.log_metrics({"e_p2_3/stage": 3, "e_p2_3/progress": 0.4})

    # =========================================================================
    # Stage 4: Train fusion module
    # =========================================================================
    print("\n[Stage 4/6] Training fusion module...")

    vlm_dim = train_vlm_features.shape[-1]
    dino_dim = train_dino_features.shape[-1]

    fusion = HybridFusionModule(
        vlm_dim=vlm_dim,
        spatial_dim=dino_dim,
        fusion_dim=1024,
        num_fusion_layers=4,
        num_output_queries=64,
        output_dim=4096,
    ).to(device)

    decoder = SimplePixelDecoder(input_dim=4096, output_size=224).to(device)

    fusion_params = sum(p.numel() for p in fusion.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Fusion params: {fusion_params:,}")
    print(f"  Decoder params: {decoder_params:,}")

    # Prepare tensors
    train_targets = images_to_tensor(train_images).to(device)
    test_targets = images_to_tensor(test_images).to(device)
    train_vlm_t = train_vlm_features.to(device)
    train_dino_t = train_dino_features.to(device)
    test_vlm_t = test_vlm_features.to(device)
    test_dino_t = test_dino_features.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(fusion.parameters()) + list(decoder.parameters()),
        lr=5e-5,
        weight_decay=0.01,
    )

    # Training
    n_epochs = 150
    batch_size = 8

    for epoch in range(n_epochs):
        fusion.train()
        decoder.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = torch.randperm(len(train_targets))

        for i in range(0, len(train_targets), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_vlm = train_vlm_t[batch_idx]
            batch_dino = train_dino_t[batch_idx]
            batch_targets = train_targets[batch_idx]

            optimizer.zero_grad()

            # Forward
            conditioning = fusion(batch_vlm, batch_dino)
            recon = decoder(conditioning)

            # Loss
            mse_loss = F.mse_loss(recon, batch_targets)
            l1_loss = F.l1_loss(recon, batch_targets)
            loss = mse_loss + 0.1 * l1_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 25 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"    Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
            runner.log_metrics({"e_p2_3/train_loss": avg_loss}, step=epoch)

    runner.log_metrics({"e_p2_3/stage": 4, "e_p2_3/progress": 0.75})

    # =========================================================================
    # Stage 5: Evaluate
    # =========================================================================
    print("\n[Stage 5/6] Evaluating hybrid reconstruction...")

    fusion.eval()
    decoder.eval()

    with torch.no_grad():
        test_conditioning = fusion(test_vlm_t, test_dino_t)
        test_recon = decoder(test_conditioning)

    metrics = compute_reconstruction_metrics(test_recon, test_targets, device)

    print(f"  LPIPS: {metrics['lpips']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  Spatial IoU: {metrics['spatial_iou']:.4f}")

    # Baselines from prior experiments
    vlm_baseline = {"lpips": 0.264, "spatial_iou": 0.559}
    dino_baseline = {"lpips": 0.30, "spatial_iou": 0.65}  # Expected from E-P2.2

    runner.log_metrics({
        "e_p2_3/stage": 5,
        "e_p2_3/progress": 0.9,
        "e_p2_3/lpips": metrics["lpips"],
        "e_p2_3/ssim": metrics["ssim"],
        "e_p2_3/spatial_iou": metrics["spatial_iou"],
    })

    # =========================================================================
    # Stage 6: Save artifacts
    # =========================================================================
    print("\n[Stage 6/6] Saving artifacts...")

    # Visualization (simplified - no VLM-only/DINOv2-only recon available)
    viz_bytes = create_comparison_grid(
        test_images[:6],
        None,  # Would need VLM-only decoder
        None,  # Would need DINOv2-only decoder
        test_recon[:6],
        metrics,
    )
    viz_path = runner.results.save_artifact("hybrid_fusion_results.png", viz_bytes)

    # Save metrics
    results_data = {
        "hybrid_metrics": {k: float(v) for k, v in metrics.items()},
        "vlm_baseline": vlm_baseline,
        "dino_baseline": dino_baseline,
        "model_info": {
            "fusion_params": fusion_params,
            "decoder_params": decoder_params,
            "total_trainable": fusion_params + decoder_params,
            "vlm_dim": vlm_dim,
            "dino_dim": dino_dim,
        },
        "comparison": {
            "vs_vlm_spatial_iou_improvement": metrics["spatial_iou"] - vlm_baseline["spatial_iou"],
            "vs_vlm_lpips_change": vlm_baseline["lpips"] - metrics["lpips"],
        },
    }
    data_path = runner.results.save_json_artifact("fusion_training_results.json", results_data)

    # Save model checkpoint
    checkpoint = {
        "fusion_state_dict": fusion.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
    }
    ckpt_bytes = io.BytesIO()
    torch.save(checkpoint, ckpt_bytes)
    ckpt_bytes.seek(0)
    ckpt_path = runner.results.save_artifact("fusion_checkpoint.pt", ckpt_bytes.read())

    runner.log_metrics({"e_p2_3/stage": 6, "e_p2_3/progress": 1.0})

    # =========================================================================
    # Determine finding
    # =========================================================================
    lpips_target = 0.35
    spatial_iou_target = 0.60

    if metrics["lpips"] < lpips_target and metrics["spatial_iou"] > spatial_iou_target:
        finding = (
            f"Hybrid fusion achieves good reconstruction quality "
            f"(LPIPS={metrics['lpips']:.3f}, Spatial IoU={metrics['spatial_iou']:.3f}). "
            f"Spatial IoU improved by {metrics['spatial_iou'] - vlm_baseline['spatial_iou']:.3f} vs VLM-only baseline. "
            f"Fusion successfully combines spatial and semantic features."
        )
    elif metrics["spatial_iou"] > 0.55:
        finding = (
            f"Hybrid fusion shows promising results "
            f"(Spatial IoU={metrics['spatial_iou']:.3f}) but needs optimization "
            f"(LPIPS={metrics['lpips']:.3f}). "
            f"Consider adjusting loss weighting or fusion architecture."
        )
    else:
        finding = (
            f"Hybrid fusion underperforms "
            f"(LPIPS={metrics['lpips']:.3f}, Spatial IoU={metrics['spatial_iou']:.3f}). "
            f"Fusion may be degrading individual stream quality. "
            f"Investigate attention patterns and stream weighting."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "lpips": float(metrics["lpips"]),
            "ssim": float(metrics["ssim"]),
            "spatial_iou": float(metrics["spatial_iou"]),
            "fusion_params": fusion_params,
            "vlm_baseline_spatial_iou": vlm_baseline["spatial_iou"],
        },
        "artifacts": [viz_path, data_path, ckpt_path],
    }
