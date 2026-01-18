"""E1.2: Linear Reconstruction Probe

Objective: Measure upper bound on information content with a simple linear probe.

Protocol:
1. Train a linear decoder: latents -> pixels
2. Evaluate reconstruction quality on held-out images
3. Compare reconstruction quality metrics (LPIPS, SSIM, PSNR)

Success Metrics:
- LPIPS < 0.35 (acceptable: < 0.45)
- SSIM > 0.75 (target: > 0.85)
- PSNR > 22 dB (target: > 25 dB)

This experiment tests whether a simple linear projection from VLM latents
can recover the original image, establishing a baseline for information content.
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from runner import ExperimentRunner


class LinearReconstructionProbe(nn.Module):
    """Simple linear projection from VLM latents to pixel space."""

    def __init__(self, latent_dim: int = 3584, output_size: tuple = (224, 224, 3)):
        super().__init__()
        self.output_size = output_size
        self.latent_dim = latent_dim
        flat_output = output_size[0] * output_size[1] * output_size[2]

        # Linear projection from pooled latents to full image
        self.proj = nn.Linear(latent_dim, flat_output)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, latent_dim] pooled latent vectors

        Returns:
            [B, C, H, W] reconstructed images
        """
        # Project to flat pixel space
        flat = self.proj(latents)  # [B, H*W*C]

        # Reshape to image
        B = latents.shape[0]
        H, W, C = self.output_size
        img = flat.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # Sigmoid to ensure [0, 1] range
        return torch.sigmoid(img)


def e1_2_reconstruction_probe(runner: ExperimentRunner) -> dict:
    """Run linear reconstruction probe sub-experiment.

    This implementation:
    1. Generates synthetic test images
    2. Extracts VLM latents
    3. Trains a linear probe to reconstruct images
    4. Evaluates reconstruction quality (LPIPS, SSIM, PSNR)

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E1.2: Linear Reconstruction Probe")
    print("=" * 60)

    # Set up environment
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e1_2/stage": 0, "e1_2/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate synthetic training and test images
    # =========================================================================
    print("\n[Stage 1/5] Generating synthetic images...")

    train_images, train_labels = generate_training_images(n_images=200)
    test_images, test_labels = generate_training_images(n_images=50)

    print(f"  Generated {len(train_images)} training images")
    print(f"  Generated {len(test_images)} test images")

    runner.log_metrics({
        "e1_2/stage": 1,
        "e1_2/progress": 0.1,
        "e1_2/n_train": len(train_images),
        "e1_2/n_test": len(test_images),
    })

    # =========================================================================
    # Stage 2: Extract latents from VLM
    # =========================================================================
    print("\n[Stage 2/5] Extracting VLM latents...")

    train_latents = extract_latents_batch(train_images, runner, prefix="train")
    test_latents = extract_latents_batch(test_images, runner, prefix="test")

    print(f"  Train latents shape: {train_latents.shape}")
    print(f"  Test latents shape: {test_latents.shape}")

    runner.log_metrics({
        "e1_2/stage": 2,
        "e1_2/progress": 0.4,
        "e1_2/latent_dim": train_latents.shape[-1],
    })

    # =========================================================================
    # Stage 3: Train linear reconstruction probe
    # =========================================================================
    print("\n[Stage 3/5] Training linear reconstruction probe...")

    # Convert images to tensors
    train_targets = images_to_tensor(train_images).to(device)
    test_targets = images_to_tensor(test_images).to(device)
    train_latents_t = torch.tensor(train_latents, dtype=torch.float32).to(device)
    test_latents_t = torch.tensor(test_latents, dtype=torch.float32).to(device)

    # Initialize probe
    probe = LinearReconstructionProbe(
        latent_dim=train_latents.shape[-1],
        output_size=(224, 224, 3),
    ).to(device)

    print(f"  Probe parameters: {sum(p.numel() for p in probe.parameters()):,}")

    # Training
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    n_epochs = 100
    batch_size = 32
    best_loss = float("inf")

    for epoch in range(n_epochs):
        probe.train()
        epoch_loss = 0.0
        n_batches = 0

        # Simple batch iteration
        for i in range(0, len(train_latents_t), batch_size):
            batch_latents = train_latents_t[i : i + batch_size]
            batch_targets = train_targets[i : i + batch_size]

            optimizer.zero_grad()

            # Forward
            recon = probe(batch_latents)

            # MSE loss
            loss = F.mse_loss(recon, batch_targets)

            # Backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if epoch % 20 == 0:
            print(f"    Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}")
            runner.log_metrics({
                "e1_2/train_loss": avg_loss,
                "e1_2/epoch": epoch,
            }, step=epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f"  Training complete. Best loss: {best_loss:.4f}")

    runner.log_metrics({"e1_2/stage": 3, "e1_2/progress": 0.7})

    # =========================================================================
    # Stage 4: Evaluate reconstruction quality
    # =========================================================================
    print("\n[Stage 4/5] Evaluating reconstruction quality...")

    probe.eval()
    with torch.no_grad():
        test_recon = probe(test_latents_t)

    # Compute metrics
    metrics = compute_reconstruction_metrics(test_recon, test_targets, device)

    print(f"  LPIPS: {metrics['lpips']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  MSE: {metrics['mse']:.6f}")

    runner.log_metrics({
        "e1_2/stage": 4,
        "e1_2/progress": 0.9,
        "e1_2/lpips": metrics["lpips"],
        "e1_2/ssim": metrics["ssim"],
        "e1_2/psnr": metrics["psnr"],
        "e1_2/mse": metrics["mse"],
    })

    # =========================================================================
    # Stage 5: Save artifacts
    # =========================================================================
    print("\n[Stage 5/5] Saving artifacts...")

    # Create visualization grid
    grid_bytes = create_reconstruction_grid(
        test_images[:8],
        test_recon[:8].cpu(),
        metrics,
    )
    grid_path = runner.results.save_artifact("reconstruction_grid.png", grid_bytes)
    print(f"  Saved reconstruction grid: {grid_path}")

    # Save metrics
    metrics_data = {
        "lpips": float(metrics["lpips"]),
        "ssim": float(metrics["ssim"]),
        "psnr": float(metrics["psnr"]),
        "mse": float(metrics["mse"]),
        "probe_params": sum(p.numel() for p in probe.parameters()),
        "latent_dim": int(train_latents.shape[-1]),
        "n_train": len(train_images),
        "n_test": len(test_images),
        "n_epochs": n_epochs,
    }
    metrics_path = runner.results.save_json_artifact("reconstruction_metrics.json", metrics_data)

    runner.log_metrics({"e1_2/stage": 5, "e1_2/progress": 1.0})

    # =========================================================================
    # Return results
    # =========================================================================
    # Assess against success criteria
    lpips_target = 0.35
    ssim_target = 0.75

    if metrics["lpips"] < lpips_target and metrics["ssim"] > ssim_target:
        finding = (
            f"Linear probe achieves acceptable reconstruction quality "
            f"(LPIPS={metrics['lpips']:.3f} < {lpips_target}, SSIM={metrics['ssim']:.3f} > {ssim_target}). "
            f"VLM latents contain sufficient information for basic reconstruction."
        )
    elif metrics["lpips"] < 0.45:
        finding = (
            f"Linear probe achieves marginal reconstruction quality "
            f"(LPIPS={metrics['lpips']:.3f}, SSIM={metrics['ssim']:.3f}). "
            f"Non-linear decoder may be needed for better results."
        )
    else:
        finding = (
            f"Linear probe fails to reconstruct images adequately "
            f"(LPIPS={metrics['lpips']:.3f}, SSIM={metrics['ssim']:.3f}). "
            f"Information may be highly entangled in VLM latents."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "lpips": float(metrics["lpips"]),
            "ssim": float(metrics["ssim"]),
            "psnr": float(metrics["psnr"]),
            "mse": float(metrics["mse"]),
        },
        "artifacts": [grid_path, metrics_path],
    }


def generate_training_images(n_images: int = 200) -> tuple[list, list]:
    """Generate diverse synthetic training images.

    Creates colored shapes at various positions:
    - Circles, squares, triangles in different colors
    - Varying positions and sizes

    Args:
        n_images: Total number of images to generate

    Returns:
        Tuple of (images, labels)
    """
    from PIL import Image, ImageDraw

    shapes = ["circle", "square", "triangle"]
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
    ]

    images = []
    labels = []

    np.random.seed(42)
    for i in range(n_images):
        # Random shape and color
        shape = shapes[i % len(shapes)]
        color = colors[i % len(colors)]
        label = shapes.index(shape)

        # Create white background
        img = Image.new("RGB", (224, 224), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Random position and size
        cx = np.random.randint(60, 164)
        cy = np.random.randint(60, 164)
        size = np.random.randint(30, 50)

        # Draw shape
        if shape == "circle":
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif shape == "square":
            draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif shape == "triangle":
            points = [
                (cx, cy - size),
                (cx - size, cy + size),
                (cx + size, cy + size),
            ]
            draw.polygon(points, fill=color)

        images.append(img)
        labels.append(label)

    return images, labels


def extract_latents_batch(images: list, runner: ExperimentRunner, prefix: str = "") -> np.ndarray:
    """Extract latents from VLM for a batch of images.

    Args:
        images: List of PIL Images
        runner: ExperimentRunner for logging
        prefix: Prefix for log metrics

    Returns:
        Numpy array of shape [N_images, latent_dim]
    """
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
            batch_images = images[i : i + batch_size]

            for img in batch_images:
                # Format for Qwen2.5-VL
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

                # Get last hidden state, pool across sequence
                hidden_states = outputs.hidden_states[-1]
                latent = hidden_states[0].float().cpu().numpy()
                latent_pooled = latent.mean(axis=0, keepdims=True)
                latents_list.append(latent_pooled)

            progress = min(1.0, (i + batch_size) / len(images))
            if prefix:
                runner.log_metrics({f"e1_2/{prefix}_extraction_progress": progress})
            print(f"    Processed {min(i + batch_size, len(images))}/{len(images)} images")

    latents = np.concatenate(latents_list, axis=0)

    # Clean up
    del model
    del processor
    torch.cuda.empty_cache()

    return latents


def images_to_tensor(images: list) -> torch.Tensor:
    """Convert list of PIL images to tensor.

    Args:
        images: List of PIL Images

    Returns:
        Tensor of shape [N, C, H, W] with values in [0, 1]
    """
    tensors = []
    for img in images:
        # Convert to numpy, normalize to [0, 1]
        arr = np.array(img).astype(np.float32) / 255.0
        # [H, W, C] -> [C, H, W]
        arr = arr.transpose(2, 0, 1)
        tensors.append(torch.tensor(arr))

    return torch.stack(tensors)


def compute_reconstruction_metrics(
    recon: torch.Tensor, target: torch.Tensor, device: torch.device
) -> dict:
    """Compute reconstruction quality metrics.

    Args:
        recon: Reconstructed images [N, C, H, W] in [0, 1]
        target: Target images [N, C, H, W] in [0, 1]
        device: Torch device

    Returns:
        Dict with lpips, ssim, psnr, mse
    """
    import lpips

    # LPIPS (perceptual similarity)
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    # LPIPS expects values in [-1, 1]
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

    # PSNR
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

    # SSIM (simple version)
    ssim_val = compute_ssim(recon, target)

    return {
        "lpips": lpips_val,
        "ssim": ssim_val,
        "psnr": psnr,
        "mse": mse,
    }


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Compute SSIM between two batches of images.

    Args:
        img1: First batch [N, C, H, W]
        img2: Second batch [N, C, H, W]
        window_size: Size of Gaussian window

    Returns:
        Mean SSIM value
    """
    from scipy.ndimage import gaussian_filter

    ssim_values = []

    for i in range(len(img1)):
        # Convert to numpy, average across channels
        im1 = img1[i].cpu().numpy().mean(axis=0)
        im2 = img2[i].cpu().numpy().mean(axis=0)

        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Compute means
        mu1 = gaussian_filter(im1, sigma=1.5)
        mu2 = gaussian_filter(im2, sigma=1.5)

        # Compute variances and covariance
        sigma1_sq = gaussian_filter(im1 ** 2, sigma=1.5) - mu1 ** 2
        sigma2_sq = gaussian_filter(im2 ** 2, sigma=1.5) - mu2 ** 2
        sigma12 = gaussian_filter(im1 * im2, sigma=1.5) - mu1 * mu2

        # SSIM formula
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
    """Create a visualization grid comparing original and reconstructed images.

    Args:
        original_images: List of original PIL images
        reconstructed: Reconstructed images tensor [N, C, H, W]
        metrics: Dictionary of metrics to display

    Returns:
        PNG image as bytes
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_images = len(original_images)
    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 5))

    for i in range(n_images):
        # Original
        axes[0, i].imshow(original_images[i])
        axes[0, i].set_title("Original" if i == 0 else "")
        axes[0, i].axis("off")

        # Reconstructed
        recon_img = reconstructed[i].permute(1, 2, 0).numpy()
        recon_img = np.clip(recon_img, 0, 1)
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title("Reconstructed" if i == 0 else "")
        axes[1, i].axis("off")

    plt.suptitle(
        f"Linear Reconstruction Probe Results\n"
        f"LPIPS: {metrics['lpips']:.3f} | SSIM: {metrics['ssim']:.3f} | PSNR: {metrics['psnr']:.1f} dB",
        fontsize=12,
    )
    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
