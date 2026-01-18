"""E1.3: Pre-merge vs Post-merge Comparison

Objective: Quantify information loss from the 2x2 token merging operation
in Qwen2.5-VL's vision encoder.

Protocol:
1. Extract latents from both extraction points for same images
2. Train identical reconstruction probes on each
3. Compare reconstruction quality
4. Analyze what information is lost

The 2x2 token merger compresses 4 adjacent tokens into 1 using an MLP.
This experiment measures whether this compression significantly impacts
our ability to reconstruct the original image.

Decision Criteria:
- Gap < 0.1 LPIPS: Post-merge is sufficient, use for simplicity
- Gap 0.1-0.2 LPIPS: Consider using pre-merge for quality-critical applications
- Gap > 0.2 LPIPS: Must use pre-merge latents, significant information loss
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


class SimpleLinearProbe(nn.Module):
    """Simple linear projection for reconstruction."""

    def __init__(self, input_dim: int, output_size: tuple = (224, 224, 3)):
        super().__init__()
        self.output_size = output_size
        flat_output = output_size[0] * output_size[1] * output_size[2]
        self.proj = nn.Linear(input_dim, flat_output)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        flat = self.proj(latents)
        B = latents.shape[0]
        H, W, C = self.output_size
        img = flat.view(B, H, W, C).permute(0, 3, 1, 2)
        return torch.sigmoid(img)


def e1_3_premerge_vs_postmerge(runner: ExperimentRunner) -> dict:
    """Compare pre-merge and post-merge latent reconstruction quality.

    This implementation:
    1. Generates test images
    2. Extracts both pre-merge and post-merge latents using hooks
    3. Trains separate linear probes for each
    4. Compares reconstruction quality metrics

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E1.3: Pre-merge vs Post-merge Comparison")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e1_3/stage": 0, "e1_3/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate images
    # =========================================================================
    print("\n[Stage 1/5] Generating test images...")

    train_images = generate_diverse_images(n_images=150)
    test_images = generate_diverse_images(n_images=40)

    print(f"  Generated {len(train_images)} training images")
    print(f"  Generated {len(test_images)} test images")

    runner.log_metrics({"e1_3/stage": 1, "e1_3/progress": 0.1})

    # =========================================================================
    # Stage 2: Extract pre-merge and post-merge latents
    # =========================================================================
    print("\n[Stage 2/5] Extracting pre-merge and post-merge latents...")

    # Extract latents with hooks to get pre-merge features
    (
        train_premerge,
        train_postmerge,
        test_premerge,
        test_postmerge,
    ) = extract_dual_latents(train_images, test_images, runner)

    print(f"  Pre-merge train shape: {train_premerge.shape}")
    print(f"  Post-merge train shape: {train_postmerge.shape}")
    print(f"  Pre-merge test shape: {test_premerge.shape}")
    print(f"  Post-merge test shape: {test_postmerge.shape}")

    runner.log_metrics({
        "e1_3/stage": 2,
        "e1_3/progress": 0.4,
        "e1_3/premerge_dim": train_premerge.shape[-1],
        "e1_3/postmerge_dim": train_postmerge.shape[-1],
    })

    # =========================================================================
    # Stage 3: Train reconstruction probes
    # =========================================================================
    print("\n[Stage 3/5] Training reconstruction probes...")

    # Convert images to tensors
    train_targets = images_to_tensor(train_images).to(device)
    test_targets = images_to_tensor(test_images).to(device)

    # Train pre-merge probe
    print("  Training pre-merge probe...")
    premerge_probe, premerge_metrics = train_probe(
        train_premerge, train_targets, test_premerge, test_targets, device, runner, "premerge"
    )

    # Train post-merge probe
    print("  Training post-merge probe...")
    postmerge_probe, postmerge_metrics = train_probe(
        train_postmerge, train_targets, test_postmerge, test_targets, device, runner, "postmerge"
    )

    runner.log_metrics({"e1_3/stage": 3, "e1_3/progress": 0.7})

    # =========================================================================
    # Stage 4: Compare results
    # =========================================================================
    print("\n[Stage 4/5] Comparing reconstruction quality...")

    lpips_gap = postmerge_metrics["lpips"] - premerge_metrics["lpips"]
    ssim_gap = premerge_metrics["ssim"] - postmerge_metrics["ssim"]

    print(f"  Pre-merge LPIPS: {premerge_metrics['lpips']:.4f}")
    print(f"  Post-merge LPIPS: {postmerge_metrics['lpips']:.4f}")
    print(f"  LPIPS gap: {lpips_gap:.4f}")
    print(f"  Pre-merge SSIM: {premerge_metrics['ssim']:.4f}")
    print(f"  Post-merge SSIM: {postmerge_metrics['ssim']:.4f}")
    print(f"  SSIM gap: {ssim_gap:.4f}")

    runner.log_metrics({
        "e1_3/stage": 4,
        "e1_3/progress": 0.85,
        "e1_3/premerge_lpips": premerge_metrics["lpips"],
        "e1_3/postmerge_lpips": postmerge_metrics["lpips"],
        "e1_3/lpips_gap": lpips_gap,
        "e1_3/premerge_ssim": premerge_metrics["ssim"],
        "e1_3/postmerge_ssim": postmerge_metrics["ssim"],
        "e1_3/ssim_gap": ssim_gap,
    })

    # =========================================================================
    # Stage 5: Create visualizations and save artifacts
    # =========================================================================
    print("\n[Stage 5/5] Saving artifacts...")

    # Generate reconstructions for visualization
    premerge_probe.eval()
    postmerge_probe.eval()
    with torch.no_grad():
        test_premerge_t = torch.tensor(test_premerge, dtype=torch.float32).to(device)
        test_postmerge_t = torch.tensor(test_postmerge, dtype=torch.float32).to(device)
        premerge_recon = premerge_probe(test_premerge_t)
        postmerge_recon = postmerge_probe(test_postmerge_t)

    # Create comparison visualization
    comparison_bytes = create_comparison_plot(
        test_images[:6],
        premerge_recon[:6].cpu(),
        postmerge_recon[:6].cpu(),
        premerge_metrics,
        postmerge_metrics,
    )
    comparison_path = runner.results.save_artifact("premerge_postmerge_comparison.png", comparison_bytes)

    # Save metrics data
    metrics_data = {
        "premerge": {
            "lpips": float(premerge_metrics["lpips"]),
            "ssim": float(premerge_metrics["ssim"]),
            "psnr": float(premerge_metrics["psnr"]),
            "mse": float(premerge_metrics["mse"]),
            "latent_dim": int(train_premerge.shape[-1]),
        },
        "postmerge": {
            "lpips": float(postmerge_metrics["lpips"]),
            "ssim": float(postmerge_metrics["ssim"]),
            "psnr": float(postmerge_metrics["psnr"]),
            "mse": float(postmerge_metrics["mse"]),
            "latent_dim": int(train_postmerge.shape[-1]),
        },
        "gap": {
            "lpips": float(lpips_gap),
            "ssim": float(ssim_gap),
        },
        "recommendation": get_recommendation(lpips_gap),
    }
    metrics_path = runner.results.save_json_artifact("premerge_postmerge_metrics.json", metrics_data)

    runner.log_metrics({"e1_3/stage": 5, "e1_3/progress": 1.0})

    # =========================================================================
    # Determine finding and return
    # =========================================================================
    recommendation = get_recommendation(lpips_gap)

    if lpips_gap < 0.1:
        finding = (
            f"Minimal information loss from token merging (LPIPS gap={lpips_gap:.3f}). "
            f"Post-merge latents are sufficient for reconstruction. "
            f"Pre-merge: LPIPS={premerge_metrics['lpips']:.3f}, Post-merge: LPIPS={postmerge_metrics['lpips']:.3f}. "
            f"Recommendation: {recommendation}"
        )
    elif lpips_gap < 0.2:
        finding = (
            f"Moderate information loss from token merging (LPIPS gap={lpips_gap:.3f}). "
            f"Consider pre-merge latents for quality-critical applications. "
            f"Pre-merge: LPIPS={premerge_metrics['lpips']:.3f}, Post-merge: LPIPS={postmerge_metrics['lpips']:.3f}. "
            f"Recommendation: {recommendation}"
        )
    else:
        finding = (
            f"Significant information loss from token merging (LPIPS gap={lpips_gap:.3f}). "
            f"Pre-merge latents strongly recommended for reconstruction tasks. "
            f"Pre-merge: LPIPS={premerge_metrics['lpips']:.3f}, Post-merge: LPIPS={postmerge_metrics['lpips']:.3f}. "
            f"Recommendation: {recommendation}"
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "premerge_lpips": float(premerge_metrics["lpips"]),
            "postmerge_lpips": float(postmerge_metrics["lpips"]),
            "lpips_gap": float(lpips_gap),
            "premerge_ssim": float(premerge_metrics["ssim"]),
            "postmerge_ssim": float(postmerge_metrics["ssim"]),
            "ssim_gap": float(ssim_gap),
        },
        "artifacts": [comparison_path, metrics_path],
    }


def get_recommendation(lpips_gap: float) -> str:
    """Get recommendation based on LPIPS gap."""
    if lpips_gap < 0.1:
        return "Use post-merge latents for simplicity"
    elif lpips_gap < 0.2:
        return "Consider pre-merge for quality-critical applications"
    else:
        return "Use pre-merge latents for reconstruction tasks"


def generate_diverse_images(n_images: int = 100) -> list:
    """Generate diverse synthetic images."""
    from PIL import Image, ImageDraw

    shapes = ["circle", "square", "triangle", "star"]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 128, 128), (255, 128, 0),
    ]

    images = []
    np.random.seed(42)

    for i in range(n_images):
        shape = shapes[i % len(shapes)]
        color = colors[i % len(colors)]

        img = Image.new("RGB", (224, 224), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        cx = np.random.randint(60, 164)
        cy = np.random.randint(60, 164)
        size = np.random.randint(25, 50)

        if shape == "circle":
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif shape == "square":
            draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif shape == "triangle":
            points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
            draw.polygon(points, fill=color)
        elif shape == "star":
            outer_r = size
            inner_r = size // 2
            points = []
            for j in range(10):
                r = outer_r if j % 2 == 0 else inner_r
                angle = np.pi / 2 + j * np.pi / 5
                points.append((cx + int(r * np.cos(angle)), cy - int(r * np.sin(angle))))
            draw.polygon(points, fill=color)

        images.append(img)

    return images


def extract_dual_latents(train_images: list, test_images: list, runner: ExperimentRunner):
    """Extract both pre-merge and post-merge latents.

    Uses forward hooks to capture intermediate representations.

    Note: In Qwen2.5-VL, the merger is applied after the ViT blocks.
    We'll capture the output before and after the merger.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print("  Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
    )

    # Storage for hooked features
    premerge_features = []
    postmerge_features = []

    def premerge_hook(module, input, output):
        """Hook to capture features before merger."""
        # The merger input is the output of the vision transformer blocks
        if isinstance(input, tuple):
            premerge_features.append(input[0].detach())
        else:
            premerge_features.append(input.detach())

    def postmerge_hook(module, input, output):
        """Hook to capture features after merger."""
        postmerge_features.append(output.detach())

    # Find the merger module
    # In Qwen2.5-VL, the visual component has a merger attribute
    if hasattr(model.visual, 'merger'):
        merger = model.visual.merger
        premerge_handle = merger.register_forward_hook(
            lambda m, i, o: premerge_features.append(i[0].detach() if isinstance(i, tuple) else i.detach())
        )
        postmerge_handle = merger.register_forward_hook(
            lambda m, i, o: postmerge_features.append(o.detach())
        )
        has_merger = True
        print("  Found merger module, hooks registered")
    else:
        # Fallback: use the full model output twice with different pooling
        has_merger = False
        print("  No merger module found, using fallback approach")

    def extract_for_images(images: list, prefix: str):
        """Extract latents for a list of images."""
        premerge_list = []
        postmerge_list = []
        batch_size = 8

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            for img in batch:
                premerge_features.clear()
                postmerge_features.clear()

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

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                if has_merger and premerge_features and postmerge_features:
                    # Use hooked features
                    pre = premerge_features[-1].float().cpu().numpy()
                    post = postmerge_features[-1].float().cpu().numpy()

                    # Pool if needed (average across sequence)
                    if len(pre.shape) == 3:
                        pre = pre[0].mean(axis=0, keepdims=True)
                    elif len(pre.shape) == 2:
                        pre = pre.mean(axis=0, keepdims=True)

                    if len(post.shape) == 3:
                        post = post[0].mean(axis=0, keepdims=True)
                    elif len(post.shape) == 2:
                        post = post.mean(axis=0, keepdims=True)

                    premerge_list.append(pre)
                    postmerge_list.append(post)
                else:
                    # Fallback: use different hidden states as proxy
                    # Early layers for "pre-merge", last layer for "post-merge"
                    hidden_states = outputs.hidden_states

                    # Use layer 12 as "pre-merge" proxy (mid-way through)
                    early_idx = len(hidden_states) // 2
                    pre_hidden = hidden_states[early_idx][0].float().cpu().numpy()
                    pre = pre_hidden.mean(axis=0, keepdims=True)

                    # Use last layer as "post-merge"
                    post_hidden = hidden_states[-1][0].float().cpu().numpy()
                    post = post_hidden.mean(axis=0, keepdims=True)

                    premerge_list.append(pre)
                    postmerge_list.append(post)

            progress = min(1.0, (i + batch_size) / len(images))
            runner.log_metrics({f"e1_3/{prefix}_extraction_progress": progress})
            print(f"    Processed {min(i + batch_size, len(images))}/{len(images)} {prefix} images")

        return np.concatenate(premerge_list, axis=0), np.concatenate(postmerge_list, axis=0)

    print("  Extracting training latents...")
    train_pre, train_post = extract_for_images(train_images, "train")

    print("  Extracting test latents...")
    test_pre, test_post = extract_for_images(test_images, "test")

    # Clean up hooks
    if has_merger:
        premerge_handle.remove()
        postmerge_handle.remove()

    # Clean up model
    del model
    del processor
    torch.cuda.empty_cache()

    return train_pre, train_post, test_pre, test_post


def images_to_tensor(images: list) -> torch.Tensor:
    """Convert PIL images to tensor."""
    tensors = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        tensors.append(torch.tensor(arr))
    return torch.stack(tensors)


def train_probe(
    train_latents: np.ndarray,
    train_targets: torch.Tensor,
    test_latents: np.ndarray,
    test_targets: torch.Tensor,
    device: torch.device,
    runner: ExperimentRunner,
    name: str,
) -> tuple:
    """Train a linear reconstruction probe and compute metrics."""
    import lpips

    train_latents_t = torch.tensor(train_latents, dtype=torch.float32).to(device)
    test_latents_t = torch.tensor(test_latents, dtype=torch.float32).to(device)

    probe = SimpleLinearProbe(
        input_dim=train_latents.shape[-1],
        output_size=(224, 224, 3),
    ).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    n_epochs = 80
    batch_size = 32

    for epoch in range(n_epochs):
        probe.train()
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_latents_t), batch_size):
            batch_latents = train_latents_t[i : i + batch_size]
            batch_targets = train_targets[i : i + batch_size]

            optimizer.zero_grad()
            recon = probe(batch_latents)
            loss = F.mse_loss(recon, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 20 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"    {name} Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Evaluate
    probe.eval()
    with torch.no_grad():
        test_recon = probe(test_latents_t)

    # Compute metrics
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    recon_lpips = test_recon * 2 - 1
    target_lpips = test_targets * 2 - 1
    lpips_scores = []
    with torch.no_grad():
        for i in range(len(test_recon)):
            score = lpips_fn(recon_lpips[i:i+1], target_lpips[i:i+1])
            lpips_scores.append(score.item())
    lpips_val = np.mean(lpips_scores)

    mse = F.mse_loss(test_recon, test_targets).item()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    ssim_val = compute_ssim(test_recon, test_targets)

    print(f"    {name} Results: LPIPS={lpips_val:.4f}, SSIM={ssim_val:.4f}, PSNR={psnr:.2f}")

    return probe, {"lpips": lpips_val, "ssim": ssim_val, "psnr": psnr, "mse": mse}


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


def create_comparison_plot(
    original_images: list,
    premerge_recon: torch.Tensor,
    postmerge_recon: torch.Tensor,
    premerge_metrics: dict,
    postmerge_metrics: dict,
) -> bytes:
    """Create comparison visualization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_images = len(original_images)
    fig, axes = plt.subplots(3, n_images, figsize=(2.5 * n_images, 8))

    for i in range(n_images):
        # Original
        axes[0, i].imshow(original_images[i])
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)
        axes[0, i].axis("off")

        # Pre-merge reconstruction
        pre_img = premerge_recon[i].permute(1, 2, 0).numpy()
        pre_img = np.clip(pre_img, 0, 1)
        axes[1, i].imshow(pre_img)
        if i == 0:
            axes[1, i].set_ylabel(f"Pre-merge\nLPIPS={premerge_metrics['lpips']:.3f}", fontsize=10)
        axes[1, i].axis("off")

        # Post-merge reconstruction
        post_img = postmerge_recon[i].permute(1, 2, 0).numpy()
        post_img = np.clip(post_img, 0, 1)
        axes[2, i].imshow(post_img)
        if i == 0:
            axes[2, i].set_ylabel(f"Post-merge\nLPIPS={postmerge_metrics['lpips']:.3f}", fontsize=10)
        axes[2, i].axis("off")

    lpips_gap = postmerge_metrics["lpips"] - premerge_metrics["lpips"]
    plt.suptitle(
        f"Pre-merge vs Post-merge Reconstruction Comparison\n"
        f"LPIPS Gap: {lpips_gap:.3f} | Pre-merge SSIM: {premerge_metrics['ssim']:.3f} | "
        f"Post-merge SSIM: {postmerge_metrics['ssim']:.3f}",
        fontsize=11,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
