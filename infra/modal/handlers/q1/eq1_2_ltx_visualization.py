"""E-Q1.2: LTX-Video Latent Space Visualization

Objective: Understand how LTX-Video organizes visual information in its VAE latent space.

Protocol:
1. Extract LTX-Video VAE latents for same images as E-Q1.1
2. Flatten spatial dimensions, analyze per-channel statistics
3. Apply dimensionality reduction (t-SNE, UMAP)
4. Color by semantic categories (same as VLM analysis)

Analysis:
- Channel-wise variance (which channels carry most information?)
- Spatial coherence (are nearby latent positions correlated?)
- Does LTX latent space cluster semantically?
- Which channels encode which aspects (content vs style vs texture)?

Success Metrics:
- Silhouette score by category
- Channel importance ranking
- Comparison with VLM clustering behavior
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
from PIL import Image

from runner import ExperimentRunner


def eq1_2_ltx_visualization(runner: ExperimentRunner) -> dict:
    """Run LTX-Video latent space visualization sub-experiment.

    This implementation:
    1. Generates same diverse test images as E-Q1.1
    2. Extracts VAE latents from LTX-Video
    3. Analyzes channel-wise statistics
    4. Runs t-SNE/UMAP visualization
    5. Computes clustering metrics

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-Q1.2: LTX-Video Latent Space Visualization")
    print("=" * 60)

    # Set up environment
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"eq1_2/stage": 0, "eq1_2/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate test images (same as E-Q1.1 for comparison)
    # =========================================================================
    print("\n[Stage 1/5] Preparing test images...")

    images, labels, label_names = generate_diverse_images(n_per_category=30)
    print(f"  Generated {len(images)} images across {len(label_names)} categories")

    runner.log_metrics({
        "eq1_2/stage": 1,
        "eq1_2/progress": 0.1,
        "eq1_2/n_images": len(images),
    })

    # =========================================================================
    # Stage 2: Load LTX-Video VAE and extract latents
    # =========================================================================
    print("\n[Stage 2/5] Loading LTX-Video VAE and extracting latents...")

    latents, latent_stats = extract_ltx_latents(images, runner)
    print(f"  Extracted latents shape: {latents.shape}")
    print(f"  Channels: {latents.shape[1]}, Spatial: {latents.shape[2]}x{latents.shape[3]}")

    runner.log_metrics({
        "eq1_2/stage": 2,
        "eq1_2/progress": 0.35,
        "eq1_2/n_channels": latents.shape[1],
        "eq1_2/spatial_h": latents.shape[2],
        "eq1_2/spatial_w": latents.shape[3],
    })

    # =========================================================================
    # Stage 3: Analyze channel statistics
    # =========================================================================
    print("\n[Stage 3/5] Analyzing channel statistics...")

    channel_analysis = analyze_channels(latents)
    print(f"  Top variance channels: {channel_analysis['top_variance_channels'][:5]}")
    print(f"  Channel variance range: [{channel_analysis['min_variance']:.4f}, {channel_analysis['max_variance']:.4f}]")

    runner.log_metrics({
        "eq1_2/stage": 3,
        "eq1_2/progress": 0.5,
        "eq1_2/max_channel_variance": channel_analysis['max_variance'],
        "eq1_2/min_channel_variance": channel_analysis['min_variance'],
    })

    # =========================================================================
    # Stage 4: Run dimensionality reduction
    # =========================================================================
    print("\n[Stage 4/5] Running dimensionality reduction...")

    from sklearn.manifold import TSNE

    try:
        from umap import UMAP
        has_umap = True
    except ImportError:
        has_umap = False
        print("  UMAP not available, using t-SNE only")

    # Flatten latents for projection: [N, C, H, W] -> [N, C*H*W]
    latents_flat = latents.reshape(latents.shape[0], -1)
    print(f"  Flattened latents shape: {latents_flat.shape}")

    # Also compute channel-mean representation
    latents_channel_mean = latents.mean(axis=(2, 3))  # [N, C]
    print(f"  Channel-mean latents shape: {latents_channel_mean.shape}")

    # t-SNE on flattened latents
    perplexity = min(30, len(images) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)

    print("  Running t-SNE on flattened latents...")
    tsne_flat = tsne.fit_transform(latents_flat)

    print("  Running t-SNE on channel-mean latents...")
    tsne_channel = tsne.fit_transform(latents_channel_mean)

    projections = {
        "flat": {"tsne": tsne_flat},
        "channel_mean": {"tsne": tsne_channel},
    }

    # UMAP if available
    if has_umap:
        print("  Running UMAP...")
        umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        projections["flat"]["umap"] = umap.fit_transform(latents_flat)
        projections["channel_mean"]["umap"] = umap.fit_transform(latents_channel_mean)

    runner.log_metrics({"eq1_2/stage": 4, "eq1_2/progress": 0.7})

    # =========================================================================
    # Stage 5: Compute metrics and create visualizations
    # =========================================================================
    print("\n[Stage 5/5] Computing metrics and creating visualizations...")

    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    metrics = {}

    # Metrics on flattened latents
    sil_flat = silhouette_score(latents_flat, labels)
    knn_flat = KNeighborsClassifier(n_neighbors=5)
    knn_flat_scores = cross_val_score(knn_flat, latents_flat, labels, cv=5)

    # Metrics on channel-mean latents
    sil_channel = silhouette_score(latents_channel_mean, labels)
    knn_channel = KNeighborsClassifier(n_neighbors=5)
    knn_channel_scores = cross_val_score(knn_channel, latents_channel_mean, labels, cv=5)

    metrics["flat"] = {
        "silhouette": float(sil_flat),
        "knn_accuracy": float(knn_flat_scores.mean()),
        "knn_std": float(knn_flat_scores.std()),
    }
    metrics["channel_mean"] = {
        "silhouette": float(sil_channel),
        "knn_accuracy": float(knn_channel_scores.mean()),
        "knn_std": float(knn_channel_scores.std()),
    }

    print(f"  Flat: silhouette={sil_flat:.3f}, k-NN={knn_flat_scores.mean():.3f}")
    print(f"  Channel-mean: silhouette={sil_channel:.3f}, k-NN={knn_channel_scores.mean():.3f}")

    # Create visualizations
    artifacts = []

    # t-SNE plots
    for rep_type in ["flat", "channel_mean"]:
        plot_bytes = create_visualization_plot(
            projections[rep_type]["tsne"],
            labels,
            label_names,
            metrics[rep_type]["silhouette"],
            f"LTX-Video Latent Space: {rep_type} (t-SNE)",
        )
        plot_path = runner.results.save_artifact(f"ltx_tsne_{rep_type}.png", plot_bytes)
        artifacts.append(plot_path)

        if "umap" in projections[rep_type]:
            plot_bytes = create_visualization_plot(
                projections[rep_type]["umap"],
                labels,
                label_names,
                metrics[rep_type]["silhouette"],
                f"LTX-Video Latent Space: {rep_type} (UMAP)",
            )
            plot_path = runner.results.save_artifact(f"ltx_umap_{rep_type}.png", plot_bytes)
            artifacts.append(plot_path)

    # Channel analysis plot
    channel_plot = create_channel_analysis_plot(channel_analysis)
    channel_path = runner.results.save_artifact("ltx_channel_analysis.png", channel_plot)
    artifacts.append(channel_path)

    # Spatial correlation analysis
    spatial_plot = create_spatial_correlation_plot(latents)
    spatial_path = runner.results.save_artifact("ltx_spatial_correlation.png", spatial_plot)
    artifacts.append(spatial_path)

    # Save raw data
    data = {
        "metrics": metrics,
        "channel_analysis": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in channel_analysis.items()
        },
        "projections": {
            rep: {k: v.tolist() for k, v in proj.items()}
            for rep, proj in projections.items()
        },
        "labels": labels.tolist(),
        "label_names": label_names,
        "latent_shape": list(latents.shape),
    }
    data_path = runner.results.save_json_artifact("ltx_latent_analysis.json", data)
    artifacts.append(data_path)

    runner.log_metrics({
        "eq1_2/stage": 5,
        "eq1_2/progress": 1.0,
        "eq1_2/silhouette_flat": sil_flat,
        "eq1_2/silhouette_channel": sil_channel,
        "eq1_2/knn_accuracy_flat": knn_flat_scores.mean(),
        "eq1_2/knn_accuracy_channel": knn_channel_scores.mean(),
    })

    # =========================================================================
    # Analyze results
    # =========================================================================
    best_sil = max(sil_flat, sil_channel)
    best_knn = max(knn_flat_scores.mean(), knn_channel_scores.mean())
    best_rep = "flattened" if sil_flat > sil_channel else "channel-mean"

    if best_sil > 0.4 and best_knn > 0.7:
        finding = (
            f"Strong semantic organization in LTX-Video latents. "
            f"Best representation: {best_rep} (silhouette={best_sil:.3f}, k-NN={best_knn:.3f}). "
            "Video decoder preserves semantic information useful for alignment."
        )
        semantic_quality = "good"
    elif best_sil > 0.2 or best_knn > 0.5:
        finding = (
            f"Moderate semantic organization in LTX-Video latents. "
            f"Best representation: {best_rep} (silhouette={best_sil:.3f}, k-NN={best_knn:.3f}). "
            "Some category structure preserved but with significant overlap."
        )
        semantic_quality = "moderate"
    else:
        finding = (
            f"Weak semantic clustering in LTX-Video latents. "
            f"Best representation: {best_rep} (silhouette={best_sil:.3f}, k-NN={best_knn:.3f}). "
            "VAE may prioritize visual reconstruction over semantic organization."
        )
        semantic_quality = "weak"

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "best_silhouette": float(best_sil),
            "best_knn_accuracy": float(best_knn),
            "best_representation": best_rep,
            "semantic_quality": semantic_quality,
            "silhouette_flat": float(sil_flat),
            "silhouette_channel": float(sil_channel),
            "n_channels": int(latents.shape[1]),
            "spatial_dims": f"{latents.shape[2]}x{latents.shape[3]}",
            "n_images": len(images),
            "n_categories": len(label_names),
            "top_variance_channels": channel_analysis["top_variance_channels"][:10],
        },
        "artifacts": artifacts,
    }


def generate_diverse_images(n_per_category: int = 30) -> tuple[list, np.ndarray, list]:
    """Generate same diverse images as E-Q1.1 for comparison.

    Uses same categories and generation logic for consistency.
    """
    from PIL import Image, ImageDraw
    import random

    categories = [
        ("circle_red", "red circle"),
        ("circle_blue", "blue circle"),
        ("square_green", "green square"),
        ("square_yellow", "yellow square"),
        ("triangle_purple", "purple triangle"),
        ("stripes_horizontal", "horizontal stripes"),
        ("stripes_vertical", "vertical stripes"),
        ("gradient_warm", "warm gradient"),
        ("gradient_cool", "cool gradient"),
    ]

    images = []
    labels = []
    label_names = [c[0] for c in categories]

    for cat_idx, (cat_name, description) in enumerate(categories):
        for i in range(n_per_category):
            img = create_category_image(cat_name, variation_seed=i)
            images.append(img)
            labels.append(cat_idx)

    return images, np.array(labels), label_names


def create_category_image(category: str, variation_seed: int = 0) -> Image.Image:
    """Create an image for a specific category with variation."""
    from PIL import Image, ImageDraw
    import random

    random.seed(variation_seed)
    np.random.seed(variation_seed)

    img = Image.new("RGB", (256, 256), (255, 255, 255))  # 256x256 for LTX-Video
    draw = ImageDraw.Draw(img)

    cx = random.randint(60, 196)
    cy = random.randint(60, 196)
    size = random.randint(35, 60)

    if category == "circle_red":
        color = (random.randint(180, 255), random.randint(0, 50), random.randint(0, 50))
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
    elif category == "circle_blue":
        color = (random.randint(0, 50), random.randint(0, 50), random.randint(180, 255))
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
    elif category == "square_green":
        color = (random.randint(0, 50), random.randint(180, 255), random.randint(0, 50))
        draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
    elif category == "square_yellow":
        color = (random.randint(200, 255), random.randint(200, 255), random.randint(0, 50))
        draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
    elif category == "triangle_purple":
        color = (random.randint(100, 180), random.randint(0, 50), random.randint(100, 180))
        points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
        draw.polygon(points, fill=color)
    elif category == "stripes_horizontal":
        stripe_width = random.randint(12, 30)
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(2)]
        for y in range(0, 256, stripe_width * 2):
            draw.rectangle([0, y, 256, y + stripe_width], fill=colors[0])
            draw.rectangle([0, y + stripe_width, 256, y + stripe_width * 2], fill=colors[1])
    elif category == "stripes_vertical":
        stripe_width = random.randint(12, 30)
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(2)]
        for x in range(0, 256, stripe_width * 2):
            draw.rectangle([x, 0, x + stripe_width, 256], fill=colors[0])
            draw.rectangle([x + stripe_width, 0, x + stripe_width * 2, 256], fill=colors[1])
    elif category == "gradient_warm":
        for y in range(256):
            r = int(255 - y * 0.3)
            g = int(100 + y * 0.5)
            b = int(50)
            draw.line([(0, y), (256, y)], fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
    elif category == "gradient_cool":
        for y in range(256):
            r = int(50)
            g = int(100 + y * 0.5)
            b = int(255 - y * 0.3)
            draw.line([(0, y), (256, y)], fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))

    return img


def extract_ltx_latents(images: list, runner: ExperimentRunner) -> tuple[np.ndarray, dict]:
    """Extract VAE latents from LTX-Video for a list of images.

    Args:
        images: List of PIL Images
        runner: ExperimentRunner for logging

    Returns:
        Tuple of (latents array [N, C, H, W], stats dict)
    """
    from diffusers import AutoencoderKLLTXVideo
    from torchvision import transforms

    print("  Loading LTX-Video VAE...")

    # Load just the VAE component
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    vae = vae.to("cuda")
    vae.eval()

    print(f"  VAE loaded on cuda")

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    latents_list = []
    stats = {"means": [], "stds": []}

    print(f"  Extracting latents from {len(images)} images...")

    with torch.no_grad():
        for idx, img in enumerate(images):
            # Preprocess image
            img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
            # Add temporal dimension for video VAE: [1, 3, 1, H, W]
            img_tensor = img_tensor.unsqueeze(2).to("cuda", dtype=torch.bfloat16)

            # Encode
            latent_dist = vae.encode(img_tensor)
            latent = latent_dist.latent_dist.sample()  # [1, C, T, H', W']

            # Remove temporal dim and convert to numpy
            latent = latent.squeeze(2).float().cpu().numpy()  # [1, C, H', W']

            latents_list.append(latent[0])  # [C, H', W']

            # Track stats
            stats["means"].append(latent.mean())
            stats["stds"].append(latent.std())

            if (idx + 1) % 30 == 0:
                progress = (idx + 1) / len(images)
                runner.log_metrics({"eq1_2/extraction_progress": progress})
                print(f"    Processed {idx + 1}/{len(images)} images")

    latents = np.stack(latents_list, axis=0)  # [N, C, H, W]

    stats["global_mean"] = float(np.mean(stats["means"]))
    stats["global_std"] = float(np.mean(stats["stds"]))

    # Clean up
    del vae
    torch.cuda.empty_cache()

    return latents, stats


def analyze_channels(latents: np.ndarray) -> dict:
    """Analyze per-channel statistics of LTX latents.

    Args:
        latents: Latent array [N, C, H, W]

    Returns:
        Dict with channel analysis results
    """
    n_samples, n_channels, h, w = latents.shape

    # Compute per-channel statistics
    channel_means = latents.mean(axis=(0, 2, 3))  # [C]
    channel_stds = latents.std(axis=(0, 2, 3))  # [C]
    channel_vars = channel_stds ** 2

    # Rank channels by variance
    variance_ranking = np.argsort(channel_vars)[::-1]
    top_variance_channels = variance_ranking.tolist()

    # Compute channel correlation matrix (subset for visualization)
    channel_data = latents.mean(axis=(2, 3))  # [N, C]
    channel_corr = np.corrcoef(channel_data.T)  # [C, C]

    return {
        "channel_means": channel_means,
        "channel_stds": channel_stds,
        "channel_vars": channel_vars,
        "top_variance_channels": top_variance_channels,
        "max_variance": float(channel_vars.max()),
        "min_variance": float(channel_vars.min()),
        "variance_ratio": float(channel_vars.max() / (channel_vars.min() + 1e-8)),
        "channel_correlation": channel_corr,
    }


def create_visualization_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    score: float,
    title: str,
) -> bytes:
    """Create a 2D visualization plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    n_classes = len(label_names)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_classes, 10)))
    if n_classes > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    for i, name in enumerate(label_names):
        mask = labels == i
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[i % len(colors)]],
            label=name,
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5,
        )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(f"{title}\nSilhouette Score: {score:.3f}", fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_channel_analysis_plot(channel_analysis: dict) -> bytes:
    """Create channel analysis visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Channel variance distribution
    vars = channel_analysis["channel_vars"]
    axes[0].bar(range(len(vars)), vars, alpha=0.7, color='steelblue')
    axes[0].set_xlabel("Channel Index")
    axes[0].set_ylabel("Variance")
    axes[0].set_title("Per-Channel Variance Distribution")
    axes[0].axhline(y=np.median(vars), color='red', linestyle='--', label='Median')
    axes[0].legend()

    # Top 20 channels by variance
    top_20 = channel_analysis["top_variance_channels"][:20]
    top_20_vars = [vars[i] for i in top_20]
    axes[1].barh(range(20), top_20_vars, alpha=0.7, color='coral')
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([f"Ch {i}" for i in top_20])
    axes[1].set_xlabel("Variance")
    axes[1].set_title("Top 20 Channels by Variance")
    axes[1].invert_yaxis()

    # Channel correlation heatmap (subset)
    corr = channel_analysis["channel_correlation"]
    # Show first 32 channels for clarity
    subset_size = min(32, corr.shape[0])
    im = axes[2].imshow(corr[:subset_size, :subset_size], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_xlabel("Channel")
    axes[2].set_ylabel("Channel")
    axes[2].set_title(f"Channel Correlation (first {subset_size})")
    plt.colorbar(im, ax=axes[2])

    plt.suptitle("LTX-Video Latent Channel Analysis", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_spatial_correlation_plot(latents: np.ndarray) -> bytes:
    """Create spatial autocorrelation analysis plot.

    Analyzes how correlated nearby positions are in the latent space.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import shift

    # Average across samples and channels
    avg_latent = latents.mean(axis=(0, 1))  # [H, W]

    # Compute spatial autocorrelation at different offsets
    max_offset = min(avg_latent.shape) // 2
    offsets = range(max_offset)
    correlations_h = []
    correlations_v = []

    for offset in offsets:
        if offset == 0:
            correlations_h.append(1.0)
            correlations_v.append(1.0)
        else:
            # Horizontal offset
            shifted_h = shift(avg_latent, [0, offset], mode='constant', cval=0)
            corr_h = np.corrcoef(avg_latent.flatten(), shifted_h.flatten())[0, 1]
            correlations_h.append(corr_h if not np.isnan(corr_h) else 0)

            # Vertical offset
            shifted_v = shift(avg_latent, [offset, 0], mode='constant', cval=0)
            corr_v = np.corrcoef(avg_latent.flatten(), shifted_v.flatten())[0, 1]
            correlations_v.append(corr_v if not np.isnan(corr_v) else 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Autocorrelation plot
    axes[0].plot(offsets, correlations_h, 'b-', label='Horizontal', linewidth=2)
    axes[0].plot(offsets, correlations_v, 'r-', label='Vertical', linewidth=2)
    axes[0].set_xlabel("Offset (pixels)")
    axes[0].set_ylabel("Correlation")
    axes[0].set_title("Spatial Autocorrelation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mean latent heatmap (first sample, first channel)
    im = axes[1].imshow(latents[0, 0], cmap='viridis')
    axes[1].set_title("Sample Latent (Ch 0)")
    plt.colorbar(im, ax=axes[1])

    # Channel-wise spatial variance
    spatial_var = latents.var(axis=(2, 3)).mean(axis=0)  # Var across space, mean across samples
    axes[2].bar(range(len(spatial_var)), spatial_var, alpha=0.7, color='mediumpurple')
    axes[2].set_xlabel("Channel")
    axes[2].set_ylabel("Spatial Variance")
    axes[2].set_title("Per-Channel Spatial Variance")

    plt.suptitle("LTX-Video Spatial Structure Analysis", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
