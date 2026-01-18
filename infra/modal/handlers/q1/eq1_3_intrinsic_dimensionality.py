"""E-Q1.3: Intrinsic Dimensionality Measurement

Objective: Determine the effective dimensionality of both VLM and LTX latent spaces.

Rationale: If VLM latents live on a low-dimensional manifold within the high-dimensional
space, alignment may be easier than the nominal dimensionality suggests.

Methods:
1. PCA Analysis: Compute explained variance ratio, find k where 95% variance explained
2. Maximum Likelihood Estimation (MLE): Levina & Bickel (2004)
3. Two-NN Estimator: Facco et al. (2017)
4. Local linear embedding analysis

Significance:
- If intrinsic dimensions are similar, linear alignment more likely
- Large difference suggests need for dimension-reducing or expanding adapter
- Very low intrinsic dim suggests representations are heavily constrained
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
from PIL import Image

from runner import ExperimentRunner


def eq1_3_intrinsic_dimensionality(runner: ExperimentRunner) -> dict:
    """Run intrinsic dimensionality measurement for both latent spaces.

    This implementation:
    1. Extracts latents from both VLM and LTX-Video
    2. Computes PCA explained variance analysis
    3. Estimates intrinsic dimensionality using MLE and Two-NN methods
    4. Compares dimensionality across spaces

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-Q1.3: Intrinsic Dimensionality Measurement")
    print("=" * 60)

    # Set up environment
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"eq1_3/stage": 0, "eq1_3/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate test images
    # =========================================================================
    print("\n[Stage 1/5] Preparing test images...")

    images, labels, label_names = generate_test_images(n_per_category=50)
    print(f"  Generated {len(images)} images across {len(label_names)} categories")

    runner.log_metrics({"eq1_3/stage": 1, "eq1_3/progress": 0.1})

    # =========================================================================
    # Stage 2: Extract VLM latents
    # =========================================================================
    print("\n[Stage 2/5] Extracting VLM latents...")

    vlm_latents = extract_vlm_latents(images, runner)
    print(f"  VLM latents shape: {vlm_latents.shape}")

    runner.log_metrics({
        "eq1_3/stage": 2,
        "eq1_3/progress": 0.3,
        "eq1_3/vlm_nominal_dim": vlm_latents.shape[1],
    })

    # =========================================================================
    # Stage 3: Extract LTX latents
    # =========================================================================
    print("\n[Stage 3/5] Extracting LTX-Video latents...")

    ltx_latents = extract_ltx_latents(images, runner)
    ltx_latents_flat = ltx_latents.reshape(ltx_latents.shape[0], -1)
    print(f"  LTX latents shape: {ltx_latents.shape}")
    print(f"  LTX latents flattened: {ltx_latents_flat.shape}")

    runner.log_metrics({
        "eq1_3/stage": 3,
        "eq1_3/progress": 0.5,
        "eq1_3/ltx_nominal_dim": ltx_latents_flat.shape[1],
    })

    # =========================================================================
    # Stage 4: Compute intrinsic dimensionality
    # =========================================================================
    print("\n[Stage 4/5] Computing intrinsic dimensionality estimates...")

    # VLM analysis
    print("  Analyzing VLM latent space...")
    vlm_results = compute_intrinsic_dimensionality(vlm_latents, "VLM")

    # LTX analysis
    print("  Analyzing LTX latent space...")
    ltx_results = compute_intrinsic_dimensionality(ltx_latents_flat, "LTX")

    # Also analyze channel-mean LTX representation
    ltx_channel_mean = ltx_latents.mean(axis=(2, 3))
    print("  Analyzing LTX channel-mean space...")
    ltx_channel_results = compute_intrinsic_dimensionality(ltx_channel_mean, "LTX_channel")

    runner.log_metrics({
        "eq1_3/stage": 4,
        "eq1_3/progress": 0.8,
        "eq1_3/vlm_pca_95": vlm_results["pca_95_components"],
        "eq1_3/vlm_mle_dim": vlm_results["mle_intrinsic_dim"],
        "eq1_3/ltx_pca_95": ltx_results["pca_95_components"],
        "eq1_3/ltx_mle_dim": ltx_results["mle_intrinsic_dim"],
    })

    # =========================================================================
    # Stage 5: Create visualizations and analysis
    # =========================================================================
    print("\n[Stage 5/5] Creating visualizations...")

    artifacts = []

    # PCA explained variance plots
    pca_plot = create_pca_comparison_plot(vlm_results, ltx_results, ltx_channel_results)
    pca_path = runner.results.save_artifact("intrinsic_dim_pca.png", pca_plot)
    artifacts.append(pca_path)

    # Dimensionality comparison summary
    summary_plot = create_dimensionality_summary(vlm_results, ltx_results, ltx_channel_results)
    summary_path = runner.results.save_artifact("intrinsic_dim_summary.png", summary_plot)
    artifacts.append(summary_path)

    # Save detailed results
    data = {
        "vlm": {k: float(v) if isinstance(v, (np.floating, float)) else
                   (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in vlm_results.items()},
        "ltx_flat": {k: float(v) if isinstance(v, (np.floating, float)) else
                       (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in ltx_results.items()},
        "ltx_channel": {k: float(v) if isinstance(v, (np.floating, float)) else
                          (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in ltx_channel_results.items()},
        "n_samples": len(images),
    }
    data_path = runner.results.save_json_artifact("intrinsic_dimensionality.json", data)
    artifacts.append(data_path)

    runner.log_metrics({"eq1_3/stage": 5, "eq1_3/progress": 1.0})

    # =========================================================================
    # Analyze results
    # =========================================================================
    vlm_intrinsic = vlm_results["mle_intrinsic_dim"]
    ltx_intrinsic = ltx_results["mle_intrinsic_dim"]
    dim_ratio = max(vlm_intrinsic, ltx_intrinsic) / min(vlm_intrinsic, ltx_intrinsic)

    vlm_pca_95 = vlm_results["pca_95_components"]
    ltx_pca_95 = ltx_results["pca_95_components"]

    if dim_ratio < 2:
        alignment_assessment = "excellent"
        finding = (
            f"Intrinsic dimensions are similar: VLM={vlm_intrinsic:.0f}, LTX={ltx_intrinsic:.0f} "
            f"(ratio {dim_ratio:.1f}x). Linear alignment is promising. "
            f"PCA 95%: VLM={vlm_pca_95}, LTX={ltx_pca_95} components."
        )
    elif dim_ratio < 5:
        alignment_assessment = "moderate"
        finding = (
            f"Intrinsic dimensions differ moderately: VLM={vlm_intrinsic:.0f}, LTX={ltx_intrinsic:.0f} "
            f"(ratio {dim_ratio:.1f}x). MLP adapter should bridge this gap. "
            f"PCA 95%: VLM={vlm_pca_95}, LTX={ltx_pca_95} components."
        )
    elif dim_ratio < 10:
        alignment_assessment = "challenging"
        finding = (
            f"Significant dimensionality mismatch: VLM={vlm_intrinsic:.0f}, LTX={ltx_intrinsic:.0f} "
            f"(ratio {dim_ratio:.1f}x). May need dimension-matching adapter. "
            f"PCA 95%: VLM={vlm_pca_95}, LTX={ltx_pca_95} components."
        )
    else:
        alignment_assessment = "concerning"
        finding = (
            f"Large dimensionality gap: VLM={vlm_intrinsic:.0f}, LTX={ltx_intrinsic:.0f} "
            f"(ratio {dim_ratio:.1f}x). Fundamental structure difference may require complex adapter. "
            f"PCA 95%: VLM={vlm_pca_95}, LTX={ltx_pca_95} components."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "vlm_nominal_dim": int(vlm_latents.shape[1]),
            "vlm_pca_95_components": int(vlm_pca_95),
            "vlm_mle_intrinsic_dim": float(vlm_intrinsic),
            "vlm_two_nn_intrinsic_dim": float(vlm_results.get("two_nn_intrinsic_dim", vlm_intrinsic)),
            "ltx_nominal_dim": int(ltx_latents_flat.shape[1]),
            "ltx_pca_95_components": int(ltx_pca_95),
            "ltx_mle_intrinsic_dim": float(ltx_intrinsic),
            "ltx_two_nn_intrinsic_dim": float(ltx_results.get("two_nn_intrinsic_dim", ltx_intrinsic)),
            "ltx_channel_intrinsic_dim": float(ltx_channel_results["mle_intrinsic_dim"]),
            "dimensionality_ratio": float(dim_ratio),
            "alignment_assessment": alignment_assessment,
            "n_samples": len(images),
        },
        "artifacts": artifacts,
    }


def generate_test_images(n_per_category: int = 50) -> tuple[list, np.ndarray, list]:
    """Generate test images with diverse categories."""
    from PIL import Image, ImageDraw
    import random

    categories = [
        "circle_red", "circle_blue", "square_green", "square_yellow",
        "triangle_purple", "stripes_horizontal", "stripes_vertical",
        "gradient_warm", "gradient_cool",
    ]

    images = []
    labels = []

    for cat_idx, cat_name in enumerate(categories):
        for i in range(n_per_category):
            img = create_category_image(cat_name, variation_seed=i)
            images.append(img)
            labels.append(cat_idx)

    return images, np.array(labels), categories


def create_category_image(category: str, variation_seed: int = 0) -> Image.Image:
    """Create an image for a specific category."""
    from PIL import Image, ImageDraw
    import random

    random.seed(variation_seed)
    np.random.seed(variation_seed)

    img = Image.new("RGB", (256, 256), (255, 255, 255))
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
    elif category == "stripes_vertical":
        stripe_width = random.randint(12, 30)
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(2)]
        for x in range(0, 256, stripe_width * 2):
            draw.rectangle([x, 0, x + stripe_width, 256], fill=colors[0])
    elif category == "gradient_warm":
        for y in range(256):
            r = int(255 - y * 0.3)
            g = int(100 + y * 0.5)
            draw.line([(0, y), (256, y)], fill=(max(0, min(255, r)), max(0, min(255, g)), 50))
    elif category == "gradient_cool":
        for y in range(256):
            g = int(100 + y * 0.5)
            b = int(255 - y * 0.3)
            draw.line([(0, y), (256, y)], fill=(50, max(0, min(255, g)), max(0, min(255, b))))

    return img


def extract_vlm_latents(images: list, runner: ExperimentRunner) -> np.ndarray:
    """Extract VLM latents (mean-pooled from final layer)."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print("    Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
    )

    latents_list = []

    with torch.no_grad():
        for idx, img in enumerate(images):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe."},
                ],
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model.device)

            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
            pooled = hidden_states.mean(dim=0).float().cpu().numpy()
            latents_list.append(pooled)

            if (idx + 1) % 50 == 0:
                print(f"      Processed {idx + 1}/{len(images)} images")

    del model, processor
    torch.cuda.empty_cache()

    return np.stack(latents_list, axis=0)


def extract_ltx_latents(images: list, runner: ExperimentRunner) -> np.ndarray:
    """Extract LTX-Video VAE latents."""
    from diffusers import AutoencoderKLLTXVideo
    from torchvision import transforms

    print("    Loading LTX-Video VAE...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    vae.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    latents_list = []

    with torch.no_grad():
        for idx, img in enumerate(images):
            img_tensor = transform(img).unsqueeze(0).unsqueeze(2).to("cuda", dtype=torch.bfloat16)
            latent_dist = vae.encode(img_tensor)
            latent = latent_dist.latent_dist.sample().squeeze(2).float().cpu().numpy()
            latents_list.append(latent[0])

            if (idx + 1) % 50 == 0:
                print(f"      Processed {idx + 1}/{len(images)} images")

    del vae
    torch.cuda.empty_cache()

    return np.stack(latents_list, axis=0)


def compute_intrinsic_dimensionality(data: np.ndarray, name: str) -> dict:
    """Compute intrinsic dimensionality using multiple methods.

    Args:
        data: Data array [N, D]
        name: Name for logging

    Returns:
        Dict with dimensionality estimates
    """
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cdist

    n_samples, nominal_dim = data.shape
    print(f"    {name}: {n_samples} samples, {nominal_dim} nominal dimensions")

    results = {
        "nominal_dim": nominal_dim,
        "n_samples": n_samples,
    }

    # =========================================================================
    # 1. PCA Analysis
    # =========================================================================
    # Limit components for computational efficiency
    max_components = min(n_samples - 1, nominal_dim, 500)
    pca = PCA(n_components=max_components)
    pca.fit(data)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find components for 90%, 95%, 99% variance
    for threshold in [0.90, 0.95, 0.99]:
        idx = np.argmax(cumulative_variance >= threshold)
        if cumulative_variance[idx] >= threshold:
            results[f"pca_{int(threshold*100)}_components"] = int(idx + 1)
        else:
            results[f"pca_{int(threshold*100)}_components"] = max_components

    results["explained_variance_ratio"] = explained_variance_ratio[:100].tolist()  # First 100
    results["cumulative_variance"] = cumulative_variance[:100].tolist()

    print(f"      PCA: 95% variance at {results['pca_95_components']} components")

    # =========================================================================
    # 2. MLE Intrinsic Dimensionality (Levina & Bickel, 2004)
    # =========================================================================
    mle_dim = estimate_mle_dimension(data, k=10)
    results["mle_intrinsic_dim"] = mle_dim
    print(f"      MLE intrinsic dim: {mle_dim:.1f}")

    # =========================================================================
    # 3. Two-NN Estimator (Facco et al., 2017)
    # =========================================================================
    two_nn_dim = estimate_two_nn_dimension(data)
    results["two_nn_intrinsic_dim"] = two_nn_dim
    print(f"      Two-NN intrinsic dim: {two_nn_dim:.1f}")

    # =========================================================================
    # 4. Effective rank (based on eigenvalue distribution)
    # =========================================================================
    eigenvalues = pca.explained_variance_
    normalized_eigs = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(normalized_eigs * np.log(normalized_eigs + 1e-10))
    effective_rank = np.exp(entropy)
    results["effective_rank"] = effective_rank
    print(f"      Effective rank: {effective_rank:.1f}")

    return results


def estimate_mle_dimension(data: np.ndarray, k: int = 10) -> float:
    """Estimate intrinsic dimension using MLE method.

    Levina & Bickel (2004): Maximum Likelihood Estimation of Intrinsic Dimension

    Args:
        data: Data array [N, D]
        k: Number of nearest neighbors

    Returns:
        Estimated intrinsic dimension
    """
    from scipy.spatial.distance import cdist

    n = data.shape[0]
    if n < k + 1:
        return float(data.shape[1])

    # Compute pairwise distances
    distances = cdist(data, data, metric='euclidean')

    # For each point, get k-nearest neighbor distances
    dim_estimates = []

    for i in range(n):
        # Get sorted distances (excluding self)
        sorted_dists = np.sort(distances[i])[1:k+1]  # k nearest (exclude self)

        if sorted_dists[-1] > 0:
            # MLE estimate for this point
            log_ratios = np.log(sorted_dists[-1] / (sorted_dists[:-1] + 1e-10))
            local_dim = (k - 1) / np.sum(log_ratios)
            if not np.isnan(local_dim) and not np.isinf(local_dim) and local_dim > 0:
                dim_estimates.append(local_dim)

    if len(dim_estimates) > 0:
        return float(np.mean(dim_estimates))
    else:
        return float(data.shape[1])


def estimate_two_nn_dimension(data: np.ndarray) -> float:
    """Estimate intrinsic dimension using Two-NN method.

    Facco et al. (2017): Estimating the intrinsic dimension using a minimal
    neighborhood information

    Args:
        data: Data array [N, D]

    Returns:
        Estimated intrinsic dimension
    """
    from scipy.spatial.distance import cdist

    n = data.shape[0]
    if n < 3:
        return float(data.shape[1])

    # Compute pairwise distances
    distances = cdist(data, data, metric='euclidean')

    # For each point, get distances to 2 nearest neighbors
    mus = []
    for i in range(n):
        sorted_dists = np.sort(distances[i])[1:3]  # 2 nearest (exclude self)
        if sorted_dists[0] > 0:
            mu = sorted_dists[1] / sorted_dists[0]
            if mu > 1:
                mus.append(mu)

    if len(mus) < 10:
        return float(data.shape[1])

    mus = np.array(mus)

    # Estimate dimension from mu distribution
    # d = n * sum(log(mu)) for empirical CDF-based estimator
    log_mus = np.log(mus)
    d = len(mus) / np.sum(log_mus)

    return float(d)


def create_pca_comparison_plot(vlm_results: dict, ltx_results: dict, ltx_channel_results: dict) -> bytes:
    """Create PCA explained variance comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Cumulative variance curves
    vlm_cumvar = np.array(vlm_results["cumulative_variance"])
    ltx_cumvar = np.array(ltx_results["cumulative_variance"])
    ltx_ch_cumvar = np.array(ltx_channel_results["cumulative_variance"])

    axes[0].plot(range(1, len(vlm_cumvar) + 1), vlm_cumvar, 'b-', linewidth=2, label='VLM')
    axes[0].plot(range(1, len(ltx_cumvar) + 1), ltx_cumvar, 'r-', linewidth=2, label='LTX (flat)')
    axes[0].plot(range(1, len(ltx_ch_cumvar) + 1), ltx_ch_cumvar, 'g-', linewidth=2, label='LTX (channel)')
    axes[0].axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("Cumulative Explained Variance")
    axes[0].set_title("PCA Cumulative Variance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(1, 100)

    # Per-component variance (scree plot)
    vlm_var = np.array(vlm_results["explained_variance_ratio"])
    ltx_var = np.array(ltx_results["explained_variance_ratio"])
    ltx_ch_var = np.array(ltx_channel_results["explained_variance_ratio"])

    axes[1].semilogy(range(1, len(vlm_var) + 1), vlm_var, 'b-', linewidth=2, label='VLM')
    axes[1].semilogy(range(1, len(ltx_var) + 1), ltx_var, 'r-', linewidth=2, label='LTX (flat)')
    axes[1].semilogy(range(1, len(ltx_ch_var) + 1), ltx_ch_var, 'g-', linewidth=2, label='LTX (channel)')
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Explained Variance Ratio (log)")
    axes[1].set_title("PCA Scree Plot")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(1, 50)

    # Dimensionality summary bar chart
    metrics = ['PCA 95%', 'MLE', 'Two-NN', 'Eff. Rank']
    vlm_vals = [
        vlm_results["pca_95_components"],
        vlm_results["mle_intrinsic_dim"],
        vlm_results["two_nn_intrinsic_dim"],
        vlm_results["effective_rank"],
    ]
    ltx_vals = [
        ltx_results["pca_95_components"],
        ltx_results["mle_intrinsic_dim"],
        ltx_results["two_nn_intrinsic_dim"],
        ltx_results["effective_rank"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    axes[2].bar(x - width/2, vlm_vals, width, label='VLM', color='steelblue', alpha=0.8)
    axes[2].bar(x + width/2, ltx_vals, width, label='LTX', color='coral', alpha=0.8)
    axes[2].set_ylabel("Estimated Dimension")
    axes[2].set_title("Dimensionality Estimates Comparison")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Intrinsic Dimensionality Analysis", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_dimensionality_summary(vlm_results: dict, ltx_results: dict, ltx_channel_results: dict) -> bytes:
    """Create dimensionality summary visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create summary table as a text-based visualization
    data = [
        ["Metric", "VLM", "LTX (flat)", "LTX (channel)"],
        ["Nominal Dim", f"{vlm_results['nominal_dim']}", f"{ltx_results['nominal_dim']}", f"{ltx_channel_results['nominal_dim']}"],
        ["PCA 90%", f"{vlm_results['pca_90_components']}", f"{ltx_results['pca_90_components']}", f"{ltx_channel_results['pca_90_components']}"],
        ["PCA 95%", f"{vlm_results['pca_95_components']}", f"{ltx_results['pca_95_components']}", f"{ltx_channel_results['pca_95_components']}"],
        ["PCA 99%", f"{vlm_results['pca_99_components']}", f"{ltx_results['pca_99_components']}", f"{ltx_channel_results['pca_99_components']}"],
        ["MLE Dim", f"{vlm_results['mle_intrinsic_dim']:.1f}", f"{ltx_results['mle_intrinsic_dim']:.1f}", f"{ltx_channel_results['mle_intrinsic_dim']:.1f}"],
        ["Two-NN Dim", f"{vlm_results['two_nn_intrinsic_dim']:.1f}", f"{ltx_results['two_nn_intrinsic_dim']:.1f}", f"{ltx_channel_results['two_nn_intrinsic_dim']:.1f}"],
        ["Effective Rank", f"{vlm_results['effective_rank']:.1f}", f"{ltx_results['effective_rank']:.1f}", f"{ltx_channel_results['effective_rank']:.1f}"],
    ]

    # Hide axes
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    # Alternate row colors
    for i in range(1, len(data)):
        color = '#D9E2F3' if i % 2 == 0 else 'white'
        for j in range(4):
            table[(i, j)].set_facecolor(color)

    ax.set_title("Intrinsic Dimensionality Summary", fontsize=16, pad=20)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
