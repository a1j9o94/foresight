"""E-Q1.5: Semantic Similarity Preservation Test

Objective: Test whether semantic relationships are preserved across VLM and LTX spaces.

Rationale: Even if the spaces aren't linearly alignable, they might preserve the same
similarity structure (which is sufficient for learning an adapter).

Protocol:
1. Compute pairwise distances in VLM space (NxN matrix)
2. Compute pairwise distances in LTX space (same images)
3. Measure correlation between distance matrices
4. Analyze which pairs are similar in one space but not the other

Metrics:
- Spearman correlation: Rank correlation of pairwise distances
- Procrustes analysis: Optimal rotation alignment of spaces
- Mantel test: Statistical significance of correlation

Success Criteria:
- Spearman rho > 0.6: Good structural alignment
- Spearman rho > 0.8: Excellent structural alignment
- Spearman rho < 0.4: Concerning structural mismatch
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
from PIL import Image
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform

from runner import ExperimentRunner


def eq1_5_semantic_similarity(runner: ExperimentRunner) -> dict:
    """Run semantic similarity preservation test.

    This implementation:
    1. Extracts paired latents from both models
    2. Computes pairwise distance matrices
    3. Calculates correlation metrics (Spearman, Procrustes)
    4. Analyzes agreement/disagreement patterns by category

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-Q1.5: Semantic Similarity Preservation Test")
    print("=" * 60)

    # Set up environment
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"eq1_5/stage": 0, "eq1_5/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate test images
    # =========================================================================
    print("\n[Stage 1/5] Preparing test images...")

    images, labels, label_names = generate_test_images(n_per_category=40)
    print(f"  Generated {len(images)} images across {len(label_names)} categories")

    runner.log_metrics({
        "eq1_5/stage": 1,
        "eq1_5/progress": 0.1,
        "eq1_5/n_images": len(images),
    })

    # =========================================================================
    # Stage 2: Extract latents from both models
    # =========================================================================
    print("\n[Stage 2/5] Extracting latents...")

    vlm_latents = extract_vlm_latents(images, runner)
    ltx_latents = extract_ltx_latents(images, runner)
    ltx_latents_flat = ltx_latents.reshape(ltx_latents.shape[0], -1)

    print(f"  VLM: {vlm_latents.shape}, LTX: {ltx_latents_flat.shape}")

    runner.log_metrics({"eq1_5/stage": 2, "eq1_5/progress": 0.4})

    # =========================================================================
    # Stage 3: Compute distance matrices
    # =========================================================================
    print("\n[Stage 3/5] Computing distance matrices...")

    # Euclidean distances
    vlm_dists = cdist(vlm_latents, vlm_latents, metric='euclidean')
    ltx_dists = cdist(ltx_latents_flat, ltx_latents_flat, metric='euclidean')

    # Cosine distances
    vlm_dists_cos = cdist(vlm_latents, vlm_latents, metric='cosine')
    ltx_dists_cos = cdist(ltx_latents_flat, ltx_latents_flat, metric='cosine')

    print(f"  Distance matrices computed: {vlm_dists.shape}")

    runner.log_metrics({"eq1_5/stage": 3, "eq1_5/progress": 0.5})

    # =========================================================================
    # Stage 4: Compute correlation metrics
    # =========================================================================
    print("\n[Stage 4/5] Computing correlation metrics...")

    # Flatten upper triangles (exclude diagonal) for correlation
    triu_idx = np.triu_indices_from(vlm_dists, k=1)

    vlm_dists_flat = vlm_dists[triu_idx]
    ltx_dists_flat = ltx_dists[triu_idx]
    vlm_dists_cos_flat = vlm_dists_cos[triu_idx]
    ltx_dists_cos_flat = ltx_dists_cos[triu_idx]

    # Spearman correlation (rank-based, robust to nonlinearity)
    spearman_euclidean, spearman_p_euclidean = stats.spearmanr(vlm_dists_flat, ltx_dists_flat)
    spearman_cosine, spearman_p_cosine = stats.spearmanr(vlm_dists_cos_flat, ltx_dists_cos_flat)

    # Pearson correlation (linear relationship)
    pearson_euclidean, pearson_p_euclidean = stats.pearsonr(vlm_dists_flat, ltx_dists_flat)
    pearson_cosine, pearson_p_cosine = stats.pearsonr(vlm_dists_cos_flat, ltx_dists_cos_flat)

    # Kendall tau (more robust but slower)
    # Only compute on a sample for efficiency
    sample_idx = np.random.choice(len(vlm_dists_flat), min(5000, len(vlm_dists_flat)), replace=False)
    kendall, kendall_p = stats.kendalltau(vlm_dists_flat[sample_idx], ltx_dists_flat[sample_idx])

    print(f"  Spearman (Euclidean): rho={spearman_euclidean:.4f}, p={spearman_p_euclidean:.2e}")
    print(f"  Spearman (Cosine): rho={spearman_cosine:.4f}, p={spearman_p_cosine:.2e}")
    print(f"  Pearson (Euclidean): r={pearson_euclidean:.4f}")
    print(f"  Kendall tau: tau={kendall:.4f}")

    # Procrustes analysis (optimal alignment)
    procrustes_disparity = compute_procrustes_alignment(vlm_latents, ltx_latents_flat)
    print(f"  Procrustes disparity: {procrustes_disparity:.4f}")

    # Mantel test (permutation-based significance)
    mantel_r, mantel_p = mantel_test(vlm_dists, ltx_dists, n_permutations=999)
    print(f"  Mantel test: r={mantel_r:.4f}, p={mantel_p:.4f}")

    runner.log_metrics({
        "eq1_5/stage": 4,
        "eq1_5/progress": 0.7,
        "eq1_5/spearman_euclidean": spearman_euclidean,
        "eq1_5/spearman_cosine": spearman_cosine,
        "eq1_5/pearson_euclidean": pearson_euclidean,
        "eq1_5/procrustes_disparity": procrustes_disparity,
        "eq1_5/mantel_r": mantel_r,
    })

    # =========================================================================
    # Stage 5: Category-specific analysis and visualization
    # =========================================================================
    print("\n[Stage 5/5] Analyzing by category and creating visualizations...")

    # Within-category vs between-category analysis
    category_analysis = analyze_by_category(vlm_dists, ltx_dists, labels, label_names)

    artifacts = []

    # Distance correlation scatter plot
    scatter_plot = create_distance_scatter_plot(
        vlm_dists_flat, ltx_dists_flat,
        spearman_euclidean, pearson_euclidean
    )
    scatter_path = runner.results.save_artifact("semantic_sim_scatter.png", scatter_plot)
    artifacts.append(scatter_path)

    # Category-specific analysis plot
    category_plot = create_category_analysis_plot(category_analysis, label_names)
    category_path = runner.results.save_artifact("semantic_sim_categories.png", category_plot)
    artifacts.append(category_path)

    # Distance matrix heatmaps
    heatmap_plot = create_distance_heatmaps(vlm_dists, ltx_dists, labels, label_names)
    heatmap_path = runner.results.save_artifact("semantic_sim_heatmaps.png", heatmap_plot)
    artifacts.append(heatmap_path)

    # Rank agreement analysis
    rank_plot = create_rank_agreement_plot(vlm_dists, ltx_dists)
    rank_path = runner.results.save_artifact("semantic_sim_ranks.png", rank_plot)
    artifacts.append(rank_path)

    # Save detailed results
    data = {
        "correlations": {
            "spearman_euclidean": float(spearman_euclidean),
            "spearman_euclidean_p": float(spearman_p_euclidean),
            "spearman_cosine": float(spearman_cosine),
            "spearman_cosine_p": float(spearman_p_cosine),
            "pearson_euclidean": float(pearson_euclidean),
            "pearson_cosine": float(pearson_cosine),
            "kendall_tau": float(kendall),
            "procrustes_disparity": float(procrustes_disparity),
            "mantel_r": float(mantel_r),
            "mantel_p": float(mantel_p),
        },
        "category_analysis": {
            k: ({kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()} if isinstance(v, dict) else float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in category_analysis.items()
        },
        "n_images": len(images),
        "n_categories": len(label_names),
        "n_pairs": len(vlm_dists_flat),
    }
    data_path = runner.results.save_json_artifact("semantic_similarity_results.json", data)
    artifacts.append(data_path)

    runner.log_metrics({"eq1_5/stage": 5, "eq1_5/progress": 1.0})

    # =========================================================================
    # Form conclusions
    # =========================================================================
    avg_spearman = (spearman_euclidean + spearman_cosine) / 2

    if avg_spearman > 0.8:
        quality = "excellent"
        finding = (
            f"Excellent semantic similarity preservation! "
            f"Spearman rho (Euclidean)={spearman_euclidean:.3f}, (Cosine)={spearman_cosine:.3f}. "
            f"Both spaces preserve the same similarity structure, enabling effective alignment. "
            f"Mantel test significant: r={mantel_r:.3f}, p={mantel_p:.4f}."
        )
    elif avg_spearman > 0.6:
        quality = "good"
        finding = (
            f"Good semantic similarity preservation. "
            f"Spearman rho (Euclidean)={spearman_euclidean:.3f}, (Cosine)={spearman_cosine:.3f}. "
            f"Most similarity relationships are preserved across spaces. "
            f"Mantel test: r={mantel_r:.3f}, p={mantel_p:.4f}."
        )
    elif avg_spearman > 0.4:
        quality = "moderate"
        finding = (
            f"Moderate semantic similarity preservation. "
            f"Spearman rho (Euclidean)={spearman_euclidean:.3f}, (Cosine)={spearman_cosine:.3f}. "
            f"Some structure preserved but with notable differences. Adapter may need more capacity. "
            f"Mantel test: r={mantel_r:.3f}, p={mantel_p:.4f}."
        )
    else:
        quality = "weak"
        finding = (
            f"Weak semantic similarity preservation. "
            f"Spearman rho (Euclidean)={spearman_euclidean:.3f}, (Cosine)={spearman_cosine:.3f}. "
            f"Spaces encode fundamentally different similarity structures. "
            f"Consider alternative alignment approaches. Mantel test: r={mantel_r:.3f}."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "spearman_euclidean": float(spearman_euclidean),
            "spearman_cosine": float(spearman_cosine),
            "average_spearman": float(avg_spearman),
            "pearson_euclidean": float(pearson_euclidean),
            "kendall_tau": float(kendall),
            "procrustes_disparity": float(procrustes_disparity),
            "mantel_r": float(mantel_r),
            "mantel_p": float(mantel_p),
            "quality": quality,
            "n_images": len(images),
            "n_pairs": len(vlm_dists_flat),
        },
        "artifacts": artifacts,
    }


def generate_test_images(n_per_category: int = 40) -> tuple[list, np.ndarray, list]:
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
    """Extract VLM latents."""
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
            hidden_states = outputs.hidden_states[-1][0]
            pooled = hidden_states.mean(dim=0).float().cpu().numpy()
            latents_list.append(pooled)

            if (idx + 1) % 50 == 0:
                print(f"    VLM: {idx + 1}/{len(images)}")

    del model, processor
    torch.cuda.empty_cache()

    return np.stack(latents_list, axis=0)


def extract_ltx_latents(images: list, runner: ExperimentRunner) -> np.ndarray:
    """Extract LTX-Video VAE latents."""
    from diffusers import AutoencoderKLLTXVideo
    from torchvision import transforms

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
                print(f"    LTX: {idx + 1}/{len(images)}")

    del vae
    torch.cuda.empty_cache()

    return np.stack(latents_list, axis=0)


def compute_procrustes_alignment(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Procrustes disparity between two sets of points.

    Lower disparity means better alignment after optimal rotation/scaling.
    """
    from scipy.linalg import orthogonal_procrustes

    # Center the data
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Reduce to same dimensionality via PCA if needed
    from sklearn.decomposition import PCA

    min_dim = min(X_centered.shape[1], Y_centered.shape[1], 100)
    pca_X = PCA(n_components=min_dim)
    pca_Y = PCA(n_components=min_dim)

    X_reduced = pca_X.fit_transform(X_centered)
    Y_reduced = pca_Y.fit_transform(Y_centered)

    # Normalize
    X_norm = X_reduced / np.linalg.norm(X_reduced)
    Y_norm = Y_reduced / np.linalg.norm(Y_reduced)

    # Find optimal rotation
    R, scale = orthogonal_procrustes(X_norm, Y_norm)

    # Compute disparity
    X_aligned = X_norm @ R
    disparity = np.sum((X_aligned - Y_norm) ** 2)

    return float(disparity)


def mantel_test(X: np.ndarray, Y: np.ndarray, n_permutations: int = 999) -> tuple[float, float]:
    """Perform Mantel test for correlation between distance matrices.

    Args:
        X, Y: Distance matrices
        n_permutations: Number of permutations for p-value

    Returns:
        (correlation, p-value)
    """
    # Get upper triangles
    triu_idx = np.triu_indices_from(X, k=1)
    x_flat = X[triu_idx]
    y_flat = Y[triu_idx]

    # Observed correlation
    r_obs, _ = stats.pearsonr(x_flat, y_flat)

    # Permutation test
    r_perms = []
    for _ in range(n_permutations):
        perm = np.random.permutation(len(Y))
        Y_perm = Y[perm][:, perm]
        y_perm_flat = Y_perm[triu_idx]
        r_perm, _ = stats.pearsonr(x_flat, y_perm_flat)
        r_perms.append(r_perm)

    # P-value: proportion of permutations with r >= r_obs
    p_value = (np.sum(np.abs(r_perms) >= np.abs(r_obs)) + 1) / (n_permutations + 1)

    return float(r_obs), float(p_value)


def analyze_by_category(
    vlm_dists: np.ndarray,
    ltx_dists: np.ndarray,
    labels: np.ndarray,
    label_names: list,
) -> dict:
    """Analyze correlation within and between categories."""
    results = {}

    unique_labels = np.unique(labels)

    # Within-category correlations
    within_corrs = []
    for label in unique_labels:
        mask = labels == label
        idx = np.where(mask)[0]
        if len(idx) > 2:
            vlm_sub = vlm_dists[np.ix_(idx, idx)]
            ltx_sub = ltx_dists[np.ix_(idx, idx)]
            triu = np.triu_indices_from(vlm_sub, k=1)
            if len(vlm_sub[triu]) > 1:
                r, _ = stats.spearmanr(vlm_sub[triu], ltx_sub[triu])
                within_corrs.append(r)
                results[f"within_{label_names[label]}"] = {"spearman": r, "n_pairs": len(vlm_sub[triu])}

    results["within_category_mean"] = np.mean(within_corrs) if within_corrs else 0

    # Between-category correlations (sample)
    between_corrs = []
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            idx1 = np.where(labels == label1)[0]
            idx2 = np.where(labels == label2)[0]

            vlm_sub = vlm_dists[np.ix_(idx1, idx2)]
            ltx_sub = ltx_dists[np.ix_(idx1, idx2)]

            r, _ = stats.spearmanr(vlm_sub.flatten(), ltx_sub.flatten())
            between_corrs.append(r)

    results["between_category_mean"] = np.mean(between_corrs) if between_corrs else 0
    results["within_vs_between_ratio"] = (
        results["within_category_mean"] / (results["between_category_mean"] + 1e-10)
    )

    return results


def create_distance_scatter_plot(
    vlm_dists: np.ndarray,
    ltx_dists: np.ndarray,
    spearman: float,
    pearson: float,
) -> bytes:
    """Create scatter plot of distances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample for visualization
    n_sample = min(10000, len(vlm_dists))
    idx = np.random.choice(len(vlm_dists), n_sample, replace=False)

    ax.scatter(vlm_dists[idx], ltx_dists[idx], alpha=0.3, s=5)

    # Add regression line
    z = np.polyfit(vlm_dists[idx], ltx_dists[idx], 1)
    p = np.poly1d(z)
    x_line = np.linspace(vlm_dists.min(), vlm_dists.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Linear fit')

    ax.set_xlabel("VLM Pairwise Distance", fontsize=12)
    ax.set_ylabel("LTX Pairwise Distance", fontsize=12)
    ax.set_title(
        f"Distance Correlation (Spearman={spearman:.3f}, Pearson={pearson:.3f})\n"
        f"n={n_sample} pairs sampled",
        fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_category_analysis_plot(category_analysis: dict, label_names: list) -> bytes:
    """Create category-specific correlation analysis plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Within-category correlations (filter out scalar entries like within_category_mean)
    within_data = {k.replace("within_", ""): v["spearman"]
                   for k, v in category_analysis.items()
                   if k.startswith("within_") and isinstance(v, dict) and "spearman" in v}

    if within_data:
        cats = list(within_data.keys())
        corrs = list(within_data.values())

        axes[0].bar(range(len(cats)), corrs, color='steelblue', alpha=0.8)
        axes[0].set_xticks(range(len(cats)))
        axes[0].set_xticklabels(cats, rotation=45, ha='right')
        axes[0].set_ylabel("Spearman Correlation")
        axes[0].set_title("Within-Category Distance Correlation")
        axes[0].axhline(y=category_analysis.get("within_category_mean", 0),
                        color='red', linestyle='--', label='Mean')
        axes[0].axhline(y=0.6, color='green', linestyle=':', alpha=0.5, label='Good threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

    # Summary comparison
    summary_data = {
        'Within-Category\nMean': category_analysis.get("within_category_mean", 0),
        'Between-Category\nMean': category_analysis.get("between_category_mean", 0),
    }

    axes[1].bar(list(summary_data.keys()), list(summary_data.values()),
                color=['steelblue', 'coral'], alpha=0.8)
    axes[1].set_ylabel("Spearman Correlation")
    axes[1].set_title("Within vs Between Category Correlation")
    axes[1].axhline(y=0.6, color='green', linestyle=':', alpha=0.5, label='Good threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Category-Specific Semantic Similarity Analysis", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_distance_heatmaps(
    vlm_dists: np.ndarray,
    ltx_dists: np.ndarray,
    labels: np.ndarray,
    label_names: list,
) -> bytes:
    """Create side-by-side distance matrix heatmaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sort by label for visualization
    sort_idx = np.argsort(labels)
    vlm_sorted = vlm_dists[np.ix_(sort_idx, sort_idx)]
    ltx_sorted = ltx_dists[np.ix_(sort_idx, sort_idx)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # VLM distances
    im1 = axes[0].imshow(vlm_sorted, cmap='viridis')
    axes[0].set_title("VLM Pairwise Distances")
    plt.colorbar(im1, ax=axes[0])

    # LTX distances
    im2 = axes[1].imshow(ltx_sorted, cmap='viridis')
    axes[1].set_title("LTX Pairwise Distances")
    plt.colorbar(im2, ax=axes[1])

    # Difference
    # Normalize both to [0, 1] for comparison
    vlm_norm = (vlm_sorted - vlm_sorted.min()) / (vlm_sorted.max() - vlm_sorted.min() + 1e-10)
    ltx_norm = (ltx_sorted - ltx_sorted.min()) / (ltx_sorted.max() - ltx_sorted.min() + 1e-10)
    diff = vlm_norm - ltx_norm

    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[2].set_title("Difference (VLM - LTX, normalized)")
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle("Pairwise Distance Matrices (sorted by category)", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_rank_agreement_plot(vlm_dists: np.ndarray, ltx_dists: np.ndarray) -> bytes:
    """Create rank agreement analysis plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = vlm_dists.shape[0]

    # For each point, get rank of neighbors
    vlm_ranks = np.argsort(np.argsort(vlm_dists, axis=1), axis=1)
    ltx_ranks = np.argsort(np.argsort(ltx_dists, axis=1), axis=1)

    # Compute rank correlation per point
    rank_correlations = []
    for i in range(n):
        r, _ = stats.spearmanr(vlm_ranks[i], ltx_ranks[i])
        if not np.isnan(r):
            rank_correlations.append(r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of rank correlations
    axes[0].hist(rank_correlations, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0].axvline(x=np.mean(rank_correlations), color='red', linestyle='--',
                    label=f'Mean: {np.mean(rank_correlations):.3f}')
    axes[0].axvline(x=np.median(rank_correlations), color='orange', linestyle='--',
                    label=f'Median: {np.median(rank_correlations):.3f}')
    axes[0].set_xlabel("Rank Correlation")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Per-Point Rank Correlation Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Rank agreement at different k
    ks = [1, 5, 10, 20, 50]
    agreements = []

    for k in ks:
        vlm_nn = np.argsort(vlm_dists, axis=1)[:, 1:k+1]  # k nearest (exclude self)
        ltx_nn = np.argsort(ltx_dists, axis=1)[:, 1:k+1]

        overlaps = []
        for i in range(n):
            overlap = len(set(vlm_nn[i]) & set(ltx_nn[i])) / k
            overlaps.append(overlap)
        agreements.append(np.mean(overlaps))

    axes[1].bar(range(len(ks)), agreements, color='coral', alpha=0.8)
    axes[1].set_xticks(range(len(ks)))
    axes[1].set_xticklabels([f'k={k}' for k in ks])
    axes[1].set_ylabel("Mean Overlap Fraction")
    axes[1].set_title("Nearest Neighbor Agreement at Various k")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add random baseline
    for i, k in enumerate(ks):
        random_baseline = k / n  # Expected overlap by chance
        axes[1].axhline(y=random_baseline, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle("Rank Agreement Analysis", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
