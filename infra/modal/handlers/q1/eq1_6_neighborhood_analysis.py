"""E-Q1.6: Neighborhood Analysis (Cross-Space Retrieval)

Objective: Test whether nearest neighbors are consistent across VLM and LTX spaces.

Rationale: For a small adapter to work, similar items should be similar in both spaces.

Protocol:
1. For each image, find k=10 nearest neighbors in VLM space
2. Check how many of those neighbors are also near in LTX space
3. Compute Recall@k for various k values
4. Analyze systematic differences

Metrics:
- Recall@k: Fraction of VLM neighbors that are also LTX neighbors
- Mean Reciprocal Rank: Average rank of VLM neighbors in LTX space
- Neighborhood overlap: Jaccard similarity of k-NN sets

Success Criteria:
- Recall@10 > 20%: Spaces share local structure
- Recall@10 > 40%: Strong local alignment
- Recall@10 < 10%: Local structure very different
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist

from runner import ExperimentRunner


def eq1_6_neighborhood_analysis(runner: ExperimentRunner) -> dict:
    """Run neighborhood analysis between VLM and LTX latent spaces.

    This implementation:
    1. Extracts paired latents from both models
    2. Computes k-NN sets for various k values
    3. Calculates Recall@k, MRR, and Jaccard overlap
    4. Analyzes patterns by category

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-Q1.6: Neighborhood Analysis (Cross-Space Retrieval)")
    print("=" * 60)

    # Set up environment
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"eq1_6/stage": 0, "eq1_6/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate test images
    # =========================================================================
    print("\n[Stage 1/5] Preparing test images...")

    images, labels, label_names = generate_test_images(n_per_category=50)
    n_images = len(images)
    print(f"  Generated {n_images} images across {len(label_names)} categories")

    runner.log_metrics({
        "eq1_6/stage": 1,
        "eq1_6/progress": 0.1,
        "eq1_6/n_images": n_images,
    })

    # =========================================================================
    # Stage 2: Extract latents
    # =========================================================================
    print("\n[Stage 2/5] Extracting latents...")

    vlm_latents = extract_vlm_latents(images, runner)
    ltx_latents = extract_ltx_latents(images, runner)
    ltx_latents_flat = ltx_latents.reshape(ltx_latents.shape[0], -1)

    print(f"  VLM: {vlm_latents.shape}, LTX: {ltx_latents_flat.shape}")

    runner.log_metrics({"eq1_6/stage": 2, "eq1_6/progress": 0.4})

    # =========================================================================
    # Stage 3: Compute k-NN for both spaces
    # =========================================================================
    print("\n[Stage 3/5] Computing nearest neighbors...")

    # Compute distance matrices (could use FAISS for larger datasets)
    vlm_dists = cdist(vlm_latents, vlm_latents, metric='euclidean')
    ltx_dists = cdist(ltx_latents_flat, ltx_latents_flat, metric='euclidean')

    # Get sorted neighbor indices (excluding self)
    vlm_nn_all = np.argsort(vlm_dists, axis=1)[:, 1:]  # Exclude self (index 0)
    ltx_nn_all = np.argsort(ltx_dists, axis=1)[:, 1:]

    print(f"  Computed neighbors for {n_images} images")

    runner.log_metrics({"eq1_6/stage": 3, "eq1_6/progress": 0.5})

    # =========================================================================
    # Stage 4: Compute retrieval metrics
    # =========================================================================
    print("\n[Stage 4/5] Computing retrieval metrics...")

    k_values = [1, 5, 10, 20, 50, 100]
    metrics = {}

    for k in k_values:
        if k >= n_images - 1:
            continue

        recall, mrr, jaccard = compute_retrieval_metrics(
            vlm_nn_all, ltx_nn_all, k
        )

        metrics[f"recall@{k}"] = recall
        metrics[f"mrr@{k}"] = mrr
        metrics[f"jaccard@{k}"] = jaccard

        # Expected by random chance
        random_recall = k / (n_images - 1)
        metrics[f"recall@{k}_random"] = random_recall
        metrics[f"recall@{k}_lift"] = recall / random_recall if random_recall > 0 else 0

        print(f"  k={k}: Recall={recall:.3f} (random={random_recall:.3f}, lift={metrics[f'recall@{k}_lift']:.1f}x), MRR={mrr:.3f}")

    runner.log_metrics({
        "eq1_6/stage": 4,
        "eq1_6/progress": 0.7,
        **{f"eq1_6/{k}": v for k, v in metrics.items() if "recall@" in k or "mrr@" in k},
    })

    # =========================================================================
    # Stage 5: Category analysis and visualizations
    # =========================================================================
    print("\n[Stage 5/5] Analyzing by category and creating visualizations...")

    # Category-specific recall
    category_metrics = compute_category_metrics(
        vlm_nn_all, ltx_nn_all, labels, label_names, k=10
    )

    # Bidirectional analysis (LTX -> VLM)
    reverse_metrics = {}
    for k in [5, 10, 20]:
        if k >= n_images - 1:
            continue
        recall_rev, mrr_rev, jaccard_rev = compute_retrieval_metrics(
            ltx_nn_all, vlm_nn_all, k
        )
        reverse_metrics[f"ltx_to_vlm_recall@{k}"] = recall_rev
        reverse_metrics[f"ltx_to_vlm_mrr@{k}"] = mrr_rev

    print(f"  Reverse (LTX->VLM) Recall@10: {reverse_metrics.get('ltx_to_vlm_recall@10', 0):.3f}")

    artifacts = []

    # Recall@k curve plot
    recall_plot = create_recall_curve_plot(metrics, n_images)
    recall_path = runner.results.save_artifact("neighborhood_recall_curve.png", recall_plot)
    artifacts.append(recall_path)

    # Category-specific analysis plot
    category_plot = create_category_recall_plot(category_metrics, label_names)
    category_path = runner.results.save_artifact("neighborhood_by_category.png", category_plot)
    artifacts.append(category_path)

    # Neighbor rank distribution
    rank_plot = create_neighbor_rank_plot(vlm_nn_all, ltx_nn_all, n_images)
    rank_path = runner.results.save_artifact("neighborhood_rank_distribution.png", rank_plot)
    artifacts.append(rank_path)

    # Example retrievals
    example_plot = create_example_retrievals_plot(
        vlm_nn_all, ltx_nn_all, labels, label_names, k=10
    )
    example_path = runner.results.save_artifact("neighborhood_examples.png", example_plot)
    artifacts.append(example_path)

    # Save detailed results
    data = {
        "vlm_to_ltx": {k: float(v) for k, v in metrics.items()},
        "ltx_to_vlm": {k: float(v) for k, v in reverse_metrics.items()},
        "category_metrics": {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()}
            for k, v in category_metrics.items()
        },
        "n_images": n_images,
        "n_categories": len(label_names),
    }
    data_path = runner.results.save_json_artifact("neighborhood_analysis.json", data)
    artifacts.append(data_path)

    runner.log_metrics({"eq1_6/stage": 5, "eq1_6/progress": 1.0})

    # =========================================================================
    # Form conclusions
    # =========================================================================
    recall_10 = metrics.get("recall@10", 0)
    recall_10_lift = metrics.get("recall@10_lift", 0)
    mrr_10 = metrics.get("mrr@10", 0)

    if recall_10 > 0.4:
        quality = "strong"
        finding = (
            f"Strong local alignment! Recall@10={recall_10:.3f} ({recall_10_lift:.1f}x over random). "
            f"VLM and LTX spaces share significant local structure. "
            f"MRR@10={mrr_10:.3f}. A small adapter should preserve neighborhood relationships."
        )
    elif recall_10 > 0.2:
        quality = "good"
        finding = (
            f"Good local alignment. Recall@10={recall_10:.3f} ({recall_10_lift:.1f}x over random). "
            f"Many VLM neighbors are also LTX neighbors. "
            f"MRR@10={mrr_10:.3f}. Adapter should work well for most cases."
        )
    elif recall_10 > 0.1:
        quality = "moderate"
        finding = (
            f"Moderate local alignment. Recall@10={recall_10:.3f} ({recall_10_lift:.1f}x over random). "
            f"Some local structure preserved but significant differences exist. "
            f"MRR@10={mrr_10:.3f}. May need larger adapter or alternative approach."
        )
    else:
        quality = "weak"
        finding = (
            f"Weak local alignment. Recall@10={recall_10:.3f} ({recall_10_lift:.1f}x over random). "
            f"Local neighborhoods differ substantially between spaces. "
            f"MRR@10={mrr_10:.3f}. Consider different adapter architecture or alignment strategy."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "recall_at_5": float(metrics.get("recall@5", 0)),
            "recall_at_10": float(recall_10),
            "recall_at_20": float(metrics.get("recall@20", 0)),
            "recall_at_10_lift": float(recall_10_lift),
            "mrr_at_10": float(mrr_10),
            "jaccard_at_10": float(metrics.get("jaccard@10", 0)),
            "ltx_to_vlm_recall_at_10": float(reverse_metrics.get("ltx_to_vlm_recall@10", 0)),
            "quality": quality,
            "n_images": n_images,
            "n_categories": len(label_names),
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


def compute_retrieval_metrics(
    source_nn: np.ndarray,
    target_nn: np.ndarray,
    k: int,
) -> tuple[float, float, float]:
    """Compute retrieval metrics for cross-space neighborhood comparison.

    Args:
        source_nn: Nearest neighbor indices for source space [N, N-1]
        target_nn: Nearest neighbor indices for target space [N, N-1]
        k: Number of neighbors to consider

    Returns:
        (recall@k, mrr@k, jaccard@k)
    """
    n = source_nn.shape[0]

    recalls = []
    mrrs = []
    jaccards = []

    for i in range(n):
        source_k = set(source_nn[i, :k])
        target_k = set(target_nn[i, :k])

        # Recall: fraction of source neighbors found in target top-k
        overlap = len(source_k & target_k)
        recall = overlap / k
        recalls.append(recall)

        # MRR: mean reciprocal rank of source neighbors in target ranking
        rr_sum = 0
        for neighbor in source_k:
            # Find rank in target
            target_rank = np.where(target_nn[i] == neighbor)[0]
            if len(target_rank) > 0:
                rr_sum += 1.0 / (target_rank[0] + 1)
        mrr = rr_sum / k
        mrrs.append(mrr)

        # Jaccard: intersection over union
        jaccard = len(source_k & target_k) / len(source_k | target_k) if source_k | target_k else 0
        jaccards.append(jaccard)

    return np.mean(recalls), np.mean(mrrs), np.mean(jaccards)


def compute_category_metrics(
    vlm_nn: np.ndarray,
    ltx_nn: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    k: int = 10,
) -> dict:
    """Compute retrieval metrics by category."""
    results = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        idx = np.where(mask)[0]

        if len(idx) > k:
            # Get neighbors for this category only
            recalls = []
            for i in idx:
                source_k = set(vlm_nn[i, :k])
                target_k = set(ltx_nn[i, :k])
                overlap = len(source_k & target_k)
                recalls.append(overlap / k)

            results[label_names[label]] = {
                "recall_at_k": np.mean(recalls),
                "n_samples": len(idx),
            }

    return results


def create_recall_curve_plot(metrics: dict, n_images: int) -> bytes:
    """Create Recall@k curve plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract k values and recalls
    k_values = []
    recalls = []
    random_recalls = []
    lifts = []

    for k in [1, 5, 10, 20, 50, 100]:
        if f"recall@{k}" in metrics:
            k_values.append(k)
            recalls.append(metrics[f"recall@{k}"])
            random_recalls.append(metrics.get(f"recall@{k}_random", k / (n_images - 1)))
            lifts.append(metrics.get(f"recall@{k}_lift", 1))

    # Recall curve
    axes[0].plot(k_values, recalls, 'b-o', linewidth=2, markersize=8, label='Actual')
    axes[0].plot(k_values, random_recalls, 'r--', linewidth=2, label='Random baseline')
    axes[0].fill_between(k_values, random_recalls, recalls, alpha=0.2)
    axes[0].set_xlabel("k (number of neighbors)", fontsize=12)
    axes[0].set_ylabel("Recall@k", fontsize=12)
    axes[0].set_title("Cross-Space Neighborhood Recall")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(recalls) * 1.1)

    # Add threshold lines
    axes[0].axhline(y=0.4, color='green', linestyle=':', alpha=0.7, label='Strong (0.4)')
    axes[0].axhline(y=0.2, color='orange', linestyle=':', alpha=0.7, label='Good (0.2)')

    # Lift over random
    axes[1].bar(range(len(k_values)), lifts, color='steelblue', alpha=0.8)
    axes[1].set_xticks(range(len(k_values)))
    axes[1].set_xticklabels([f'k={k}' for k in k_values])
    axes[1].set_ylabel("Lift over random", fontsize=12)
    axes[1].set_title("Improvement over Random Baseline")
    axes[1].axhline(y=1, color='red', linestyle='--', label='Random baseline')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, lift in enumerate(lifts):
        axes[1].annotate(f'{lift:.1f}x',
                         xy=(i, lift),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=10)

    plt.suptitle("Neighborhood Retrieval Analysis", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_category_recall_plot(category_metrics: dict, label_names: list) -> bytes:
    """Create category-specific recall plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    categories = list(category_metrics.keys())
    recalls = [category_metrics[c]["recall_at_k"] for c in categories]
    n_samples = [category_metrics[c]["n_samples"] for c in categories]

    bars = ax.bar(range(len(categories)), recalls, color='steelblue', alpha=0.8)

    # Color bars by performance
    for bar, recall in zip(bars, recalls):
        if recall > 0.4:
            bar.set_color('green')
        elif recall > 0.2:
            bar.set_color('steelblue')
        else:
            bar.set_color('coral')

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("Neighborhood Recall by Category")

    # Add threshold lines
    ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.7, label='Strong')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Good')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_neighbor_rank_plot(vlm_nn: np.ndarray, ltx_nn: np.ndarray, n_images: int) -> bytes:
    """Create neighbor rank distribution plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # For each VLM top-10 neighbor, find its rank in LTX
    k = 10
    all_ranks = []

    for i in range(n_images):
        vlm_k = vlm_nn[i, :k]
        for neighbor in vlm_k:
            ltx_rank = np.where(ltx_nn[i] == neighbor)[0]
            if len(ltx_rank) > 0:
                all_ranks.append(ltx_rank[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of ranks
    axes[0].hist(all_ranks, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0].axvline(x=np.median(all_ranks), color='red', linestyle='--',
                    label=f'Median: {np.median(all_ranks):.0f}')
    axes[0].axvline(x=np.mean(all_ranks), color='orange', linestyle='--',
                    label=f'Mean: {np.mean(all_ranks):.0f}')
    axes[0].set_xlabel("LTX Rank of VLM Top-10 Neighbors", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Distribution of Cross-Space Neighbor Ranks")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_ranks = np.sort(all_ranks)
    cumulative = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)

    axes[1].plot(sorted_ranks, cumulative, 'b-', linewidth=2)
    axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    axes[1].axvline(x=10, color='green', linestyle='--', alpha=0.7, label='k=10')
    axes[1].axvline(x=20, color='orange', linestyle='--', alpha=0.7, label='k=20')

    # Mark specific percentiles
    for percentile in [0.25, 0.5, 0.75]:
        rank_at_percentile = sorted_ranks[int(percentile * len(sorted_ranks))]
        axes[1].annotate(f'{percentile*100:.0f}%: rank {rank_at_percentile:.0f}',
                         xy=(rank_at_percentile, percentile),
                         xytext=(10, 0),
                         textcoords="offset points",
                         fontsize=9)

    axes[1].set_xlabel("LTX Rank", fontsize=12)
    axes[1].set_ylabel("Cumulative Proportion", fontsize=12)
    axes[1].set_title("Cumulative Distribution of Neighbor Ranks")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, min(n_images - 1, 200))

    plt.suptitle("Cross-Space Neighbor Rank Analysis", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_example_retrievals_plot(
    vlm_nn: np.ndarray,
    ltx_nn: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    k: int = 10,
) -> bytes:
    """Create example retrieval visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Find examples with high and low agreement
    n = vlm_nn.shape[0]
    agreements = []

    for i in range(n):
        vlm_k = set(vlm_nn[i, :k])
        ltx_k = set(ltx_nn[i, :k])
        overlap = len(vlm_k & ltx_k) / k
        agreements.append(overlap)

    # Sort by agreement
    sorted_idx = np.argsort(agreements)

    # Select examples: worst, median, best
    example_idx = [
        sorted_idx[0],  # Worst agreement
        sorted_idx[len(sorted_idx) // 2],  # Median
        sorted_idx[-1],  # Best agreement
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    titles = ["Lowest Agreement", "Median Agreement", "Highest Agreement"]

    for ax_idx, (idx, title) in enumerate(zip(example_idx, titles)):
        vlm_neighbors = vlm_nn[idx, :k]
        ltx_neighbors = ltx_nn[idx, :k]

        # Get labels
        query_label = label_names[labels[idx]]
        vlm_neighbor_labels = [label_names[labels[n]] for n in vlm_neighbors]
        ltx_neighbor_labels = [label_names[labels[n]] for n in ltx_neighbors]

        # Check which are shared
        shared = set(vlm_neighbors) & set(ltx_neighbors)

        # Create text representation
        vlm_text = f"VLM: " + ", ".join([
            f"[{vlm_neighbor_labels[i]}]" if vlm_neighbors[i] in shared else vlm_neighbor_labels[i]
            for i in range(k)
        ])
        ltx_text = f"LTX: " + ", ".join([
            f"[{ltx_neighbor_labels[i]}]" if ltx_neighbors[i] in shared else ltx_neighbor_labels[i]
            for i in range(k)
        ])

        axes[ax_idx].axis('off')
        axes[ax_idx].text(0.5, 0.7, f"Query: {query_label} (Agreement: {agreements[idx]:.0%})",
                          ha='center', va='center', fontsize=14, fontweight='bold',
                          transform=axes[ax_idx].transAxes)
        axes[ax_idx].text(0.5, 0.4, vlm_text,
                          ha='center', va='center', fontsize=10,
                          transform=axes[ax_idx].transAxes)
        axes[ax_idx].text(0.5, 0.2, ltx_text,
                          ha='center', va='center', fontsize=10,
                          transform=axes[ax_idx].transAxes)
        axes[ax_idx].text(0.5, 0.05, "[bracketed] = shared neighbors",
                          ha='center', va='center', fontsize=9, style='italic',
                          transform=axes[ax_idx].transAxes)
        axes[ax_idx].set_title(title, fontsize=12)

    plt.suptitle(f"Example Retrievals (k={k})", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
