"""E-Q1.1: VLM Latent Space Visualization

Objective: Understand how Qwen2.5-VL organizes visual information internally.

Protocol:
1. Extract VLM latents for diverse images across multiple extraction points
2. Apply dimensionality reduction (PCA, t-SNE, UMAP)
3. Color-code by semantic category to assess clustering
4. Analyze clustering patterns at different layers

Extraction Points:
- ViT layer 12 (mid-depth)
- ViT layer 24 (late)
- ViT output (post-merge)
- LLM layer 8, 16, 24 (visual token positions)

Success Metrics:
- Silhouette score by category (measures cluster quality)
- k-NN classification accuracy
- Visual inspection of clustering behavior
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
from PIL import Image

from runner import ExperimentRunner


def eq1_1_vlm_visualization(runner: ExperimentRunner) -> dict:
    """Run VLM latent space visualization sub-experiment.

    This implementation:
    1. Generates/loads diverse test images across semantic categories
    2. Extracts latents from multiple layers of Qwen2.5-VL
    3. Runs t-SNE and UMAP visualization
    4. Computes silhouette scores and k-NN accuracy per layer
    5. Identifies which layer shows best semantic organization

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-Q1.1: VLM Latent Space Visualization")
    print("=" * 60)

    # Set up environment
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"eq1_1/stage": 0, "eq1_1/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate/Load test images
    # =========================================================================
    print("\n[Stage 1/5] Preparing test images...")

    images, labels, label_names = generate_diverse_images(n_per_category=30)
    print(f"  Generated {len(images)} images across {len(label_names)} categories")

    runner.log_metrics({
        "eq1_1/stage": 1,
        "eq1_1/progress": 0.1,
        "eq1_1/n_images": len(images),
        "eq1_1/n_categories": len(label_names),
    })

    # =========================================================================
    # Stage 2: Load model and extract latents from multiple layers
    # =========================================================================
    print("\n[Stage 2/5] Loading Qwen2.5-VL and extracting multi-layer latents...")

    latents_by_layer = extract_multilayer_latents(images, runner)
    print(f"  Extracted latents from {len(latents_by_layer)} layers")

    runner.log_metrics({"eq1_1/stage": 2, "eq1_1/progress": 0.4})

    # =========================================================================
    # Stage 3: Run dimensionality reduction
    # =========================================================================
    print("\n[Stage 3/5] Running dimensionality reduction...")

    from sklearn.manifold import TSNE

    try:
        from umap import UMAP
        has_umap = True
    except ImportError:
        has_umap = False
        print("  UMAP not available, using t-SNE only")

    projections = {}
    for layer_name, latents in latents_by_layer.items():
        print(f"  Processing {layer_name}...")

        # t-SNE projection
        perplexity = min(30, len(images) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        tsne_result = tsne.fit_transform(latents)

        projections[layer_name] = {"tsne": tsne_result}

        # UMAP projection if available
        if has_umap:
            umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
            umap_result = umap.fit_transform(latents)
            projections[layer_name]["umap"] = umap_result

    runner.log_metrics({"eq1_1/stage": 3, "eq1_1/progress": 0.6})

    # =========================================================================
    # Stage 4: Compute metrics per layer
    # =========================================================================
    print("\n[Stage 4/5] Computing clustering metrics...")

    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    metrics_by_layer = {}
    for layer_name, latents in latents_by_layer.items():
        print(f"  Analyzing {layer_name}...")

        # Silhouette score on raw latents
        sil_raw = silhouette_score(latents, labels)

        # Silhouette on t-SNE projection
        sil_tsne = silhouette_score(projections[layer_name]["tsne"], labels)

        # k-NN classification accuracy (5-fold CV)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn_scores = cross_val_score(knn, latents, labels, cv=5)
        knn_acc = knn_scores.mean()

        # Compute intra-class vs inter-class distance ratio
        intra, inter = compute_class_distances(latents, labels)
        distance_ratio = intra / inter if inter > 0 else float('inf')

        metrics_by_layer[layer_name] = {
            "silhouette_raw": float(sil_raw),
            "silhouette_tsne": float(sil_tsne),
            "knn_accuracy": float(knn_acc),
            "knn_std": float(knn_scores.std()),
            "intra_class_dist": float(intra),
            "inter_class_dist": float(inter),
            "distance_ratio": float(distance_ratio),
            "latent_dim": int(latents.shape[1]),
        }

        print(f"    Silhouette: {sil_raw:.3f}, k-NN acc: {knn_acc:.3f}")

    runner.log_metrics({"eq1_1/stage": 4, "eq1_1/progress": 0.8})

    # =========================================================================
    # Stage 5: Create visualizations and save artifacts
    # =========================================================================
    print("\n[Stage 5/5] Creating visualizations...")

    artifacts = []

    # Create t-SNE plots for each layer
    for layer_name, proj_data in projections.items():
        # t-SNE plot
        plot_bytes = create_visualization_plot(
            proj_data["tsne"],
            labels,
            label_names,
            metrics_by_layer[layer_name]["silhouette_tsne"],
            f"VLM Latent Space: {layer_name} (t-SNE)",
        )
        plot_path = runner.results.save_artifact(f"vlm_tsne_{layer_name}.png", plot_bytes)
        artifacts.append(plot_path)

        # UMAP plot if available
        if "umap" in proj_data:
            plot_bytes = create_visualization_plot(
                proj_data["umap"],
                labels,
                label_names,
                metrics_by_layer[layer_name]["silhouette_raw"],
                f"VLM Latent Space: {layer_name} (UMAP)",
            )
            plot_path = runner.results.save_artifact(f"vlm_umap_{layer_name}.png", plot_bytes)
            artifacts.append(plot_path)

    # Create comparison plot across layers
    comparison_plot = create_layer_comparison_plot(metrics_by_layer)
    comparison_path = runner.results.save_artifact("vlm_layer_comparison.png", comparison_plot)
    artifacts.append(comparison_path)

    # Save raw metrics data
    data = {
        "metrics_by_layer": metrics_by_layer,
        "projections": {
            layer: {k: v.tolist() for k, v in proj.items()}
            for layer, proj in projections.items()
        },
        "labels": labels.tolist(),
        "label_names": label_names,
        "n_images": len(images),
    }
    data_path = runner.results.save_json_artifact("vlm_latent_analysis.json", data)
    artifacts.append(data_path)

    runner.log_metrics({
        "eq1_1/stage": 5,
        "eq1_1/progress": 1.0,
        **{f"eq1_1/{layer}/silhouette": m["silhouette_raw"]
           for layer, m in metrics_by_layer.items()},
        **{f"eq1_1/{layer}/knn_acc": m["knn_accuracy"]
           for layer, m in metrics_by_layer.items()},
    })

    # =========================================================================
    # Analyze results and form findings
    # =========================================================================
    best_layer = max(metrics_by_layer.keys(),
                     key=lambda k: metrics_by_layer[k]["silhouette_raw"])
    best_sil = metrics_by_layer[best_layer]["silhouette_raw"]
    best_knn = metrics_by_layer[best_layer]["knn_accuracy"]

    if best_sil > 0.5 and best_knn > 0.8:
        finding = (
            f"Strong semantic organization in VLM latents. Best layer: {best_layer} "
            f"(silhouette={best_sil:.3f}, k-NN={best_knn:.3f}). "
            "VLM encodes clear category structure that may facilitate alignment."
        )
        semantic_quality = "excellent"
    elif best_sil > 0.3 or best_knn > 0.6:
        finding = (
            f"Moderate semantic organization in VLM latents. Best layer: {best_layer} "
            f"(silhouette={best_sil:.3f}, k-NN={best_knn:.3f}). "
            "Categories are distinguishable but with overlap."
        )
        semantic_quality = "good"
    else:
        finding = (
            f"Weak semantic clustering in VLM latents. Best layer: {best_layer} "
            f"(silhouette={best_sil:.3f}, k-NN={best_knn:.3f}). "
            "VLM may prioritize other features over category membership."
        )
        semantic_quality = "weak"

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "best_layer": best_layer,
            "best_silhouette": float(best_sil),
            "best_knn_accuracy": float(best_knn),
            "semantic_quality": semantic_quality,
            "n_images": len(images),
            "n_categories": len(label_names),
            "layers_analyzed": list(metrics_by_layer.keys()),
            **{f"{layer}_silhouette": m["silhouette_raw"]
               for layer, m in metrics_by_layer.items()},
        },
        "artifacts": artifacts,
    }


def generate_diverse_images(n_per_category: int = 30) -> tuple[list, np.ndarray, list]:
    """Generate diverse test images with semantic categories.

    Creates images across multiple semantic categories with visual variation:
    - Objects (shapes with colors)
    - Scenes (gradients, patterns)
    - Actions/motion (directional elements)

    Args:
        n_per_category: Number of images per category

    Returns:
        Tuple of (images, labels, label_names)
    """
    from PIL import Image, ImageDraw
    import random

    categories = [
        # Object categories
        ("circle_red", "red circle"),
        ("circle_blue", "blue circle"),
        ("square_green", "green square"),
        ("square_yellow", "yellow square"),
        ("triangle_purple", "purple triangle"),
        # Pattern categories
        ("stripes_horizontal", "horizontal stripes"),
        ("stripes_vertical", "vertical stripes"),
        # Scene categories
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
    """Create an image for a specific category with variation.

    Args:
        category: Category name
        variation_seed: Seed for variation within category

    Returns:
        PIL Image
    """
    from PIL import Image, ImageDraw
    import random

    random.seed(variation_seed)
    np.random.seed(variation_seed)

    img = Image.new("RGB", (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Position and size variation
    cx = random.randint(60, 164)
    cy = random.randint(60, 164)
    size = random.randint(30, 50)

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
        points = [
            (cx, cy - size),
            (cx - size, cy + size),
            (cx + size, cy + size),
        ]
        draw.polygon(points, fill=color)
    elif category == "stripes_horizontal":
        stripe_width = random.randint(10, 25)
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(2)]
        for y in range(0, 224, stripe_width * 2):
            draw.rectangle([0, y, 224, y + stripe_width], fill=colors[0])
            draw.rectangle([0, y + stripe_width, 224, y + stripe_width * 2], fill=colors[1])
    elif category == "stripes_vertical":
        stripe_width = random.randint(10, 25)
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(2)]
        for x in range(0, 224, stripe_width * 2):
            draw.rectangle([x, 0, x + stripe_width, 224], fill=colors[0])
            draw.rectangle([x + stripe_width, 0, x + stripe_width * 2, 224], fill=colors[1])
    elif category == "gradient_warm":
        # Warm gradient (red/orange/yellow)
        for y in range(224):
            r = int(255 - y * 0.3)
            g = int(100 + y * 0.5)
            b = int(50)
            draw.line([(0, y), (224, y)], fill=(r, g, b))
    elif category == "gradient_cool":
        # Cool gradient (blue/cyan/green)
        for y in range(224):
            r = int(50)
            g = int(100 + y * 0.5)
            b = int(255 - y * 0.3)
            draw.line([(0, y), (224, y)], fill=(r, g, b))

    return img


def extract_multilayer_latents(images: list, runner: ExperimentRunner) -> dict[str, np.ndarray]:
    """Extract latents from multiple layers of Qwen2.5-VL.

    Extraction points:
    - LLM hidden states at layers 8, 16, 24, and final layer
    - Mean pooled across sequence

    Args:
        images: List of PIL Images
        runner: ExperimentRunner for logging

    Returns:
        Dict mapping layer names to latent arrays [N_images, latent_dim]
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

    print(f"  Model loaded, extracting latents from {len(images)} images...")

    # Layers to extract (0-indexed, Qwen2.5-VL-7B has 28 layers)
    target_layers = [8, 16, 24, -1]  # -1 for final layer
    layer_names = ["layer_8", "layer_16", "layer_24", "layer_final"]

    latents_by_layer = {name: [] for name in layer_names}

    with torch.no_grad():
        for idx, img in enumerate(images):
            # Prepare input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image."},
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

            # Forward pass with hidden states
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract from target layers
            hidden_states = outputs.hidden_states  # Tuple of [batch, seq_len, hidden_dim]

            for layer_idx, layer_name in zip(target_layers, layer_names):
                # Get hidden state for this layer
                hs = hidden_states[layer_idx][0]  # [seq_len, hidden_dim]

                # Mean pool across sequence (simpler than identifying image tokens)
                pooled = hs.mean(dim=0).float().cpu().numpy()
                latents_by_layer[layer_name].append(pooled)

            if (idx + 1) % 20 == 0:
                progress = (idx + 1) / len(images)
                runner.log_metrics({"eq1_1/extraction_progress": progress})
                print(f"    Processed {idx + 1}/{len(images)} images")

    # Stack latents
    for layer_name in layer_names:
        latents_by_layer[layer_name] = np.stack(latents_by_layer[layer_name], axis=0)
        print(f"  {layer_name}: shape {latents_by_layer[layer_name].shape}")

    # Clean up
    del model
    del processor
    torch.cuda.empty_cache()

    return latents_by_layer


def compute_class_distances(latents: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute mean intra-class and inter-class distances.

    Args:
        latents: Latent vectors [N, D]
        labels: Class labels [N]

    Returns:
        Tuple of (mean_intra_class_dist, mean_inter_class_dist)
    """
    from scipy.spatial.distance import cdist

    unique_labels = np.unique(labels)

    intra_distances = []
    inter_distances = []

    for label in unique_labels:
        mask = labels == label
        class_latents = latents[mask]
        other_latents = latents[~mask]

        if len(class_latents) > 1:
            # Intra-class: distances within this class
            intra_dists = cdist(class_latents, class_latents, metric='euclidean')
            # Get upper triangle (exclude diagonal)
            triu_idx = np.triu_indices_from(intra_dists, k=1)
            intra_distances.extend(intra_dists[triu_idx])

        if len(other_latents) > 0:
            # Inter-class: distances to other classes
            inter_dists = cdist(class_latents, other_latents, metric='euclidean')
            inter_distances.extend(inter_dists.flatten())

    mean_intra = np.mean(intra_distances) if intra_distances else 0
    mean_inter = np.mean(inter_distances) if inter_distances else 1

    return mean_intra, mean_inter


def create_visualization_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    score: float,
    title: str,
) -> bytes:
    """Create a 2D visualization plot.

    Args:
        coords: 2D coordinates [N, 2]
        labels: Category labels for each point
        label_names: Names for each category
        score: Score to display (e.g., silhouette)
        title: Plot title

    Returns:
        PNG image as bytes
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a good colormap
    n_classes = len(label_names)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_classes, 10)))
    if n_classes > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    # Plot each category
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

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_layer_comparison_plot(metrics_by_layer: dict) -> bytes:
    """Create a comparison plot of metrics across layers.

    Args:
        metrics_by_layer: Dict mapping layer names to metric dicts

    Returns:
        PNG image as bytes
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(metrics_by_layer.keys())
    silhouettes = [metrics_by_layer[l]["silhouette_raw"] for l in layers]
    knn_accs = [metrics_by_layer[l]["knn_accuracy"] for l in layers]
    dist_ratios = [metrics_by_layer[l]["distance_ratio"] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Silhouette scores
    axes[0].bar(layers, silhouettes, color='steelblue', alpha=0.8)
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Semantic Clustering Quality")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    axes[0].axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    # k-NN accuracy
    axes[1].bar(layers, knn_accs, color='coral', alpha=0.8)
    axes[1].set_ylabel("k-NN Accuracy")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    # Distance ratio
    axes[2].bar(layers, dist_ratios, color='mediumpurple', alpha=0.8)
    axes[2].set_ylabel("Intra/Inter Distance Ratio")
    axes[2].set_title("Class Separation (lower is better)")
    axes[2].tick_params(axis='x', rotation=45)

    plt.suptitle("VLM Latent Space Analysis Across Layers", fontsize=14)
    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
