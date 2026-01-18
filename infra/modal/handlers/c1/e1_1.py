"""E1.1: Latent Space Visualization

Objective: Understand the structure of Qwen2.5-VL's visual latent space
before attempting reconstruction.

Protocol:
1. Extract pre-merge and post-merge latents from diverse images
2. Apply dimensionality reduction (t-SNE, UMAP) to visualize
3. Color-code by image category to assess semantic clustering

Success Metrics:
- Silhouette score by category (measures cluster quality)
- Visual inspection of clustering behavior

This is a STARTER implementation. Extend it to:
- Process more images
- Extract pre-merge latents (requires model hooks)
- Add UMAP visualization
- Analyze temporal structure for video
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
from PIL import Image

from runner import ExperimentRunner


def e1_1_latent_visualization(runner: ExperimentRunner) -> dict:
    """Run latent space visualization sub-experiment.

    This implementation:
    1. Generates synthetic test images (colored shapes)
    2. Extracts post-merge latents from Qwen2.5-VL
    3. Runs t-SNE visualization
    4. Computes silhouette score for semantic clustering

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E1.1: Latent Space Visualization")
    print("=" * 60)

    # Set up environment
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"e1_1/stage": 0, "e1_1/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate synthetic test images
    # =========================================================================
    print("\n[Stage 1/4] Generating synthetic test images...")

    images, labels, label_names = generate_synthetic_images(n_per_category=20)
    print(f"  Generated {len(images)} images across {len(label_names)} categories")

    runner.log_metrics({"e1_1/stage": 1, "e1_1/progress": 0.25, "e1_1/n_images": len(images)})

    # =========================================================================
    # Stage 2: Load model and extract latents
    # =========================================================================
    print("\n[Stage 2/4] Loading Qwen2.5-VL and extracting latents...")

    latents = extract_latents(images, runner)
    print(f"  Extracted latents shape: {latents.shape}")

    runner.log_metrics(
        {
            "e1_1/stage": 2,
            "e1_1/progress": 0.5,
            "e1_1/latent_dim": latents.shape[-1],
        }
    )

    # =========================================================================
    # Stage 3: Run t-SNE
    # =========================================================================
    print("\n[Stage 3/4] Running t-SNE dimensionality reduction...")

    from sklearn.manifold import TSNE

    # Latents are already pooled to [N_images, hidden_dim] in extract_latents
    # If they have 3 dimensions, pool across tokens
    if len(latents.shape) == 3:
        latents_pooled = latents.mean(axis=1)
    else:
        latents_pooled = latents
    print(f"  Latents shape for t-SNE: {latents_pooled.shape}")

    tsne = TSNE(n_components=2, perplexity=min(30, len(images) - 1), random_state=42)
    tsne_result = tsne.fit_transform(latents_pooled)
    print(f"  t-SNE result shape: {tsne_result.shape}")

    runner.log_metrics({"e1_1/stage": 3, "e1_1/progress": 0.75})

    # =========================================================================
    # Stage 4: Compute metrics and create visualization
    # =========================================================================
    print("\n[Stage 4/4] Computing metrics and creating visualization...")

    # Compute silhouette score (measures cluster quality)
    from sklearn.metrics import silhouette_score

    sil_score = silhouette_score(tsne_result, labels)
    print(f"  Silhouette score: {sil_score:.3f}")

    # Create visualization
    plot_bytes = create_tsne_plot(tsne_result, labels, label_names, sil_score)
    plot_path = runner.results.save_artifact("tsne_visualization.png", plot_bytes)
    print(f"  Saved visualization: {plot_path}")

    # Save raw data for further analysis
    data = {
        "tsne_coords": tsne_result.tolist(),
        "labels": labels.tolist(),
        "label_names": label_names,
        "silhouette_score": float(sil_score),
        "n_images": len(images),
        "latent_dim": int(latents.shape[-1]),
    }
    data_path = runner.results.save_json_artifact("latent_analysis.json", data)

    runner.log_metrics(
        {
            "e1_1/stage": 4,
            "e1_1/progress": 1.0,
            "e1_1/silhouette_score": sil_score,
        }
    )

    # =========================================================================
    # Return results
    # =========================================================================
    # Interpret results
    if sil_score > 0.5:
        finding = f"Strong semantic clustering (silhouette={sil_score:.3f}). Latents clearly encode category information."
    elif sil_score > 0.25:
        finding = f"Moderate semantic clustering (silhouette={sil_score:.3f}). Categories partially separable in latent space."
    else:
        finding = f"Weak semantic clustering (silhouette={sil_score:.3f}). Categories not well separated - may need different features."

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "silhouette_score": float(sil_score),
            "n_images": len(images),
            "n_categories": len(label_names),
            "latent_dim": int(latents.shape[-1]),
        },
        "artifacts": [plot_path, data_path],
    }


def generate_synthetic_images(n_per_category: int = 20) -> tuple[list, np.ndarray, list]:
    """Generate synthetic test images with known categories.

    Creates simple colored shapes on white backgrounds:
    - Red circles
    - Blue squares
    - Green triangles

    Args:
        n_per_category: Number of images per category

    Returns:
        Tuple of (images, labels, label_names)
    """
    from PIL import Image, ImageDraw

    categories = [
        ("red_circle", (255, 0, 0), "circle"),
        ("blue_square", (0, 0, 255), "square"),
        ("green_triangle", (0, 255, 0), "triangle"),
    ]

    images = []
    labels = []
    label_names = [c[0] for c in categories]

    for cat_idx, (name, color, shape) in enumerate(categories):
        for i in range(n_per_category):
            # Create white background
            img = Image.new("RGB", (224, 224), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            # Random position and size
            cx = np.random.randint(50, 174)
            cy = np.random.randint(50, 174)
            size = np.random.randint(30, 60)

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
            labels.append(cat_idx)

    return images, np.array(labels), label_names


def extract_latents(images: list, runner: ExperimentRunner) -> np.ndarray:
    """Extract latents from Qwen2.5-VL for a list of images.

    Args:
        images: List of PIL Images
        runner: ExperimentRunner for logging

    Returns:
        Numpy array of shape [N_images, N_tokens, latent_dim]
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

    print(f"  Model loaded on {model.device}")
    print(f"  Extracting latents from {len(images)} images...")

    latents_list = []
    batch_size = 8  # Process in batches to manage memory

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            # Process images through the vision encoder
            # We need to use the model's visual processing pipeline
            for img in batch_images:
                # Format for Qwen2.5-VL
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Describe this image briefly."},
                        ],
                    }
                ]

                # Process
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text],
                    images=[img],
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                # Get vision embeddings through forward pass
                # Access the visual features from the model
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Extract the last hidden state
                # Shape: [1, seq_len, hidden_dim]
                hidden_states = outputs.hidden_states[-1]

                # Get the image token positions (first N tokens after special tokens)
                # For simplicity, take mean of all non-padding tokens
                # In practice, you'd want to identify image tokens specifically
                latent = hidden_states[0].float().cpu().numpy()

                # Take mean across sequence for this simple version
                # Shape: [1, hidden_dim]
                latent_pooled = latent.mean(axis=0, keepdims=True)
                latents_list.append(latent_pooled)

            progress = min(1.0, (i + batch_size) / len(images))
            runner.log_metrics({"e1_1/extraction_progress": progress})
            print(f"    Processed {min(i + batch_size, len(images))}/{len(images)} images")

    # Stack all latents
    latents = np.concatenate(latents_list, axis=0)

    # Clean up GPU memory
    del model
    del processor
    torch.cuda.empty_cache()

    return latents


def create_tsne_plot(
    tsne_result: np.ndarray, labels: np.ndarray, label_names: list, silhouette: float
) -> bytes:
    """Create a t-SNE visualization plot.

    Args:
        tsne_result: 2D array of t-SNE coordinates [N, 2]
        labels: Category labels for each point
        label_names: Names for each category
        silhouette: Silhouette score to display

    Returns:
        PNG image as bytes
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))

    # Plot each category
    for i, name in enumerate(label_names):
        mask = labels == i
        ax.scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            c=[colors[i]],
            label=name,
            alpha=0.7,
            s=50,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Qwen2.5-VL Latent Space Visualization\nSilhouette Score: {silhouette:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
