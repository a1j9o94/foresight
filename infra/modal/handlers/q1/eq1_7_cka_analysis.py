"""E-Q1.7: Cross-Space CKA Analysis (Optional)

Objective: Use Centered Kernel Alignment (CKA) to measure representational similarity
between VLM layers and LTX-Video latent space.

Protocol:
1. Extract representations from multiple VLM layers
2. Compute CKA between each VLM layer and LTX latent space
3. Create CKA heatmap showing which VLM layers align best with LTX
4. Identify optimal extraction point for adapter training

CKA (Kornblith et al., 2019):
- Measures representational similarity that is invariant to orthogonal transformations
- Works across different dimensionalities
- Standard tool for comparing neural network representations

Success Criteria:
- CKA > 0.4: Meaningful representational overlap
- Identify which VLM layer has highest alignment with LTX
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
from PIL import Image

from runner import ExperimentRunner


def eq1_7_cka_analysis(runner: ExperimentRunner) -> dict:
    """Run CKA analysis between VLM layers and LTX latent space.

    This implementation:
    1. Extracts representations from multiple VLM layers
    2. Extracts LTX VAE latents
    3. Computes CKA for each VLM layer vs LTX
    4. Creates heatmap showing alignment patterns

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-Q1.7: CKA Analysis (Representational Similarity)")
    print("=" * 60)

    # Set up environment
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"eq1_7/stage": 0, "eq1_7/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate test images
    # =========================================================================
    print("\n[Stage 1/5] Preparing test images...")

    images, labels, label_names = generate_test_images(n_per_category=40)
    print(f"  Generated {len(images)} images")

    runner.log_metrics({"eq1_7/stage": 1, "eq1_7/progress": 0.1})

    # =========================================================================
    # Stage 2: Extract multi-layer VLM representations
    # =========================================================================
    print("\n[Stage 2/5] Extracting multi-layer VLM representations...")

    vlm_by_layer = extract_vlm_multilayer(images, runner)
    vlm_layers = list(vlm_by_layer.keys())
    print(f"  Extracted {len(vlm_layers)} VLM layer representations")
    for layer, data in vlm_by_layer.items():
        print(f"    {layer}: {data.shape}")

    runner.log_metrics({"eq1_7/stage": 2, "eq1_7/progress": 0.4})

    # =========================================================================
    # Stage 3: Extract LTX representations
    # =========================================================================
    print("\n[Stage 3/5] Extracting LTX representations...")

    ltx_latents = extract_ltx_latents(images, runner)
    ltx_flat = ltx_latents.reshape(ltx_latents.shape[0], -1)
    ltx_channel_mean = ltx_latents.mean(axis=(2, 3))

    print(f"  LTX flat: {ltx_flat.shape}")
    print(f"  LTX channel-mean: {ltx_channel_mean.shape}")

    runner.log_metrics({"eq1_7/stage": 3, "eq1_7/progress": 0.5})

    # =========================================================================
    # Stage 4: Compute CKA for all pairs
    # =========================================================================
    print("\n[Stage 4/5] Computing CKA for all layer pairs...")

    cka_results = {}

    # VLM layers vs LTX (flat)
    print("  Computing VLM vs LTX (flat)...")
    for layer_name, vlm_data in vlm_by_layer.items():
        cka_value = compute_cka(vlm_data, ltx_flat)
        cka_results[f"{layer_name}_vs_ltx_flat"] = cka_value
        print(f"    {layer_name} vs LTX flat: CKA = {cka_value:.4f}")

    # VLM layers vs LTX (channel mean)
    print("  Computing VLM vs LTX (channel-mean)...")
    for layer_name, vlm_data in vlm_by_layer.items():
        cka_value = compute_cka(vlm_data, ltx_channel_mean)
        cka_results[f"{layer_name}_vs_ltx_channel"] = cka_value
        print(f"    {layer_name} vs LTX channel: CKA = {cka_value:.4f}")

    # VLM self-similarity (layer-to-layer)
    print("  Computing VLM layer-to-layer CKA...")
    vlm_cka_matrix = np.zeros((len(vlm_layers), len(vlm_layers)))
    for i, layer1 in enumerate(vlm_layers):
        for j, layer2 in enumerate(vlm_layers):
            if i <= j:
                cka_value = compute_cka(vlm_by_layer[layer1], vlm_by_layer[layer2])
                vlm_cka_matrix[i, j] = cka_value
                vlm_cka_matrix[j, i] = cka_value

    runner.log_metrics({
        "eq1_7/stage": 4,
        "eq1_7/progress": 0.8,
        **{f"eq1_7/cka_{k}": v for k, v in cka_results.items()},
    })

    # =========================================================================
    # Stage 5: Analysis and visualization
    # =========================================================================
    print("\n[Stage 5/5] Creating visualizations...")

    artifacts = []

    # Find best VLM layer for LTX alignment
    best_layer_flat = max(
        [(k, v) for k, v in cka_results.items() if "ltx_flat" in k],
        key=lambda x: x[1]
    )
    best_layer_channel = max(
        [(k, v) for k, v in cka_results.items() if "ltx_channel" in k],
        key=lambda x: x[1]
    )

    print(f"  Best VLM layer (vs LTX flat): {best_layer_flat[0]} (CKA={best_layer_flat[1]:.4f})")
    print(f"  Best VLM layer (vs LTX channel): {best_layer_channel[0]} (CKA={best_layer_channel[1]:.4f})")

    # CKA heatmap (VLM layers vs LTX)
    heatmap_plot = create_cka_heatmap(cka_results, vlm_layers)
    heatmap_path = runner.results.save_artifact("cka_heatmap.png", heatmap_plot)
    artifacts.append(heatmap_path)

    # VLM self-similarity heatmap
    self_sim_plot = create_vlm_self_similarity_plot(vlm_cka_matrix, vlm_layers)
    self_sim_path = runner.results.save_artifact("cka_vlm_self_similarity.png", self_sim_plot)
    artifacts.append(self_sim_path)

    # CKA curve across VLM layers
    curve_plot = create_cka_curve_plot(cka_results, vlm_layers)
    curve_path = runner.results.save_artifact("cka_layer_curve.png", curve_plot)
    artifacts.append(curve_path)

    # Save detailed results
    data = {
        "cka_results": {k: float(v) for k, v in cka_results.items()},
        "vlm_cka_matrix": vlm_cka_matrix.tolist(),
        "vlm_layers": vlm_layers,
        "best_layer_flat": {
            "layer": best_layer_flat[0].replace("_vs_ltx_flat", ""),
            "cka": float(best_layer_flat[1]),
        },
        "best_layer_channel": {
            "layer": best_layer_channel[0].replace("_vs_ltx_channel", ""),
            "cka": float(best_layer_channel[1]),
        },
        "n_images": len(images),
    }
    data_path = runner.results.save_json_artifact("cka_analysis.json", data)
    artifacts.append(data_path)

    runner.log_metrics({"eq1_7/stage": 5, "eq1_7/progress": 1.0})

    # =========================================================================
    # Form conclusions
    # =========================================================================
    max_cka = max(cka_results.values())
    best_layer_name = best_layer_flat[0].replace("_vs_ltx_flat", "")

    if max_cka > 0.5:
        quality = "strong"
        finding = (
            f"Strong representational alignment found! Best CKA={max_cka:.3f} "
            f"at {best_layer_name}. This layer preserves similar representations to LTX "
            "and should be optimal for adapter training."
        )
    elif max_cka > 0.4:
        quality = "good"
        finding = (
            f"Good representational alignment. Best CKA={max_cka:.3f} "
            f"at {best_layer_name}. Meaningful overlap exists between VLM and LTX spaces."
        )
    elif max_cka > 0.25:
        quality = "moderate"
        finding = (
            f"Moderate representational alignment. Best CKA={max_cka:.3f} "
            f"at {best_layer_name}. Some structural similarity but non-linear adapter "
            "may be needed for effective bridging."
        )
    else:
        quality = "weak"
        finding = (
            f"Weak representational alignment. Best CKA={max_cka:.3f} "
            f"at {best_layer_name}. Limited structural overlap suggests significant "
            "adapter capacity may be needed."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "max_cka": float(max_cka),
            "best_layer": best_layer_name,
            "best_layer_cka_flat": float(best_layer_flat[1]),
            "best_layer_cka_channel": float(best_layer_channel[1]),
            "cka_layer_4": float(cka_results.get("layer_4_vs_ltx_flat", 0)),
            "cka_layer_8": float(cka_results.get("layer_8_vs_ltx_flat", 0)),
            "cka_layer_16": float(cka_results.get("layer_16_vs_ltx_flat", 0)),
            "cka_layer_24": float(cka_results.get("layer_24_vs_ltx_flat", 0)),
            "quality": quality,
            "n_images": len(images),
            "n_vlm_layers": len(vlm_layers),
        },
        "artifacts": artifacts,
    }


def generate_test_images(n_per_category: int = 40) -> tuple[list, np.ndarray, list]:
    """Generate test images."""
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


def extract_vlm_multilayer(images: list, runner: ExperimentRunner) -> dict[str, np.ndarray]:
    """Extract representations from multiple VLM layers."""
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

    # Layers to extract (0-indexed)
    target_layers = [4, 8, 12, 16, 20, 24, -1]
    layer_names = ["layer_4", "layer_8", "layer_12", "layer_16", "layer_20", "layer_24", "layer_final"]

    latents_by_layer = {name: [] for name in layer_names}

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
            hidden_states = outputs.hidden_states

            for layer_idx, layer_name in zip(target_layers, layer_names):
                hs = hidden_states[layer_idx][0]
                pooled = hs.mean(dim=0).float().cpu().numpy()
                latents_by_layer[layer_name].append(pooled)

            if (idx + 1) % 50 == 0:
                print(f"    VLM: {idx + 1}/{len(images)}")

    del model, processor
    torch.cuda.empty_cache()

    return {k: np.stack(v, axis=0) for k, v in latents_by_layer.items()}


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


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    CKA (Centered Kernel Alignment) measures representational similarity
    that is invariant to orthogonal transformations and isotropic scaling.

    Args:
        X: First representation matrix [N, D1]
        Y: Second representation matrix [N, D2]

    Returns:
        CKA similarity score in [0, 1]
    """
    # Center the representations
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Compute Gram matrices
    XtX = X @ X.T  # [N, N]
    YtY = Y @ Y.T  # [N, N]

    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_xy = np.sum(XtX * YtY)
    hsic_xx = np.sum(XtX * XtX)
    hsic_yy = np.sum(YtY * YtY)

    # CKA
    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)

    return float(cka)


def create_cka_heatmap(cka_results: dict, vlm_layers: list) -> bytes:
    """Create CKA heatmap visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Organize data for heatmap
    ltx_types = ["ltx_flat", "ltx_channel"]
    heatmap_data = np.zeros((len(vlm_layers), len(ltx_types)))

    for i, layer in enumerate(vlm_layers):
        for j, ltx_type in enumerate(ltx_types):
            key = f"{layer}_vs_{ltx_type}"
            heatmap_data[i, j] = cka_results.get(key, 0)

    fig, ax = plt.subplots(figsize=(8, 10))

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(ltx_types)))
    ax.set_xticklabels(["LTX Flat", "LTX Channel-Mean"])
    ax.set_yticks(range(len(vlm_layers)))
    ax.set_yticklabels(vlm_layers)

    # Add values
    for i in range(len(vlm_layers)):
        for j in range(len(ltx_types)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                           ha="center", va="center", color="black" if heatmap_data[i, j] < 0.5 else "white",
                           fontsize=10)

    ax.set_xlabel("LTX Representation", fontsize=12)
    ax.set_ylabel("VLM Layer", fontsize=12)
    ax.set_title("CKA: VLM Layers vs LTX Representations", fontsize=14)

    plt.colorbar(im, ax=ax, label="CKA")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_vlm_self_similarity_plot(cka_matrix: np.ndarray, vlm_layers: list) -> bytes:
    """Create VLM layer-to-layer CKA heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cka_matrix, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(len(vlm_layers)))
    ax.set_xticklabels(vlm_layers, rotation=45, ha='right')
    ax.set_yticks(range(len(vlm_layers)))
    ax.set_yticklabels(vlm_layers)

    # Add values
    for i in range(len(vlm_layers)):
        for j in range(len(vlm_layers)):
            text = ax.text(j, i, f'{cka_matrix[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if cka_matrix[i, j] > 0.5 else "black",
                           fontsize=9)

    ax.set_title("VLM Layer-to-Layer CKA Similarity", fontsize=14)
    plt.colorbar(im, ax=ax, label="CKA")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_cka_curve_plot(cka_results: dict, vlm_layers: list) -> bytes:
    """Create CKA curve across VLM layers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract CKA values
    cka_flat = [cka_results.get(f"{layer}_vs_ltx_flat", 0) for layer in vlm_layers]
    cka_channel = [cka_results.get(f"{layer}_vs_ltx_channel", 0) for layer in vlm_layers]

    x = range(len(vlm_layers))

    ax.plot(x, cka_flat, 'b-o', linewidth=2, markersize=10, label='LTX Flat')
    ax.plot(x, cka_channel, 'r-s', linewidth=2, markersize=10, label='LTX Channel-Mean')

    ax.set_xticks(x)
    ax.set_xticklabels(vlm_layers, rotation=45, ha='right')
    ax.set_xlabel("VLM Layer", fontsize=12)
    ax.set_ylabel("CKA Similarity", fontsize=12)
    ax.set_title("CKA Across VLM Layers", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add threshold lines
    ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.7, label='Good threshold (0.4)')
    ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold (0.25)')

    # Mark best layers
    best_flat_idx = np.argmax(cka_flat)
    best_channel_idx = np.argmax(cka_channel)

    ax.annotate(f'Best (flat): {cka_flat[best_flat_idx]:.3f}',
                xy=(best_flat_idx, cka_flat[best_flat_idx]),
                xytext=(5, 10), textcoords='offset points',
                fontsize=10, color='blue')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
