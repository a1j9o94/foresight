"""E-Q1.4: Linear Probing (Predict One Space from Other)

Objective: Test whether a simple linear transformation can map between VLM and LTX spaces.

Protocol:
1. Train linear probe: VLM latent -> LTX latent
2. Train reverse probe: LTX latent -> VLM latent
3. Evaluate with R^2, cosine similarity, and reconstruction quality
4. Compare against random projection baseline

Success Criteria:
- R^2 > 0.5: Linear alignment promising
- R^2 > 0.7: Linear alignment likely sufficient
- R^2 < 0.3: Need non-linear adapter

This is a critical experiment for determining adapter architecture complexity.
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from runner import ExperimentRunner


def eq1_4_linear_probe(runner: ExperimentRunner) -> dict:
    """Run linear probing experiment between VLM and LTX latent spaces.

    This implementation:
    1. Extracts paired latents from both models for the same images
    2. Trains bidirectional linear probes
    3. Evaluates alignment quality with multiple metrics
    4. Compares against random baseline

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-Q1.4: Linear Probing for Latent Alignment")
    print("=" * 60)

    # Set up environment
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"eq1_4/stage": 0, "eq1_4/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate and split data
    # =========================================================================
    print("\n[Stage 1/6] Preparing dataset...")

    images, labels, label_names = generate_test_images(n_per_category=60)
    n_total = len(images)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    # Shuffle indices
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_images = [images[i] for i in train_idx]
    test_images = [images[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    print(f"  Total: {n_total}, Train: {n_train}, Test: {n_test}")

    runner.log_metrics({"eq1_4/stage": 1, "eq1_4/progress": 0.1})

    # =========================================================================
    # Stage 2: Extract VLM latents
    # =========================================================================
    print("\n[Stage 2/6] Extracting VLM latents...")

    vlm_train = extract_vlm_latents(train_images, runner)
    vlm_test = extract_vlm_latents(test_images, runner)
    print(f"  VLM train: {vlm_train.shape}, test: {vlm_test.shape}")

    runner.log_metrics({
        "eq1_4/stage": 2,
        "eq1_4/progress": 0.3,
        "eq1_4/vlm_dim": vlm_train.shape[1],
    })

    # =========================================================================
    # Stage 3: Extract LTX latents
    # =========================================================================
    print("\n[Stage 3/6] Extracting LTX latents...")

    ltx_train = extract_ltx_latents(train_images, runner)
    ltx_test = extract_ltx_latents(test_images, runner)

    # Flatten for linear probing
    ltx_train_flat = ltx_train.reshape(ltx_train.shape[0], -1)
    ltx_test_flat = ltx_test.reshape(ltx_test.shape[0], -1)
    print(f"  LTX train: {ltx_train_flat.shape}, test: {ltx_test_flat.shape}")

    runner.log_metrics({
        "eq1_4/stage": 3,
        "eq1_4/progress": 0.5,
        "eq1_4/ltx_dim": ltx_train_flat.shape[1],
    })

    # =========================================================================
    # Stage 4: Train linear probes
    # =========================================================================
    print("\n[Stage 4/6] Training linear probes...")

    # VLM -> LTX probe
    print("  Training VLM -> LTX probe...")
    vlm_to_ltx_results = train_linear_probe(
        vlm_train, ltx_train_flat,
        vlm_test, ltx_test_flat,
        "vlm_to_ltx", runner
    )

    # LTX -> VLM probe
    print("  Training LTX -> VLM probe...")
    ltx_to_vlm_results = train_linear_probe(
        ltx_train_flat, vlm_train,
        ltx_test_flat, vlm_test,
        "ltx_to_vlm", runner
    )

    runner.log_metrics({
        "eq1_4/stage": 4,
        "eq1_4/progress": 0.7,
        "eq1_4/vlm_to_ltx_r2": vlm_to_ltx_results["r2"],
        "eq1_4/ltx_to_vlm_r2": ltx_to_vlm_results["r2"],
    })

    # =========================================================================
    # Stage 5: Compute baselines and additional metrics
    # =========================================================================
    print("\n[Stage 5/6] Computing baselines and additional metrics...")

    # Random projection baseline
    print("  Computing random projection baseline...")
    random_baseline = compute_random_baseline(
        vlm_test, ltx_test_flat,
        n_random=10
    )

    # Mean prediction baseline
    mean_baseline_vlm_ltx = compute_mean_baseline(ltx_train_flat, ltx_test_flat)
    mean_baseline_ltx_vlm = compute_mean_baseline(vlm_train, vlm_test)

    # Cosine similarity analysis
    print("  Computing cosine similarity metrics...")
    cosine_metrics = compute_cosine_metrics(
        vlm_to_ltx_results["predictions"],
        ltx_test_flat,
        ltx_to_vlm_results["predictions"],
        vlm_test
    )

    runner.log_metrics({
        "eq1_4/stage": 5,
        "eq1_4/progress": 0.85,
        "eq1_4/random_baseline_r2": random_baseline["mean_r2"],
        "eq1_4/cosine_sim_vlm_ltx": cosine_metrics["vlm_to_ltx_cosine"],
        "eq1_4/cosine_sim_ltx_vlm": cosine_metrics["ltx_to_vlm_cosine"],
    })

    # =========================================================================
    # Stage 6: Create visualizations
    # =========================================================================
    print("\n[Stage 6/6] Creating visualizations...")

    artifacts = []

    # Training curves plot
    curves_plot = create_training_curves_plot(
        vlm_to_ltx_results["history"],
        ltx_to_vlm_results["history"]
    )
    curves_path = runner.results.save_artifact("linear_probe_training.png", curves_plot)
    artifacts.append(curves_path)

    # Prediction scatter plots
    scatter_plot = create_prediction_scatter_plot(
        vlm_to_ltx_results["predictions"],
        ltx_test_flat,
        ltx_to_vlm_results["predictions"],
        vlm_test
    )
    scatter_path = runner.results.save_artifact("linear_probe_scatter.png", scatter_plot)
    artifacts.append(scatter_path)

    # Summary comparison plot
    summary_plot = create_summary_plot(
        vlm_to_ltx_results,
        ltx_to_vlm_results,
        random_baseline,
        mean_baseline_vlm_ltx,
        mean_baseline_ltx_vlm
    )
    summary_path = runner.results.save_artifact("linear_probe_summary.png", summary_plot)
    artifacts.append(summary_path)

    # Save detailed results
    data = {
        "vlm_to_ltx": {
            "r2": float(vlm_to_ltx_results["r2"]),
            "mse": float(vlm_to_ltx_results["mse"]),
            "final_train_loss": float(vlm_to_ltx_results["history"]["train_loss"][-1]),
            "final_val_loss": float(vlm_to_ltx_results["history"]["val_loss"][-1]),
        },
        "ltx_to_vlm": {
            "r2": float(ltx_to_vlm_results["r2"]),
            "mse": float(ltx_to_vlm_results["mse"]),
            "final_train_loss": float(ltx_to_vlm_results["history"]["train_loss"][-1]),
            "final_val_loss": float(ltx_to_vlm_results["history"]["val_loss"][-1]),
        },
        "random_baseline": {
            "mean_r2": float(random_baseline["mean_r2"]),
            "std_r2": float(random_baseline["std_r2"]),
        },
        "mean_baseline": {
            "vlm_to_ltx_r2": float(mean_baseline_vlm_ltx),
            "ltx_to_vlm_r2": float(mean_baseline_ltx_vlm),
        },
        "cosine_metrics": {k: float(v) for k, v in cosine_metrics.items()},
        "n_train": n_train,
        "n_test": n_test,
        "vlm_dim": int(vlm_train.shape[1]),
        "ltx_dim": int(ltx_train_flat.shape[1]),
    }
    data_path = runner.results.save_json_artifact("linear_probe_results.json", data)
    artifacts.append(data_path)

    runner.log_metrics({"eq1_4/stage": 6, "eq1_4/progress": 1.0})

    # =========================================================================
    # Form conclusions
    # =========================================================================
    avg_r2 = (vlm_to_ltx_results["r2"] + ltx_to_vlm_results["r2"]) / 2
    best_r2 = max(vlm_to_ltx_results["r2"], ltx_to_vlm_results["r2"])
    best_direction = "VLM->LTX" if vlm_to_ltx_results["r2"] > ltx_to_vlm_results["r2"] else "LTX->VLM"

    improvement_over_random = avg_r2 - random_baseline["mean_r2"]

    if avg_r2 > 0.7:
        alignment_quality = "excellent"
        finding = (
            f"Strong linear alignment found! Average R^2={avg_r2:.3f} (best: {best_direction} R^2={best_r2:.3f}). "
            f"Linear adapter likely sufficient for bridging VLM and LTX spaces. "
            f"Improvement over random: {improvement_over_random:.3f}."
        )
    elif avg_r2 > 0.5:
        alignment_quality = "good"
        finding = (
            f"Good linear alignment. Average R^2={avg_r2:.3f} (best: {best_direction} R^2={best_r2:.3f}). "
            f"Linear adapter should work well, but MLP may improve further. "
            f"Improvement over random: {improvement_over_random:.3f}."
        )
    elif avg_r2 > 0.3:
        alignment_quality = "moderate"
        finding = (
            f"Moderate linear alignment. Average R^2={avg_r2:.3f} (best: {best_direction} R^2={best_r2:.3f}). "
            f"Small MLP adapter recommended over pure linear projection. "
            f"Improvement over random: {improvement_over_random:.3f}."
        )
    else:
        alignment_quality = "weak"
        finding = (
            f"Weak linear alignment. Average R^2={avg_r2:.3f} (best: {best_direction} R^2={best_r2:.3f}). "
            f"Significant non-linear adapter or alternative approach may be needed. "
            f"Improvement over random: {improvement_over_random:.3f}."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "vlm_to_ltx_r2": float(vlm_to_ltx_results["r2"]),
            "ltx_to_vlm_r2": float(ltx_to_vlm_results["r2"]),
            "average_r2": float(avg_r2),
            "best_r2": float(best_r2),
            "best_direction": best_direction,
            "random_baseline_r2": float(random_baseline["mean_r2"]),
            "improvement_over_random": float(improvement_over_random),
            "cosine_similarity_vlm_ltx": float(cosine_metrics["vlm_to_ltx_cosine"]),
            "cosine_similarity_ltx_vlm": float(cosine_metrics["ltx_to_vlm_cosine"]),
            "alignment_quality": alignment_quality,
            "n_train": n_train,
            "n_test": n_test,
        },
        "artifacts": artifacts,
    }


def generate_test_images(n_per_category: int = 60) -> tuple[list, np.ndarray, list]:
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

            if (idx + 1) % 100 == 0:
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

            if (idx + 1) % 100 == 0:
                print(f"    LTX: {idx + 1}/{len(images)}")

    del vae
    torch.cuda.empty_cache()

    return np.stack(latents_list, axis=0)


class LinearProbe(nn.Module):
    """Simple linear projection for probing alignment."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def train_linear_probe(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    name: str,
    runner: ExperimentRunner,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict:
    """Train a linear probe to predict one space from another.

    Args:
        X_train, Y_train: Training data (input, target)
        X_test, Y_test: Test data
        name: Probe name for logging
        runner: ExperimentRunner
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization

    Returns:
        Dict with results and history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32).to(device)

    # Create model
    probe = LinearProbe(X_train.shape[1], Y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        # Training
        probe.train()
        optimizer.zero_grad()
        pred = probe(X_train_t)
        train_loss = F.mse_loss(pred, Y_train_t)
        train_loss.backward()
        optimizer.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_test_t)
            val_loss = F.mse_loss(val_pred, Y_test_t)

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss.item())

        scheduler.step(val_loss)

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = probe.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train_loss={train_loss.item():.6f}, val_loss={val_loss.item():.6f}")

    # Load best model
    if best_state is not None:
        probe.load_state_dict(best_state)

    # Final evaluation
    probe.eval()
    with torch.no_grad():
        predictions = probe(X_test_t).cpu().numpy()

    # Compute metrics
    mse = np.mean((predictions - Y_test) ** 2)
    ss_res = np.sum((Y_test - predictions) ** 2)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    return {
        "r2": float(r2),
        "mse": float(mse),
        "predictions": predictions,
        "history": history,
    }


def compute_random_baseline(X_test: np.ndarray, Y_test: np.ndarray, n_random: int = 10) -> dict:
    """Compute random projection baseline R^2."""
    r2_scores = []
    input_dim = X_test.shape[1]
    output_dim = Y_test.shape[1]

    for _ in range(n_random):
        # Random projection matrix - normalize columns for stability
        random_proj = np.random.randn(input_dim, output_dim)
        # Normalize columns instead of QR (handles any dimension ratio)
        random_proj = random_proj / (np.linalg.norm(random_proj, axis=0, keepdims=True) + 1e-10)

        predictions = X_test @ random_proj

        ss_res = np.sum((Y_test - predictions) ** 2)
        ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        r2_scores.append(r2)

    return {
        "mean_r2": float(np.mean(r2_scores)),
        "std_r2": float(np.std(r2_scores)),
        "scores": r2_scores,
    }


def compute_mean_baseline(Y_train: np.ndarray, Y_test: np.ndarray) -> float:
    """Compute mean prediction baseline R^2."""
    mean_pred = Y_train.mean(axis=0, keepdims=True)
    predictions = np.tile(mean_pred, (len(Y_test), 1))

    ss_res = np.sum((Y_test - predictions) ** 2)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    return float(r2)


def compute_cosine_metrics(
    vlm_ltx_pred: np.ndarray,
    ltx_target: np.ndarray,
    ltx_vlm_pred: np.ndarray,
    vlm_target: np.ndarray,
) -> dict:
    """Compute cosine similarity metrics."""
    from sklearn.metrics.pairwise import cosine_similarity

    # VLM -> LTX: average cosine similarity between predictions and targets
    vlm_ltx_cos = np.mean([
        cosine_similarity(vlm_ltx_pred[i:i+1], ltx_target[i:i+1])[0, 0]
        for i in range(len(vlm_ltx_pred))
    ])

    # LTX -> VLM
    ltx_vlm_cos = np.mean([
        cosine_similarity(ltx_vlm_pred[i:i+1], vlm_target[i:i+1])[0, 0]
        for i in range(len(ltx_vlm_pred))
    ])

    return {
        "vlm_to_ltx_cosine": vlm_ltx_cos,
        "ltx_to_vlm_cosine": ltx_vlm_cos,
        "average_cosine": (vlm_ltx_cos + ltx_vlm_cos) / 2,
    }


def create_training_curves_plot(history1: dict, history2: dict) -> bytes:
    """Create training curves visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # VLM -> LTX
    axes[0].plot(history1["train_loss"], 'b-', label='Train', alpha=0.8)
    axes[0].plot(history1["val_loss"], 'r-', label='Validation', alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("VLM -> LTX Linear Probe")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # LTX -> VLM
    axes[1].plot(history2["train_loss"], 'b-', label='Train', alpha=0.8)
    axes[1].plot(history2["val_loss"], 'r-', label='Validation', alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title("LTX -> VLM Linear Probe")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.suptitle("Linear Probe Training Curves", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_prediction_scatter_plot(
    vlm_ltx_pred: np.ndarray,
    ltx_target: np.ndarray,
    ltx_vlm_pred: np.ndarray,
    vlm_target: np.ndarray,
) -> bytes:
    """Create prediction vs target scatter plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sample dimensions for visualization
    n_dims_to_show = 3
    dims = np.random.choice(min(vlm_ltx_pred.shape[1], ltx_target.shape[1]), n_dims_to_show, replace=False)

    # VLM -> LTX
    for i, dim in enumerate(dims):
        axes[0].scatter(
            ltx_target[:, dim],
            vlm_ltx_pred[:, dim],
            alpha=0.5,
            s=20,
            label=f'Dim {dim}'
        )
    axes[0].plot(
        [ltx_target[:, dims[0]].min(), ltx_target[:, dims[0]].max()],
        [ltx_target[:, dims[0]].min(), ltx_target[:, dims[0]].max()],
        'k--', alpha=0.5
    )
    axes[0].set_xlabel("Target (LTX)")
    axes[0].set_ylabel("Prediction")
    axes[0].set_title("VLM -> LTX Predictions")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # LTX -> VLM
    dims = np.random.choice(min(ltx_vlm_pred.shape[1], vlm_target.shape[1]), n_dims_to_show, replace=False)
    for i, dim in enumerate(dims):
        axes[1].scatter(
            vlm_target[:, dim],
            ltx_vlm_pred[:, dim],
            alpha=0.5,
            s=20,
            label=f'Dim {dim}'
        )
    axes[1].plot(
        [vlm_target[:, dims[0]].min(), vlm_target[:, dims[0]].max()],
        [vlm_target[:, dims[0]].min(), vlm_target[:, dims[0]].max()],
        'k--', alpha=0.5
    )
    axes[1].set_xlabel("Target (VLM)")
    axes[1].set_ylabel("Prediction")
    axes[1].set_title("LTX -> VLM Predictions")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Linear Probe Predictions vs Targets (Sample Dimensions)", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def create_summary_plot(
    vlm_ltx: dict,
    ltx_vlm: dict,
    random_baseline: dict,
    mean_baseline_vlm_ltx: float,
    mean_baseline_ltx_vlm: float,
) -> bytes:
    """Create summary comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['VLM->LTX\n(Linear)', 'LTX->VLM\n(Linear)', 'Random\nBaseline', 'Mean\nBaseline']
    r2_scores = [
        vlm_ltx["r2"],
        ltx_vlm["r2"],
        random_baseline["mean_r2"],
        (mean_baseline_vlm_ltx + mean_baseline_ltx_vlm) / 2,
    ]
    errors = [0, 0, random_baseline["std_r2"], 0]

    colors = ['steelblue', 'coral', 'gray', 'lightgray']

    bars = ax.bar(methods, r2_scores, yerr=errors, capsize=5, color=colors, alpha=0.8)

    # Add value labels
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.annotate(f'{score:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    # Threshold lines
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Good threshold (0.5)')
    ax.axhline(y=0.7, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent threshold (0.7)')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Weak threshold (0.3)')

    ax.set_ylabel("R^2 Score", fontsize=12)
    ax.set_title("Linear Probe Alignment Quality", fontsize=14)
    ax.legend(loc='upper right')
    ax.set_ylim(min(0, min(r2_scores) - 0.1), max(1, max(r2_scores) + 0.1))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
