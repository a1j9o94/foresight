"""E1.6: Ablation Studies

Objective: Understand which factors most affect reconstruction quality.

Ablations to run:
1. Layer depth: Extract from different layers of the VLM
2. Adapter capacity: Compare different adapter sizes
3. Training data size: Data efficiency analysis

This experiment helps identify the optimal configuration for production use.
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


def e1_6_ablations(runner: ExperimentRunner) -> dict:
    """Run ablation studies sub-experiment.

    This implementation tests:
    1. Layer depth ablation - which VLM layer has best reconstruction info
    2. Adapter capacity ablation - how many parameters needed
    3. Training data size ablation - data efficiency

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E1.6: Ablation Studies")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e1_6/stage": 0, "e1_6/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate dataset
    # =========================================================================
    print("\n[Stage 1/4] Generating dataset...")

    all_images = generate_diverse_images(n_images=200)
    train_images = all_images[:150]
    test_images = all_images[150:]

    print(f"  Generated {len(train_images)} training images")
    print(f"  Generated {len(test_images)} test images")

    runner.log_metrics({"e1_6/stage": 1, "e1_6/progress": 0.1})

    # =========================================================================
    # Stage 2: Layer depth ablation
    # =========================================================================
    print("\n[Stage 2/4] Running layer depth ablation...")

    layer_results = run_layer_ablation(train_images, test_images, device, runner)

    runner.log_metrics({"e1_6/stage": 2, "e1_6/progress": 0.5})

    # =========================================================================
    # Stage 3: Adapter capacity ablation
    # =========================================================================
    print("\n[Stage 3/4] Running adapter capacity ablation...")

    # Use last layer latents for this ablation
    train_latents, test_latents = extract_latents_from_layer(
        train_images, test_images, layer_idx=-1, runner=runner
    )

    # Convert images to targets
    train_targets = images_to_tensor(train_images).to(device)
    test_targets = images_to_tensor(test_images).to(device)

    capacity_results = run_capacity_ablation(
        train_latents, train_targets, test_latents, test_targets, device, runner
    )

    runner.log_metrics({"e1_6/stage": 3, "e1_6/progress": 0.8})

    # =========================================================================
    # Stage 4: Training data size ablation
    # =========================================================================
    print("\n[Stage 4/4] Running training data size ablation...")

    datasize_results = run_datasize_ablation(
        train_latents, train_targets, test_latents, test_targets, device, runner
    )

    runner.log_metrics({"e1_6/stage": 4, "e1_6/progress": 0.95})

    # =========================================================================
    # Create visualizations and save artifacts
    # =========================================================================
    print("\n[Final] Creating visualizations and saving artifacts...")

    # Create ablation plots
    plot_bytes = create_ablation_plots(layer_results, capacity_results, datasize_results)
    plot_path = runner.results.save_artifact("ablation_results.png", plot_bytes)

    # Save detailed results
    all_results = {
        "layer_ablation": layer_results,
        "capacity_ablation": capacity_results,
        "datasize_ablation": datasize_results,
    }
    results_path = runner.results.save_json_artifact("ablation_details.json", all_results)

    runner.log_metrics({"e1_6/progress": 1.0})

    # =========================================================================
    # Analyze findings
    # =========================================================================
    # Find best layer
    best_layer = min(layer_results, key=lambda x: x["lpips"])
    best_layer_idx = best_layer["layer_idx"]

    # Find sufficient capacity
    target_lpips = 0.35
    sufficient_capacity = None
    for cap in capacity_results:
        if cap["lpips"] < target_lpips:
            sufficient_capacity = cap
            break

    # Find minimum data needed
    min_data = None
    for ds in datasize_results:
        if ds["lpips"] < target_lpips:
            min_data = ds
            break

    findings = []
    findings.append(f"Best layer: {best_layer_idx} (LPIPS={best_layer['lpips']:.3f})")

    if sufficient_capacity:
        findings.append(
            f"Sufficient capacity: {sufficient_capacity['hidden_dim']} hidden dim "
            f"({sufficient_capacity['n_params']:,} params, LPIPS={sufficient_capacity['lpips']:.3f})"
        )
    else:
        findings.append("No tested capacity achieved target LPIPS < 0.35")

    if min_data:
        findings.append(
            f"Minimum training data: {min_data['n_train']} images "
            f"(LPIPS={min_data['lpips']:.3f})"
        )
    else:
        findings.append("All data sizes needed for acceptable quality")

    finding = " | ".join(findings)

    print(f"\n{finding}")
    print("=" * 60)

    # Summary metrics
    summary_metrics = {
        "best_layer_idx": best_layer_idx,
        "best_layer_lpips": float(best_layer["lpips"]),
    }

    if sufficient_capacity:
        summary_metrics["sufficient_capacity_params"] = sufficient_capacity["n_params"]
        summary_metrics["sufficient_capacity_lpips"] = float(sufficient_capacity["lpips"])

    if min_data:
        summary_metrics["min_data_n_train"] = min_data["n_train"]
        summary_metrics["min_data_lpips"] = float(min_data["lpips"])

    return {
        "finding": finding,
        "metrics": summary_metrics,
        "artifacts": [plot_path, results_path],
    }


def generate_diverse_images(n_images: int = 200) -> list:
    """Generate diverse synthetic images."""
    from PIL import Image, ImageDraw

    shapes = ["circle", "square", "triangle", "star"]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
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


def run_layer_ablation(train_images, test_images, device, runner) -> list:
    """Test different VLM layers for reconstruction quality."""
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

    # Get total number of layers by running a forward pass and counting hidden states
    # The hidden_states tuple length tells us the number of layers + 1 (for embeddings)
    # We'll test a subset: early, middle, late layers
    # Sample one image to get hidden state count
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_images[0]},
                {"type": "text", "text": "Describe."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[train_images[0]], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    n_layers = len(outputs.hidden_states)

    # Test early, middle, and late layers
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, -1]
    print(f"  Testing layers: {test_layers} (out of {n_layers} hidden states)")

    results = []

    for layer_idx in test_layers:
        print(f"\n  --- Layer {layer_idx} ---")

        # Extract latents from this layer
        train_latents = []
        test_latents = []

        # Extract train latents
        for img in train_images[:50]:  # Use subset for speed
            latent = extract_single_layer(model, processor, img, layer_idx)
            train_latents.append(latent)

        # Extract test latents
        for img in test_images:
            latent = extract_single_layer(model, processor, img, layer_idx)
            test_latents.append(latent)

        train_latents = np.concatenate(train_latents, axis=0)
        test_latents = np.concatenate(test_latents, axis=0)

        # Train quick probe
        train_targets = images_to_tensor(train_images[:50]).to(device)
        test_targets = images_to_tensor(test_images).to(device)

        lpips_val = train_quick_probe(
            train_latents, train_targets, test_latents, test_targets, device
        )

        results.append({
            "layer_idx": layer_idx,
            "lpips": float(lpips_val),
        })
        print(f"    LPIPS: {lpips_val:.4f}")

        runner.log_metrics({f"e1_6/layer_{layer_idx}_lpips": lpips_val})

    # Clean up
    del model
    del processor
    torch.cuda.empty_cache()

    return results


def extract_single_layer(model, processor, img, layer_idx: int) -> np.ndarray:
    """Extract latents from a specific layer."""
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

    hidden_states = outputs.hidden_states[layer_idx]
    latent = hidden_states[0].float().cpu().numpy()
    latent_pooled = latent.mean(axis=0, keepdims=True)

    return latent_pooled


def extract_latents_from_layer(train_images, test_images, layer_idx: int, runner):
    """Extract latents from a specific layer for all images."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"  Loading model for layer {layer_idx} extraction...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
    )

    train_latents = []
    test_latents = []

    print(f"  Extracting train latents...")
    for i, img in enumerate(train_images):
        latent = extract_single_layer(model, processor, img, layer_idx)
        train_latents.append(latent)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(train_images)}")

    print(f"  Extracting test latents...")
    for img in test_images:
        latent = extract_single_layer(model, processor, img, layer_idx)
        test_latents.append(latent)

    del model
    del processor
    torch.cuda.empty_cache()

    return np.concatenate(train_latents, axis=0), np.concatenate(test_latents, axis=0)


def run_capacity_ablation(train_latents, train_targets, test_latents, test_targets, device, runner) -> list:
    """Test different adapter capacities."""
    hidden_dims = [256, 512, 1024, 2048]
    results = []

    train_latents_t = torch.tensor(train_latents, dtype=torch.float32).to(device)
    test_latents_t = torch.tensor(test_latents, dtype=torch.float32).to(device)

    for hidden_dim in hidden_dims:
        print(f"\n  --- Hidden dim: {hidden_dim} ---")

        # Create probe with specified capacity
        latent_dim = train_latents.shape[-1]
        output_size = train_targets.shape[-2] * train_targets.shape[-1] * train_targets.shape[-3]

        probe = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid(),
        ).to(device)

        n_params = sum(p.numel() for p in probe.parameters())
        print(f"    Parameters: {n_params:,}")

        # Train
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for epoch in range(60):
            probe.train()
            for i in range(0, len(train_latents_t), 32):
                batch_latents = train_latents_t[i : i + 32]
                batch_targets = train_targets[i : i + 32].flatten(start_dim=1)

                optimizer.zero_grad()
                pred = probe(batch_latents)
                loss = F.mse_loss(pred, batch_targets)
                loss.backward()
                optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            pred = probe(test_latents_t)
            pred_img = pred.view(-1, 3, 224, 224)

        lpips_val = compute_lpips(pred_img, test_targets, device)

        results.append({
            "hidden_dim": hidden_dim,
            "n_params": n_params,
            "lpips": float(lpips_val),
        })
        print(f"    LPIPS: {lpips_val:.4f}")

        runner.log_metrics({f"e1_6/capacity_{hidden_dim}_lpips": lpips_val})

    return results


def run_datasize_ablation(train_latents, train_targets, test_latents, test_targets, device, runner) -> list:
    """Test different training data sizes."""
    data_sizes = [25, 50, 100, 150]
    results = []

    train_latents_t = torch.tensor(train_latents, dtype=torch.float32).to(device)
    test_latents_t = torch.tensor(test_latents, dtype=torch.float32).to(device)

    for n_train in data_sizes:
        print(f"\n  --- N train: {n_train} ---")

        subset_latents = train_latents_t[:n_train]
        subset_targets = train_targets[:n_train]

        # Create probe
        latent_dim = train_latents.shape[-1]
        output_size = train_targets.shape[-2] * train_targets.shape[-1] * train_targets.shape[-3]

        probe = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.Sigmoid(),
        ).to(device)

        # Train
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for epoch in range(60):
            probe.train()
            for i in range(0, len(subset_latents), 32):
                batch_latents = subset_latents[i : i + 32]
                batch_targets = subset_targets[i : i + 32].flatten(start_dim=1)

                optimizer.zero_grad()
                pred = probe(batch_latents)
                loss = F.mse_loss(pred, batch_targets)
                loss.backward()
                optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            pred = probe(test_latents_t)
            pred_img = pred.view(-1, 3, 224, 224)

        lpips_val = compute_lpips(pred_img, test_targets, device)

        results.append({
            "n_train": n_train,
            "lpips": float(lpips_val),
        })
        print(f"    LPIPS: {lpips_val:.4f}")

        runner.log_metrics({f"e1_6/datasize_{n_train}_lpips": lpips_val})

    return results


def train_quick_probe(train_latents, train_targets, test_latents, test_targets, device) -> float:
    """Train a quick probe and return LPIPS."""
    train_latents_t = torch.tensor(train_latents, dtype=torch.float32).to(device)
    test_latents_t = torch.tensor(test_latents, dtype=torch.float32).to(device)

    latent_dim = train_latents.shape[-1]
    output_size = train_targets.shape[-2] * train_targets.shape[-1] * train_targets.shape[-3]

    probe = nn.Sequential(
        nn.Linear(latent_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, output_size),
        nn.Sigmoid(),
    ).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for epoch in range(50):
        probe.train()
        for i in range(0, len(train_latents_t), 32):
            batch_latents = train_latents_t[i : i + 32]
            batch_targets = train_targets[i : i + 32].flatten(start_dim=1)

            optimizer.zero_grad()
            pred = probe(batch_latents)
            loss = F.mse_loss(pred, batch_targets)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(test_latents_t)
        pred_img = pred.view(-1, 3, 224, 224)

    return compute_lpips(pred_img, test_targets, device)


def compute_lpips(recon: torch.Tensor, target: torch.Tensor, device) -> float:
    """Compute LPIPS."""
    import lpips

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    recon_lpips = recon * 2 - 1
    target_lpips = target * 2 - 1

    scores = []
    with torch.no_grad():
        for i in range(len(recon)):
            score = lpips_fn(recon_lpips[i:i+1], target_lpips[i:i+1])
            scores.append(score.item())

    return np.mean(scores)


def images_to_tensor(images: list) -> torch.Tensor:
    """Convert PIL images to tensor."""
    tensors = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        tensors.append(torch.tensor(arr))
    return torch.stack(tensors)


def create_ablation_plots(layer_results, capacity_results, datasize_results) -> bytes:
    """Create visualization of ablation results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Layer ablation
    layers = [r["layer_idx"] for r in layer_results]
    lpips_vals = [r["lpips"] for r in layer_results]
    axes[0].bar(range(len(layers)), lpips_vals)
    axes[0].set_xticks(range(len(layers)))
    axes[0].set_xticklabels([str(l) for l in layers])
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("LPIPS (lower is better)")
    axes[0].set_title("Layer Depth Ablation")
    axes[0].axhline(y=0.35, color="r", linestyle="--", label="Target")
    axes[0].legend()

    # Capacity ablation
    hidden_dims = [r["hidden_dim"] for r in capacity_results]
    lpips_vals = [r["lpips"] for r in capacity_results]
    axes[1].plot(hidden_dims, lpips_vals, "o-")
    axes[1].set_xlabel("Hidden Dimension")
    axes[1].set_ylabel("LPIPS (lower is better)")
    axes[1].set_title("Adapter Capacity Ablation")
    axes[1].axhline(y=0.35, color="r", linestyle="--", label="Target")
    axes[1].legend()

    # Data size ablation
    n_trains = [r["n_train"] for r in datasize_results]
    lpips_vals = [r["lpips"] for r in datasize_results]
    axes[2].plot(n_trains, lpips_vals, "o-")
    axes[2].set_xlabel("Training Set Size")
    axes[2].set_ylabel("LPIPS (lower is better)")
    axes[2].set_title("Training Data Size Ablation")
    axes[2].axhline(y=0.35, color="r", linestyle="--", label="Target")
    axes[2].legend()

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()
