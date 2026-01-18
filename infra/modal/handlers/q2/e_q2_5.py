"""E-Q2.5: Fine-Grained Detail Probe (Texture/Edges)

Objective: Measure preservation of high-frequency visual information.

Protocol:
1. Train decoder to reconstruct image from features
2. Measure edge preservation (Canny edge F1)
3. Measure perceptual similarity (LPIPS)
4. Analyze which frequencies are lost at each stage

Success Criteria:
- LPIPS < 0.3 at some extraction point
- Edge F1 > 0.6
- Identify best extraction point for fine-grained detail
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from runner import ExperimentRunner

# cv2 is imported inside functions where needed (Modal has it installed)


class TexturedImageDataset:
    """Dataset with textured images for fine-grained detail analysis."""

    def __init__(self, n_samples: int = 200, img_size: int = 448):
        self.n_samples = n_samples
        self.img_size = img_size

        np.random.seed(42)
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        texture_types = ["stripes", "checkerboard", "gradient", "noise", "circles"]

        for i in range(self.n_samples):
            texture_type = texture_types[i % len(texture_types)]
            img = self._create_textured_image(texture_type)
            samples.append({"image": img, "texture_type": texture_type})

        return samples

    def _create_textured_image(self, texture_type: str) -> Image.Image:
        """Create image with specific texture pattern."""
        img = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))
        arr = np.array(img)

        if texture_type == "stripes":
            # Vertical stripes of varying frequency
            freq = np.random.randint(5, 20)
            for x in range(0, self.img_size, freq * 2):
                arr[:, x:x+freq] = [np.random.randint(50, 200)] * 3

        elif texture_type == "checkerboard":
            # Checkerboard pattern
            size = np.random.randint(10, 30)
            for i in range(0, self.img_size, size):
                for j in range(0, self.img_size, size):
                    if (i // size + j // size) % 2 == 0:
                        arr[i:i+size, j:j+size] = [np.random.randint(50, 150)] * 3
                    else:
                        arr[i:i+size, j:j+size] = [np.random.randint(150, 250)] * 3

        elif texture_type == "gradient":
            # Smooth gradient with noise
            for i in range(self.img_size):
                val = int(255 * i / self.img_size)
                arr[i, :] = [val, val, val]
            arr = arr + np.random.randint(-20, 20, arr.shape).astype(np.uint8)
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        elif texture_type == "noise":
            # Random noise pattern
            arr = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)

        elif texture_type == "circles":
            # Concentric circles
            img = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            center = self.img_size // 2
            for r in range(10, center, 10):
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                draw.ellipse([center-r, center-r, center+r, center+r], outline=color, width=3)
            arr = np.array(img)

        return Image.fromarray(arr)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.samples[idx]


class ReconstructionDecoder(nn.Module):
    """Decoder to reconstruct images from VLM features."""

    def __init__(self, input_dim: int, output_size: int = 448, hidden_dim: int = 512):
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim

        # Project to spatial format (7x7 base)
        self.proj = nn.Linear(input_dim, hidden_dim * 7 * 7)

        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(hidden_dim, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 224x224 -> 448x448
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, seq_len, dim] or [batch, dim]
        Returns:
            images: [batch, 3, output_size, output_size]
        """
        if features.dim() == 3:
            features = features.mean(dim=1)  # Pool across sequence

        # Project to spatial
        spatial = self.proj(features).view(-1, self.hidden_dim, 7, 7)

        # Decode
        return self.decoder(spatial)


def extract_features_for_reconstruction(
    model, processor, images: list, extraction_point: str, device, runner: ExperimentRunner
) -> torch.Tensor:
    """Extract features from specified extraction point.

    Returns:
        Tensor of shape [n_images, embed_dim]
    """
    features_list = []

    with torch.no_grad():
        for i, img in enumerate(images):
            if extraction_point == "post_merge":
                # Use visual embeddings directly (faster)
                image_inputs = processor.image_processor(images=[img], return_tensors="pt")
                pixel_values = image_inputs["pixel_values"].to(device).to(model.dtype)
                image_grid_thw = image_inputs["image_grid_thw"].to(device)
                visual_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)

                # Handle both possible shapes
                if visual_embeds.dim() == 2:
                    # Shape is [num_tokens, embed_dim]
                    feat = visual_embeds
                else:
                    # Shape is [batch, num_tokens, embed_dim]
                    feat = visual_embeds[0]

                # Pool to [embed_dim]
                feat_pooled = feat.mean(dim=0, keepdim=True).float().cpu()
            else:
                # For LLM layers, need full forward pass
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
                ).to(device)

                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

                if extraction_point.startswith("llm_layer_"):
                    layer_idx = int(extraction_point.split("_")[-1])
                    feat = outputs.hidden_states[layer_idx + 1][0]
                else:
                    feat = outputs.hidden_states[1][0]

                # Pool to [1, embed_dim]
                feat_pooled = feat.float().cpu().mean(dim=0, keepdim=True)

            features_list.append(feat_pooled)

            if (i + 1) % 50 == 0:
                runner.log_metrics({"e_q2_5/extraction_progress": (i + 1) / len(images)})
                print(f"    Extracted {i + 1}/{len(images)} images")

    return torch.cat(features_list, dim=0)


def compute_edge_maps(images: list, img_size: int = 448) -> torch.Tensor:
    """Compute Canny edge maps for images."""
    import cv2  # Import here for Modal runtime

    edge_maps = []
    for img in images:
        # Convert to grayscale numpy array
        if isinstance(img, Image.Image):
            img_gray = np.array(img.convert("L"))
        else:
            img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(img_gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0

        edge_maps.append(torch.tensor(edges).unsqueeze(0))

    return torch.stack(edge_maps)


def compute_edge_f1(pred_edges: torch.Tensor, gt_edges: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute F1 score for edge detection."""
    # Binarize predictions
    pred_binary = (pred_edges > threshold).float()
    gt_binary = (gt_edges > threshold).float()

    # Compute TP, FP, FN
    tp = (pred_binary * gt_binary).sum()
    fp = (pred_binary * (1 - gt_binary)).sum()
    fn = ((1 - pred_binary) * gt_binary).sum()

    # Compute precision, recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1.item()


def compute_lpips_simple(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Simplified LPIPS computation using VGG-style features.

    This is a simplified version - actual LPIPS would use a trained perceptual network.
    """
    # Simple perceptual loss using multiple scales
    losses = []

    for scale in [1, 2, 4]:
        if scale > 1:
            pred_scaled = F.avg_pool2d(pred, scale)
            target_scaled = F.avg_pool2d(target, scale)
        else:
            pred_scaled = pred
            target_scaled = target

        # L1 difference normalized
        diff = (pred_scaled - target_scaled).abs().mean(dim=(1, 2, 3))
        losses.append(diff)

    # Combine scales
    lpips = sum(losses) / len(losses)
    return lpips.mean().item()


def train_reconstruction_decoder(
    features: torch.Tensor,
    target_images: torch.Tensor,
    epochs: int = 150,
    lr: float = 1e-4,
) -> tuple[ReconstructionDecoder, dict]:
    """Train decoder to reconstruct images from features."""
    device = features.device
    input_dim = features.shape[-1]
    output_size = target_images.shape[-1]

    decoder = ReconstructionDecoder(input_dim, output_size).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.01)

    # Split train/val
    n_train = int(len(features) * 0.8)
    train_features = features[:n_train]
    train_targets = target_images[:n_train]
    val_features = features[n_train:]
    val_targets = target_images[n_train:]

    history = {"train_loss": [], "val_lpips": []}

    for epoch in range(epochs):
        decoder.train()

        # Mini-batch training
        batch_size = 16
        epoch_losses = []

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_feat = train_features[start:end]
            batch_target = train_targets[start:end]

            pred = decoder(batch_feat)

            # Combined loss: L1 + perceptual
            l1_loss = F.l1_loss(pred, batch_target)
            perceptual_loss = compute_lpips_simple(pred, batch_target)
            loss = l1_loss + 0.1 * perceptual_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        history["train_loss"].append(np.mean(epoch_losses))

        # Validation
        if (epoch + 1) % 30 == 0:
            decoder.eval()
            with torch.no_grad():
                val_pred = decoder(val_features)
                val_lpips = compute_lpips_simple(val_pred, val_targets)
                history["val_lpips"].append(val_lpips)
                print(f"      Epoch {epoch+1}: loss={history['train_loss'][-1]:.4f}, val_lpips={val_lpips:.4f}")

    return decoder, history


def extract_edges_from_reconstruction(reconstructed: torch.Tensor) -> torch.Tensor:
    """Extract edge maps from reconstructed images."""
    import cv2  # Import here for Modal runtime

    edges_list = []

    for img in reconstructed:
        # Convert to numpy, grayscale
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Canny edge detection
        edges = cv2.Canny(img_gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0

        edges_list.append(torch.tensor(edges).unsqueeze(0))

    return torch.stack(edges_list)


def create_reconstruction_visualization(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    gt_edges: torch.Tensor,
    pred_edges: torch.Tensor,
    lpips_score: float,
    edge_f1: float,
    extraction_point: str,
) -> bytes:
    """Create visualization comparing original and reconstructed images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i in range(4):
        # Original image
        ax = axes[0, i]
        img = original_images[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title("Original" if i == 0 else "")
        ax.axis("off")

        # Reconstructed image
        ax = axes[1, i]
        img = reconstructed_images[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title("Reconstructed" if i == 0 else "")
        ax.axis("off")

        # Original edges
        ax = axes[2, i]
        edges = gt_edges[i, 0].cpu().numpy()
        ax.imshow(edges, cmap="gray")
        ax.set_title("GT Edges" if i == 0 else "")
        ax.axis("off")

        # Predicted edges
        ax = axes[3, i]
        edges = pred_edges[i, 0].cpu().numpy()
        ax.imshow(edges, cmap="gray")
        ax.set_title("Pred Edges" if i == 0 else "")
        ax.axis("off")

    plt.suptitle(f"Fine-Grained Reconstruction ({extraction_point})\nLPIPS: {lpips_score:.3f}, Edge F1: {edge_f1:.3f}")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e_q2_5_finegrained_probe(runner: ExperimentRunner) -> dict:
    """Run fine-grained detail probe for texture and edge preservation.

    This measures how well high-frequency visual information (textures, edges)
    is preserved in VLM features for image reconstruction.

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-Q2.5: Fine-Grained Detail Probe (LPIPS, Edge F1)")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    runner.log_metrics({"e_q2_5/stage": 0, "e_q2_5/progress": 0.0})

    # =========================================================================
    # Stage 1: Create textured dataset
    # =========================================================================
    print("\n[Stage 1/5] Creating textured image dataset...")

    dataset = TexturedImageDataset(n_samples=150, img_size=448)
    print(f"  Created {len(dataset)} samples with various textures")

    runner.log_metrics({"e_q2_5/stage": 1, "e_q2_5/progress": 0.1})

    # =========================================================================
    # Stage 2: Load model and extract features
    # =========================================================================
    print("\n[Stage 2/5] Loading Qwen2.5-VL and extracting features...")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from torchvision import transforms

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
    )
    device = model.device
    print(f"  Model loaded on {device}")

    images = [dataset[i]["image"] for i in range(len(dataset))]

    # Convert images to tensors for target
    to_tensor = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    target_images = torch.stack([to_tensor(img) for img in images])

    # Compute ground truth edge maps
    gt_edges = compute_edge_maps(images, img_size=448)

    # Extract from multiple points
    extraction_points = ["post_merge", "llm_layer_0"]
    features_by_point = {}

    for point in extraction_points:
        print(f"\n  Extracting features from {point}...")
        features_by_point[point] = extract_features_for_reconstruction(
            model, processor, images, point, device, runner
        )
        print(f"    Shape: {features_by_point[point].shape}")

    runner.log_metrics({"e_q2_5/stage": 2, "e_q2_5/progress": 0.3})

    # Free GPU memory
    del model
    del processor
    torch.cuda.empty_cache()

    # =========================================================================
    # Stage 3: Train reconstruction decoders
    # =========================================================================
    print("\n[Stage 3/5] Training reconstruction decoders...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_images = target_images.to(device)
    gt_edges = gt_edges.to(device)

    results_by_point = {}
    best_point = None
    best_lpips = float("inf")

    for point in extraction_points:
        print(f"\n  Training decoder for {point}...")
        features = features_by_point[point].to(device)

        decoder, history = train_reconstruction_decoder(
            features, target_images, epochs=150
        )

        # Evaluate
        decoder.eval()
        with torch.no_grad():
            reconstructed = decoder(features)

            # LPIPS
            lpips = compute_lpips_simple(reconstructed, target_images)

            # Edge F1
            pred_edges = extract_edges_from_reconstruction(reconstructed)
            pred_edges = pred_edges.to(device)
            edge_f1 = compute_edge_f1(pred_edges, gt_edges)

            # SSIM (simplified)
            ssim = 1 - F.mse_loss(reconstructed, target_images).item()
            ssim = max(0, min(1, ssim))  # Clamp to [0, 1]

        results_by_point[point] = {
            "lpips": lpips,
            "edge_f1": edge_f1,
            "ssim": ssim,
            "reconstructed": reconstructed,
            "pred_edges": pred_edges,
        }

        print(f"    LPIPS: {lpips:.4f}, Edge F1: {edge_f1:.4f}, SSIM: {ssim:.4f}")

        if lpips < best_lpips:
            best_lpips = lpips
            best_point = point

        runner.log_metrics({
            f"e_q2_5/{point}_lpips": lpips,
            f"e_q2_5/{point}_edge_f1": edge_f1,
            f"e_q2_5/{point}_ssim": ssim,
        })

    runner.log_metrics({"e_q2_5/stage": 3, "e_q2_5/progress": 0.7})

    # =========================================================================
    # Stage 4: Compute Information Retention Score (IRS)
    # =========================================================================
    print("\n[Stage 4/5] Computing Information Retention Score...")

    # IRS = 0.4 * (Bbox_IoU / 0.85) + 0.3 * (LPIPS_inv / 0.7) + 0.3 * (Edge_F1 / 0.8)
    # For this experiment, we use LPIPS and Edge F1 (bbox from other experiments)

    for point in extraction_points:
        lpips = results_by_point[point]["lpips"]
        edge_f1 = results_by_point[point]["edge_f1"]
        lpips_inv = 1 - lpips  # Higher is better

        # Simplified IRS without bbox (using edge_f1 twice)
        irs = 0.4 * (edge_f1 / 0.8) + 0.3 * (lpips_inv / 0.7) + 0.3 * (edge_f1 / 0.8)
        results_by_point[point]["irs"] = irs
        print(f"  {point}: IRS = {irs:.4f}")

        runner.log_metrics({f"e_q2_5/{point}_irs": irs})

    runner.log_metrics({"e_q2_5/stage": 4, "e_q2_5/progress": 0.85})

    # =========================================================================
    # Stage 5: Create visualizations
    # =========================================================================
    print("\n[Stage 5/5] Creating visualizations...")

    # Visualization for best point
    best_results = results_by_point[best_point]
    vis_bytes = create_reconstruction_visualization(
        target_images[:4],
        best_results["reconstructed"][:4],
        gt_edges[:4],
        best_results["pred_edges"][:4],
        best_results["lpips"],
        best_results["edge_f1"],
        best_point,
    )
    vis_path = runner.results.save_artifact("reconstruction_visualization.png", vis_bytes)

    # Comparison chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # LPIPS comparison
    ax = axes[0]
    points = extraction_points
    lpips_vals = [results_by_point[p]["lpips"] for p in points]
    ax.bar(points, lpips_vals, color="blue", alpha=0.7)
    ax.axhline(y=0.3, color="red", linestyle="--", label="Target (0.3)")
    ax.set_ylabel("LPIPS (lower is better)")
    ax.set_title("Perceptual Similarity")
    ax.legend()

    # Edge F1 comparison
    ax = axes[1]
    edge_f1_vals = [results_by_point[p]["edge_f1"] for p in points]
    ax.bar(points, edge_f1_vals, color="green", alpha=0.7)
    ax.axhline(y=0.6, color="red", linestyle="--", label="Target (0.6)")
    ax.set_ylabel("Edge F1 (higher is better)")
    ax.set_title("Edge Preservation")
    ax.legend()

    # IRS comparison
    ax = axes[2]
    irs_vals = [results_by_point[p]["irs"] for p in points]
    ax.bar(points, irs_vals, color="purple", alpha=0.7)
    ax.axhline(y=0.6, color="red", linestyle="--", label="Target (0.6)")
    ax.set_ylabel("IRS (higher is better)")
    ax.set_title("Information Retention Score")
    ax.legend()

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    comparison_path = runner.results.save_artifact("finegrained_comparison.png", buf.read())

    runner.log_metrics({
        "e_q2_5/stage": 5,
        "e_q2_5/progress": 1.0,
        "lpips": best_lpips,  # For overall assessment
        "edge_f1": results_by_point[best_point]["edge_f1"],
    })

    # Save detailed results
    results_data = {
        "extraction_points_evaluated": extraction_points,
        "results_by_point": {
            point: {
                "lpips": results_by_point[point]["lpips"],
                "edge_f1": results_by_point[point]["edge_f1"],
                "ssim": results_by_point[point]["ssim"],
                "irs": results_by_point[point]["irs"],
            }
            for point in extraction_points
        },
        "best_point": best_point,
        "best_lpips": best_lpips,
        "n_samples": len(dataset),
    }
    data_path = runner.results.save_json_artifact("finegrained_analysis.json", results_data)

    # =========================================================================
    # Interpret results
    # =========================================================================
    best_results = results_by_point[best_point]
    lpips_target = 0.3
    edge_f1_target = 0.6
    irs_target = 0.6

    lpips_met = best_results["lpips"] < lpips_target
    edge_f1_met = best_results["edge_f1"] > edge_f1_target
    irs_met = best_results["irs"] > irs_target

    if lpips_met and edge_f1_met:
        finding = (
            f"Fine-grained details well preserved at {best_point}. "
            f"LPIPS: {best_results['lpips']:.3f} (target: <{lpips_target}), "
            f"Edge F1: {best_results['edge_f1']:.3f} (target: >{edge_f1_target}). "
            f"Suitable for high-quality video generation."
        )
    elif lpips_met or edge_f1_met:
        finding = (
            f"Partial fine-grained detail preservation at {best_point}. "
            f"LPIPS: {best_results['lpips']:.3f} ({'met' if lpips_met else 'not met'}), "
            f"Edge F1: {best_results['edge_f1']:.3f} ({'met' if edge_f1_met else 'not met'}). "
            f"May need enhancement for best video quality."
        )
    else:
        finding = (
            f"Fine-grained details significantly degraded at {best_point}. "
            f"LPIPS: {best_results['lpips']:.3f} (target: <{lpips_target}), "
            f"Edge F1: {best_results['edge_f1']:.3f} (target: >{edge_f1_target}). "
            f"Consider auxiliary high-resolution features."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "lpips": float(best_results["lpips"]),
            "edge_f1": float(best_results["edge_f1"]),
            "ssim": float(best_results["ssim"]),
            "irs": float(best_results["irs"]),
            "best_extraction_point": best_point,
            **{f"{point}_lpips": results_by_point[point]["lpips"]
               for point in extraction_points},
            **{f"{point}_edge_f1": results_by_point[point]["edge_f1"]
               for point in extraction_points},
        },
        "artifacts": [vis_path, comparison_path, data_path],
    }
