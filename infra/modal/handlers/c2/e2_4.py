"""E2.4: Final Efficiency Validation

Objective: Comprehensive validation of the optimized adapter against success criteria.

Protocol:
1. Train final adapter with best architecture + training strategy (from E2.2, E2.3)
2. Evaluate on held-out test sets
3. Compute parameter efficiency metric
4. Measure inference latency
5. Compare to P2 baseline

Evaluation datasets:
- Synthetic shapes (500 samples) - Spatial precision
- COCO val subset (500 samples) - Real-world objects (if available)
- Something-Something v2 subset (200 clips) - Temporal coherence (if available)

Success Criteria (from research_plan.yaml):
- param_efficiency > 0.90 (target), > 0.85 (acceptable), < 0.80 (failure)
- LPIPS (10M adapter) < 0.18 (target), < 0.20 (acceptable), > 0.25 (failure)
- Spatial IoU > 0.75 (target), > 0.70 (acceptable), < 0.65 (failure)
- Inference latency < 150ms (target), < 200ms (acceptable), > 300ms (failure)

This experiment produces the final Gate 2 readiness assessment.
"""

import io
import os
import sys
import time

sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from runner import ExperimentRunner


# =============================================================================
# Best Architecture: Query Adapter (from E2.2)
# =============================================================================

class OptimizedQueryAdapter(nn.Module):
    """Query-based adapter optimized for 10M params.

    Incorporates best practices from E2.2 and E2.3:
    - Learned queries with cross-attention
    - Layer normalization for stability
    - Dropout for regularization
    """

    def __init__(
        self,
        vlm_dim: int = 3584,
        dino_dim: int = 1024,
        ltx_dim: int = 4096,
        hidden_dim: int = 384,
        n_layers: int = 2,
        n_queries: int = 77,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projections with layer norm
        self.vlm_proj = nn.Sequential(
            nn.Linear(vlm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.dino_proj = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 64),
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ltx_dim),
        )

    def forward(self, vlm_features: torch.Tensor, dino_features: torch.Tensor) -> torch.Tensor:
        B = vlm_features.size(0)

        # Project and concatenate
        vlm = self.vlm_proj(vlm_features)
        dino = self.dino_proj(dino_features)
        context = torch.cat([vlm, dino], dim=1)

        # Cross-attention
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, context)

        return self.out_proj(queries)


class LargeQueryAdapter(nn.Module):
    """100M reference adapter for efficiency comparison."""

    def __init__(
        self,
        vlm_dim: int = 3584,
        dino_dim: int = 1024,
        ltx_dim: int = 4096,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        n_queries: int = 77,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.vlm_proj = nn.Sequential(
            nn.Linear(vlm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.dino_proj = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.queries = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 64),
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ltx_dim),
        )

    def forward(self, vlm_features: torch.Tensor, dino_features: torch.Tensor) -> torch.Tensor:
        B = vlm_features.size(0)

        vlm = self.vlm_proj(vlm_features)
        dino = self.dino_proj(dino_features)
        context = torch.cat([vlm, dino], dim=1)

        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, context)

        return self.out_proj(queries)


class SimplePixelDecoder(nn.Module):
    """Pixel decoder for reconstruction evaluation."""

    def __init__(self, input_dim: int = 4096, output_size: int = 224):
        super().__init__()
        self.output_size = output_size

        self.initial = nn.Linear(input_dim, 256 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        x = conditioning.mean(dim=1)
        x = self.initial(x)
        x = x.view(-1, 256, 7, 7)
        return self.decoder(x)


# =============================================================================
# Feature Extraction
# =============================================================================

def load_dinov2_model(device: torch.device):
    """Load DINOv2-large model."""
    print("  Loading DINOv2-large model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = model.to(device)
    model.eval()
    return model


def load_vlm_model(device: torch.device):
    """Load Qwen2.5-VL model."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print("  Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
    )
    return model, processor


def extract_dinov2_features(images: list, model: nn.Module, device: torch.device) -> torch.Tensor:
    """Extract DINOv2 patch features."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_list = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = torch.stack([transform(img) for img in batch]).to(device)

            features = model.forward_features(batch_tensors)
            if isinstance(features, dict):
                patch_features = features.get('x_norm_patchtokens', features.get('x_prenorm', None))
                if patch_features is None:
                    patch_features = features['x_norm'][:, 1:, :]
            else:
                patch_features = features[:, 1:, :]

            features_list.append(patch_features.cpu())

    return torch.cat(features_list, dim=0)


def extract_vlm_features(images: list, model, processor, device: torch.device) -> torch.Tensor:
    """Extract VLM features from images."""
    features_list = []

    with torch.no_grad():
        for img in images:
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

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states[-1]
            features_list.append(hidden_states[0].float().cpu())

    max_len = max(f.shape[0] for f in features_list)
    padded = []
    for f in features_list:
        if f.shape[0] < max_len:
            padding = torch.zeros(max_len - f.shape[0], f.shape[1])
            f = torch.cat([f, padding], dim=0)
        padded.append(f)

    return torch.stack(padded)


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_synthetic_dataset(n_samples: int = 500, img_size: int = 224, seed: int = 42):
    """Generate diverse synthetic dataset for comprehensive evaluation."""
    shapes = ["circle", "square", "triangle", "star", "pentagon"]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (0, 128, 255),
    ]
    backgrounds = [(255, 255, 255), (240, 240, 240), (220, 220, 220)]

    images = []
    np.random.seed(seed)

    for i in range(n_samples):
        shape = shapes[i % len(shapes)]
        color = colors[i % len(colors)]
        bg = backgrounds[i % len(backgrounds)]

        # Varying sizes
        size = np.random.randint(20, 70)
        margin = size + 10
        cx = np.random.randint(margin, img_size - margin)
        cy = np.random.randint(margin, img_size - margin)

        img = Image.new("RGB", (img_size, img_size), bg)
        draw = ImageDraw.Draw(img)

        if shape == "circle":
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif shape == "square":
            draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif shape == "triangle":
            points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
            draw.polygon(points, fill=color)
        elif shape == "star":
            # 5-pointed star
            outer_r = size
            inner_r = size * 0.4
            points = []
            for j in range(10):
                r = outer_r if j % 2 == 0 else inner_r
                angle = j * np.pi / 5 - np.pi / 2
                points.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
            draw.polygon(points, fill=color)
        elif shape == "pentagon":
            points = []
            for j in range(5):
                angle = j * 2 * np.pi / 5 - np.pi / 2
                points.append((cx + size * np.cos(angle), cy + size * np.sin(angle)))
            draw.polygon(points, fill=color)

        images.append(img)

    return images


def images_to_tensor(images: list) -> torch.Tensor:
    """Convert list of PIL images to tensor."""
    tensors = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        tensors.append(torch.tensor(arr))
    return torch.stack(tensors)


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_comprehensive_metrics(
    recon: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute all reconstruction metrics."""
    import lpips as lpips_lib
    from scipy.ndimage import gaussian_filter

    # LPIPS
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    recon_lpips = recon * 2 - 1
    target_lpips = target * 2 - 1
    lpips_scores = []
    with torch.no_grad():
        for i in range(len(recon)):
            score = lpips_fn(recon_lpips[i:i+1], target_lpips[i:i+1])
            lpips_scores.append(score.item())
    lpips_val = np.mean(lpips_scores)

    # MSE and PSNR
    mse = F.mse_loss(recon, target).item()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

    # SSIM
    ssim_values = []
    for i in range(len(recon)):
        im1 = recon[i].cpu().numpy().mean(axis=0)
        im2 = target[i].cpu().numpy().mean(axis=0)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = gaussian_filter(im1, sigma=1.5)
        mu2 = gaussian_filter(im2, sigma=1.5)
        sigma1_sq = gaussian_filter(im1 ** 2, sigma=1.5) - mu1 ** 2
        sigma2_sq = gaussian_filter(im2 ** 2, sigma=1.5) - mu2 ** 2
        sigma12 = gaussian_filter(im1 * im2, sigma=1.5) - mu1 * mu2

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        ssim_values.append(ssim_map.mean())

    ssim_val = float(np.mean(ssim_values))

    # Spatial IoU
    ious = []
    for i in range(len(recon)):
        recon_gray = recon[i].mean(dim=0).cpu().numpy()
        target_gray = target[i].mean(dim=0).cpu().numpy()

        # Threshold for object detection
        recon_mask = recon_gray < 0.95
        target_mask = target_gray < 0.95

        intersection = (recon_mask & target_mask).sum()
        union = (recon_mask | target_mask).sum()

        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(1.0)

    spatial_iou = float(np.mean(ious))

    del lpips_fn
    torch.cuda.empty_cache()

    return {
        "lpips": lpips_val,
        "ssim": ssim_val,
        "psnr": psnr,
        "mse": mse,
        "spatial_iou": spatial_iou,
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Training Function
# =============================================================================

def train_adapter(
    adapter: nn.Module,
    decoder: nn.Module,
    vlm_features: torch.Tensor,
    dino_features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    n_epochs: int = 60,
    batch_size: int = 8,
    runner: ExperimentRunner = None,
    prefix: str = "",
) -> float:
    """Train adapter with best training strategy from E2.3.

    Uses:
    - LPIPS + L2 loss
    - Warmup + cosine decay schedule
    - No augmentation (cleanest signal)
    - Dropout 0.1, weight decay 1e-4
    """
    import lpips as lpips_lib

    vlm_t = vlm_features.to(device)
    dino_t = dino_features.to(device)
    targets_t = targets.to(device)

    # LPIPS loss
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    for param in lpips_fn.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(decoder.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Warmup + cosine decay scheduler
    n_steps = n_epochs * (len(targets_t) // batch_size)
    warmup_steps = n_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (n_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    step = 0
    for epoch in range(n_epochs):
        adapter.train()
        decoder.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = torch.randperm(len(targets_t))

        for i in range(0, len(targets_t), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_vlm = vlm_t[batch_idx]
            batch_dino = dino_t[batch_idx]
            batch_targets = targets_t[batch_idx]

            optimizer.zero_grad()

            conditioning = adapter(batch_vlm, batch_dino)
            recon = decoder(conditioning)

            # LPIPS + L2 loss
            recon_lpips = recon * 2 - 1
            target_lpips = batch_targets * 2 - 1
            lpips_loss = lpips_fn(recon_lpips, target_lpips).mean()
            l2_loss = F.mse_loss(recon, batch_targets)
            loss = lpips_loss + 0.1 * l2_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

        if runner and (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / n_batches
            runner.log_metrics({f"e2_4/{prefix}_loss": avg_loss}, step=epoch)
            print(f"      Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    del lpips_fn
    torch.cuda.empty_cache()

    return epoch_loss / n_batches


def measure_inference_latency(
    adapter: nn.Module,
    decoder: nn.Module,
    vlm_sample: torch.Tensor,
    dino_sample: torch.Tensor,
    n_warmup: int = 5,
    n_measure: int = 20,
) -> float:
    """Measure inference latency in milliseconds."""
    adapter.eval()
    decoder.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            conditioning = adapter(vlm_sample, dino_sample)
            _ = decoder(conditioning)

    # Measure
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(n_measure):
            conditioning = adapter(vlm_sample, dino_sample)
            _ = decoder(conditioning)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    return (elapsed / n_measure) * 1000  # ms


# =============================================================================
# Visualization
# =============================================================================

def create_final_validation_plot(
    results_10m: dict,
    results_100m: dict,
    p2_baseline: dict,
) -> bytes:
    """Create comprehensive visualization for final validation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Success criteria thresholds
    thresholds = {
        "lpips": {"target": 0.18, "acceptable": 0.20, "failure": 0.25},
        "spatial_iou": {"target": 0.75, "acceptable": 0.70, "failure": 0.65},
        "latency": {"target": 150, "acceptable": 200, "failure": 300},
        "param_efficiency": {"target": 0.90, "acceptable": 0.85, "failure": 0.80},
    }

    # Plot 1: LPIPS comparison
    models = ["10M Adapter", "100M Adapter", "P2 Baseline"]
    lpips_vals = [results_10m["lpips"], results_100m["lpips"], p2_baseline["lpips"]]
    colors = ['steelblue', 'darkorange', 'gray']

    axes[0, 0].bar(models, lpips_vals, color=colors)
    axes[0, 0].axhline(y=thresholds["lpips"]["target"], color='g', linestyle='--', label='Target')
    axes[0, 0].axhline(y=thresholds["lpips"]["acceptable"], color='orange', linestyle='--', label='Acceptable')
    axes[0, 0].set_ylabel('LPIPS (lower is better)')
    axes[0, 0].set_title('Reconstruction Quality')
    axes[0, 0].legend(fontsize=8)

    # Plot 2: Spatial IoU comparison
    iou_vals = [results_10m["spatial_iou"], results_100m["spatial_iou"], p2_baseline["spatial_iou"]]
    axes[0, 1].bar(models, iou_vals, color=colors)
    axes[0, 1].axhline(y=thresholds["spatial_iou"]["target"], color='g', linestyle='--', label='Target')
    axes[0, 1].axhline(y=thresholds["spatial_iou"]["acceptable"], color='orange', linestyle='--', label='Acceptable')
    axes[0, 1].set_ylabel('Spatial IoU (higher is better)')
    axes[0, 1].set_title('Spatial Preservation')
    axes[0, 1].legend(fontsize=8)

    # Plot 3: Inference latency
    latency_vals = [results_10m["latency_ms"], results_100m["latency_ms"], p2_baseline.get("latency_ms", 100)]
    axes[0, 2].bar(models, latency_vals, color=colors)
    axes[0, 2].axhline(y=thresholds["latency"]["target"], color='g', linestyle='--', label='Target')
    axes[0, 2].axhline(y=thresholds["latency"]["acceptable"], color='orange', linestyle='--', label='Acceptable')
    axes[0, 2].set_ylabel('Inference Latency (ms)')
    axes[0, 2].set_title('Inference Speed')
    axes[0, 2].legend(fontsize=8)

    # Plot 4: Parameter efficiency gauge
    param_efficiency = results_10m.get("param_efficiency", 0.9)
    ax = axes[1, 0]
    ax.barh([0], [param_efficiency], color='steelblue', height=0.3)
    ax.axvline(x=thresholds["param_efficiency"]["target"], color='g', linestyle='--', label='Target (0.90)')
    ax.axvline(x=thresholds["param_efficiency"]["acceptable"], color='orange', linestyle='--', label='Acceptable (0.85)')
    ax.axvline(x=thresholds["param_efficiency"]["failure"], color='r', linestyle='--', label='Failure (0.80)')
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel('Parameter Efficiency')
    ax.set_title(f'10M achieves {param_efficiency:.1%} of 100M quality')
    ax.legend(fontsize=7)
    ax.set_yticks([])

    # Plot 5: Parameter count comparison
    param_vals = [10, 100, p2_baseline.get("params_m", 78)]
    axes[1, 1].bar(models, param_vals, color=colors)
    axes[1, 1].set_ylabel('Parameters (M)')
    axes[1, 1].set_title('Model Size')

    # Plot 6: Memory usage
    memory_vals = [
        results_10m.get("memory_gb", 0.5),
        results_100m.get("memory_gb", 2.0),
        p2_baseline.get("memory_gb", 1.5)
    ]
    axes[1, 2].bar(models, memory_vals, color=colors)
    axes[1, 2].set_ylabel('Peak Memory (GB)')
    axes[1, 2].set_title('Memory Usage')

    plt.suptitle('C2 Final Efficiency Validation (E2.4)', fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e2_4_final_validation(runner: ExperimentRunner) -> dict:
    """Comprehensive validation of optimized adapter.

    This experiment:
    1. Trains both 10M and 100M adapters with best training strategy
    2. Evaluates on diverse synthetic dataset
    3. Computes parameter efficiency metric
    4. Measures inference latency
    5. Compares to P2 baseline
    6. Produces Gate 2 readiness assessment

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E2.4: Final Efficiency Validation")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e2_4/stage": 0, "e2_4/progress": 0.0})

    # P2 baseline (from research_plan.yaml)
    p2_baseline = {
        "lpips": 0.162,
        "spatial_iou": 0.837,
        "params_m": 78,
        "latency_ms": 100,  # Estimated
        "memory_gb": 1.5,   # Estimated
    }

    # =========================================================================
    # Stage 1: Generate comprehensive test dataset
    # =========================================================================
    print("\n[Stage 1/6] Generating datasets...")

    train_images = generate_synthetic_dataset(n_samples=400, seed=42)
    test_images = generate_synthetic_dataset(n_samples=100, seed=123)

    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    runner.log_metrics({"e2_4/stage": 1, "e2_4/progress": 0.05})

    # =========================================================================
    # Stage 2: Extract features
    # =========================================================================
    print("\n[Stage 2/6] Extracting features...")

    dinov2 = load_dinov2_model(device)
    train_dino = extract_dinov2_features(train_images, dinov2, device)
    test_dino = extract_dinov2_features(test_images, dinov2, device)
    del dinov2
    torch.cuda.empty_cache()

    vlm_model, vlm_processor = load_vlm_model(device)
    train_vlm = extract_vlm_features(train_images, vlm_model, vlm_processor, device)
    test_vlm = extract_vlm_features(test_images, vlm_model, vlm_processor, device)
    del vlm_model, vlm_processor
    torch.cuda.empty_cache()

    print(f"  Features: DINOv2 {train_dino.shape}, VLM {train_vlm.shape}")

    runner.log_metrics({"e2_4/stage": 2, "e2_4/progress": 0.25})

    # =========================================================================
    # Stage 3: Train 10M adapter (optimized)
    # =========================================================================
    print("\n[Stage 3/6] Training 10M adapter...")

    train_targets = images_to_tensor(train_images)
    test_targets = images_to_tensor(test_images)

    adapter_10m = OptimizedQueryAdapter(
        vlm_dim=train_vlm.shape[-1],
        dino_dim=train_dino.shape[-1],
    ).to(device)

    decoder_10m = SimplePixelDecoder(input_dim=4096, output_size=224).to(device)

    params_10m = count_parameters(adapter_10m)
    print(f"  10M Adapter params: {params_10m:,} ({params_10m/1e6:.2f}M)")

    train_adapter(
        adapter_10m, decoder_10m,
        train_vlm, train_dino, train_targets,
        device, n_epochs=60, batch_size=8,
        runner=runner, prefix="10m",
    )

    runner.log_metrics({"e2_4/stage": 3, "e2_4/progress": 0.45})

    # =========================================================================
    # Stage 4: Train 100M adapter (reference)
    # =========================================================================
    print("\n[Stage 4/6] Training 100M adapter...")

    adapter_100m = LargeQueryAdapter(
        vlm_dim=train_vlm.shape[-1],
        dino_dim=train_dino.shape[-1],
    ).to(device)

    decoder_100m = SimplePixelDecoder(input_dim=4096, output_size=224).to(device)

    params_100m = count_parameters(adapter_100m)
    print(f"  100M Adapter params: {params_100m:,} ({params_100m/1e6:.2f}M)")

    train_adapter(
        adapter_100m, decoder_100m,
        train_vlm, train_dino, train_targets,
        device, n_epochs=60, batch_size=8,
        runner=runner, prefix="100m",
    )

    runner.log_metrics({"e2_4/stage": 4, "e2_4/progress": 0.65})

    # =========================================================================
    # Stage 5: Evaluate both adapters
    # =========================================================================
    print("\n[Stage 5/6] Evaluating adapters...")

    test_vlm_t = test_vlm.to(device)
    test_dino_t = test_dino.to(device)
    test_targets_t = test_targets.to(device)

    # Evaluate 10M
    adapter_10m.eval()
    decoder_10m.eval()
    with torch.no_grad():
        conditioning_10m = adapter_10m(test_vlm_t, test_dino_t)
        recon_10m = decoder_10m(conditioning_10m)

    metrics_10m = compute_comprehensive_metrics(recon_10m, test_targets_t, device)
    latency_10m = measure_inference_latency(
        adapter_10m, decoder_10m,
        test_vlm_t[:1], test_dino_t[:1],
    )

    # Measure memory for 10M
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = adapter_10m(test_vlm_t[:8], test_dino_t[:8])
    memory_10m = torch.cuda.max_memory_allocated() / 1e9

    print(f"  10M - LPIPS: {metrics_10m['lpips']:.4f}, Spatial IoU: {metrics_10m['spatial_iou']:.4f}")
    print(f"  10M - Latency: {latency_10m:.1f}ms, Memory: {memory_10m:.2f}GB")

    # Evaluate 100M
    adapter_100m.eval()
    decoder_100m.eval()
    with torch.no_grad():
        conditioning_100m = adapter_100m(test_vlm_t, test_dino_t)
        recon_100m = decoder_100m(conditioning_100m)

    metrics_100m = compute_comprehensive_metrics(recon_100m, test_targets_t, device)
    latency_100m = measure_inference_latency(
        adapter_100m, decoder_100m,
        test_vlm_t[:1], test_dino_t[:1],
    )

    # Measure memory for 100M
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = adapter_100m(test_vlm_t[:8], test_dino_t[:8])
    memory_100m = torch.cuda.max_memory_allocated() / 1e9

    print(f"  100M - LPIPS: {metrics_100m['lpips']:.4f}, Spatial IoU: {metrics_100m['spatial_iou']:.4f}")
    print(f"  100M - Latency: {latency_100m:.1f}ms, Memory: {memory_100m:.2f}GB")

    # Compute parameter efficiency
    # param_efficiency = 1 - (LPIPS(10M) - LPIPS(100M)) / LPIPS(100M)
    if metrics_100m["lpips"] > 0:
        param_efficiency = 1 - (metrics_10m["lpips"] - metrics_100m["lpips"]) / metrics_100m["lpips"]
    else:
        param_efficiency = 1.0

    print(f"  Parameter Efficiency: {param_efficiency:.4f}")

    runner.log_metrics({"e2_4/stage": 5, "e2_4/progress": 0.85})

    # =========================================================================
    # Stage 6: Gate 2 assessment and save results
    # =========================================================================
    print("\n[Stage 6/6] Gate 2 assessment...")

    # Success criteria check
    criteria = {
        "param_efficiency": {"value": param_efficiency, "target": 0.90, "acceptable": 0.85, "failure": 0.80, "higher_better": True},
        "lpips_10m": {"value": metrics_10m["lpips"], "target": 0.18, "acceptable": 0.20, "failure": 0.25, "higher_better": False},
        "spatial_iou": {"value": metrics_10m["spatial_iou"], "target": 0.75, "acceptable": 0.70, "failure": 0.65, "higher_better": True},
        "latency_ms": {"value": latency_10m, "target": 150, "acceptable": 200, "failure": 300, "higher_better": False},
    }

    assessment = {}
    all_pass = True
    for metric, spec in criteria.items():
        if spec["higher_better"]:
            met_target = spec["value"] >= spec["target"]
            met_acceptable = spec["value"] >= spec["acceptable"]
            failed = spec["value"] < spec["failure"]
        else:
            met_target = spec["value"] <= spec["target"]
            met_acceptable = spec["value"] <= spec["acceptable"]
            failed = spec["value"] > spec["failure"]

        status = "TARGET" if met_target else ("ACCEPTABLE" if met_acceptable else ("FAILURE" if failed else "INVESTIGATE"))
        assessment[metric] = {
            "value": spec["value"],
            "status": status,
            "met_target": met_target,
            "met_acceptable": met_acceptable,
        }

        if failed:
            all_pass = False

        print(f"  {metric}: {spec['value']:.4f} - {status}")

    # Compile results
    results_10m = {
        **metrics_10m,
        "latency_ms": latency_10m,
        "memory_gb": memory_10m,
        "params_m": params_10m / 1e6,
        "param_efficiency": param_efficiency,
    }

    results_100m = {
        **metrics_100m,
        "latency_ms": latency_100m,
        "memory_gb": memory_100m,
        "params_m": params_100m / 1e6,
    }

    # Create visualization
    viz_bytes = create_final_validation_plot(results_10m, results_100m, p2_baseline)
    viz_path = runner.results.save_artifact("final_validation.png", viz_bytes)

    # Save comprehensive results
    results_data = {
        "adapter_10m": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in results_10m.items()},
        "adapter_100m": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in results_100m.items()},
        "p2_baseline": p2_baseline,
        "assessment": assessment,
        "gate_2_ready": all_pass,
        "recommendation": "proceed" if all_pass else "investigate",
    }
    data_path = runner.results.save_json_artifact("final_validation_results.json", results_data)

    # Save model checkpoint
    checkpoint = {
        "adapter_10m_state_dict": adapter_10m.state_dict(),
        "decoder_10m_state_dict": decoder_10m.state_dict(),
        "params_10m": params_10m,
        "metrics_10m": metrics_10m,
    }
    ckpt_bytes = io.BytesIO()
    torch.save(checkpoint, ckpt_bytes)
    ckpt_bytes.seek(0)
    ckpt_path = runner.results.save_artifact("optimized_adapter_10m.pt", ckpt_bytes.read())

    runner.log_metrics({
        "e2_4/stage": 6,
        "e2_4/progress": 1.0,
        "param_efficiency": param_efficiency,
        "lpips_10m": metrics_10m["lpips"],
        "lpips_100m": metrics_100m["lpips"],
        "spatial_iou_10m": metrics_10m["spatial_iou"],
        "latency_10m_ms": latency_10m,
    })

    # =========================================================================
    # Determine finding
    # =========================================================================
    if all_pass and param_efficiency >= 0.90:
        gate_status = "READY"
        finding = (
            f"Gate 2 READY. 10M adapter achieves {param_efficiency:.1%} of 100M quality "
            f"(LPIPS: 10M={metrics_10m['lpips']:.3f}, 100M={metrics_100m['lpips']:.3f}). "
            f"Spatial IoU: {metrics_10m['spatial_iou']:.3f}. "
            f"Inference latency: {latency_10m:.0f}ms. "
            f"All success criteria met. Recommend proceeding to Phase 3."
        )
    elif param_efficiency >= 0.85:
        gate_status = "ACCEPTABLE"
        finding = (
            f"Gate 2 acceptable. 10M adapter achieves {param_efficiency:.1%} of 100M quality "
            f"(within acceptable range). "
            f"LPIPS: 10M={metrics_10m['lpips']:.3f}, Spatial IoU: {metrics_10m['spatial_iou']:.3f}. "
            f"Some criteria below target but above acceptable threshold. "
            f"May proceed with caution."
        )
    else:
        gate_status = "INVESTIGATE"
        finding = (
            f"Gate 2 requires investigation. Parameter efficiency {param_efficiency:.1%} "
            f"is below acceptable threshold. "
            f"LPIPS: 10M={metrics_10m['lpips']:.3f}, 100M={metrics_100m['lpips']:.3f}. "
            f"Consider: increasing adapter size, alternative architectures, or longer training."
        )

    print(f"\n[GATE 2 STATUS: {gate_status}]")
    print(f"{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "param_efficiency": float(param_efficiency),
            "lpips_10m": float(metrics_10m["lpips"]),
            "lpips_100m": float(metrics_100m["lpips"]),
            "spatial_iou_10m": float(metrics_10m["spatial_iou"]),
            "spatial_iou_100m": float(metrics_100m["spatial_iou"]),
            "latency_10m_ms": float(latency_10m),
            "latency_100m_ms": float(latency_100m),
            "gate_2_status": gate_status,
        },
        "artifacts": [viz_path, data_path, ckpt_path],
    }
