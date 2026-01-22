"""E2.3: Training Strategy Optimization

Objective: Find optimal training strategy for the chosen adapter architecture.

Strategies to test:
A. Loss Functions: LPIPS only, LPIPS + L2, LPIPS + SSIM, Multi-scale LPIPS
B. Learning Rate Schedules: Constant, Cosine decay, Warmup + decay, Cyclic
C. Data Augmentation: None, Color jitter, Random crop, MixUp
D. Regularization: Dropout 0.1/0.2, Weight decay 1e-4/1e-3

Protocol:
1. Grid search over loss functions
2. Grid search over learning rate schedules
3. Ablate augmentation strategies
4. Combine best settings

Success Metrics:
- Final LPIPS (minimum achievable)
- Training stability (low variance)
- Convergence speed (epochs to plateau)

This experiment produces the optimal training recipe for C2.
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
# Best Architecture from E2.2 (Query-based, configurable)
# =============================================================================

class QueryAdapter(nn.Module):
    """Query-based adapter (best from E2.2 or configurable)."""

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

        self.vlm_proj = nn.Linear(vlm_dim, hidden_dim)
        self.dino_proj = nn.Linear(dino_dim, hidden_dim)

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

        self.out_proj = nn.Linear(hidden_dim, ltx_dim)

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
    """Simple pixel decoder for reconstruction."""

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
# Loss Functions
# =============================================================================

class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss."""

    def __init__(self, device):
        super().__init__()
        import lpips
        self.lpips = lpips.LPIPS(net="alex").to(device)
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert from [0,1] to [-1,1]
        pred = pred * 2 - 1
        target = target * 2 - 1
        return self.lpips(pred, target).mean()


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss."""

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.window = self._create_window(window_size, channels)

    def _create_window(self, window_size: int, channels: int) -> torch.Tensor:
        sigma = 1.5
        gauss = torch.tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self.window.to(pred.device).to(pred.dtype)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=self.channels)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=self.channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return 1 - ssim_map.mean()


class MultiScaleLPIPSLoss(nn.Module):
    """Multi-scale LPIPS loss."""

    def __init__(self, device, scales: list = [1.0, 0.5, 0.25]):
        super().__init__()
        import lpips
        self.lpips = lpips.LPIPS(net="alex").to(device)
        for param in self.lpips.parameters():
            param.requires_grad = False
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred * 2 - 1
        target = target * 2 - 1

        total_loss = 0
        for scale in self.scales:
            if scale < 1.0:
                size = int(pred.shape[-1] * scale)
                pred_scaled = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
            else:
                pred_scaled = pred
                target_scaled = target

            total_loss += self.lpips(pred_scaled, target_scaled).mean()

        return total_loss / len(self.scales)


def get_loss_function(loss_name: str, device: torch.device):
    """Get loss function by name."""
    if loss_name == "lpips":
        lpips_loss = LPIPSLoss(device)
        def loss_fn(pred, target):
            return lpips_loss(pred, target)
        return loss_fn

    elif loss_name == "lpips_l2":
        lpips_loss = LPIPSLoss(device)
        def loss_fn(pred, target):
            return lpips_loss(pred, target) + 0.1 * F.mse_loss(pred, target)
        return loss_fn

    elif loss_name == "lpips_ssim":
        lpips_loss = LPIPSLoss(device)
        ssim_loss = SSIMLoss()
        def loss_fn(pred, target):
            return lpips_loss(pred, target) + 0.5 * ssim_loss(pred, target)
        return loss_fn

    elif loss_name == "multiscale_lpips":
        ms_lpips = MultiScaleLPIPSLoss(device)
        def loss_fn(pred, target):
            return ms_lpips(pred, target)
        return loss_fn

    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

def get_scheduler(scheduler_name: str, optimizer, n_epochs: int, n_steps_per_epoch: int):
    """Get learning rate scheduler by name."""
    total_steps = n_epochs * n_steps_per_epoch

    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps)

    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    elif scheduler_name == "warmup_cosine":
        warmup_steps = total_steps // 10

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_name == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-5,
            max_lr=1e-4,
            step_size_up=n_steps_per_epoch * 5,
            mode='triangular2',
            cycle_momentum=False,
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


# =============================================================================
# Data Augmentation
# =============================================================================

def apply_augmentation(images: torch.Tensor, aug_name: str) -> torch.Tensor:
    """Apply data augmentation to a batch of images."""
    from torchvision import transforms as T

    if aug_name == "none":
        return images

    elif aug_name == "color_jitter":
        # Apply per-sample color jitter
        augmented = []
        for img in images:
            brightness = 0.2 * (torch.rand(1).item() - 0.5)
            contrast = 1 + 0.2 * (torch.rand(1).item() - 0.5)
            aug_img = torch.clamp((img + brightness) * contrast, 0, 1)
            augmented.append(aug_img)
        return torch.stack(augmented)

    elif aug_name == "random_crop":
        # Random crop and resize
        B, C, H, W = images.shape
        scale = 0.8 + 0.2 * torch.rand(B)
        augmented = []
        for i, img in enumerate(images):
            s = scale[i].item()
            new_h, new_w = int(H * s), int(W * s)
            top = torch.randint(0, H - new_h + 1, (1,)).item()
            left = torch.randint(0, W - new_w + 1, (1,)).item()
            cropped = img[:, top:top+new_h, left:left+new_w]
            resized = F.interpolate(cropped.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
            augmented.append(resized.squeeze(0))
        return torch.stack(augmented)

    elif aug_name == "mixup":
        # MixUp augmentation
        B = images.shape[0]
        lam = torch.distributions.Beta(0.4, 0.4).sample((B,)).to(images.device)
        lam = lam.view(-1, 1, 1, 1)
        indices = torch.randperm(B)
        mixed = lam * images + (1 - lam) * images[indices]
        return mixed

    else:
        raise ValueError(f"Unknown augmentation: {aug_name}")


# =============================================================================
# Feature Extraction (reused)
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


def generate_training_images(n_images: int = 200, img_size: int = 224):
    """Generate synthetic training images."""
    shapes = ["circle", "square", "triangle"]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    images = []
    np.random.seed(42)

    for i in range(n_images):
        shape = shapes[i % len(shapes)]
        color = colors[i % len(colors)]

        size = np.random.randint(25, 60)
        margin = size + 10
        cx = np.random.randint(margin, img_size - margin)
        cy = np.random.randint(margin, img_size - margin)

        img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        if shape == "circle":
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif shape == "square":
            draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
        elif shape == "triangle":
            points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
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


def compute_lpips_metric(recon: torch.Tensor, target: torch.Tensor, device: torch.device) -> float:
    """Compute LPIPS metric for evaluation."""
    import lpips as lpips_lib

    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    recon_lpips = recon * 2 - 1
    target_lpips = target * 2 - 1
    lpips_scores = []
    with torch.no_grad():
        for i in range(len(recon)):
            score = lpips_fn(recon_lpips[i:i+1], target_lpips[i:i+1])
            lpips_scores.append(score.item())

    del lpips_fn
    torch.cuda.empty_cache()

    return float(np.mean(lpips_scores))


# =============================================================================
# Training Function
# =============================================================================

def train_with_strategy(
    config: dict,
    vlm_features: torch.Tensor,
    dino_features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    n_epochs: int = 30,
    batch_size: int = 8,
) -> dict:
    """Train adapter with specific strategy configuration.

    Config contains:
    - loss: Loss function name
    - scheduler: LR scheduler name
    - augmentation: Data augmentation name
    - dropout: Dropout rate
    - weight_decay: Weight decay value
    """

    strategy_name = f"{config['loss']}_{config['scheduler']}_{config['augmentation']}"
    print(f"\n  Training with strategy: {strategy_name}")

    # Create models
    adapter = QueryAdapter(
        vlm_dim=vlm_features.shape[-1],
        dino_dim=dino_features.shape[-1],
        dropout=config.get("dropout", 0.1),
    ).to(device)

    decoder = SimplePixelDecoder(input_dim=4096, output_size=224).to(device)

    # Move data to device
    vlm_t = vlm_features.to(device)
    dino_t = dino_features.to(device)
    targets_t = targets.to(device)

    # Setup loss
    loss_fn = get_loss_function(config["loss"], device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(decoder.parameters()),
        lr=1e-4,
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Setup scheduler
    n_steps_per_epoch = len(targets_t) // batch_size
    scheduler = get_scheduler(config["scheduler"], optimizer, n_epochs, n_steps_per_epoch)

    # Training loop
    loss_history = []
    start_time = time.time()

    for epoch in range(n_epochs):
        adapter.train()
        decoder.train()
        epoch_losses = []

        indices = torch.randperm(len(targets_t))

        for i in range(0, len(targets_t), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_vlm = vlm_t[batch_idx]
            batch_dino = dino_t[batch_idx]
            batch_targets = targets_t[batch_idx]

            # Apply augmentation to targets
            batch_targets_aug = apply_augmentation(batch_targets, config["augmentation"])

            optimizer.zero_grad()

            conditioning = adapter(batch_vlm, batch_dino)
            recon = decoder(conditioning)

            loss = loss_fn(recon, batch_targets_aug)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        loss_history.append(epoch_loss)

    training_time = time.time() - start_time

    # Evaluate
    adapter.eval()
    decoder.eval()

    with torch.no_grad():
        conditioning = adapter(vlm_t, dino_t)
        recon = decoder(conditioning)

    lpips = compute_lpips_metric(recon, targets_t, device)

    # Compute metrics
    loss_variance = np.var(loss_history[-10:]) if len(loss_history) >= 10 else np.var(loss_history)

    # Find convergence epoch (first epoch where loss < 1.1 * final loss)
    final_loss = loss_history[-1]
    convergence_epoch = n_epochs
    for i, loss_val in enumerate(loss_history):
        if loss_val < 1.1 * final_loss:
            convergence_epoch = i
            break

    results = {
        "lpips": lpips,
        "final_loss": float(loss_history[-1]),
        "loss_variance": float(loss_variance),
        "convergence_epoch": convergence_epoch,
        "training_time": training_time,
    }

    print(f"    LPIPS: {lpips:.4f}, Convergence: epoch {convergence_epoch}")

    # Cleanup
    del adapter, decoder
    torch.cuda.empty_cache()

    return results


def create_strategy_comparison_plot(results: dict) -> bytes:
    """Create visualization of strategy comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Group results by category
    loss_results = {k: v for k, v in results.items() if "loss" in k.lower() or any(l in k for l in ["lpips", "l2", "ssim", "multiscale"])}
    scheduler_results = {k: v for k, v in results.items() if any(s in k for s in ["constant", "cosine", "warmup", "cyclic"])}

    # Plot 1: LPIPS by strategy
    strategies = list(results.keys())[:8]  # Limit for readability
    lpips_vals = [results[s]["lpips"] for s in strategies]
    colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))

    axes[0, 0].barh(range(len(strategies)), lpips_vals, color=colors)
    axes[0, 0].set_yticks(range(len(strategies)))
    axes[0, 0].set_yticklabels(strategies, fontsize=8)
    axes[0, 0].set_xlabel('LPIPS (lower is better)')
    axes[0, 0].set_title('Reconstruction Quality by Strategy')
    axes[0, 0].axvline(x=0.20, color='g', linestyle='--', alpha=0.5, label='Target')

    # Plot 2: Convergence speed
    conv_vals = [results[s]["convergence_epoch"] for s in strategies]
    axes[0, 1].barh(range(len(strategies)), conv_vals, color=colors)
    axes[0, 1].set_yticks(range(len(strategies)))
    axes[0, 1].set_yticklabels(strategies, fontsize=8)
    axes[0, 1].set_xlabel('Convergence Epoch (lower is better)')
    axes[0, 1].set_title('Convergence Speed by Strategy')

    # Plot 3: Training stability
    var_vals = [results[s]["loss_variance"] for s in strategies]
    axes[1, 0].barh(range(len(strategies)), var_vals, color=colors)
    axes[1, 0].set_yticks(range(len(strategies)))
    axes[1, 0].set_yticklabels(strategies, fontsize=8)
    axes[1, 0].set_xlabel('Loss Variance (lower is better)')
    axes[1, 0].set_title('Training Stability by Strategy')

    # Plot 4: Pareto frontier (LPIPS vs convergence)
    for i, s in enumerate(strategies):
        axes[1, 1].scatter(results[s]["lpips"], results[s]["convergence_epoch"],
                          c=[colors[i]], s=100, label=s[:15])
    axes[1, 1].set_xlabel('LPIPS (lower is better)')
    axes[1, 1].set_ylabel('Convergence Epoch (lower is better)')
    axes[1, 1].set_title('Quality vs Speed Tradeoff')
    axes[1, 1].legend(fontsize=6, loc='upper right')

    plt.suptitle('Training Strategy Comparison (E2.3)', fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e2_3_training_strategy(runner: ExperimentRunner) -> dict:
    """Find optimal training strategy for adapter.

    Tests combinations of:
    - Loss functions: lpips, lpips_l2, lpips_ssim, multiscale_lpips
    - LR schedules: constant, cosine, warmup_cosine, cyclic
    - Augmentations: none, color_jitter, random_crop, mixup

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E2.3: Training Strategy Optimization")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e2_3/stage": 0, "e2_3/progress": 0.0})

    # Strategy configurations to test
    # Using a focused search to reduce compute time
    strategies = [
        # Loss function comparison (with baseline settings)
        {"loss": "lpips", "scheduler": "cosine", "augmentation": "none", "dropout": 0.1, "weight_decay": 1e-4},
        {"loss": "lpips_l2", "scheduler": "cosine", "augmentation": "none", "dropout": 0.1, "weight_decay": 1e-4},
        {"loss": "lpips_ssim", "scheduler": "cosine", "augmentation": "none", "dropout": 0.1, "weight_decay": 1e-4},
        {"loss": "multiscale_lpips", "scheduler": "cosine", "augmentation": "none", "dropout": 0.1, "weight_decay": 1e-4},

        # Scheduler comparison (with best loss)
        {"loss": "lpips_l2", "scheduler": "constant", "augmentation": "none", "dropout": 0.1, "weight_decay": 1e-4},
        {"loss": "lpips_l2", "scheduler": "warmup_cosine", "augmentation": "none", "dropout": 0.1, "weight_decay": 1e-4},
        {"loss": "lpips_l2", "scheduler": "cyclic", "augmentation": "none", "dropout": 0.1, "weight_decay": 1e-4},

        # Augmentation comparison
        {"loss": "lpips_l2", "scheduler": "warmup_cosine", "augmentation": "color_jitter", "dropout": 0.1, "weight_decay": 1e-4},
        {"loss": "lpips_l2", "scheduler": "warmup_cosine", "augmentation": "random_crop", "dropout": 0.1, "weight_decay": 1e-4},
        {"loss": "lpips_l2", "scheduler": "warmup_cosine", "augmentation": "mixup", "dropout": 0.1, "weight_decay": 1e-4},

        # Regularization comparison
        {"loss": "lpips_l2", "scheduler": "warmup_cosine", "augmentation": "none", "dropout": 0.2, "weight_decay": 1e-4},
        {"loss": "lpips_l2", "scheduler": "warmup_cosine", "augmentation": "none", "dropout": 0.1, "weight_decay": 1e-3},
    ]

    # =========================================================================
    # Stage 1: Generate data
    # =========================================================================
    print("\n[Stage 1/4] Generating training images...")

    train_images = generate_training_images(n_images=250)
    print(f"  Generated {len(train_images)} images")

    runner.log_metrics({"e2_3/stage": 1, "e2_3/progress": 0.05})

    # =========================================================================
    # Stage 2: Extract features
    # =========================================================================
    print("\n[Stage 2/4] Extracting features...")

    dinov2 = load_dinov2_model(device)
    dino_features = extract_dinov2_features(train_images, dinov2, device)
    del dinov2
    torch.cuda.empty_cache()

    vlm_model, vlm_processor = load_vlm_model(device)
    vlm_features = extract_vlm_features(train_images, vlm_model, vlm_processor, device)
    del vlm_model, vlm_processor
    torch.cuda.empty_cache()

    print(f"  DINOv2: {dino_features.shape}, VLM: {vlm_features.shape}")

    runner.log_metrics({"e2_3/stage": 2, "e2_3/progress": 0.25})

    # =========================================================================
    # Stage 3: Test each strategy
    # =========================================================================
    print("\n[Stage 3/4] Testing strategies...")

    targets = images_to_tensor(train_images)
    strategy_results = {}

    for idx, config in enumerate(strategies):
        strategy_name = f"{config['loss']}_{config['scheduler']}_{config['augmentation']}"
        progress = 0.25 + (0.65 * (idx / len(strategies)))
        runner.log_metrics({"e2_3/progress": progress})

        results = train_with_strategy(
            config,
            vlm_features,
            dino_features,
            targets,
            device,
            n_epochs=30,
            batch_size=8,
        )

        strategy_results[strategy_name] = results

        runner.log_metrics({
            f"e2_3/{strategy_name}_lpips": results["lpips"],
            f"e2_3/{strategy_name}_convergence": results["convergence_epoch"],
        })

    runner.log_metrics({"e2_3/stage": 3, "e2_3/progress": 0.9})

    # =========================================================================
    # Stage 4: Analyze and save results
    # =========================================================================
    print("\n[Stage 4/4] Analyzing results...")

    # Find best strategy
    best_strategy = min(strategy_results.keys(), key=lambda s: strategy_results[s]["lpips"])
    best_lpips = strategy_results[best_strategy]["lpips"]
    best_convergence = strategy_results[best_strategy]["convergence_epoch"]

    # Create visualization
    viz_bytes = create_strategy_comparison_plot(strategy_results)
    viz_path = runner.results.save_artifact("strategy_comparison.png", viz_bytes)

    # Save detailed results
    results_data = {
        "strategy_results": {
            k: {key: float(val) for key, val in v.items()}
            for k, v in strategy_results.items()
        },
        "best_strategy": best_strategy,
        "best_lpips": float(best_lpips),
        "best_convergence": best_convergence,
        "rankings": {
            "by_lpips": sorted(strategy_results.keys(), key=lambda s: strategy_results[s]["lpips"]),
            "by_convergence": sorted(strategy_results.keys(), key=lambda s: strategy_results[s]["convergence_epoch"]),
            "by_stability": sorted(strategy_results.keys(), key=lambda s: strategy_results[s]["loss_variance"]),
        },
        "recommended_config": {
            "loss": best_strategy.split("_")[0] if "_" in best_strategy else "lpips_l2",
            "scheduler": "warmup_cosine",
            "augmentation": "none",
            "dropout": 0.1,
            "weight_decay": 1e-4,
        },
    }
    data_path = runner.results.save_json_artifact("training_strategy_results.json", results_data)

    runner.log_metrics({
        "e2_3/stage": 4,
        "e2_3/progress": 1.0,
        "e2_3/best_strategy": best_strategy,
        "e2_3/best_lpips": best_lpips,
    })

    # =========================================================================
    # Determine finding
    # =========================================================================
    finding = (
        f"Best training strategy: {best_strategy} "
        f"(LPIPS={best_lpips:.3f}, converges at epoch {best_convergence}). "
        f"Rankings by quality: {results_data['rankings']['by_lpips'][:3]}. "
    )

    # Identify key insights
    loss_insights = []
    for loss in ["lpips", "lpips_l2", "lpips_ssim", "multiscale_lpips"]:
        matching = [s for s in strategy_results.keys() if s.startswith(loss)]
        if matching:
            avg_lpips = np.mean([strategy_results[s]["lpips"] for s in matching])
            loss_insights.append((loss, avg_lpips))

    if loss_insights:
        best_loss = min(loss_insights, key=lambda x: x[1])
        finding += f"Best loss function: {best_loss[0]}. "

    if best_lpips < 0.20:
        finding += "Achieves target quality."
    elif best_lpips < 0.22:
        finding += "Achieves acceptable quality."
    else:
        finding += "Needs further optimization."

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "best_strategy": best_strategy,
            "best_lpips": float(best_lpips),
            "best_convergence_epoch": best_convergence,
            "best_loss_variance": float(strategy_results[best_strategy]["loss_variance"]),
            "n_strategies_tested": len(strategies),
        },
        "artifacts": [viz_path, data_path],
    }
