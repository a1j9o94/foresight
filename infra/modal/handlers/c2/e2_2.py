"""E2.2: Architecture Comparison

Objective: Compare different adapter architectures at fixed parameter budget (~10M params).

Architectures to test:
A. Query-based Adapter (baseline from E2.1) - Learned queries attend to concatenated features
B. Bottleneck Adapter - Compress features through bottleneck, then expand
C. LoRA-style Adapter - Low-rank projections from each stream
D. Perceiver-style Adapter - Iterative cross-attention with fixed latent array

Protocol:
1. Implement each architecture at ~10M params
2. Train with identical procedure
3. Compare reconstruction quality and inference speed

Success Metrics:
- LPIPS (lower is better)
- Spatial IoU (higher is better)
- Inference latency (lower is better)
- Training stability (loss variance)
- Memory usage (peak VRAM)

This experiment identifies the best adapter architecture for efficient bridging.
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
# Architecture A: Query-based Adapter (baseline)
# =============================================================================

class QueryAdapter(nn.Module):
    """Query-based adapter using learned queries to attend to hybrid features."""

    def __init__(
        self,
        vlm_dim: int = 3584,
        dino_dim: int = 1024,
        ltx_dim: int = 4096,
        hidden_dim: int = 384,
        n_layers: int = 2,
        n_queries: int = 77,
    ):
        super().__init__()
        self.name = "query_adapter"

        # Input projections
        self.vlm_proj = nn.Linear(vlm_dim, hidden_dim)
        self.dino_proj = nn.Linear(dino_dim, hidden_dim)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 64),
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.1,
            )
            for _ in range(n_layers)
        ])

        # Output projection
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


# =============================================================================
# Architecture B: Bottleneck Adapter
# =============================================================================

class BottleneckAdapter(nn.Module):
    """Compress features through bottleneck, then expand to output."""

    def __init__(
        self,
        vlm_dim: int = 3584,
        dino_dim: int = 1024,
        ltx_dim: int = 4096,
        bottleneck_dim: int = 256,
        hidden_dim: int = 512,
        n_tokens: int = 77,
    ):
        super().__init__()
        self.name = "bottleneck_adapter"
        self.n_tokens = n_tokens
        self.ltx_dim = ltx_dim

        # Compress each stream
        self.vlm_compress = nn.Sequential(
            nn.Linear(vlm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.dino_compress = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        # Bottleneck fusion
        self.fusion = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Expand to output tokens
        self.token_generator = nn.Linear(hidden_dim, n_tokens * hidden_dim)

        # Per-token refinement
        self.token_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, ltx_dim),
        )

    def forward(self, vlm_features: torch.Tensor, dino_features: torch.Tensor) -> torch.Tensor:
        B = vlm_features.size(0)

        # Compress with global pooling
        vlm = self.vlm_compress(vlm_features.mean(1))
        dino = self.dino_compress(dino_features.mean(1))

        # Fuse
        fused = self.fusion(torch.cat([vlm, dino], dim=-1))

        # Generate tokens
        tokens = self.token_generator(fused)
        tokens = tokens.view(B, self.n_tokens, -1)

        # Refine each token
        return self.token_refine(tokens)


# =============================================================================
# Architecture C: LoRA-style Adapter
# =============================================================================

class LoRAAdapter(nn.Module):
    """Low-rank projections for efficient bridging."""

    def __init__(
        self,
        vlm_dim: int = 3584,
        dino_dim: int = 1024,
        ltx_dim: int = 4096,
        rank: int = 128,
        n_tokens: int = 77,
    ):
        super().__init__()
        self.name = "lora_adapter"
        self.n_tokens = n_tokens
        self.ltx_dim = ltx_dim

        # Low-rank projections for VLM
        self.vlm_down = nn.Linear(vlm_dim, rank, bias=False)
        self.vlm_up = nn.Linear(rank, ltx_dim, bias=False)

        # Low-rank projections for DINOv2
        self.dino_down = nn.Linear(dino_dim, rank, bias=False)
        self.dino_up = nn.Linear(rank, ltx_dim, bias=False)

        # Attention-based pooling for each stream
        self.vlm_pool = nn.MultiheadAttention(rank, 4, batch_first=True)
        self.dino_pool = nn.MultiheadAttention(rank, 4, batch_first=True)

        # Learned queries for output generation
        self.output_queries = nn.Parameter(torch.randn(n_tokens, rank) * 0.02)

        # Fusion and output
        self.fusion_layer = nn.TransformerDecoderLayer(
            d_model=rank,
            nhead=4,
            dim_feedforward=rank * 4,
            batch_first=True,
        )

        # Final projection
        self.out_proj = nn.Linear(rank, ltx_dim)

    def forward(self, vlm_features: torch.Tensor, dino_features: torch.Tensor) -> torch.Tensor:
        B = vlm_features.size(0)

        # Project to low-rank space
        vlm_low = self.vlm_down(vlm_features)
        dino_low = self.dino_down(dino_features)

        # Concatenate as context
        context = torch.cat([vlm_low, dino_low], dim=1)

        # Queries attend to context
        queries = self.output_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.fusion_layer(queries, context)

        # Project to output space
        return self.out_proj(queries)


# =============================================================================
# Architecture D: Perceiver-style Adapter
# =============================================================================

class PerceiverAdapter(nn.Module):
    """Iterative cross-attention with fixed latent array (Perceiver-style)."""

    def __init__(
        self,
        vlm_dim: int = 3584,
        dino_dim: int = 1024,
        ltx_dim: int = 4096,
        latent_dim: int = 384,
        n_latents: int = 77,
        n_iterations: int = 3,
    ):
        super().__init__()
        self.name = "perceiver_adapter"
        self.n_iterations = n_iterations

        # Latent array
        self.latents = nn.Parameter(torch.randn(n_latents, latent_dim) * 0.02)

        # Input projections
        self.vlm_proj = nn.Linear(vlm_dim, latent_dim)
        self.dino_proj = nn.Linear(dino_dim, latent_dim)

        # Cross-attention (latents attend to input)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(latent_dim, 4, batch_first=True)
            for _ in range(n_iterations)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim)
            for _ in range(n_iterations)
        ])

        # Self-attention (latents attend to themselves)
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(latent_dim, 4, batch_first=True)
            for _ in range(n_iterations)
        ])
        self.self_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim)
            for _ in range(n_iterations)
        ])

        # FFN for each iteration
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 4),
                nn.GELU(),
                nn.Linear(latent_dim * 4, latent_dim),
            )
            for _ in range(n_iterations)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim)
            for _ in range(n_iterations)
        ])

        # Output projection
        self.out_proj = nn.Linear(latent_dim, ltx_dim)

    def forward(self, vlm_features: torch.Tensor, dino_features: torch.Tensor) -> torch.Tensor:
        B = vlm_features.size(0)

        # Project inputs
        context = torch.cat([
            self.vlm_proj(vlm_features),
            self.dino_proj(dino_features)
        ], dim=1)

        # Initialize latents
        x = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Iterative refinement
        for i in range(self.n_iterations):
            # Cross-attention to input
            attn_out, _ = self.cross_attns[i](x, context, context)
            x = self.cross_norms[i](x + attn_out)

            # Self-attention
            self_out, _ = self.self_attns[i](x, x, x)
            x = self.self_norms[i](x + self_out)

            # FFN
            ffn_out = self.ffns[i](x)
            x = self.ffn_norms[i](x + ffn_out)

        return self.out_proj(x)


# =============================================================================
# Shared Components
# =============================================================================

class SimplePixelDecoder(nn.Module):
    """Simple pixel decoder for reconstruction evaluation."""

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


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_adapter(arch_name: str, vlm_dim: int, dino_dim: int) -> nn.Module:
    """Create adapter of specified architecture."""
    if arch_name == "query":
        return QueryAdapter(vlm_dim=vlm_dim, dino_dim=dino_dim)
    elif arch_name == "bottleneck":
        return BottleneckAdapter(vlm_dim=vlm_dim, dino_dim=dino_dim)
    elif arch_name == "lora":
        return LoRAAdapter(vlm_dim=vlm_dim, dino_dim=dino_dim)
    elif arch_name == "perceiver":
        return PerceiverAdapter(vlm_dim=vlm_dim, dino_dim=dino_dim)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


# =============================================================================
# Feature Extraction (reused from E2.1)
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

    # Pad to same length
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


def compute_metrics(recon: torch.Tensor, target: torch.Tensor, device: torch.device) -> dict:
    """Compute reconstruction metrics."""
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
        "spatial_iou": spatial_iou,
    }


def train_architecture(
    arch_name: str,
    vlm_features: torch.Tensor,
    dino_features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    n_epochs: int = 50,
    batch_size: int = 8,
    runner: ExperimentRunner = None,
) -> dict:
    """Train an adapter architecture and return metrics."""

    print(f"\n  Training {arch_name} adapter...")

    # Create adapter
    adapter = create_adapter(
        arch_name,
        vlm_dim=vlm_features.shape[-1],
        dino_dim=dino_features.shape[-1],
    ).to(device)

    decoder = SimplePixelDecoder(input_dim=4096, output_size=224).to(device)

    adapter_params = count_parameters(adapter)
    decoder_params = count_parameters(decoder)
    print(f"    Adapter params: {adapter_params:,} ({adapter_params/1e6:.2f}M)")

    # Move data to device
    vlm_t = vlm_features.to(device)
    dino_t = dino_features.to(device)
    targets_t = targets.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(decoder.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Training with loss variance tracking
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

            optimizer.zero_grad()

            conditioning = adapter(batch_vlm, batch_dino)
            recon = decoder(conditioning)

            mse_loss = F.mse_loss(recon, batch_targets)
            l1_loss = F.l1_loss(recon, batch_targets)
            loss = mse_loss + 0.1 * l1_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        loss_history.append(epoch_loss)

        if runner and (epoch + 1) % 10 == 0:
            runner.log_metrics({
                f"e2_2/{arch_name}_loss": epoch_loss,
            }, step=epoch)

    training_time = time.time() - start_time

    # Compute loss variance (training stability)
    loss_variance = np.var(loss_history[-10:]) if len(loss_history) >= 10 else np.var(loss_history)

    # Measure inference latency
    adapter.eval()
    decoder.eval()

    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = adapter(vlm_t[:1], dino_t[:1])

    # Measure
    torch.cuda.synchronize()
    latency_start = time.time()
    n_inference = 10
    with torch.no_grad():
        for _ in range(n_inference):
            conditioning = adapter(vlm_t[:1], dino_t[:1])
            _ = decoder(conditioning)
    torch.cuda.synchronize()
    inference_latency = (time.time() - latency_start) / n_inference * 1000  # ms

    # Measure peak memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        conditioning = adapter(vlm_t[:8], dino_t[:8])
        _ = decoder(conditioning)
    peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB

    # Evaluate reconstruction quality
    with torch.no_grad():
        conditioning = adapter(vlm_t, dino_t)
        recon = decoder(conditioning)

    metrics = compute_metrics(recon, targets_t, device)

    results = {
        "lpips": metrics["lpips"],
        "ssim": metrics["ssim"],
        "spatial_iou": metrics["spatial_iou"],
        "params": adapter_params,
        "training_time": training_time,
        "inference_latency_ms": inference_latency,
        "loss_variance": float(loss_variance),
        "peak_memory_gb": peak_memory,
    }

    print(f"    LPIPS: {metrics['lpips']:.4f}, Spatial IoU: {metrics['spatial_iou']:.4f}")
    print(f"    Latency: {inference_latency:.1f}ms, Memory: {peak_memory:.2f}GB")

    # Cleanup
    del adapter, decoder
    torch.cuda.empty_cache()

    return results


def create_comparison_plot(results: dict) -> bytes:
    """Create visualization comparing architectures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    architectures = list(results.keys())
    n_arch = len(architectures)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Colors for each architecture
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # LPIPS comparison
    lpips_vals = [results[a]["lpips"] for a in architectures]
    axes[0, 0].bar(range(n_arch), lpips_vals, color=colors[:n_arch])
    axes[0, 0].set_xticks(range(n_arch))
    axes[0, 0].set_xticklabels(architectures, rotation=45, ha='right')
    axes[0, 0].set_ylabel('LPIPS (lower is better)')
    axes[0, 0].set_title('Reconstruction Quality')
    axes[0, 0].axhline(y=0.20, color='g', linestyle='--', alpha=0.5)

    # Spatial IoU comparison
    iou_vals = [results[a]["spatial_iou"] for a in architectures]
    axes[0, 1].bar(range(n_arch), iou_vals, color=colors[:n_arch])
    axes[0, 1].set_xticks(range(n_arch))
    axes[0, 1].set_xticklabels(architectures, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Spatial IoU (higher is better)')
    axes[0, 1].set_title('Spatial Preservation')
    axes[0, 1].axhline(y=0.75, color='g', linestyle='--', alpha=0.5)

    # Inference latency comparison
    latency_vals = [results[a]["inference_latency_ms"] for a in architectures]
    axes[0, 2].bar(range(n_arch), latency_vals, color=colors[:n_arch])
    axes[0, 2].set_xticks(range(n_arch))
    axes[0, 2].set_xticklabels(architectures, rotation=45, ha='right')
    axes[0, 2].set_ylabel('Inference Latency (ms)')
    axes[0, 2].set_title('Inference Speed')

    # Training stability (loss variance)
    var_vals = [results[a]["loss_variance"] for a in architectures]
    axes[1, 0].bar(range(n_arch), var_vals, color=colors[:n_arch])
    axes[1, 0].set_xticks(range(n_arch))
    axes[1, 0].set_xticklabels(architectures, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Loss Variance (lower is better)')
    axes[1, 0].set_title('Training Stability')

    # Memory usage
    mem_vals = [results[a]["peak_memory_gb"] for a in architectures]
    axes[1, 1].bar(range(n_arch), mem_vals, color=colors[:n_arch])
    axes[1, 1].set_xticks(range(n_arch))
    axes[1, 1].set_xticklabels(architectures, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Peak Memory (GB)')
    axes[1, 1].set_title('Memory Usage')

    # Parameter count
    param_vals = [results[a]["params"] / 1e6 for a in architectures]
    axes[1, 2].bar(range(n_arch), param_vals, color=colors[:n_arch])
    axes[1, 2].set_xticks(range(n_arch))
    axes[1, 2].set_xticklabels(architectures, rotation=45, ha='right')
    axes[1, 2].set_ylabel('Parameters (M)')
    axes[1, 2].set_title('Model Size')

    plt.suptitle('Architecture Comparison (E2.2)', fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e2_2_architecture_comparison(runner: ExperimentRunner) -> dict:
    """Compare different adapter architectures at fixed parameter budget.

    Architectures compared:
    1. Query-based (baseline)
    2. Bottleneck
    3. LoRA-style
    4. Perceiver-style

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E2.2: Architecture Comparison")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e2_2/stage": 0, "e2_2/progress": 0.0})

    architectures = ["query", "bottleneck", "lora", "perceiver"]

    # =========================================================================
    # Stage 1: Generate data
    # =========================================================================
    print("\n[Stage 1/5] Generating training images...")

    train_images = generate_training_images(n_images=300)
    print(f"  Generated {len(train_images)} images")

    runner.log_metrics({"e2_2/stage": 1, "e2_2/progress": 0.05})

    # =========================================================================
    # Stage 2: Extract DINOv2 features
    # =========================================================================
    print("\n[Stage 2/5] Extracting DINOv2 features...")

    dinov2 = load_dinov2_model(device)
    dino_features = extract_dinov2_features(train_images, dinov2, device)
    print(f"  DINOv2 features: {dino_features.shape}")

    del dinov2
    torch.cuda.empty_cache()

    runner.log_metrics({"e2_2/stage": 2, "e2_2/progress": 0.15})

    # =========================================================================
    # Stage 3: Extract VLM features
    # =========================================================================
    print("\n[Stage 3/5] Extracting VLM features...")

    vlm_model, vlm_processor = load_vlm_model(device)
    vlm_features = extract_vlm_features(train_images, vlm_model, vlm_processor, device)
    print(f"  VLM features: {vlm_features.shape}")

    del vlm_model, vlm_processor
    torch.cuda.empty_cache()

    runner.log_metrics({"e2_2/stage": 3, "e2_2/progress": 0.35})

    # =========================================================================
    # Stage 4: Train each architecture
    # =========================================================================
    print("\n[Stage 4/5] Training architectures...")

    targets = images_to_tensor(train_images)
    arch_results = {}

    for idx, arch_name in enumerate(architectures):
        progress = 0.35 + (0.55 * (idx / len(architectures)))
        runner.log_metrics({"e2_2/progress": progress})

        results = train_architecture(
            arch_name,
            vlm_features,
            dino_features,
            targets,
            device,
            n_epochs=50,
            batch_size=8,
            runner=runner,
        )

        arch_results[arch_name] = results

        runner.log_metrics({
            f"e2_2/{arch_name}_lpips": results["lpips"],
            f"e2_2/{arch_name}_spatial_iou": results["spatial_iou"],
            f"e2_2/{arch_name}_latency_ms": results["inference_latency_ms"],
        })

    runner.log_metrics({"e2_2/stage": 4, "e2_2/progress": 0.9})

    # =========================================================================
    # Stage 5: Analysis and save artifacts
    # =========================================================================
    print("\n[Stage 5/5] Analyzing results...")

    # Find best architecture
    best_arch = min(arch_results.keys(), key=lambda a: arch_results[a]["lpips"])
    best_lpips = arch_results[best_arch]["lpips"]
    best_spatial_iou = arch_results[best_arch]["spatial_iou"]

    # Create comparison visualization
    viz_bytes = create_comparison_plot(arch_results)
    viz_path = runner.results.save_artifact("architecture_comparison.png", viz_bytes)

    # Save detailed results
    results_data = {
        "architecture_results": {
            k: {key: float(val) for key, val in v.items()}
            for k, v in arch_results.items()
        },
        "best_architecture": best_arch,
        "rankings": {
            "by_lpips": sorted(architectures, key=lambda a: arch_results[a]["lpips"]),
            "by_spatial_iou": sorted(architectures, key=lambda a: -arch_results[a]["spatial_iou"]),
            "by_latency": sorted(architectures, key=lambda a: arch_results[a]["inference_latency_ms"]),
            "by_stability": sorted(architectures, key=lambda a: arch_results[a]["loss_variance"]),
        },
    }
    data_path = runner.results.save_json_artifact("architecture_comparison.json", results_data)

    runner.log_metrics({
        "e2_2/stage": 5,
        "e2_2/progress": 1.0,
        "e2_2/best_architecture": best_arch,
        "e2_2/best_lpips": best_lpips,
        "e2_2/best_spatial_iou": best_spatial_iou,
    })

    # =========================================================================
    # Determine finding
    # =========================================================================
    finding = (
        f"Best architecture: {best_arch} "
        f"(LPIPS={best_lpips:.3f}, Spatial IoU={best_spatial_iou:.3f}). "
        f"Rankings by LPIPS: {results_data['rankings']['by_lpips']}. "
        f"Rankings by latency: {results_data['rankings']['by_latency']}. "
    )

    if best_lpips < 0.20:
        finding += f"The {best_arch} architecture meets target quality."
    elif best_lpips < 0.22:
        finding += f"The {best_arch} architecture meets acceptable quality."
    else:
        finding += f"No architecture meets quality targets; consider hybrid approaches."

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "best_architecture": best_arch,
            "best_lpips": float(best_lpips),
            "best_spatial_iou": float(best_spatial_iou),
            "best_latency_ms": float(arch_results[best_arch]["inference_latency_ms"]),
            "query_lpips": float(arch_results["query"]["lpips"]),
            "bottleneck_lpips": float(arch_results["bottleneck"]["lpips"]),
            "lora_lpips": float(arch_results["lora"]["lpips"]),
            "perceiver_lpips": float(arch_results["perceiver"]["lpips"]),
        },
        "artifacts": [viz_path, data_path],
    }
