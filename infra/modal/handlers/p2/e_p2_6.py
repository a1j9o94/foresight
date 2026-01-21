"""E-P2.6: Latency and Efficiency Analysis

Objective: Measure computational overhead and identify optimization opportunities.

Protocol:
1. Benchmark inference latency for each component
2. Measure memory usage
3. Profile bottlenecks
4. Test optimization strategies

Success Criteria:
- Latency overhead < 25% over VLM-only baseline
- Total pipeline < 1500ms
"""

import io
import os
import sys
import time

sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from runner import ExperimentRunner


class HybridFusionModule(nn.Module):
    """Fusion module for latency testing."""

    def __init__(self, vlm_dim=3584, spatial_dim=1024, fusion_dim=1024, n_layers=4, n_queries=64, output_dim=4096):
        super().__init__()
        self.vlm_proj = nn.Linear(vlm_dim, fusion_dim)
        self.spatial_proj = nn.Linear(spatial_dim, fusion_dim)
        self.spatial_pos = nn.Parameter(torch.randn(256, fusion_dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(n_queries, fusion_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=fusion_dim, nhead=8, dim_feedforward=fusion_dim * 4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(fusion_dim, output_dim)

    def forward(self, vlm_feat, spatial_feat):
        B = vlm_feat.size(0)
        vlm_p = self.vlm_proj(vlm_feat)
        sp_p = self.spatial_proj(spatial_feat) + self.spatial_pos[:spatial_feat.size(1)]
        ctx = torch.cat([vlm_p, sp_p], dim=1)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, ctx)
        return self.out_proj(q)


class SimplePixelDecoder(nn.Module):
    """Pixel decoder."""

    def __init__(self, input_dim=4096):
        super().__init__()
        self.initial = nn.Linear(input_dim, 256 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, cond):
        x = cond.mean(dim=1)
        x = self.initial(x).view(-1, 256, 7, 7)
        return self.decoder(x)


def generate_test_image():
    """Generate a single test image."""
    img = Image.new("RGB", (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy, size = 112, 112, 40
    draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=(255, 0, 0))
    return img


def measure_latency(fn, n_warmup=3, n_runs=10):
    """Measure latency of a function."""
    # Warmup
    for _ in range(n_warmup):
        fn()

    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def get_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
    return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}


def e_p2_6_latency_analysis(runner: ExperimentRunner) -> dict:
    """Measure latency and memory usage of hybrid pipeline.

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E-P2.6: Latency and Efficiency Analysis")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e_p2_6/stage": 0, "e_p2_6/progress": 0.0})

    # Generate test image
    test_image = generate_test_image()

    results = {}

    # =========================================================================
    # Benchmark 1: DINOv2 Encoding
    # =========================================================================
    print("\n[1/5] Benchmarking DINOv2-large encoding...")

    torch.cuda.reset_peak_memory_stats()
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2 = dinov2.to(device).eval()

    from torchvision import transforms
    dino_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = dino_transform(test_image).unsqueeze(0).to(device)

    def dinov2_forward():
        with torch.no_grad():
            f = dinov2.forward_features(img_tensor)
            if isinstance(f, dict):
                patch_f = f.get('x_norm_patchtokens', f.get('x_prenorm', None))
                if patch_f is None:
                    patch_f = f['x_norm'][:, 1:, :]  # Exclude CLS
                return patch_f
            return f[:, 1:, :]

    dino_latency = measure_latency(dinov2_forward)
    dino_memory = get_memory_usage()

    # Cache features for later
    dino_features = dinov2_forward()

    print(f"  Latency: {dino_latency['mean_ms']:.1f} ± {dino_latency['std_ms']:.1f} ms")
    print(f"  Memory: {dino_memory['max_allocated_mb']:.0f} MB")

    results["dinov2"] = {"latency": dino_latency, "memory": dino_memory}

    del dinov2
    torch.cuda.empty_cache()

    runner.log_metrics({
        "e_p2_6/stage": 1,
        "e_p2_6/progress": 0.2,
        "e_p2_6/dinov2_latency_ms": dino_latency["mean_ms"],
    })

    # =========================================================================
    # Benchmark 2: VLM Encoding
    # =========================================================================
    print("\n[2/5] Benchmarking VLM encoding...")

    torch.cuda.reset_peak_memory_stats()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": test_image},
            {"type": "text", "text": "Describe."},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[test_image], padding=True, return_tensors="pt").to(vlm.device)

    def vlm_forward():
        with torch.no_grad():
            outputs = vlm(**inputs, output_hidden_states=True, return_dict=True)
            return outputs.hidden_states[-1][0].float()

    vlm_latency = measure_latency(vlm_forward, n_warmup=2, n_runs=5)  # Fewer runs due to cost
    vlm_memory = get_memory_usage()

    # Cache features
    vlm_features = vlm_forward().unsqueeze(0)

    print(f"  Latency: {vlm_latency['mean_ms']:.1f} ± {vlm_latency['std_ms']:.1f} ms")
    print(f"  Memory: {vlm_memory['max_allocated_mb']:.0f} MB")

    results["vlm"] = {"latency": vlm_latency, "memory": vlm_memory}

    del vlm
    del processor
    torch.cuda.empty_cache()

    runner.log_metrics({
        "e_p2_6/stage": 2,
        "e_p2_6/progress": 0.4,
        "e_p2_6/vlm_latency_ms": vlm_latency["mean_ms"],
    })

    # =========================================================================
    # Benchmark 3: Fusion Module
    # =========================================================================
    print("\n[3/5] Benchmarking fusion module...")

    torch.cuda.reset_peak_memory_stats()

    vlm_dim = vlm_features.shape[-1]
    dino_dim = dino_features.shape[-1]

    fusion = HybridFusionModule(
        vlm_dim=vlm_dim,
        spatial_dim=dino_dim,
        fusion_dim=1024,
        n_layers=4,
        n_queries=64,
        output_dim=4096,
    ).to(device).eval()

    vlm_t = vlm_features.to(device)
    dino_t = dino_features.to(device)

    def fusion_forward():
        with torch.no_grad():
            return fusion(vlm_t, dino_t)

    fusion_latency = measure_latency(fusion_forward)
    fusion_memory = get_memory_usage()

    fusion_output = fusion_forward()

    print(f"  Latency: {fusion_latency['mean_ms']:.1f} ± {fusion_latency['std_ms']:.1f} ms")
    print(f"  Memory: {fusion_memory['max_allocated_mb']:.0f} MB")

    results["fusion"] = {"latency": fusion_latency, "memory": fusion_memory}

    runner.log_metrics({
        "e_p2_6/stage": 3,
        "e_p2_6/progress": 0.6,
        "e_p2_6/fusion_latency_ms": fusion_latency["mean_ms"],
    })

    # =========================================================================
    # Benchmark 4: Decoder
    # =========================================================================
    print("\n[4/5] Benchmarking pixel decoder...")

    torch.cuda.reset_peak_memory_stats()

    decoder = SimplePixelDecoder(input_dim=4096).to(device).eval()

    def decoder_forward():
        with torch.no_grad():
            return decoder(fusion_output)

    decoder_latency = measure_latency(decoder_forward)
    decoder_memory = get_memory_usage()

    print(f"  Latency: {decoder_latency['mean_ms']:.1f} ± {decoder_latency['std_ms']:.1f} ms")
    print(f"  Memory: {decoder_memory['max_allocated_mb']:.0f} MB")

    results["decoder"] = {"latency": decoder_latency, "memory": decoder_memory}

    runner.log_metrics({
        "e_p2_6/stage": 4,
        "e_p2_6/progress": 0.8,
        "e_p2_6/decoder_latency_ms": decoder_latency["mean_ms"],
    })

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n[5/5] Computing totals and analysis...")

    # Total latency
    total_hybrid = (
        dino_latency["mean_ms"] +
        vlm_latency["mean_ms"] +
        fusion_latency["mean_ms"] +
        decoder_latency["mean_ms"]
    )

    # VLM-only baseline (VLM + decoder, no DINOv2/fusion)
    vlm_only_baseline = vlm_latency["mean_ms"] + decoder_latency["mean_ms"]

    # Overhead from hybrid approach
    hybrid_overhead = (total_hybrid - vlm_only_baseline) / vlm_only_baseline

    # Memory totals (peak during full pipeline)
    peak_memory = max(
        dino_memory["max_allocated_mb"],
        vlm_memory["max_allocated_mb"],
    )

    print(f"\n  === Summary ===")
    print(f"  Total hybrid pipeline: {total_hybrid:.0f} ms")
    print(f"  VLM-only baseline: {vlm_only_baseline:.0f} ms")
    print(f"  Overhead: {hybrid_overhead * 100:.1f}%")
    print(f"  Peak memory: {peak_memory:.0f} MB")

    # Component breakdown
    print(f"\n  === Component Breakdown ===")
    print(f"  DINOv2:  {dino_latency['mean_ms']:6.0f} ms ({dino_latency['mean_ms']/total_hybrid*100:4.1f}%)")
    print(f"  VLM:     {vlm_latency['mean_ms']:6.0f} ms ({vlm_latency['mean_ms']/total_hybrid*100:4.1f}%)")
    print(f"  Fusion:  {fusion_latency['mean_ms']:6.0f} ms ({fusion_latency['mean_ms']/total_hybrid*100:4.1f}%)")
    print(f"  Decoder: {decoder_latency['mean_ms']:6.0f} ms ({decoder_latency['mean_ms']/total_hybrid*100:4.1f}%)")

    results["summary"] = {
        "total_hybrid_ms": total_hybrid,
        "vlm_only_baseline_ms": vlm_only_baseline,
        "overhead_percent": hybrid_overhead * 100,
        "peak_memory_mb": peak_memory,
    }

    # Target assessment
    overhead_target = 0.25  # 25%
    latency_target = 1500  # ms

    overhead_passed = hybrid_overhead < overhead_target
    latency_passed = total_hybrid < latency_target

    runner.log_metrics({
        "e_p2_6/stage": 5,
        "e_p2_6/progress": 1.0,
        "e_p2_6/total_latency_ms": total_hybrid,
        "e_p2_6/overhead_percent": hybrid_overhead * 100,
        "e_p2_6/peak_memory_mb": peak_memory,
    })

    # Save results
    results_data = {
        "component_latencies": {
            "dinov2_ms": dino_latency["mean_ms"],
            "vlm_ms": vlm_latency["mean_ms"],
            "fusion_ms": fusion_latency["mean_ms"],
            "decoder_ms": decoder_latency["mean_ms"],
        },
        "memory_usage": {
            "dinov2_mb": dino_memory["max_allocated_mb"],
            "vlm_mb": vlm_memory["max_allocated_mb"],
            "fusion_mb": fusion_memory["max_allocated_mb"],
            "decoder_mb": decoder_memory["max_allocated_mb"],
        },
        "summary": results["summary"],
        "assessment": {
            "overhead_target": overhead_target,
            "overhead_actual": hybrid_overhead,
            "overhead_passed": overhead_passed,
            "latency_target": latency_target,
            "latency_actual": total_hybrid,
            "latency_passed": latency_passed,
        },
    }
    data_path = runner.results.save_json_artifact("latency_analysis.json", results_data)

    # Create visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Latency breakdown
    components = ["DINOv2", "VLM", "Fusion", "Decoder"]
    latencies = [
        dino_latency["mean_ms"],
        vlm_latency["mean_ms"],
        fusion_latency["mean_ms"],
        decoder_latency["mean_ms"],
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    axes[0].bar(components, latencies, color=colors)
    axes[0].axhline(y=latency_target, color="r", linestyle="--", label=f"Target: {latency_target}ms")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title(f"Component Latency\nTotal: {total_hybrid:.0f}ms")
    axes[0].legend()

    # Pie chart
    axes[1].pie(latencies, labels=components, autopct='%1.1f%%', colors=colors)
    axes[1].set_title("Latency Distribution")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    viz_path = runner.results.save_artifact("latency_breakdown.png", buf.read())

    # Determine finding
    if overhead_passed and latency_passed:
        finding = (
            f"Latency requirements met. "
            f"Total pipeline: {total_hybrid:.0f}ms (target: <{latency_target}ms). "
            f"Overhead: {hybrid_overhead*100:.1f}% (target: <{overhead_target*100:.0f}%). "
            f"VLM encoding dominates ({vlm_latency['mean_ms']/total_hybrid*100:.0f}%). "
            f"DINOv2 adds {dino_latency['mean_ms']:.0f}ms, fusion adds only {fusion_latency['mean_ms']:.0f}ms."
        )
    elif latency_passed:
        finding = (
            f"Latency target met but overhead exceeds threshold. "
            f"Total: {total_hybrid:.0f}ms, Overhead: {hybrid_overhead*100:.1f}% (target: <{overhead_target*100:.0f}%). "
            f"Consider: DINOv2 model size reduction (ViT-L vs ViT-G), feature caching, batch optimization."
        )
    else:
        finding = (
            f"Latency requirements not met. "
            f"Total: {total_hybrid:.0f}ms (target: <{latency_target}ms). "
            f"Optimization needed: quantization, smaller models, parallelization."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "total_latency_ms": float(total_hybrid),
            "overhead_percent": float(hybrid_overhead * 100),
            "dinov2_latency_ms": float(dino_latency["mean_ms"]),
            "vlm_latency_ms": float(vlm_latency["mean_ms"]),
            "fusion_latency_ms": float(fusion_latency["mean_ms"]),
            "decoder_latency_ms": float(decoder_latency["mean_ms"]),
            "peak_memory_mb": float(peak_memory),
            "latency_overhead": float(hybrid_overhead),
        },
        "artifacts": [data_path, viz_path],
    }
