"""E3.2: Single Frame Future Prediction

Objective: Train query tokens to predict the latent representation of the *next*
frame (frame 17 given frames 1-16).

This is the core test of future prediction capability. The query tokens must learn
to extrapolate from the current visual state to predict what comes next.

Protocol:
1. Generate synthetic video data with predictable motion
2. Extract VLM features for context frames (1-16)
3. Extract VLM features for target frame (17)
4. Train query tokens to predict frame 17's latent from frames 1-16
5. Compare against copy baseline (using frame 16 as prediction)

Success Criteria:
- Cosine similarity > 0.7 on validation set
- Predicted future is closer to actual future than copy baseline
- Improvement is statistically significant (p < 0.01)

Failure Criteria:
- Predicted latents indistinguishable from copy baseline
- High variance across samples (memorizing, not generalizing)

Duration: ~1 day
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import stats

from runner import ExperimentRunner


class FuturePredictionQueries(nn.Module):
    """Learnable query tokens for future prediction.

    These query tokens attend to VLM hidden states via cross-attention
    and learn to extract information relevant to future prediction.
    """

    def __init__(
        self,
        num_queries: int = 32,
        hidden_dim: int = 3584,  # Qwen2.5-VL-7B hidden size
        num_layers: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable query embeddings
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # FFN for each layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, vlm_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_hidden_states: [batch, seq_len, hidden_dim] from VLM

        Returns:
            predicted_future: [batch, num_queries, hidden_dim]
        """
        B = vlm_hidden_states.size(0)

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention layers with FFN
        for attn, norm, ffn, ffn_norm in zip(
            self.cross_attention_layers,
            self.layer_norms,
            self.ffns,
            self.ffn_norms,
        ):
            # Cross-attention
            attn_out, _ = attn(queries, vlm_hidden_states, vlm_hidden_states)
            queries = norm(queries + attn_out)

            # FFN
            ffn_out = ffn(queries)
            queries = ffn_norm(queries + ffn_out)

        # Project to output space
        return self.output_proj(queries)


def generate_synthetic_video_data(
    n_videos: int = 200,
    n_frames: int = 24,
    img_size: int = 224,
) -> list[list[Image.Image]]:
    """Generate synthetic video data with predictable motion."""
    videos = []
    shapes = ["circle", "square", "triangle"]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    np.random.seed(42)

    for video_idx in range(n_videos):
        shape = shapes[video_idx % len(shapes)]
        color = colors[video_idx % len(colors)]
        size = np.random.randint(20, 40)

        # Random motion with some noise
        start_x = np.random.randint(size + 10, img_size // 2)
        start_y = np.random.randint(size + 10, img_size - size - 10)
        direction = np.random.choice(["right", "left", "down", "up"])

        # Velocity with slight variation
        base_velocity = np.random.randint(4, 8)

        frames = []
        for frame_idx in range(n_frames):
            img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            # Calculate position based on frame
            velocity = base_velocity + np.random.uniform(-0.5, 0.5)
            if direction == "right":
                cx = start_x + frame_idx * velocity
                cy = start_y + np.random.uniform(-1, 1)
            elif direction == "left":
                cx = start_x - frame_idx * velocity
                cy = start_y + np.random.uniform(-1, 1)
            elif direction == "down":
                cx = start_x + np.random.uniform(-1, 1)
                cy = start_y + frame_idx * velocity
            else:  # up
                cx = start_x + np.random.uniform(-1, 1)
                cy = start_y - frame_idx * velocity

            # Clamp to bounds
            cx = max(size, min(img_size - size, cx))
            cy = max(size, min(img_size - size, cy))

            # Draw shape
            if shape == "circle":
                draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
            elif shape == "square":
                draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
            elif shape == "triangle":
                points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
                draw.polygon(points, fill=color)

            frames.append(img)

        videos.append(frames)

    return videos


def load_vlm_model(device: torch.device):
    """Load Qwen2.5-VL model for feature extraction."""
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
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, processor


def extract_vlm_features(
    images: list[Image.Image],
    model,
    processor,
    device: torch.device,
) -> list[torch.Tensor]:
    """Extract VLM hidden states for multiple images."""
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

    return features_list


def extract_prediction_data(
    videos: list[list[Image.Image]],
    model,
    processor,
    device: torch.device,
    runner: ExperimentRunner,
    context_frames: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract features for future prediction.

    Returns:
        context_features: VLM encoding of last context frame (frame 16)
        target_features: VLM encoding of next frame (frame 17)
        copy_baseline_features: Same as context (for baseline comparison)
    """
    context_list = []
    target_list = []

    for idx, video in enumerate(videos):
        # Get features for context (frame 16) and target (frame 17)
        context_frame = video[context_frames - 1]  # Frame 16 (0-indexed as 15)
        target_frame = video[context_frames]  # Frame 17 (0-indexed as 16)

        context_feat = extract_vlm_features([context_frame], model, processor, device)[0]
        target_feat = extract_vlm_features([target_frame], model, processor, device)[0]

        context_list.append(context_feat)
        target_list.append(target_feat)

        if (idx + 1) % 20 == 0:
            progress = (idx + 1) / len(videos)
            runner.log_metrics({"e3_2/extraction_progress": progress})
            print(f"    Extracted {idx + 1}/{len(videos)} videos")

    # Pad to same length
    max_len = max(max(f.shape[0] for f in context_list), max(f.shape[0] for f in target_list))
    hidden_dim = context_list[0].shape[1]

    def pad_features(features_list):
        padded = []
        for f in features_list:
            if f.shape[0] < max_len:
                padding = torch.zeros(max_len - f.shape[0], hidden_dim)
                f = torch.cat([f, padding], dim=0)
            padded.append(f)
        return torch.stack(padded)

    context_features = pad_features(context_list)
    target_features = pad_features(target_list)
    copy_baseline = context_features.clone()  # Copy baseline uses frame 16 as prediction

    return context_features, target_features, copy_baseline


def compute_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute per-sample cosine similarity."""
    pred_pooled = pred.mean(dim=1)  # [batch, hidden_dim]
    target_pooled = target.mean(dim=1)

    pred_norm = F.normalize(pred_pooled, dim=-1)
    target_norm = F.normalize(target_pooled, dim=-1)

    return (pred_norm * target_norm).sum(dim=-1)


def create_results_plot(
    losses: list[float],
    val_metrics: list[dict],
) -> bytes:
    """Create visualization of training results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss curve
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")

    # Cosine similarity comparison
    if val_metrics:
        steps = [m["step"] for m in val_metrics]
        pred_sims = [m["predicted_cos_sim"] for m in val_metrics]
        copy_sims = [m["copy_baseline_cos_sim"] for m in val_metrics]

        axes[0, 1].plot(steps, pred_sims, label="Predicted", marker="o")
        axes[0, 1].plot(steps, copy_sims, label="Copy Baseline", marker="s")
        axes[0, 1].axhline(y=0.7, color="g", linestyle="--", alpha=0.5, label="Target (0.7)")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Cosine Similarity")
        axes[0, 1].set_title("Prediction vs Copy Baseline")
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)

        # Improvement over baseline
        improvements = [m["improvement"] for m in val_metrics]
        axes[1, 0].plot(steps, improvements, marker="o", color="purple")
        axes[1, 0].axhline(y=0.05, color="g", linestyle="--", alpha=0.5, label="Target (0.05)")
        axes[1, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Improvement over Baseline")
        axes[1, 0].set_title("Copy Baseline Delta")
        axes[1, 0].legend()

        # Final histogram
        final_pred = val_metrics[-1]["per_sample_sims"] if "per_sample_sims" in val_metrics[-1] else []
        if final_pred:
            axes[1, 1].hist(final_pred, bins=20, alpha=0.7, label="Predicted")
            axes[1, 1].axvline(x=0.7, color="g", linestyle="--", label="Target")
            axes[1, 1].set_xlabel("Cosine Similarity")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("Final Prediction Distribution")
            axes[1, 1].legend()

    plt.suptitle("E3.2: Single Frame Future Prediction", fontsize=12)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e3_2_single_frame_prediction(runner: ExperimentRunner) -> dict:
    """Run E3.2: Single frame future prediction.

    This experiment trains query tokens to predict the next frame's latent
    representation from the current frame's VLM encoding.

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E3.2: Single Frame Future Prediction")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e3_2/stage": 0, "e3_2/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate synthetic video data
    # =========================================================================
    print("\n[Stage 1/5] Generating synthetic video data...")

    n_videos = 100
    videos = generate_synthetic_video_data(n_videos=n_videos, n_frames=24)

    # Split into train/val
    n_train = int(0.8 * n_videos)
    train_videos = videos[:n_train]
    val_videos = videos[n_train:]

    print(f"  Generated {len(videos)} videos")
    print(f"  Train: {len(train_videos)}, Val: {len(val_videos)}")

    runner.log_metrics({
        "e3_2/stage": 1,
        "e3_2/progress": 0.1,
        "e3_2/n_train": len(train_videos),
        "e3_2/n_val": len(val_videos),
    })

    # =========================================================================
    # Stage 2: Load VLM and extract features
    # =========================================================================
    print("\n[Stage 2/5] Loading VLM and extracting features...")

    vlm_model, vlm_processor = load_vlm_model(device)

    print("  Extracting training features...")
    train_context, train_target, train_copy = extract_prediction_data(
        train_videos, vlm_model, vlm_processor, device, runner, context_frames=16
    )

    print("  Extracting validation features...")
    val_context, val_target, val_copy = extract_prediction_data(
        val_videos, vlm_model, vlm_processor, device, runner, context_frames=16
    )

    hidden_dim = train_context.shape[-1]
    print(f"  Train context: {train_context.shape}")
    print(f"  Val context: {val_context.shape}")
    print(f"  Hidden dim: {hidden_dim}")

    runner.log_metrics({
        "e3_2/stage": 2,
        "e3_2/progress": 0.4,
        "e3_2/hidden_dim": hidden_dim,
    })

    # Free VLM memory
    del vlm_model, vlm_processor
    torch.cuda.empty_cache()

    # =========================================================================
    # Stage 3: Initialize query tokens
    # =========================================================================
    print("\n[Stage 3/5] Initializing query tokens...")

    query_model = FuturePredictionQueries(
        num_queries=32,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_heads=8,
    ).to(device)

    n_params = sum(p.numel() for p in query_model.parameters())
    print(f"  Query model params: {n_params:,} ({n_params/1e6:.2f}M)")

    runner.log_metrics({
        "e3_2/stage": 3,
        "e3_2/progress": 0.45,
        "e3_2/query_params": n_params,
    })

    # =========================================================================
    # Stage 4: Train query tokens
    # =========================================================================
    print("\n[Stage 4/5] Training query tokens...")

    train_context_t = train_context.to(device)
    train_target_t = train_target.to(device)
    val_context_t = val_context.to(device)
    val_target_t = val_target.to(device)
    val_copy_t = val_copy.to(device)

    n_steps = 5000
    batch_size = 8
    lr = 1e-4

    optimizer = torch.optim.AdamW(query_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, eta_min=1e-6)

    losses = []
    val_metrics = []

    for step in range(n_steps):
        query_model.train()

        # Sample batch
        batch_idx = torch.randint(0, len(train_context_t), (batch_size,))
        batch_context = train_context_t[batch_idx]
        batch_target = train_target_t[batch_idx]

        # Forward pass
        optimizer.zero_grad()
        predicted = query_model(batch_context)

        # Compute loss
        pred_pooled = predicted.mean(dim=1)
        target_pooled = batch_target.mean(dim=1)

        # Cosine similarity loss
        pred_norm = F.normalize(pred_pooled, dim=-1)
        target_norm = F.normalize(target_pooled, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        loss_cos = 1 - cos_sim.mean()

        # MSE loss in normalized space
        loss_mse = F.mse_loss(pred_norm, target_norm)

        # Combined loss
        loss = loss_cos + 0.1 * loss_mse

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(query_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        # Validate periodically
        if (step + 1) % 500 == 0 or step == 0:
            query_model.eval()
            with torch.no_grad():
                # Predict on validation set
                val_pred = query_model(val_context_t)

                # Compute similarities
                pred_sims = compute_cosine_similarity(val_pred, val_target_t)
                copy_sims = compute_cosine_similarity(val_copy_t, val_target_t)

                pred_mean = float(pred_sims.mean().item())
                copy_mean = float(copy_sims.mean().item())
                improvement = pred_mean - copy_mean

                val_metrics.append({
                    "step": step + 1,
                    "predicted_cos_sim": pred_mean,
                    "copy_baseline_cos_sim": copy_mean,
                    "improvement": improvement,
                    "per_sample_sims": pred_sims.cpu().tolist(),
                })

            print(f"    Step {step + 1}/{n_steps}: "
                  f"Pred={pred_mean:.4f}, Copy={copy_mean:.4f}, "
                  f"Improvement={improvement:+.4f}")

            runner.log_metrics({
                "e3_2/loss": loss.item(),
                "e3_2/val_predicted_cos_sim": pred_mean,
                "e3_2/val_copy_cos_sim": copy_mean,
                "e3_2/val_improvement": improvement,
            }, step=step)

    runner.log_metrics({
        "e3_2/stage": 4,
        "e3_2/progress": 0.9,
    })

    # =========================================================================
    # Stage 5: Final evaluation and statistical tests
    # =========================================================================
    print("\n[Stage 5/5] Final evaluation...")

    query_model.eval()
    with torch.no_grad():
        val_pred = query_model(val_context_t)

        # Per-sample similarities
        pred_sims = compute_cosine_similarity(val_pred, val_target_t)
        copy_sims = compute_cosine_similarity(val_copy_t, val_target_t)

        # Random baseline (random vectors)
        random_pred = torch.randn_like(val_pred)
        random_sims = compute_cosine_similarity(random_pred, val_target_t)

    pred_sims_np = pred_sims.cpu().numpy()
    copy_sims_np = copy_sims.cpu().numpy()
    random_sims_np = random_sims.cpu().numpy()

    # Statistical tests
    t_stat, p_value = stats.ttest_rel(pred_sims_np, copy_sims_np)

    # Final metrics
    final_pred_sim = float(np.mean(pred_sims_np))
    final_copy_sim = float(np.mean(copy_sims_np))
    final_random_sim = float(np.mean(random_sims_np))
    final_improvement = final_pred_sim - final_copy_sim
    random_delta = final_pred_sim - final_random_sim

    print(f"  Predicted cos_sim: {final_pred_sim:.4f}")
    print(f"  Copy baseline cos_sim: {final_copy_sim:.4f}")
    print(f"  Random baseline cos_sim: {final_random_sim:.4f}")
    print(f"  Improvement over copy: {final_improvement:+.4f}")
    print(f"  Improvement over random: {random_delta:+.4f}")
    print(f"  Statistical significance: t={t_stat:.3f}, p={p_value:.4f}")

    # Create visualization
    viz_bytes = create_results_plot(losses, val_metrics)
    viz_path = runner.results.save_artifact("single_frame_prediction.png", viz_bytes)

    # Save results
    results_data = {
        "final_predicted_cos_sim": final_pred_sim,
        "final_copy_baseline_cos_sim": final_copy_sim,
        "final_random_cos_sim": final_random_sim,
        "improvement_over_copy": final_improvement,
        "improvement_over_random": random_delta,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "statistically_significant": p_value < 0.01,
        "per_sample_pred_sims": pred_sims_np.tolist(),
        "per_sample_copy_sims": copy_sims_np.tolist(),
    }
    data_path = runner.results.save_json_artifact("single_frame_results.json", results_data)

    runner.log_metrics({
        "e3_2/stage": 5,
        "e3_2/progress": 1.0,
        "e3_2/final_predicted_cos_sim": final_pred_sim,
        "e3_2/final_improvement": final_improvement,
        "e3_2/p_value": p_value,
    })

    # =========================================================================
    # Determine finding
    # =========================================================================
    cos_sim_target = 0.7
    improvement_target = 0.05

    success = (
        final_pred_sim > cos_sim_target and
        final_improvement > improvement_target and
        p_value < 0.01
    )

    if success:
        finding = (
            f"SINGLE FRAME PREDICTION SUCCESSFUL: Query tokens predict next frame latent "
            f"(cos_sim={final_pred_sim:.3f} > {cos_sim_target}). "
            f"Improvement over copy baseline: {final_improvement:+.3f} (p={p_value:.4f}). "
            f"Ready to proceed with action-conditioned prediction (E3.3)."
        )
    elif final_pred_sim > 0.5 and final_improvement > 0:
        finding = (
            f"SINGLE FRAME PREDICTION PARTIAL: Query tokens show some prediction capability "
            f"(cos_sim={final_pred_sim:.3f}), but below target ({cos_sim_target}). "
            f"Improvement over copy: {final_improvement:+.3f}. "
            f"Consider longer training or architecture improvements."
        )
    else:
        finding = (
            f"SINGLE FRAME PREDICTION FAILED: Query tokens fail to predict next frame "
            f"(cos_sim={final_pred_sim:.3f}). "
            f"No significant improvement over copy baseline ({final_improvement:+.3f}). "
            f"Investigate architecture or use contrastive training."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "cosine_similarity": final_pred_sim,
            "copy_baseline_delta": final_improvement,
            "random_baseline_delta": random_delta,
            "p_value": float(p_value),
            "statistically_significant": p_value < 0.01,
            "passed": success,
        },
        "artifacts": [viz_path, data_path],
    }
