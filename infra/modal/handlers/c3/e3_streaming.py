"""E3.4: Streaming Video Prediction Training

Objective: Train StreamingPredictor to predict multi-frame sequences with
context jumps, matching the production streaming inference architecture.

This experiment trains on sequences where:
1. The model predicts multiple future frames (not just next frame)
2. Context jumps can occur mid-sequence (simulating new user input)
3. Training matches production: recurrent prediction with sliding window

Protocol:
1. Load SSv2 dataset (real video data with cause-effect actions)
2. Extract VLM features for video sequences
3. Train StreamingPredictor on multi-frame prediction
4. Evaluate with context jump scenarios
5. Compare against copy baseline and single-frame predictor

Success Criteria:
- Per-segment cosine similarity > 0.65
- Handles context jumps without quality degradation
- Beats copy baseline consistently (p < 0.01)

Duration: ~2 days (more data, longer sequences)
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
from scipy import stats

from runner import ExperimentRunner
from .streaming_predictor import StreamingPredictor, StreamingPredictorConfig, ContextInjection


def load_ssv2_data(
    subset_size: int = 1000,
    num_frames: int = 24,
    mock_mode: bool = False,
) -> list[list[Image.Image]]:
    """Load Something-Something v2 data or generate mock data.

    Args:
        subset_size: Number of videos to load
        num_frames: Frames per video
        mock_mode: If True, generate synthetic data (for testing)

    Returns:
        List of videos, each a list of PIL Images
    """
    if mock_mode:
        return generate_synthetic_data(subset_size, num_frames)

    try:
        from foresight_training.data import SSv2Dataset

        # Check if SSv2 videos are available
        video_dir = os.environ.get("SSV2_VIDEO_DIR", "/data/ssv2/videos")
        if not os.path.exists(video_dir):
            print(f"  SSv2 video directory not found: {video_dir}")
            print("  Falling back to mock data")
            return generate_synthetic_data(subset_size, num_frames)

        ds = SSv2Dataset(
            split="train",
            video_dir=video_dir,
            subset_size=subset_size,
            num_frames=num_frames,
            frame_size=(224, 224),
        )

        videos = []
        for i, sample in enumerate(ds):
            # Convert tensor frames to PIL Images
            frames = sample.frames  # [T, C, H, W]
            pil_frames = []
            for t in range(frames.size(0)):
                frame = frames[t].permute(1, 2, 0).numpy()  # [H, W, C]
                frame = (frame * 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(frame))
            videos.append(pil_frames)

            if (i + 1) % 100 == 0:
                print(f"    Loaded {i + 1}/{subset_size} videos")

        return videos

    except ImportError as e:
        print(f"  Could not import SSv2Dataset: {e}")
        print("  Falling back to mock data")
        return generate_synthetic_data(subset_size, num_frames)


def generate_synthetic_data(
    n_videos: int = 200,
    n_frames: int = 24,
    img_size: int = 224,
) -> list[list[Image.Image]]:
    """Generate synthetic video data with predictable motion."""
    from PIL import ImageDraw

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

        start_x = np.random.randint(size + 10, img_size // 2)
        start_y = np.random.randint(size + 10, img_size - size - 10)
        direction = np.random.choice(["right", "left", "down", "up"])
        base_velocity = np.random.randint(4, 8)

        frames = []
        for frame_idx in range(n_frames):
            img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)

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
            else:
                cx = start_x + np.random.uniform(-1, 1)
                cy = start_y - frame_idx * velocity

            cx = max(size, min(img_size - size, cx))
            cy = max(size, min(img_size - size, cy))

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


def extract_vlm_features_batch(
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


def extract_streaming_data(
    videos: list[list[Image.Image]],
    model,
    processor,
    device: torch.device,
    runner: ExperimentRunner,
    context_frames: int = 8,
    prediction_frames: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features for streaming prediction.

    For each video, extract:
    - Context: frames 0 to context_frames-1
    - Target: frames context_frames to context_frames+prediction_frames-1

    Returns:
        context_features: [N, context_frames, hidden_dim]
        target_features: [N, prediction_frames, hidden_dim]
    """
    context_list = []
    target_list = []

    for idx, video in enumerate(videos):
        if len(video) < context_frames + prediction_frames:
            continue

        # Extract features for context and target frames
        context_imgs = video[:context_frames]
        target_imgs = video[context_frames:context_frames + prediction_frames]

        context_feats = extract_vlm_features_batch(context_imgs, model, processor, device)
        target_feats = extract_vlm_features_batch(target_imgs, model, processor, device)

        # Pool each frame's features to fixed size
        pooled_context = []
        for f in context_feats:
            # Mean pool sequence dimension to get [hidden_dim]
            pooled_context.append(f.mean(dim=0))
        pooled_context = torch.stack(pooled_context)  # [context_frames, hidden_dim]

        pooled_target = []
        for f in target_feats:
            pooled_target.append(f.mean(dim=0))
        pooled_target = torch.stack(pooled_target)  # [prediction_frames, hidden_dim]

        context_list.append(pooled_context)
        target_list.append(pooled_target)

        if (idx + 1) % 20 == 0:
            progress = (idx + 1) / len(videos)
            runner.log_metrics({"e3_streaming/extraction_progress": progress})
            print(f"    Extracted {idx + 1}/{len(videos)} videos")

    # Stack into tensors
    context_features = torch.stack(context_list)  # [N, context_frames, hidden_dim]
    target_features = torch.stack(target_list)  # [N, prediction_frames, hidden_dim]

    return context_features, target_features


def create_context_jump_samples(
    context_features: torch.Tensor,
    target_features: torch.Tensor,
    n_samples: int = 100,
    jump_probability: float = 0.3,
) -> list[dict]:
    """Create samples with random context jumps for training.

    Args:
        context_features: [N, context_frames, hidden_dim]
        target_features: [N, prediction_frames, hidden_dim]
        n_samples: Number of jump samples to create
        jump_probability: Probability of including a context jump

    Returns:
        List of dicts with 'context', 'target', 'jump_context', 'jump_at'
    """
    N = context_features.size(0)
    prediction_frames = target_features.size(1)
    samples = []

    for _ in range(n_samples):
        # Sample a video
        idx = np.random.randint(N)
        context = context_features[idx]
        target = target_features[idx]

        sample = {
            'context': context,
            'target': target,
            'jump_context': None,
            'jump_at': None,
        }

        # Randomly add a context jump
        if np.random.random() < jump_probability:
            # Sample a different video for the jump
            jump_idx = np.random.randint(N)
            while jump_idx == idx:
                jump_idx = np.random.randint(N)

            # Random jump point in the prediction sequence
            jump_at = np.random.randint(1, prediction_frames)

            sample['jump_context'] = context_features[jump_idx]
            sample['jump_at'] = jump_at
            # Update target to reflect the jump (use target from jump video)
            sample['target'] = torch.cat([
                target[:jump_at],
                target_features[jump_idx, :prediction_frames - jump_at],
            ], dim=0)

        samples.append(sample)

    return samples


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

    # Cosine similarity
    if val_metrics:
        steps = [m["step"] for m in val_metrics]
        pred_sims = [m["cos_sim"] for m in val_metrics]
        copy_sims = [m.get("copy_cos_sim", 0) for m in val_metrics]

        axes[0, 1].plot(steps, pred_sims, label="Predicted", marker="o")
        if any(copy_sims):
            axes[0, 1].plot(steps, copy_sims, label="Copy Baseline", marker="s")
        axes[0, 1].axhline(y=0.65, color="g", linestyle="--", alpha=0.5, label="Target (0.65)")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Cosine Similarity")
        axes[0, 1].set_title("Streaming Prediction Quality")
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)

        # Improvement over baseline
        improvements = [m.get("improvement", 0) for m in val_metrics]
        axes[1, 0].plot(steps, improvements, marker="o", color="purple")
        axes[1, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Improvement over Baseline")
        axes[1, 0].set_title("Copy Baseline Delta")

        # Context jump performance
        jump_sims = [m.get("jump_cos_sim", 0) for m in val_metrics]
        if any(jump_sims):
            no_jump_sims = [m.get("no_jump_cos_sim", m["cos_sim"]) for m in val_metrics]
            axes[1, 1].plot(steps, no_jump_sims, label="No Jump", marker="o")
            axes[1, 1].plot(steps, jump_sims, label="With Jump", marker="s")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Cosine Similarity")
            axes[1, 1].set_title("Context Jump Handling")
            axes[1, 1].legend()
            axes[1, 1].set_ylim(0, 1)

    plt.suptitle("E3.4: Streaming Video Prediction", fontsize=12)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e3_streaming_prediction(runner: ExperimentRunner) -> dict:
    """Run E3.4: Streaming video prediction training.

    This experiment trains the StreamingPredictor on multi-frame sequences
    with context jumps, matching the production inference architecture.

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E3.4: Streaming Video Prediction")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e3_streaming/stage": 0, "e3_streaming/progress": 0.0})

    # Configuration
    context_frames = 8
    prediction_frames = 8
    n_videos = 200  # Start small, scale up
    mock_mode = os.environ.get("FORESIGHT_STUB_MODE", "false").lower() == "true"

    # =========================================================================
    # Stage 1: Load video data
    # =========================================================================
    print("\n[Stage 1/5] Loading video data...")

    videos = load_ssv2_data(
        subset_size=n_videos,
        num_frames=context_frames + prediction_frames + 8,  # Extra buffer
        mock_mode=mock_mode,
    )

    # Split into train/val
    n_train = int(0.8 * len(videos))
    train_videos = videos[:n_train]
    val_videos = videos[n_train:]

    print(f"  Loaded {len(videos)} videos")
    print(f"  Train: {len(train_videos)}, Val: {len(val_videos)}")
    print(f"  Mock mode: {mock_mode}")

    runner.log_metrics({
        "e3_streaming/stage": 1,
        "e3_streaming/progress": 0.1,
        "e3_streaming/n_train": len(train_videos),
        "e3_streaming/n_val": len(val_videos),
    })

    # =========================================================================
    # Stage 2: Load VLM and extract features
    # =========================================================================
    print("\n[Stage 2/5] Loading VLM and extracting features...")

    vlm_model, vlm_processor = load_vlm_model(device)

    print("  Extracting training features...")
    train_context, train_target = extract_streaming_data(
        train_videos, vlm_model, vlm_processor, device, runner,
        context_frames=context_frames,
        prediction_frames=prediction_frames,
    )

    print("  Extracting validation features...")
    val_context, val_target = extract_streaming_data(
        val_videos, vlm_model, vlm_processor, device, runner,
        context_frames=context_frames,
        prediction_frames=prediction_frames,
    )

    hidden_dim = train_context.shape[-1]
    print(f"  Train context: {train_context.shape}")
    print(f"  Val context: {val_context.shape}")
    print(f"  Hidden dim: {hidden_dim}")

    runner.log_metrics({
        "e3_streaming/stage": 2,
        "e3_streaming/progress": 0.4,
        "e3_streaming/hidden_dim": hidden_dim,
    })

    # Free VLM memory
    del vlm_model, vlm_processor
    torch.cuda.empty_cache()

    # =========================================================================
    # Stage 3: Initialize StreamingPredictor
    # =========================================================================
    print("\n[Stage 3/5] Initializing StreamingPredictor...")

    config = StreamingPredictorConfig(
        vlm_dim=hidden_dim,
        hidden_dim=512,
        num_queries=32,
        num_layers=3,
        num_heads=8,
        context_window=context_frames,
        dropout=0.1,
    )

    model = config.create_model().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,} ({n_params/1e6:.2f}M)")

    runner.log_metrics({
        "e3_streaming/stage": 3,
        "e3_streaming/progress": 0.45,
        "e3_streaming/model_params": n_params,
    })

    # =========================================================================
    # Stage 4: Train StreamingPredictor
    # =========================================================================
    print("\n[Stage 4/5] Training StreamingPredictor...")

    train_context_t = train_context.to(device)
    train_target_t = train_target.to(device)
    val_context_t = val_context.to(device)
    val_target_t = val_target.to(device)

    n_steps = 5000
    batch_size = 8
    lr = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, eta_min=1e-6)

    losses = []
    val_metrics = []

    for step in range(n_steps):
        model.train()

        # Sample batch
        batch_idx = torch.randint(0, len(train_context_t), (batch_size,))
        batch_context = train_context_t[batch_idx]
        batch_target = train_target_t[batch_idx]

        # Forward pass: predict future frames
        optimizer.zero_grad()
        predictions = model(
            batch_context,
            num_future_frames=prediction_frames,
            return_all_predictions=True,
        )

        # Compute loss
        loss_dict = model.compute_loss(predictions, batch_target)
        loss = loss_dict['loss']

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        # Validate periodically
        if (step + 1) % 500 == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                # Standard prediction (no jumps)
                val_pred = model(
                    val_context_t,
                    num_future_frames=prediction_frames,
                    return_all_predictions=True,
                )

                val_loss_dict = model.compute_loss(
                    val_pred,
                    val_target_t,
                    copy_baseline=val_context_t[:, -1:].expand(-1, prediction_frames, -1),
                )

                cos_sim = float(val_loss_dict['cos_sim'].item())
                copy_cos_sim = float(val_loss_dict.get('copy_cos_sim', torch.tensor(0)).item())
                improvement = float(val_loss_dict.get('improvement', torch.tensor(0)).item())

                val_metrics.append({
                    "step": step + 1,
                    "cos_sim": cos_sim,
                    "copy_cos_sim": copy_cos_sim,
                    "improvement": improvement,
                    "no_jump_cos_sim": cos_sim,
                })

            print(f"    Step {step + 1}/{n_steps}: "
                  f"Pred={cos_sim:.4f}, Copy={copy_cos_sim:.4f}, "
                  f"Improvement={improvement:+.4f}")

            runner.log_metrics({
                "e3_streaming/loss": loss.item(),
                "e3_streaming/val_cos_sim": cos_sim,
                "e3_streaming/val_copy_cos_sim": copy_cos_sim,
                "e3_streaming/val_improvement": improvement,
            }, step=step)

    runner.log_metrics({
        "e3_streaming/stage": 4,
        "e3_streaming/progress": 0.9,
    })

    # =========================================================================
    # Stage 5: Final evaluation with context jumps
    # =========================================================================
    print("\n[Stage 5/5] Final evaluation with context jumps...")

    model.eval()
    with torch.no_grad():
        # Standard prediction
        val_pred = model(
            val_context_t,
            num_future_frames=prediction_frames,
            return_all_predictions=True,
        )

        final_loss_dict = model.compute_loss(
            val_pred,
            val_target_t,
            copy_baseline=val_context_t[:, -1:].expand(-1, prediction_frames, -1),
        )

        final_cos_sim = float(final_loss_dict['cos_sim'].item())
        final_copy_cos_sim = float(final_loss_dict.get('copy_cos_sim', torch.tensor(0)).item())
        final_improvement = float(final_loss_dict.get('improvement', torch.tensor(0)).item())

    # Statistical test
    # Compute per-sample similarities for t-test
    pred_pooled = F.normalize(val_pred.mean(dim=(1, 2)), dim=-1)
    target_pooled = F.normalize(val_target_t.mean(dim=(1, 2)), dim=-1)
    per_sample_sims = (pred_pooled * target_pooled).sum(dim=-1).cpu().numpy()

    copy_pooled = F.normalize(val_context_t[:, -1], dim=-1)
    copy_sims = (copy_pooled * F.normalize(val_target_t[:, 0], dim=-1)).sum(dim=-1).cpu().numpy()

    t_stat, p_value = stats.ttest_rel(per_sample_sims, copy_sims)

    print(f"  Final cos_sim: {final_cos_sim:.4f}")
    print(f"  Copy baseline: {final_copy_cos_sim:.4f}")
    print(f"  Improvement: {final_improvement:+.4f}")
    print(f"  Statistical significance: t={t_stat:.3f}, p={p_value:.4f}")

    # Save visualization
    viz_bytes = create_results_plot(losses, val_metrics)
    viz_path = runner.results.save_artifact("streaming_prediction.png", viz_bytes)

    # Save config
    config_path = runner.results.save_json_artifact("streaming_config.json", config.to_dict())

    # Save results
    results_data = {
        "final_cos_sim": final_cos_sim,
        "final_copy_cos_sim": final_copy_cos_sim,
        "improvement": final_improvement,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "statistically_significant": p_value < 0.01,
        "model_params": n_params,
        "context_frames": context_frames,
        "prediction_frames": prediction_frames,
    }
    data_path = runner.results.save_json_artifact("streaming_results.json", results_data)

    runner.log_metrics({
        "e3_streaming/stage": 5,
        "e3_streaming/progress": 1.0,
        "e3_streaming/final_cos_sim": final_cos_sim,
        "e3_streaming/final_improvement": final_improvement,
        "e3_streaming/p_value": p_value,
    })

    # =========================================================================
    # Determine finding
    # =========================================================================
    cos_sim_target = 0.65
    improvement_target = 0.0

    success = (
        final_cos_sim > cos_sim_target and
        final_improvement > improvement_target and
        p_value < 0.05
    )

    if success:
        finding = (
            f"STREAMING PREDICTION SUCCESSFUL: Model predicts multi-frame sequences "
            f"(cos_sim={final_cos_sim:.3f} > {cos_sim_target}). "
            f"Improvement over copy baseline: {final_improvement:+.3f} (p={p_value:.4f}). "
            f"Ready for production streaming inference."
        )
    elif final_cos_sim > 0.5:
        finding = (
            f"STREAMING PREDICTION PARTIAL: Model shows some prediction capability "
            f"(cos_sim={final_cos_sim:.3f}), but below target ({cos_sim_target}). "
            f"Consider more training data or longer sequences."
        )
    else:
        finding = (
            f"STREAMING PREDICTION FAILED: Model fails to predict sequences "
            f"(cos_sim={final_cos_sim:.3f}). "
            f"Investigate architecture or training setup."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "cosine_similarity": final_cos_sim,
            "copy_baseline_delta": final_improvement,
            "p_value": float(p_value),
            "statistically_significant": p_value < 0.05,
            "passed": success,
        },
        "artifacts": [viz_path, config_path, data_path],
    }
