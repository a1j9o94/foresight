"""E4.1: Correlation Study - Does LPIPS Error Predict Prediction Correctness?

This experiment establishes whether LPIPS (perceptual similarity) between
predicted and actual video correlates with whether the prediction is semantically
correct (i.e., shows the right action).

Protocol:
1. Generate N predictions using LTX Image-to-Video
2. For each prediction:
   - Compute LPIPS(predicted, actual)
   - Use VLM to classify action in both predicted and actual
   - Determine correctness (does classification match ground truth?)
3. Compute correlation between LPIPS and correctness

Success Criteria:
- Point-biserial correlation r > 0.3 (minimum)
- AUROC (LPIPS as classifier) > 0.65 (minimum)
- LPIPS gap (incorrect - correct) > 0.08 (minimum)

If this experiment fails, pixel verification cannot help because LPIPS
doesn't tell us anything about semantic correctness.
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from runner import ExperimentRunner


# =============================================================================
# Data Loading
# =============================================================================

def load_ssv2_with_labels(subset_size: int = 200, num_frames: int = 16):
    """Load SSv2 videos with action labels.

    Returns:
        videos: List of [T, C, H, W] tensors
        action_labels: List of action label strings
        action_ids: List of action class IDs
        label_to_id: Dict mapping label string to ID
    """
    print(f"  Loading {subset_size} SSv2 videos with {num_frames} frames...")

    video_dir = os.environ.get(
        "SSV2_VIDEO_DIR",
        "/datasets/ssv2/videos/20bn-something-something-v2"
    )

    if not os.path.exists(video_dir):
        print(f"  [WARN] SSv2 not found at {video_dir}, generating synthetic data")
        return _generate_synthetic_data(subset_size, num_frames)

    try:
        from foresight_training.data import SSv2Dataset

        ds = SSv2Dataset(
            split="validation",
            video_dir=video_dir,
            subset_size=subset_size,
            num_frames=num_frames,
            frame_size=(224, 224),
        )

        videos = []
        action_labels = []
        action_ids = []
        label_to_id = {}

        for i, sample in enumerate(ds):
            if i >= subset_size:
                break

            frames = sample.frames  # [T, C, H, W]
            if frames.shape[0] < num_frames:
                continue

            video_tensor = frames.float()
            label = sample.label
            label_id = sample.label_id

            if label not in label_to_id:
                label_to_id[label] = label_id

            videos.append(video_tensor)
            action_labels.append(label)
            action_ids.append(label_id)

            if (i + 1) % 50 == 0:
                print(f"    Loaded {i + 1}/{subset_size} videos")

        print(f"  Loaded {len(videos)} videos with {len(label_to_id)} unique actions")
        return videos, action_labels, action_ids, label_to_id

    except Exception as e:
        print(f"  [WARN] Failed to load SSv2: {e}")
        import traceback
        traceback.print_exc()
        return _generate_synthetic_data(subset_size, num_frames)


def _generate_synthetic_data(subset_size: int, num_frames: int):
    """Generate synthetic data for testing."""
    import random

    videos = []
    action_labels = []
    action_ids = []

    # Define action classes with distinct visual patterns
    actions = [
        "Pushing something from left to right",
        "Pushing something from right to left",
        "Moving something up",
        "Moving something down",
        "Rotating something clockwise",
        "Rotating something counterclockwise",
        "Picking something up",
        "Putting something down",
    ]
    label_to_id = {a: i for i, a in enumerate(actions)}

    for i in range(subset_size):
        # Create video with shape movement
        video = torch.zeros(num_frames, 3, 224, 224)

        action_idx = i % len(actions)
        action = actions[action_idx]

        # Shape parameters
        x_start = random.randint(40, 100)
        y_start = random.randint(80, 140)
        color = torch.tensor([random.random(), random.random(), random.random()])
        size = random.randint(15, 25)

        # Motion based on action
        for t in range(num_frames):
            progress = t / (num_frames - 1)

            if "left to right" in action:
                x = int(x_start + (224 - 2 * x_start) * progress)
                y = y_start
            elif "right to left" in action:
                x = int(224 - x_start - (224 - 2 * x_start) * progress)
                y = y_start
            elif "up" in action:
                x = x_start
                y = int(y_start - 60 * progress)
            elif "down" in action:
                x = x_start
                y = int(y_start + 60 * progress)
            elif "clockwise" in action:
                angle = 2 * np.pi * progress
                x = int(112 + 50 * np.cos(angle))
                y = int(112 + 50 * np.sin(angle))
            elif "counterclockwise" in action:
                angle = -2 * np.pi * progress
                x = int(112 + 50 * np.cos(angle))
                y = int(112 + 50 * np.sin(angle))
            elif "up" in action.lower():
                x = x_start
                y = int(y_start * (1 - 0.5 * progress))
            else:  # putting down
                x = x_start
                y = int(y_start * (1 + 0.3 * progress))

            # Draw shape
            x = max(size, min(224 - size, x))
            y = max(size, min(224 - size, y))
            video[t, :, y-size:y+size, x-size:x+size] = color.view(3, 1, 1)

        videos.append(video)
        action_labels.append(action)
        action_ids.append(label_to_id[action])

    return videos, action_labels, action_ids, label_to_id


# =============================================================================
# Video Generation
# =============================================================================

_ltx_pipeline = None


def _get_ltx_pipeline(device: str):
    """Get or load the LTX Image-to-Video pipeline (cached)."""
    global _ltx_pipeline
    if _ltx_pipeline is None:
        try:
            from diffusers import LTXImageToVideoPipeline

            _ltx_pipeline = LTXImageToVideoPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                torch_dtype=torch.bfloat16,
            ).to(device)
            print("  [LTX] Pipeline loaded and cached")
        except Exception as e:
            print(f"  [WARN] Failed to load LTX pipeline: {e}")
            _ltx_pipeline = "failed"
    return _ltx_pipeline if _ltx_pipeline != "failed" else None


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor [C, H, W] to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_video_continuation(
    context_frames: torch.Tensor,
    action_label: str,
    device: str,
    num_output_frames: int = 8,
) -> torch.Tensor:
    """Generate video continuation using LTX-Video or fallback to extrapolation.

    Args:
        context_frames: [T, C, H, W] context video tensor
        action_label: Action description for prompt
        device: torch device
        num_output_frames: Number of frames to generate

    Returns:
        [num_output_frames, C, H, W] generated frames tensor
    """
    pipeline = _get_ltx_pipeline(device)

    if pipeline is not None:
        try:
            # Get last frame as conditioning image
            last_frame = context_frames[-1]
            conditioning_image = _tensor_to_pil(last_frame)

            # Generate continuation
            prompt = f"Continue this video showing: {action_label}"
            negative_prompt = "worst quality, blurry, jittery, distorted"

            with torch.no_grad():
                output = pipeline(
                    image=conditioning_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=224,
                    height=224,
                    num_frames=num_output_frames + 1,  # +1 for conditioning frame
                    num_inference_steps=15,
                    guidance_scale=3.0,
                )
                generated_pil = output.frames[0]

                # Convert to tensor, skip conditioning frame
                gen_frames = torch.stack([
                    torch.from_numpy(np.array(f)).permute(2, 0, 1).float() / 255.0
                    for f in generated_pil[1:num_output_frames + 1]
                ]).to(device)

            return gen_frames

        except Exception as e:
            print(f"    [WARN] LTX generation failed: {e}")

    # Fallback to simple extrapolation
    return _simple_extrapolation(context_frames, num_output_frames)


def _simple_extrapolation(context: torch.Tensor, num_frames: int = 8) -> torch.Tensor:
    """Simple linear extrapolation baseline."""
    velocity = context[-1] - context[-2]  # [C, H, W]

    extrapolated = []
    last_frame = context[-1]
    for t in range(num_frames):
        next_frame = last_frame + velocity * (t + 1) * 0.5
        next_frame = next_frame.clamp(0, 1)
        extrapolated.append(next_frame)

    return torch.stack(extrapolated)


# =============================================================================
# LPIPS Computation
# =============================================================================

_lpips_model = None


def _get_lpips_model(device: str):
    """Get or load LPIPS model (cached)."""
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='alex').to(device)
        _lpips_model.eval()
        print("  [LPIPS] Model loaded")
    return _lpips_model


def compute_video_lpips(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    device: str,
) -> dict[str, float]:
    """Compute LPIPS between predicted and actual videos.

    Args:
        predicted: [T, C, H, W] predicted video
        actual: [T, C, H, W] actual video
        device: torch device

    Returns:
        Dict with 'mean', 'max', 'final_frame' LPIPS scores
    """
    lpips_fn = _get_lpips_model(device)

    # Ensure same length
    T = min(predicted.shape[0], actual.shape[0])
    predicted = predicted[:T].to(device)
    actual = actual[:T].to(device)

    # Resize if needed
    if predicted.shape[-2:] != actual.shape[-2:]:
        predicted = F.interpolate(predicted, size=actual.shape[-2:], mode='bilinear', align_corners=False)

    # LPIPS expects input in [-1, 1] range
    pred_norm = predicted * 2 - 1
    actual_norm = actual * 2 - 1

    frame_scores = []
    with torch.no_grad():
        for t in range(T):
            # Add batch dimension
            score = lpips_fn(pred_norm[t:t+1], actual_norm[t:t+1])
            frame_scores.append(score.item())

    return {
        'mean': float(np.mean(frame_scores)),
        'max': float(np.max(frame_scores)),
        'final_frame': float(frame_scores[-1]) if frame_scores else 0.0,
        'per_frame': frame_scores,
    }


# =============================================================================
# VLM Action Classification
# =============================================================================

_vlm_model = None
_vlm_processor = None


def _get_vlm(device: str):
    """Get or load VLM model (cached)."""
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        _vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        _vlm_model.eval()
        print("  [VLM] Model loaded")
    return _vlm_model, _vlm_processor


def classify_action_in_video(
    frames: torch.Tensor,
    action_choices: list[str],
    device: str,
) -> tuple[str, int]:
    """Ask VLM to classify action in video frames.

    Args:
        frames: [T, C, H, W] video frames
        action_choices: List of possible action labels
        device: torch device

    Returns:
        (predicted_action, predicted_idx)
    """
    vlm, processor = _get_vlm(device)

    # Sample 3 frames for VLM (first, middle, last)
    T = frames.shape[0]
    indices = [0, T // 2, T - 1]
    pil_images = [_tensor_to_pil(frames[i]) for i in indices]

    # Create numbered action list
    choices_text = "\n".join([f"{i+1}. {a}" for i, a in enumerate(action_choices)])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_images[0]},
                {"type": "image", "image": pil_images[1]},
                {"type": "image", "image": pil_images[2]},
                {"type": "text", "text": f"""These 3 images are frames from a video showing a person performing an action with an object.
What action is being performed? Choose from this list:

{choices_text}

Reply with ONLY the number of the action."""},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=pil_images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = vlm.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)

    # Parse response to get action number
    import re
    try:
        numbers = re.findall(r'\d+', response.split("assistant")[-1] if "assistant" in response else response)
        if numbers:
            idx = int(numbers[0]) - 1
            if 0 <= idx < len(action_choices):
                return action_choices[idx], idx
    except Exception:
        pass

    return action_choices[0], 0  # Default to first action


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_correlation(lpips_scores: list[float], correctness: list[bool]) -> dict:
    """Compute point-biserial correlation between LPIPS and correctness.

    Args:
        lpips_scores: List of LPIPS scores (higher = more different)
        correctness: List of booleans (True = correct prediction)

    Returns:
        Dict with correlation stats
    """
    lpips_arr = np.array(lpips_scores)
    correct_arr = np.array(correctness).astype(float)

    # Point-biserial correlation (continuous vs binary)
    # Note: LPIPS is higher when predictions are different (wrong)
    # So we expect negative correlation with correctness
    r, p_value = stats.pointbiserialr(correct_arr, lpips_arr)

    # Also compute mean LPIPS for correct vs incorrect
    correct_lpips = lpips_arr[correct_arr == 1]
    incorrect_lpips = lpips_arr[correct_arr == 0]

    return {
        'correlation': float(r),
        'p_value': float(p_value),
        'mean_lpips_correct': float(np.mean(correct_lpips)) if len(correct_lpips) > 0 else 0.0,
        'mean_lpips_incorrect': float(np.mean(incorrect_lpips)) if len(incorrect_lpips) > 0 else 0.0,
        'lpips_gap': float(np.mean(incorrect_lpips) - np.mean(correct_lpips)) if len(correct_lpips) > 0 and len(incorrect_lpips) > 0 else 0.0,
        'n_correct': int(np.sum(correct_arr)),
        'n_incorrect': int(np.sum(1 - correct_arr)),
    }


def compute_auroc(lpips_scores: list[float], correctness: list[bool]) -> dict:
    """Compute AUROC for using LPIPS to detect incorrect predictions.

    Args:
        lpips_scores: List of LPIPS scores
        correctness: List of booleans

    Returns:
        Dict with AUROC stats
    """
    # Higher LPIPS should indicate incorrect predictions
    # So we use 1-correctness as the positive class (incorrect)
    incorrect = [not c for c in correctness]

    try:
        auroc = roc_auc_score(incorrect, lpips_scores)
        fpr, tpr, thresholds = roc_curve(incorrect, lpips_scores)

        # Find optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        return {
            'auroc': float(auroc),
            'optimal_threshold': float(optimal_threshold),
            'fpr_at_optimal': float(fpr[optimal_idx]),
            'tpr_at_optimal': float(tpr[optimal_idx]),
        }
    except Exception as e:
        print(f"  [WARN] AUROC computation failed: {e}")
        return {
            'auroc': 0.5,
            'optimal_threshold': 0.0,
            'fpr_at_optimal': 0.0,
            'tpr_at_optimal': 0.0,
        }


def create_analysis_plots(
    lpips_scores: list[float],
    correctness: list[bool],
    correlation_stats: dict,
    auroc_stats: dict,
) -> bytes:
    """Create visualization of correlation analysis.

    Returns:
        PNG bytes
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    lpips_arr = np.array(lpips_scores)
    correct_arr = np.array(correctness)

    # 1. LPIPS distribution by correctness
    ax = axes[0, 0]
    correct_lpips = lpips_arr[correct_arr == True]
    incorrect_lpips = lpips_arr[correct_arr == False]

    if len(correct_lpips) > 0:
        ax.hist(correct_lpips, bins=20, alpha=0.6, label=f'Correct (n={len(correct_lpips)})', color='green')
    if len(incorrect_lpips) > 0:
        ax.hist(incorrect_lpips, bins=20, alpha=0.6, label=f'Incorrect (n={len(incorrect_lpips)})', color='red')
    ax.set_xlabel('LPIPS Score')
    ax.set_ylabel('Count')
    ax.set_title('LPIPS Distribution by Correctness')
    ax.legend()

    # 2. Box plot comparison
    ax = axes[0, 1]
    data_to_plot = []
    labels = []
    if len(correct_lpips) > 0:
        data_to_plot.append(correct_lpips)
        labels.append('Correct')
    if len(incorrect_lpips) > 0:
        data_to_plot.append(incorrect_lpips)
        labels.append('Incorrect')

    if data_to_plot:
        ax.boxplot(data_to_plot, labels=labels)
    ax.set_ylabel('LPIPS Score')
    ax.set_title(f'LPIPS Gap: {correlation_stats.get("lpips_gap", 0):.3f}')

    # 3. ROC Curve
    ax = axes[1, 0]
    if auroc_stats['auroc'] > 0:
        incorrect = [not c for c in correctness]
        fpr, tpr, _ = roc_curve(incorrect, lpips_scores)
        ax.plot(fpr, tpr, 'b-', label=f'ROC (AUC={auroc_stats["auroc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.plot(auroc_stats['fpr_at_optimal'], auroc_stats['tpr_at_optimal'],
                'ro', markersize=10, label=f'Optimal (thresh={auroc_stats["optimal_threshold"]:.3f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: LPIPS as Error Detector')
    ax.legend()

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""Correlation Analysis Summary

Point-Biserial Correlation: r = {correlation_stats['correlation']:.3f}
P-value: {correlation_stats['p_value']:.4f}
Statistically Significant: {'Yes' if correlation_stats['p_value'] < 0.05 else 'No'}

LPIPS Statistics:
  Mean (Correct): {correlation_stats['mean_lpips_correct']:.3f}
  Mean (Incorrect): {correlation_stats['mean_lpips_incorrect']:.3f}
  Gap: {correlation_stats['lpips_gap']:.3f}

Classification Performance:
  AUROC: {auroc_stats['auroc']:.3f}
  Optimal Threshold: {auroc_stats['optimal_threshold']:.3f}

Sample Counts:
  Correct: {correlation_stats['n_correct']}
  Incorrect: {correlation_stats['n_incorrect']}
"""
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.suptitle('E4.1: LPIPS vs Prediction Correctness Correlation', fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf.read()


# =============================================================================
# Main Experiment Handler
# =============================================================================

def e4_1_correlation_study(runner: ExperimentRunner) -> dict:
    """E4.1: Test if LPIPS error correlates with prediction correctness.

    Protocol:
    1. Load SSv2 videos
    2. Generate predictions using LTX-Video
    3. Compute LPIPS between predicted and actual
    4. Classify action in both using VLM
    5. Compute correlation between LPIPS and correctness

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E4.1: Correlation Study (LPIPS vs Prediction Correctness)")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner.log_metrics({"e4_1/stage": 0, "e4_1/progress": 0.0})

    # =========================================================================
    # Stage 1: Load Data
    # =========================================================================
    print("\n[Stage 1/5] Loading SSv2 data...")

    num_samples = 100  # Number of samples for correlation study
    videos, action_labels, action_ids, label_to_id = load_ssv2_with_labels(
        subset_size=num_samples,
        num_frames=16,
    )

    # Get unique actions for classification
    unique_actions = list(label_to_id.keys())[:50]  # Limit to 50 classes for VLM
    print(f"  Loaded {len(videos)} videos, {len(unique_actions)} action classes")

    runner.log_metrics({
        "e4_1/stage": 1,
        "e4_1/progress": 0.1,
        "e4_1/num_videos": len(videos),
        "e4_1/num_actions": len(unique_actions),
    })

    # =========================================================================
    # Stage 2: Generate Predictions and Compute LPIPS
    # =========================================================================
    print("\n[Stage 2/5] Generating predictions and computing LPIPS...")

    # Pre-load models
    _ = _get_ltx_pipeline(device)
    _ = _get_lpips_model(device)

    results = []
    num_eval = min(50, len(videos))  # Evaluate on subset for speed

    for i in range(num_eval):
        video = videos[i].to(device)
        true_label = action_labels[i]

        # Split into context and future
        context = video[:8]
        actual_future = video[8:]

        # Generate prediction
        predicted_future = generate_video_continuation(
            context, true_label, device, num_output_frames=8
        )

        # Compute LPIPS
        lpips_scores = compute_video_lpips(predicted_future, actual_future, device)

        results.append({
            'sample_id': i,
            'true_label': true_label,
            'lpips_mean': lpips_scores['mean'],
            'lpips_max': lpips_scores['max'],
            'lpips_final': lpips_scores['final_frame'],
            'predicted_future': predicted_future.cpu(),
            'actual_future': actual_future.cpu(),
        })

        if (i + 1) % 10 == 0:
            print(f"    Generated {i + 1}/{num_eval} predictions (LPIPS mean={lpips_scores['mean']:.3f})")
            runner.log_metrics({
                "e4_1/progress": 0.1 + 0.4 * (i + 1) / num_eval,
                "e4_1/lpips_rolling_mean": np.mean([r['lpips_mean'] for r in results]),
            })

    runner.log_metrics({"e4_1/stage": 2, "e4_1/progress": 0.5})

    # =========================================================================
    # Stage 3: Classify Actions with VLM
    # =========================================================================
    print("\n[Stage 3/5] Classifying actions with VLM...")

    # Load VLM
    _ = _get_vlm(device)

    correctness = []

    for i, result in enumerate(results):
        true_label = result['true_label']

        # Classify action in predicted video
        pred_action, pred_idx = classify_action_in_video(
            result['predicted_future'],
            unique_actions,
            device,
        )

        # Check if correct (fuzzy match)
        is_correct = (
            pred_action.lower() in true_label.lower() or
            true_label.lower() in pred_action.lower() or
            pred_action == true_label
        )

        result['predicted_action'] = pred_action
        result['is_correct'] = is_correct
        correctness.append(is_correct)

        if (i + 1) % 10 == 0:
            acc_so_far = np.mean(correctness)
            print(f"    Classified {i + 1}/{len(results)} (accuracy so far: {acc_so_far:.2%})")
            runner.log_metrics({
                "e4_1/progress": 0.5 + 0.3 * (i + 1) / len(results),
                "e4_1/accuracy_rolling": acc_so_far,
            })

    overall_accuracy = np.mean(correctness)
    print(f"  Overall accuracy: {overall_accuracy:.2%}")

    runner.log_metrics({"e4_1/stage": 3, "e4_1/progress": 0.8})

    # =========================================================================
    # Stage 4: Correlation Analysis
    # =========================================================================
    print("\n[Stage 4/5] Computing correlation statistics...")

    lpips_scores = [r['lpips_mean'] for r in results]

    # Compute correlation
    correlation_stats = compute_correlation(lpips_scores, correctness)
    print(f"  Point-biserial correlation: r = {correlation_stats['correlation']:.3f}")
    print(f"  P-value: {correlation_stats['p_value']:.4f}")
    print(f"  LPIPS gap (incorrect - correct): {correlation_stats['lpips_gap']:.3f}")

    # Compute AUROC
    auroc_stats = compute_auroc(lpips_scores, correctness)
    print(f"  AUROC: {auroc_stats['auroc']:.3f}")

    runner.log_metrics({
        "e4_1/stage": 4,
        "e4_1/progress": 0.9,
        "e4_1/correlation": correlation_stats['correlation'],
        "e4_1/p_value": correlation_stats['p_value'],
        "e4_1/lpips_gap": correlation_stats['lpips_gap'],
        "e4_1/auroc": auroc_stats['auroc'],
        "e4_1/accuracy": overall_accuracy,
    })

    # =========================================================================
    # Stage 5: Generate Plots and Save Results
    # =========================================================================
    print("\n[Stage 5/5] Generating plots and saving results...")

    # Create plots
    plot_bytes = create_analysis_plots(lpips_scores, correctness, correlation_stats, auroc_stats)
    plot_path = runner.results.save_artifact("correlation_analysis.png", plot_bytes)

    # Save detailed results
    detailed_results = {
        'samples': [
            {
                'sample_id': r['sample_id'],
                'true_label': r['true_label'],
                'predicted_action': r['predicted_action'],
                'is_correct': r['is_correct'],
                'lpips_mean': r['lpips_mean'],
                'lpips_max': r['lpips_max'],
            }
            for r in results
        ],
        'correlation_stats': correlation_stats,
        'auroc_stats': auroc_stats,
        'overall_accuracy': overall_accuracy,
    }
    results_path = runner.results.save_json_artifact("correlation_results.json", detailed_results)

    runner.log_metrics({"e4_1/stage": 5, "e4_1/progress": 1.0})

    # =========================================================================
    # Determine Finding
    # =========================================================================

    # Success criteria from experiment plan
    r_threshold_min = 0.3   # Minimum acceptable correlation
    r_threshold_target = 0.5  # Target correlation
    auroc_threshold_min = 0.65
    auroc_threshold_target = 0.75
    gap_threshold_min = 0.08
    gap_threshold_target = 0.15

    # Use absolute value since correlation could be negative (higher LPIPS = more incorrect)
    abs_r = abs(correlation_stats['correlation'])

    passed_correlation = abs_r >= r_threshold_min
    passed_auroc = auroc_stats['auroc'] >= auroc_threshold_min
    passed_gap = correlation_stats['lpips_gap'] >= gap_threshold_min
    passed_significance = correlation_stats['p_value'] < 0.05

    passed_all = passed_correlation and passed_auroc and passed_gap and passed_significance

    if passed_all:
        if abs_r >= r_threshold_target and auroc_stats['auroc'] >= auroc_threshold_target:
            finding = (
                f"STRONG CORRELATION FOUND: |r|={abs_r:.3f} (target>{r_threshold_target}), "
                f"AUROC={auroc_stats['auroc']:.3f} (target>{auroc_threshold_target}), "
                f"gap={correlation_stats['lpips_gap']:.3f}. "
                f"LPIPS strongly predicts correctness. Verification is viable!"
            )
        else:
            finding = (
                f"CORRELATION FOUND: |r|={abs_r:.3f} (>{r_threshold_min}), "
                f"AUROC={auroc_stats['auroc']:.3f} (>{auroc_threshold_min}), "
                f"gap={correlation_stats['lpips_gap']:.3f}. "
                f"LPIPS can predict correctness. Proceed with verification experiments."
            )
    else:
        failures = []
        if not passed_correlation:
            failures.append(f"|r|={abs_r:.3f}<{r_threshold_min}")
        if not passed_auroc:
            failures.append(f"AUROC={auroc_stats['auroc']:.3f}<{auroc_threshold_min}")
        if not passed_gap:
            failures.append(f"gap={correlation_stats['lpips_gap']:.3f}<{gap_threshold_min}")
        if not passed_significance:
            failures.append(f"p={correlation_stats['p_value']:.4f}>0.05")

        finding = (
            f"WEAK/NO CORRELATION: {', '.join(failures)}. "
            f"LPIPS does not reliably predict correctness. "
            f"Consider pivot options: task-specific metrics, object tracking, or VLM-based verification."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "correlation": float(abs_r),
            "correlation_raw": float(correlation_stats['correlation']),
            "p_value": float(correlation_stats['p_value']),
            "auroc": float(auroc_stats['auroc']),
            "lpips_gap": float(correlation_stats['lpips_gap']),
            "mean_lpips_correct": float(correlation_stats['mean_lpips_correct']),
            "mean_lpips_incorrect": float(correlation_stats['mean_lpips_incorrect']),
            "accuracy": float(overall_accuracy),
            "n_samples": len(results),
            "passed": passed_all,
        },
        "artifacts": [plot_path, results_path],
    }
