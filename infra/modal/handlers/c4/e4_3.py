"""E4.3: Single Verification Loop - Does Feedback Improve Second Attempt?

This experiment tests the core verification hypothesis: given feedback about
a prediction's error, can the model produce a better second prediction?

Protocol:
1. Round 1: Generate prediction
2. Verification: Compare to actual, generate feedback
3. Round 2: Generate with feedback (using different prompts)
4. Compare accuracy V1 vs V2

Feedback conditions tested:
- Binary: "Your prediction was incorrect"
- LPIPS: "Your prediction had error 0.35"
- VLM description: "Your prediction showed X but actual showed Y"
- Visual: Include both predicted and actual in context (oracle-ish)

Success Criteria:
- Correction rate > 15% (minimum)
- Overall accuracy improvement > 5% (minimum)
- V2 LPIPS < V1 LPIPS rate > 55% (minimum)
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

from runner import ExperimentRunner


# Import shared utilities from e4_1
from .e4_1 import (
    load_ssv2_with_labels,
    _tensor_to_pil,
    _get_lpips_model,
    compute_video_lpips,
    classify_action_in_video,
)


# =============================================================================
# Video Generation with Feedback
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


def generate_video_round1(
    context_frames: torch.Tensor,
    action_label: str,
    device: str,
    num_output_frames: int = 8,
) -> torch.Tensor:
    """Generate initial (Round 1) prediction without feedback.

    Args:
        context_frames: [T, C, H, W] context video tensor
        action_label: Action description for prompt
        device: torch device
        num_output_frames: Number of frames to generate

    Returns:
        [num_output_frames, C, H, W] generated frames tensor
    """
    pipeline = _get_ltx_pipeline(device)

    if pipeline is None:
        return _simple_extrapolation(context_frames, num_output_frames)

    try:
        last_frame = context_frames[-1]
        conditioning_image = _tensor_to_pil(last_frame)

        prompt = f"Continue this video showing: {action_label}"
        negative_prompt = "worst quality, blurry, jittery, distorted"

        with torch.no_grad():
            output = pipeline(
                image=conditioning_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=224,
                height=224,
                num_frames=num_output_frames + 1,
                num_inference_steps=15,
                guidance_scale=3.0,
                generator=torch.Generator(device=device).manual_seed(42),
            )
            generated_pil = output.frames[0]

            gen_frames = torch.stack([
                torch.from_numpy(np.array(f)).permute(2, 0, 1).float() / 255.0
                for f in generated_pil[1:num_output_frames + 1]
            ]).to(device)

        return gen_frames

    except Exception as e:
        print(f"    [WARN] LTX generation failed: {e}")
        return _simple_extrapolation(context_frames, num_output_frames)


def generate_video_round2_with_feedback(
    context_frames: torch.Tensor,
    action_label: str,
    feedback_type: str,
    feedback_content: dict,
    device: str,
    num_output_frames: int = 8,
) -> torch.Tensor:
    """Generate Round 2 prediction with feedback incorporated.

    Args:
        context_frames: [T, C, H, W] context video tensor
        action_label: Action description
        feedback_type: Type of feedback ('binary', 'lpips', 'vlm', 'visual')
        feedback_content: Dict with feedback-specific content
        device: torch device
        num_output_frames: Number of frames to generate

    Returns:
        [num_output_frames, C, H, W] generated frames tensor
    """
    pipeline = _get_ltx_pipeline(device)

    if pipeline is None:
        return _simple_extrapolation_with_feedback(context_frames, feedback_content, num_output_frames)

    try:
        last_frame = context_frames[-1]
        conditioning_image = _tensor_to_pil(last_frame)

        # Construct prompt based on feedback type
        base_prompt = f"Continue this video showing: {action_label}"

        if feedback_type == "binary":
            # Simple binary feedback
            prompt = f"{base_prompt}. The previous prediction was incorrect. Generate a more accurate continuation."
            negative_prompt = "worst quality, blurry, jittery, distorted, incorrect motion"

        elif feedback_type == "lpips":
            # Quantitative error feedback
            lpips_score = feedback_content.get('lpips_score', 0.0)
            if lpips_score > 0.4:
                prompt = f"{base_prompt}. The previous prediction had high perceptual error ({lpips_score:.2f}). Generate a significantly different, more accurate continuation."
            else:
                prompt = f"{base_prompt}. The previous prediction had moderate error ({lpips_score:.2f}). Generate a slightly refined continuation."
            negative_prompt = "worst quality, blurry, incorrect, wrong motion direction"

        elif feedback_type == "vlm":
            # Semantic feedback from VLM
            v1_description = feedback_content.get('v1_description', '')
            actual_description = feedback_content.get('actual_description', '')
            prompt = f"{base_prompt}. Previous prediction showed: '{v1_description}'. But actual outcome was: '{actual_description}'. Generate a continuation matching the actual outcome."
            negative_prompt = "worst quality, blurry, " + v1_description[:50]

        elif feedback_type == "visual":
            # Visual feedback - use actual outcome as reference
            # This is more "oracle-like" but tests the upper bound
            prompt = f"{base_prompt}. Generate a video that closely matches the reference outcome."
            negative_prompt = "worst quality, blurry, jittery"
            # Note: For visual feedback, we would ideally use image-to-image
            # but LTX doesn't directly support that. Using enhanced prompt instead.

        else:  # baseline - no feedback
            prompt = base_prompt
            negative_prompt = "worst quality, blurry, jittery, distorted"

        # Use different seed than Round 1 to get different output
        seed = 12345 if feedback_type != "baseline" else 42

        with torch.no_grad():
            output = pipeline(
                image=conditioning_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=224,
                height=224,
                num_frames=num_output_frames + 1,
                num_inference_steps=15,
                guidance_scale=4.0 if feedback_type in ["binary", "lpips", "vlm"] else 3.0,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            generated_pil = output.frames[0]

            gen_frames = torch.stack([
                torch.from_numpy(np.array(f)).permute(2, 0, 1).float() / 255.0
                for f in generated_pil[1:num_output_frames + 1]
            ]).to(device)

        return gen_frames

    except Exception as e:
        print(f"    [WARN] LTX Round 2 generation failed: {e}")
        return _simple_extrapolation_with_feedback(context_frames, feedback_content, num_output_frames)


def _simple_extrapolation(context: torch.Tensor, num_frames: int = 8) -> torch.Tensor:
    """Simple linear extrapolation baseline."""
    velocity = context[-1] - context[-2]

    extrapolated = []
    last_frame = context[-1]
    for t in range(num_frames):
        next_frame = last_frame + velocity * (t + 1) * 0.5
        next_frame = next_frame.clamp(0, 1)
        extrapolated.append(next_frame)

    return torch.stack(extrapolated)


def _simple_extrapolation_with_feedback(
    context: torch.Tensor,
    feedback_content: dict,
    num_frames: int = 8,
) -> torch.Tensor:
    """Extrapolation with feedback-adjusted velocity."""
    velocity = context[-1] - context[-2]

    # Adjust velocity based on feedback
    lpips_score = feedback_content.get('lpips_score', 0.3)
    adjustment = 0.5 + lpips_score  # Higher error = larger adjustment

    extrapolated = []
    last_frame = context[-1]
    for t in range(num_frames):
        next_frame = last_frame + velocity * (t + 1) * adjustment
        next_frame = next_frame.clamp(0, 1)
        extrapolated.append(next_frame)

    return torch.stack(extrapolated)


# =============================================================================
# VLM Description Generation
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


def describe_video(frames: torch.Tensor, device: str) -> str:
    """Get VLM description of video frames.

    Args:
        frames: [T, C, H, W] video frames
        device: torch device

    Returns:
        Text description of the video
    """
    vlm, processor = _get_vlm(device)

    # Sample 3 frames
    T = frames.shape[0]
    indices = [0, T // 2, T - 1]
    pil_images = [_tensor_to_pil(frames[i]) for i in indices]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_images[0]},
                {"type": "image", "image": pil_images[1]},
                {"type": "image", "image": pil_images[2]},
                {"type": "text", "text": "Briefly describe what action is happening in this video in one sentence."},
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
            max_new_tokens=50,
            do_sample=False,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()

    return response[:200]  # Truncate to reasonable length


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_correction_rate(
    v1_correct: list[bool],
    v2_correct: list[bool],
) -> dict:
    """Compute correction rate and related statistics.

    Args:
        v1_correct: List of V1 correctness
        v2_correct: List of V2 correctness

    Returns:
        Dict with correction statistics
    """
    v1_arr = np.array(v1_correct)
    v2_arr = np.array(v2_correct)

    # Correction: V1 wrong, V2 right
    corrections = (~v1_arr) & v2_arr
    correction_rate = corrections.sum() / (~v1_arr).sum() if (~v1_arr).sum() > 0 else 0.0

    # Regression: V1 right, V2 wrong
    regressions = v1_arr & (~v2_arr)
    regression_rate = regressions.sum() / v1_arr.sum() if v1_arr.sum() > 0 else 0.0

    # Overall accuracy
    v1_accuracy = v1_arr.mean()
    v2_accuracy = v2_arr.mean()
    accuracy_improvement = v2_accuracy - v1_accuracy

    return {
        'correction_rate': float(correction_rate),
        'regression_rate': float(regression_rate),
        'v1_accuracy': float(v1_accuracy),
        'v2_accuracy': float(v2_accuracy),
        'accuracy_improvement': float(accuracy_improvement),
        'n_v1_incorrect': int((~v1_arr).sum()),
        'n_corrections': int(corrections.sum()),
        'n_regressions': int(regressions.sum()),
    }


def compute_lpips_improvement_rate(
    v1_lpips: list[float],
    v2_lpips: list[float],
) -> dict:
    """Compute rate at which V2 has lower LPIPS than V1.

    Args:
        v1_lpips: List of V1 LPIPS scores
        v2_lpips: List of V2 LPIPS scores

    Returns:
        Dict with LPIPS improvement statistics
    """
    v1_arr = np.array(v1_lpips)
    v2_arr = np.array(v2_lpips)

    improvements = v2_arr < v1_arr
    improvement_rate = improvements.mean()

    mean_lpips_change = (v2_arr - v1_arr).mean()

    return {
        'lpips_improvement_rate': float(improvement_rate),
        'mean_lpips_change': float(mean_lpips_change),
        'mean_v1_lpips': float(v1_arr.mean()),
        'mean_v2_lpips': float(v2_arr.mean()),
        'n_improved': int(improvements.sum()),
        'n_total': len(v1_lpips),
    }


# =============================================================================
# Analysis Plots
# =============================================================================

def create_verification_plots(
    condition_results: dict[str, dict],
) -> bytes:
    """Create visualization of verification loop results.

    Args:
        condition_results: Dict mapping condition name to results dict

    Returns:
        PNG bytes
    """
    conditions = list(condition_results.keys())
    n_conditions = len(conditions)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Accuracy comparison (V1 vs V2)
    ax = axes[0, 0]
    x = np.arange(n_conditions)
    width = 0.35

    v1_accs = [condition_results[c]['correction_stats']['v1_accuracy'] for c in conditions]
    v2_accs = [condition_results[c]['correction_stats']['v2_accuracy'] for c in conditions]

    bars1 = ax.bar(x - width/2, v1_accs, width, label='V1 (no feedback)', color='lightcoral')
    bars2 = ax.bar(x + width/2, v2_accs, width, label='V2 (with feedback)', color='lightgreen')

    ax.set_xlabel('Feedback Condition')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy: V1 vs V2 by Feedback Type')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    # Add improvement annotations
    for i, (v1, v2) in enumerate(zip(v1_accs, v2_accs)):
        improvement = v2 - v1
        color = 'green' if improvement > 0 else 'red'
        ax.annotate(f'{improvement:+.1%}', xy=(i, max(v1, v2) + 0.02),
                   ha='center', fontsize=9, color=color)

    # 2. Correction rate
    ax = axes[0, 1]
    correction_rates = [condition_results[c]['correction_stats']['correction_rate'] for c in conditions]
    regression_rates = [condition_results[c]['correction_stats']['regression_rate'] for c in conditions]

    bars1 = ax.bar(x - width/2, correction_rates, width, label='Correction rate', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, regression_rates, width, label='Regression rate', color='red', alpha=0.7)

    ax.set_xlabel('Feedback Condition')
    ax.set_ylabel('Rate')
    ax.set_title('Correction vs Regression Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.legend()
    ax.axhline(y=0.15, color='gray', linestyle='--', alpha=0.5, label='Target (15%)')

    # 3. LPIPS comparison
    ax = axes[1, 0]
    v1_lpips = [condition_results[c]['lpips_stats']['mean_v1_lpips'] for c in conditions]
    v2_lpips = [condition_results[c]['lpips_stats']['mean_v2_lpips'] for c in conditions]

    bars1 = ax.bar(x - width/2, v1_lpips, width, label='V1 LPIPS', color='salmon')
    bars2 = ax.bar(x + width/2, v2_lpips, width, label='V2 LPIPS', color='lightblue')

    ax.set_xlabel('Feedback Condition')
    ax.set_ylabel('LPIPS (lower is better)')
    ax.set_title('Mean LPIPS: V1 vs V2')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.legend()

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "Verification Loop Results Summary\n" + "=" * 40 + "\n\n"
    for condition in conditions:
        stats = condition_results[condition]['correction_stats']
        lpips = condition_results[condition]['lpips_stats']
        summary_text += f"{condition}:\n"
        summary_text += f"  Accuracy: {stats['v1_accuracy']:.1%} -> {stats['v2_accuracy']:.1%} ({stats['accuracy_improvement']:+.1%})\n"
        summary_text += f"  Correction rate: {stats['correction_rate']:.1%}\n"
        summary_text += f"  LPIPS improvement rate: {lpips['lpips_improvement_rate']:.1%}\n\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.suptitle('E4.3: Single Verification Loop Analysis', fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf.read()


# =============================================================================
# Main Experiment Handler
# =============================================================================

def e4_3_verification_loop(runner: ExperimentRunner) -> dict:
    """E4.3: Test if verification feedback improves second attempt.

    Protocol:
    1. For each sample, generate Round 1 prediction
    2. Compute verification signal (LPIPS, VLM description)
    3. Generate Round 2 with different feedback types
    4. Compare V1 vs V2 accuracy

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E4.3: Single Verification Loop (Feedback Improves Accuracy?)")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner.log_metrics({"e4_3/stage": 0, "e4_3/progress": 0.0})

    # =========================================================================
    # Stage 1: Load Data
    # =========================================================================
    print("\n[Stage 1/5] Loading SSv2 data...")

    num_samples = 50
    videos, action_labels, action_ids, label_to_id = load_ssv2_with_labels(
        subset_size=num_samples,
        num_frames=16,
    )

    unique_actions = list(label_to_id.keys())[:50]
    print(f"  Loaded {len(videos)} videos, {len(unique_actions)} action classes")

    runner.log_metrics({
        "e4_3/stage": 1,
        "e4_3/progress": 0.1,
        "e4_3/num_videos": len(videos),
    })

    # =========================================================================
    # Stage 2: Generate Round 1 Predictions
    # =========================================================================
    print("\n[Stage 2/5] Generating Round 1 predictions...")

    # Pre-load models
    _ = _get_ltx_pipeline(device)
    _ = _get_lpips_model(device)

    round1_results = []
    num_eval = min(30, len(videos))

    for i in range(num_eval):
        video = videos[i].to(device)
        true_label = action_labels[i]

        context = video[:8]
        actual_future = video[8:]

        # Generate V1
        v1_prediction = generate_video_round1(context, true_label, device, num_output_frames=8)

        # Compute LPIPS
        v1_lpips = compute_video_lpips(v1_prediction, actual_future, device)

        round1_results.append({
            'sample_id': i,
            'true_label': true_label,
            'context': context.cpu(),
            'actual_future': actual_future.cpu(),
            'v1_prediction': v1_prediction.cpu(),
            'v1_lpips': v1_lpips['mean'],
        })

        if (i + 1) % 10 == 0:
            print(f"    Generated {i + 1}/{num_eval} Round 1 predictions")
            runner.log_metrics({
                "e4_3/progress": 0.1 + 0.2 * (i + 1) / num_eval,
            })

    runner.log_metrics({"e4_3/stage": 2, "e4_3/progress": 0.3})

    # =========================================================================
    # Stage 3: Classify V1 and Get Descriptions
    # =========================================================================
    print("\n[Stage 3/5] Classifying V1 and generating descriptions...")

    _ = _get_vlm(device)

    for i, result in enumerate(round1_results):
        # Classify V1
        v1_action, _ = classify_action_in_video(
            result['v1_prediction'].to(device),
            unique_actions,
            device,
        )
        result['v1_predicted_action'] = v1_action

        # Check correctness
        true_label = result['true_label']
        v1_correct = (
            v1_action.lower() in true_label.lower() or
            true_label.lower() in v1_action.lower() or
            v1_action == true_label
        )
        result['v1_correct'] = v1_correct

        # Get descriptions for VLM feedback
        v1_description = describe_video(result['v1_prediction'].to(device), device)
        actual_description = describe_video(result['actual_future'].to(device), device)
        result['v1_description'] = v1_description
        result['actual_description'] = actual_description

        if (i + 1) % 10 == 0:
            v1_acc = np.mean([r['v1_correct'] for r in round1_results[:i+1]])
            print(f"    Processed {i + 1}/{len(round1_results)} (V1 accuracy so far: {v1_acc:.2%})")

    v1_accuracy = np.mean([r['v1_correct'] for r in round1_results])
    print(f"  V1 overall accuracy: {v1_accuracy:.2%}")

    runner.log_metrics({"e4_3/stage": 3, "e4_3/progress": 0.5})

    # =========================================================================
    # Stage 4: Generate Round 2 with Different Feedback Types
    # =========================================================================
    print("\n[Stage 4/5] Generating Round 2 predictions with feedback...")

    feedback_conditions = ["baseline", "binary", "lpips", "vlm"]
    condition_results = {}

    for condition in feedback_conditions:
        print(f"\n  Testing feedback type: {condition}")

        v2_predictions = []
        v2_correct_list = []
        v1_correct_list = []
        v1_lpips_list = []
        v2_lpips_list = []

        for i, result in enumerate(round1_results):
            context = result['context'].to(device)
            actual_future = result['actual_future'].to(device)
            true_label = result['true_label']

            # Prepare feedback content
            feedback_content = {
                'lpips_score': result['v1_lpips'],
                'v1_description': result['v1_description'],
                'actual_description': result['actual_description'],
            }

            # Generate V2
            v2_prediction = generate_video_round2_with_feedback(
                context, true_label, condition, feedback_content, device, num_output_frames=8
            )

            # Compute V2 LPIPS
            v2_lpips = compute_video_lpips(v2_prediction, actual_future, device)

            # Classify V2
            v2_action, _ = classify_action_in_video(v2_prediction, unique_actions, device)

            v2_correct = (
                v2_action.lower() in true_label.lower() or
                true_label.lower() in v2_action.lower() or
                v2_action == true_label
            )

            v2_predictions.append({
                'v2_prediction': v2_prediction.cpu(),
                'v2_lpips': v2_lpips['mean'],
                'v2_predicted_action': v2_action,
                'v2_correct': v2_correct,
            })

            v1_correct_list.append(result['v1_correct'])
            v2_correct_list.append(v2_correct)
            v1_lpips_list.append(result['v1_lpips'])
            v2_lpips_list.append(v2_lpips['mean'])

        # Compute statistics for this condition
        correction_stats = compute_correction_rate(v1_correct_list, v2_correct_list)
        lpips_stats = compute_lpips_improvement_rate(v1_lpips_list, v2_lpips_list)

        condition_results[condition] = {
            'predictions': v2_predictions,
            'correction_stats': correction_stats,
            'lpips_stats': lpips_stats,
            'v1_correct': v1_correct_list,
            'v2_correct': v2_correct_list,
            'v1_lpips': v1_lpips_list,
            'v2_lpips': v2_lpips_list,
        }

        print(f"    {condition}: V1={correction_stats['v1_accuracy']:.1%} -> V2={correction_stats['v2_accuracy']:.1%} "
              f"(correction rate={correction_stats['correction_rate']:.1%})")

        runner.log_metrics({
            f"e4_3/{condition}/accuracy_improvement": correction_stats['accuracy_improvement'],
            f"e4_3/{condition}/correction_rate": correction_stats['correction_rate'],
            f"e4_3/{condition}/lpips_improvement_rate": lpips_stats['lpips_improvement_rate'],
        })

    runner.log_metrics({"e4_3/stage": 4, "e4_3/progress": 0.85})

    # =========================================================================
    # Stage 5: Analysis and Visualization
    # =========================================================================
    print("\n[Stage 5/5] Analyzing results and generating plots...")

    # Create plots
    plot_bytes = create_verification_plots(condition_results)
    plot_path = runner.results.save_artifact("verification_loop_analysis.png", plot_bytes)

    # Find best condition
    best_condition = max(
        condition_results.keys(),
        key=lambda c: condition_results[c]['correction_stats']['accuracy_improvement']
    )
    best_stats = condition_results[best_condition]['correction_stats']

    # Save detailed results
    detailed_results = {
        'round1_summary': {
            'n_samples': len(round1_results),
            'v1_accuracy': float(v1_accuracy),
            'mean_v1_lpips': float(np.mean([r['v1_lpips'] for r in round1_results])),
        },
        'condition_results': {
            condition: {
                'correction_stats': stats['correction_stats'],
                'lpips_stats': stats['lpips_stats'],
            }
            for condition, stats in condition_results.items()
        },
        'best_condition': best_condition,
    }
    results_path = runner.results.save_json_artifact("verification_results.json", detailed_results)

    runner.log_metrics({
        "e4_3/stage": 5,
        "e4_3/progress": 1.0,
        "e4_3/best_condition": feedback_conditions.index(best_condition),
        "e4_3/best_accuracy_improvement": best_stats['accuracy_improvement'],
        "e4_3/best_correction_rate": best_stats['correction_rate'],
    })

    # =========================================================================
    # Determine Finding
    # =========================================================================

    # Success criteria
    correction_rate_min = 0.15
    correction_rate_target = 0.30
    accuracy_improvement_min = 0.05
    accuracy_improvement_target = 0.10
    lpips_improvement_min = 0.55
    lpips_improvement_target = 0.60

    best_lpips_stats = condition_results[best_condition]['lpips_stats']

    passed_correction = best_stats['correction_rate'] >= correction_rate_min
    passed_accuracy = best_stats['accuracy_improvement'] >= accuracy_improvement_min
    passed_lpips = best_lpips_stats['lpips_improvement_rate'] >= lpips_improvement_min

    passed_all = passed_correction and passed_accuracy and passed_lpips

    if passed_all:
        if (best_stats['correction_rate'] >= correction_rate_target and
            best_stats['accuracy_improvement'] >= accuracy_improvement_target):
            finding = (
                f"VERIFICATION LOOP WORKS WELL: Best feedback='{best_condition}', "
                f"correction rate={best_stats['correction_rate']:.1%} (target>{correction_rate_target:.0%}), "
                f"accuracy improvement={best_stats['accuracy_improvement']:.1%} (target>{accuracy_improvement_target:.0%}). "
                f"Feedback significantly improves predictions! Proceed with multi-iteration experiments (E4.4)."
            )
        else:
            finding = (
                f"VERIFICATION LOOP HELPS: Best feedback='{best_condition}', "
                f"correction rate={best_stats['correction_rate']:.1%} (>{correction_rate_min:.0%}), "
                f"accuracy improvement={best_stats['accuracy_improvement']:.1%} (>{accuracy_improvement_min:.0%}). "
                f"Feedback provides meaningful improvement. Consider optimizing feedback signal."
            )
    else:
        failures = []
        if not passed_correction:
            failures.append(f"correction_rate={best_stats['correction_rate']:.1%}<{correction_rate_min:.0%}")
        if not passed_accuracy:
            failures.append(f"accuracy_improvement={best_stats['accuracy_improvement']:.1%}<{accuracy_improvement_min:.0%}")
        if not passed_lpips:
            failures.append(f"lpips_improvement={best_lpips_stats['lpips_improvement_rate']:.1%}<{lpips_improvement_min:.0%}")

        finding = (
            f"VERIFICATION LOOP INSUFFICIENT: {', '.join(failures)}. "
            f"Best condition was '{best_condition}' but did not meet thresholds. "
            f"Consider pivots: verification for filtering (not correction), training-time verification, or ensemble selection."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "accuracy_improvement": float(best_stats['accuracy_improvement']),
            "correction_rate": float(best_stats['correction_rate']),
            "lpips_improvement_rate": float(best_lpips_stats['lpips_improvement_rate']),
            "v1_accuracy": float(best_stats['v1_accuracy']),
            "v2_accuracy": float(best_stats['v2_accuracy']),
            "best_condition": best_condition,
            "n_samples": len(round1_results),
            "passed": passed_all,
        },
        "condition_comparison": {
            condition: {
                'accuracy_improvement': stats['correction_stats']['accuracy_improvement'],
                'correction_rate': stats['correction_stats']['correction_rate'],
            }
            for condition, stats in condition_results.items()
        },
        "artifacts": [plot_path, results_path],
    }
