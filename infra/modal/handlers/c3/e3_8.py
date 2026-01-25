"""E3.8: Video Predicts → VLM Describes

Pivot from failed VLM prediction approach. Use each model for its strength:
- LTX-Video: Generate plausible future frames (trained for temporal dynamics)
- Qwen2.5-VL: Describe/reason about generated content (trained for understanding)

Sub-experiments:
- E3.8a: Video Continuation Quality - Can LTX-Video generate coherent continuations?
- E3.8b: Action Recognition on Generated - Can VLM recognize actions in generated video?
- E3.8c: Description Alignment - Does VLM's description match ground truth?

Success Criteria:
- action_accuracy > 0.40 (acceptable), > 0.60 (target)
- semantic_similarity > 0.50 (acceptable), > 0.70 (target)
"""

import os
import sys

sys.path.insert(0, "/root")

import json
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats

from runner import ExperimentRunner


# =============================================================================
# Data Loading (shared across sub-experiments)
# =============================================================================

def load_ssv2_with_labels(subset_size: int = 200, num_frames: int = 16):
    """Load SSv2 videos with action labels for E3.8.

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
            split="train",
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

            # VideoSample has .frames, .label, .label_id attributes
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
    videos = []
    action_labels = []
    action_ids = []
    label_to_id = {f"action_{i}": i for i in range(50)}

    for i in range(subset_size):
        # Create moving shape video
        video = torch.zeros(num_frames, 3, 224, 224)
        x_start = random.randint(20, 180)
        y_start = random.randint(20, 180)
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        color = torch.rand(3)

        for t in range(num_frames):
            x = int(x_start + dx * t) % 200 + 12
            y = int(y_start + dy * t) % 200 + 12
            video[t, :, y-10:y+10, x-10:x+10] = color.view(3, 1, 1)

        videos.append(video)
        label = f"action_{i % 50}"
        action_labels.append(label)
        action_ids.append(label_to_id[label])

    return videos, action_labels, action_ids, label_to_id


# =============================================================================
# E3.8a: Video Continuation Quality
# =============================================================================

def e3_8a_video_continuation(runner: ExperimentRunner) -> dict:
    """E3.8a: Test if LTX-Video can generate coherent continuations.

    Protocol:
    1. Load SSv2 videos (16 frames each)
    2. Use first 8 frames as context
    3. Generate next 8 frames with LTX-Video
    4. Compare generated vs real future frames
    """
    print("=" * 60)
    print("E3.8a: Video Continuation Quality")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print("\n[Stage 1/4] Loading SSv2 data...")
    videos, action_labels, action_ids, label_to_id = load_ssv2_with_labels(
        subset_size=100,
        num_frames=16,
    )

    # Load LTX-Video Image-to-Video pipeline
    print("\n[Stage 2/4] Loading LTX-Video (Image-to-Video)...")
    try:
        from diffusers import LTXImageToVideoPipeline

        pipeline = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        ).to(device)
        print("  LTX-Video Image-to-Video pipeline loaded successfully")
        has_ltx = True
    except Exception as e:
        print(f"  [WARN] Failed to load LTX-Video: {e}")
        print("  Using simple frame extrapolation as baseline")
        has_ltx = False

    # Generate continuations
    print("\n[Stage 3/4] Generating video continuations...")

    results = []
    num_samples = min(20, len(videos))

    for i in range(num_samples):
        video = videos[i].to(device)  # [16, C, H, W]
        context_frames = video[:8]     # First 8 frames
        real_future = video[8:]        # Last 8 frames (ground truth)

        if has_ltx:
            # Use LTX-Video Image-to-Video pipeline
            try:
                # Get last frame as conditioning image
                last_frame = context_frames[-1]
                conditioning_image = _tensor_to_pil(last_frame)

                # Get action label for prompt
                action = action_labels[i]
                prompt = f"Continue this video showing: {action}"
                negative_prompt = "worst quality, blurry, jittery, distorted"

                # Generate continuation frames
                with torch.no_grad():
                    output = pipeline(
                        image=conditioning_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=224,
                        height=224,
                        num_frames=9,  # 1 conditioning + 8 generated
                        num_inference_steps=15,  # Faster for evaluation
                        guidance_scale=3.0,
                    )
                    generated_pil = output.frames[0]

                    # Convert PIL frames to tensor, skip first (conditioning) frame
                    gen_frames = torch.stack([
                        torch.from_numpy(np.array(f)).permute(2, 0, 1).float() / 255.0
                        for f in generated_pil[1:9]  # Skip conditioning frame
                    ]).to(device)

                if (i + 1) % 5 == 0:
                    print(f"    [LTX] Generated {len(gen_frames)} frames for sample {i+1}")

            except Exception as e:
                print(f"    [WARN] LTX generation failed for sample {i}: {e}")
                # Fallback: simple extrapolation
                gen_frames = _simple_extrapolation(context_frames)
        else:
            # Simple baseline: extrapolate from last frames
            gen_frames = _simple_extrapolation(context_frames)

        # Compute metrics
        # Resize if needed
        if gen_frames.shape[-2:] != real_future.shape[-2:]:
            gen_frames = F.interpolate(
                gen_frames,
                size=real_future.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        # Pixel-level metrics
        mse = F.mse_loss(gen_frames, real_future).item()
        l1 = F.l1_loss(gen_frames, real_future).item()

        # Temporal consistency (difference between consecutive frames)
        gen_diff = (gen_frames[1:] - gen_frames[:-1]).abs().mean().item()
        real_diff = (real_future[1:] - real_future[:-1]).abs().mean().item()
        temporal_ratio = gen_diff / (real_diff + 1e-6)

        results.append({
            "sample_id": i,
            "mse": mse,
            "l1": l1,
            "temporal_ratio": temporal_ratio,
            "action": action_labels[i],
        })

        if (i + 1) % 5 == 0:
            print(f"    Processed {i + 1}/{num_samples} videos")

    # Aggregate metrics
    print("\n[Stage 4/4] Computing final metrics...")

    avg_mse = np.mean([r["mse"] for r in results])
    avg_l1 = np.mean([r["l1"] for r in results])
    avg_temporal = np.mean([r["temporal_ratio"] for r in results])

    # Copy baseline comparison
    copy_l1_scores = []
    for i in range(num_samples):
        video = videos[i]
        last_context = video[7:8].expand(8, -1, -1, -1)  # Repeat last frame
        real_future = video[8:]
        copy_l1 = F.l1_loss(last_context, real_future).item()
        copy_l1_scores.append(copy_l1)

    avg_copy_l1 = np.mean(copy_l1_scores)
    improvement = avg_copy_l1 - avg_l1

    print(f"\n  Results:")
    print(f"    Generated L1: {avg_l1:.4f}")
    print(f"    Copy baseline L1: {avg_copy_l1:.4f}")
    print(f"    Improvement: {improvement:.4f}")
    print(f"    Temporal ratio: {avg_temporal:.4f} (1.0 = matches real dynamics)")

    # Log metrics
    runner.log_metrics({
        "e3_8a/l1_loss": avg_l1,
        "e3_8a/mse_loss": avg_mse,
        "e3_8a/copy_baseline": avg_copy_l1,
        "e3_8a/improvement": improvement,
        "e3_8a/temporal_ratio": avg_temporal,
    })

    return {
        "status": "completed",
        "finding": f"VIDEO CONTINUATION: L1={avg_l1:.4f}, copy baseline={avg_copy_l1:.4f}, improvement={improvement:.4f}",
        "metrics": {
            "l1_loss": float(avg_l1),
            "mse_loss": float(avg_mse),
            "copy_baseline_l1": float(avg_copy_l1),
            "improvement_over_copy": float(improvement),
            "temporal_ratio": float(avg_temporal),
            "num_samples": num_samples,
        },
        "artifacts": [],
    }


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor [C, H, W] to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _simple_extrapolation(context: torch.Tensor) -> torch.Tensor:
    """Simple linear extrapolation baseline."""
    # Use last two frames to estimate velocity, extrapolate
    T = context.shape[0]
    velocity = context[-1] - context[-2]  # [C, H, W]

    extrapolated = []
    last_frame = context[-1]
    for t in range(8):
        next_frame = last_frame + velocity * (t + 1) * 0.5
        next_frame = next_frame.clamp(0, 1)
        extrapolated.append(next_frame)

    return torch.stack(extrapolated)


# Global LTX pipeline cache
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


def _generate_video_continuation(
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
    return _simple_extrapolation(context_frames)[:num_output_frames]


# =============================================================================
# E3.8b: Action Recognition on Generated Video
# =============================================================================

def e3_8b_action_recognition(runner: ExperimentRunner) -> dict:
    """E3.8b: Test if VLM can recognize actions in generated video.

    Protocol:
    1. Generate video continuations (from E3.8a approach)
    2. Ask VLM to classify the action in generated video
    3. Compare to ground truth action labels
    4. Also test on real video as baseline
    """
    print("=" * 60)
    print("E3.8b: Action Recognition on Generated Video")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print("\n[Stage 1/5] Loading SSv2 data...")
    videos, action_labels, action_ids, label_to_id = load_ssv2_with_labels(
        subset_size=100,
        num_frames=16,
    )
    id_to_label = {v: k for k, v in label_to_id.items()}

    # Load VLM
    print("\n[Stage 2/5] Loading VLM...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    vlm.eval()
    print("  VLM loaded")

    # Create action prompt with choices
    action_list = list(label_to_id.keys())[:50]  # Top 50 actions
    action_choices = "\n".join([f"{i+1}. {a}" for i, a in enumerate(action_list)])

    def classify_action(frames: torch.Tensor) -> tuple[str, float]:
        """Ask VLM to classify action in video frames."""
        # Sample frames for VLM (first, middle, last)
        indices = [0, len(frames)//2, len(frames)-1]
        pil_images = [_tensor_to_pil(frames[i]) for i in indices]

        # Create multi-image prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_images[0]},
                    {"type": "image", "image": pil_images[1]},
                    {"type": "image", "image": pil_images[2]},
                    {"type": "text", "text": f"""These are 3 frames from a video showing an action.
What action is being performed? Choose from this list:

{action_choices}

Reply with ONLY the number of the matching action."""},
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
                max_new_tokens=20,
                do_sample=False,
            )

        response = processor.decode(outputs[0], skip_special_tokens=True)

        # Parse response to get action number
        try:
            # Extract number from response
            import re
            numbers = re.findall(r'\d+', response.split("assistant")[-1])
            if numbers:
                idx = int(numbers[0]) - 1
                if 0 <= idx < len(action_list):
                    return action_list[idx], 1.0
        except:
            pass

        return "unknown", 0.0

    # Test on real videos first (baseline)
    print("\n[Stage 3/5] Testing action recognition on REAL videos...")

    real_correct = 0
    real_total = 0
    num_samples = min(30, len(videos))

    for i in range(num_samples):
        video = videos[i]
        true_label = action_labels[i]

        # Use future frames (what we'd generate)
        future_frames = video[8:]

        predicted, conf = classify_action(future_frames)

        if predicted.lower() in true_label.lower() or true_label.lower() in predicted.lower():
            real_correct += 1
        real_total += 1

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{num_samples} (acc so far: {real_correct/real_total:.2%})")

    real_accuracy = real_correct / real_total
    print(f"  Real video accuracy: {real_accuracy:.2%}")

    # Test on generated videos
    print("\n[Stage 4/5] Testing action recognition on GENERATED videos...")
    print("  Loading LTX-Video for generation...")
    _ = _get_ltx_pipeline(device)  # Pre-load pipeline

    gen_correct = 0
    gen_total = 0

    for i in range(num_samples):
        video = videos[i].to(device)
        true_label = action_labels[i]

        # Generate continuation using LTX-Video (or fallback)
        context = video[:8]
        generated = _generate_video_continuation(context, true_label, device, num_output_frames=8)

        predicted, conf = classify_action(generated)

        if predicted.lower() in true_label.lower() or true_label.lower() in predicted.lower():
            gen_correct += 1
        gen_total += 1

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{num_samples} (acc so far: {gen_correct/gen_total:.2%})")

    gen_accuracy = gen_correct / gen_total

    # Results
    print("\n[Stage 5/5] Results...")
    print(f"  Real video accuracy: {real_accuracy:.2%}")
    print(f"  Generated video accuracy: {gen_accuracy:.2%}")
    print(f"  Accuracy drop: {real_accuracy - gen_accuracy:.2%}")

    runner.log_metrics({
        "e3_8b/real_accuracy": real_accuracy,
        "e3_8b/gen_accuracy": gen_accuracy,
        "e3_8b/accuracy_drop": real_accuracy - gen_accuracy,
    })

    passed = gen_accuracy >= 0.40  # Acceptable threshold

    return {
        "status": "completed",
        "finding": f"ACTION RECOGNITION: real={real_accuracy:.2%}, generated={gen_accuracy:.2%}, drop={real_accuracy-gen_accuracy:.2%}",
        "metrics": {
            "real_video_accuracy": float(real_accuracy),
            "generated_video_accuracy": float(gen_accuracy),
            "accuracy_drop": float(real_accuracy - gen_accuracy),
            "num_samples": num_samples,
            "passed": passed,
        },
        "artifacts": [],
    }


# =============================================================================
# E3.8c: Description Alignment
# =============================================================================

def e3_8c_description_alignment(runner: ExperimentRunner) -> dict:
    """E3.8c: Test if VLM descriptions of generated video match reality.

    Protocol:
    1. Generate video continuation
    2. Ask VLM to describe both generated and real future
    3. Compute semantic similarity between descriptions
    4. Check factual consistency
    """
    print("=" * 60)
    print("E3.8c: Description Alignment")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print("\n[Stage 1/4] Loading SSv2 data...")
    videos, action_labels, action_ids, label_to_id = load_ssv2_with_labels(
        subset_size=50,
        num_frames=16,
    )

    # Load VLM
    print("\n[Stage 2/4] Loading VLM...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    vlm.eval()

    # Load sentence transformer for similarity
    print("  Loading sentence transformer for similarity...")
    try:
        from sentence_transformers import SentenceTransformer
        sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        has_sim = True
    except:
        print("  [WARN] sentence-transformers not available, using word overlap")
        has_sim = False

    def describe_video(frames: torch.Tensor) -> str:
        """Get VLM description of video frames."""
        indices = [0, len(frames)//2, len(frames)-1]
        pil_images = [_tensor_to_pil(frames[i]) for i in indices]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_images[0]},
                    {"type": "image", "image": pil_images[1]},
                    {"type": "image", "image": pil_images[2]},
                    {"type": "text", "text": "Describe what is happening in this video sequence in 2-3 sentences. Focus on the action and any objects involved."},
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
                max_new_tokens=100,
                do_sample=False,
            )

        response = processor.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response

    def compute_similarity(text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if has_sim:
            emb1 = sim_model.encode(text1)
            emb2 = sim_model.encode(text2)
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        else:
            # Simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            overlap = len(words1 & words2)
            return overlap / max(len(words1), len(words2))

    # Compare descriptions
    print("\n[Stage 3/4] Comparing descriptions...")
    print("  Loading LTX-Video for generation...")
    _ = _get_ltx_pipeline(device)  # Pre-load pipeline

    results = []
    num_samples = min(20, len(videos))

    for i in range(num_samples):
        video = videos[i].to(device)
        true_label = action_labels[i]

        # Get real and generated futures
        real_future = video[8:]
        context = video[:8]
        gen_future = _generate_video_continuation(context, true_label, device, num_output_frames=8)

        # Get descriptions
        real_desc = describe_video(real_future)
        gen_desc = describe_video(gen_future)

        # Compute similarity
        similarity = compute_similarity(real_desc, gen_desc)

        # Check if action is mentioned
        action_in_real = any(word in real_desc.lower() for word in true_label.lower().split())
        action_in_gen = any(word in gen_desc.lower() for word in true_label.lower().split())

        results.append({
            "sample_id": i,
            "action": true_label,
            "real_description": real_desc[:200],
            "gen_description": gen_desc[:200],
            "similarity": similarity,
            "action_in_real": action_in_real,
            "action_in_gen": action_in_gen,
        })

        print(f"    Sample {i+1}: sim={similarity:.3f}, action_match={action_in_gen}")

    # Aggregate
    print("\n[Stage 4/4] Final results...")

    avg_similarity = np.mean([r["similarity"] for r in results])
    action_recall_real = np.mean([r["action_in_real"] for r in results])
    action_recall_gen = np.mean([r["action_in_gen"] for r in results])

    print(f"  Average semantic similarity: {avg_similarity:.3f}")
    print(f"  Action mentioned in real desc: {action_recall_real:.2%}")
    print(f"  Action mentioned in gen desc: {action_recall_gen:.2%}")

    runner.log_metrics({
        "e3_8c/semantic_similarity": avg_similarity,
        "e3_8c/action_recall_real": action_recall_real,
        "e3_8c/action_recall_gen": action_recall_gen,
    })

    passed = avg_similarity >= 0.50  # Acceptable threshold

    return {
        "status": "completed",
        "finding": f"DESCRIPTION ALIGNMENT: similarity={avg_similarity:.3f}, action_recall_gen={action_recall_gen:.2%}",
        "metrics": {
            "semantic_similarity": float(avg_similarity),
            "action_recall_real": float(action_recall_real),
            "action_recall_gen": float(action_recall_gen),
            "num_samples": num_samples,
            "passed": passed,
        },
        "sample_descriptions": results[:5],  # Include a few examples
        "artifacts": [],
    }


# =============================================================================
# Entry Points
# =============================================================================

def e3_8a(runner: ExperimentRunner) -> dict:
    """Entry point for E3.8a: Video Continuation Quality."""
    return e3_8a_video_continuation(runner)


def e3_8b(runner: ExperimentRunner) -> dict:
    """Entry point for E3.8b: Action Recognition on Generated."""
    return e3_8b_action_recognition(runner)


def e3_8c(runner: ExperimentRunner) -> dict:
    """Entry point for E3.8c: Description Alignment."""
    return e3_8c_description_alignment(runner)


def e3_8(runner: ExperimentRunner) -> dict:
    """Run all E3.8 sub-experiments sequentially."""
    results = {}

    print("\n" + "=" * 70)
    print("E3.8: VIDEO PREDICTS → VLM DESCRIBES")
    print("=" * 70)

    # Run each sub-experiment
    print("\n[1/3] Running E3.8a: Video Continuation Quality...")
    results["e3_8a"] = e3_8a_video_continuation(runner)

    print("\n[2/3] Running E3.8b: Action Recognition...")
    results["e3_8b"] = e3_8b_action_recognition(runner)

    print("\n[3/3] Running E3.8c: Description Alignment...")
    results["e3_8c"] = e3_8c_description_alignment(runner)

    # Aggregate results
    print("\n" + "=" * 70)
    print("E3.8 SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        print(f"\n{name}: {result['finding']}")

    overall_passed = (
        results["e3_8b"]["metrics"].get("passed", False) or
        results["e3_8c"]["metrics"].get("passed", False)
    )

    return {
        "status": "completed",
        "finding": "VIDEO→VLM APPROACH: " + ("SHOWS PROMISE" if overall_passed else "NEEDS WORK"),
        "sub_experiments": results,
        "overall_passed": overall_passed,
    }
