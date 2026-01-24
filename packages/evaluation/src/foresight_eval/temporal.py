"""Temporal coherence evaluation for video predictions.

This module provides metrics for evaluating temporal consistency:
- Optical flow consistency
- Frame-to-frame smoothness
- Temporal LPIPS (consecutive frames)

Usage:
    from foresight_eval.temporal import evaluate_temporal_coherence

    result = evaluate_temporal_coherence(video_frames)
    print(f"Temporal consistency: {result['consistency']}")
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from PIL import Image

from .metrics import lpips_score, ssim_score, to_tensor


@dataclass
class TemporalMetrics:
    """Metrics for temporal coherence evaluation."""
    consistency: float  # Overall consistency score [0, 1]
    mean_frame_lpips: float  # Mean LPIPS between consecutive frames
    mean_frame_ssim: float  # Mean SSIM between consecutive frames
    smoothness: float  # Variance in frame differences (lower = smoother)
    num_frames: int


def evaluate_temporal_coherence(
    frames: list[Union[Image.Image, torch.Tensor]],
    device: Optional[torch.device] = None,
) -> TemporalMetrics:
    """Evaluate temporal coherence of video frames.

    Args:
        frames: List of video frames (PIL Images or tensors)
        device: Computation device

    Returns:
        TemporalMetrics with coherence scores
    """
    if len(frames) < 2:
        return TemporalMetrics(
            consistency=1.0,
            mean_frame_lpips=0.0,
            mean_frame_ssim=1.0,
            smoothness=0.0,
            num_frames=len(frames),
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute frame-to-frame metrics
    lpips_scores = []
    ssim_scores = []
    frame_diffs = []

    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]

        # LPIPS between consecutive frames
        lpips = lpips_score(frame1, frame2, device=device)
        lpips_scores.append(lpips)

        # SSIM between consecutive frames
        ssim = ssim_score(frame1, frame2, device=device)
        ssim_scores.append(ssim)

        # Frame difference magnitude
        t1 = to_tensor(frame1, device)
        t2 = to_tensor(frame2, device)
        diff = (t1 - t2).abs().mean().item()
        frame_diffs.append(diff)

    mean_lpips = sum(lpips_scores) / len(lpips_scores)
    mean_ssim = sum(ssim_scores) / len(ssim_scores)

    # Smoothness: variance in frame differences (lower = more consistent motion)
    if len(frame_diffs) > 1:
        diff_mean = sum(frame_diffs) / len(frame_diffs)
        smoothness = sum((d - diff_mean) ** 2 for d in frame_diffs) / len(frame_diffs)
    else:
        smoothness = 0.0

    # Overall consistency combines SSIM (high good) and variance (low good)
    # Normalize smoothness to [0, 1] range (assuming max variance of 0.1)
    smoothness_normalized = min(1.0, smoothness / 0.1)
    consistency = 0.6 * mean_ssim + 0.4 * (1 - smoothness_normalized)

    return TemporalMetrics(
        consistency=consistency,
        mean_frame_lpips=mean_lpips,
        mean_frame_ssim=mean_ssim,
        smoothness=smoothness,
        num_frames=len(frames),
    )


def compute_optical_flow_consistency(
    frames: list[Union[Image.Image, torch.Tensor]],
    device: Optional[torch.device] = None,
) -> dict:
    """Compute optical flow-based temporal consistency.

    This requires OpenCV for optical flow computation.

    Args:
        frames: List of video frames
        device: Computation device

    Returns:
        Dict with flow-based metrics
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {
            "error": "OpenCV required for optical flow computation",
            "flow_consistency": None,
        }

    if len(frames) < 3:
        return {
            "flow_consistency": 1.0,
            "flow_magnitudes": [],
        }

    # Convert frames to numpy
    np_frames = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            np_frames.append(np.array(frame))
        else:
            np_frames.append(frame.cpu().numpy().transpose(1, 2, 0) * 255)

    # Convert to grayscale
    gray_frames = [cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in np_frames]

    # Compute optical flow between consecutive frames
    flows = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append(flow)

    # Compute flow magnitudes
    magnitudes = []
    for flow in flows:
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitudes.append(float(mag.mean()))

    # Flow consistency: variance in flow magnitudes (lower = more consistent)
    if len(magnitudes) > 1:
        mag_mean = sum(magnitudes) / len(magnitudes)
        variance = sum((m - mag_mean) ** 2 for m in magnitudes) / len(magnitudes)
        # Normalize to [0, 1] with 1 being perfect consistency
        consistency = max(0, 1 - variance / (mag_mean + 1e-6))
    else:
        consistency = 1.0

    return {
        "flow_consistency": consistency,
        "flow_magnitudes": magnitudes,
        "mean_flow_magnitude": sum(magnitudes) / len(magnitudes) if magnitudes else 0,
    }


def evaluate_segment_transition(
    segment1_frames: list[Union[Image.Image, torch.Tensor]],
    segment2_frames: list[Union[Image.Image, torch.Tensor]],
    device: Optional[torch.device] = None,
) -> dict:
    """Evaluate transition quality between two video segments.

    Useful for evaluating context jumps in streaming prediction.

    Args:
        segment1_frames: Frames from first segment
        segment2_frames: Frames from second segment
        device: Computation device

    Returns:
        Dict with transition metrics
    """
    if not segment1_frames or not segment2_frames:
        return {"error": "Both segments must have frames"}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Metrics for the transition frame
    last_frame_s1 = segment1_frames[-1]
    first_frame_s2 = segment2_frames[0]

    transition_lpips = lpips_score(last_frame_s1, first_frame_s2, device=device)
    transition_ssim = ssim_score(last_frame_s1, first_frame_s2, device=device)

    # Individual segment coherence
    metrics_s1 = evaluate_temporal_coherence(segment1_frames, device)
    metrics_s2 = evaluate_temporal_coherence(segment2_frames, device)

    return {
        "transition_lpips": transition_lpips,
        "transition_ssim": transition_ssim,
        "segment1_consistency": metrics_s1.consistency,
        "segment2_consistency": metrics_s2.consistency,
        "is_hard_jump": transition_lpips > 0.3,  # Threshold for detecting context jump
    }
