"""Foresight Evaluation - Metrics, verification, and benchmarks.

This package provides evaluation tools for video prediction:
- Core metrics: LPIPS, SSIM, cosine similarity, MSE/MAE
- First frame fidelity: Evaluate how well first frames match input images
- Temporal coherence: Evaluate temporal consistency of generated videos

Usage:
    from foresight_eval import lpips_score, ssim_score
    from foresight_eval.first_frame import evaluate_first_frame_fidelity
    from foresight_eval.temporal import evaluate_temporal_coherence
"""

__version__ = "0.1.0"

from .metrics import (
    lpips_score,
    ssim_score,
    cosine_similarity,
    mse,
    mae,
    psnr,
    to_tensor,
)
from .first_frame import (
    FirstFrameMetrics,
    evaluate_first_frame_fidelity,
    batch_evaluate_first_frame,
)
from .temporal import (
    TemporalMetrics,
    evaluate_temporal_coherence,
    compute_optical_flow_consistency,
    evaluate_segment_transition,
)

__all__ = [
    # Core metrics
    "lpips_score",
    "ssim_score",
    "cosine_similarity",
    "mse",
    "mae",
    "psnr",
    "to_tensor",
    # First frame
    "FirstFrameMetrics",
    "evaluate_first_frame_fidelity",
    "batch_evaluate_first_frame",
    # Temporal
    "TemporalMetrics",
    "evaluate_temporal_coherence",
    "compute_optical_flow_consistency",
    "evaluate_segment_transition",
]
