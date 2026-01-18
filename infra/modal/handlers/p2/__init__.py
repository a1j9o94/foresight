"""P2 Experiment Handlers: Hybrid Encoder (VLM + DINOv2)

This package contains handlers for the P2 experiment which tests whether
combining DINOv2 spatial features with Qwen2.5-VL semantic features can
achieve Spatial IoU > 0.6 (the Gate 1 threshold).

Sub-experiments:
- E-P2.1: DINOv2 spatial feature analysis
- E-P2.2: DINOv2-only reconstruction baseline
- E-P2.3: Cross-attention fusion module training
- E-P2.4: End-to-end hybrid pipeline evaluation
- E-P2.5: Ablation studies
- E-P2.6: Latency and efficiency analysis

Key Hypothesis:
A cross-attention fusion module combining DINOv2 spatial features with VLM
semantic features will achieve:
1. Spatial IoU > 0.6 (Gate 1 threshold)
2. LPIPS < 0.35 (maintain perceptual quality)
3. mAP@0.5 > 0.4 (object detection capability)

Success Criteria:
- Spatial IoU > 0.6 (must pass)
- LPIPS < 0.35
- mAP@0.5 > 0.4
- Latency overhead < 25%
"""

from typing import Callable

from .e_p2_1 import e_p2_1_dinov2_spatial_analysis
from .e_p2_2 import e_p2_2_dinov2_reconstruction
from .e_p2_3 import e_p2_3_fusion_training
from .e_p2_4 import e_p2_4_pipeline_evaluation
from .e_p2_5 import e_p2_5_ablations
from .e_p2_6 import e_p2_6_latency_analysis


def get_handlers() -> dict[str, Callable]:
    """Return all handlers for P2 experiment.

    Returns:
        Dict mapping sub-experiment IDs to handler functions.
    """
    return {
        # DINOv2 analysis
        "e_p2_1": e_p2_1_dinov2_spatial_analysis,
        "e_p2_2": e_p2_2_dinov2_reconstruction,
        # Fusion training
        "e_p2_3": e_p2_3_fusion_training,
        # Evaluation
        "e_p2_4": e_p2_4_pipeline_evaluation,
        "e_p2_5": e_p2_5_ablations,
        "e_p2_6": e_p2_6_latency_analysis,
    }
