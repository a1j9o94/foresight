"""Q2 Experiment Handlers: Information Preservation Through VLM

This package contains handlers for the Q2 experiment which quantifies
information loss at each stage of Qwen2.5-VL's visual processing pipeline
and identifies the best extraction point for video generation conditioning.

Sub-experiments:
- E-Q2.1: Pre-merge ViT analysis (baseline spatial accuracy)
- E-Q2.2: Post-merge analysis (quantify 2x2 merge impact)
- E-Q2.3: LLM layer-wise decay analysis
- E-Q2.4: Bounding box detection probe (mAP metrics)
- E-Q2.5: Fine-grained detail probe (LPIPS, edge F1)
- E-Q2.6: Temporal information probe (video inputs)

Key Extraction Points:
- E1: ViT patch embeddings [N_patches, 1536]
- E4: ViT final (pre-merge) [N_patches, 1536]
- E5: Post-merge [N_tokens, 1536]
- E6-E8: LLM layers 1, 14, 28 [seq_len, 3584]

Success Criteria:
- Bbox IoU > 0.7 at some extraction point
- LPIPS < 0.3 for reconstruction
- Edge F1 > 0.6
- Identify optimal extraction point with IRS > 0.6
"""

from typing import Callable

from .e_q2_1 import e_q2_1_premerge_analysis
from .e_q2_2 import e_q2_2_postmerge_analysis
from .e_q2_3 import e_q2_3_llm_layer_analysis
from .e_q2_4 import e_q2_4_detection_probe
from .e_q2_5 import e_q2_5_finegrained_probe
from .e_q2_6 import e_q2_6_temporal_probe


def get_handlers() -> dict[str, Callable]:
    """Return all handlers for Q2 experiment.

    Returns:
        Dict mapping sub-experiment IDs to handler functions.
    """
    return {
        # Pre-merge and post-merge analysis
        "e_q2_1": e_q2_1_premerge_analysis,
        "e_q2_2": e_q2_2_postmerge_analysis,
        # LLM layer analysis
        "e_q2_3": e_q2_3_llm_layer_analysis,
        # Probing experiments
        "e_q2_4": e_q2_4_detection_probe,
        "e_q2_5": e_q2_5_finegrained_probe,
        "e_q2_6": e_q2_6_temporal_probe,
    }
