"""Q3 Experiment Handlers: Temporal Coherence

This package contains handlers for the Q3 experiment which tests whether
conditioning injection from the hybrid encoder disrupts LTX-Video's temporal
dynamics.

Sub-experiments:
- E-Q3.1: Baseline temporal coherence measurement
- E-Q3.2: Conditioning strength vs coherence tradeoff

Key Question:
Does conditioning injection from the hybrid encoder disrupt video decoder's
temporal dynamics?

Success Criteria (from research_plan.yaml):
- temporal_consistency > 0.80 (target)
- temporal_consistency > 0.70 (acceptable)
- temporal_consistency < 0.50 (failure)

This experiment is required for Gate 2 (Bridging) alongside C2 (Adapter Bridging).

Dependencies:
- P2 Hybrid Encoder (PASSED - spatial_iou=0.837, lpips=0.162)
"""

from typing import Callable

from .e_q3_1 import run as e_q3_1_baseline_measurement
from .e_q3_2 import run as e_q3_2_conditioning_tradeoff


def get_handlers() -> dict[str, Callable]:
    """Return all handlers for Q3 experiment.

    Returns:
        Dict mapping sub-experiment IDs to handler functions.
    """
    return {
        # Baseline measurement
        "e_q3_1": e_q3_1_baseline_measurement,
        # Conditioning strength tradeoff
        "e_q3_2": e_q3_2_conditioning_tradeoff,
    }
