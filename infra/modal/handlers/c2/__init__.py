"""C2 Experiment Handlers: Adapter Bridging

This package contains handlers for the C2 experiment which tests whether
a small adapter (~10-50M params) can effectively bridge the hybrid encoder
(DINOv2 + VLM) to the LTX-Video decoder.

Sub-experiments:
- E2.1: Baseline Adapter Scaling Study - Establish scaling relationship
- E2.2: Architecture Comparison - Compare adapter architectures at ~10M params
- E2.3: Training Strategy Optimization - Find optimal loss/scheduler/augmentation
- E2.4: Final Efficiency Validation - Comprehensive Gate 2 readiness assessment

Key Hypothesis:
A 10M parameter adapter using cross-attention with learned query tokens can
achieve >90% of the reconstruction quality of a 100M parameter adapter.

Success Criteria (from research_plan.yaml):
- param_efficiency > 0.90 (target), > 0.85 (acceptable), < 0.80 (failure)
- Definition: 10M adapter achieves X% of 100M adapter quality

Dependencies:
- P2 (Hybrid Encoder) - provides DINOv2 + VLM fusion features

Blocks:
- C3 (Future Prediction) - needs efficient adapter for prediction training
- Q3 (Temporal Coherence) - needs stable adapter for temporal analysis
- Gate 2 - cannot proceed without C2 completion
"""

from typing import Callable

from .e2_1 import e2_1_baseline_adapter_scaling
from .e2_2 import e2_2_architecture_comparison
from .e2_3 import e2_3_training_strategy
from .e2_4 import e2_4_final_validation


def get_handlers() -> dict[str, Callable]:
    """Return all handlers for C2 experiment.

    Returns:
        Dict mapping sub-experiment IDs to handler functions.
    """
    return {
        # Scaling study
        "e2_1": e2_1_baseline_adapter_scaling,
        # Architecture comparison
        "e2_2": e2_2_architecture_comparison,
        # Training optimization
        "e2_3": e2_3_training_strategy,
        # Final validation
        "e2_4": e2_4_final_validation,
    }
