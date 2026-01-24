"""C3 Experiment Handlers: VLM Future State Prediction in Latent Space

This package contains handlers for the C3 experiment which tests whether a VLM
(Qwen2.5-VL-7B-Instruct), augmented with learned query tokens, can predict latent
representations of future video frames.

Sub-experiments:
- E3.1: Sanity Check - Reconstruct current frame latent (verify query tokens learn)
- E3.2: Single Frame Future Prediction - Predict next frame latent
- E3.3: Action-Conditioned Prediction - Test if action descriptions improve prediction
        Tests SSv2-style actions: push right/left/up/down, rotate CW/CCW
        Compares: with action vs no action vs wrong (opposite) action
        Success: Action gain > 0.05, Action specificity > 0.1
- E3.4: Multi-Frame Future Prediction (TODO)
- E3.5: Action Discrimination Test (TODO)
- E3.6: Contrastive Evaluation / Retrieval Task (TODO)

Key Hypothesis:
Query tokens can learn to extract "imagined future" representations from the VLM
that meaningfully correlate with the VLM's encoding of actual future frames.

Success Criteria (from research_plan.yaml):
- cosine_sim_t5 > 0.75 (target)
- cosine_sim_t5 > 0.65 (acceptable)
- cosine_sim_t5 < 0.50 (failure)

Dependencies:
- P2 Hybrid Encoder (PASSED - for feature extraction baseline)
- C2 Adapter Bridging (PASSED - for conditioning injection)

Unlocks:
- C4 (Full System Verification)
- Q4, Q5 (Data Efficiency, Prediction Horizon)
- Gate 3 (Prediction Gate)
"""

from typing import Callable

from .e3_1 import e3_1_sanity_check
from .e3_2 import e3_2_single_frame_prediction
from .e3_3 import e3_3_action_conditioned_prediction


def get_handlers() -> dict[str, Callable]:
    """Return all handlers for C3 experiment.

    Returns:
        Dict mapping sub-experiment IDs to handler functions.
    """
    return {
        # Sanity check - reconstruct current frame
        "e3_1": e3_1_sanity_check,
        # Single frame future prediction
        "e3_2": e3_2_single_frame_prediction,
        # Action-conditioned prediction
        "e3_3": e3_3_action_conditioned_prediction,
    }
