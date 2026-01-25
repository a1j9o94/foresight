"""C3 Experiment Handlers: VLM Future State Prediction in Latent Space

This package contains handlers for the C3 experiment which tests whether a VLM
(Qwen2.5-VL-7B-Instruct), augmented with learned query tokens, can predict latent
representations of future video frames.

Sub-experiments:
- E3.1: Sanity Check - Reconstruct current frame latent (verify query tokens learn)
- E3.2: Single Frame Future Prediction - Predict next frame latent (FAILED - can't beat copy)
- E3.3: Action-Conditioned Prediction - Test if action descriptions improve prediction
        Tests SSv2-style actions: push right/left/up/down, rotate CW/CCW
        Compares: with action vs no action vs wrong (opposite) action
        Success: Action gain > 0.05, Action specificity > 0.1
- E3.4: Multi-Frame Context - Use 4-8 context frames instead of 1 (FAILED)
        Hypothesis: More temporal context provides motion trajectory info
        Success: improvement over copy > 0, p < 0.01, cos_sim > 0.65
- E3.5: Temporal Transformer - Explicit temporal modeling with future token (FAILED)
        Tests standard vs causal temporal transformer
        Success: improvement over copy > 0, p < 0.01, cos_sim > 0.65
- E3.6: Contrastive Loss - InfoNCE loss instead of cosine similarity (FAILED)
        Uses hard negatives (copy baseline) for harder discrimination
        Success: improvement over copy > 0, contrastive accuracy > 50%
- E3.streaming: Streaming Prediction - Multi-frame sequences with context jumps
        Uses StreamingPredictor for recurrent prediction matching production
        Trains on SSv2 real video data
        Success: cos_sim > 0.65, handles context switches
- E3.7a: Pixel Feedback (Frozen VLM) - True autoregressive with pixel feedback
        Model sees its own generated outputs during training
        Tests if pixel feedback helps even with frozen VLM
        Success: improvement over copy > 0, p < 0.01
- E3.7b: Pixel Feedback + VLM LoRA - Fine-tune VLM while seeing own predictions
        VLM learns prediction-aware features via LoRA adapters
        Success: improvement over copy > 0, p < 0.01, beats E3.7a
- E3.8: PIVOT - Video Predicts → VLM Describes (NEW APPROACH)
        After E3.1-E3.7b all failed, pivot to using each model for its strength:
        - Video model (LTX-Video): Generate plausible future frames
        - VLM (Qwen2.5-VL): Describe/reason about generated content
        E3.8a: Video continuation quality (can LTX-Video extend coherently?)
        E3.8b: Action recognition (can VLM identify actions in generated video?)
        E3.8c: Description alignment (do descriptions of generated match real?)
        Success: action_accuracy > 0.40, semantic_similarity > 0.50

Key Hypothesis:
Query tokens can learn to extract "imagined future" representations from the VLM
that meaningfully correlate with the VLM's encoding of actual future frames.

SUCCESS REQUIRES: True autoregressive training where model sees its own outputs.

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
from .e3_4 import e3_4_multiframe_prediction
from .e3_5 import e3_5_temporal_transformer
from .e3_6 import e3_6_contrastive_prediction
from .e3_streaming import e3_streaming_prediction
from .e3_7a import e3_7a
from .e3_7b import e3_7b
from .e3_8 import e3_8a, e3_8b, e3_8c, e3_8


def get_handlers() -> dict[str, Callable]:
    """Return all handlers for C3 experiment.

    Returns:
        Dict mapping sub-experiment IDs to handler functions.
    """
    return {
        # Sanity check - reconstruct current frame
        "e3_1": e3_1_sanity_check,
        # Single frame future prediction (baseline - failed)
        "e3_2": e3_2_single_frame_prediction,
        # Action-conditioned prediction
        "e3_3": e3_3_action_conditioned_prediction,
        # Architecture investigation experiments (all failed)
        "e3_4": e3_4_multiframe_prediction,      # Multi-frame context
        "e3_5": e3_5_temporal_transformer,       # Temporal transformer
        "e3_6": e3_6_contrastive_prediction,     # Contrastive loss
        # Streaming multi-frame prediction
        "e3_streaming": e3_streaming_prediction,
        # E3.7: True autoregressive with pixel feedback
        "e3_7a": e3_7a,                          # Pixel feedback (frozen VLM)
        "e3_7b": e3_7b,                          # Pixel feedback + VLM LoRA
        # E3.8: PIVOT - Video predicts → VLM describes
        "e3_8a": e3_8a,                          # Video continuation quality
        "e3_8b": e3_8b,                          # Action recognition on generated
        "e3_8c": e3_8c,                          # Description alignment
        "e3_8": e3_8,                            # Run all E3.8 sub-experiments
    }
