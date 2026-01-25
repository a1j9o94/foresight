"""C4 Experiment Handlers: Pixel Verification Improves Accuracy

This package contains handlers for the C4 experiment which tests whether
comparing predicted video to actual outcomes provides a signal that improves
prediction accuracy.

Architecture (E3.8 Pivot):
    Context frames -> LTX-Video -> Generated future -> VLM describes
                                        |
                              Compare to actual outcome
                                        |
                              Verification signal

Sub-experiments:
- E4.1: Correlation Study - Does LPIPS error predict prediction correctness?
        Generates N predictions, computes LPIPS vs actual, measures correlation
        with VLM action classification correctness.
        Success: Point-biserial r > 0.3, AUROC > 0.65

- E4.2: Calibration Study - Does model uncertainty correlate with error?
        Extracts attention entropy, sampling variance as uncertainty signals.
        Compares to actual LPIPS error.
        Success: ECE < 0.15, Uncertainty-LPIPS r > 0.2

- E4.3: Single Verification Loop - Does feedback improve second attempt?
        Round 1: Generate prediction
        Verification: Compare to actual, generate feedback
        Round 2: Generate with feedback
        Success: Correction rate > 15%, Overall accuracy improvement > 5%

Key Hypothesis:
Perceptual similarity metrics (LPIPS) between predicted and actual video
correlate with task-relevant prediction accuracy, enabling self-correction
through a verification loop.

Dependencies:
- C3 (E3.8 Pivot PASSED) - LTX-Video generates, VLM describes
- Q3 (Temporal Coherence) - Video maintains realistic motion
"""

from typing import Callable

from .e4_1 import e4_1_correlation_study
from .e4_2 import e4_2_calibration_study
from .e4_3 import e4_3_verification_loop


def get_handlers() -> dict[str, Callable]:
    """Return all handlers for C4 experiment.

    Returns:
        Dict mapping sub-experiment IDs to handler functions.
    """
    return {
        # E4.1: Correlation Study - LPIPS vs correctness
        "e4_1": e4_1_correlation_study,
        # E4.2: Calibration Study - uncertainty vs error
        "e4_2": e4_2_calibration_study,
        # E4.3: Single Verification Loop - feedback improves accuracy
        "e4_3": e4_3_verification_loop,
    }
