"""C1 Experiment Handlers: VLM Latent Sufficiency

This package contains handlers for the C1 experiment which tests whether
Qwen2.5-VL's latent representations contain enough information to reconstruct
input video at reasonable fidelity.

Sub-experiments:
- E1.1: Latent Space Visualization
- E1.2: Reconstruction Probe (Linear)
- E1.3: Pre-merge vs Post-merge Comparison
- E1.4: Spatial Information Test
- E1.5: Full Reconstruction via Video Decoder
- E1.6: Ablation Studies
"""

from typing import Callable

from .example import example_handler
from .e1_1 import e1_1_latent_visualization
from .e1_2 import e1_2_reconstruction_probe
from .e1_3 import e1_3_premerge_vs_postmerge
from .e1_4 import e1_4_spatial_information
from .e1_5 import e1_5_full_reconstruction
from .e1_6 import e1_6_ablations


def get_handlers() -> dict[str, Callable]:
    """Return all handlers for C1 experiment.

    Returns:
        Dict mapping sub-experiment IDs to handler functions.
        Use 'example' to run a minimal test of the infrastructure.
    """
    return {
        # Example handler for testing
        "example": example_handler,
        # Real experiment handlers
        "e1_1": e1_1_latent_visualization,
        "e1_2": e1_2_reconstruction_probe,
        "e1_3": e1_3_premerge_vs_postmerge,
        "e1_4": e1_4_spatial_information,
        "e1_5": e1_5_full_reconstruction,
        "e1_6": e1_6_ablations,
    }
