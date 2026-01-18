"""Q1 Experiment Handlers: Latent Space Alignment

This package contains handlers for the Q1 experiment which analyzes the geometric
structure of VLM (Qwen2.5-VL) and video decoder (LTX-Video) latent spaces, and
determines if they can be bridged with a small adapter.

Sub-experiments:
- E-Q1.1: VLM latent space visualization
- E-Q1.2: LTX-Video latent space visualization
- E-Q1.3: Intrinsic dimensionality measurement
- E-Q1.4: Linear probing (predict one space from other)
- E-Q1.5: Semantic similarity preservation test
- E-Q1.6: Neighborhood analysis (cross-space retrieval)
- E-Q1.7: CKA analysis (optional)

Success Criteria:
- Linear probe R^2 > 0.5 (good: > 0.7)
- Spearman correlation > 0.6 (excellent: > 0.8)
- Neighborhood Recall@10 > 20% (strong: > 40%)
- CKA > 0.4
"""

from typing import Callable

from .eq1_1_vlm_visualization import eq1_1_vlm_visualization
from .eq1_2_ltx_visualization import eq1_2_ltx_visualization
from .eq1_3_intrinsic_dimensionality import eq1_3_intrinsic_dimensionality
from .eq1_4_linear_probe import eq1_4_linear_probe
from .eq1_5_semantic_similarity import eq1_5_semantic_similarity
from .eq1_6_neighborhood_analysis import eq1_6_neighborhood_analysis
from .eq1_7_cka_analysis import eq1_7_cka_analysis


def get_handlers() -> dict[str, Callable]:
    """Return all handlers for Q1 experiment.

    Returns:
        Dict mapping sub-experiment IDs to handler functions.
    """
    return {
        # Core visualization experiments
        "eq1_1": eq1_1_vlm_visualization,
        "eq1_2": eq1_2_ltx_visualization,
        # Dimensionality analysis
        "eq1_3": eq1_3_intrinsic_dimensionality,
        # Alignment experiments
        "eq1_4": eq1_4_linear_probe,
        "eq1_5": eq1_5_semantic_similarity,
        "eq1_6": eq1_6_neighborhood_analysis,
        # Optional
        "eq1_7": eq1_7_cka_analysis,
    }
