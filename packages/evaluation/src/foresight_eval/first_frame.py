"""First frame fidelity evaluation.

This module provides metrics for evaluating how well generated video
first frames match the conditioning input image.

Usage:
    from foresight_eval.first_frame import evaluate_first_frame_fidelity

    result = evaluate_first_frame_fidelity(
        input_image=conditioning_image,
        generated_frame_0=video.frames[0],
    )
    print(f"LPIPS: {result['lpips']}")
    print(f"SSIM: {result['ssim']}")
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from PIL import Image

from .metrics import lpips_score, ssim_score, psnr, to_tensor


@dataclass
class FirstFrameMetrics:
    """Metrics for first frame fidelity evaluation."""
    lpips: float
    ssim: float
    psnr: float
    mse: float
    passed_threshold: bool
    threshold_used: float


def evaluate_first_frame_fidelity(
    input_image: Union[Image.Image, torch.Tensor],
    generated_frame_0: Union[Image.Image, torch.Tensor],
    lpips_threshold: float = 0.1,
    device: Optional[torch.device] = None,
) -> FirstFrameMetrics:
    """Evaluate how well the generated first frame matches the input image.

    Args:
        input_image: The conditioning input image
        generated_frame_0: The first frame of the generated video
        lpips_threshold: LPIPS threshold for pass/fail (default 0.1)
        device: Computation device

    Returns:
        FirstFrameMetrics with all computed metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    input_t = to_tensor(input_image, device)
    gen_t = to_tensor(generated_frame_0, device)

    # Resize generated to match input if needed
    if input_t.shape[2:] != gen_t.shape[2:]:
        import torch.nn.functional as F
        gen_t = F.interpolate(
            gen_t,
            size=input_t.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    # Compute metrics
    lpips = lpips_score(input_image, generated_frame_0, device=device)
    ssim = ssim_score(input_image, generated_frame_0, device=device)
    psnr_val = psnr(input_image, generated_frame_0, device=device)

    # MSE
    mse = float(torch.nn.functional.mse_loss(input_t, gen_t).item())

    return FirstFrameMetrics(
        lpips=lpips,
        ssim=ssim,
        psnr=psnr_val,
        mse=mse,
        passed_threshold=lpips < lpips_threshold,
        threshold_used=lpips_threshold,
    )


def batch_evaluate_first_frame(
    input_images: list[Union[Image.Image, torch.Tensor]],
    generated_first_frames: list[Union[Image.Image, torch.Tensor]],
    lpips_threshold: float = 0.1,
    device: Optional[torch.device] = None,
) -> dict:
    """Evaluate first frame fidelity for a batch of videos.

    Args:
        input_images: List of conditioning input images
        generated_first_frames: List of first frames from generated videos
        lpips_threshold: LPIPS threshold for pass/fail
        device: Computation device

    Returns:
        Dict with aggregated metrics and per-sample results
    """
    if len(input_images) != len(generated_first_frames):
        raise ValueError("Number of input images must match number of generated frames")

    results = []
    for inp, gen in zip(input_images, generated_first_frames):
        metrics = evaluate_first_frame_fidelity(
            inp, gen, lpips_threshold=lpips_threshold, device=device
        )
        results.append(metrics)

    # Aggregate
    lpips_scores = [r.lpips for r in results]
    ssim_scores = [r.ssim for r in results]
    psnr_scores = [r.psnr for r in results]
    pass_rate = sum(1 for r in results if r.passed_threshold) / len(results)

    return {
        "mean_lpips": sum(lpips_scores) / len(lpips_scores),
        "mean_ssim": sum(ssim_scores) / len(ssim_scores),
        "mean_psnr": sum(psnr_scores) / len(psnr_scores),
        "pass_rate": pass_rate,
        "num_samples": len(results),
        "threshold": lpips_threshold,
        "per_sample": results,
    }
