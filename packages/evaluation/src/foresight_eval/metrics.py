"""Core metrics for video prediction evaluation.

This module provides common metrics used throughout the evaluation framework:
- LPIPS: Perceptual similarity
- SSIM: Structural similarity
- Cosine similarity in latent space
- MSE/MAE for direct comparison

Usage:
    from foresight_eval.metrics import lpips_score, ssim_score, cosine_similarity

    # Compare two images
    lpips = lpips_score(image1, image2)
    ssim = ssim_score(image1, image2)

    # Compare latent vectors
    cos_sim = cosine_similarity(latent1, latent2)
"""

from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image


def to_tensor(
    image: Union[Image.Image, torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert PIL Image to tensor.

    Args:
        image: PIL Image or tensor
        device: Target device

    Returns:
        Tensor of shape [1, C, H, W] normalized to [0, 1]
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if device is not None:
            image = image.to(device)
        return image

    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def lpips_score(
    image1: Union[Image.Image, torch.Tensor],
    image2: Union[Image.Image, torch.Tensor],
    net: str = "alex",
    device: Optional[torch.device] = None,
) -> float:
    """Compute LPIPS perceptual similarity between two images.

    Lower values indicate more similar images.

    Args:
        image1: First image (PIL or tensor)
        image2: Second image (PIL or tensor)
        net: Network to use ('alex', 'vgg', 'squeeze')
        device: Computation device

    Returns:
        LPIPS score (lower = more similar)
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips package required: pip install lpips")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LPIPS model
    lpips_fn = lpips.LPIPS(net=net).to(device)
    lpips_fn.eval()

    # Convert images to tensors
    img1 = to_tensor(image1, device)
    img2 = to_tensor(image2, device)

    # Normalize to [-1, 1] for LPIPS
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    # Resize if needed
    if img1.shape != img2.shape:
        target_size = (min(img1.shape[2], img2.shape[2]), min(img1.shape[3], img2.shape[3]))
        img1 = F.interpolate(img1, size=target_size, mode="bilinear", align_corners=False)
        img2 = F.interpolate(img2, size=target_size, mode="bilinear", align_corners=False)

    with torch.no_grad():
        score = lpips_fn(img1, img2)

    return float(score.item())


def ssim_score(
    image1: Union[Image.Image, torch.Tensor],
    image2: Union[Image.Image, torch.Tensor],
    window_size: int = 11,
    device: Optional[torch.device] = None,
) -> float:
    """Compute SSIM structural similarity between two images.

    Higher values indicate more similar images.

    Args:
        image1: First image (PIL or tensor)
        image2: Second image (PIL or tensor)
        window_size: Gaussian window size
        device: Computation device

    Returns:
        SSIM score (higher = more similar, max 1.0)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img1 = to_tensor(image1, device)
    img2 = to_tensor(image2, device)

    # Resize if needed
    if img1.shape != img2.shape:
        target_size = (min(img1.shape[2], img2.shape[2]), min(img1.shape[3], img2.shape[3]))
        img1 = F.interpolate(img1, size=target_size, mode="bilinear", align_corners=False)
        img2 = F.interpolate(img2, size=target_size, mode="bilinear", align_corners=False)

    return float(_compute_ssim(img1, img2, window_size))


def _compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Compute SSIM using Gaussian window.

    Implementation based on Wang et al. "Image Quality Assessment: From Error
    Visibility to Structural Similarity".
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)

    # Expand for all channels
    channels = img1.shape[1]
    window = window.expand(channels, 1, window_size, window_size)

    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def cosine_similarity(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute cosine similarity between vectors.

    Args:
        vec1: First vector(s)
        vec2: Second vector(s)
        dim: Dimension to compute similarity over

    Returns:
        Cosine similarity (higher = more similar, range [-1, 1])
    """
    vec1_norm = F.normalize(vec1, dim=dim)
    vec2_norm = F.normalize(vec2, dim=dim)
    return (vec1_norm * vec2_norm).sum(dim=dim)


def mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute mean squared error.

    Args:
        pred: Predicted values
        target: Target values
        reduction: 'mean', 'sum', or 'none'

    Returns:
        MSE value
    """
    return F.mse_loss(pred, target, reduction=reduction)


def mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute mean absolute error.

    Args:
        pred: Predicted values
        target: Target values
        reduction: 'mean', 'sum', or 'none'

    Returns:
        MAE value
    """
    return F.l1_loss(pred, target, reduction=reduction)


def psnr(
    image1: Union[Image.Image, torch.Tensor],
    image2: Union[Image.Image, torch.Tensor],
    max_val: float = 1.0,
    device: Optional[torch.device] = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Higher values indicate more similar images.

    Args:
        image1: First image
        image2: Second image
        max_val: Maximum pixel value (1.0 for normalized, 255 for uint8)
        device: Computation device

    Returns:
        PSNR in dB
    """
    import math

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img1 = to_tensor(image1, device)
    img2 = to_tensor(image2, device)

    # Resize if needed
    if img1.shape != img2.shape:
        target_size = (min(img1.shape[2], img2.shape[2]), min(img1.shape[3], img2.shape[3]))
        img1 = F.interpolate(img1, size=target_size, mode="bilinear", align_corners=False)
        img2 = F.interpolate(img2, size=target_size, mode="bilinear", align_corners=False)

    mse_val = F.mse_loss(img1, img2)
    if mse_val == 0:
        return float("inf")

    psnr_val = 20 * math.log10(max_val) - 10 * math.log10(mse_val.item())
    return psnr_val
