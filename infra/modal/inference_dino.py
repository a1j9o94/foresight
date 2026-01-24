"""DINOv2 feature extraction module for Foresight inference pipeline.

This module provides a standalone DINOv2-ViT-L feature extractor using HuggingFace
transformers. It extracts spatial features that preserve object positions and
relationships, which are critical for the hybrid encoder architecture.

Usage:
    from inference_dino import DINOv2FeatureExtractor

    extractor = DINOv2FeatureExtractor()
    features = extractor.extract_features(image)  # [1, 257, 1024]

Model: facebook/dinov2-large (ViT-L/14)
- Parameters: ~304M
- Feature dim: 1024
- Output: [batch, 257, 1024] (1 CLS token + 256 patch tokens for 224x224 input)
"""

from typing import Union

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOv2FeatureExtractor:
    """DINOv2-ViT-L feature extractor for spatial feature extraction.

    This extractor uses the HuggingFace transformers implementation of DINOv2
    to extract features that preserve spatial information about objects in images.

    Attributes:
        model: The DINOv2-ViT-L model
        processor: The image processor for DINOv2
        device: The device to run inference on
    """

    MODEL_ID = "facebook/dinov2-large"
    FEATURE_DIM = 1024  # ViT-L hidden size
    NUM_TOKENS = 257  # 1 CLS + 256 patches (for 224x224 input with patch_size=14)

    def __init__(
        self,
        device: Union[str, torch.device, None] = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the DINOv2 feature extractor.

        Args:
            device: Device to load the model on. Defaults to CUDA if available.
            cache_dir: Directory to cache the model weights.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self._load_model(cache_dir)

    def _load_model(self, cache_dir: str | None = None) -> None:
        """Load the DINOv2-ViT-L model and processor.

        Args:
            cache_dir: Optional directory to cache model weights.
        """
        print(f"Loading DINOv2-ViT-L ({self.MODEL_ID})...")

        self.processor = AutoImageProcessor.from_pretrained(
            self.MODEL_ID,
            cache_dir=cache_dir,
        )

        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            cache_dir=cache_dir,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"DINOv2-ViT-L loaded: {n_params / 1e6:.1f}M parameters")
        print(f"  Device: {self.device}")
        print(f"  Feature dim: {self.FEATURE_DIM}")
        print(f"  Output shape: [batch, {self.NUM_TOKENS}, {self.FEATURE_DIM}]")

    def extract_features(
        self,
        image: Union[Image.Image, list[Image.Image]],
    ) -> torch.Tensor:
        """Extract spatial features from an image using DINOv2.

        Args:
            image: A PIL Image or list of PIL Images to extract features from.

        Returns:
            Tensor of shape [batch, 257, 1024] containing the last hidden state.
            This includes the CLS token (index 0) and 256 patch tokens.
            For single images, batch=1.
        """
        # Handle single image input
        if isinstance(image, Image.Image):
            images = [image]
        else:
            images = image

        # Ensure RGB format
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Return last hidden state: [batch, 257, 1024]
        return outputs.last_hidden_state

    def extract_patch_features(
        self,
        image: Union[Image.Image, list[Image.Image]],
    ) -> torch.Tensor:
        """Extract only patch features (excluding CLS token).

        This is useful when you need spatial features without the global CLS token,
        for example when computing spatial attention maps or position probing.

        Args:
            image: A PIL Image or list of PIL Images.

        Returns:
            Tensor of shape [batch, 256, 1024] containing patch tokens only.
        """
        features = self.extract_features(image)
        # Exclude CLS token (index 0)
        return features[:, 1:, :]

    def extract_cls_features(
        self,
        image: Union[Image.Image, list[Image.Image]],
    ) -> torch.Tensor:
        """Extract only the CLS token features (global image representation).

        Args:
            image: A PIL Image or list of PIL Images.

        Returns:
            Tensor of shape [batch, 1024] containing the CLS token.
        """
        features = self.extract_features(image)
        # Return CLS token (index 0)
        return features[:, 0, :]


def extract_features(image: Image.Image, device: str = "cuda") -> torch.Tensor:
    """Convenience function to extract DINOv2 features from an image.

    This is a stateless wrapper that creates a new extractor each time.
    For repeated calls, use DINOv2FeatureExtractor class directly.

    Args:
        image: PIL Image to extract features from.
        device: Device to run inference on.

    Returns:
        Tensor of shape [1, 257, 1024].
    """
    extractor = DINOv2FeatureExtractor(device=device)
    return extractor.extract_features(image)


# For direct testing
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("DINOv2 Feature Extraction Test")
    print("=" * 60)

    # Create a test image
    test_image = Image.new("RGB", (224, 224), color=(128, 64, 32))

    # Initialize extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    extractor = DINOv2FeatureExtractor(device=device)

    # Test feature extraction
    print("\nTesting extract_features()...")
    features = extractor.extract_features(test_image)
    print(f"  Output shape: {features.shape}")
    assert features.shape == (1, 257, 1024), f"Expected [1, 257, 1024], got {features.shape}"
    print("  PASS: Shape is correct [1, 257, 1024]")

    # Test patch features
    print("\nTesting extract_patch_features()...")
    patch_features = extractor.extract_patch_features(test_image)
    print(f"  Output shape: {patch_features.shape}")
    assert patch_features.shape == (1, 256, 1024), f"Expected [1, 256, 1024], got {patch_features.shape}"
    print("  PASS: Shape is correct [1, 256, 1024]")

    # Test CLS features
    print("\nTesting extract_cls_features()...")
    cls_features = extractor.extract_cls_features(test_image)
    print(f"  Output shape: {cls_features.shape}")
    assert cls_features.shape == (1, 1024), f"Expected [1, 1024], got {cls_features.shape}"
    print("  PASS: Shape is correct [1, 1024]")

    # Test batch processing
    print("\nTesting batch processing...")
    batch_images = [test_image, test_image, test_image]
    batch_features = extractor.extract_features(batch_images)
    print(f"  Output shape: {batch_features.shape}")
    assert batch_features.shape == (3, 257, 1024), f"Expected [3, 257, 1024], got {batch_features.shape}"
    print("  PASS: Shape is correct [3, 257, 1024]")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
