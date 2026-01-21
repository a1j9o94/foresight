"""Temporal Coherence Metrics for Q3 Experiment.

This module implements the temporal consistency metrics defined in the Q3 experiment plan:
- Flow smoothness: Optical flow acceleration variance
- Temporal LPIPS variance: Frame-to-frame perceptual consistency
- Identity preservation: DINO feature correlation across frames
- Warping accuracy: How well optical flow predicts next frame

The combined temporal_consistency score is:
    0.30 * flow_smoothness + 0.25 * temporal_lpips + 0.25 * identity + 0.20 * warp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class TemporalMetricsResult:
    """Results from temporal metrics computation."""
    temporal_consistency: float
    flow_smoothness: float
    temporal_lpips_score: float  # 1 - normalized variance
    identity_preservation: float
    warp_accuracy: float

    # Raw values for debugging
    flow_acceleration_variance: float
    temporal_lpips_variance: float
    warp_error: float

    def to_dict(self) -> dict:
        return {
            "temporal_consistency": self.temporal_consistency,
            "flow_smoothness": self.flow_smoothness,
            "temporal_lpips_score": self.temporal_lpips_score,
            "identity_preservation": self.identity_preservation,
            "warp_accuracy": self.warp_accuracy,
            "flow_acceleration_variance": self.flow_acceleration_variance,
            "temporal_lpips_variance": self.temporal_lpips_variance,
            "warp_error": self.warp_error,
        }


class TemporalMetrics:
    """Compute temporal coherence metrics for generated videos.

    This class manages model loading and provides efficient batch computation
    of temporal consistency metrics.
    """

    # Normalization constants (empirically determined)
    MAX_ACCEL = 5.0  # Maximum expected flow acceleration
    MAX_LPIPS_VAR = 0.05  # Maximum expected LPIPS variance
    MAX_WARP_ERROR = 0.15  # Maximum expected warping error

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize temporal metrics with required models.

        Args:
            device: Device to use for computation. Auto-detected if None.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded models
        self._raft_model = None
        self._lpips_model = None
        self._dino_model = None

    @property
    def raft_model(self):
        """Lazy-load RAFT optical flow model."""
        if self._raft_model is None:
            print("  Loading RAFT optical flow model...")
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self._raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            self._raft_model = self._raft_model.to(self.device)
            self._raft_model.eval()
        return self._raft_model

    @property
    def lpips_model(self):
        """Lazy-load LPIPS model."""
        if self._lpips_model is None:
            print("  Loading LPIPS model...")
            import lpips
            self._lpips_model = lpips.LPIPS(net='alex')
            self._lpips_model = self._lpips_model.to(self.device)
            self._lpips_model.eval()
        return self._lpips_model

    @property
    def dino_model(self):
        """Lazy-load DINOv2 model for identity preservation."""
        if self._dino_model is None:
            print("  Loading DINOv2 model...")
            self._dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self._dino_model = self._dino_model.to(self.device)
            self._dino_model.eval()
        return self._dino_model

    def compute_optical_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> torch.Tensor:
        """Compute optical flow between two frames using RAFT.

        Args:
            frame1: First frame [C, H, W] or [B, C, H, W] in range [0, 1]
            frame2: Second frame [C, H, W] or [B, C, H, W] in range [0, 1]

        Returns:
            Optical flow tensor [B, 2, H, W]
        """
        # Ensure batch dimension
        if frame1.dim() == 3:
            frame1 = frame1.unsqueeze(0)
            frame2 = frame2.unsqueeze(0)

        # RAFT expects images in range [0, 255]
        frame1_scaled = (frame1 * 255).to(self.device)
        frame2_scaled = (frame2 * 255).to(self.device)

        # Resize to multiple of 8 for RAFT
        _, _, h, w = frame1_scaled.shape
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8

        if new_h != h or new_w != w:
            frame1_scaled = F.interpolate(frame1_scaled, (new_h, new_w), mode='bilinear', align_corners=False)
            frame2_scaled = F.interpolate(frame2_scaled, (new_h, new_w), mode='bilinear', align_corners=False)

        with torch.no_grad():
            # RAFT returns list of flow predictions, use the last one
            flows = self.raft_model(frame1_scaled, frame2_scaled)
            flow = flows[-1]  # [B, 2, H, W]

        return flow

    def flow_smoothness_score(self, video: torch.Tensor) -> tuple[float, float]:
        """Compute flow smoothness from optical flow acceleration.

        Lower acceleration variance = smoother motion.

        Args:
            video: Video tensor [T, C, H, W] in range [0, 1]

        Returns:
            Tuple of (smoothness_score, acceleration_variance)
            Smoothness score is normalized to [0, 1], higher is better.
        """
        T = video.shape[0]

        if T < 4:
            # Need at least 4 frames for acceleration
            return 1.0, 0.0

        # Compute optical flow for consecutive frames
        flows = []
        for t in range(T - 1):
            flow = self.compute_optical_flow(video[t], video[t + 1])
            flows.append(flow)

        # Compute velocity (flow difference between consecutive flows)
        velocities = []
        for t in range(len(flows) - 1):
            vel = flows[t + 1] - flows[t]
            velocities.append(vel)

        # Compute acceleration (velocity difference)
        accelerations = []
        for t in range(len(velocities) - 1):
            acc = velocities[t + 1] - velocities[t]
            accelerations.append(acc)

        if not accelerations:
            return 1.0, 0.0

        # Compute acceleration magnitude variance
        acc_magnitudes = []
        for acc in accelerations:
            mag = torch.norm(acc, dim=1).mean().item()  # Mean over spatial dims
            acc_magnitudes.append(mag)

        mean_acc = np.mean(acc_magnitudes)

        # Normalize: higher score = smoother
        smoothness = max(0.0, 1.0 - mean_acc / self.MAX_ACCEL)

        return smoothness, mean_acc

    def temporal_lpips_variance_score(self, video: torch.Tensor) -> tuple[float, float]:
        """Compute temporal LPIPS variance.

        High variance in LPIPS between consecutive frames indicates
        inconsistent appearance.

        Args:
            video: Video tensor [T, C, H, W] in range [0, 1]

        Returns:
            Tuple of (consistency_score, lpips_variance)
            Score is normalized to [0, 1], higher is better (lower variance).
        """
        T = video.shape[0]

        if T < 2:
            return 1.0, 0.0

        # Compute LPIPS between consecutive frames
        lpips_scores = []
        for t in range(T - 1):
            frame1 = video[t:t+1].to(self.device)
            frame2 = video[t+1:t+2].to(self.device)

            # LPIPS expects [-1, 1] range
            frame1 = frame1 * 2 - 1
            frame2 = frame2 * 2 - 1

            with torch.no_grad():
                score = self.lpips_model(frame1, frame2).item()
            lpips_scores.append(score)

        variance = np.var(lpips_scores)

        # Also compute mean LPIPS for reference
        mean_lpips = np.mean(lpips_scores)

        # Normalize: higher score = more consistent (lower variance)
        consistency = max(0.0, 1.0 - variance / self.MAX_LPIPS_VAR)

        return consistency, variance

    def identity_preservation_score(self, video: torch.Tensor) -> float:
        """Compute identity preservation using DINO features.

        High feature correlation across frames indicates consistent
        object/scene appearance.

        Args:
            video: Video tensor [T, C, H, W] in range [0, 1]

        Returns:
            Mean pairwise DINO feature correlation (0-1, higher is better)
        """
        from torchvision import transforms

        T = video.shape[0]

        if T < 2:
            return 1.0

        # DINOv2 preprocessing
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Extract features for each frame
        features = []
        for t in range(T):
            frame = video[t:t+1].to(self.device)
            # Resize to DINOv2 expected size
            frame = F.interpolate(frame, (224, 224), mode='bilinear', align_corners=False)
            frame = normalize(frame)

            with torch.no_grad():
                feat = self.dino_model(frame)  # [1, D]
            features.append(feat.squeeze())

        features = torch.stack(features)  # [T, D]

        # Compute pairwise cosine similarities between consecutive frames
        similarities = []
        for t in range(T - 1):
            sim = F.cosine_similarity(
                features[t:t+1],
                features[t+1:t+2]
            ).item()
            similarities.append(sim)

        return np.mean(similarities)

    def warping_accuracy_score(self, video: torch.Tensor) -> tuple[float, float]:
        """Compute warping accuracy using optical flow prediction.

        Measures how well the optical flow can predict the next frame.
        Lower warping error indicates more coherent motion.

        Args:
            video: Video tensor [T, C, H, W] in range [0, 1]

        Returns:
            Tuple of (accuracy_score, warp_error)
            Score is normalized to [0, 1], higher is better (lower error).
        """
        T = video.shape[0]

        if T < 2:
            return 1.0, 0.0

        errors = []
        for t in range(T - 1):
            frame1 = video[t].to(self.device)
            frame2 = video[t + 1].to(self.device)

            # Compute flow from frame1 to frame2
            flow = self.compute_optical_flow(frame1, frame2)  # [1, 2, H, W]

            # Warp frame1 using the flow
            warped = self._warp_frame(frame1.unsqueeze(0), flow)  # [1, C, H, W]

            # Compute L1 error between warped frame1 and actual frame2
            frame2_resized = F.interpolate(
                frame2.unsqueeze(0),
                warped.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            error = F.l1_loss(warped, frame2_resized).item()
            errors.append(error)

        mean_error = np.mean(errors)

        # Normalize: higher score = more accurate (lower error)
        accuracy = max(0.0, 1.0 - mean_error / self.MAX_WARP_ERROR)

        return accuracy, mean_error

    def _warp_frame(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp a frame using optical flow.

        Args:
            frame: Frame tensor [B, C, H, W]
            flow: Optical flow [B, 2, H, W]

        Returns:
            Warped frame [B, C, H, W]
        """
        B, C, H, W = frame.shape

        # Resize frame to flow size if needed
        _, _, fH, fW = flow.shape
        if H != fH or W != fW:
            frame = F.interpolate(frame, (fH, fW), mode='bilinear', align_corners=False)
            H, W = fH, fW

        # Create base grid
        y, x = torch.meshgrid(
            torch.arange(H, device=flow.device),
            torch.arange(W, device=flow.device),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=-1).float()  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # Add flow to grid
        flow_permuted = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        new_grid = grid + flow_permuted

        # Normalize grid to [-1, 1]
        new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0

        # Warp using grid_sample
        warped = F.grid_sample(
            frame,
            new_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped

    def temporal_consistency(self, video: torch.Tensor) -> TemporalMetricsResult:
        """Compute combined temporal consistency score.

        This is the primary metric for Q3 experiment.

        Args:
            video: Video tensor [T, C, H, W] in range [0, 1]

        Returns:
            TemporalMetricsResult with all component scores and combined score.
        """
        # Compute all component metrics
        flow_smoothness, acc_var = self.flow_smoothness_score(video)
        lpips_score, lpips_var = self.temporal_lpips_variance_score(video)
        identity = self.identity_preservation_score(video)
        warp_acc, warp_err = self.warping_accuracy_score(video)

        # Weighted combination (from experiment plan)
        combined = (
            0.30 * flow_smoothness +
            0.25 * lpips_score +
            0.25 * identity +
            0.20 * warp_acc
        )

        return TemporalMetricsResult(
            temporal_consistency=combined,
            flow_smoothness=flow_smoothness,
            temporal_lpips_score=lpips_score,
            identity_preservation=identity,
            warp_accuracy=warp_acc,
            flow_acceleration_variance=acc_var,
            temporal_lpips_variance=lpips_var,
            warp_error=warp_err,
        )

    def batch_temporal_consistency(
        self,
        videos: list[torch.Tensor],
        verbose: bool = True
    ) -> list[TemporalMetricsResult]:
        """Compute temporal consistency for multiple videos.

        Args:
            videos: List of video tensors, each [T, C, H, W]
            verbose: Whether to print progress

        Returns:
            List of TemporalMetricsResult for each video.
        """
        results = []
        for i, video in enumerate(videos):
            if verbose and (i + 1) % 10 == 0:
                print(f"    Processing video {i + 1}/{len(videos)}")
            result = self.temporal_consistency(video)
            results.append(result)
        return results


class SemanticAccuracy:
    """Compute semantic accuracy using CLIP similarity to prompt.

    Used in E-Q3.2 to measure semantic control vs temporal coherence tradeoff.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_model = None
        self._clip_processor = None

    def _load_clip(self):
        """Load CLIP model on demand."""
        if self._clip_model is None:
            print("  Loading CLIP model...")
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()

    def score(self, video: torch.Tensor, prompt: str) -> float:
        """Compute CLIP similarity between video frames and text prompt.

        Args:
            video: Video tensor [T, C, H, W] in range [0, 1]
            prompt: Text description

        Returns:
            Mean CLIP similarity across frames (0-1)
        """
        self._load_clip()

        T = video.shape[0]
        similarities = []

        # Sample frames (max 8 for efficiency)
        sample_indices = np.linspace(0, T - 1, min(8, T), dtype=int)

        for t in sample_indices:
            frame = video[t]
            # Convert to PIL for CLIP processor
            frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            from PIL import Image
            frame_pil = Image.fromarray(frame_np)

            # Process inputs
            inputs = self._clip_processor(
                text=[prompt],
                images=frame_pil,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                # Get image-text similarity
                logits = outputs.logits_per_image  # [1, 1]
                sim = torch.sigmoid(logits / 100).item()  # Normalize to ~[0, 1]

            similarities.append(sim)

        return np.mean(similarities)
