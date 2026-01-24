"""Streaming prediction models for Foresight training.

This module contains reusable model components for streaming video prediction:
- StreamingPredictor: Recurrent prediction with sliding window and context injection
- FuturePredictionQueries: Learnable queries for future prediction (simpler variant)
- ContextInjection: Dataclass for representing context switches

These models are designed to match the production inference architecture,
so training learns the right behavior for real-time streaming.

Usage:
    from foresight_training.models import StreamingPredictor, ContextInjection

    predictor = StreamingPredictor(
        vlm_dim=3584,
        hidden_dim=512,
        num_queries=32,
    ).to(device)

    # Predict future frames
    predictions = predictor(context_frames, num_future_frames=30)

    # With context injection
    predictions = predictor(
        context_frames,
        new_input=ContextInjection(context=new_context, inject_at=10),
        num_future_frames=30,
    )
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContextInjection:
    """Represents a context injection (hard switch) during prediction.

    Attributes:
        context: New context features to switch to [B, T, D]
        inject_at: Frame index at which to inject the new context
    """
    context: torch.Tensor
    inject_at: int


class FuturePredictionQueries(nn.Module):
    """Learnable query tokens for future prediction.

    This is a simpler version of StreamingPredictor that predicts a single
    future frame rather than a sequence. Useful for ablation studies and
    as a building block.

    The query tokens attend to VLM hidden states via cross-attention
    and learn to extract information relevant to future prediction.
    """

    def __init__(
        self,
        num_queries: int = 32,
        hidden_dim: int = 3584,  # Qwen2.5-VL hidden size
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable query embeddings
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # FFN for each layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, vlm_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_hidden_states: [batch, seq_len, hidden_dim] from VLM

        Returns:
            predicted_future: [batch, num_queries, hidden_dim]
        """
        B = vlm_hidden_states.size(0)

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention layers with FFN
        for attn, norm, ffn, ffn_norm in zip(
            self.cross_attention_layers,
            self.layer_norms,
            self.ffns,
            self.ffn_norms,
        ):
            # Cross-attention
            attn_out, _ = attn(queries, vlm_hidden_states, vlm_hidden_states)
            queries = norm(queries + attn_out)

            # FFN
            ffn_out = ffn(queries)
            queries = ffn_norm(queries + ffn_out)

        # Project to output space
        return self.output_proj(queries)


class StreamingPredictor(nn.Module):
    """Streaming prediction module for recurrent video prediction.

    This module implements a recurrent prediction loop that:
    1. Maintains a sliding context window
    2. Predicts future latent representations
    3. Handles hard context switches when new inputs arrive

    The architecture is designed to match production inference exactly,
    so training learns the right behavior for streaming.
    """

    def __init__(
        self,
        vlm_dim: int = 3584,  # Qwen2.5-VL hidden size
        hidden_dim: int = 512,
        num_queries: int = 32,
        num_layers: int = 3,
        num_heads: int = 8,
        context_window: int = 16,
        dropout: float = 0.1,
    ):
        """Initialize the streaming predictor.

        Args:
            vlm_dim: VLM hidden dimension (Qwen2.5-VL is 3584)
            hidden_dim: Internal hidden dimension for processing
            num_queries: Number of learnable query tokens
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            context_window: Size of sliding context window
            dropout: Dropout probability
        """
        super().__init__()
        self.vlm_dim = vlm_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.context_window = context_window

        # Input projection: VLM features → hidden dim
        self.input_proj = nn.Linear(vlm_dim, hidden_dim)

        # Positional encoding for temporal awareness
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, context_window, hidden_dim) * 0.02
        )

        # Learnable query tokens for prediction
        self.query_tokens = nn.Parameter(
            torch.randn(num_queries, hidden_dim) * 0.02
        )

        # Cross-attention layers for querying context
        self.cross_attention_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Temporal aggregation: combine query outputs into single prediction
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(num_queries * hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Output projection: hidden dim → VLM dim (for loss computation)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, vlm_dim),
        )

        # Recurrent state projection for auto-regressive prediction
        self.recurrent_proj = nn.Linear(hidden_dim, vlm_dim)

    def forward(
        self,
        context_features: torch.Tensor,
        new_input: Optional[ContextInjection] = None,
        num_future_frames: int = 30,
        return_all_predictions: bool = True,
    ) -> torch.Tensor:
        """Predict future frames using recurrent prediction loop.

        Args:
            context_features: Initial context features [B, T, vlm_dim]
            new_input: Optional context injection for hard switch
            num_future_frames: Number of future frames to predict
            return_all_predictions: If True, return all predictions; else just final

        Returns:
            predictions: Predicted latent features [B, num_future_frames, vlm_dim]
                        or [B, vlm_dim] if return_all_predictions=False
        """
        B = context_features.size(0)
        device = context_features.device

        # Project to hidden dim and add temporal position
        current_context = self.input_proj(context_features)  # [B, T, hidden_dim]

        # Ensure context fits in window
        if current_context.size(1) > self.context_window:
            current_context = current_context[:, -self.context_window:]
        elif current_context.size(1) < self.context_window:
            # Pad with zeros if context is shorter
            pad_len = self.context_window - current_context.size(1)
            padding = torch.zeros(B, pad_len, self.hidden_dim, device=device)
            current_context = torch.cat([padding, current_context], dim=1)

        # Add temporal position encoding
        current_context = current_context + self.temporal_pos_encoding

        predictions = []

        for t in range(num_future_frames):
            # Check for context injection (hard switch)
            if new_input is not None and t == new_input.inject_at:
                # Hard switch to new context - no blending
                new_context = self.input_proj(new_input.context)
                if new_context.size(1) > self.context_window:
                    new_context = new_context[:, -self.context_window:]
                elif new_context.size(1) < self.context_window:
                    pad_len = self.context_window - new_context.size(1)
                    padding = torch.zeros(B, pad_len, self.hidden_dim, device=device)
                    new_context = torch.cat([padding, new_context], dim=1)
                current_context = new_context + self.temporal_pos_encoding

            # Predict next frame using cross-attention
            next_latent = self._predict_single_frame(current_context)
            predictions.append(next_latent)

            # Update context with sliding window
            # Convert prediction back to hidden dim for recurrence
            pred_hidden = self.input_proj(next_latent).unsqueeze(1)  # [B, 1, hidden_dim]

            # Slide window: drop oldest, add predicted
            current_context = torch.cat([
                current_context[:, 1:],
                pred_hidden + self.temporal_pos_encoding[:, -1:],
            ], dim=1)

        if return_all_predictions:
            return torch.stack(predictions, dim=1)  # [B, num_future_frames, vlm_dim]
        else:
            return predictions[-1]  # [B, vlm_dim]

    def _predict_single_frame(self, context: torch.Tensor) -> torch.Tensor:
        """Predict the next frame given current context.

        Args:
            context: Current context features [B, context_window, hidden_dim]

        Returns:
            next_latent: Predicted next frame latent [B, vlm_dim]
        """
        B = context.size(0)

        # Expand query tokens for batch
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, hidden_dim]

        # Cross-attention: queries attend to context
        for layer in self.cross_attention_layers:
            queries = layer(queries, context)

        # Flatten queries and aggregate temporally
        queries_flat = queries.reshape(B, -1)  # [B, num_queries * hidden_dim]
        aggregated = self.temporal_aggregator(queries_flat)  # [B, hidden_dim]

        # Project to VLM dimension
        return self.output_proj(aggregated)  # [B, vlm_dim]

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        copy_baseline: Optional[torch.Tensor] = None,
        first_frame_weight: float = 0.0,
        first_frame_target: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss with multiple components.

        Args:
            predictions: Predicted latents [B, T, vlm_dim] or [B, vlm_dim]
            targets: Ground truth latents [B, T, vlm_dim] or [B, vlm_dim]
            copy_baseline: Optional copy baseline for improvement tracking
            first_frame_weight: Weight for first frame reconstruction loss
            first_frame_target: Target for first frame (if different from targets)

        Returns:
            Dict with 'loss', 'cos_sim', 'mse', and optionally 'improvement'
        """
        # Handle both single-frame and multi-frame cases
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(1)
            targets = targets.unsqueeze(1)
            if copy_baseline is not None:
                copy_baseline = copy_baseline.unsqueeze(1)

        # Pool over sequence if multi-frame
        pred_pooled = predictions.mean(dim=1)  # [B, vlm_dim]
        target_pooled = targets.mean(dim=1)

        # Normalize for cosine similarity
        pred_norm = F.normalize(pred_pooled, dim=-1)
        target_norm = F.normalize(target_pooled, dim=-1)

        # Cosine similarity loss (want to maximize, so 1 - cos_sim)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        loss_cos = 1 - cos_sim.mean()

        # MSE loss in normalized space
        loss_mse = F.mse_loss(pred_norm, target_norm)

        # Combined loss
        loss = loss_cos + 0.1 * loss_mse

        # First frame reconstruction loss (for anchoring)
        if first_frame_weight > 0 and first_frame_target is not None:
            first_pred = predictions[:, 0] if predictions.dim() == 3 else predictions
            first_frame_loss = F.mse_loss(first_pred, first_frame_target)
            loss = loss + first_frame_weight * first_frame_loss

        result = {
            'loss': loss,
            'cos_sim': cos_sim.mean(),
            'mse': loss_mse,
        }

        # Compute improvement over copy baseline if provided
        if copy_baseline is not None:
            copy_pooled = copy_baseline.mean(dim=1)
            copy_norm = F.normalize(copy_pooled, dim=-1)
            copy_cos_sim = (copy_norm * target_norm).sum(dim=-1)
            improvement = cos_sim.mean() - copy_cos_sim.mean()
            result['copy_cos_sim'] = copy_cos_sim.mean()
            result['improvement'] = improvement

        return result


class StreamingPredictorConfig:
    """Configuration for StreamingPredictor."""

    def __init__(
        self,
        vlm_dim: int = 3584,
        hidden_dim: int = 512,
        num_queries: int = 32,
        num_layers: int = 3,
        num_heads: int = 8,
        context_window: int = 16,
        dropout: float = 0.1,
    ):
        self.vlm_dim = vlm_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_window = context_window
        self.dropout = dropout

    def create_model(self) -> StreamingPredictor:
        """Create a StreamingPredictor from this config."""
        return StreamingPredictor(
            vlm_dim=self.vlm_dim,
            hidden_dim=self.hidden_dim,
            num_queries=self.num_queries,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            context_window=self.context_window,
            dropout=self.dropout,
        )

    def to_dict(self) -> dict:
        """Convert config to dict for serialization."""
        return {
            'vlm_dim': self.vlm_dim,
            'hidden_dim': self.hidden_dim,
            'num_queries': self.num_queries,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'context_window': self.context_window,
            'dropout': self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'StreamingPredictorConfig':
        """Create config from dict."""
        return cls(**d)


def compute_cosine_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    pool_sequence: bool = True,
) -> torch.Tensor:
    """Compute cosine similarity between prediction and target.

    Args:
        pred: Predicted features [B, T, D] or [B, D]
        target: Target features [B, T, D] or [B, D]
        pool_sequence: Whether to pool over sequence dimension

    Returns:
        Per-sample cosine similarities [B]
    """
    if pool_sequence and pred.dim() == 3:
        pred = pred.mean(dim=1)
        target = target.mean(dim=1)

    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)

    return (pred_norm * target_norm).sum(dim=-1)
