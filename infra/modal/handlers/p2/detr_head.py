"""DETR-style Detection Head for P2 Experiments

This module provides a full DETR architecture for object detection from DINOv2 features:
- 6 encoder layers (processes spatial features with positional encoding)
- 6 decoder layers (processes learnable query embeddings)
- Hungarian matching for training
- GIoU + L1 box loss + focal classification loss

Usage:
    from detr_head import DETRDetectionHead, HungarianMatcher, detr_loss

    head = DETRDetectionHead(input_dim=1024, n_classes=12)
    boxes, class_logits = head(features)  # features: [B, N_patches, 1024]

    matcher = HungarianMatcher()
    loss = detr_loss(boxes, class_logits, gt_boxes, gt_labels, matcher)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional


class PositionalEncoding2D(nn.Module):
    """2D sine positional encoding for spatial patch features.

    Creates positional encodings for a grid of patches (e.g., 16x16 for 224x224 image
    with 14x14 patch size from DINOv2).
    """

    def __init__(self, d_model: int, max_h: int = 16, max_w: int = 16, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        self.temperature = temperature

        # Pre-compute positional encodings
        pe = self._make_pe(max_h, max_w, d_model)
        self.register_buffer('pe', pe)

    def _make_pe(self, h: int, w: int, d_model: int) -> torch.Tensor:
        """Generate 2D positional encoding."""
        y_embed = torch.arange(h, dtype=torch.float32).unsqueeze(1).expand(h, w)
        x_embed = torch.arange(w, dtype=torch.float32).unsqueeze(0).expand(h, w)

        # Normalize to [0, 1]
        y_embed = y_embed / h
        x_embed = x_embed / w

        dim_t = torch.arange(d_model // 2, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (d_model // 2))

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        # Interleave sin and cos
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)

        # Concatenate x and y
        pe = torch.cat([pos_x, pos_y], dim=2)  # [H, W, d_model]

        # Flatten to [H*W, d_model]
        return pe.view(-1, d_model)

    def forward(self, n_patches: int) -> torch.Tensor:
        """Return positional encoding for n_patches.

        Args:
            n_patches: Number of spatial patches (should be H*W)

        Returns:
            [n_patches, d_model] positional encoding
        """
        return self.pe[:n_patches]


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DETRDetectionHead(nn.Module):
    """Full DETR-style detection head for object detection from spatial features.

    Architecture:
    - Input projection: Maps input features to hidden_dim
    - Positional encoding: 2D sine encoding for spatial awareness
    - Transformer encoder: 6 layers processing spatial features
    - Transformer decoder: 6 layers processing learnable queries
    - Box head: 3-layer MLP predicting (x_center, y_center, width, height)
    - Class head: Linear layer predicting class logits + no-object class
    """

    def __init__(
        self,
        input_dim: int = 1024,      # ViT-L feature dimension
        hidden_dim: int = 256,
        num_queries: int = 20,
        n_classes: int = 12,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.n_classes = n_classes

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding for spatial patches
        self.pos_encoder = PositionalEncoding2D(hidden_dim)

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Prediction heads
        self.box_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.class_head = nn.Linear(hidden_dim, n_classes + 1)  # +1 for no-object

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: [B, N_patches, input_dim] DINOv2 spatial features

        Returns:
            boxes: [B, num_queries, 4] predicted boxes (x_c, y_c, w, h) in [0, 1]
            class_logits: [B, num_queries, n_classes + 1] class predictions
        """
        B, N, _ = features.shape

        # Project to hidden dim
        src = self.input_proj(features)  # [B, N, hidden_dim]

        # Add positional encoding
        pos = self.pos_encoder(N).unsqueeze(0).expand(B, -1, -1)  # [B, N, hidden_dim]
        src = src + pos

        # Encode spatial features
        memory = self.encoder(src)  # [B, N, hidden_dim]

        # Decode with learnable queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, hidden_dim]
        hs = self.decoder(queries, memory)  # [B, num_queries, hidden_dim]

        # Predict boxes and classes
        # Fix: Use sigmoid for center coords (valid [0,1]) but softplus for w,h (always positive)
        # This fixes the issue where sigmoid squashes width/height targets (0.2-0.5) incorrectly
        raw = self.box_head(hs)
        cx = raw[..., 0].sigmoid()      # Center x: [0,1]
        cy = raw[..., 1].sigmoid()      # Center y: [0,1]
        w = F.softplus(raw[..., 2]).clamp(max=1.0)   # Width: (0,1]
        h = F.softplus(raw[..., 3]).clamp(max=1.0)   # Height: (0,1]
        boxes = torch.stack([cx, cy, w, h], dim=-1)  # [B, num_queries, 4]
        class_logits = self.class_head(hs)  # [B, num_queries, n_classes + 1]

        return boxes, class_logits


class HungarianMatcher(nn.Module):
    """Hungarian matcher for DETR-style bipartite matching.

    Matches predictions to ground truth based on a cost matrix that combines:
    - Classification cost (cross-entropy like)
    - L1 box cost
    - GIoU cost
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_boxes: torch.Tensor,      # [B, num_queries, 4]
        pred_logits: torch.Tensor,     # [B, num_queries, n_classes + 1]
        gt_boxes: List[torch.Tensor],  # List of [N_gt, 4] per image
        gt_labels: List[torch.Tensor], # List of [N_gt] per image
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute optimal assignment between predictions and ground truth.

        Returns:
            List of (pred_indices, gt_indices) tuples for each image in batch
        """
        B, num_queries = pred_boxes.shape[:2]

        indices = []
        for b in range(B):
            if len(gt_boxes[b]) == 0:
                indices.append((
                    torch.tensor([], dtype=torch.int64, device=pred_boxes.device),
                    torch.tensor([], dtype=torch.int64, device=pred_boxes.device),
                ))
                continue

            # Get predictions for this image
            out_prob = pred_logits[b].softmax(-1)  # [num_queries, n_classes + 1]
            out_bbox = pred_boxes[b]  # [num_queries, 4]

            # Get ground truth
            tgt_bbox = gt_boxes[b]  # [N_gt, 4]
            tgt_labels = gt_labels[b]  # [N_gt]

            # Classification cost: negative probability of correct class
            cost_class = -out_prob[:, tgt_labels]  # [num_queries, N_gt]

            # L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [num_queries, N_gt]

            # GIoU cost
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox),
            )  # [num_queries, N_gt]

            # Final cost matrix
            C = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )

            # Hungarian algorithm
            C_np = C.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C_np)

            indices.append((
                torch.tensor(row_ind, dtype=torch.int64, device=pred_boxes.device),
                torch.tensor(col_ind, dtype=torch.int64, device=pred_boxes.device),
            ))

        return indices


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute generalized IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in (x1, y1, x2, y2) format
        boxes2: [M, 4] boxes in (x1, y1, x2, y2) format

    Returns:
        [N, M] GIoU matrix
    """
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    union = area1[:, None] + area2[None, :] - inter  # [N, M]

    # IoU
    iou = inter / (union + 1e-6)

    # Enclosing box
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh_enc = rb_enc - lt_enc  # [N, M, 2]
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]  # [N, M]

    # GIoU
    giou = iou - (area_enc - union) / (area_enc + 1e-6)

    return giou


def detr_loss(
    pred_boxes: torch.Tensor,       # [B, num_queries, 4]
    pred_logits: torch.Tensor,      # [B, num_queries, n_classes + 1]
    gt_boxes: List[torch.Tensor],   # List of [N_gt, 4] per image
    gt_labels: List[torch.Tensor],  # List of [N_gt] per image
    matcher: HungarianMatcher,
    n_classes: int,
    eos_coef: float = 0.1,          # Weight for no-object class
    loss_bbox_weight: float = 5.0,
    loss_giou_weight: float = 2.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute DETR loss.

    Args:
        pred_boxes: Predicted boxes [B, num_queries, 4] in (cx, cy, w, h) format
        pred_logits: Predicted class logits [B, num_queries, n_classes + 1]
        gt_boxes: Ground truth boxes, list of [N_gt, 4] per image
        gt_labels: Ground truth labels, list of [N_gt] per image
        matcher: Hungarian matcher
        n_classes: Number of object classes
        eos_coef: Weight for no-object (background) class
        loss_bbox_weight: Weight for L1 box loss
        loss_giou_weight: Weight for GIoU loss

    Returns:
        total_loss: Scalar loss
        loss_dict: Dictionary with individual loss components
    """
    device = pred_boxes.device
    B = pred_boxes.shape[0]

    # Match predictions to ground truth
    indices = matcher(pred_boxes, pred_logits, gt_boxes, gt_labels)

    # Classification loss
    # Create targets: all predictions map to "no object" (class n_classes) by default
    target_classes = torch.full(
        (B, pred_logits.shape[1]),
        n_classes,  # no-object class index
        dtype=torch.int64,
        device=device,
    )

    # Set matched predictions to their target class
    for b, (pred_idx, gt_idx) in enumerate(indices):
        if len(pred_idx) > 0:
            target_classes[b, pred_idx] = gt_labels[b][gt_idx]

    # Class weights: down-weight the no-object class
    weight = torch.ones(n_classes + 1, device=device)
    weight[-1] = eos_coef

    loss_ce = F.cross_entropy(
        pred_logits.transpose(1, 2),  # [B, n_classes+1, num_queries]
        target_classes,  # [B, num_queries]
        weight=weight,
    )

    # Box losses (only for matched predictions)
    loss_bbox = torch.tensor(0.0, device=device)
    loss_giou = torch.tensor(0.0, device=device)
    num_boxes = 0

    for b, (pred_idx, gt_idx) in enumerate(indices):
        if len(pred_idx) == 0:
            continue

        src_boxes = pred_boxes[b, pred_idx]  # [N_matched, 4]
        tgt_boxes = gt_boxes[b][gt_idx]  # [N_matched, 4]

        # L1 loss
        loss_bbox = loss_bbox + F.l1_loss(src_boxes, tgt_boxes, reduction='sum')

        # GIoU loss
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )
        loss_giou = loss_giou + (1 - giou.diag()).sum()

        num_boxes += len(pred_idx)

    # Normalize by number of boxes
    num_boxes = max(num_boxes, 1)
    loss_bbox = loss_bbox / num_boxes
    loss_giou = loss_giou / num_boxes

    # Total loss
    total_loss = loss_ce + loss_bbox_weight * loss_bbox + loss_giou_weight * loss_giou

    loss_dict = {
        'loss_ce': loss_ce.item(),
        'loss_bbox': loss_bbox.item(),
        'loss_giou': loss_giou.item(),
        'total_loss': total_loss.item(),
    }

    return total_loss, loss_dict
