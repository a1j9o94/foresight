"""E3.3: Action-Conditioned Prediction

Objective: Test whether action descriptions improve prediction accuracy. The same
initial video can lead to different futures depending on the action.

This experiment validates that the VLM's understanding of actions can be leveraged
by query tokens to improve prediction quality. Following the C3 experiment plan,
we use SSv2-style action templates and test action specificity.

Protocol:
1. Generate synthetic videos with labeled actions (SSv2-style: "Pushing [something] from left to right")
2. Train query tokens with action text conditioning via learned embeddings
3. Compare prediction quality: with action vs without action vs wrong (opposite) action
4. Test action specificity: correct action should beat wrong action
5. Evaluate per-action breakdown to ensure model distinguishes directions

Success Criteria (from research_plan.yaml):
- Action Gain > 0.05 (action conditioning improves prediction)
- Action Specificity > 0.1 (correct action beats wrong action)
- Model distinguishes between opposite actions (e.g., push left vs push right)
- Statistical significance (p < 0.01)

Failure Criteria:
- No significant difference with/without action
- Model ignores action, predicts "average" future

Duration: ~1.5 days
"""

import io
import os
import sys

sys.path.insert(0, "/root")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import stats

from runner import ExperimentRunner


class ActionConditionedQueries(nn.Module):
    """Query tokens that can be conditioned on action text embeddings."""

    def __init__(
        self,
        num_queries: int = 32,
        hidden_dim: int = 3584,  # Qwen2.5-VL-7B hidden size
        action_embed_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable query embeddings
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # Action embedding projection
        self.action_proj = nn.Sequential(
            nn.Linear(action_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Action modulation (adds action info to queries)
        self.action_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # FFN for each layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
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

    def forward(
        self,
        vlm_hidden_states: torch.Tensor,
        action_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            vlm_hidden_states: [batch, seq_len, hidden_dim] from VLM
            action_embedding: [batch, action_embed_dim] optional action conditioning

        Returns:
            predicted_future: [batch, num_queries, hidden_dim]
        """
        B = vlm_hidden_states.size(0)

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Add action conditioning if provided
        if action_embedding is not None:
            action_feat = self.action_proj(action_embedding)  # [B, hidden_dim]
            action_gate = self.action_gate(action_feat)  # [B, hidden_dim]
            # Modulate queries with action information
            queries = queries + action_gate.unsqueeze(1) * action_feat.unsqueeze(1)

        # Cross-attention layers
        for attn, norm, ffn, ffn_norm in zip(
            self.cross_attention_layers,
            self.layer_norms,
            self.ffns,
            self.ffn_norms,
        ):
            # Cross-attention to visual context
            attn_out, _ = attn(queries, vlm_hidden_states, vlm_hidden_states)
            queries = norm(queries + attn_out)

            # FFN
            ffn_out = ffn(queries)
            queries = ffn_norm(queries + ffn_out)

        return self.output_proj(queries)


class SimpleActionEncoder(nn.Module):
    """Simple action encoder using learnable embeddings.

    This encodes discrete action IDs into continuous embeddings that
    can be used to modulate the query tokens for action-conditioned prediction.
    """

    def __init__(self, num_actions: int, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embed_dim)
        # Add a small MLP for richer action representations
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(action_ids)
        return self.mlp(embed)


# SSv2-style action labels and descriptions
ACTION_LABELS = {
    "push_right": 0,
    "push_left": 1,
    "push_down": 2,
    "push_up": 3,
    "rotate_cw": 4,
    "rotate_ccw": 5,
}

# SSv2-style action templates (following the experiment plan)
ACTION_TEMPLATES = {
    0: "Pushing [something] from left to right",
    1: "Pushing [something] from right to left",
    2: "Pushing [something] downward",
    3: "Pushing [something] upward",
    4: "Rotating [something] clockwise",
    5: "Rotating [something] counterclockwise",
}

# Full descriptions with object details
ACTION_DESCRIPTIONS = {
    0: "A hand pushes a {color} {shape} from the left side to the right side.",
    1: "A hand pushes a {color} {shape} from the right side to the left side.",
    2: "A hand pushes a {color} {shape} downward toward the bottom.",
    3: "A hand pushes a {color} {shape} upward toward the top.",
    4: "A hand rotates a {color} {shape} in a clockwise direction.",
    5: "A hand rotates a {color} {shape} in a counterclockwise direction.",
}

# Opposite actions for specificity testing
OPPOSITE_ACTIONS = {
    0: 1,  # push_right <-> push_left
    1: 0,
    2: 3,  # push_down <-> push_up
    3: 2,
    4: 5,  # rotate_cw <-> rotate_ccw
    5: 4,
}


def generate_action_conditioned_videos(
    n_videos_per_action: int = 50,
    n_frames: int = 24,
    img_size: int = 224,
) -> list[dict]:
    """Generate synthetic videos with action labels.

    Each video has a clear action that determines the motion direction.
    Includes push actions (4 directions) and rotation actions (CW/CCW).
    """
    samples = []
    shapes = ["circle", "square", "triangle"]
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "magenta": (255, 0, 255),
        "cyan": (0, 255, 255),
    }
    color_names = list(color_map.keys())

    np.random.seed(42)

    for action_name, action_id in ACTION_LABELS.items():
        for video_idx in range(n_videos_per_action):
            shape_name = shapes[video_idx % len(shapes)]
            color_name = color_names[video_idx % len(color_names)]
            color = color_map[color_name]
            size = np.random.randint(20, 40)

            # Start position depends on action type
            if action_name.startswith("push"):
                # For push actions, start from edge to show clear motion
                if action_name == "push_right":
                    start_x = np.random.randint(size + 10, img_size // 3)
                    start_y = np.random.randint(size + 10, img_size - size - 10)
                elif action_name == "push_left":
                    start_x = np.random.randint(2 * img_size // 3, img_size - size - 10)
                    start_y = np.random.randint(size + 10, img_size - size - 10)
                elif action_name == "push_down":
                    start_x = np.random.randint(size + 10, img_size - size - 10)
                    start_y = np.random.randint(size + 10, img_size // 3)
                else:  # push_up
                    start_x = np.random.randint(size + 10, img_size - size - 10)
                    start_y = np.random.randint(2 * img_size // 3, img_size - size - 10)
            else:
                # For rotation, start from center
                start_x = img_size // 2
                start_y = img_size // 2

            # Velocity based on action
            base_velocity = np.random.randint(5, 8)

            frames = []
            for frame_idx in range(n_frames):
                img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
                draw = ImageDraw.Draw(img)

                # Calculate position based on action
                if action_name == "push_right":
                    cx = start_x + frame_idx * base_velocity
                    cy = start_y
                elif action_name == "push_left":
                    cx = start_x - frame_idx * base_velocity
                    cy = start_y
                elif action_name == "push_down":
                    cx = start_x
                    cy = start_y + frame_idx * base_velocity
                elif action_name == "push_up":
                    cx = start_x
                    cy = start_y - frame_idx * base_velocity
                elif action_name == "rotate_cw":
                    # Clockwise rotation around center
                    angle = frame_idx * 15 * np.pi / 180  # 15 degrees per frame
                    radius = 40
                    cx = start_x + int(radius * np.cos(angle))
                    cy = start_y + int(radius * np.sin(angle))
                else:  # rotate_ccw
                    # Counter-clockwise rotation
                    angle = -frame_idx * 15 * np.pi / 180
                    radius = 40
                    cx = start_x + int(radius * np.cos(angle))
                    cy = start_y + int(radius * np.sin(angle))

                # Clamp to bounds
                cx = max(size, min(img_size - size, cx))
                cy = max(size, min(img_size - size, cy))

                # Draw shape
                if shape_name == "circle":
                    draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
                elif shape_name == "square":
                    draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
                elif shape_name == "triangle":
                    points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
                    draw.polygon(points, fill=color)

                frames.append(img)

            # Create action description with object details
            action_desc = ACTION_DESCRIPTIONS[action_id].format(
                color=color_name, shape=shape_name
            )

            samples.append({
                "frames": frames,
                "action_id": action_id,
                "action_name": action_name,
                "action_template": ACTION_TEMPLATES[action_id],
                "action_description": action_desc,
                "color": color_name,
                "shape": shape_name,
            })

    # Shuffle
    np.random.shuffle(samples)
    return samples


def load_vlm_model(device: torch.device):
    """Load Qwen2.5-VL model for feature extraction."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print("  Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, processor


def extract_vlm_features(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
) -> torch.Tensor:
    """Extract VLM hidden states for a single image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]

    return hidden_states[0].float().cpu()


def extract_action_conditioned_data(
    samples: list[dict],
    model,
    processor,
    device: torch.device,
    runner: ExperimentRunner,
    context_frames: int = 16,
    future_frames: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract features for action-conditioned prediction.

    Returns:
        context_features: [n_samples, seq_len, hidden_dim]
        target_features: [n_samples, seq_len, hidden_dim]
        action_ids: [n_samples]
    """
    context_list = []
    target_list = []
    action_ids = []

    for idx, sample in enumerate(samples):
        frames = sample["frames"]

        # Context: frame 16
        context_frame = frames[context_frames - 1]
        context_feat = extract_vlm_features(context_frame, model, processor, device)

        # Target: frame 17-24 (we use frame 20 as middle of future)
        target_frame = frames[context_frames + future_frames // 2]
        target_feat = extract_vlm_features(target_frame, model, processor, device)

        context_list.append(context_feat)
        target_list.append(target_feat)
        action_ids.append(sample["action_id"])

        if (idx + 1) % 20 == 0:
            progress = (idx + 1) / len(samples)
            runner.log_metrics({"e3_3/extraction_progress": progress})
            print(f"    Extracted {idx + 1}/{len(samples)} samples")

    # Pad to same length
    max_len = max(
        max(f.shape[0] for f in context_list),
        max(f.shape[0] for f in target_list)
    )
    hidden_dim = context_list[0].shape[1]

    def pad_features(features_list):
        padded = []
        for f in features_list:
            if f.shape[0] < max_len:
                padding = torch.zeros(max_len - f.shape[0], hidden_dim)
                f = torch.cat([f, padding], dim=0)
            padded.append(f)
        return torch.stack(padded)

    return (
        pad_features(context_list),
        pad_features(target_list),
        torch.tensor(action_ids, dtype=torch.long),
    )


def compute_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute per-sample cosine similarity."""
    pred_pooled = pred.mean(dim=1)
    target_pooled = target.mean(dim=1)

    pred_norm = F.normalize(pred_pooled, dim=-1)
    target_norm = F.normalize(target_pooled, dim=-1)

    return (pred_norm * target_norm).sum(dim=-1)


def get_opposite_action(action_id: int) -> int:
    """Get the opposite action for testing action specificity.

    Uses the OPPOSITE_ACTIONS mapping which pairs:
    - push_right <-> push_left
    - push_down <-> push_up
    - rotate_cw <-> rotate_ccw
    """
    return OPPOSITE_ACTIONS[action_id]


def compute_per_action_metrics(
    sim_with: torch.Tensor,
    sim_without: torch.Tensor,
    sim_wrong: torch.Tensor,
    action_ids: torch.Tensor,
) -> dict:
    """Compute metrics broken down by action type.

    Returns dict mapping action_name -> {cos_sim_with, cos_sim_without, cos_sim_wrong, gain, specificity}
    """
    # Convert to numpy
    sim_with_np = sim_with.cpu().numpy()
    sim_without_np = sim_without.cpu().numpy()
    sim_wrong_np = sim_wrong.cpu().numpy()
    action_ids_np = action_ids.cpu().numpy()

    # Invert ACTION_LABELS for lookup
    id_to_name = {v: k for k, v in ACTION_LABELS.items()}

    per_action = {}
    for action_id in np.unique(action_ids_np):
        mask = action_ids_np == action_id
        action_name = id_to_name[action_id]

        with_sims = sim_with_np[mask]
        without_sims = sim_without_np[mask]
        wrong_sims = sim_wrong_np[mask]

        per_action[action_name] = {
            "cos_sim_with_action": float(np.mean(with_sims)),
            "cos_sim_without_action": float(np.mean(without_sims)),
            "cos_sim_wrong_action": float(np.mean(wrong_sims)),
            "action_gain": float(np.mean(with_sims - without_sims)),
            "action_specificity": float(np.mean(with_sims - wrong_sims)),
            "n_samples": int(np.sum(mask)),
        }

    return per_action


def create_results_plot(
    losses: list[float],
    eval_history: list[dict],
    per_action_metrics: dict | None = None,
) -> bytes:
    """Create visualization of action conditioning results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Loss curve
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")

    if eval_history:
        steps = [e["step"] for e in eval_history]

        # With vs without action
        with_action = [e["with_action_cos_sim"] for e in eval_history]
        without_action = [e["without_action_cos_sim"] for e in eval_history]

        axes[0, 1].plot(steps, with_action, label="With Action", marker="o")
        axes[0, 1].plot(steps, without_action, label="Without Action", marker="s")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Cosine Similarity")
        axes[0, 1].set_title("Action Conditioning Effect")
        axes[0, 1].legend()

        # Action gain over time
        action_gains = [e["action_gain"] for e in eval_history]
        axes[1, 0].plot(steps, action_gains, marker="o", color="purple")
        axes[1, 0].axhline(y=0.05, color="g", linestyle="--", alpha=0.5, label="Target (0.05)")
        axes[1, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Action Gain")
        axes[1, 0].set_title("Action Conditioning Gain")
        axes[1, 0].legend()

        # Correct vs wrong action
        if "wrong_action_cos_sim" in eval_history[-1]:
            correct = [e["with_action_cos_sim"] for e in eval_history]
            wrong = [e.get("wrong_action_cos_sim", 0) for e in eval_history]

            axes[1, 1].plot(steps, correct, label="Correct Action", marker="o")
            axes[1, 1].plot(steps, wrong, label="Wrong Action", marker="x")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Cosine Similarity")
            axes[1, 1].set_title("Action Specificity")
            axes[1, 1].legend()

    # Per-action breakdown
    if per_action_metrics:
        action_names = list(per_action_metrics.keys())
        n_actions = len(action_names)
        x = np.arange(n_actions)
        width = 0.25

        gains = [per_action_metrics[a]["action_gain"] for a in action_names]
        specificities = [per_action_metrics[a]["action_specificity"] for a in action_names]

        axes[0, 2].bar(x - width/2, gains, width, label="Action Gain", color="#2ecc71")
        axes[0, 2].bar(x + width/2, specificities, width, label="Specificity", color="#9b59b6")
        axes[0, 2].axhline(y=0.05, color="#2ecc71", linestyle="--", alpha=0.5)
        axes[0, 2].axhline(y=0.1, color="#9b59b6", linestyle="--", alpha=0.5)
        axes[0, 2].set_xlabel("Action Type")
        axes[0, 2].set_ylabel("Improvement")
        axes[0, 2].set_title("Per-Action: Gain and Specificity")
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(action_names, rotation=45, ha="right", fontsize=8)
        axes[0, 2].legend(fontsize=8)

        # Per-action cos_sim comparison
        with_sims = [per_action_metrics[a]["cos_sim_with_action"] for a in action_names]
        without_sims = [per_action_metrics[a]["cos_sim_without_action"] for a in action_names]
        wrong_sims = [per_action_metrics[a]["cos_sim_wrong_action"] for a in action_names]

        width_small = 0.2
        axes[1, 2].bar(x - width_small, with_sims, width_small, label="Correct", color="#2ecc71")
        axes[1, 2].bar(x, without_sims, width_small, label="No Action", color="#3498db")
        axes[1, 2].bar(x + width_small, wrong_sims, width_small, label="Wrong", color="#e74c3c")
        axes[1, 2].set_xlabel("Action Type")
        axes[1, 2].set_ylabel("Cosine Similarity")
        axes[1, 2].set_title("Per-Action: Prediction Quality")
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(action_names, rotation=45, ha="right", fontsize=8)
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].set_ylim(0, 1)

    plt.suptitle("E3.3: Action-Conditioned Prediction Results", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def e3_3_action_conditioned_prediction(runner: ExperimentRunner) -> dict:
    """Run E3.3: Action-conditioned prediction.

    This experiment tests whether action text descriptions improve future
    prediction accuracy.

    Args:
        runner: ExperimentRunner instance

    Returns:
        Dict with finding, metrics, and artifacts
    """
    print("=" * 60)
    print("E3.3: Action-Conditioned Prediction")
    print("=" * 60)

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.log_metrics({"e3_3/stage": 0, "e3_3/progress": 0.0})

    # =========================================================================
    # Stage 1: Generate action-conditioned videos
    # =========================================================================
    print("\n[Stage 1/5] Generating action-conditioned videos...")

    n_per_action = 20  # 20 videos per action * 6 actions = 120 total
    samples = generate_action_conditioned_videos(
        n_videos_per_action=n_per_action,
        n_frames=24,
    )

    # Split into train/val
    n_train = int(0.8 * len(samples))
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    print(f"  Generated {len(samples)} videos across {len(ACTION_LABELS)} actions")
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    runner.log_metrics({
        "e3_3/stage": 1,
        "e3_3/progress": 0.1,
        "e3_3/n_train": len(train_samples),
        "e3_3/n_val": len(val_samples),
        "e3_3/n_actions": len(ACTION_LABELS),
    })

    # =========================================================================
    # Stage 2: Load VLM and extract features
    # =========================================================================
    print("\n[Stage 2/5] Loading VLM and extracting features...")

    vlm_model, vlm_processor = load_vlm_model(device)

    print("  Extracting training features...")
    train_context, train_target, train_actions = extract_action_conditioned_data(
        train_samples, vlm_model, vlm_processor, device, runner
    )

    print("  Extracting validation features...")
    val_context, val_target, val_actions = extract_action_conditioned_data(
        val_samples, vlm_model, vlm_processor, device, runner
    )

    hidden_dim = train_context.shape[-1]
    print(f"  Train context: {train_context.shape}")
    print(f"  Hidden dim: {hidden_dim}")

    runner.log_metrics({
        "e3_3/stage": 2,
        "e3_3/progress": 0.4,
        "e3_3/hidden_dim": hidden_dim,
    })

    # Free VLM memory
    del vlm_model, vlm_processor
    torch.cuda.empty_cache()

    # =========================================================================
    # Stage 3: Initialize models
    # =========================================================================
    print("\n[Stage 3/5] Initializing models...")

    action_embed_dim = 256
    action_encoder = SimpleActionEncoder(
        num_actions=len(ACTION_LABELS),
        embed_dim=action_embed_dim,
    ).to(device)

    query_model = ActionConditionedQueries(
        num_queries=32,
        hidden_dim=hidden_dim,
        action_embed_dim=action_embed_dim,
        num_layers=3,
        num_heads=8,
    ).to(device)

    n_params = (
        sum(p.numel() for p in query_model.parameters()) +
        sum(p.numel() for p in action_encoder.parameters())
    )
    print(f"  Total params: {n_params:,} ({n_params/1e6:.2f}M)")

    runner.log_metrics({
        "e3_3/stage": 3,
        "e3_3/progress": 0.45,
        "e3_3/total_params": n_params,
    })

    # =========================================================================
    # Stage 4: Train with action conditioning
    # =========================================================================
    print("\n[Stage 4/5] Training with action conditioning...")

    train_context_t = train_context.to(device)
    train_target_t = train_target.to(device)
    train_actions_t = train_actions.to(device)

    val_context_t = val_context.to(device)
    val_target_t = val_target.to(device)
    val_actions_t = val_actions.to(device)

    n_steps = 5000
    batch_size = 8
    lr = 1e-4

    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(action_encoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, eta_min=1e-6)

    losses = []
    eval_history = []

    for step in range(n_steps):
        query_model.train()
        action_encoder.train()

        # Sample batch
        batch_idx = torch.randint(0, len(train_context_t), (batch_size,))
        batch_context = train_context_t[batch_idx]
        batch_target = train_target_t[batch_idx]
        batch_actions = train_actions_t[batch_idx]

        # Forward with action conditioning
        optimizer.zero_grad()
        action_embed = action_encoder(batch_actions)
        predicted = query_model(batch_context, action_embed)

        # Compute loss
        pred_pooled = predicted.mean(dim=1)
        target_pooled = batch_target.mean(dim=1)

        pred_norm = F.normalize(pred_pooled, dim=-1)
        target_norm = F.normalize(target_pooled, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        loss = 1 - cos_sim.mean()

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(query_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        # Evaluate periodically
        if (step + 1) % 500 == 0 or step == 0:
            query_model.eval()
            action_encoder.eval()

            with torch.no_grad():
                # With action conditioning
                val_action_embed = action_encoder(val_actions_t)
                val_pred_with = query_model(val_context_t, val_action_embed)
                sim_with = compute_cosine_similarity(val_pred_with, val_target_t)

                # Without action conditioning
                val_pred_without = query_model(val_context_t, None)
                sim_without = compute_cosine_similarity(val_pred_without, val_target_t)

                # With wrong action
                wrong_actions = torch.tensor(
                    [get_opposite_action(a.item()) for a in val_actions_t],
                    device=device,
                )
                wrong_action_embed = action_encoder(wrong_actions)
                val_pred_wrong = query_model(val_context_t, wrong_action_embed)
                sim_wrong = compute_cosine_similarity(val_pred_wrong, val_target_t)

                with_mean = float(sim_with.mean().item())
                without_mean = float(sim_without.mean().item())
                wrong_mean = float(sim_wrong.mean().item())
                action_gain = with_mean - without_mean
                action_specificity = with_mean - wrong_mean

                eval_history.append({
                    "step": step + 1,
                    "with_action_cos_sim": with_mean,
                    "without_action_cos_sim": without_mean,
                    "wrong_action_cos_sim": wrong_mean,
                    "action_gain": action_gain,
                    "action_specificity": action_specificity,
                })

            print(f"    Step {step + 1}/{n_steps}: "
                  f"With={with_mean:.4f}, Without={without_mean:.4f}, Wrong={wrong_mean:.4f}, "
                  f"Gain={action_gain:+.4f}, Specificity={action_specificity:+.4f}")

            runner.log_metrics({
                "e3_3/loss": loss.item(),
                "e3_3/with_action_cos_sim": with_mean,
                "e3_3/without_action_cos_sim": without_mean,
                "e3_3/wrong_action_cos_sim": wrong_mean,
                "e3_3/action_gain": action_gain,
                "e3_3/action_specificity": action_specificity,
            }, step=step)

    runner.log_metrics({
        "e3_3/stage": 4,
        "e3_3/progress": 0.9,
    })

    # =========================================================================
    # Stage 5: Final evaluation and statistical tests
    # =========================================================================
    print("\n[Stage 5/5] Final evaluation...")

    query_model.eval()
    action_encoder.eval()

    with torch.no_grad():
        # Final predictions
        val_action_embed = action_encoder(val_actions_t)
        val_pred_with = query_model(val_context_t, val_action_embed)
        val_pred_without = query_model(val_context_t, None)

        wrong_actions = torch.tensor(
            [get_opposite_action(a.item()) for a in val_actions_t],
            device=device,
        )
        wrong_embed = action_encoder(wrong_actions)
        val_pred_wrong = query_model(val_context_t, wrong_embed)

        sim_with = compute_cosine_similarity(val_pred_with, val_target_t)
        sim_without = compute_cosine_similarity(val_pred_without, val_target_t)
        sim_wrong = compute_cosine_similarity(val_pred_wrong, val_target_t)

    sim_with_np = sim_with.cpu().numpy()
    sim_without_np = sim_without.cpu().numpy()
    sim_wrong_np = sim_wrong.cpu().numpy()

    # Statistical tests
    t_stat_gain, p_value_gain = stats.ttest_rel(sim_with_np, sim_without_np)
    t_stat_spec, p_value_spec = stats.ttest_rel(sim_with_np, sim_wrong_np)

    # Final metrics
    final_with = float(np.mean(sim_with_np))
    final_without = float(np.mean(sim_without_np))
    final_wrong = float(np.mean(sim_wrong_np))
    final_action_gain = final_with - final_without
    final_action_specificity = final_with - final_wrong

    print(f"  With action: {final_with:.4f}")
    print(f"  Without action: {final_without:.4f}")
    print(f"  Wrong action: {final_wrong:.4f}")
    print(f"  Action gain: {final_action_gain:+.4f} (p={p_value_gain:.4f})")
    print(f"  Action specificity: {final_action_specificity:+.4f} (p={p_value_spec:.4f})")

    # Compute per-action metrics
    per_action_metrics = compute_per_action_metrics(
        sim_with, sim_without, sim_wrong, val_actions_t
    )

    print("\n  Per-action breakdown:")
    for action_name, metrics in per_action_metrics.items():
        print(f"    {action_name}: gain={metrics['action_gain']:+.3f}, "
              f"spec={metrics['action_specificity']:+.3f}, n={metrics['n_samples']}")

    # Create visualization with per-action breakdown
    viz_bytes = create_results_plot(losses, eval_history, per_action_metrics)
    viz_path = runner.results.save_artifact("action_conditioned_prediction.png", viz_bytes)

    # Save results
    results_data = {
        "final_with_action_cos_sim": final_with,
        "final_without_action_cos_sim": final_without,
        "final_wrong_action_cos_sim": final_wrong,
        "action_gain": final_action_gain,
        "action_specificity": final_action_specificity,
        "p_value_gain": float(p_value_gain),
        "p_value_specificity": float(p_value_spec),
        "action_gain_significant": p_value_gain < 0.01,
        "specificity_significant": p_value_spec < 0.01,
        "per_action_metrics": per_action_metrics,
        "per_sample_with": sim_with_np.tolist(),
        "per_sample_without": sim_without_np.tolist(),
        "per_sample_wrong": sim_wrong_np.tolist(),
    }
    data_path = runner.results.save_json_artifact("action_conditioned_results.json", results_data)

    runner.log_metrics({
        "e3_3/stage": 5,
        "e3_3/progress": 1.0,
        "e3_3/final_action_gain": final_action_gain,
        "e3_3/final_action_specificity": final_action_specificity,
        "e3_3/p_value_gain": p_value_gain,
    })

    # =========================================================================
    # Determine finding
    # =========================================================================
    gain_target = 0.05
    specificity_target = 0.1

    success = (
        final_action_gain > gain_target and
        final_action_specificity > 0 and  # Correct should beat wrong
        p_value_gain < 0.01
    )

    if success and final_action_specificity > specificity_target:
        finding = (
            f"ACTION CONDITIONING SUCCESSFUL: Action text significantly improves prediction "
            f"(gain={final_action_gain:+.3f} > {gain_target}, p={p_value_gain:.4f}). "
            f"Model distinguishes correct vs wrong actions (specificity={final_action_specificity:+.3f}). "
            f"VLM's action understanding successfully leveraged for future prediction."
        )
    elif success:
        finding = (
            f"ACTION CONDITIONING PARTIAL SUCCESS: Action text improves prediction "
            f"(gain={final_action_gain:+.3f} > {gain_target}), but action specificity is limited "
            f"({final_action_specificity:+.3f} < {specificity_target}). "
            f"Consider stronger action representations."
        )
    elif final_action_gain > 0:
        finding = (
            f"ACTION CONDITIONING MARGINAL: Action text shows small positive effect "
            f"(gain={final_action_gain:+.3f}), but below target ({gain_target}). "
            f"Model may be ignoring some action information."
        )
    else:
        finding = (
            f"ACTION CONDITIONING FAILED: Action text does not improve prediction "
            f"(gain={final_action_gain:+.3f} <= 0). "
            f"Model ignores action conditioning. Investigate integration method."
        )

    print(f"\n{finding}")
    print("=" * 60)

    return {
        "finding": finding,
        "metrics": {
            "action_gain": final_action_gain,
            "action_specificity": final_action_specificity,
            "with_action_cos_sim": final_with,
            "without_action_cos_sim": final_without,
            "wrong_action_cos_sim": final_wrong,
            "p_value_gain": float(p_value_gain),
            "p_value_specificity": float(p_value_spec),
            "passed": success,
        },
        "artifacts": [viz_path, data_path],
    }
