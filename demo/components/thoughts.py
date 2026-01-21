"""
Thoughts Panel Component

Displays the model's visual predictions and internal states.
"""

from typing import Optional

import gradio as gr


def create_thoughts_panel(config: dict) -> dict:
    """
    Create the thoughts visualization panel.

    This panel shows:
    - Current video prediction
    - Quality metrics (confidence, LPIPS, latency)
    - Frame timeline
    - Before/after comparison

    Returns:
        Dictionary of component references
    """
    components = {}

    gr.Markdown("### Model's Predicted Future")

    # Main prediction display
    components["video"] = gr.Video(
        label="Current Prediction",
        height=300,
        autoplay=config["ui"]["auto_play_predictions"],
    )

    # Metrics row
    if config["ui"]["show_metrics"]:
        with gr.Row():
            components["confidence"] = gr.Textbox(
                label="Confidence",
                value="--",
                interactive=False,
                scale=1,
            )
            components["lpips"] = gr.Textbox(
                label="LPIPS",
                value="--",
                interactive=False,
                scale=1,
            )
            components["latency"] = gr.Textbox(
                label="Latency",
                value="--",
                interactive=False,
                scale=1,
            )

    # Advanced latent visualization (hidden by default)
    if config["ui"].get("show_latents", False):
        with gr.Accordion("Internal Representations", open=False):
            components["attention_map"] = gr.Image(
                label="Attention Heatmap",
                height=150,
            )
            components["query_tokens"] = gr.Textbox(
                label="Query Token Stats",
                value="",
                interactive=False,
            )

    return components


def update_metrics(
    confidence: Optional[float] = None,
    lpips: Optional[float] = None,
    latency: Optional[float] = None,
) -> tuple:
    """
    Format metrics for display.

    Returns:
        Tuple of (confidence_str, lpips_str, latency_str)
    """
    conf_str = f"{confidence:.2f}" if confidence is not None else "--"
    lpips_str = f"{lpips:.3f}" if lpips is not None else "--"
    latency_str = f"{latency:.2f}s" if latency is not None else "--"

    return conf_str, lpips_str, latency_str
