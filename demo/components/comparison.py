"""
Comparison View Component

Side-by-side display of input image and predicted outcome.
"""

from typing import Optional

import gradio as gr


def create_comparison_view(config: dict) -> dict:
    """
    Create the before/after comparison component.

    Shows the input image alongside the predicted outcome
    for easy visual comparison.

    Returns:
        Dictionary of component references
    """
    components = {}

    gr.Markdown("#### Input vs Prediction")

    with gr.Row():
        components["input_image"] = gr.Image(
            label="Input (What I Saw)",
            height=150,
            interactive=False,
        )
        components["output_image"] = gr.Image(
            label="Predicted (What I Think Will Happen)",
            height=150,
            interactive=False,
        )

    # Comparison controls
    with gr.Row():
        components["overlay_btn"] = gr.Button("Overlay", size="sm", visible=False)
        components["diff_btn"] = gr.Button("Show Diff", size="sm", visible=False)

    return components


def create_overlay_view(
    input_image: "PIL.Image.Image",
    output_image: "PIL.Image.Image",
    alpha: float = 0.5,
) -> "PIL.Image.Image":
    """
    Create an overlay blend of input and output images.

    Args:
        input_image: The original input image
        output_image: The predicted output image
        alpha: Blend factor (0 = all input, 1 = all output)

    Returns:
        Blended PIL Image
    """
    # TODO: Implement with PIL
    # return Image.blend(input_image, output_image, alpha)
    pass


def create_diff_view(
    input_image: "PIL.Image.Image",
    output_image: "PIL.Image.Image",
) -> "PIL.Image.Image":
    """
    Create a difference map between input and output.

    Highlights regions that changed between input and prediction.

    Args:
        input_image: The original input image
        output_image: The predicted output image

    Returns:
        Difference map as PIL Image
    """
    # TODO: Implement with PIL/numpy
    # diff = np.abs(np.array(input_image) - np.array(output_image))
    # return Image.fromarray(diff)
    pass
