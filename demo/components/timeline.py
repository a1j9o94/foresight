"""
Prediction Timeline Component

Displays a timeline of predicted video frames.
"""

from pathlib import Path
from typing import Optional

import gradio as gr


def create_timeline(config: dict) -> dict:
    """
    Create the prediction timeline component.

    The timeline shows individual frames from the predicted video,
    allowing users to scrub through the prediction.

    Returns:
        Dictionary of component references
    """
    components = {}

    gr.Markdown("#### Prediction Timeline")

    # Frame gallery
    components["gallery"] = gr.Gallery(
        label="Frames",
        columns=6,
        rows=1,
        height=100,
        object_fit="cover",
        allow_preview=True,
    )

    # Frame selection info
    with gr.Row():
        components["frame_info"] = gr.Markdown(
            value="Select a frame to view details",
            visible=True,
        )

    return components


def extract_frames(
    video_path: Path,
    num_frames: int = 6,
    size: tuple = (128, 128),
) -> list:
    """
    Extract frames from a video for timeline display.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        size: Size to resize frames to

    Returns:
        List of PIL Images
    """
    # TODO: Implement with opencv/PIL
    # This is a placeholder for the actual implementation
    return []


def frames_to_gallery_format(
    frames: list,
    timestamps: Optional[list] = None,
) -> list:
    """
    Convert frames to Gradio Gallery format.

    Args:
        frames: List of PIL Images or numpy arrays
        timestamps: Optional list of timestamps for labels

    Returns:
        List of (image, label) tuples for gr.Gallery
    """
    if timestamps is None:
        timestamps = [f"t={i}" for i in range(len(frames))]

    return [(frame, ts) for frame, ts in zip(frames, timestamps)]
