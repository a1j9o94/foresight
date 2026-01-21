"""
Chat Interface Component

Provides the main chat interface for user interaction.
"""

from typing import Optional

import gradio as gr


def create_chat_interface(config: dict) -> tuple:
    """
    Create the chat interface components.

    Returns:
        Tuple of (chatbot, text_input, image_input, submit_btn, clear_btn)
    """
    chatbot = gr.Chatbot(
        label="Chat",
        height=500,
        show_copy_button=True,
        bubble_full_width=False,
    )

    with gr.Row():
        image_input = gr.Image(
            label="Upload Image",
            type="pil",
            height=100,
            scale=1,
        )
        with gr.Column(scale=4):
            text_input = gr.Textbox(
                label="Message",
                placeholder="Ask about what will happen...",
                lines=2,
            )

    with gr.Row():
        clear_btn = gr.Button("Clear", size="sm")
        submit_btn = gr.Button("Send", variant="primary")

    return chatbot, text_input, image_input, submit_btn, clear_btn


def format_user_message(message: str, image: Optional["PIL.Image.Image"] = None) -> str:
    """Format a user message for display."""
    if image:
        return f"[Image uploaded]\n{message}"
    return message


def format_assistant_message(text: str, prediction_available: bool = False) -> str:
    """Format an assistant message for display."""
    if prediction_available:
        return f"{text}\n\n*[Visual prediction available in Thoughts panel]*"
    return text
