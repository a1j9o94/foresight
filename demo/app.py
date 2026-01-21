"""
Foresight Demo Application

A chat interface with visual "thoughts" - showing the model's predicted future states.

Run with:
    python demo/app.py

Or with custom config:
    python demo/app.py --config demo/config.yaml
"""

import argparse
from pathlib import Path
from typing import Generator, Optional

import gradio as gr
import yaml

# Demo components (to be implemented)
from demo.components.chat import create_chat_interface
from demo.components.thoughts import create_thoughts_panel
from demo.pipeline.demo_pipeline import DemoPipeline, MockPipeline

# Configuration
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(config_path: Path) -> dict:
    """Load demo configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_demo_app(config: dict) -> gr.Blocks:
    """
    Create the main Gradio application.

    Layout:
    +----------------------------------+
    |            HEADER                |
    +----------------+-----------------+
    |   CHAT PANEL   | THOUGHTS PANEL  |
    |     (60%)      |     (40%)       |
    +----------------+-----------------+
    |          STATUS BAR              |
    +----------------------------------+
    """

    # Initialize pipeline
    if config.get("dev", {}).get("mock_pipeline", False):
        pipeline = MockPipeline(config)
    else:
        pipeline = DemoPipeline(config)

    # Custom CSS
    custom_css = """
    .thoughts-panel {
        border-left: 1px solid #e0e0e0;
        padding-left: 16px;
    }
    .prediction-video {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metrics-badge {
        font-size: 0.85em;
        padding: 4px 8px;
        border-radius: 4px;
        background: #f0f0f0;
    }
    .timeline-frame {
        cursor: pointer;
        transition: transform 0.2s;
    }
    .timeline-frame:hover {
        transform: scale(1.05);
    }
    """

    # Build the interface
    with gr.Blocks(
        title=config["ui"]["title"],
        theme=gr.themes.Soft() if config["ui"]["theme"] == "soft" else gr.themes.Default(),
        css=custom_css,
    ) as demo:

        # State management
        session_state = gr.State({
            "predictions": [],
            "current_image": None,
        })

        # Header
        with gr.Row():
            gr.Markdown(f"# {config['ui']['title']}")
            gr.Markdown(f"*{config['ui']['description']}*", elem_classes=["description"])
            with gr.Column(scale=0, min_width=100):
                settings_btn = gr.Button("Settings", size="sm")

        # Main content area
        with gr.Row():
            # Left panel: Chat
            with gr.Column(scale=config["ui"]["chat_panel_width"]):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_copy_button=True,
                    avatar_images=(None, "demo/static/assets/avatar.png"),
                )

                with gr.Row():
                    image_input = gr.Image(
                        label="Upload Image (optional)",
                        type="pil",
                        height=100,
                        scale=1,
                    )
                    with gr.Column(scale=4):
                        text_input = gr.Textbox(
                            label="Message",
                            placeholder="Ask about the future... e.g., 'What will happen if I push this ball?'",
                            lines=2,
                        )

                with gr.Row():
                    clear_btn = gr.Button("Clear", size="sm")
                    submit_btn = gr.Button("Send", variant="primary")

            # Right panel: Thoughts visualization
            with gr.Column(
                scale=config["ui"]["thoughts_panel_width"],
                elem_classes=["thoughts-panel"],
            ):
                gr.Markdown("### Model's Predicted Future")

                # Current prediction display
                prediction_video = gr.Video(
                    label="Current Prediction",
                    height=300,
                    autoplay=config["ui"]["auto_play_predictions"],
                    elem_classes=["prediction-video"],
                )

                # Metrics
                if config["ui"]["show_metrics"]:
                    with gr.Row():
                        confidence_display = gr.Textbox(
                            label="Confidence",
                            value="--",
                            interactive=False,
                            scale=1,
                        )
                        lpips_display = gr.Textbox(
                            label="LPIPS",
                            value="--",
                            interactive=False,
                            scale=1,
                        )
                        latency_display = gr.Textbox(
                            label="Latency",
                            value="--",
                            interactive=False,
                            scale=1,
                        )

                # Timeline
                gr.Markdown("#### Prediction Timeline")
                timeline_gallery = gr.Gallery(
                    label="Frames",
                    columns=6,
                    rows=1,
                    height=100,
                    object_fit="cover",
                    elem_classes=["timeline-frame"],
                )

                # Before/After comparison
                gr.Markdown("#### Input vs Prediction")
                with gr.Row():
                    input_image_display = gr.Image(
                        label="Input",
                        height=150,
                        interactive=False,
                    )
                    output_frame_display = gr.Image(
                        label="Predicted",
                        height=150,
                        interactive=False,
                    )

        # Status bar
        with gr.Row():
            status_text = gr.Markdown(
                f"**Model:** {config['model']['vlm']} + {config['model']['video_decoder']} | "
                f"**Status:** Ready"
            )

        # Event handlers
        def on_submit(
            message: str,
            image: Optional["PIL.Image.Image"],
            history: list,
            state: dict,
        ) -> Generator:
            """Handle chat submission with streaming response."""
            if not message.strip():
                yield history, None, "--", "--", "--", [], None, None, state
                return

            # Add user message to history
            history = history + [[message, None]]
            yield history, None, "--", "--", "Processing...", [], image, None, state

            # Generate response and prediction
            for result in pipeline.predict_streaming(message, image, state):
                # Update chat with streaming text
                history[-1][1] = result.get("text", "")

                # Update prediction displays
                video = result.get("video")
                confidence = result.get("confidence", "--")
                lpips = result.get("lpips", "--")
                latency = result.get("latency", "--")
                frames = result.get("frames", [])
                final_frame = result.get("final_frame")

                yield (
                    history,
                    video,
                    str(confidence),
                    str(lpips),
                    str(latency),
                    frames,
                    image,
                    final_frame,
                    state,
                )

        def on_clear():
            """Clear conversation and predictions."""
            return [], None, "--", "--", "--", [], None, None, {"predictions": [], "current_image": None}

        # Wire up events
        submit_btn.click(
            on_submit,
            inputs=[text_input, image_input, chatbot, session_state],
            outputs=[
                chatbot,
                prediction_video,
                confidence_display,
                lpips_display,
                latency_display,
                timeline_gallery,
                input_image_display,
                output_frame_display,
                session_state,
            ],
        )

        text_input.submit(
            on_submit,
            inputs=[text_input, image_input, chatbot, session_state],
            outputs=[
                chatbot,
                prediction_video,
                confidence_display,
                lpips_display,
                latency_display,
                timeline_gallery,
                input_image_display,
                output_frame_display,
                session_state,
            ],
        )

        clear_btn.click(
            on_clear,
            outputs=[
                chatbot,
                prediction_video,
                confidence_display,
                lpips_display,
                latency_display,
                timeline_gallery,
                input_image_display,
                output_frame_display,
                session_state,
            ],
        )

    return demo


def main():
    """Entry point for the demo application."""
    parser = argparse.ArgumentParser(description="Foresight Demo UI")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with CLI args
    if args.share:
        config["server"]["share"] = True
    if args.debug:
        config["server"]["debug"] = True

    # Create and launch app
    demo = create_demo_app(config)
    demo.launch(
        server_name=config["server"]["host"],
        server_port=config["server"]["port"],
        share=config["server"]["share"],
        debug=config["server"]["debug"],
        auth=config["server"]["auth"],
    )


if __name__ == "__main__":
    main()
