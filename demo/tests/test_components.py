"""
Tests for demo UI components.
"""

import pytest

from demo.components.chat import format_user_message, format_assistant_message
from demo.components.thoughts import update_metrics
from demo.components.timeline import frames_to_gallery_format


class TestChatFormatting:
    """Tests for chat message formatting."""

    def test_format_user_message_text_only(self):
        """Test formatting a text-only message."""
        result = format_user_message("Hello world")
        assert result == "Hello world"

    def test_format_user_message_with_image(self):
        """Test formatting a message with image."""
        result = format_user_message("What is this?", image="mock_image")
        assert "[Image uploaded]" in result
        assert "What is this?" in result

    def test_format_assistant_message_without_prediction(self):
        """Test assistant message without prediction."""
        result = format_assistant_message("The answer is 42")
        assert result == "The answer is 42"
        assert "prediction" not in result.lower()

    def test_format_assistant_message_with_prediction(self):
        """Test assistant message with prediction available."""
        result = format_assistant_message("The answer is 42", prediction_available=True)
        assert "The answer is 42" in result
        assert "Thoughts panel" in result


class TestMetricsUpdate:
    """Tests for metrics formatting."""

    def test_update_metrics_all_values(self):
        """Test formatting all metrics."""
        conf, lpips, latency = update_metrics(
            confidence=0.87,
            lpips=0.162,
            latency=2.5,
        )

        assert conf == "0.87"
        assert lpips == "0.162"
        assert latency == "2.50s"

    def test_update_metrics_none_values(self):
        """Test formatting with None values."""
        conf, lpips, latency = update_metrics()

        assert conf == "--"
        assert lpips == "--"
        assert latency == "--"

    def test_update_metrics_partial_values(self):
        """Test formatting with some None values."""
        conf, lpips, latency = update_metrics(confidence=0.5)

        assert conf == "0.50"
        assert lpips == "--"
        assert latency == "--"


class TestTimeline:
    """Tests for timeline component."""

    def test_frames_to_gallery_format_with_timestamps(self):
        """Test converting frames to gallery format."""
        frames = ["frame1", "frame2", "frame3"]
        timestamps = ["t=0", "t=1", "t=2"]

        result = frames_to_gallery_format(frames, timestamps)

        assert len(result) == 3
        assert result[0] == ("frame1", "t=0")
        assert result[2] == ("frame3", "t=2")

    def test_frames_to_gallery_format_auto_timestamps(self):
        """Test automatic timestamp generation."""
        frames = ["frame1", "frame2"]

        result = frames_to_gallery_format(frames)

        assert len(result) == 2
        assert result[0][1] == "t=0"
        assert result[1][1] == "t=1"

    def test_frames_to_gallery_format_empty(self):
        """Test with empty frame list."""
        result = frames_to_gallery_format([])
        assert result == []
