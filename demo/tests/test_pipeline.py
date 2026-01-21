"""
Tests for the demo pipeline.
"""

import pytest

from demo.pipeline.demo_pipeline import MockPipeline, PredictionResult


class TestMockPipeline:
    """Tests for MockPipeline."""

    @pytest.fixture
    def mock_config(self):
        """Minimal config for testing."""
        return {
            "dev": {
                "mock_pipeline": True,
                "mock_latency_ms": 100,  # Fast for tests
                "sample_video_path": None,
            }
        }

    def test_mock_pipeline_initialization(self, mock_config):
        """Test MockPipeline can be initialized."""
        pipeline = MockPipeline(mock_config)
        assert pipeline.mock_latency == 0.1  # 100ms in seconds

    def test_predict_streaming_yields_results(self, mock_config):
        """Test that predict_streaming yields results."""
        pipeline = MockPipeline(mock_config)

        results = list(pipeline.predict_streaming("What will happen?"))

        # Should yield multiple results (progress stages + final)
        assert len(results) >= 2

        # Final result should have text
        final = results[-1]
        assert "text" in final
        assert final["text"] is not None

    def test_predict_streaming_with_image(self, mock_config):
        """Test prediction with image input."""
        pipeline = MockPipeline(mock_config)

        # Mock image (just need to pass something)
        mock_image = "placeholder"

        results = list(pipeline.predict_streaming("What happens?", image=mock_image))
        final = results[-1]

        # Should return the image as final_frame in mock mode
        assert final["final_frame"] == mock_image

    def test_predict_streaming_responds_to_keywords(self, mock_config):
        """Test that responses vary based on keywords."""
        pipeline = MockPipeline(mock_config)

        # Test 'pour' keyword
        pour_results = list(pipeline.predict_streaming("What if I pour water?"))
        pour_text = pour_results[-1]["text"]
        assert "liquid" in pour_text.lower() or "flow" in pour_text.lower()

        # Test 'push' keyword
        push_results = list(pipeline.predict_streaming("What if I push this?"))
        push_text = push_results[-1]["text"]
        assert "move" in push_text.lower() or "force" in push_text.lower()

    def test_predict_streaming_includes_metrics(self, mock_config):
        """Test that final result includes metrics."""
        pipeline = MockPipeline(mock_config)

        results = list(pipeline.predict_streaming("Test message"))
        final = results[-1]

        assert final["confidence"] is not None
        assert final["lpips"] is not None
        assert final["latency"] is not None


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_default_frames_is_list(self):
        """Test that frames defaults to empty list."""
        result = PredictionResult(text="test")
        assert result.frames == []
        assert result.video_path is None

    def test_full_result(self):
        """Test creating a full result."""
        result = PredictionResult(
            text="Prediction text",
            video_path="/path/to/video.mp4",
            frames=[1, 2, 3],
            confidence=0.85,
            lpips=0.18,
            latency=2.5,
        )

        assert result.text == "Prediction text"
        assert result.confidence == 0.85
        assert len(result.frames) == 3
