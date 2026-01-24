"""Tests for Something-Something v2 dataset loader."""

import pytest
import torch

from foresight_training.data import SSv2Dataset, VideoSample


class TestSSv2Dataset:
    """Tests for SSv2Dataset class."""

    @pytest.fixture
    def small_dataset(self) -> SSv2Dataset:
        """Load a small subset of SSv2 for testing."""
        return SSv2Dataset(
            split="validation",
            subset_size=5,
            num_frames=8,
            frame_size=(112, 112),
            mock_videos=True,
        )

    def test_dataset_creation(self, small_dataset: SSv2Dataset) -> None:
        """Test that dataset can be created."""
        assert small_dataset is not None
        assert len(small_dataset) == 5

    def test_getitem_returns_video_sample(self, small_dataset: SSv2Dataset) -> None:
        """Test that __getitem__ returns a VideoSample dataclass."""
        sample = small_dataset[0]
        assert isinstance(sample, VideoSample)
        assert hasattr(sample, "frames")
        assert hasattr(sample, "label")
        assert hasattr(sample, "label_id")
        assert hasattr(sample, "metadata")

    def test_video_sample_has_correct_frame_shape(
        self, small_dataset: SSv2Dataset
    ) -> None:
        """Test that frames have correct shape (T, C, H, W)."""
        sample = small_dataset[0]
        frames = sample.frames

        assert isinstance(frames, torch.Tensor)
        assert frames.dim() == 4  # (T, C, H, W)
        assert frames.shape[0] == 8  # num_frames
        assert frames.shape[1] == 3  # RGB channels
        assert frames.shape[2] == 112  # height
        assert frames.shape[3] == 112  # width

    def test_video_sample_has_label(self, small_dataset: SSv2Dataset) -> None:
        """Test that sample has label information."""
        sample = small_dataset[0]

        assert isinstance(sample.label, str)
        assert len(sample.label) > 0
        assert isinstance(sample.label_id, int)
        assert sample.label_id >= 0

    def test_video_sample_has_video_id_in_metadata(
        self, small_dataset: SSv2Dataset
    ) -> None:
        """Test that sample has video ID in metadata."""
        sample = small_dataset[0]
        assert "video_id" in sample.metadata
        assert isinstance(sample.metadata["video_id"], str)
        assert len(sample.metadata["video_id"]) > 0

    def test_video_sample_has_metadata(self, small_dataset: SSv2Dataset) -> None:
        """Test that sample has expected metadata fields."""
        sample = small_dataset[0]
        metadata = sample.metadata

        assert isinstance(metadata, dict)
        assert "video_id" in metadata
        assert "template" in metadata
        assert "placeholders" in metadata
        assert "label_text" in metadata

    def test_frames_are_normalized(self, small_dataset: SSv2Dataset) -> None:
        """Test that frame values are normalized to [0, 1]."""
        sample = small_dataset[0]
        frames = sample.frames

        assert frames.min() >= 0.0
        assert frames.max() <= 1.0

    def test_num_classes(self, small_dataset: SSv2Dataset) -> None:
        """Test num_classes property returns correct count for subset."""
        # For a small subset, num_classes equals unique templates found
        assert small_dataset.num_classes == 5

    def test_total_classes(self, small_dataset: SSv2Dataset) -> None:
        """Test total_classes property returns 174 (SSv2 constant)."""
        assert small_dataset.total_classes == 174

    def test_class_names_property(self, small_dataset: SSv2Dataset) -> None:
        """Test class_names property returns list of templates."""
        names = small_dataset.class_names
        assert isinstance(names, list)
        assert len(names) == small_dataset.num_classes
        for name in names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_iteration(self, small_dataset: SSv2Dataset) -> None:
        """Test that dataset is iterable."""
        samples = list(small_dataset)
        assert len(samples) == 5
        for sample in samples:
            assert isinstance(sample, VideoSample)

    def test_subset_sizes(self) -> None:
        """Test loading different subset sizes."""
        ds_10 = SSv2Dataset(
            split="validation", subset_size=10, num_frames=4, mock_videos=True
        )
        ds_20 = SSv2Dataset(
            split="validation", subset_size=20, num_frames=4, mock_videos=True
        )

        assert len(ds_10) == 10
        assert len(ds_20) == 20

    def test_mock_videos_deterministic(self) -> None:
        """Test that mock video generation is deterministic."""
        ds1 = SSv2Dataset(
            split="validation",
            subset_size=3,
            num_frames=4,
            frame_size=(64, 64),
            mock_videos=True,
        )
        ds2 = SSv2Dataset(
            split="validation",
            subset_size=3,
            num_frames=4,
            frame_size=(64, 64),
            mock_videos=True,
        )

        for i in range(3):
            frames1 = ds1[i].frames
            frames2 = ds2[i].frames
            assert torch.allclose(frames1, frames2), f"Frames differ at index {i}"

    def test_video_dir_required_without_mock(self) -> None:
        """Test that video_dir is required when mock_videos=False."""
        with pytest.raises(ValueError, match="video_dir is required"):
            SSv2Dataset(
                split="validation",
                subset_size=5,
                mock_videos=False,
            )


@pytest.mark.slow
class TestSSv2DatasetLarger:
    """Slower tests that access more data."""

    def test_validation_split_100_samples(self) -> None:
        """Test loading 100 samples from validation split."""
        ds = SSv2Dataset(
            split="validation",
            subset_size=100,
            num_frames=8,
            mock_videos=True,
        )
        assert len(ds) == 100

        # Verify we can access samples at different indices
        sample_0 = ds[0]
        sample_50 = ds[50]
        sample_99 = ds[99]

        assert sample_0.metadata["video_id"] != sample_50.metadata["video_id"]
        assert sample_50.metadata["video_id"] != sample_99.metadata["video_id"]

    def test_train_split_loadable(self) -> None:
        """Test that train split can be loaded."""
        ds = SSv2Dataset(
            split="train",
            subset_size=50,
            num_frames=4,
            mock_videos=True,
        )
        assert len(ds) == 50


if __name__ == "__main__":
    # Quick manual test
    print("Loading SSv2 dataset with mock_videos=True (subset_size=5)...")
    ds = SSv2Dataset(
        split="validation",
        subset_size=5,
        num_frames=8,
        frame_size=(112, 112),
        mock_videos=True,
    )
    print(f"Dataset loaded with {len(ds)} samples")
    print(f"Number of unique classes in subset: {ds.num_classes}")
    print(f"Total SSv2 classes: {ds.total_classes}")

    print("\nFetching first sample...")
    sample = ds[0]
    print(f"  Video ID: {sample.metadata['video_id']}")
    print(f"  Label (template): {sample.label}")
    print(f"  Label ID: {sample.label_id}")
    print(f"  Frames shape: {sample.frames.shape}")
    print(f"  Frame dtype: {sample.frames.dtype}")
    print(f"  Frame range: [{sample.frames.min():.3f}, {sample.frames.max():.3f}]")
    print(f"  Metadata: {sample.metadata}")

    print("\nSSv2 dataset test PASSED!")
