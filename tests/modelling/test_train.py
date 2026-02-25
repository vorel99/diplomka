import pandas as pd
import pytest

from geoscore_de.modelling.config import FeatureFilteringConfig, ModelConfig, TrainingConfig
from geoscore_de.modelling.train import Trainer


class TestFeatureFiltering:
    """Test feature filtering functionality."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample DataFrame with features."""
        return pd.DataFrame(
            {
                "feature_a": [1, 2, 3, 4],
                "feature_b": [5, 6, 7, 8],
                "feature_c": [9, 10, 11, 12],
                "feature_d": [13, 14, 15, 16],
            }
        )

    @pytest.fixture
    def base_config(self) -> TrainingConfig:
        """Create base training config."""
        return TrainingConfig(target_variable="target", feature_filtering=FeatureFilteringConfig(), model=ModelConfig())

    def test_no_filtering(self, sample_data, base_config):
        """Test that all features are kept when no filtering is specified."""
        trainer = Trainer(base_config)
        result = trainer._filter_features(sample_data.copy())

        assert list(result.columns) == list(sample_data.columns)
        pd.testing.assert_frame_equal(result, sample_data)

    def test_use_features_only(self, sample_data, base_config):
        """Test selecting specific features with use_features."""
        base_config.feature_filtering.use_features = ["feature_a", "feature_c"]
        trainer = Trainer(base_config)

        result = trainer._filter_features(sample_data.copy())

        assert list(result.columns) == ["feature_a", "feature_c"]
        assert len(result) == 4  # All rows preserved
        pd.testing.assert_frame_equal(result, sample_data[["feature_a", "feature_c"]])

    def test_omit_features_only(self, sample_data, base_config):
        """Test excluding specific features with omit_features."""
        base_config.feature_filtering.omit_features = ["feature_b", "feature_d"]
        trainer = Trainer(base_config)

        result = trainer._filter_features(sample_data.copy())

        assert list(result.columns) == ["feature_a", "feature_c"]
        assert len(result) == 4  # All rows preserved
        pd.testing.assert_frame_equal(result, sample_data[["feature_a", "feature_c"]])

    def test_use_and_omit_features(self, sample_data, base_config):
        """Test using both use_features and omit_features together.

        First use_features selects subset, then omit_features excludes from that subset.
        """
        # First select a, b, c, then omit b
        base_config.feature_filtering.use_features = ["feature_a", "feature_b", "feature_c"]
        base_config.feature_filtering.omit_features = ["feature_b"]
        trainer = Trainer(base_config)

        result = trainer._filter_features(sample_data.copy())

        # Should have a and c only (d excluded by use_features, b excluded by omit_features)
        assert list(result.columns) == ["feature_a", "feature_c"]
        pd.testing.assert_frame_equal(result, sample_data[["feature_a", "feature_c"]])

    def test_omit_nonexistent_feature(self, sample_data, base_config):
        """Test that omitting non-existent feature doesn't raise error."""
        base_config.feature_filtering.omit_features = ["nonexistent_feature", "feature_b"]
        trainer = Trainer(base_config)

        # Should not raise error due to errors="ignore"
        result = trainer._filter_features(sample_data.copy())

        # Only feature_b should be removed
        assert list(result.columns) == ["feature_a", "feature_c", "feature_d"]

    def test_use_single_feature(self, sample_data, base_config):
        """Test selecting a single feature."""
        base_config.feature_filtering.use_features = ["feature_c"]
        trainer = Trainer(base_config)

        result = trainer._filter_features(sample_data.copy())

        assert list(result.columns) == ["feature_c"]
        assert result.shape == (4, 1)

    def test_omit_all_features(self, sample_data, base_config):
        """Test omitting all features results in empty DataFrame."""
        base_config.feature_filtering.omit_features = ["feature_a", "feature_b", "feature_c", "feature_d"]
        trainer = Trainer(base_config)

        result = trainer._filter_features(sample_data.copy())

        assert len(result.columns) == 0
        assert len(result) == 4  # Rows still present

    def test_use_features_preserves_order(self, sample_data, base_config):
        """Test that use_features preserves the specified order."""
        # Specify features in different order than DataFrame
        base_config.feature_filtering.use_features = ["feature_d", "feature_a", "feature_c"]
        trainer = Trainer(base_config)

        result = trainer._filter_features(sample_data.copy())

        # Should maintain order from use_features
        assert list(result.columns) == ["feature_d", "feature_a", "feature_c"]
