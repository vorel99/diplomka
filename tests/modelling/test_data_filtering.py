import pandas as pd
import pytest

from geoscore_de.modelling.config import FeatureFilteringConfig, ModelConfig, RowFilteringConfig, TrainingConfig
from geoscore_de.modelling.data_filtering import filter_features, filter_rows


class TestFeatureFiltering:
    """Test feature filtering functionality."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample DataFrame with features."""
        return pd.DataFrame(
            {
                "census_a": [1, 2, 3, 4],
                "census_b": [5, 6, 7, 8],
                "feature_c": [9, 10, 11, 12],
                "feature_d": [13, 14, 15, 16],
            }
        )

    @pytest.fixture
    def base_config(self) -> TrainingConfig:
        """Create base training config."""
        return TrainingConfig(target_variable="target", feature_filtering=FeatureFilteringConfig(), model=ModelConfig())

    def test_no_filtering(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test that all features are kept when no filtering is specified."""
        result = filter_features(sample_data.copy(), base_config)

        assert list(result.columns) == list(sample_data.columns)
        pd.testing.assert_frame_equal(result, sample_data)

    def test_use_features_only(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test selecting specific features with use_features."""
        base_config.feature_filtering.use_features = ["census_a", "feature_c"]
        result = filter_features(sample_data.copy(), base_config)

        assert list(result.columns) == ["census_a", "feature_c"]
        assert len(result) == 4  # All rows preserved
        pd.testing.assert_frame_equal(result, sample_data[["census_a", "feature_c"]])

    def test_omit_features_only(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test excluding specific features with omit_features."""
        base_config.feature_filtering.omit_features = ["census_b", "feature_d"]
        result = filter_features(sample_data.copy(), base_config)

        assert list(result.columns) == ["census_a", "feature_c"]
        assert len(result) == 4  # All rows preserved
        pd.testing.assert_frame_equal(result, sample_data[["census_a", "feature_c"]])

    def test_use_and_omit_features(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test using both use_features and omit_features together.

        First use_features selects subset, then omit_features excludes from that subset.
        """
        # First select a, b, c, then omit b
        base_config.feature_filtering.use_features = ["census_a", "census_b", "feature_c"]
        base_config.feature_filtering.omit_features = ["census_b"]
        result = filter_features(sample_data.copy(), base_config)

        # Should have a and c only (d excluded by use_features, b excluded by omit_features)
        assert list(result.columns) == ["census_a", "feature_c"]
        pd.testing.assert_frame_equal(result, sample_data[["census_a", "feature_c"]])

    def test_use_nonexistent_feature(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test that using non-existent feature raises warning."""
        base_config.feature_filtering.use_features = ["census_a", "nonexistent_feature"]
        with pytest.warns(UserWarning, match="No columns matched for use_features patterns"):
            result = filter_features(sample_data.copy(), base_config)

        assert list(result.columns) == ["census_a"]

    def test_omit_nonexistent_feature(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test that omitting non-existent feature raises warning."""
        base_config.feature_filtering.omit_features = ["nonexistent_feature", "census_b"]
        with pytest.warns(UserWarning, match="No columns matched for omit_features patterns"):
            result = filter_features(sample_data.copy(), base_config)

        # Only census_b should be removed
        assert list(result.columns) == ["census_a", "feature_c", "feature_d"]

    def test_use_single_feature(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test selecting a single feature."""
        base_config.feature_filtering.use_features = ["feature_c"]
        result = filter_features(sample_data.copy(), base_config)

        assert list(result.columns) == ["feature_c"]
        assert result.shape == (4, 1)

    def test_omit_all_features(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test omitting all features results in empty DataFrame."""
        base_config.feature_filtering.omit_features = ["census_a", "census_b", "feature_c", "feature_d"]
        result = filter_features(sample_data.copy(), base_config)

        assert len(result.columns) == 0
        assert len(result) == 4  # Rows still present

    def test_use_features_preserves_order(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test that use_features preserves the specified order."""
        # Specify features in different order than DataFrame
        base_config.feature_filtering.use_features = ["feature_d", "census_a", "feature_c"]
        result = filter_features(sample_data.copy(), base_config)

        # Should maintain order from use_features
        assert list(result.columns) == ["feature_d", "census_a", "feature_c"]

    def test_use_features_regex(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test using regex patterns in use_features."""
        base_config.feature_filtering.use_features = ["census_*"]
        result = filter_features(sample_data.copy(), base_config)

        assert list(result.columns) == ["census_a", "census_b"]
        pd.testing.assert_frame_equal(result, sample_data[["census_a", "census_b"]])

    def test_omit_features_regex(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test using regex patterns in omit_features."""
        base_config.feature_filtering.omit_features = ["feature_*"]
        result = filter_features(sample_data.copy(), base_config)

        assert list(result.columns) == ["census_a", "census_b"]
        pd.testing.assert_frame_equal(result, sample_data[["census_a", "census_b"]])


class TestRowFiltering:
    """Test row filtering functionality."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample DataFrame with AGS and region columns."""
        return pd.DataFrame(
            {
                "AGS": ["02000", "02123", "03000", "09162", "09163"],
                "land": ["Hamburg", "Hamburg", "Niedersachsen", "Bayern", "Bayern"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

    @pytest.fixture
    def base_config(self) -> TrainingConfig:
        """Create base training config with no row filtering."""
        return TrainingConfig(
            target_variable="value",
            feature_filtering=FeatureFilteringConfig(),
            row_filtering=RowFilteringConfig(),
            model=ModelConfig(),
        )

    def test_no_filtering(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test that all rows are kept when no row filtering is configured."""
        result = filter_rows(sample_data.copy(), base_config)

        pd.testing.assert_frame_equal(result, sample_data)

    def test_omit_exact_value(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test omitting rows with an exact column value."""
        base_config.row_filtering.omit_rows = {"AGS": ["03000"]}
        result = filter_rows(sample_data.copy(), base_config)

        assert "03000" not in result["AGS"].values
        assert len(result) == 4

    def test_omit_glob_pattern(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test omitting rows using a glob-style wildcard pattern (e.g. '02*')."""
        base_config.row_filtering.omit_rows = {"AGS": ["02*"]}
        result = filter_rows(sample_data.copy(), base_config)

        assert not result["AGS"].str.startswith("02").any()
        assert len(result) == 3
        assert list(result["AGS"]) == ["03000", "09162", "09163"]

    def test_omit_multiple_patterns(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test omitting rows matching any of multiple patterns."""
        base_config.row_filtering.omit_rows = {"AGS": ["02*", "03*"]}
        result = filter_rows(sample_data.copy(), base_config)

        assert len(result) == 2
        assert list(result["AGS"]) == ["09162", "09163"]

    def test_omit_multiple_columns(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test omitting rows based on filters applied to two different columns."""
        base_config.row_filtering.omit_rows = {"AGS": ["09*"], "land": ["Hamburg"]}
        result = filter_rows(sample_data.copy(), base_config)

        # Hamburg rows (02000, 02123) and Bayern rows (09162, 09163) should be omitted
        assert len(result) == 1
        assert list(result["AGS"]) == ["03000"]

    def test_omit_regex_pattern(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test omitting rows using a full regular expression."""
        # Regex: AGS starts with 09 and ends with 2 or 3
        base_config.row_filtering.omit_rows = {"AGS": ["0916[23]"]}
        result = filter_rows(sample_data.copy(), base_config)

        assert len(result) == 3
        assert "09162" not in result["AGS"].values
        assert "09163" not in result["AGS"].values

    def test_omit_nonexistent_column_warns(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test that a warning is issued when the filter column is not in the data."""
        base_config.row_filtering.omit_rows = {"nonexistent": ["02*"]}
        with pytest.warns(UserWarning, match="Column 'nonexistent'"):
            result = filter_rows(sample_data.copy(), base_config)

        # All rows preserved when column is missing
        pd.testing.assert_frame_equal(result, sample_data)

    def test_omit_all_rows(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test that filtering can remove all rows."""
        base_config.row_filtering.omit_rows = {"AGS": [".*"]}
        result = filter_rows(sample_data.copy(), base_config)

        assert len(result) == 0
        assert list(result.columns) == list(sample_data.columns)

    def test_pattern_matches_no_rows(self, sample_data: pd.DataFrame, base_config: TrainingConfig):
        """Test that a pattern matching no rows leaves the DataFrame unchanged."""
        base_config.row_filtering.omit_rows = {"AGS": ["99*"]}
        result = filter_rows(sample_data.copy(), base_config)

        pd.testing.assert_frame_equal(result, sample_data)
