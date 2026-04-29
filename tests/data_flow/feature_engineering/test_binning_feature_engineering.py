import pandas as pd
import pytest

from geoscore_de.data_flow.feature_engineering.binning import BinningFeatureEngineering


def _make_df(data):
    """Helper to build a minimal dataframe for binning tests."""
    return pd.DataFrame({"AGS": range(len(data)), "value": data})


class TestBinningFeatureEngineeringInit:
    def test_valid_single_column(self):
        binning = BinningFeatureEngineering(input_columns=["value"], output_column="binned")
        assert binning.input_columns == ["value"]
        assert binning.output_column == "binned"
        assert binning.strategy == "quantile"
        assert binning.n_bins == 10

    def test_custom_strategy_and_bins(self):
        binning = BinningFeatureEngineering(
            input_columns=["value"], output_column="binned", strategy="equal_width", n_bins=5
        )
        assert binning.strategy == "equal_width"
        assert binning.n_bins == 5

    def test_too_few_columns_raises(self):
        with pytest.raises(ValueError, match="exactly 1"):
            BinningFeatureEngineering(input_columns=[], output_column="binned")

    def test_too_many_columns_raises(self):
        with pytest.raises(ValueError, match="exactly 1"):
            BinningFeatureEngineering(input_columns=["a", "b"], output_column="binned")

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            BinningFeatureEngineering(input_columns=["value"], output_column="binned", strategy="invalid")

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError, match="n_bins must be at least 2"):
            BinningFeatureEngineering(input_columns=["value"], output_column="binned", n_bins=1)


class TestBinningFeatureEngineeringApply:
    def test_quantile_binning_basic(self):
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        binning = BinningFeatureEngineering(
            input_columns=["value"], output_column="binned", strategy="quantile", n_bins=2
        )
        result = binning.apply(df)

        assert "binned" in result.columns
        # With 10 values and 2 bins, roughly 5 should be in each bin
        assert result["binned"].nunique() <= 2
        assert result["binned"].isin([0, 1]).all()

    def test_equal_width_binning_basic(self):
        df = _make_df([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        binning = BinningFeatureEngineering(
            input_columns=["value"], output_column="binned", strategy="equal_width", n_bins=2
        )
        result = binning.apply(df)

        assert "binned" in result.columns
        assert result["binned"].isin([0, 1]).all()

    def test_quantile_binning_many_bins(self):
        df = _make_df(list(range(100)))
        binning = BinningFeatureEngineering(
            input_columns=["value"], output_column="binned", strategy="quantile", n_bins=10
        )
        result = binning.apply(df)

        assert result["binned"].min() == 0
        assert result["binned"].max() <= 9

    def test_null_values_preserved(self):
        df = _make_df([1.0, None, 3.0, 4.0, 5.0])
        binning = BinningFeatureEngineering(
            input_columns=["value"], output_column="binned", strategy="quantile", n_bins=2
        )
        result = binning.apply(df)

        assert pd.isna(result["binned"].iloc[1])
        assert not pd.isna(result["binned"].iloc[0])
        assert not pd.isna(result["binned"].iloc[2])

    def test_all_same_values_quantile(self):
        # When all values are identical, qcut with duplicates='drop' returns none edges
        df = _make_df([5.0, 5.0, 5.0, 5.0])
        binning = BinningFeatureEngineering(
            input_columns=["value"], output_column="binned", strategy="quantile", n_bins=4
        )
        result = binning.apply(df)

        # All should be None
        assert pd.isna(result["binned"]).all()

    def test_negative_values(self):
        df = _make_df([-5.0, -3.0, 0.0, 3.0, 5.0])
        binning = BinningFeatureEngineering(
            input_columns=["value"], output_column="binned", strategy="equal_width", n_bins=2
        )
        result = binning.apply(df)

        assert result["binned"].isin([0, 1]).all()
        assert not result["binned"].isna().any()

    def test_default_params(self):
        # Test that default params are applied
        df = _make_df(list(range(100)))
        binning = BinningFeatureEngineering(input_columns=["value"], output_column="binned")
        result = binning.apply(df)

        assert "binned" in result.columns
        # Default is quantile with 10 bins
        assert result["binned"].max() <= 9


class TestBinningFeatureEngineeringValidation:
    def test_non_numeric_column_fails(self):
        df = pd.DataFrame({"AGS": [1, 2, 3], "value": ["a", "b", "c"]})
        binning = BinningFeatureEngineering(input_columns=["value"], output_column="binned")
        with pytest.raises(ValueError, match="failed validation"):
            binning.apply(df)

    def test_missing_column_fails(self):
        df = _make_df([1.0, 2.0, 3.0])
        binning = BinningFeatureEngineering(input_columns=["nonexistent"], output_column="binned")
        with pytest.raises(ValueError, match="failed validation"):
            binning.apply(df)

    def test_missing_ags_column_fails(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        binning = BinningFeatureEngineering(input_columns=["value"], output_column="binned")
        with pytest.raises(ValueError, match="failed validation"):
            binning.apply(df)
