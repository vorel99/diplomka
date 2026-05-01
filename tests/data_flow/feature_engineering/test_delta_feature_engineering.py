import numpy as np
import pandas as pd
import pytest

from geoscore_de.data_flow.feature_engineering.delta import DeltaFeatureEngineering


def _make_df(newer, older):
    """Helper to build a minimal dataframe for delta tests."""
    return pd.DataFrame({"AGS": range(len(newer)), "newer": newer, "older": older})


class TestDeltaFeatureEngineeringInit:
    def test_valid_two_columns(self):
        delta = DeltaFeatureEngineering(input_columns=["a", "b"], output_column="delta_ab")
        assert delta.input_columns == ["a", "b"]
        assert delta.output_column == "delta_ab"

    def test_too_few_columns_raises(self):
        with pytest.raises(ValueError, match="exactly 2"):
            DeltaFeatureEngineering(input_columns=["a"], output_column="out")

    def test_too_many_columns_raises(self):
        with pytest.raises(ValueError, match="exactly 2"):
            DeltaFeatureEngineering(input_columns=["a", "b", "c"], output_column="out")


class TestDeltaFeatureEngineeringApply:
    def test_basic_delta_computation(self):
        df = _make_df(newer=[10.0, 20.0, 30.0], older=[5.0, 15.0, 25.0])
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        result = delta.transform(df)

        assert "delta" in result.columns
        assert list(result["delta"]) == [5.0, 5.0, 5.0]

    def test_negative_delta(self):
        df = _make_df(newer=[5.0, 10.0], older=[10.0, 15.0])
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        result = delta.transform(df)

        assert list(result["delta"]) == [-5.0, -5.0]

    def test_zero_delta(self):
        df = _make_df(newer=[7.0, 7.0], older=[7.0, 7.0])
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        result = delta.transform(df)

        assert list(result["delta"]) == [0.0, 0.0]

    def test_null_in_newer_propagates(self):
        df = _make_df(newer=[None, 20.0], older=[5.0, 15.0])
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        result = delta.transform(df)

        assert np.isnan(result["delta"].iloc[0])
        assert result["delta"].iloc[1] == 5.0

    def test_null_in_older_propagates(self):
        df = _make_df(newer=[10.0, 20.0], older=[None, 15.0])
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        result = delta.transform(df)

        assert np.isnan(result["delta"].iloc[0])
        assert result["delta"].iloc[1] == 5.0

    def test_both_null_gives_null(self):
        df = _make_df(newer=[float("nan")], older=[float("nan")])
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        result = delta.transform(df)

        assert np.isnan(result["delta"].iloc[0])

    def test_original_columns_preserved(self):
        df = _make_df(newer=[10.0], older=[5.0])
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        result = delta.transform(df)

        assert "newer" in result.columns
        assert "older" in result.columns
        assert "AGS" in result.columns

    def test_output_column_name(self):
        df = _make_df(newer=[1.0], older=[0.0])
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="my_custom_delta")
        result = delta.transform(df)

        assert "my_custom_delta" in result.columns


class TestDeltaFeatureEngineeringValidation:
    def test_missing_newer_column_raises(self):
        df = pd.DataFrame({"AGS": [1], "older": [5.0]})
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        with pytest.raises(ValueError, match="validation"):
            delta.transform(df)

    def test_missing_older_column_raises(self):
        df = pd.DataFrame({"AGS": [1], "newer": [10.0]})
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        with pytest.raises(ValueError, match="validation"):
            delta.transform(df)

    def test_missing_ags_column_raises(self):
        df = pd.DataFrame({"newer": [10.0], "older": [5.0]})
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        with pytest.raises(ValueError, match="validation"):
            delta.transform(df)

    def test_non_numeric_column_raises(self):
        df = pd.DataFrame({"AGS": [1], "newer": ["hello"], "older": [5.0]})
        delta = DeltaFeatureEngineering(input_columns=["newer", "older"], output_column="delta")
        with pytest.raises(ValueError, match="validation"):
            delta.transform(df)


def test_instantiate_delta_via_config():
    """Ensure DeltaFeatureEngineering can be loaded via the dynamic factory."""
    from geoscore_de.data_flow.feature_engineering.base import instantiate_feature_engineering_class
    from geoscore_de.data_flow.feature_engineering.config import FeatureEngineeringConfig

    config = FeatureEngineeringConfig(
        name="test_delta",
        class_name="DeltaFeatureEngineering",
        input_columns=["col_new", "col_old"],
        output_column="col_delta",
        params={},
    )
    instance = instantiate_feature_engineering_class(config)
    assert isinstance(instance, DeltaFeatureEngineering)
    assert instance.input_columns == ["col_new", "col_old"]
    assert instance.output_column == "col_delta"
