import numpy as np
import pandas as pd
import pytest

from geoscore_de.data_flow.feature_engineering.sum import SumFeatureEngineering


def _make_df(col_a, col_b, col_c):
    """Helper to build a minimal dataframe for sum tests."""
    return pd.DataFrame({"AGS": range(len(col_a)), "a": col_a, "b": col_b, "c": col_c})


class TestSumFeatureEngineeringInit:
    def test_valid_two_columns(self):
        sum_fe = SumFeatureEngineering(input_columns=["a", "b"], output_column="sum_ab")
        assert sum_fe.input_columns == ["a", "b"]
        assert sum_fe.output_column == "sum_ab"

    def test_valid_three_columns(self):
        sum_fe = SumFeatureEngineering(input_columns=["a", "b", "c"], output_column="sum_abc")
        assert sum_fe.input_columns == ["a", "b", "c"]
        assert sum_fe.output_column == "sum_abc"

    def test_too_few_columns_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            SumFeatureEngineering(input_columns=["a"], output_column="out")


class TestSumFeatureEngineeringApply:
    def test_basic_sum_computation_two_columns(self):
        df = pd.DataFrame({"AGS": [0, 1, 2], "a": [10.0, 20.0, 30.0], "b": [1.0, 2.0, 3.0]})
        sum_fe = SumFeatureEngineering(input_columns=["a", "b"], output_column="sum")
        result = sum_fe.transform(df)

        assert "sum" in result.columns
        assert list(result["sum"]) == [11.0, 22.0, 33.0]

    def test_basic_sum_computation_three_columns(self):
        df = _make_df(col_a=[10.0, 20.0], col_b=[1.0, 2.0], col_c=[100.0, 200.0])
        sum_fe = SumFeatureEngineering(input_columns=["a", "b", "c"], output_column="sum")
        result = sum_fe.transform(df)

        assert list(result["sum"]) == [111.0, 222.0]

    def test_negative_and_positive_values(self):
        df = _make_df(col_a=[10.0, -5.0], col_b=[-2.0, 1.0], col_c=[-3.0, 4.0])
        sum_fe = SumFeatureEngineering(input_columns=["a", "b", "c"], output_column="sum")
        result = sum_fe.transform(df)

        assert list(result["sum"]) == [5.0, 0.0]

    def test_partial_nulls_are_ignored(self):
        df = _make_df(col_a=[10.0, None], col_b=[None, 2.0], col_c=[1.0, 3.0])
        sum_fe = SumFeatureEngineering(input_columns=["a", "b", "c"], output_column="sum")
        result = sum_fe.transform(df)

        assert result["sum"].iloc[0] == 11.0
        assert result["sum"].iloc[1] == 5.0

    def test_all_nulls_give_null(self):
        df = _make_df(col_a=[np.nan], col_b=[np.nan], col_c=[np.nan])
        sum_fe = SumFeatureEngineering(input_columns=["a", "b", "c"], output_column="sum")
        result = sum_fe.transform(df)

        assert np.isnan(result["sum"].iloc[0])

    def test_original_columns_preserved(self):
        df = _make_df(col_a=[10.0], col_b=[5.0], col_c=[1.0])
        sum_fe = SumFeatureEngineering(input_columns=["a", "b", "c"], output_column="sum")
        result = sum_fe.transform(df)

        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns
        assert "AGS" in result.columns

    def test_output_column_name(self):
        df = _make_df(col_a=[1.0], col_b=[2.0], col_c=[3.0])
        sum_fe = SumFeatureEngineering(input_columns=["a", "b", "c"], output_column="my_custom_sum")
        result = sum_fe.transform(df)

        assert "my_custom_sum" in result.columns


class TestSumFeatureEngineeringValidation:
    def test_missing_input_column_raises(self):
        df = pd.DataFrame({"AGS": [1], "a": [10.0], "b": [5.0]})
        sum_fe = SumFeatureEngineering(input_columns=["a", "b", "c"], output_column="sum")
        with pytest.raises(ValueError, match="validation"):
            sum_fe.transform(df)

    def test_missing_ags_column_raises(self):
        df = pd.DataFrame({"a": [10.0], "b": [5.0], "c": [1.0]})
        sum_fe = SumFeatureEngineering(input_columns=["a", "b", "c"], output_column="sum")
        with pytest.raises(ValueError, match="validation"):
            sum_fe.transform(df)

    def test_non_numeric_column_raises(self):
        df = pd.DataFrame({"AGS": [1], "a": [10.0], "b": ["hello"], "c": [1.0]})
        sum_fe = SumFeatureEngineering(input_columns=["a", "b", "c"], output_column="sum")
        with pytest.raises(ValueError, match="validation"):
            sum_fe.transform(df)


def test_instantiate_sum_via_config():
    """Ensure SumFeatureEngineering can be loaded via the dynamic factory."""
    from geoscore_de.data_flow.feature_engineering.base import instantiate_feature_engineering_class
    from geoscore_de.data_flow.feature_engineering.config import FeatureEngineeringConfig

    config = FeatureEngineeringConfig(
        name="test_sum",
        class_name="SumFeatureEngineering",
        input_columns=["col_1", "col_2", "col_3"],
        output_column="col_sum",
        params={},
    )
    instance = instantiate_feature_engineering_class(config)
    assert isinstance(instance, SumFeatureEngineering)
    assert instance.input_columns == ["col_1", "col_2", "col_3"]
    assert instance.output_column == "col_sum"
