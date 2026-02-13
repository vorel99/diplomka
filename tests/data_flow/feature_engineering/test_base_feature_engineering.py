import pytest

from geoscore_de.data_flow.feature_engineering.base import instantiate_feature_engineering_class
from geoscore_de.data_flow.feature_engineering.config import FeatureEngineeringConfig
from geoscore_de.data_flow.feature_engineering.homogeneity import HomogeneityFeatureEngineering


def test_instantiate_feature_engineering_class_homogeneity():
    config = FeatureEngineeringConfig(
        name="homogeneity",
        class_name="HomogeneityFeatureEngineering",
        input_columns=["voting_percentage", "population"],
        output_column="homogeneity_score",
        params={
            "weight_column": "population",
        },
    )
    feature_engineering_class = instantiate_feature_engineering_class(config)
    assert isinstance(feature_engineering_class, HomogeneityFeatureEngineering), (
        "Expected an instance of HomogeneityFeatureEngineering"
    )
    assert feature_engineering_class.input_columns == ["voting_percentage", "population"], "Input columns do not match"
    assert feature_engineering_class.output_column == "homogeneity_score", "Output column does not match"
    assert feature_engineering_class.weight_column == "population", "Weight column does not match"


def test_instantiate_feature_engineering_class_invalid_module():
    config = FeatureEngineeringConfig(
        name="invalid_module",
        module="non_existent_module",
        class_name="SomeClass",
        input_columns=["col1"],
        output_column="col2",
        params={},
    )
    with pytest.raises(ImportError):
        instantiate_feature_engineering_class(config)
