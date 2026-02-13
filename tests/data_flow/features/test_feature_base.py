from geoscore_de.data_flow.features.base import instantiate_feature
from geoscore_de.data_flow.features.config import FeatureConfig


def test_load_transform_with_before_transforms(mock_feature_class, feature_engineering_config):
    """Test the load_transform method of BaseFeature."""
    # Create a mock feature config with a before_transform
    feature_instance = mock_feature_class(before_transforms=[feature_engineering_config])
    df = feature_instance.load_transform()
    # Check that the transformed column is created
    assert "transformed_value" in df.columns
    # Check that the transformation is correct (sum of value and weight)
    expected_transformed = feature_instance.data["value"] + feature_instance.data["weight"]
    assert all(df["transformed_value"] == expected_transformed)


def test_instantiate_feature(mock_feature_class, mock_feature_module):
    """Test feature instantiation."""
    config = FeatureConfig(
        name="mock_feature",
        class_name="MockFeature",
        module=mock_feature_module,
    )
    feature_instance = instantiate_feature(config)
    assert isinstance(feature_instance, mock_feature_class)
