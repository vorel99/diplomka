from geoscore_de.data_flow.features.base import get_feature_class
from geoscore_de.data_flow.features.config import FeatureConfig


def test_get_feature_class(mock_feature_class, mock_feature_module):
    """Test feature instantiation."""
    config = FeatureConfig(
        name="mock_feature",
        class_name="MockFeature",
        module=mock_feature_module,
    )
    feature_instance = get_feature_class(config)
    assert isinstance(feature_instance, mock_feature_class)
