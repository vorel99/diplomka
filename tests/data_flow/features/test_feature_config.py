"""Tests for feature configuration Pydantic models."""

import pytest
from pydantic import ValidationError

from geoscore_de.data_flow.features.config import ComponentConfig, FeaturesYAMLConfig, MatrixConfig


class TestFeatureConfig:
    """Tests for FeatureConfig model."""

    def test_valid_feature_config(self):
        """Test creating a valid feature config."""
        config = ComponentConfig(
            name="test_feature",
            class_name="TestFeature",
            module="test.module",
            params={"param1": "value1"},
        )
        assert config.name == "test_feature"
        assert config.class_name == "TestFeature"
        assert config.module == "test.module"
        assert config.params == {"param1": "value1"}

    def test_feature_config_with_alias(self):
        """Test that 'class' alias works for class_name."""
        config = ComponentConfig(
            name="test_feature",
            **{"class": "TestFeature"},  # Using the alias
            module="test.module",
        )
        assert config.class_name == "TestFeature"

    def test_feature_config_defaults(self):
        """Test default values for optional fields."""
        config = ComponentConfig(
            name="test_feature",
            class_name="TestFeature",
            module="test.module",
        )
        assert config.params == {}

    def test_feature_config_missing_required_field(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            ComponentConfig(
                name="test_feature",
                # Missing class_name and module
            )


class TestMatrixConfig:
    """Tests for MatrixConfig model."""

    def test_valid_matrix_config(self):
        """Test creating a valid matrix config."""
        config = MatrixConfig(
            join_key="ID",
            save_output=False,
            output_path="custom/path.csv",
            missing_values="fill",
            fill_value=999,
        )
        assert config.join_key == "ID"
        assert config.save_output is False
        assert config.output_path == "custom/path.csv"
        assert config.missing_values == "fill"
        assert config.fill_value == 999

    def test_matrix_config_defaults(self):
        """Test default values for matrix config."""
        config = MatrixConfig()
        assert config.join_key == "AGS"
        assert config.save_output is True
        assert config.output_path == "data/final/feature_matrix.csv"
        assert config.missing_values is None
        assert config.fill_value == 0

    def test_invalid_missing_values_strategy(self):
        """Test that invalid missing values strategy raises validation error."""
        with pytest.raises(ValidationError):
            MatrixConfig(missing_values="invalid")


class TestFeaturesYAMLConfig:
    """Tests for the root FeaturesYAMLConfig model."""

    def test_valid_full_config(self):
        """Test creating a complete valid config."""
        config_dict = {
            "municipalities": {"class": "MockFeature", "module": __name__},
            "features": [
                {
                    "name": "feature1",
                    "class": "Feature1",
                    "module": "module1",
                    "params": {"param1": "value1"},
                },
                {
                    "name": "feature2",
                    "class": "Feature2",
                    "module": "module2",
                },
            ],
            "matrix": {
                "join_key": "ID",
                "join_method": "left",
                "save_output": True,
                "output_path": "output.csv",
            },
        }
        config = FeaturesYAMLConfig(**config_dict)
        assert len(config.features) == 2
        assert config.features[0].name == "feature1"
        assert config.matrix.join_key == "ID"

    def test_empty_config(self):
        """Test creating config with default values."""
        with pytest.raises(ValidationError):
            FeaturesYAMLConfig()

    def test_config_with_invalid_feature(self):
        """Test that invalid feature raises validation error."""
        config_dict = {
            "municipalities": {"class": "MockFeature", "module": __name__},
            "features": [
                {
                    "name": "feature1",
                    # Missing required 'class' and 'module'
                }
            ],
        }
        with pytest.raises(ValidationError):
            FeaturesYAMLConfig(**config_dict)

    def test_config_with_invalid_matrix(self):
        """Test that invalid matrix config raises validation error."""
        config_dict = {
            "municipalities": {"class": "MockFeature", "module": __name__},
            "matrix": {
                "missing_values": "invalid_value",  # Invalid choice
            },
        }
        with pytest.raises(ValidationError):
            FeaturesYAMLConfig(**config_dict)
