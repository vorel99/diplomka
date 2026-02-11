"""Tests for FeatureMatrixBuilder."""

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.matrix_builder import FeatureMatrixBuilder


class MockFeature(BaseFeature):
    """Mock feature class for testing."""

    def __init__(self, data: pd.DataFrame | None = None):
        self.data = data if data is not None else pd.DataFrame({"AGS": [1, 2, 3], "value": [10, 20, 30]})

    def load(self) -> pd.DataFrame:
        """Return mock data."""
        return self.data.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return data as-is."""
        return df


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_data = {
        "municipalities": {
            "class": "MockFeature",
            "module": __name__,
            "params": {},
        },
        "features": [
            {
                "name": "test_feature",
                "class": "MockFeature",
                "module": __name__,
                "params": {},
            }
        ],
        "matrix": {
            "join_key": "AGS",
            "join_method": "inner",
            "save_output": False,
            "output_path": "test_output.csv",
        },
    }

    temp_path = tmp_path / "config.yaml"
    with open(temp_path, "w") as f:
        yaml.dump(config_data, f)

    yield temp_path


@pytest.fixture
def multi_feature_config(tmp_path):
    """Create a config with multiple features."""
    config_data = {
        "municipalities": {
            "class": "MockFeature",
            "module": __name__,
            "params": {},
        },
        "features": [
            {
                "name": "feature1",
                "class": "MockFeature",
                "module": __name__,
                "params": {},
            },
            {
                "name": "feature2",
                "class": "MockFeature",
                "module": __name__,
                "params": {},
            },
            {
                "name": "feature3",
                "class": "MockFeature",
                "module": __name__,
                "params": {},
            },
        ],
        "matrix": {
            "join_key": "AGS",
            "join_method": "inner",
            "save_output": False,
        },
    }

    config_file = tmp_path / "multi_config.yaml"
    config_file.write_text(yaml.dump(config_data))
    return str(config_file)


class TestFeatureMatrixBuilder:
    """Tests for FeatureMatrixBuilder class."""

    def test_init_with_valid_config(self, temp_config_file):
        """Test initialization with a valid config file."""
        builder = FeatureMatrixBuilder(config_path=temp_config_file)
        assert builder.config_path == temp_config_file
        assert len(builder.config.features) == 1
        assert "test_feature" in builder.features
        assert isinstance(builder.features["test_feature"], MockFeature)

    def test_init_with_missing_config(self):
        """Test initialization with missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            FeatureMatrixBuilder(config_path="nonexistent.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises error."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            FeatureMatrixBuilder(config_path=str(config_file))

    def test_load_config_invalid_schema(self, tmp_path):
        """Test loading config with invalid schema raises ValidationError."""
        invalid_config = {
            "features": [
                {
                    "name": "test",
                    # Missing required 'class' and 'module'
                }
            ]
        }

        config_file = tmp_path / "invalid_schema.yaml"
        config_file.write_text(yaml.dump(invalid_config))

        with pytest.raises(ValidationError):
            FeatureMatrixBuilder(config_path=str(config_file))

    def test_feature_nonexisting_module(self, tmp_path):
        """Test that import errors are handled gracefully."""
        config_data = {
            "municipalities": {"class": "MockFeature", "module": __name__},
            "features": [
                {
                    "name": "bad_feature",
                    "class": "NonExistentClass",
                    "module": "nonexistent.module",
                }
            ],
            "matrix": {"join_key": "AGS"},
        }

        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        builder = FeatureMatrixBuilder(config_path=str(config_file))
        # Should not raise, but features dict should be empty
        assert len(builder.features) == 0

    def test_instantiate_feature(self, temp_config_file):
        """Test feature instantiation."""
        builder = FeatureMatrixBuilder(config_path=temp_config_file)
        feature_config = builder.config.features[0]
        feature = builder._instantiate_feature(feature_config)

        assert isinstance(feature, MockFeature)

    def test_build_matrix_single_feature(self, temp_config_file):
        """Test building matrix with a single feature."""
        builder = FeatureMatrixBuilder(config_path=temp_config_file)
        matrix = builder.build_matrix()

        assert isinstance(matrix, pd.DataFrame)
        assert "AGS" in matrix.columns
        assert "test_feature_value" in matrix.columns
        assert len(matrix) == 3

    def test_build_matrix_multiple_features(self, multi_feature_config):
        """Test building matrix with multiple features."""
        builder = FeatureMatrixBuilder(config_path=multi_feature_config)

        # Create different data for each feature
        builder.municipalities = MockFeature(pd.DataFrame({"AGS": [1, 2], "muni_val": [100, 200]}))
        builder.features["feature1"] = MockFeature(pd.DataFrame({"AGS": [1, 2], "val1": [10, 20]}))
        builder.features["feature2"] = MockFeature(pd.DataFrame({"AGS": [1, 2], "val2": [100, 200]}))

        matrix = builder.build_matrix()

        assert "AGS" in matrix.columns
        assert "feature1_val1" in matrix.columns
        assert "feature2_val2" in matrix.columns
        assert len(matrix) == 2

    def test_build_matrix_missing_join_key(self, temp_config_file):
        """Test handling of missing join key in feature data."""
        builder = FeatureMatrixBuilder(config_path=temp_config_file)

        # Replace feature with one that doesn't have AGS column
        builder.features["test_feature"] = MockFeature(pd.DataFrame({"other_col": [1, 2, 3]}))

        with pytest.raises(KeyError, match="Join key 'AGS' not found in test_feature dataframe"):
            builder.build_matrix()

    def test_build_matrix_with_missing_values_drop(self, tmp_path):
        """Test handling missing values with drop strategy."""
        config_data = {
            "municipalities": {"class": "MockFeature", "module": __name__},
            "features": [{"name": "f1", "class": "MockFeature", "module": __name__}],
            "matrix": {"join_key": "AGS", "missing_values": "drop", "save_output": False},
        }

        config_file = tmp_path / "drop_missing.yaml"
        config_file.write_text(yaml.dump(config_data))

        builder = FeatureMatrixBuilder(config_path=str(config_file))

        # Create data with missing values
        df = pd.DataFrame({"AGS": [1, 2, 3], "val": [10, None, 30]})
        builder.features["f1"] = MockFeature(df)

        matrix = builder.build_matrix()

        # Should drop row with missing value
        assert len(matrix) == 2
        assert matrix["f1_val"].notna().all()

    def test_build_matrix_with_missing_values_fill(self, tmp_path):
        """Test handling missing values with fill strategy."""
        config_data = {
            "municipalities": {"class": "MockFeature", "module": __name__},
            "features": [{"name": "f1", "class": "MockFeature", "module": __name__}],
            "matrix": {"join_key": "AGS", "missing_values": "fill", "fill_value": -999, "save_output": False},
        }

        config_file = tmp_path / "fill_missing.yaml"
        config_file.write_text(yaml.dump(config_data))

        builder = FeatureMatrixBuilder(config_path=str(config_file))

        df = pd.DataFrame({"AGS": [1, 2, 3], "val": [10, None, 30]})
        builder.features["f1"] = MockFeature(df)

        matrix = builder.build_matrix()

        # Should fill missing value with -999
        assert len(matrix) == 3
        assert matrix.loc[matrix["AGS"] == 2, "f1_val"].values[0] == -999

    def test_build_matrix_saves_output(self, tmp_path):
        """Test that matrix is saved when save_output is True."""
        output_path = tmp_path / "output" / "matrix.csv"

        config_data = {
            "municipalities": {"class": "MockFeature", "module": __name__},
            "features": [{"name": "f1", "class": "MockFeature", "module": __name__}],
            "matrix": {"join_key": "AGS", "save_output": True, "output_path": str(output_path)},
        }

        config_file = tmp_path / "save_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        builder = FeatureMatrixBuilder(config_path=str(config_file))
        builder.build_matrix()

        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == 3

    def test_get_feature_names(self, temp_config_file):
        """Test getting feature names."""
        builder = FeatureMatrixBuilder(config_path=temp_config_file)

        names = builder.get_feature_names()
        assert names == ["test_feature"]

    def test_get_feature(self, temp_config_file):
        """Test getting a specific feature."""
        builder = FeatureMatrixBuilder(config_path=temp_config_file)

        feature = builder.get_feature("test_feature")
        assert feature is not None
        assert isinstance(feature, MockFeature)

        # Test getting non-existent feature
        assert builder.get_feature("nonexistent") is None

    def test_column_prefixing(self, multi_feature_config):
        """Test that feature names are prefixed to columns."""
        builder = FeatureMatrixBuilder(config_path=multi_feature_config)

        builder.features["feature1"] = MockFeature(pd.DataFrame({"AGS": [1, 2], "col1": [10, 20]}))
        builder.features["feature2"] = MockFeature(pd.DataFrame({"AGS": [1, 2], "col2": [100, 200]}))

        matrix = builder.build_matrix()

        # Check that columns are prefixed
        assert "feature1_col1" in matrix.columns
        assert "feature2_col2" in matrix.columns
        # Join key should not be prefixed
        assert "AGS" in matrix.columns
        assert "feature1_AGS" not in matrix.columns
        assert "feature2_AGS" not in matrix.columns
