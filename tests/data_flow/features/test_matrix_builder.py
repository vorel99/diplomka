"""Tests for FeatureMatrixBuilder."""

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from geoscore_de.data_flow.features.config import ColumnFilteringConfig
from geoscore_de.data_flow.features.matrix_builder import FeatureMatrixBuilder

# Module path for DeltaFeatureEngineering (used in after_transforms configs)
DELTA_MODULE = "geoscore_de.data_flow.feature_engineering"


@pytest.fixture
def temp_config_file(tmp_path, mock_feature_module):
    """Create a temporary config file for testing."""
    config_data = {
        "municipalities": {
            "class": "MockFeature",
            "module": mock_feature_module,
            "params": {},
        },
        "features": [
            {
                "name": "test_feature",
                "class": "MockFeature",
                "module": mock_feature_module,
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
def multi_feature_config(tmp_path, mock_feature_module):
    """Create a config with multiple features."""
    config_data = {
        "municipalities": {
            "class": "MockFeature",
            "module": mock_feature_module,
            "params": {},
        },
        "features": [
            {
                "name": "feature1",
                "class": "MockFeature",
                "module": mock_feature_module,
                "params": {},
            },
            {
                "name": "feature2",
                "class": "MockFeature",
                "module": mock_feature_module,
                "params": {},
            },
            {
                "name": "feature3",
                "class": "MockFeature",
                "module": mock_feature_module,
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

    @pytest.fixture(autouse=True)
    def setup_mock(self, mock_feature_class):
        """Inject MockFeature into module namespace for all tests."""
        self.MockFeature = mock_feature_class

    def test_init_with_valid_config(self, temp_config_file):
        """Test initialization with a valid config file."""
        builder = FeatureMatrixBuilder(config_path=temp_config_file)
        assert builder.config_path == temp_config_file
        assert len(builder.config.features) == 1
        assert "test_feature" in builder.features
        assert isinstance(builder.features["test_feature"], self.MockFeature)

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

    def test_feature_nonexisting_module(self, tmp_path, mock_feature_module):
        """Test that import errors are handled gracefully."""
        config_data = {
            "municipalities": {"class": "MockFeature", "module": mock_feature_module},
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
        builder.municipalities = self.MockFeature(pd.DataFrame({"AGS": [1, 2], "muni_val": [100, 200]}))
        builder.features["feature1"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2], "val1": [10, 20]}))
        builder.features["feature2"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2], "val2": [100, 200]}))

        matrix = builder.build_matrix()

        assert "AGS" in matrix.columns
        assert "feature1_val1" in matrix.columns
        assert "feature2_val2" in matrix.columns
        assert len(matrix) == 2

    def test_build_matrix_missing_join_key(self, temp_config_file):
        """Test handling of missing join key in feature data."""
        builder = FeatureMatrixBuilder(config_path=temp_config_file)

        # Replace feature with one that doesn't have AGS column
        builder.features["test_feature"] = self.MockFeature(pd.DataFrame({"other_col": [1, 2, 3]}))

        with pytest.raises(KeyError, match="Join key 'AGS' not found in test_feature dataframe"):
            builder.build_matrix()

    def test_build_matrix_with_missing_values_drop(self, tmp_path, mock_feature_module):
        """Test handling missing values with drop strategy."""
        config_data = {
            "municipalities": {"class": "MockFeature", "module": mock_feature_module},
            "features": [{"name": "f1", "class": "MockFeature", "module": mock_feature_module}],
            "matrix": {"join_key": "AGS", "missing_values": "drop", "save_output": False},
        }

        config_file = tmp_path / "drop_missing.yaml"
        config_file.write_text(yaml.dump(config_data))

        builder = FeatureMatrixBuilder(config_path=str(config_file))

        # Create data with missing values
        df = pd.DataFrame({"AGS": [1, 2, 3], "val": [10, None, 30]})
        builder.features["f1"] = self.MockFeature(df)

        matrix = builder.build_matrix()

        # Should drop row with missing value
        assert len(matrix) == 2
        assert matrix["f1_val"].notna().all()

    def test_build_matrix_with_missing_values_fill(self, tmp_path, mock_feature_module):
        """Test handling missing values with fill strategy."""
        config_data = {
            "municipalities": {"class": "MockFeature", "module": mock_feature_module},
            "features": [{"name": "f1", "class": "MockFeature", "module": mock_feature_module}],
            "matrix": {"join_key": "AGS", "missing_values": "fill", "fill_value": -999, "save_output": False},
        }

        config_file = tmp_path / "fill_missing.yaml"
        config_file.write_text(yaml.dump(config_data))

        builder = FeatureMatrixBuilder(config_path=str(config_file))

        df = pd.DataFrame({"AGS": [1, 2, 3], "val": [10, None, 30]})
        builder.features["f1"] = self.MockFeature(df)

        matrix = builder.build_matrix()

        # Should fill missing value with -999
        assert len(matrix) == 3
        assert matrix.loc[matrix["AGS"] == 2, "f1_val"].values[0] == -999

    def test_build_matrix_saves_output(self, tmp_path, mock_feature_module):
        """Test that matrix is saved when save_output is True."""
        output_path = tmp_path / "output" / "matrix.csv"

        config_data = {
            "municipalities": {"class": "MockFeature", "module": mock_feature_module},
            "features": [{"name": "f1", "class": "MockFeature", "module": mock_feature_module}],
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
        assert isinstance(feature, self.MockFeature)

        # Test getting non-existent feature
        assert builder.get_feature("nonexistent") is None

    def test_column_prefixing(self, multi_feature_config):
        """Test that feature names are prefixed to columns."""
        builder = FeatureMatrixBuilder(config_path=multi_feature_config)

        builder.features["feature1"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2], "col1": [10, 20]}))
        builder.features["feature2"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2], "col2": [100, 200]}))

        matrix = builder.build_matrix()

        # Check that columns are prefixed
        assert "feature1_col1" in matrix.columns
        assert "feature2_col2" in matrix.columns
        # Join key should not be prefixed
        assert "AGS" in matrix.columns
        assert "feature1_AGS" not in matrix.columns
        assert "feature2_AGS" not in matrix.columns


class TestAfterTransforms:
    """Integration tests for after_transforms (post-merge delta features)."""

    @pytest.fixture(autouse=True)
    def setup_mock(self, mock_feature_class):
        self.MockFeature = mock_feature_class

    def _build_config_with_after_transforms(self, tmp_path, mock_feature_module, after_transforms):
        """Helper to create a config file with after_transforms."""
        config_data = {
            "municipalities": {"class": "MockFeature", "module": mock_feature_module},
            "features": [
                {"name": "f1", "class": "MockFeature", "module": mock_feature_module},
                {"name": "f2", "class": "MockFeature", "module": mock_feature_module},
            ],
            "after_transforms": after_transforms,
            "matrix": {"join_key": "AGS", "save_output": False},
        }
        config_file = tmp_path / "after_tform_config.yaml"
        config_file.write_text(yaml.dump(config_data))
        return str(config_file)

    def test_delta_column_added_to_matrix(self, tmp_path, mock_feature_module):
        """Delta column is computed and appended to the merged matrix."""
        after_transforms = [
            {
                "name": "val_delta",
                "class": "DeltaFeatureEngineering",
                "module": DELTA_MODULE,
                "input_columns": ["f2_value", "f1_value"],
                "output_column": "val_delta",
                "params": {},
            }
        ]
        config_path = self._build_config_with_after_transforms(tmp_path, mock_feature_module, after_transforms)
        builder = FeatureMatrixBuilder(config_path=config_path)

        # Override features so f1_value and f2_value land in matrix with known values
        builder.features["f1"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2, 3], "value": [10.0, 20.0, 30.0]}))
        builder.features["f2"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2, 3], "value": [15.0, 25.0, 35.0]}))

        matrix = builder.build_matrix()

        assert "val_delta" in matrix.columns
        # delta = f2_value - f1_value = 5 for all rows
        assert list(matrix["val_delta"]) == [5.0, 5.0, 5.0]

    def test_delta_null_propagation(self, tmp_path, mock_feature_module):
        """NaN in either input column propagates to the delta column."""
        after_transforms = [
            {
                "name": "val_delta",
                "class": "DeltaFeatureEngineering",
                "module": DELTA_MODULE,
                "input_columns": ["f2_value", "f1_value"],
                "output_column": "val_delta",
                "params": {},
            }
        ]
        config_path = self._build_config_with_after_transforms(tmp_path, mock_feature_module, after_transforms)
        builder = FeatureMatrixBuilder(config_path=config_path)

        builder.features["f1"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2, 3], "value": [10.0, None, 30.0]}))
        builder.features["f2"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2, 3], "value": [15.0, 25.0, 35.0]}))

        matrix = builder.build_matrix()

        import numpy as np

        assert matrix["val_delta"].iloc[0] == 5.0
        assert np.isnan(matrix["val_delta"].iloc[1])
        assert matrix["val_delta"].iloc[2] == 5.0

    def test_multiple_after_transforms_applied_in_order(self, tmp_path, mock_feature_module):
        """Multiple after_transforms are applied sequentially."""
        after_transforms = [
            {
                "name": "delta_a",
                "class": "DeltaFeatureEngineering",
                "module": DELTA_MODULE,
                "input_columns": ["f2_value", "f1_value"],
                "output_column": "delta_a",
                "params": {},
            },
            {
                "name": "delta_b",
                "class": "DeltaFeatureEngineering",
                "module": DELTA_MODULE,
                "input_columns": ["f2_weight", "f1_weight"],
                "output_column": "delta_b",
                "params": {},
            },
        ]
        config_path = self._build_config_with_after_transforms(tmp_path, mock_feature_module, after_transforms)
        builder = FeatureMatrixBuilder(config_path=config_path)

        builder.municipalities = self.MockFeature(pd.DataFrame({"AGS": [1, 2]}))
        builder.features["f1"] = self.MockFeature(
            pd.DataFrame({"AGS": [1, 2], "value": [10.0, 20.0], "weight": [100.0, 200.0]})
        )
        builder.features["f2"] = self.MockFeature(
            pd.DataFrame({"AGS": [1, 2], "value": [15.0, 25.0], "weight": [110.0, 210.0]})
        )

        matrix = builder.build_matrix()

        assert "delta_a" in matrix.columns
        assert "delta_b" in matrix.columns
        assert list(matrix["delta_a"]) == [5.0, 5.0]
        assert list(matrix["delta_b"]) == [10.0, 10.0]

    def test_original_columns_still_present(self, tmp_path, mock_feature_module):
        """After delta transform, the original source columns are still in the matrix."""
        after_transforms = [
            {
                "name": "val_delta",
                "class": "DeltaFeatureEngineering",
                "module": DELTA_MODULE,
                "input_columns": ["f2_value", "f1_value"],
                "output_column": "val_delta",
                "params": {},
            }
        ]
        config_path = self._build_config_with_after_transforms(tmp_path, mock_feature_module, after_transforms)
        builder = FeatureMatrixBuilder(config_path=config_path)

        builder.features["f1"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2], "value": [10.0, 20.0]}))
        builder.features["f2"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2], "value": [15.0, 25.0]}))

        matrix = builder.build_matrix()

        # originals and delta all present
        assert "f1_value" in matrix.columns
        assert "f2_value" in matrix.columns
        assert "val_delta" in matrix.columns

    def test_invalid_after_transform_column_raises(self, tmp_path, mock_feature_module):
        """An after_transform referencing a non-existent column raises an error."""
        after_transforms = [
            {
                "name": "bad_delta",
                "class": "DeltaFeatureEngineering",
                "module": DELTA_MODULE,
                "input_columns": ["nonexistent_col", "f1_value"],
                "output_column": "bad_delta",
                "params": {},
            }
        ]
        config_path = self._build_config_with_after_transforms(tmp_path, mock_feature_module, after_transforms)
        builder = FeatureMatrixBuilder(config_path=config_path)

        builder.features["f1"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2], "value": [10.0, 20.0]}))
        builder.features["f2"] = self.MockFeature(pd.DataFrame({"AGS": [1, 2], "value": [15.0, 25.0]}))

        with pytest.raises(ValueError):
            builder.build_matrix()


def test_default_config():
    """Test that default config is set correctly."""
    builder = FeatureMatrixBuilder()
    assert builder.municipalities is not None


class TestColumnFiltering:
    """Tests for column-level filtering in FeatureMatrixBuilder."""

    @pytest.fixture(autouse=True)
    def setup_mock(self, mock_feature_class):
        self.MockFeature = mock_feature_class

    def _make_config_file(self, tmp_path, mock_feature_module, column_filter: dict | None = None):
        """Helper to build a config YAML with an optional column_filter."""
        feature_cfg = {
            "name": "feat",
            "class": "MockFeature",
            "module": mock_feature_module,
            "params": {},
        }
        if column_filter is not None:
            feature_cfg["column_filter"] = column_filter

        config_data = {
            "municipalities": {"class": "MockFeature", "module": mock_feature_module},
            "features": [feature_cfg],
            "matrix": {"join_key": "AGS", "save_output": False},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))
        return str(config_file)

    # ------------------------------------------------------------------
    # _apply_column_filter unit tests
    # ------------------------------------------------------------------

    def test_apply_column_filter_select_keeps_only_selected(self):
        """select_columns keeps only the listed columns (plus join key)."""
        df = pd.DataFrame({"AGS": [1, 2], "a": [10, 20], "b": [30, 40], "c": [50, 60]})
        cf = ColumnFilteringConfig(select_columns=["a", "c"])
        result = FeatureMatrixBuilder._apply_column_filter(df, "AGS", cf)
        assert list(result.columns) == ["AGS", "a", "c"]

    def test_apply_column_filter_omit_drops_listed_columns(self):
        """omit_columns removes the listed columns but keeps the rest."""
        df = pd.DataFrame({"AGS": [1, 2], "a": [10, 20], "b": [30, 40], "c": [50, 60]})
        cf = ColumnFilteringConfig(omit_columns=["b"])
        result = FeatureMatrixBuilder._apply_column_filter(df, "AGS", cf)
        assert "b" not in result.columns
        assert "a" in result.columns
        assert "c" in result.columns
        assert "AGS" in result.columns

    def test_apply_column_filter_combined(self):
        """select_columns then omit_columns narrows down correctly."""
        df = pd.DataFrame({"AGS": [1], "a": [1], "b": [2], "c": [3], "d": [4]})
        cf = ColumnFilteringConfig(select_columns=["a", "b", "c"], omit_columns=["b"])
        result = FeatureMatrixBuilder._apply_column_filter(df, "AGS", cf)
        assert list(result.columns) == ["AGS", "a", "c"]

    def test_apply_column_filter_join_key_always_preserved(self):
        """Join key is never removed even if all feature columns are filtered out."""
        df = pd.DataFrame({"AGS": [1, 2], "x": [10, 20]})
        cf = ColumnFilteringConfig(omit_columns=["x"])
        result = FeatureMatrixBuilder._apply_column_filter(df, "AGS", cf)
        assert "AGS" in result.columns

    def test_apply_column_filter_glob_select(self):
        """select_columns supports fnmatch glob patterns."""
        df = pd.DataFrame({"AGS": [1], "unemp_total": [5], "unemp_male": [3], "pop_total": [100]})
        cf = ColumnFilteringConfig(select_columns=["unemp_*"])
        result = FeatureMatrixBuilder._apply_column_filter(df, "AGS", cf)
        assert set(result.columns) == {"AGS", "unemp_total", "unemp_male"}
        assert "pop_total" not in result.columns

    def test_apply_column_filter_glob_omit(self):
        """omit_columns supports fnmatch glob patterns."""
        df = pd.DataFrame({"AGS": [1], "unemp_total": [5], "unemp_male": [3], "pop_total": [100]})
        cf = ColumnFilteringConfig(omit_columns=["unemp_*"])
        result = FeatureMatrixBuilder._apply_column_filter(df, "AGS", cf)
        assert "unemp_total" not in result.columns
        assert "unemp_male" not in result.columns
        assert "pop_total" in result.columns

    def test_apply_column_filter_no_filter_is_no_op(self):
        """An empty ColumnFilteringConfig leaves all columns unchanged."""
        df = pd.DataFrame({"AGS": [1, 2], "a": [10, 20], "b": [30, 40]})
        cf = ColumnFilteringConfig()  # select_columns=None, omit_columns=[]
        result = FeatureMatrixBuilder._apply_column_filter(df, "AGS", cf)
        assert list(result.columns) == ["AGS", "a", "b"]

    def test_apply_column_filter_unknown_pattern_warns(self, caplog):
        """A select_columns pattern that matches nothing issues a warning."""
        import logging

        df = pd.DataFrame({"AGS": [1], "a": [10]})
        cf = ColumnFilteringConfig(select_columns=["nonexistent_col"])
        with caplog.at_level(logging.WARNING):
            FeatureMatrixBuilder._apply_column_filter(df, "AGS", cf)
        assert any("matched no columns" in msg for msg in caplog.messages)

    # ------------------------------------------------------------------
    # Integration: filtering applied during build_matrix
    # ------------------------------------------------------------------

    def test_build_matrix_select_columns(self, tmp_path, mock_feature_module):
        """select_columns filters columns before prefixing in build_matrix."""
        config_path = self._make_config_file(
            tmp_path, mock_feature_module, column_filter={"select_columns": ["value"]}
        )
        builder = FeatureMatrixBuilder(config_path=config_path)
        # MockFeature returns AGS, value, weight
        matrix = builder.build_matrix()
        assert "feat_value" in matrix.columns
        assert "feat_weight" not in matrix.columns
        assert "AGS" in matrix.columns

    def test_build_matrix_omit_columns(self, tmp_path, mock_feature_module):
        """omit_columns drops the specified column before prefixing in build_matrix."""
        config_path = self._make_config_file(
            tmp_path, mock_feature_module, column_filter={"omit_columns": ["weight"]}
        )
        builder = FeatureMatrixBuilder(config_path=config_path)
        matrix = builder.build_matrix()
        assert "feat_value" in matrix.columns
        assert "feat_weight" not in matrix.columns
        assert "AGS" in matrix.columns

    def test_build_matrix_no_column_filter(self, tmp_path, mock_feature_module):
        """Without column_filter all columns are kept."""
        config_path = self._make_config_file(tmp_path, mock_feature_module, column_filter=None)
        builder = FeatureMatrixBuilder(config_path=config_path)
        matrix = builder.build_matrix()
        assert "feat_value" in matrix.columns
        assert "feat_weight" in matrix.columns
        assert "AGS" in matrix.columns
