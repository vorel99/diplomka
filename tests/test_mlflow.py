# tests/test_mlflow.py
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from geoscore_de import mlflow_wrapper


class TestRequireActiveRun:
    """Test the decorator behavior."""

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run")
    def test_decorator_with_active_run(self, mock_active_run: MagicMock):
        """Function executes when run is active."""
        mock_active_run.return_value = MagicMock()  # Simulate active run

        with patch("mlflow.log_param") as mock_log:
            _ = mlflow_wrapper.log_param("key", "value")
            mock_log.assert_called_once_with("key", "value")

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run")
    def test_decorator_without_active_run(self, mock_active_run: MagicMock):
        """Function returns None when no active run."""
        mock_active_run.return_value = None  # No active run

        with patch("mlflow.log_param") as mock_log:
            result = mlflow_wrapper.log_param("key", "value")
            assert result is None
            mock_log.assert_not_called()


class TestStandardLogging:
    """Test wrapper functions for standard MLflow operations."""

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run", return_value=MagicMock())
    @patch("mlflow.log_metric")
    def test_log_metric(self, mock_log_metric: MagicMock, mock_active):
        mlflow_wrapper.log_metric("accuracy", 0.95, step=1)
        mock_log_metric.assert_called_once_with("accuracy", 0.95, 1)

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run", return_value=MagicMock())
    @patch("mlflow.log_params")
    def test_log_params(self, mock_log_params: MagicMock, mock_active):
        params = {"lr": 0.01, "epochs": 10}
        mlflow_wrapper.log_params(params)
        mock_log_params.assert_called_once_with(params)


class TestCustomLogging:
    """Test custom logging functions."""

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run", return_value=MagicMock())
    @patch("mlflow.log_artifact")
    def test_log_data_csv(self, mock_log_artifact: MagicMock, mock_active):
        """Test logging DataFrame as CSV."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        mlflow_wrapper.log_data(df, "folder/data.csv")

        # Verify log_artifact was called
        mock_log_artifact.assert_called_once()
        # Get the call arguments
        call_kwargs = mock_log_artifact.call_args.kwargs

        # Check local_path (temp file) ends with correct filename
        assert call_kwargs["local_path"].endswith("data.csv")

        # Check artifact_path is the directory
        assert call_kwargs["artifact_path"] == "folder"

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run", return_value=MagicMock())
    @patch("mlflow.log_artifact")
    def test_log_data_parquet(self, mock_log_artifact: MagicMock, mock_active):
        """Test logging DataFrame as Parquet."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        mlflow_wrapper.log_data(df, "folder/data.parquet")

        mock_log_artifact.assert_called_once()
        call_kwargs = mock_log_artifact.call_args.kwargs
        assert call_kwargs["local_path"].endswith("data.parquet")
        assert call_kwargs["artifact_path"] == "folder"

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run", return_value=MagicMock())
    def test_log_data_unsupported_format(self, mock_active):
        """Test error on unsupported format."""
        df = pd.DataFrame({"a": [1, 2]})

        with pytest.raises(ValueError, match="Unsupported format"):
            mlflow_wrapper.log_data(df, "data.txt")

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run", return_value=MagicMock())
    @patch("mlflow.log_artifact")
    def test_log_pickle(self, mock_log_artifact: MagicMock, mock_active):
        """Test pickling objects."""
        obj = {"key": "value"}

        mlflow_wrapper.log_pickle(obj, "object.pickle")

        mock_log_artifact.assert_called_once()


class TestModelLogging:
    """Test model logging with type detection."""

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run", return_value=MagicMock())
    def test_log_lightgbm_model(self, mock_active_run):
        """Test auto-detection of LightGBM model."""
        from lightgbm import LGBMRegressor

        model = LGBMRegressor()

        # Mock the entire lightgbm submodule to prevent real MLflow calls
        with patch("mlflow.lightgbm") as mock_lightgbm:
            mock_log_model = MagicMock()
            mock_lightgbm.log_model = mock_log_model

            mlflow_wrapper.log_model(model, "model")

            mock_log_model.assert_called_once_with(model, "model")

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run", return_value=MagicMock())
    def test_log_sklearn_model(self, mock_active):
        """Test auto-detection of sklearn model."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()

        # Mock the entire sklearn submodule to prevent real MLflow calls
        with patch("mlflow.sklearn") as mock_sklearn:
            mock_log_model = MagicMock()
            mock_sklearn.log_model = mock_log_model

            mlflow_wrapper.log_model(model, "model")

            mock_log_model.assert_called_once_with(model, "model")

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run", return_value=MagicMock())
    def test_log_model_none_raises_error(self, mock_active):
        """Test that None model raises ValueError."""
        with pytest.raises(ValueError, match="log_model.*was called with model=None"):
            mlflow_wrapper.log_model(None, "model")

    @patch("geoscore_de.mlflow_wrapper.mlflow.active_run", return_value=MagicMock())
    def test_log_model_invalid_type_raises_error(self, mock_active):
        """Test that invalid model type raises TypeError."""
        # Create object with no __module__
        invalid_model = type("BadModel", (), {})()
        invalid_model.__class__.__module__ = None

        with pytest.raises(TypeError, match="could not determine the model type"):
            mlflow_wrapper.log_model(invalid_model, "model")
