"""Tests for the geoscore CLI."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from geoscore_de.cli import app


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def training_config_file(tmp_path):
    """Write a minimal valid training YAML config and return its path."""
    config = {
        "target_variable": "y",
        "train_test_split_ratio": 0.8,
        "model": {
            "model_type": "lightgbm",
            "param_grid": {"n_estimators": [10]},
        },
    }
    path = tmp_path / "training.yaml"
    path.write_text(yaml.dump(config))
    return path


@pytest.fixture()
def input_csv(tmp_path):
    """Write a small CSV file and return its path."""
    df = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def report_qmd(tmp_path):
    """Create a dummy .qmd file and return its path."""
    path = tmp_path / "training_report.qmd"
    path.write_text("# dummy report")
    return path


# ---------------------------------------------------------------------------
# create_feature_matrix command
# ---------------------------------------------------------------------------


class TestCreateFeatureMatrixCommand:
    def test_missing_config_shows_error(self, runner, tmp_path):
        """Non-existent config path should cause a Typer error (exit != 0)."""
        result = runner.invoke(app, ["create-feature-matrix", "--config", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code != 0

    def test_calls_builder_build_matrix(self, runner, tmp_path):
        """Command should call FeatureMatrixBuilder.build_matrix() with the given config."""
        config_path = tmp_path / "features.yaml"
        config_path.write_text("# dummy")

        mock_matrix = pd.DataFrame({"AGS": ["01001"], "col": [1]})
        with patch("geoscore_de.cli.FeatureMatrixBuilder") as MockBuilder:
            instance = MockBuilder.return_value
            instance.build_matrix.return_value = mock_matrix

            result = runner.invoke(app, ["create-feature-matrix", "--config", str(config_path)])

        assert result.exit_code == 0, result.output
        MockBuilder.assert_called_once_with(config_path=str(config_path))
        instance.build_matrix.assert_called_once()
        assert "Shape:" in result.output


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------


class TestTrainCommand:
    def test_missing_config_shows_error(self, runner, tmp_path, input_csv):
        """Non-existent config path should cause a Typer error."""
        result = runner.invoke(app, ["train", str(tmp_path / "no.yaml"), str(input_csv)])
        assert result.exit_code != 0

    def test_missing_input_shows_error(self, runner, training_config_file, tmp_path):
        """Non-existent input CSV path should cause a Typer error."""
        result = runner.invoke(app, ["train", str(training_config_file), str(tmp_path / "no.csv")])
        assert result.exit_code != 0

    def test_starts_mlflow_run(self, runner, training_config_file, input_csv, tmp_path):
        """The train command must start exactly one MLflow run."""
        with (
            patch("geoscore_de.cli.mlflow") as mock_mlflow,
            patch("geoscore_de.cli.Trainer") as MockTrainer,
            patch("geoscore_de.cli._render_and_log_report"),
        ):
            mock_run_ctx = MagicMock()
            mock_mlflow.start_run.return_value = mock_run_ctx
            mock_run_ctx.__enter__ = MagicMock(return_value=MagicMock())
            mock_run_ctx.__exit__ = MagicMock(return_value=False)

            MockTrainer.return_value.train.return_value = MagicMock()

            result = runner.invoke(app, ["train", str(training_config_file), str(input_csv)])

        assert result.exit_code == 0, result.output
        mock_mlflow.start_run.assert_called_once()

    def test_logs_config_and_input_params(self, runner, training_config_file, input_csv):
        """train should log config_path and input_path as MLflow params."""
        with (
            patch("geoscore_de.cli.mlflow") as mock_mlflow,
            patch("geoscore_de.cli.mlflow_wrapper") as mock_wrapper,
            patch("geoscore_de.cli.Trainer") as MockTrainer,
            patch("geoscore_de.cli._render_and_log_report"),
        ):
            mock_run_ctx = MagicMock()
            mock_mlflow.start_run.return_value = mock_run_ctx
            mock_run_ctx.__enter__ = MagicMock(return_value=MagicMock())
            mock_run_ctx.__exit__ = MagicMock(return_value=False)

            MockTrainer.return_value.train.return_value = MagicMock()

            result = runner.invoke(app, ["train", str(training_config_file), str(input_csv)])

        assert result.exit_code == 0, result.output
        param_calls = mock_wrapper.log_param.call_args_list
        keys_logged = {c.args[0] for c in param_calls}
        assert "config_path" in keys_logged
        assert "input_path" in keys_logged

    def test_trainer_called_with_correct_config(self, runner, training_config_file, input_csv):
        """Trainer should be instantiated with a TrainingConfig and called with the DataFrame."""
        from geoscore_de.modelling.config import TrainingConfig

        with (
            patch("geoscore_de.cli.mlflow") as mock_mlflow,
            patch("geoscore_de.cli.mlflow_wrapper"),
            patch("geoscore_de.cli.Trainer") as MockTrainer,
            patch("geoscore_de.cli._render_and_log_report"),
        ):
            mock_run_ctx = MagicMock()
            mock_mlflow.start_run.return_value = mock_run_ctx
            mock_run_ctx.__enter__ = MagicMock(return_value=MagicMock())
            mock_run_ctx.__exit__ = MagicMock(return_value=False)
            MockTrainer.return_value.train.return_value = MagicMock()

            result = runner.invoke(app, ["train", str(training_config_file), str(input_csv)])

        assert result.exit_code == 0, result.output
        config_arg = MockTrainer.call_args.args[0]
        assert isinstance(config_arg, TrainingConfig)
        assert config_arg.target_variable == "y"

        df_arg = MockTrainer.return_value.train.call_args.args[0]
        assert isinstance(df_arg, pd.DataFrame)

    def test_render_report_called_after_training(self, runner, training_config_file, input_csv):
        """_render_and_log_report should be called after Trainer.train()."""
        with (
            patch("geoscore_de.cli.mlflow") as mock_mlflow,
            patch("geoscore_de.cli.mlflow_wrapper"),
            patch("geoscore_de.cli.Trainer") as MockTrainer,
            patch("geoscore_de.cli._render_and_log_report") as mock_render,
        ):
            mock_run_ctx = MagicMock()
            mock_mlflow.start_run.return_value = mock_run_ctx
            mock_run_ctx.__enter__ = MagicMock(return_value=MagicMock())
            mock_run_ctx.__exit__ = MagicMock(return_value=False)
            MockTrainer.return_value.train.return_value = MagicMock()

            result = runner.invoke(app, ["train", str(training_config_file), str(input_csv)])

        assert result.exit_code == 0, result.output
        mock_render.assert_called_once()


# ---------------------------------------------------------------------------
# _render_and_log_report helper
# ---------------------------------------------------------------------------


class TestRenderAndLogReport:
    def test_missing_report_warns_and_returns(self, tmp_path, capsys):
        """When the .qmd file does not exist, a warning is printed and no further action taken."""
        from geoscore_de.cli import _render_and_log_report

        with patch("geoscore_de.cli.mlflow_wrapper") as mock_wrapper:
            _render_and_log_report(tmp_path / "no.qmd", tmp_path / "cfg.yaml", tmp_path / "data.csv")
            mock_wrapper.log_artifact.assert_not_called()

    def test_quarto_not_found_warns(self, tmp_path):
        """When quarto is not installed, a warning is printed and no artifact logged."""
        from geoscore_de.cli import _render_and_log_report

        qmd = tmp_path / "report.qmd"
        qmd.write_text("# hello")

        with (
            patch("geoscore_de.cli.subprocess.run", side_effect=FileNotFoundError),
            patch("geoscore_de.cli.mlflow_wrapper") as mock_wrapper,
        ):
            _render_and_log_report(qmd, tmp_path / "cfg.yaml", tmp_path / "data.csv")
            mock_wrapper.log_artifact.assert_not_called()

    def test_quarto_failure_warns(self, tmp_path):
        """When quarto exits with non-zero code, a warning is printed and no artifact logged."""
        import subprocess

        from geoscore_de.cli import _render_and_log_report

        qmd = tmp_path / "report.qmd"
        qmd.write_text("# hello")

        with (
            patch(
                "geoscore_de.cli.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "quarto", stderr="oops"),
            ),
            patch("geoscore_de.cli.mlflow_wrapper") as mock_wrapper,
        ):
            _render_and_log_report(qmd, tmp_path / "cfg.yaml", tmp_path / "data.csv")
            mock_wrapper.log_artifact.assert_not_called()

    def test_logs_qmd_and_html_on_success(self, tmp_path):
        """When Quarto succeeds and the HTML exists, both source and HTML are logged."""
        from geoscore_de.cli import _render_and_log_report

        qmd = tmp_path / "report.qmd"
        qmd.write_text("# hello")
        html = qmd.with_suffix(".html")
        html.write_text("<html/>")

        with (
            patch("geoscore_de.cli.subprocess.run", return_value=MagicMock(returncode=0, stdout="")),
            patch("geoscore_de.cli.mlflow_wrapper") as mock_wrapper,
        ):
            _render_and_log_report(qmd, tmp_path / "cfg.yaml", tmp_path / "data.csv")

        assert mock_wrapper.log_artifact.call_count == 2
        logged_paths = {c.args[0] for c in mock_wrapper.log_artifact.call_args_list}
        assert str(qmd) in logged_paths
        assert str(html) in logged_paths

    def test_logs_only_qmd_when_html_missing(self, tmp_path):
        """When Quarto succeeds but no HTML output is found, only the .qmd is logged."""
        from geoscore_de.cli import _render_and_log_report

        qmd = tmp_path / "report.qmd"
        qmd.write_text("# hello")

        with (
            patch("geoscore_de.cli.subprocess.run", return_value=MagicMock(returncode=0, stdout="")),
            patch("geoscore_de.cli.mlflow_wrapper") as mock_wrapper,
        ):
            _render_and_log_report(qmd, tmp_path / "cfg.yaml", tmp_path / "data.csv")

        mock_wrapper.log_artifact.assert_called_once_with(str(qmd), artifact_path="report")
