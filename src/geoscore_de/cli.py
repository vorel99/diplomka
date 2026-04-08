"""Typer CLI for geoscore_de: feature-matrix creation and model training."""

import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import typer
import yaml

from geoscore_de import mlflow_wrapper
from geoscore_de.data_flow.features.matrix_builder import FeatureMatrixBuilder
from geoscore_de.modelling.config import TrainingConfig
from geoscore_de.modelling.train import Trainer

app = typer.Typer(name="geoscore", help="GeoScore Germany CLI.")


@app.command()
def create_feature_matrix(
    config_path: Path = typer.Option(
        Path("configs/features.yaml"),
        "--config",
        "-c",
        help="Path to the features YAML configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Build the feature matrix from the features configuration and save to disk."""
    typer.echo(f"Building feature matrix using config: {config_path}")
    builder = FeatureMatrixBuilder(config_path=str(config_path))
    matrix = builder.build_matrix()
    typer.echo(f"Feature matrix built successfully. Shape: {matrix.shape}")


@app.command()
def train(
    config_path: Path = typer.Argument(
        ...,
        help="Path to the training YAML configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    input_path: Path = typer.Argument(
        ...,
        help="Path to the input CSV file containing the feature matrix.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    report_path: Path = typer.Option(
        Path("reports/training_report.qmd"),
        "--report",
        "-r",
        help="Path to the Quarto report template to render after training.",
        resolve_path=True,
    ),
) -> None:
    """Train the model using the provided config and input CSV, then render and log the report."""
    typer.echo(f"Loading training config from: {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    training_config = TrainingConfig(**config_dict)

    typer.echo(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    with mlflow.start_run():
        mlflow_wrapper.log_param("config_path", str(config_path))
        mlflow_wrapper.log_param("input_path", str(input_path))

        typer.echo("Starting model training...")
        trainer = Trainer(training_config)
        trainer.train(df)
        typer.echo("Training complete.")

        _render_and_log_report(report_path, config_path, input_path)


def _render_and_log_report(report_path: Path, config_path: Path, input_path: Path) -> None:
    """Render the Quarto report and log both the .qmd source and the HTML output to MLflow."""
    if not report_path.exists():
        typer.echo(f"Warning: report template not found at {report_path}, skipping report rendering.", err=True)
        return

    output_html = report_path.with_suffix(".html")
    typer.echo(f"Rendering Quarto report: {report_path}")

    try:
        result = subprocess.run(
            [
                "quarto",
                "render",
                str(report_path),
                "--to",
                "html",
                "--execute-param",
                f"training_config_path:{config_path}",
                "--execute-param",
                f"input_path:{input_path}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        typer.echo(result.stdout)
    except FileNotFoundError:
        typer.echo("Warning: quarto executable not found, skipping report rendering.", err=True)
        return
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Warning: Quarto rendering failed:\n{exc.stderr}", err=True)
        return

    mlflow_wrapper.log_artifact(str(report_path), artifact_path="report")
    typer.echo(f"Logged report source: {report_path}")

    if output_html.exists():
        mlflow_wrapper.log_artifact(str(output_html), artifact_path="report")
        typer.echo(f"Logged rendered report: {output_html}")
    else:
        typer.echo(f"Warning: expected rendered HTML at {output_html} but file not found.", err=True)


def main() -> None:
    """Entry point for the geoscore CLI."""
    app()


if __name__ == "__main__":
    main()
