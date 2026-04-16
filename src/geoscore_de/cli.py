"""Typer CLI for geoscore_de: feature-matrix creation and model training."""

import logging
from pathlib import Path

import mlflow
import quarto
import typer

from geoscore_de import mlflow_wrapper
from geoscore_de.data_flow.matrix_builder import FeatureMatrixBuilder

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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Build the feature matrix from the features configuration and save to disk."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

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
        help="Path to the input feature matrix file (.csv or .parquet).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    report_path: Path = typer.Option(
        Path("reports/training_report.qmd"),
        "--report",
        "-r",
        help="Path to the Quarto report template to render (training runs inside the report).",
        resolve_path=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Render the training report (training runs inside it) and log artifacts to MLflow."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    with mlflow.start_run():
        mlflow_wrapper.log_param("config_path", str(config_path))
        mlflow_wrapper.log_param("input_path", str(input_path))

        _render_and_log_report(report_path, config_path, input_path)


def _render_and_log_report(report_path: Path, config_path: Path, input_path: Path) -> None:
    """Render the Quarto report and log both the .qmd source and the HTML output to MLflow."""
    if not report_path.exists():
        typer.echo(f"Warning: report template not found at {report_path}, skipping report rendering.", err=True)
        return

    output_html = report_path.with_suffix(".html")
    typer.echo(f"Rendering Quarto report: {report_path}")

    try:
        quarto.render(
            str(report_path),
            output_format="html",
            execute_params={
                "training_config_path": str(config_path),
                "input_path": str(input_path),
            },
            cache=False,
        )
    except Exception as exc:
        typer.echo(f"Warning: Quarto rendering failed: {exc}", err=True)
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
