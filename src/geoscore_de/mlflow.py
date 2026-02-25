import pickle
import shutil
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import pandas as pd


def require_active_run(func):
    """Decorator that makes function no-op when no active run."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if mlflow.active_run() is None:
            return None
        return func(*args, **kwargs)

    return wrapper


# Standard MLflow logging functions
@require_active_run
def log_metric(key: str, value: float, step: int = None):
    """Log a metric under the current run."""
    return mlflow.log_metric(key, value, step)


@require_active_run
def log_metrics(metrics: dict, step: int = None):
    """Log multiple metrics for the current run."""
    return mlflow.log_metrics(metrics, step)


@require_active_run
def log_param(key: str, value: Any):
    """Log a parameter under the current run."""
    return mlflow.log_param(key, value)


@require_active_run
def log_params(params: dict):
    """Log multiple parameters for the current run."""
    return mlflow.log_params(params)


@require_active_run
def log_artifact(local_path: str, artifact_path: str = None):
    """Log a local file or directory as an artifact."""
    return mlflow.log_artifact(local_path, artifact_path)


@require_active_run
def log_artifacts(local_dir: str, artifact_path: str = None):
    """Log all contents of a local directory as artifacts."""
    return mlflow.log_artifacts(local_dir, artifact_path)


@require_active_run
def log_dict(dictionary: dict, artifact_file: str):
    """Log a dictionary as a JSON or YAML artifact."""
    return mlflow.log_dict(dictionary, artifact_file)


@require_active_run
def log_figure(figure, artifact_file: str):
    """Log a figure as an artifact."""
    return mlflow.log_figure(figure, artifact_file)


@require_active_run
def log_text(text: str, artifact_file: str):
    """Log text as an artifact."""
    return mlflow.log_text(text, artifact_file)


@require_active_run
def log_image(image, artifact_file: str = None, key: str = None):
    """Log an image as an artifact."""
    return mlflow.log_image(image, artifact_file, key)


@require_active_run
def set_tag(key: str, value: Any):
    """Set a tag under the current run."""
    return mlflow.set_tag(key, value)


@require_active_run
def set_tags(tags: dict):
    """Set multiple tags for the current run."""
    return mlflow.set_tags(tags)


# Custom logging functions
@require_active_run
def log_data(data: pd.DataFrame | pd.Series, artifact_file: str, **pandas_kwargs):
    """
    Logs data as an artifact (csv or parquet).
    Works similarly as log_dict(...), log_figure(...), log_text(...).
    """
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    pth = Path(artifact_file)
    suffix = pth.suffix
    basename = pth.name
    dirname = str(pth.parent) if str(pth.parent) != "." else None
    with TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / basename
        if suffix == ".csv":
            data.to_csv(local_path, **pandas_kwargs)
        elif suffix == ".parquet":
            data.to_parquet(local_path, **pandas_kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix.lstrip('.')}")

        mlflow.log_artifact(local_path=str(local_path), artifact_path=dirname)


@require_active_run
def log_html(html_path: Path | str, artifact_path: str):
    """Logs an HTML file as an artifact."""
    pth = Path(artifact_path)
    basename = pth.name
    dirname = str(pth.parent) if str(pth.parent) != "." else None

    with TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / basename
        shutil.copy(html_path, local_path)
        mlflow.log_artifact(local_path=str(local_path), artifact_path=dirname)


@require_active_run
def log_pickle(obj: Any, artifact_file: str):
    """
    Logs pickle-able object as an artifact.
    Workaround for logging models without knowing proper API.
    """
    pth = Path(artifact_file)
    suffix = pth.suffix
    basename = pth.name
    dirname = str(pth.parent) if str(pth.parent) != "." else None
    with TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / basename
        if suffix == ".pickle":
            with open(local_path, "wb") as file_:
                pickle.dump(obj, file_)
        else:
            raise ValueError(f"Unsupported format: {suffix.lstrip('.')}")

        mlflow.log_artifact(local_path=str(local_path), artifact_path=dirname)


# Model logging functions
@require_active_run
def log_catboost_model(model, artifact_path: str = "model", **kwargs):
    """Log a CatBoost model."""
    import mlflow.catboost

    return mlflow.catboost.log_model(model, artifact_path, **kwargs)
