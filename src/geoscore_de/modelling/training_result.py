from pathlib import Path
from typing import Any

import pandas as pd
import plotnine as gg
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from geoscore_de import mlflow_wrapper
from geoscore_de.modelling.plots.diagnostic_plots import build_predicted_vs_actual_plot, build_residual_plot
from geoscore_de.modelling.plots.grid_search_plots import build_grid_search_results_plot


class TrainingResult:
    """Container for training artifacts with post-training utilities.

    The object wraps the fitted sklearn search object and keeps holdout data so callers can
    evaluate, plot diagnostics, and log artifacts without going back to ``Trainer``.
    """

    X_train: pd.DataFrame
    y_train: pd.Series

    X_val: pd.DataFrame | None
    y_val: pd.Series | None

    X_test: pd.DataFrame
    y_test: pd.Series

    def __init__(
        self,
        grid_search: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        test_indices: pd.Index | None = None,
        best_estimator_override: Any | None = None,
    ):
        self.grid_search = grid_search

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.test_indices = test_indices if test_indices is not None else X_test.index
        self._best_estimator_override = best_estimator_override
        self.metrics: dict[str, float | None] | None = None

    @property
    def best_estimator(self):
        if self._best_estimator_override is not None:
            return self._best_estimator_override
        return self.grid_search.best_estimator_

    @property
    def best_estimator_(self):
        """Backwards-compatible alias with sklearn naming style."""
        return self.best_estimator

    @property
    def best_params_(self):
        """Backwards-compatible alias with sklearn naming style."""
        return self.grid_search.best_params_

    def log_grid_search_results(self) -> None:
        """Log detailed grid-search outputs and plots to MLflow."""
        print("\n" + "=" * 60)
        print("GRID SEARCH RESULTS")
        print("=" * 60)
        print(f"Best CV Score (R²): {self.grid_search.best_score_:.4f}")
        print("\nBest Parameters:")

        for param_name, param_value in self.grid_search.best_params_.items():
            print(f"  {param_name}: {param_value}")
            mlflow_wrapper.log_param(f"best_{param_name}", param_value)

        mlflow_wrapper.log_metric("cv_best_score", self.grid_search.best_score_)

        cv_results_path = "grid_search_cv_results.csv"
        cv_results_df = pd.DataFrame(self.grid_search.cv_results_)
        mlflow_wrapper.log_data(cv_results_df, artifact_file=cv_results_path, index=False)
        print(f"\nGrid search results logged to MLflow as artifact: {cv_results_path}")

        try:
            grid_search_plot = build_grid_search_results_plot(cv_results_df, self.grid_search.best_params_)
            if grid_search_plot is not None:
                grid_search_plot_path = "grid_search_results.png"
                grid_search_plot.save(grid_search_plot_path, dpi=300, width=14, height=5, units="in", verbose=False)
                print(f"Grid search plots saved to: {grid_search_plot_path}")
                mlflow_wrapper.log_artifact(grid_search_plot_path)
                Path(grid_search_plot_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: Could not create grid search plots: {e}")

        n_models = len(cv_results_df)
        mlflow_wrapper.log_metric("n_models_trained", n_models)
        print(f"\nTotal models evaluated: {n_models}")

        if hasattr(self.grid_search, "param_grid"):
            param_space = self.grid_search.param_grid
        elif hasattr(self.grid_search, "param_distributions"):
            param_space = self.grid_search.param_distributions
        else:
            param_space = {}

        n_params = len(param_space)
        mlflow_wrapper.log_metric("n_hyperparameters_tuned", n_params)

        print("=" * 60 + "\n")

    def evaluate(self, create_plots: bool = True) -> dict[str, float | None]:
        """Evaluate the best model on holdout set and store metrics."""
        y_pred = self.best_estimator.predict(self.X_test)

        self.metrics = self._compute_metrics(self.y_test, y_pred, metric_prefix="test")

        for metric_name, metric_value in self.metrics.items():
            if metric_value is not None:
                mlflow_wrapper.log_metric(metric_name, metric_value)

        r2 = self.metrics["test_r2_score"]
        mae = self.metrics["test_mae"]
        rmse = self.metrics["test_rmse"]
        mse = self.metrics["test_mse"]
        mape = self.metrics["test_mape"]
        med_ae = self.metrics["test_median_ae"]
        max_err = self.metrics["test_max_error"]
        explained_var = self.metrics["test_explained_variance"]

        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"R² Score:              {r2:.4f}")
        print(f"Mean Absolute Error:   {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Squared Error:    {mse:.4f}")
        if mape is not None:
            print(f"Mean Abs % Error:      {mape:.4f} ({mape * 100:.2f}%)")
        print(f"Median Absolute Error: {med_ae:.4f}")
        print(f"Max Error:             {max_err:.4f}")
        print(f"Explained Variance:    {explained_var:.4f}")
        print("=" * 60 + "\n")

        if create_plots:
            self._plot_diagnostics(self.y_test, y_pred, save_path="model_diagnostics.png")

        return self.metrics

    def evaluate_subset(
        self, X_subset: pd.DataFrame, y_subset: pd.Series, metric_prefix: str
    ) -> dict[str, float | None]:
        """Evaluate model on a subset using the same metric suite as holdout evaluation."""
        y_pred = self.best_estimator.predict(X_subset)
        return self._compute_metrics(y_subset, y_pred, metric_prefix=metric_prefix)

    def _compute_metrics(
        self, y_true: pd.Series, y_pred: pd.Series | list[float], metric_prefix: str
    ) -> dict[str, float | None]:
        """Compute a consistent set of regression metrics with resilient MAPE handling."""
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except (ZeroDivisionError, ValueError):
            mape = None

        med_ae = median_absolute_error(y_true, y_pred)
        max_err = max_error(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)

        return {
            f"{metric_prefix}_r2_score": r2,
            f"{metric_prefix}_mae": mae,
            f"{metric_prefix}_mse": mse,
            f"{metric_prefix}_rmse": rmse,
            f"{metric_prefix}_mape": mape,
            f"{metric_prefix}_median_ae": med_ae,
            f"{metric_prefix}_max_error": max_err,
            f"{metric_prefix}_explained_variance": explained_var,
        }

    def plot_diagnostics(self, save_path: str | None = None) -> None | gg.ggplot:
        """Plot predicted-vs-actual and residual diagnostics for holdout set.

        Args:
            save_path: If provided, saves the plot to this path in MLflow.
        """
        y_pred = self.best_estimator.predict(self.X_test)
        return self._plot_diagnostics(self.y_test, y_pred, save_path=save_path)

    def log_best_model(self, artifact_path: str = "best_model") -> None:
        """Log the best estimator to MLflow."""
        mlflow_wrapper.log_model(self.best_estimator, artifact_path)

    def _plot_diagnostics(self, y_true, y_pred, save_path: str | None = None) -> None | Any:
        """Create diagnostic plots for model evaluation."""
        try:
            predicted_vs_actual_plot = build_predicted_vs_actual_plot(y_true, y_pred)
            residuals_plot = build_residual_plot(y_true, y_pred)
            diagnostic_plot = predicted_vs_actual_plot | residuals_plot

            if save_path:
                diagnostic_plot.save(save_path, dpi=300, width=14, height=5, units="in", verbose=False)
                mlflow_wrapper.log_artifact(save_path)
                Path(save_path).unlink(missing_ok=True)

            return diagnostic_plot

        except Exception as e:
            print(f"Warning: Could not create diagnostic plots: {e}")
