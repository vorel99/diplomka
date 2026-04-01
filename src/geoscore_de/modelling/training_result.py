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
from sklearn.model_selection import GridSearchCV

from geoscore_de import mlflow_wrapper
from geoscore_de.modelling.plots.diagnostic_plots import build_predicted_vs_actual_plot, build_residual_plot
from geoscore_de.modelling.plots.grid_search_plots import build_plot_grid_search_results


class TrainingResult:
    """Container for training artifacts with post-training utilities.

    The object wraps the fitted ``GridSearchCV`` and keeps holdout data so callers can
    evaluate, plot diagnostics, and log artifacts without going back to ``Trainer``.
    """

    def __init__(
        self,
        grid_search: GridSearchCV,
        X_train_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train_val: pd.Series,
        y_test: pd.Series,
    ):
        self.grid_search = grid_search
        self.X_train_val = X_train_val
        self.X_test = X_test
        self.y_train_val = y_train_val
        self.y_test = y_test
        self.metrics: dict[str, float | None] | None = None

    @property
    def best_estimator(self):
        return self.grid_search.best_estimator_

    @property
    def best_estimator_(self):
        """Backwards-compatible alias with sklearn naming style."""
        return self.grid_search.best_estimator_

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
            grid_search_plot = build_plot_grid_search_results(cv_results_df, self.grid_search.best_params_)
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

        param_grid = self.grid_search.param_grid
        n_params = len(param_grid)
        mlflow_wrapper.log_metric("n_hyperparameters_tuned", n_params)

        print("=" * 60 + "\n")

    def evaluate(self, create_plots: bool = True) -> dict[str, float | None]:
        """Evaluate the best model on holdout set and store metrics."""
        y_pred = self.best_estimator.predict(self.X_test)

        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = root_mean_squared_error(self.y_test, y_pred)

        try:
            mape = mean_absolute_percentage_error(self.y_test, y_pred)
        except (ZeroDivisionError, ValueError):
            mape = None

        med_ae = median_absolute_error(self.y_test, y_pred)
        max_err = max_error(self.y_test, y_pred)
        explained_var = explained_variance_score(self.y_test, y_pred)

        self.metrics = {
            "test_r2_score": r2,
            "test_mae": mae,
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mape": mape,
            "test_median_ae": med_ae,
            "test_max_error": max_err,
            "test_explained_variance": explained_var,
        }

        for metric_name, metric_value in self.metrics.items():
            if metric_value is not None:
                mlflow_wrapper.log_metric(metric_name, metric_value)

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

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to wrapped GridSearchCV for compatibility."""
        return getattr(self.grid_search, name)
