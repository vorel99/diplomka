from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
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
        print(f"\nGrid search results saved to: {cv_results_path}")

        try:
            self._plot_grid_search_results(cv_results_df, self.grid_search.best_params_)
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

    def plot_diagnostics(self, save_path: str | None = None) -> None:
        """Plot predicted-vs-actual and residual diagnostics for holdout set."""
        y_pred = self.best_estimator.predict(self.X_test)
        self._plot_diagnostics(self.y_test, y_pred, save_path=save_path)

    def log_best_model(self, artifact_path: str = "best_model") -> None:
        """Log the best estimator to MLflow."""
        mlflow_wrapper.log_model(self.best_estimator, artifact_path)

    def _plot_diagnostics(self, y_true, y_pred, save_path: str | None = None) -> None:
        """Create diagnostic plots for model evaluation."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
            axes[0].set_xlabel("Actual Values", fontsize=12)
            axes[0].set_ylabel("Predicted Values", fontsize=12)
            axes[0].set_title("Predicted vs Actual Values", fontsize=14, fontweight="bold")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            residuals = y_true - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
            axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
            axes[1].set_xlabel("Predicted Values", fontsize=12)
            axes[1].set_ylabel("Residuals", fontsize=12)
            axes[1].set_title("Residual Plot", fontsize=14, fontweight="bold")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Diagnostic plots saved to: {save_path}")
            else:
                plt.show()

            plt.close()

            if save_path:
                mlflow_wrapper.log_artifact(save_path)
                Path(save_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"Warning: Could not create diagnostic plots: {e}")

    def _plot_grid_search_results(self, cv_results_df: pd.DataFrame, best_params: dict) -> None:
        """Create visualization of grid search results."""
        try:
            param_cols = [col for col in cv_results_df.columns if col.startswith("param_") and col != "params"]

            if not param_cols:
                return

            n_params = len(param_cols)
            fig, axes = plt.subplots(1, min(n_params, 3), figsize=(6 * min(n_params, 3), 5))
            if n_params == 1:
                axes = [axes]

            for idx, param_col in enumerate(param_cols[:3]):
                ax = axes[idx]
                param_name = param_col.replace("param_", "")

                grouped = cv_results_df.groupby(param_col)["mean_test_score"].agg(["mean", "std"])

                x_values = grouped.index.astype(str)
                y_values = grouped["mean"]
                y_std = grouped["std"]

                ax.errorbar(x_values, y_values, yerr=y_std, marker="o", capsize=5, capthick=2, linewidth=2)

                best_value = str(best_params.get(param_name, ""))
                if best_value in x_values:
                    best_idx = list(x_values).index(best_value)
                    ax.plot(x_values[best_idx], y_values.iloc[best_idx], "r*", markersize=20, label="Best", zorder=5)

                ax.set_xlabel(param_name, fontsize=12, fontweight="bold")
                ax.set_ylabel("Mean CV Score (R²)", fontsize=12)
                ax.set_title(f"Impact of {param_name}", fontsize=13, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.legend()

                if len(x_values) > 5:
                    ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plot_path = "grid_search_param_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            mlflow_wrapper.log_artifact(plot_path)
            Path(plot_path).unlink(missing_ok=True)

            print(f"Parameter importance plot saved to: {plot_path}")

        except Exception as e:
            print(f"Warning: Could not create grid search visualization: {e}")

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to wrapped GridSearchCV for compatibility."""
        return getattr(self.grid_search, name)
