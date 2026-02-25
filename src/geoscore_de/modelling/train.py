import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
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
from sklearn.model_selection import GridSearchCV, train_test_split

from geoscore_de import mlflow_wrapper
from geoscore_de.modelling.config import TrainingConfig


class Trainer:
    """Model trainer with comprehensive evaluation and hyperparameter logging."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    # TODO: implement method to filter rows based on config
    def _filter_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    # TODO: add support for regex in feature filtering
    def _filter_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter features based on use_features and omit_features lists in config.
        Features in use_features are selected first, then features in omit_features are dropped from the result.

        Args:
            X (pd.DataFrame): Input features DataFrame before filtering.
        Returns:
            pd.DataFrame: Filtered features DataFrame.
        """
        feature_filter = self.config.feature_filtering
        if feature_filter.use_features:
            missing_features = set(feature_filter.use_features) - set(X.columns)
            if missing_features:
                warnings.warn(f"Missing features in use_features: {missing_features}")
            available_features = [f for f in feature_filter.use_features if f in X.columns]
            X = X[available_features]

        if feature_filter.omit_features:
            missing_features = set(feature_filter.omit_features) - set(X.columns)
            if missing_features:
                warnings.warn(f"Missing features in omit_features: {missing_features}")
            X = X.drop(columns=feature_filter.omit_features, errors="ignore")
        return X

    def _prepare_data(self, data: pd.DataFrame):
        """Prepare data for model training by applying row and feature filtering
        and splitting into train and test sets.

        Args:
            data (pd.DataFrame): The input DataFrame containing features and target variable.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Split data into
            X_train_val, X_test, y_train_val, y_test.
        """
        # remove ID column if it exists
        if self.config.id_column in data.columns:
            data = data.drop(columns=[self.config.id_column])

        # filter rows
        data = self._filter_rows(data)

        # drop rows with missing target variable
        data = data.dropna(subset=[self.config.target_variable])

        # filter features
        X = self._filter_features(data.drop(columns=[self.config.target_variable]))
        y = data[self.config.target_variable]

        # Split into train and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=1 - self.config.train_test_split_ratio, random_state=self.config.random_state
        )

        return X_train_val, X_test, y_train_val, y_test

    # TODO: implement method to get model based on config
    def _get_model(self):
        """Get LGBMRegressor with sensible defaults.

        These base parameters reduce warnings and improve training:
        - verbosity=-1: Suppress info/warning messages
        - min_child_samples=20: Prevent overfitting to small groups
        - min_split_gain=0.0: Allow any beneficial split
        - n_jobs=-1: Use all CPU cores

        GridSearchCV will override these with param_grid values during tuning.
        """
        return LGBMRegressor(
            random_state=self.config.random_state,
            verbosity=-1,  # Suppress warnings
            min_child_samples=20,  # Reasonable default for stability
            min_split_gain=0.0,  # Allow splits with any gain
            n_jobs=-1,  # Parallel processing
        )

    def _plot_diagnostics(self, y_true, y_pred, save_path: str | None = None):
        """Create diagnostic plots for model evaluation.

        Args:
            y_true: True target values
            y_pred: Predicted values
            save_path (str | None): Optional path to save the plot
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Predicted vs Actual
            axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
            axes[0].set_xlabel("Actual Values", fontsize=12)
            axes[0].set_ylabel("Predicted Values", fontsize=12)
            axes[0].set_title("Predicted vs Actual Values", fontsize=14, fontweight="bold")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Residuals vs Predicted
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

            # Log to MLflow if path provided
            if save_path:
                mlflow_wrapper.log_artifact(save_path)

                # clean up plot file if it was saved locally
                Path(save_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"Warning: Could not create diagnostic plots: {e}")

    def _log_grid_search_results(self, grid_search: GridSearchCV):
        """Log comprehensive grid search results to MLflow.

        Logging includes:
        1. Best parameters individually for easy comparison
        2. Save full cv_results as CSV artifact for detailed analysis
        3. Create visualization of parameter importance
        4. Log CV score statistics (mean, std)
        5. Track total training time

        Args:
            grid_search (GridSearchCV): Fitted GridSearchCV object
        """
        # 1. Log best parameters individually
        print("\n" + "=" * 60)
        print("GRID SEARCH RESULTS")
        print("=" * 60)
        print(f"Best CV Score (R²): {grid_search.best_score_:.4f}")
        print("\nBest Parameters:")

        for param_name, param_value in grid_search.best_params_.items():
            print(f"  {param_name}: {param_value}")
            # Log to MLflow with 'param_' prefix
            mlflow_wrapper.log_param(f"best_{param_name}", param_value)

        # Log best CV score
        mlflow_wrapper.log_metric("cv_best_score", grid_search.best_score_)

        # 2. Create DataFrame from cv_results and log as artifact
        cv_results_path = "grid_search_cv_results.csv"
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        mlflow_wrapper.log_data(cv_results_df, artifact_file=cv_results_path, index=False)
        print(f"\nGrid search results saved to: {cv_results_path}")

        # 3. Create visualization if matplotlib available
        try:
            self._plot_grid_search_results(cv_results_df, grid_search.best_params_)
        except Exception as e:
            print(f"Warning: Could not create grid search plots: {e}")

        # 4. Log total number of models trained
        n_models = len(cv_results_df)
        mlflow_wrapper.log_metric("n_models_trained", n_models)
        print(f"\nTotal models evaluated: {n_models}")

        # 5. Log parameter grid info
        param_grid = grid_search.param_grid
        n_params = len(param_grid)
        mlflow_wrapper.log_metric("n_hyperparameters_tuned", n_params)

        print("=" * 60 + "\n")

    def _plot_grid_search_results(self, cv_results_df: pd.DataFrame, best_params: dict):
        """Create visualization of grid search results.

        Args:
            cv_results_df (pd.DataFrame): DataFrame of cv_results_
            best_params (dict): Best parameters found
        """
        try:
            # Extract parameter columns
            param_cols = [col for col in cv_results_df.columns if col.startswith("param_") and col != "params"]

            if not param_cols:
                return

            n_params = len(param_cols)

            # Create subplot for each parameter
            fig, axes = plt.subplots(1, min(n_params, 3), figsize=(6 * min(n_params, 3), 5))
            if n_params == 1:
                axes = [axes]

            for idx, param_col in enumerate(param_cols[:3]):  # Limit to 3 plots
                ax = axes[idx]
                param_name = param_col.replace("param_", "")

                # Group by parameter value and calculate mean score
                grouped = cv_results_df.groupby(param_col)["mean_test_score"].agg(["mean", "std"])

                # Plot
                x_values = grouped.index.astype(str)
                y_values = grouped["mean"]
                y_std = grouped["std"]

                ax.errorbar(x_values, y_values, yerr=y_std, marker="o", capsize=5, capthick=2, linewidth=2)

                # Highlight best parameter
                best_value = str(best_params.get(param_name, ""))
                if best_value in x_values:
                    best_idx = list(x_values).index(best_value)
                    ax.plot(x_values[best_idx], y_values.iloc[best_idx], "r*", markersize=20, label="Best", zorder=5)

                ax.set_xlabel(param_name, fontsize=12, fontweight="bold")
                ax.set_ylabel("Mean CV Score (R²)", fontsize=12)
                ax.set_title(f"Impact of {param_name}", fontsize=13, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Rotate x-labels if needed
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

    def _evaluate(self, model, X_test, y_test, create_plots=True):
        """Comprehensive evaluation of the model using multiple metrics.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            create_plots: Whether to create diagnostic plots

        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate core regression metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        # Additional metrics
        try:
            # MAPE can fail if y_test contains zeros
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except (ZeroDivisionError, ValueError):
            mape = None

        med_ae = median_absolute_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)

        # Log all metrics to MLflow
        metrics = {
            "test_r2_score": r2,
            "test_mae": mae,
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mape": mape,
            "test_median_ae": med_ae,
            "test_max_error": max_err,
            "test_explained_variance": explained_var,
        }

        # Log to MLflow
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow_wrapper.log_metric(metric_name, metric_value)

        # Print summary
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

        # Create diagnostic plots
        if create_plots:
            self._plot_diagnostics(y_test, y_pred, save_path="model_diagnostics.png")

        return metrics

    def train(self, data: pd.DataFrame):
        """Train the model using the provided data and configuration.

        Args:
            data (pd.DataFrame): The input DataFrame containing features and target variable.

        Returns:
            GridSearchCV: The fitted GridSearchCV object containing the best model and results.
        """
        # Log training configuration to MLflow for reproducibility
        config_dict = self.config.model_dump()
        mlflow_wrapper.log_dict(config_dict, "config.json")

        X_train_val, X_test, y_train_val, y_test = self._prepare_data(data)

        # Save first 100 rows for reference
        sample = X_train_val.head(100)
        mlflow_wrapper.log_data(sample, "train_sample.csv", index=False)

        # Train with GridSearchCV
        model = self._get_model()
        grid_search = GridSearchCV(model, self.config.model.param_grid, cv=5, scoring="r2")
        grid_search.fit(X_train_val, y_train_val)

        # Log grid search results
        self._log_grid_search_results(grid_search)

        # Get best model
        model = grid_search.best_estimator_

        # Evaluate on test set
        self._evaluate(model, X_test, y_test)

        # Log the best model to MLflow
        mlflow_wrapper.log_model(grid_search.best_estimator_, "best_model")

        return grid_search
