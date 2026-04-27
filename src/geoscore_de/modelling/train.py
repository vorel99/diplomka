import warnings
from math import ceil

import pandas as pd
from lightgbm import early_stopping
from sklearn.base import clone
from sklearn.metrics import (
    explained_variance_score,
    make_scorer,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

from geoscore_de import mlflow_wrapper
from geoscore_de.modelling.config import TrainingConfig
from geoscore_de.modelling.data_filtering import filter_features, filter_rows
from geoscore_de.modelling.models import get_model_instance
from geoscore_de.modelling.training_result import TrainingResult


class Trainer:
    """Model trainer with comprehensive evaluation and hyperparameter logging."""

    X_train_: pd.DataFrame | None = None
    y_train_: pd.Series | None = None
    X_val_: pd.DataFrame | None = None
    y_val_: pd.Series | None = None

    def __init__(self, config: TrainingConfig):
        self.config = config

    def _filter_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        return filter_rows(data, self.config)

    def _derive_state_labels(self, data: pd.DataFrame) -> pd.Series:
        """Build state labels used for stratified train/test split."""
        if self.config.federal_state_column in data.columns:
            return data[self.config.federal_state_column].astype(str)

        raise ValueError(
            "State stratification requires either federal_state_column or id_column to be present in input data."
        )

    def _is_stratification_feasible(self, labels: pd.Series, total_samples: int) -> tuple[bool, str]:
        """Check whether sklearn stratified split constraints are satisfiable."""
        class_counts = labels.value_counts(dropna=False)
        if class_counts.empty:
            return False, "No labels available for stratification."

        if class_counts.min() < 2:
            return False, "At least one class has fewer than 2 samples."

        test_size = 1 - self.config.train_test_split_ratio
        n_test = ceil(total_samples * test_size)
        n_train = total_samples - n_test
        n_classes = class_counts.size

        if n_test < n_classes or n_train < n_classes:
            return (
                False,
                f"Split sizes are too small for {n_classes} classes (n_train={n_train}, n_test={n_test}).",
            )

        return True, ""

    def _prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for model training by applying row and feature filtering
        and splitting into train and test sets.

        Args:
            data (pd.DataFrame): The input DataFrame containing features and target variable.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Split data into
            X_train_val, X_test, y_train_val, y_test.
        """
        # filter rows
        data = self._filter_rows(data)

        # remove ID column if it exists
        if self.config.id_column in data.columns:
            data = data.drop(columns=[self.config.id_column])

        # drop rows with missing target variable
        data = data.dropna(subset=[self.config.target_variable])
        # filter features
        X = filter_features(data.drop(columns=[self.config.target_variable]), self.config.feature_filtering)
        y = data[self.config.target_variable]

        state_labels = None
        if self.config.split_strategy == "stratified_federal_state":
            candidate_labels = self._derive_state_labels(data)
            feasible, reason = self._is_stratification_feasible(candidate_labels, total_samples=len(data))
            if feasible:
                state_labels = candidate_labels
            else:
                warnings.warn(
                    "Stratified federal-state split is not feasible for this dataset. "
                    f"Falling back to random split. Reason: {reason}",
                    UserWarning,
                )

        # Split into train and test sets
        split_kwargs = {
            "test_size": 1 - self.config.train_test_split_ratio,
            "random_state": self.config.random_state,
        }
        if state_labels is not None:
            split_kwargs["stratify"] = state_labels

        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, **split_kwargs)
        except ValueError as exc:
            if "stratify" not in split_kwargs:
                raise

            warnings.warn(
                "Stratified federal-state split is not feasible for this dataset. "
                f"Falling back to random split. Original error: {exc}",
                UserWarning,
            )
            split_kwargs.pop("stratify", None)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, **split_kwargs)

        return X_train_val, X_test, y_train_val, y_test

    def _get_model(self):
        """Instantiate the configured regression model with sensible defaults.

        Model type and optional single-value overrides are read from
        ``self.config.model``.  GridSearchCV will apply the ``param_grid``
        values on top of these defaults during hyperparameter tuning.
        """
        return get_model_instance(
            model_type=self.config.model.model_type,
            random_state=self.config.random_state,
        )

    def _get_catboost_fit_params(self, X_train_val: pd.DataFrame) -> dict[str, list[str]]:
        """Return CatBoost-specific fit parameters derived from the training frame."""
        if self.config.model.model_type != "catboost":
            return {}

        cat_features = list(X_train_val.select_dtypes(include=["category", "object", "string"]).columns)
        cat_cols_indices = [i for i, x in enumerate(X_train_val.columns) if x in cat_features]
        if not cat_cols_indices:
            return {}

        return {"cat_features": cat_cols_indices}

    def _build_search(self, model, scoring: dict):
        """Build configured hyperparameter search object."""
        search_config = self.config.search
        search_type = search_config.search_type

        if search_type == "randomized":
            if not search_config.param_grid:
                raise ValueError("Randomized search requires search.param_grid to be defined.")

            return RandomizedSearchCV(
                estimator=model,
                param_distributions=search_config.param_grid,
                n_iter=search_config.n_iter,
                cv=search_config.cv,
                scoring=scoring,
                refit=search_config.refit_metric,
                return_train_score=True,
                random_state=self.config.random_state,
                n_jobs=-1,
            )

        return GridSearchCV(
            estimator=model,
            param_grid=search_config.param_grid or {},
            cv=search_config.cv,
            scoring=scoring,
            refit=search_config.refit_metric,
            return_train_score=True,
            n_jobs=-1,
        )

    def _fit_best_model_with_early_stopping(
        self,
        best_estimator,
        X_train_val: pd.DataFrame,
        y_train_val: pd.Series,
    ):
        """Refit best estimator with early stopping on an internal validation split."""
        early_stopping_config = self.config.early_stopping
        rounds = early_stopping_config.early_stopping_rounds
        if rounds is None:
            return best_estimator

        # TODO: add support for early stopping with other model types if needed
        # like xgboost or catboost or gradient_boosting
        if self.config.model.model_type.lower() not in ["lightgbm", "catboost"]:
            return best_estimator

        X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
            X_train_val,
            y_train_val,
            test_size=early_stopping_config.early_stopping_validation_fraction,
            random_state=self.config.random_state,
        )

        self.X_train_ = X_train_fit
        self.y_train_ = y_train_fit
        self.X_val_ = X_val_fit
        self.y_val_ = y_val_fit

        estimator = clone(best_estimator)

        if self.config.model.model_type.lower() == "catboost":
            estimator.fit(
                X_train_fit,
                y_train_fit,
                eval_set=(X_val_fit, y_val_fit),
                early_stopping_rounds=rounds,
                use_best_model=True,
                verbose=False,
                cat_features=self._get_catboost_fit_params(X_train_val).get("cat_features", []),
            )
        elif self.config.model.model_type.lower() == "lightgbm":
            estimator.fit(
                X_train_fit,
                y_train_fit,
                eval_set=[(X_val_fit, y_val_fit)],
                eval_metric="l2",
                callbacks=[early_stopping(stopping_rounds=rounds, verbose=False)],
            )
        return estimator

    def train(self, data: pd.DataFrame) -> TrainingResult:
        """Train the model using the provided data and configuration.

        Args:
            data (pd.DataFrame): The input DataFrame containing features and target variable.

        Returns:
            TrainingResult: Object with fitted model, splits, and post-training methods.
        """
        # Log training configuration to MLflow for reproducibility
        config_dict = self.config.model_dump()
        mlflow_wrapper.log_dict(config_dict, "config.json")

        X_train_val, X_test, y_train_val, y_test = self._prepare_data(data)

        # Save first 100 rows for reference
        sample = X_train_val.head(100)
        mlflow_wrapper.log_data(sample, "train_sample.csv", index=False)

        # Train with configured hyperparameter search
        model = self._get_model()
        fit_params = self._get_catboost_fit_params(X_train_val)
        scoring = {
            "r2": "r2",
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "mse": make_scorer(mean_squared_error, greater_is_better=False),
            "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
            "mape": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
            "median_ae": make_scorer(median_absolute_error, greater_is_better=False),
            "max_error": make_scorer(max_error, greater_is_better=False),
            "explained_variance": make_scorer(explained_variance_score),
        }
        grid_search = self._build_search(model, scoring)
        grid_search.fit(X_train_val, y_train_val, **fit_params)

        best_estimator = grid_search.best_estimator_
        try:
            best_estimator = self._fit_best_model_with_early_stopping(best_estimator, X_train_val, y_train_val)
        except Exception as exc:
            warnings.warn(
                f"Early stopping refit failed and will be skipped. Reason: {exc}",
                UserWarning,
            )

        result = TrainingResult(
            grid_search=grid_search,
            X_train=self.X_train_ if self.X_train_ is not None else X_train_val,
            X_val=self.X_val_,
            X_test=X_test,
            y_train=self.y_train_ if self.y_train_ is not None else y_train_val,
            y_val=self.y_val_,
            y_test=y_test,
            test_indices=X_test.index,
            best_estimator_override=best_estimator,
        )

        result.log_grid_search_results()
        result.log_best_model("best_model")

        return result
