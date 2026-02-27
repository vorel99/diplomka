import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from geoscore_de import mlflow_wrapper
from geoscore_de.modelling.config import TrainingConfig
from geoscore_de.modelling.data_filtering import filter_features, filter_rows
from geoscore_de.modelling.training_result import TrainingResult


class Trainer:
    """Model trainer with comprehensive evaluation and hyperparameter logging."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def _filter_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        return filter_rows(data, self.config)

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
        X = filter_features(data.drop(columns=[self.config.target_variable]), self.config)
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

        # Train with GridSearchCV
        model = self._get_model()
        grid_search = GridSearchCV(model, self.config.model.param_grid, cv=5, scoring="r2")
        grid_search.fit(X_train_val, y_train_val)

        result = TrainingResult(
            grid_search=grid_search,
            X_train_val=X_train_val,
            X_test=X_test,
            y_train_val=y_train_val,
            y_test=y_test,
        )

        # Keep previous side effects for existing workflows
        result.log_grid_search_results()
        result.evaluate(create_plots=True)
        result.log_best_model("best_model")

        return result
