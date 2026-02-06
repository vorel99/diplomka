import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from geoscore_de.modelling.config import TrainingConfig


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    # TODO: implement method to filter rows based on config
    def _filter_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    # TODO: implement method to filter features based on config
    def _filter_features(self, X):
        return X

    def _prepare_data(self, data: pd.DataFrame):
        """Prepare data for model training by applying row and feature filtering,
        and splitting into train, validation, and test sets.
        """
        # filter rows
        data = self._filter_rows(data)

        # filter features
        X = self._filter_features(data.drop(columns=[self.config.target_variable]))
        y = data[self.config.target_variable]

        # Split into train and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=1 - self.config.train_test_split_ratio, random_state=self.config.random_state
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_val,
            y_train_val,
            test_size=1 - self.config.train_valid_split_ratio,
            random_state=self.config.random_state,
        )
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    # TODO: implement method to get model based on config
    def _get_model(self):
        return LGBMRegressor()

    def _evaluate(self, model, X_test, y_test):
        test_score = model.score(X_test, y_test)
        print(f"Test R^2 Score: {test_score}")
        return test_score

    def train(self, data: pd.DataFrame):
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._prepare_data(data)

        # Train
        model = self._get_model()
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=10, verbose=True)

        # Evaluate on test set
        self._evaluate(model, X_test, y_test)
        return model
