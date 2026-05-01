import importlib
import logging
from abc import abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from geoscore_de.data_flow.feature_engineering.config import FeatureEngineeringConfig

logger = logging.getLogger(__name__)


class BaseFeatureEngineering(BaseEstimator, TransformerMixin):
    """Abstract base class for feature engineering (sklearn-compatible)."""

    requires_join_key: bool = True

    def __init__(self, input_columns: list[str], output_column: str):
        self.input_columns = input_columns
        self.output_column = output_column

    @abstractmethod
    def _validate(self, df: pd.DataFrame) -> bool:
        """Custom validation logic for the feature engineering transformation.

        Args:
            df: Input dataframe to validate.

        Returns:
            True if validation passes, False otherwise.
        """
        return True

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate the input dataframe before applying transformations.
        Check that required input columns are present.

        Args:
            df: Input dataframe to validate.

        Returns:
            True if validation passes, False otherwise.
        """
        missing_columns = [col for col in self.input_columns if col not in df.columns]
        if missing_columns:
            logger.error(
                f"Missing required input columns for transformation '{self.__class__.__name__}': {missing_columns}"
            )
            return False

        if self.requires_join_key and "AGS" not in df.columns:
            logger.error(f"Missing required 'AGS' column for transformation '{self.__class__.__name__}'")
            return False

        if not self._validate(df):
            return False

        return True

    def fit(self, X, y=None):
        """Fit step for stateless transformers (no-op by default).

        Subclasses that are stateful should override this method.

        Args:
            X: Input dataframe or array to fit on.
            y: Target values (unused, for sklearn compatibility).

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input dataframe using the feature engineering transformation.

        Args:
            X: Input dataframe to transform.

        Returns:
            DataFrame containing one or more engineered feature columns.

        Raises:
            ValueError: If the input dataframe fails validation checks.
        """
        logger.info(f"Applying transformation '{self.__class__.__name__}'")
        if not self.validate(X):
            raise ValueError("Input dataframe failed validation checks.")

        return self._apply(X.copy())

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backward compatibility wrapper for transform() method.

        Deprecated: Use transform() instead.

        Args:
            df: Input dataframe to transform.

        Returns:
            Transformed dataframe.
        """
        return self.transform(df)

    @abstractmethod
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to implement the actual feature engineering transformation.

        Args:
            df: Input dataframe to transform.

        Returns:
            DataFrame containing one or more engineered feature columns.
        """


class StatefulFeatureEngineering(BaseFeatureEngineering):
    """Base class for stateful feature engineering transformations that require fitting on training data.

    Subclasses must implement fit() method with sklearn-compatible signature: fit(X, y=None)
    """

    requires_join_key = False

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the transformation on the provided data.

        This method should be called on training data only to avoid leakage.

        Args:
            X: Training DataFrame containing the input columns to fit on.
            y: Target values (unused, for sklearn compatibility).

        Returns:
            self
        """


def instantiate_feature_engineering_class(config: FeatureEngineeringConfig) -> BaseFeatureEngineering:
    """Dynamically import and return a feature class instance based on the provided configuration.

    Args:
        config: FeatureEngineeringConfig containing the module and class name to import.

    Returns:
        An instance of the imported feature class.
    """
    module_name = config.module
    class_name = config.class_name

    try:
        module = importlib.import_module(module_name)
        feature_class = getattr(module, class_name)
        feature_instance = feature_class(
            input_columns=config.input_columns, output_column=config.output_column, **config.params
        )
        return feature_instance
    except ImportError as e:
        logger.error(f"Error importing module '{module_name}': {e}")
        raise
    except AttributeError as e:
        logger.error(f"Module '{module_name}' does not have a class '{class_name}': {e}")
        raise
