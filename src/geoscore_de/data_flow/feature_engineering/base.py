import importlib
import logging
from abc import ABCMeta, abstractmethod

import pandas as pd

from geoscore_de.data_flow.feature_engineering.config import FeatureEngineeringConfig

logger = logging.getLogger(__name__)


class BaseFeatureEngineering(metaclass=ABCMeta):
    """Abstract base class for feature engineering."""

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

        if not self._validate(df):
            return False

        return True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the feature engineering transformation to the input dataframe.

        Args:
            df: Input dataframe to transform.

        Returns:
            DataFrame containing one or more engineered feature columns.

        Raises:
            ValueError: If the input dataframe fails validation checks.
        """
        logger.info(f"Applying transformation '{self.__class__.__name__}'")
        if not self.validate(df):
            raise ValueError("Input dataframe failed validation checks.")

        return self._apply(df.copy())

    @abstractmethod
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to implement the actual feature engineering transformation.

        Args:
            df: Input dataframe to transform.

        Returns:
            DataFrame containing one or more engineered feature columns.
        """


def get_feature_engineering_class(config: FeatureEngineeringConfig) -> type[BaseFeatureEngineering]:
    """Dynamically import and return a feature class based on the provided configuration.

    Args:
        config: FeatureEngineeringConfig containing the module and class name to import.

    Returns:
        The imported feature class.
    """
    module_name = config.module
    class_name = config.class_name

    try:
        module = importlib.import_module(module_name)
        feature_class = getattr(module, class_name)
        return feature_class
    except ImportError as e:
        logger.error(f"Error importing module '{module_name}': {e}")
        raise
    except AttributeError as e:
        logger.error(f"Module '{module_name}' does not have a class '{class_name}': {e}")
        raise
