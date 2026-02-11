import logging
from abc import ABCMeta, abstractmethod

import pandas as pd

from geoscore_de.data_flow.features.config import FeatureEngineeringConfig

logger = logging.getLogger(__name__)


class BaseFeatureEngineering(metaclass=ABCMeta):
    """Abstract base class for feature engineering."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    @property
    def output_column(self) -> str:
        """Get the name of the output column produced by this transformation."""
        return self.config.output_column

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate the input dataframe before applying transformations.
        Check that required input columns are present.

        Args:
            df: Input dataframe to validate.

        Returns:
            True if validation passes, False otherwise.
        """
        missing_columns = [col for col in self.config.input_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required input columns for transformation '{self.config.name}': {missing_columns}")
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
        logger.info(f"Applying transformation '{self.config.name}'")
        if not self.validate(df):
            raise ValueError("Input dataframe failed validation checks.")

        return self._apply(df)

    @abstractmethod
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to implement the actual feature engineering transformation.

        Args:
            df: Input dataframe to transform.

        Returns:
            DataFrame containing one or more engineered feature columns.
        """
