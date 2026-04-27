import logging

import pandas as pd

from geoscore_de.data_flow.feature_engineering.base import BaseFeatureEngineering

logger = logging.getLogger(__name__)


class SumFeatureEngineering(BaseFeatureEngineering):
    """Compute the sum of multiple columns.

    sum = column_1 + column_2 + ... + column_n

    A positive sum indicates an increase; a negative sum indicates a decrease.
    If all input values are missing for a row, the sum for that row will be NaN.
    The transform adds the output column to the existing dataframe and returns it in full.
    """

    def __init__(self, input_columns: list[str], output_column: str, **kwargs):
        if len(input_columns) < 2:
            raise ValueError(f"SumFeatureEngineering requires at least 2 input columns, got {len(input_columns)}")
        super().__init__(input_columns, output_column)

    def _validate(self, df: pd.DataFrame) -> bool:
        """Validate that all input columns contain numeric data."""
        for col in self.input_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(
                    f"SumFeatureEngineering: column '{col}' is not numeric (dtype={df[col].dtype}). Cannot compute sum."
                )
                return False
        return True

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the sum column to the dataframe.

        Args:
            df: Input dataframe containing all input columns.

        Returns:
            DataFrame with the sum column appended.
        """
        # Keep missing values when all source columns are missing in a row.
        df[self.output_column] = df[self.input_columns].sum(axis=1, min_count=1)
        return df
