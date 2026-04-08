import logging

import pandas as pd

from geoscore_de.data_flow.feature_engineering.base import BaseFeatureEngineering

logger = logging.getLogger(__name__)


class DeltaFeatureEngineering(BaseFeatureEngineering):
    """Compute the pairwise delta (difference) between two numeric columns.

    delta = newer_column - older_column

    A positive delta indicates an increase; a negative delta indicates a decrease.
    If either column has a missing value for a row, the delta for that row will be NaN.
    The transform adds the output column to the existing dataframe and returns it in full.
    """

    def __init__(self, input_columns: list[str], output_column: str, **kwargs):
        if len(input_columns) != 2:
            raise ValueError(f"DeltaFeatureEngineering requires exactly 2 input columns, got {len(input_columns)}")
        super().__init__(input_columns, output_column)

    def _validate(self, df: pd.DataFrame) -> bool:
        """Validate that both input columns contain numeric data."""
        newer_col, older_col = self.input_columns
        for col in (newer_col, older_col):
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(
                    f"DeltaFeatureEngineering: column '{col}' is not numeric "
                    f"(dtype={df[col].dtype}). Cannot compute delta."
                )
                return False
        return True

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the delta column to the dataframe.

        Args:
            df: Input dataframe containing both input columns.

        Returns:
            DataFrame with the delta column appended.
        """
        newer_col, older_col = self.input_columns
        df[self.output_column] = df[newer_col] - df[older_col]
        return df
