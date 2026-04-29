import logging

import pandas as pd

from geoscore_de.data_flow.feature_engineering.base import BaseFeatureEngineering

logger = logging.getLogger(__name__)


class BinningFeatureEngineering(BaseFeatureEngineering):
    """Bin a numeric column into ordered categories using numeric labels.

    Supports two binning strategies:
    - 'quantile': Equal-frequency bins (pd.qcut).
    - 'equal_width': Equal-width bins (pd.cut).

    Output column contains integer labels 0, 1, 2, ..., n_bins-1.
    Missing values in the input are preserved as NaN in the output.
    """

    def __init__(
        self,
        input_columns: list[str],
        output_column: str,
        strategy: str = "quantile",
        n_bins: int = 10,
        **kwargs,
    ):
        if len(input_columns) != 1:
            raise ValueError(f"BinningFeatureEngineering requires exactly 1 input column, got {len(input_columns)}")
        if strategy not in ("quantile", "equal_width"):
            raise ValueError(f"Invalid strategy '{strategy}'. Must be 'quantile' or 'equal_width'.")
        if n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {n_bins}")
        super().__init__(input_columns, output_column)
        self.strategy = strategy
        self.n_bins = n_bins

    def _validate(self, df: pd.DataFrame) -> bool:
        """Validate that the input column is numeric."""
        col = self.input_columns[0]
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(
                f"BinningFeatureEngineering: column '{col}' is not numeric (dtype={df[col].dtype}). Cannot bin."
            )
            return False
        return True

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the binned column to the dataframe.

        Args:
            df: Input dataframe containing the input column.

        Returns:
            DataFrame with the binned column appended.
        """
        col = self.input_columns[0]
        col_data = df[col]

        if self.strategy == "quantile":
            df[self.output_column] = pd.qcut(col_data, q=self.n_bins, labels=False, duplicates="drop")
        else:  # equal_width
            df[self.output_column] = pd.cut(col_data, bins=self.n_bins, labels=False)

        return df
