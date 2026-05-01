"""Train-fitted binning using scikit-learn's KBinsDiscretizer.

This transformer is designed for use in the training pipeline to avoid data leakage.
It fits bin boundaries on training data only and applies them to test data.
"""

import logging

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from geoscore_de.data_flow.feature_engineering.base import StatefulFeatureEngineering

logger = logging.getLogger(__name__)


class KBinsDiscretizerBinning(StatefulFeatureEngineering):
    """Bin numeric columns using scikit-learn's KBinsDiscretizer with train-fitted bin edges.

    Supports two strategies:
    - 'quantile': Equal-frequency bins, fitted on training data.
    - 'uniform': Equal-width bins, fitted on training data.

    This transformer must be fitted on training data only to avoid leakage. It stores the fitted
    discretizer state and applies it to test data.

    Output column contains ordinal bin labels in the range 0..n_bins-1, stored as floats
    (for example, 0.0, 1.0, ..., n_bins-1.0) so missing values can be preserved as NaN.
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
            raise ValueError(f"KBinsDiscretizerBinning requires exactly 1 input column, got {len(input_columns)}")
        if strategy not in ("quantile", "uniform"):
            raise ValueError(f"Invalid strategy '{strategy}'. Must be 'quantile' or 'uniform'.")
        if n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {n_bins}")

        super().__init__(input_columns, output_column)
        self.strategy = strategy
        self.n_bins = n_bins
        self.discretizer = None
        self._constant_output = False

    def _validate(self, df: pd.DataFrame) -> bool:
        """Validate that the input column is numeric."""
        col = self.input_columns[0]
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"KBinsDiscretizerBinning: column '{col}' is not numeric (dtype={df[col].dtype}). Cannot bin.")
            return False
        return True

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the discretizer on the provided data.

        This method should be called on training data only to avoid leakage.

        Args:
            df: Training DataFrame containing the input column to fit on.

        Raises:
            ValueError: If the input column is not numeric or missing.
        """
        if not self.validate(df):
            raise ValueError("Input dataframe failed validation checks.")

        col = self.input_columns[0]
        col_data = df[[col]].dropna()

        if col_data.empty or col_data[col].nunique(dropna=True) < 2:
            self.discretizer = None
            self._constant_output = True
            logger.info(f"Skipping KBins fit for column '{col}' because there are not enough distinct values")
            return

        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, encode="ordinal", strategy=self.strategy, subsample=None
        )
        self.discretizer.fit(col_data)
        self._constant_output = False
        logger.info(f"Fitted KBinsDiscretizerBinning on column '{col}' with strategy '{self.strategy}'")

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted discretizer to the dataframe.

        Args:
            df: Input dataframe containing the input column.

        Returns:
            DataFrame with the binned column appended.

        Raises:
            ValueError: If the discretizer has not been fitted.
        """
        if self.discretizer is None:
            if self._constant_output:
                df[self.output_column] = pd.Series(index=df.index, dtype="float64")
                return df
            raise ValueError("Discretizer has not been fitted. Call fit() on training data first.")

        col = self.input_columns[0]
        col_data = df[[col]]
        output = pd.Series(index=df.index, dtype="float64")

        non_null_mask = col_data[col].notna()
        if non_null_mask.any():
            transformed = self.discretizer.transform(col_data.loc[non_null_mask])
            output.loc[non_null_mask] = transformed[:, 0]

        df[self.output_column] = output

        return df
