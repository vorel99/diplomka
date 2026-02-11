import numpy as np
import pandas as pd

from geoscore_de.data_flow.feature_engineering.base import BaseFeatureEngineering


class HomogeneityFeatureEngineering(BaseFeatureEngineering):
    """Feature engineering transformation to calculate the homogeneity of a feature.
    This transformation calculates the weighted coefficient of variation (CV) for specified input columns,
    grouped by AGS. The CV is a measure of relative variability and can be used
    to assess the homogeneity of a feature across municipalities within a district.
    The resulting CV values are merged back into the main feature dataframe as new columns specified in output_column.
    """

    def __init__(self, input_columns: list[str], output_column: str, weight_column: str):
        super().__init__(input_columns, output_column)
        self.weight_column = weight_column

    @classmethod
    def _weighted_cv(cls, values, weights):
        """Calculate weighted coefficient of variation.

        Args:
            values: Array of feature values for a group of municipalities.
            weights: Array of weights corresponding to the values (e.g., population).
        """
        weighted_mean = np.average(values, weights=weights)
        if weighted_mean == 0:
            return np.nan
        weighted_var = np.average((values - weighted_mean) ** 2, weights=weights)
        weighted_std = np.sqrt(weighted_var)
        return weighted_std / weighted_mean

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted Coefficient of Variation (CV) as a measure of homogeneity for the specified input columns.

        Args:
            df: Input dataframe containing multiple municipalities and the feature columns to calculate homogeneity for.

        Returns:
            DataFrame containing AGS code and the calculated homogeneity feature column.
        """

        results = []

        for ags, group in df.groupby("AGS"):
            if len(group) < 2:  # Need at least 2 districts to measure variance
                continue

            metrics = {"AGS": ags}
            weights = group[self.weight_column].values

            # Calculate CV for each input column
            cvs = []
            for col in self.input_columns:
                values = group[col].values
                if np.sum(weights) > 0 and not np.all(values == 0):
                    cv = self._weighted_cv(values, weights)
                    if not np.isnan(cv):
                        cvs.append(cv)

            # For each output column, calculate the mean CV across input columns
            if cvs:
                metrics[self.output_column] = np.mean(cvs)
            else:
                metrics[self.output_column] = np.nan

            results.append(metrics)

        return pd.DataFrame(results)
