from unittest.mock import patch

import numpy as np
import pandas as pd

from geoscore_de.data_flow.feature_engineering.homogeneity import HomogeneityFeatureEngineering


def test_weighted_cv():
    # Test with simple values: 3 districts with 90%, 100%, 110% voting for a party
    # All districts have equal weight (1000 voters each)
    values = np.array([90, 100, 110])
    weights = np.array([1000, 1000, 1000])

    # Expected calculation:
    # Weighted mean = 100
    # Deviations: -10, 0, 10
    # Variance = ((−10)² + 0² + 10²) / 3 = 200/3 = 66.67
    # Std = sqrt(66.67) = 8.165
    # CV = 8.165 / 100 = 0.0816
    expected_cv = 0.0816

    calculated_cv = HomogeneityFeatureEngineering._weighted_cv(values, weights)

    assert np.isclose(calculated_cv, expected_cv, atol=0.001), f"Expected {expected_cv:.4f}, got {calculated_cv:.4f}"


def test_homogeneity_apply():
    data = {
        "AGS": ["001", "001", "002", "002", "003"],
        "voting_percentage": [90, 100, 80, 120, 100],
        "population": [1000, 1000, 500, 500, 2000],
    }
    df = pd.DataFrame(data)

    homogeneity_engineering = HomogeneityFeatureEngineering(
        input_columns=["voting_percentage"],
        output_column="voting_homogeneity",
        weight_column="population",
    )
    with patch.object(HomogeneityFeatureEngineering, "_weighted_cv", return_value=0.1):
        result_df = homogeneity_engineering._apply(df)

    assert "voting_homogeneity" in result_df.columns, "Output column 'voting_homogeneity' not found in result"
    assert len(result_df) == 3, "Result should have one row per AGS"
    assert result_df["voting_homogeneity"].iloc[0] == 0.1, "Expected homogeneity value of 0.1 for AGS '001'"
    assert result_df["voting_homogeneity"].iloc[1] == 0.1, "Expected homogeneity value of 0.1 for AGS '002'"
    assert np.isnan(result_df["voting_homogeneity"].iloc[2]), (
        "Expected homogeneity value of NaN for AGS '003' (only one district)"
    )
