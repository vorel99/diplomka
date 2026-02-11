import numpy as np

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
