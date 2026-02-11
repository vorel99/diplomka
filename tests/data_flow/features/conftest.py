"""Shared test fixtures and utilities for feature tests."""

import pandas as pd
import pytest

from geoscore_de.data_flow.feature_engineering.base import BaseFeatureEngineering
from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.config import FeatureEngineeringConfig


class MockFeatureEngineering(BaseFeatureEngineering):
    """Mock feature engineering transformation for testing."""

    def __init__(self, input_columns: list[str], output_column: str, **kwargs):
        super().__init__(input_columns, output_column)

    def _validate(self, df: pd.DataFrame) -> bool:
        """Validate that the required input columns are present in the dataframe."""
        return True

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a simple transformation that sums the specified input columns."""
        df[self.output_column] = df[self.input_columns].sum(axis=1)
        return df[["AGS", self.output_column]]


@pytest.fixture
def mock_feature_engineering_class() -> type[BaseFeatureEngineering]:
    """Provide MockFeatureEngineering class for tests."""
    return MockFeatureEngineering


@pytest.fixture
def feature_engineering_config(mock_feature_module) -> FeatureEngineeringConfig:
    """Provide a feature engineering configuration for tests."""
    return FeatureEngineeringConfig(
        name="mock_feature_engineering",
        class_name="MockFeatureEngineering",
        module=mock_feature_module,
        input_columns=["value", "weight"],
        output_column="transformed_value",
    )


class MockFeature(BaseFeature):
    """Mock feature class for testing."""

    def __init__(self, data: pd.DataFrame | None = None, **kwargs):
        super().__init__(**kwargs)
        self.data = (
            data
            if data is not None
            else pd.DataFrame({"AGS": [1, 2, 3], "value": [10, 20, 30], "weight": [100, 200, 300]})
        )

    def load(self) -> pd.DataFrame:
        """Return mock data."""
        return self.data.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return data as-is."""
        return df


@pytest.fixture
def mock_feature_class() -> type[BaseFeature]:
    """Provide MockFeature class for tests."""
    return MockFeature


@pytest.fixture
def mock_feature_module():
    """Provide the module path where MockFeature lives."""
    return __name__
