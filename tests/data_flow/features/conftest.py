"""Shared test fixtures and utilities for feature tests."""

import pandas as pd
import pytest

from geoscore_de.data_flow.features.base import BaseFeature


class MockFeature(BaseFeature):
    """Mock feature class for testing."""

    def __init__(self, data: pd.DataFrame | None = None, **kwargs):
        super().__init__(**kwargs)
        self.data = data if data is not None else pd.DataFrame({"AGS": [1, 2, 3], "value": [10, 20, 30]})

    def load(self) -> pd.DataFrame:
        """Return mock data."""
        return self.data.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return data as-is."""
        return df


@pytest.fixture
def mock_feature_class():
    """Provide MockFeature class for tests."""
    return MockFeature
