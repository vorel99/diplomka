from abc import ABCMeta, abstractmethod

import pandas as pd


class BaseFeature(metaclass=ABCMeta):
    """Abstract base class for data flow features."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load raw data for the feature."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into feature data."""
        pass

    def load_transform(self) -> pd.DataFrame:
        """Load and transform data in one step."""
        raw_data = self.load()
        transformed_data = self.transform(raw_data)
        return transformed_data
