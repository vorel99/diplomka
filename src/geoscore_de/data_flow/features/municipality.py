import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature

DEFAULT_RAW_DATA_PATH = "data/raw/municipalities_2022.csv"


class MunicipalityFeature(BaseFeature):
    """Load and transform municipality data."""

    def __init__(self, raw_data_path: str = DEFAULT_RAW_DATA_PATH, **kwargs):
        """Initialize the municipality feature.

        Args:
            raw_data_path (str): Path to the CSV file containing municipality data.
        """
        super().__init__(**kwargs)
        self.raw_data_path = raw_data_path

    def load(self) -> pd.DataFrame:
        """Load raw municipality data from CSV and add AGS column."""
        df = pd.read_csv(
            self.raw_data_path,
            skiprows=6,
            sep=";",
            skipfooter=4,
            engine="python",
            header=None,
            names=["MU_ID", "Municipality", "Persons", "Area", "Population Density"],
            dtype={"MU_ID": str},
        )

        # Create AGS column by removing the Verbandsgemeinde (collective municipality) level from MU_ID
        df["AGS"] = df["MU_ID"].str.slice(0, 5) + df["MU_ID"].str.slice(9, 12)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform municipality data:
        - delete unnecessary columns
        - add `federal_state_id` and `admin_region_id` columns derived from `AGS`
        """
        df.drop(columns=["MU_ID", "Municipality"], inplace=True)
        df["federal_state_id"] = df["AGS"].str.slice(0, 2).astype("int16")  # first two digits for federal state level
        df["admin_region_id"] = df["AGS"].str.slice(2, 3).astype("int8")  # one digit for administrative region level
        return df


def load_municipality_data(path: str) -> pd.DataFrame:
    """Load municipality data from a CSV file (legacy function).

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded municipality data.
            DataFrame includes columns `AGS` with 8-character municipality codes.
            `MU_ID`, `Municipality`, `Persons`, `Area`, `Population Density` and `AGS`.
    """
    feature = MunicipalityFeature(path)
    return feature.load()
