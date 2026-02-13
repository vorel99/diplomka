import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.municipality import DEFAULT_RAW_DATA_PATH as MUNICIPALITY_RAW_DATA_PATH
from geoscore_de.data_flow.features.municipality import MunicipalityFeature

DEFAULT_RAW_DATA_PATH = "data/raw/features/unemployment.csv"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/unemployment.csv"


class UnemploymentFeature(BaseFeature):
    """Load and transform unemployment data from GENESIS-Online."""

    def __init__(
        self,
        raw_data_path: str = DEFAULT_RAW_DATA_PATH,
        tform_data_path: str = DEFAULT_TFORM_DATA_PATH,
        municipality_data_path: str = MUNICIPALITY_RAW_DATA_PATH,
        **kwargs,
    ):
        """Initialize the unemployment feature.

        Args:
            raw_data_path (str): Path to the CSV file containing raw unemployment data.
            tform_data_path (str): Path to the CSV file containing transformed unemployment data.
                Data source: https://www.regionalstatistik.de/genesis//online?operation=table&code=13211-01-03-5&bypass=true&levelindex=1&levelid=1768376127943#abreadcrumb
            municipality_data_path (str): Path to the CSV file containing municipality data for normalization.
        """
        super().__init__(**kwargs)
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path
        self.municipality_data_path = municipality_data_path

    def load(self) -> pd.DataFrame:
        """Load raw unemployment data from CSV."""
        df = pd.read_csv(
            self.raw_data_path,
            sep=";",
            encoding="latin1",
            skiprows=7,
            skipfooter=4,
            engine="python",
            na_values=["-", "."],
            dtype={"MU_ID": str},
        )
        df.rename(
            columns={"Unnamed: 0": "MU_ID", "Unnamed: 1": "Municipality", "Unnamed: 2": "unemployment_total"},
            inplace=True,
        )

        # drop first row which contains column descriptions
        df = df.iloc[1:].reset_index(drop=True)

        df["unemployment_total"] = pd.to_numeric(df["unemployment_total"])

        # fill MU_ID on the right with zeros to a total length of 8 characters
        df["AGS"] = df["MU_ID"].str.ljust(8, "0")

        return df

    # TODO: implement any transformations if needed (e.g., normalization by population)
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw unemployment data and normalize by municipality population."""

        # Load municipality data to get population (Persons column) for normalization
        municipality_feature = MunicipalityFeature(self.municipality_data_path)
        municipality_df = municipality_feature.load()[["AGS", "Persons"]]

        # Convert Persons to numeric (it may be a string)
        municipality_df["Persons"] = pd.to_numeric(municipality_df["Persons"], errors="coerce")

        # Merge with municipality data
        df = df.merge(municipality_df, on="AGS", how="left")

        # Normalize unemployment by population
        df["unemployment_per_capita"] = (df["unemployment_total"] / df["Persons"]).round(6)

        # Drop temporary columns
        df = df.drop(columns=["MU_ID", "Municipality", "Persons"])

        df.to_csv(self.tform_data_path, index=False)

        return df


def load_unemployment_data(path: str) -> pd.DataFrame:
    """Load and transform unemployment data from a CSV file (legacy function).

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded unemployment data.
    """
    feature = UnemploymentFeature(path)
    return feature.load()
