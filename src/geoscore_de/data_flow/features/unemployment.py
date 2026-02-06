import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature

DEFAULT_RAW_DATA_PATH = "data/raw/features/unemployment.csv"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/unemployment.csv"


class UnemploymentFeature(BaseFeature):
    """Load and transform unemployment data from GENESIS-Online."""

    def __init__(self, raw_data_path: str = DEFAULT_RAW_DATA_PATH, tform_data_path: str = DEFAULT_TFORM_DATA_PATH):
        """Initialize the unemployment feature.

        Args:
            raw_data_path (str): Path to the CSV file containing raw unemployment data.
            tform_data_path (str): Path to the CSV file containing transformed unemployment data.
                Data source: https://www.regionalstatistik.de/genesis//online?operation=table&code=13211-01-03-5&bypass=true&levelindex=1&levelid=1768376127943#abreadcrumb
        """
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path

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
        )
        df.rename(
            columns={"Unnamed: 0": "MU_ID", "Unnamed: 1": "Municipality", "Unnamed: 2": "unemployment_total"},
            inplace=True,
        )

        # drop first row which contains column descriptions
        df = df.iloc[1:].reset_index(drop=True)

        df["MU_ID"] = df["MU_ID"].astype(str)
        df["unemployment_total"] = pd.to_numeric(df["unemployment_total"])

        # fill MU_ID on the right with zeros to a total length of 8 characters
        df["AGS"] = df["MU_ID"].str.ljust(8, "0")

        return df

    # TODO: implement any transformations if needed (e.g., normalization by population)
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw unemployment data."""

        # drop MU_ID and Municipality columns
        df = df.drop(columns=["MU_ID", "Municipality"])
        return df


def load_unemployment_data(path: str) -> pd.DataFrame:
    """Load and transform unemployment data from a CSV file (legacy function).

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded unemployment data.
    """
    feature = UnemploymentFeature(path)
    return feature.load_transform()
