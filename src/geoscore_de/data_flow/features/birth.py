import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.municipality import DEFAULT_RAW_DATA_PATH as MUNICIPALITY_RAW_DATA_PATH
from geoscore_de.data_flow.features.municipality import MunicipalityFeature

DEFAULT_RAW_DATA_PATH = "data/raw/features/12612-91-01-5-births.csv"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/births.csv"


class BirthsFeature(BaseFeature):
    """Initialize the birth feature.
    Data source: https://www.regionalstatistik.de/genesis//online?operation=table&code=12612-91-01-5

    Args:
        raw_data_path (str): Path to the CSV file containing raw birth data.
        tform_data_path (str): Path to the CSV file containing transformed birth data.
        municipality_data_path (str): Path to the CSV file containing municipality data for normalization.
    """

    def __init__(
        self,
        raw_data_path: str = DEFAULT_RAW_DATA_PATH,
        tform_data_path: str = DEFAULT_TFORM_DATA_PATH,
        municipality_data_path: str = MUNICIPALITY_RAW_DATA_PATH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path
        self.municipality_data_path = municipality_data_path

    def load(self) -> pd.DataFrame:
        """Load birth data from a CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the loaded birth data.
                Dataframe includes columns: `AGS` with 8-character municipality codes,
                `births` with the total number of births in the municipality.
        """
        df = pd.read_csv(
            self.raw_data_path,
            sep=";",
            encoding="latin1",
            skiprows=5,
            skipfooter=4,
            engine="python",
            na_values=["-", "."],
            names=["MU_ID", "MU_name", "births"],
            dtype={"MU_ID": str},
            header=None,
        )

        # add AGS column by right-padding MU_ID with zeros to 8 characters (adds trailing zeros if necessary)
        df["AGS"] = df["MU_ID"].str.ljust(8, "0")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw birth data by normalizing with municipality population.

        Args:
            df (pd.DataFrame): Raw birth DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with columns `AGS` and `births` normalized by population.
        """
        municipality_feature = MunicipalityFeature(self.municipality_data_path)
        municipality_df = municipality_feature.load()[["AGS", "Persons"]]

        # merge muni_df with filtered_df to get Persons column
        merged_df = df.merge(municipality_df[["AGS", "Persons"]], on="AGS", how="left")

        # weight all accident columns by Persons (per capita)
        merged_df["births"] = merged_df["births"] / merged_df["Persons"]

        return merged_df[["AGS", "births"]]
