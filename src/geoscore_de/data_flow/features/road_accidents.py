import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.municipality import DEFAULT_RAW_DATA_PATH as MUNICIPALITY_RAW_DATA_PATH

DEFAULT_RAW_DATA_PATH = "data/raw/features/46241-01-04-5-road-accidents.csv"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/road_accidents.csv"


class RoadAccidentsFeature(BaseFeature):
    """Initialize the road accidents feature.
    Data source: https://www.regionalstatistik.de/genesis//online?operation=table&code=46241-01-04-5


    Args:
        raw_data_path (str): Path to the CSV file containing raw road accident data.
        tform_data_path (str): Path to the CSV file containing transformed road accident data.
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
        self.accident_columns = [
            "accident_count",
            "injury_accidents",
            "property_damage_accidents",
            "fatalities",
            "injured",
        ]

    def load(self) -> pd.DataFrame:
        """Load road accident data from a CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the loaded road accident data.
                Dataframe includes columns: `AGS` with 8-character municipality codes,
                `accident_count` with the total number of accidents,
                `injury_accidents` with the number of accidents that resulted in injuries,
                `property_damage_accidents` with the number of accidents that resulted in property damage,
                `fatalities` with the number of fatalities resulting from road accidents,
                `injured` with the number of people injured in road accidents.
        """
        df = pd.read_csv(
            self.raw_data_path,
            sep=";",
            encoding="latin1",
            skiprows=7,
            skipfooter=4,
            engine="python",
            na_values=["-", "."],
            names=["MU_ID", "MU_name"] + self.accident_columns,
            header=None,
        )

        # add AGS column by right-padding MU_ID with zeros to 8 characters (adds trailing zeros if necessary)
        df["AGS"] = df["MU_ID"].str.ljust(8, "0")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw road accident data by normalizing with municipality population.
        Replaces raw accident counts with per capita values to account for population differences.

        Args:
            df (pd.DataFrame): Raw road accident DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with columns `AGS`, `accident_count`, `fatalities`,
            and `injury_accidents` normalized by population.
        """
        muni_df = pd.read_csv(self.municipality_data_path, sep=";", encoding="latin1")

        # merge muni_df with filtered_df to get Persons column
        merged_df = df.merge(muni_df[["AGS", "Persons"]], on="AGS", how="left")

        # weight all accident columns by Persons (per capita)
        for col in self.accident_columns:
            merged_df[col] = merged_df[col] / merged_df["Persons"]

        return merged_df
