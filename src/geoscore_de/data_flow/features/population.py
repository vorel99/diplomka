import logging

import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature

logger = logging.getLogger(__name__)

DEFAULT_RAW_DATA_PATH = "data/raw/features/population.csv"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/population.csv"


class PopulationFeature(BaseFeature):
    """Feature class for population data."""

    def __init__(
        self, raw_data_path: str = DEFAULT_RAW_DATA_PATH, tform_data_path: str = DEFAULT_TFORM_DATA_PATH, **kwargs
    ):
        super().__init__(**kwargs)
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path

    def load(self) -> pd.DataFrame:
        """Load population data from a CSV file.
        Data were obtained from https://www.regionalstatistik.de/genesis//online?operation=table&code=12411-02-03-5&bypass=true&levelindex=1&levelid=1765292926381#abreadcrumb

        Returns:
            pd.DataFrame: DataFrame containing the loaded population data.
                Dataframe includes columns: `AGS` with 8-character municipality codes,
                `age_group` with age group descriptions, `people_count`, `male_count`, `female_count`.
        """
        df = pd.read_csv(
            self.raw_data_path,
            sep=";",
            encoding="latin1",
            skiprows=6,
            skipfooter=4,
            engine="python",
            na_values=["-", "."],
            names=["date", "MU_ID", "Municipality", "age_group", "people_count", "male_count", "female_count"],
            dtype={"MU_ID": str},
            header=None,
        )

        # add AGS column by right-padding MU_ID with zeros to 8 characters (adds trailing zeros if necessary)
        df["AGS"] = df["MU_ID"].str.ljust(8, "0")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw population data into a pivoted format with age groups as columns.
        - rename age group columns from German to English
        - convert absolute counts to proportions of the total population. (except for the total population column)

        Args:
            df (pd.DataFrame): Raw population DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with age groups as columns.
        """
        # Pivot the data to have age groups as columns
        tform_df = df.pivot_table(
            index=["AGS"],
            columns="age_group",
            values="people_count",
            aggfunc="sum",
        ).reset_index()

        # rename german age group columns to english
        age_group_rename_map = {
            "unter 3 Jahre": "age_under_3",
            "3 bis unter 6 Jahre": "age_3_to_5",
            "6 bis unter 10 Jahre": "age_6_to_9",
            "10 bis unter 15 Jahre": "age_10_to_14",
            "15 bis unter 18 Jahre": "age_15_to_17",
            "18 bis unter 20 Jahre": "age_18_to_19",
            "20 bis unter 25 Jahre": "age_20_to_24",
            "25 bis unter 30 Jahre": "age_25_to_29",
            "30 bis unter 35 Jahre": "age_30_to_34",
            "35 bis unter 40 Jahre": "age_35_to_39",
            "40 bis unter 45 Jahre": "age_40_to_44",
            "45 bis unter 50 Jahre": "age_45_to_49",
            "50 bis unter 55 Jahre": "age_50_to_54",
            "55 bis unter 60 Jahre": "age_55_to_59",
            "60 bis unter 65 Jahre": "age_60_to_64",
            "65 bis unter 75 Jahre": "age_65_to_74",
            "75 Jahre und mehr": "age_75_and_over",
            "Insgesamt": "total_population",
        }
        tform_df.rename(columns=age_group_rename_map, inplace=True)

        # Change all columns from absolute counts to proportions of the total population
        # Replace 0 with NaN to avoid division by zero (inf values)
        total_pop = tform_df["total_population"].replace(0, pd.NA)

        for col in tform_df.columns:
            if col != "AGS" and col != "total_population":
                tform_df[col] = tform_df[col] / total_pop

        tform_df.to_csv(self.tform_data_path, index=False)
        return tform_df
