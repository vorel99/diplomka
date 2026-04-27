from pathlib import Path

import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.municipality import (
    DEFAULT_RAW_DATA_PATH as MUNICIPALITY_RAW_DATA_PATH,
)
from geoscore_de.data_flow.features.municipality import (
    MunicipalityFeature,
)

DEFAULT_RAW_DATA_PATH = "data/raw/features/12711-91-01-5-migration.csv"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/migration.csv"


class MigrationFeature(BaseFeature):
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
        """Load migration data from a CSV file.
        Data obtained from the GENESIS-Online regional statistics database of the German Federal Statistical Office.
        https://www.regionalstatistik.de/genesis//online?operation=table&code=12711-91-01-5&bypass=true&levelindex=0&levelid=1768376496916#abreadcrumb

        Returns:
            pd.DataFrame: DataFrame containing the loaded migration data.
                DataFrame includes columns: `AGS` with 8-character municipality codes,
                `in_migration` with the number of in-migrants, and `out_migration` with the number of out-migrants.
        """
        df = pd.read_csv(
            self.raw_data_path,
            sep=";",
            encoding="latin1",
            skiprows=6,
            skipfooter=4,
            engine="python",
            na_values=["-", "."],
            names=["Year", "MU_ID", "Municipality", "in_migration", "out_migration"],
            header=None,
            dtype={"MU_ID": str},
        )

        df["MU_ID"] = df["MU_ID"].astype(str)

        # fill MU_ID on the right with zeros to a total length of 8 characters
        df["AGS"] = df["MU_ID"].str.ljust(8, "0")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw migration data into a wide format with year-suffixed migration columns.

        Args:
            df (pd.DataFrame): Raw migration DataFrame.
        Returns:
            pd.DataFrame: One row per `AGS` with columns for each year, e.g.
                `in_migration_2022`, `out_migration_2022`, `net_migration_2022`.
        """
        df = df.copy()

        # Ensure arithmetic and column naming work consistently across years.
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["in_migration"] = pd.to_numeric(df["in_migration"], errors="coerce")
        df["out_migration"] = pd.to_numeric(df["out_migration"], errors="coerce")

        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)
        df["net_migration"] = df["in_migration"] - df["out_migration"]

        # Pivot each metric to one column per year and flatten the MultiIndex columns.
        pivoted = df.pivot_table(
            index="AGS",
            columns="Year",
            values=["in_migration", "out_migration", "net_migration"],
            aggfunc="first",
        )

        pivoted.columns = [f"{metric}_{year}" for metric, year in pivoted.columns]
        transformed_df = pivoted.reset_index()

        # Load municipality data to get population (Persons column) for normalization
        municipality_feature = MunicipalityFeature(self.municipality_data_path)
        municipality_df = municipality_feature.load()[["AGS", "Persons"]]

        # merge muni_df with filtered_df to get Persons column
        merged_df = transformed_df.merge(municipality_df, on="AGS", how="left")

        # Normalize migration by population
        for col in merged_df.columns:
            if col.startswith(("in_migration_", "out_migration_", "net_migration_")):
                merged_df[col] = (merged_df[col] / merged_df["Persons"]).round(6)

        output_path = Path(self.tform_data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merged_df = merged_df.drop(columns=["Persons"])
        merged_df.to_csv(output_path, index=False)

        return merged_df
