import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature

DEFAULT_RAW_DATA_PATH = "data/raw/features/12711-91-01-5-migration.csv"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/migration.csv"


class MigrationFeature(BaseFeature):
    def __init__(
        self, raw_data_path: str = DEFAULT_RAW_DATA_PATH, tform_data_path: str = DEFAULT_TFORM_DATA_PATH, **kwargs
    ):
        super().__init__(**kwargs)
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path

    def load(self) -> pd.DataFrame:
        """Load migration data from a CSV file.

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
        df_tform = df.copy()

        # Ensure arithmetic and column naming work consistently across years.
        df_tform["Year"] = pd.to_numeric(df_tform["Year"], errors="coerce").astype("Int64")
        df_tform["in_migration"] = pd.to_numeric(df_tform["in_migration"], errors="coerce")
        df_tform["out_migration"] = pd.to_numeric(df_tform["out_migration"], errors="coerce")

        df_tform = df_tform.dropna(subset=["Year"])
        df_tform["Year"] = df_tform["Year"].astype(int)
        df_tform["net_migration"] = df_tform["in_migration"] - df_tform["out_migration"]

        # Pivot each metric to one column per year and flatten the MultiIndex columns.
        pivoted = df_tform.pivot_table(
            index="AGS",
            columns="Year",
            values=["in_migration", "out_migration", "net_migration"],
            aggfunc="first",
        )

        pivoted.columns = [f"{metric}_{year}" for metric, year in pivoted.columns]
        return pivoted.reset_index()


def load_migration_data(path: str) -> pd.DataFrame:
    """Load migration data from a CSV file.
    Data were obtained from the GENESIS-Online regional statistics database of the German Federal Statistical Office.
    https://www.regionalstatistik.de/genesis//online?operation=table&code=12711-91-01-5&bypass=true&levelindex=0&levelid=1768376496916#abreadcrumb

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded migration data.
    """
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        skiprows=6,
        skipfooter=4,
        engine="python",
        na_values=["-", "."],
        names=["Year", "MU_ID", "Municipality", "in_migration", "out_migration"],
        header=None,
    )

    df["MU_ID"] = df["MU_ID"].astype(str)
    df["in_migration"] = pd.to_numeric(df["in_migration"])
    df["out_migration"] = pd.to_numeric(df["out_migration"])

    # fill MU_ID on the right with zeros to a total length of 8 characters
    df["AGS"] = df["MU_ID"].str.ljust(8, "0")

    return df
