import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature

DEFAULT_RAW_DATA_PATH = "data/raw/features/31111-01-01-5-area.csv"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/area.csv"

# Column mapping from raw data (1-indexed in the source, 0-indexed in pandas)
AREA_COLUMNS = {
    "total_land_area": "Total Land Area",  # Bodenfläche Insgesamt (ha)
    "total_settlement_area": "Settlement Area",  # Siedlung Insgesamt (ha)
    "total_traffic_area": "Traffic Area",  # Verkehr Insgesamt (ha)
    "total_vegetation": "Agricultural Vegetation",  # Landwirtschaft Insgesamt (ha)
    "christmas_tree_cultivation": "Christmas Tree Cultivation",  # Weihnachtsbaumkultur (ha)
    "forestry_area": "Forestry Area",  # Forstwirtschaftsfläche Insgesamt (ha)
    "forest_burial_area": "Forest Burial Area",  # Waldbestattungsfläche (ha)
    "forest_area": "Forest Area",  # Wald Insgesamt (ha)
    "shrubland": "Shrubland",  # Gehölz Insgesamt (ha)
    "heathland": "Heathland",  # Heide Insgesamt (ha)
    "moorland": "Moorland",  # Moor Insgesamt (ha)
    "swamp": "Swamp",  # Sumpf Insgesamt (ha)
    "barren_area": "Barren/Vegetation-free Area",  # Unland, Vegetationslose Fläche Insgesamt (ha)
    "total_water_area": "Total Water Area",  # Gewässer Insgesamt (ha)
    "flowing_water": "Flowing Water",  # Fließgewässer Insgesamt (ha)
    "canal": "Canal",  # Kanal Insgesamt (ha)
    "harbor_basin": "Harbor Basin",  # Hafenbecken Insgesamt (ha)
    "standing_water": "Standing Water",  # Stehendes Gewässer Insgesamt (ha)
    "reservoir": "Reservoir",  # Stausee Insgesamt (ha)
    "storage_basin": "Storage Basin",  # Speicherbecken Insgesamt (ha)
    "sea": "Sea",  # Meer Insgesamt (ha)
}


class AreaFeature(BaseFeature):
    """Feature class for land area composition data.
    Data source: https://www.regionalstatistik.de/genesis/online?operation=table&code=31111-01-01-5

    The data includes breakdown of total land area into different categories:
    settlement, traffic, agriculture, forestry, water, and other land uses.
    Provides proportional distribution of each area type relative to total land area.

    Args:
        raw_data_path (str): Path to the CSV file containing raw area data.
        tform_data_path (str): Path to the CSV file containing transformed area data.
    """

    def __init__(
        self,
        raw_data_path: str = DEFAULT_RAW_DATA_PATH,
        tform_data_path: str = DEFAULT_TFORM_DATA_PATH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path

    def load(self) -> pd.DataFrame:
        """Load area data from a CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the loaded area data.
                DataFrame includes columns: `AGS` with 8-character municipality codes,
                and area type columns (settlement, traffic, agriculture, forestry, water, etc.) in hectares.
        """
        df = pd.read_csv(
            self.raw_data_path,
            sep=";",
            encoding="latin1",
            skiprows=5,
            skipfooter=4,
            engine="python",
            na_values=["-", "."],
            dtype=str,
            header=None,
        )

        # Columns 1-3 are identifiers, columns 4-24 contain area data
        # Rename to: AGS, MU_name, and area type columns
        column_names = ["date", "MU_ID", "MU_name"] + list(AREA_COLUMNS.keys())
        df.columns = column_names[: len(df.columns)]

        # Add AGS column by right-padding MU_ID with zeros to 8 characters
        df["AGS"] = df["MU_ID"].str.ljust(8, "0")

        # Convert area columns to numeric (handle missing values)
        for col in AREA_COLUMNS.keys():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw area data by calculating proportions relative to total land area.

        Each area type is normalized as a proportion (0-1) of the total land area.

        Args:
            df (pd.DataFrame): Raw area DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with columns `AGS` and proportional area columns.
        """
        # Create result dataframe with AGS
        result_df = df[["AGS"]].copy()

        # Calculate proportions for each area type
        total_land_area = df["total_land_area"]

        for col_key, col_display_name in AREA_COLUMNS.items():
            if col_key != "total_land_area":  # Skip total land area itself
                result_df[col_key] = df[col_key] / total_land_area

        result_df.to_csv(self.tform_data_path, index=False)
        return result_df
