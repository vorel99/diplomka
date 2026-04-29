import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.municipality import DEFAULT_RAW_DATA_PATH as MUNICIPALITY_RAW_DATA_PATH
from geoscore_de.data_flow.features.municipality import MunicipalityFeature


class BaseElectionFeature(BaseFeature):
    def __init__(self, municipality_data_path: str = MUNICIPALITY_RAW_DATA_PATH, **kwargs):
        super().__init__(**kwargs)
        self.municipality_data_path = municipality_data_path

    def _fix_missing(self, df: pd.DataFrame, muni_name_col: str = "Gemeindename") -> pd.DataFrame:
        """Municipalities that are missing in the election data but present in the municipality data
        are filled using einschl. keyword in the municipality name. If a municipality is missing
        but is included in another municipality's data add the missing municipality with the same values
        as the including municipality.

        Args:
            df (pd.DataFrame): DataFrame containing the election data.
            muni_name_col (str): Name of the column containing the municipality name.

        Returns:
            pd.DataFrame: DataFrame with added rows for missing municipalities.
        """
        municipality_feature = MunicipalityFeature(self.municipality_data_path)
        df_muni = municipality_feature.load()[["AGS", "Municipality"]]

        df_merged = df.merge(df_muni, on="AGS", how="outer", indicator=True)

        missing_munis = df_merged[df_merged["_merge"] == "right_only"]
        einschl_munis = df_merged[df_merged[muni_name_col].str.contains("einschl.", na=False)]
        rows_to_add: list[pd.DataFrame] = []

        # iterate over missing municipalities and check if their name is included in any of the einschl. municipalities
        for _, missing_row in missing_munis.iterrows():
            missing_name = missing_row["Municipality"]
            # Find first matching einschl. municipality
            for _, einschl_row in einschl_munis.iterrows():
                einschl_name = einschl_row[muni_name_col]
                if missing_name in einschl_name:
                    # If the missing municipality is included in the einschl. municipality
                    # add row to original df with the same values as the einschl. municipality
                    # except for AGS
                    new_row = df[df["AGS"] == einschl_row["AGS"]].copy()
                    new_row["AGS"] = missing_row["AGS"]
                    rows_to_add.append(new_row)
                    break

        df = pd.concat([df, *rows_to_add], ignore_index=True)
        return df
