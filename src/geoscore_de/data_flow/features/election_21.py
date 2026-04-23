import shutil
from pathlib import Path

import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.utils import load_election_zip, move_extracted_file

ZIP_URL = "https://www.bundeswahlleiterin.de/en/dam/jcr/c2cd99e6-064e-4ebc-b634-f86b5c0e14b3/btw21_wbz.zip"

DEFAULT_RAW_DATA_PATH = "data/raw/features/election_2021"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/federal_election_21.csv"
HAMBURG_CITY_AGS = "02000000"
BERLIN_CITY_AGS = "11000000"


class Election21Feature(BaseFeature):
    """Feature class for election 2021 data."""

    def __init__(
        self,
        url: str = ZIP_URL,
        raw_data_path: str = DEFAULT_RAW_DATA_PATH,
        tform_data_path: str = DEFAULT_TFORM_DATA_PATH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.url = url
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path

    def load(self) -> pd.DataFrame:
        """Load and extract election 21 data from a ZIP file.

        This method:
        - adds an AGS column by concatenating the Land, Regierungsbezirk, Kreis, and Gemeinde columns
        - renames key columns to normalized English identifiers

        Returns:
            pd.DataFrame: DataFrame containing the election data.
        """
        temp_dir = load_election_zip(self.url)
        try:
            # Ensure destination directory exists
            Path(self.raw_data_path).mkdir(parents=True, exist_ok=True)
            move_extracted_file(temp_dir, self.raw_data_path)
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Load the election data CSV into a DataFrame
        election_csv_path = Path(self.raw_data_path) / "btw21_wbz_ergebnisse.csv"
        dtypes = {
            "Land": str,
            "Regierungsbezirk": str,
            "Kreis": str,
            "Gemeinde": str,
        }
        df = pd.read_csv(election_csv_path, sep=";", dtype=dtypes)

        # create AGS column by concatenating Land, Regierungsbezirk, Kreis, and Gemeinde columns
        # Convert to string first to handle potential numeric types
        df["AGS"] = (
            df["Land"].astype(str)
            + df["Regierungsbezirk"].astype(str)
            + df["Kreis"].astype(str)
            + df["Gemeinde"].astype(str).str.zfill(3)
        )

        # rename columns
        df = df.rename(
            columns={
                "Wahlberechtigte (A)": "eligible_voters",
                "Wählende (B)": "total_voters",
                # first votes
                "E_Ungültige": "E_invalid_votes",
                "E_Gültige": "E_valid_votes",
                # second votes
                "Z_Ungültige": "Z_invalid_votes",
                "Z_Gültige": "Z_valid_votes",
            },
        )

        # drop rows with missing AGS components
        df = df.dropna(subset=["Land", "Regierungsbezirk", "Kreis", "Gemeinde"])

        return df

    @staticmethod
    def _normalize_city_state_ags(ags: pd.Series) -> pd.Series:
        """Map city-state district AGS to official municipality-level AGS codes."""

        hamburg_district_mask = ags.str.match(r"^0200[1-7]000$", na=False)
        berlin_district_mask = ags.str.match(r"^11[12]\d{5}$", na=False)

        normalized_ags = ags.where(~hamburg_district_mask, HAMBURG_CITY_AGS)
        normalized_ags = normalized_ags.where(~berlin_district_mask, BERLIN_CITY_AGS)
        return normalized_ags

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw election 21 data to include only relevant columns.

        - group data by municipality
        - replace absolute vote counts with proportions in each municipality.

        Args:
            df (pd.DataFrame): Raw election data DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with relevant election data.
        """

        # Merge city-state district records to one municipality per city.
        df["AGS"] = self._normalize_city_state_ags(df["AGS"].astype("string"))

        # select only columns started with "E_" or "Z_" or AGS or eligible_voters or total_voters
        df = df[
            [
                col
                for col in df.columns
                if col.startswith(("E_", "Z_")) or col in ("AGS", "eligible_voters", "total_voters")
            ]
        ]

        # group by municipality (AGS)
        df = df.groupby("AGS").sum().reset_index()

        vote_columns = [col for col in df.columns if col.startswith(("E_", "Z_"))]
        numeric_columns = ["eligible_voters", "total_voters", *vote_columns]

        # Ensure numeric dtypes before ratio calculations
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        # Use NaN (float-compatible) instead of pd.NA to keep float dtypes
        df["eligible_voters"] = df["eligible_voters"].replace(0, float("nan"))
        df["total_voters"] = df["total_voters"].replace(0, float("nan"))

        df["election_participation"] = (df["total_voters"] / df["eligible_voters"]).astype(float)

        # Relative votes per party in each municipality
        df[vote_columns] = df[vote_columns].div(df["total_voters"], axis=0).astype(float)

        # Save the transformed DataFrame to CSV
        df.to_csv(self.tform_data_path, index=False)

        return df
