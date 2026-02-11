import shutil
from pathlib import Path

import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.utils import load_election_zip, move_extracted_file

ZIP_URL = "https://www.bundeswahlleiterin.de/en/dam/jcr/c2cd99e6-064e-4ebc-b634-f86b5c0e14b3/btw21_wbz.zip"

DEFAULT_RAW_DATA_PATH = "data/raw/features/election_2021"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/federal_election_21.csv"


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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw election 21 data to include only relevant columns.

        - group data by municipality
        - replace absolute vote counts with proportions in each municipality.

        Args:
            df (pd.DataFrame): Raw election data DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with relevant election data.
        """

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

        df["eligible_voters"] = df["eligible_voters"].replace(0, pd.NA)
        df["election_participation"] = df["total_voters"] / df["eligible_voters"]

        # # relative votes per party in each municipality
        vote_columns = [col for col in df.columns if col.startswith(("E_", "Z_"))]
        for col in vote_columns:
            df[col] = df[col] / df["total_voters"].replace(0, pd.NA)

        # Save the transformed DataFrame to CSV
        df.to_csv(self.tform_data_path, index=False)

        return df
