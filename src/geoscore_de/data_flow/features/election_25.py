import shutil
from pathlib import Path

import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.utils import load_election_zip, move_extracted_file

ZIP_URL = "https://www.bundeswahlleiterin.de/en/dam/jcr/e79a7bd3-0607-4e87-9752-8e601e299e00/btw25_wbz.zip"


class Election25Feature(BaseFeature):
    """Feature class for election 2025 data."""

    def __init__(
        self,
        url: str = ZIP_URL,
        raw_data_path: str = "data/raw/features/election_2025",
        tform_data_path: str = "data/tform/features/election_25.csv",
    ):
        self.url = url
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path

    def load(self) -> pd.DataFrame:
        """Load and extract election 25 data from a ZIP file."""
        temp_dir = load_election_zip(self.url)
        try:
            # Ensure destination directory exists
            Path(self.raw_data_path).mkdir(parents=True, exist_ok=True)
            move_extracted_file(temp_dir, self.raw_data_path)
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Load the election data CSV into a DataFrame
        election_csv_path = Path(self.raw_data_path) / "btw25_wbz_ergebnisse.csv"
        dtypes = {
            "Land": str,
            "Regierungsbezirk": str,
            "Kreis": str,
            "Gemeinde": str,
        }
        df = pd.read_csv(election_csv_path, sep=";", dtype=dtypes, skiprows=4)

        # create AGS column by concatenating Land, Regierungsbezirk, Kreis, and Gemeinde columns
        # Convert to string first to handle potential numeric types
        df["AGS"] = (
            df["Land"].astype(str)
            + df["Regierungsbezirk"].astype(str)
            + df["Kreis"].astype(str)
            + df["Gemeinde"].astype(str).str.zfill(3)
        )
        # drop none rows with missing AGS
        df = df.dropna(subset=["Land", "Regierungsbezirk", "Kreis", "Gemeinde"])
        return df

    def transform(self, df) -> pd.DataFrame:
        """No transformation implemented for election 25 data."""

        # select only columns ending with "Erststimmen" or "Zweitstimmen" or AGS or Wahlberechtigte (A) or Wählende (B)
        df = df[
            [
                col
                for col in df.columns
                if col.endswith(("Erststimmen", "Zweitstimmen"))
                or col in ("AGS", "Wahlberechtigte (A)", "Wählende (B)")
            ]
        ]

        # rename columns
        df = df.rename(
            columns={
                "Wahlberechtigte (A)": "eligible_voters",
                "Wählende (B)": "total_voters",
                "Ungültige - Zweitstimmen": "invalid_votes_zweitstimmen",
                "Gültige - Zweitstimmen": "valid_votes_zweitstimmen",
                "Ungültige - Erststimmen": "invalid_votes_erststimmen",
                "Gültige - Erststimmen": "valid_votes_erststimmen",
            }
        )

        # group by AGS and sum all other columns
        df = df.groupby("AGS").sum().reset_index()

        # Change all columns from absolute counts to proportions of the total voters
        df["eligible_voters"] = df["eligible_voters"].replace(0, pd.NA)
        df["election_participation"] = df["total_voters"] / df["eligible_voters"]

        for col in df.columns:
            if col not in ["AGS", "eligible_voters", "total_voters", "election_participation"]:
                df[col] = df[col] / df["total_voters"].replace(0, pd.NA)

        return df
