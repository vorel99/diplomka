import shutil
from pathlib import Path

import pandas as pd

from geoscore_de.data_flow.features.election import BaseElectionFeature
from geoscore_de.data_flow.features.municipality import DEFAULT_RAW_DATA_PATH as MUNICIPALITY_RAW_DATA_PATH
from geoscore_de.data_flow.features.utils import load_election_zip, move_extracted_file

ZIP_URL = "https://www.bundeswahlleiterin.de/en/dam/jcr/e79a7bd3-0607-4e87-9752-8e601e299e00/btw25_wbz.zip"

DEFAULT_RAW_DATA_PATH = "data/raw/features/election_2025"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/federal_election_25.csv"
HAMBURG_CITY_AGS = "02000000"
BERLIN_CITY_AGS = "11000000"


class Election25Feature(BaseElectionFeature):
    """Feature class for election 2025 data."""

    def __init__(
        self,
        url: str = ZIP_URL,
        raw_data_path: str = DEFAULT_RAW_DATA_PATH,
        tform_data_path: str = DEFAULT_TFORM_DATA_PATH,
        municipality_data_path: str = MUNICIPALITY_RAW_DATA_PATH,
        fix_missing: bool = True,
        **kwargs,
    ):
        super().__init__(municipality_data_path=municipality_data_path, **kwargs)
        self.url = url
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path
        self.fix_missing = fix_missing

    def load(self) -> pd.DataFrame:
        """Load and extract election 25 data from a ZIP file.

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

        # rename columns
        df = df.rename(
            columns={
                "Wahlberechtigte (A)": "eligible_voters",
                "Wählende (B)": "total_voters",
                "Ungültige - Erststimmen": "E_invalid_votes",
                "Gültige - Erststimmen": "E_valid_votes",
                "Ungültige - Zweitstimmen": "Z_invalid_votes",
                "Gültige - Zweitstimmen": "Z_valid_votes",
            }
        )

        # drop rows with missing AGS
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
        """Transform raw election 2025 data into aggregated, proportional metrics.

        This method:
        - selects first- and second-vote columns, along with AGS, eligible voters, and total voters,
        - aggregates all vote counts by AGS via summation, and
        - computes election participation and converts vote counts to proportions of total voters.
        """
        # Merge city-state district records to one municipality per city.
        df["AGS"] = self._normalize_city_state_ags(df["AGS"].astype("string"))

        # select only columns ending with "Erststimmen" or "Zweitstimmen" or AGS or Wahlberechtigte (A) or Wählende (B)
        df = df[
            [
                col
                for col in df.columns
                if col.endswith(("Erststimmen", "Zweitstimmen"))
                or col
                in (
                    "AGS",
                    "eligible_voters",
                    "total_voters",
                    "E_invalid_votes",
                    "E_valid_votes",
                    "Z_invalid_votes",
                    "Z_valid_votes",
                    "Gemeindename",
                )
            ]
        ]

        # rename Erststimmen and Zweitstimmen columns to start with E_ and Z_ respectively
        df = df.rename(
            columns={
                col: "E_" + col.replace(" - Erststimmen", "")
                if col.endswith("Erststimmen")
                else "Z_" + col.replace(" - Zweitstimmen", "")
                if col.endswith("Zweitstimmen")
                else col
                for col in df.columns
            }
        )

        # group by AGS and sum all other columns
        df = df.groupby("AGS").sum().reset_index()

        if self.fix_missing:
            df = self._fix_missing(df)

        df = df.drop(columns=["Gemeindename"], errors="ignore")

        # Change all columns from absolute counts to proportions of the total voters
        df["eligible_voters"] = df["eligible_voters"].replace(0, float("nan"))
        df["election_participation"] = df["total_voters"] / df["eligible_voters"]

        for col in df.columns:
            if col not in ["AGS", "eligible_voters", "total_voters", "election_participation"]:
                df[col] = df[col] / df["total_voters"].replace(0, float("nan"))

        df.to_csv(self.tform_data_path, index=False)
        return df
