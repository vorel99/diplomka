import shutil
from pathlib import Path

import pandas as pd

from geoscore_de.data_flow.election.utils import load_election_zip, move_extracted_file

ZIP_URL = "https://www.bundeswahlleiterin.de/en/dam/jcr/c2cd99e6-064e-4ebc-b634-f86b5c0e14b3/btw21_wbz.zip"

DEFAULT_RAW_DATA_PATH = "data/raw/features/election_2021"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/federal_election_21.csv"


def load_raw_election_21_data(url: str = ZIP_URL, dest_path: str = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """Load and extract election 21 data from a ZIP file.

    Args:
        url (str): URL to the ZIP file containing election data.
        dest_path (str): Destination path where the extracted files should be saved.

    Returns:
        pd.DataFrame: DataFrame containing the election data.
    """
    temp_dir = load_election_zip(url)
    try:
        # Ensure destination directory exists
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        move_extracted_file(temp_dir, dest_path)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Load the election data CSV into a DataFrame
    election_csv_path = Path(dest_path) / "btw21_wbz_ergebnisse.csv"
    dtypes = {
        "Land": str,
        "Regierungsbezirk": str,
        "Kreis": str,
        "Gemeinde": str,
    }
    df = pd.read_csv(election_csv_path, sep=";", dtype=dtypes)

    # create AGS column by concatenating Land, Regierungsbezirk, Kreis, and Gemeinde columns
    # Convert to string first to handle potential numeric types
    df["AGS"] = df["Land"] + df["Regierungsbezirk"] + df["Kreis"] + df["Gemeinde"].str.zfill(3)

    return df


def transform_election_21_data(
    in_path: str = DEFAULT_RAW_DATA_PATH,
    out_path: str = DEFAULT_TFORM_DATA_PATH,
) -> pd.DataFrame:
    """Transform raw election 21 data to include only relevant columns.

    - group data by municipality
    - replace absolute vote counts with proportions in each municipality.

    Args:
        in_path (str): Path to the input raw election data folder.
        out_path (str): Path to save the transformed election data.

    Returns:
        pd.DataFrame: Transformed DataFrame with relevant election data.
    """
    df = load_raw_election_21_data(dest_path=in_path)

    # TODO: rename columns
    df.rename(
        columns={
            "Wahlberechtigte (A)": "eligible_voters",
            "Wähler (B)": "total_voters",
            # first votes
            "E_Ungültige": "E_invalid_votes",
            # second votes
            "Z_Ungültige": "Z_invalid_votes",
        },
        inplace=True,
    )

    # TODO: group by municipality (AGS)

    # TODO: relative votes per party in each municipality

    # Save the transformed DataFrame to CSV
    df.to_csv(out_path, index=False)

    return df
