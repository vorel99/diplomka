import shutil
from pathlib import Path

from geoscore_de.data_flow.election.utils import load_election_zip, move_extracted_file

ZIP_URL = "https://www.bundeswahlleiterin.de/en/dam/jcr/c2cd99e6-064e-4ebc-b634-f86b5c0e14b3/btw21_wbz.zip"


def load_election_21_data(url: str = ZIP_URL, dest_path: str = "data/raw/election_2021") -> None:
    """Load and extract election 21 data from a ZIP file.

    Args:
        url (str): URL to the ZIP file containing election data.
        dest_path (str): Destination path where the extracted files should be saved.
    """
    temp_dir = load_election_zip(url)
    try:
        # Ensure destination directory exists
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        move_extracted_file(temp_dir, dest_path)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
