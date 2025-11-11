import shutil
from pathlib import Path

from geoscore_de.data_flow.election.utils import load_election_zip, move_extracted_file

EXTRACT_FILES = [
    "btw25_wbz_ergebnisse.csv",
    "btw25_wbz_dsb_ergebnisse.pdf",
]

ZIP_URL = "https://www.bundeswahlleiterin.de/en/dam/jcr/e79a7bd3-0607-4e87-9752-8e601e299e00/btw25_wbz.zip"


def load_election_25_data(url: str = ZIP_URL, dest_path: str = "data/raw/election_2025") -> None:
    """Load and extract election 25 data from a ZIP file.

    Args:
        url (str): URL to the ZIP file containing election data.
        dest_path (str): Destination path where the extracted files should be saved.
    """
    temp_dir = load_election_zip(url)
    try:
        # Ensure destination directory exists
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        move_extracted_file(temp_dir, dest_path, EXTRACT_FILES)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
