import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

EXTRACT_FILES = [
    "btw25_wbz_ergebnisse.csv",
    "btw25_wbz_dsb_ergebnisse.pdf",
]

ZIP_URL = "https://www.bundeswahlleiterin.de/en/dam/jcr/e79a7bd3-0607-4e87-9752-8e601e299e00/btw25_wbz.zip"


def load_election_zip(url: str) -> str:
    """Load election data from a ZIP file and save it extracted to temp directory.

    Args:
        url (str): URL to the ZIP file containing election data.

    Returns:
        str: Path to the extracted data file."""

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    # Download the ZIP file
    print(f"Downloading from: {url}")
    response = requests.get(url)
    zip_path = Path(temp_dir) / "election_data.zip"
    with open(zip_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded ZIP file to: {zip_path}")

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        extracted_files = zip_ref.namelist()
        print(f"Files in ZIP: {extracted_files}")
        zip_ref.extractall(temp_dir)

    # Remove the ZIP file after extraction
    zip_path.unlink()
    print(f"Extracted files to: {temp_dir}")

    # List what's actually in the temp directory
    extracted_contents = list(Path(temp_dir).iterdir())
    print(f"Contents of temp directory: {[f.name for f in extracted_contents]}")

    return temp_dir


def move_extracted_file(temp_dir: str, dest_path: str) -> None:
    """Move the extracted file from the temporary directory to the destination path.

    Args:
        temp_dir (str): Path to the temporary directory containing the extracted file.
        dest_path (str): Destination path where the file should be moved.
    """
    temp_path = Path(temp_dir)
    dest_path_obj = Path(dest_path)

    # Ensure destination directory exists
    dest_path_obj.mkdir(parents=True, exist_ok=True)

    for file_name in EXTRACT_FILES:
        src_file = temp_path / file_name
        dest_file = dest_path_obj / file_name
        if src_file.exists():
            # Use copy2 to preserve metadata, then remove source
            shutil.copy2(src_file, dest_file)
            print(f"Successfully moved {file_name} to {dest_file}")
        else:
            print(f"Warning: {file_name} not found in extracted files")


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
        move_extracted_file(temp_dir, dest_path)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
