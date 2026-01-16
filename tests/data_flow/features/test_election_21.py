"""Tests for election 2021 data loading and transformation."""

from unittest.mock import patch

import pandas as pd
import pytest

from geoscore_de.data_flow.features.election_21 import Election21Feature


@pytest.fixture
def mock_raw_election_csv_content():
    """Create mock raw CSV content matching the expected format."""
    return """Land;Regierungsbezirk;Kreis;Gemeinde;Wahlberechtigte (A);Wählende (B);E_Ungültige;E_Gültige;E_CDU;E_SPD;E_GRÜNE;Z_Ungültige;Z_Gültige;Z_CDU;Z_SPD;Z_GRÜNE
01;0;01;000;50000;40000;200;39800;15000;12000;8000;150;39850;14000;13000;9000
01;0;01;001;30000;25000;100;24900;10000;8000;5000;80;24920;9500;8500;5500
02;0;02;000;40000;35000;150;34850;12000;10000;7000;120;34880;11500;11000;7500"""  # noqa: E501


@pytest.fixture
def mock_raw_election_csv_file(tmp_path, mock_raw_election_csv_content):
    """Create a temporary CSV file with mock election data."""
    csv_file = tmp_path / "btw21_wbz_ergebnisse.csv"
    csv_file.write_text(mock_raw_election_csv_content)
    return tmp_path


def test_load_election_21_data(mock_raw_election_csv_file):
    """Test loading raw election 21 data from CSV."""
    feature = Election21Feature(raw_data_path=str(mock_raw_election_csv_file))

    # Patch load_election_zip to avoid actual download
    with patch("geoscore_de.data_flow.features.election_21.load_election_zip"):
        with patch("geoscore_de.data_flow.features.election_21.move_extracted_file"):
            df = feature.load()

    # Check required columns exist
    assert "AGS" in df.columns
    assert "Wahlberechtigte (A)" in df.columns
    assert "Wählende (B)" in df.columns
    assert "E_CDU" in df.columns
    assert "Z_CDU" in df.columns

    # Check AGS is properly created
    assert df["AGS"].dtype == object
    assert len(df) == 3


def test_load_election_21_data_ags_format(mock_raw_election_csv_file):
    """Test that AGS codes are properly formatted."""
    feature = Election21Feature(raw_data_path=str(mock_raw_election_csv_file))

    # Patch load_election_zip to avoid actual download
    with patch("geoscore_de.data_flow.features.election_21.load_election_zip"):
        with patch("geoscore_de.data_flow.features.election_21.move_extracted_file"):
            df = feature.load()

    # Check AGS format: Land (2) + Regierungsbezirk (1) + Kreis (3) + Gemeinde (3)
    ags_values = df["AGS"].unique()

    # Should have AGS codes like "01001000", "01001001", "02002000"
    assert "01001000" in ags_values
    assert "01001001" in ags_values
    assert "02002000" in ags_values


def test_transform_election_21_data_structure(tmp_path, mock_raw_election_csv_file):
    """Test the structure of transformed election data."""
    out_path = tmp_path / "output.csv"

    feature = Election21Feature(raw_data_path=str(mock_raw_election_csv_file), tform_data_path=str(out_path))
    with patch("geoscore_de.data_flow.features.election_21.load_election_zip"):
        with patch("geoscore_de.data_flow.features.election_21.move_extracted_file"):
            df = feature.transform(feature.load())

    # Check that AGS column exists
    assert "AGS" in df.columns

    # Check renamed columns exist
    assert "eligible_voters" in df.columns
    assert "total_voters" in df.columns
    assert "election_participation" in df.columns
    assert "E_invalid_votes" in df.columns
    assert "Z_invalid_votes" in df.columns

    # Check that vote columns still exist (now as proportions)
    assert "E_CDU" in df.columns
    assert "Z_CDU" in df.columns
    assert "E_SPD" in df.columns
    assert "Z_SPD" in df.columns


def test_transform_election_21_data_grouping(tmp_path, mock_raw_election_csv_file):
    """Test that data is properly grouped by AGS."""
    out_path = tmp_path / "output.csv"

    feature = Election21Feature(raw_data_path=str(mock_raw_election_csv_file), tform_data_path=str(out_path))
    with patch("geoscore_de.data_flow.features.election_21.load_election_zip"):
        with patch("geoscore_de.data_flow.features.election_21.move_extracted_file"):
            raw_df = feature.load()
            transformed_df = feature.transform(raw_df)

    # After grouping, we should have fewer rows
    # Original has 3 municipalities, but some may share AGS prefix
    assert len(transformed_df) <= len(raw_df)

    # Check that AGS values are unique
    assert len(transformed_df["AGS"].unique()) == len(transformed_df)


def test_transform_election_21_proportions(tmp_path, mock_raw_election_csv_file):
    """Test that vote counts are converted to proportions."""
    out_path = tmp_path / "output.csv"

    feature = Election21Feature(raw_data_path=str(mock_raw_election_csv_file), tform_data_path=str(out_path))
    with patch("geoscore_de.data_flow.features.election_21.load_election_zip"):
        with patch("geoscore_de.data_flow.features.election_21.move_extracted_file"):
            df = feature.transform(feature.load())

    # Check that vote proportions are between 0 and 1 (excluding NaN)
    vote_columns = [col for col in df.columns if col.startswith(("E_", "Z_"))]

    for col in vote_columns:
        non_na_values = df[col].dropna()
        if len(non_na_values) > 0:
            assert (non_na_values >= 0).all(), f"{col} has negative values"
            assert (non_na_values <= 1).all(), f"{col} has values > 1"


def test_transform_election_21_participation(tmp_path, mock_raw_election_csv_file):
    """Test that election participation is calculated correctly."""
    out_path = tmp_path / "output.csv"

    feature = Election21Feature(raw_data_path=str(mock_raw_election_csv_file), tform_data_path=str(out_path))
    with patch("geoscore_de.data_flow.features.election_21.load_election_zip"):
        with patch("geoscore_de.data_flow.features.election_21.move_extracted_file"):
            df = feature.transform(feature.load())

    # Check that participation is between 0 and 1
    participation = df["election_participation"].dropna()
    assert (participation >= 0).all()
    assert (participation <= 1).all()


def test_transform_election_21_zero_voters_handling(tmp_path):
    """Test handling of zero total voters to avoid division by zero."""
    # Create data with zero voters
    csv_content = """Land;Regierungsbezirk;Kreis;Gemeinde;Wahlberechtigte (A);Wählende (B);E_Ungültige;E_Gültige;E_CDU;E_SPD;Z_Ungültige;Z_Gültige;Z_CDU;Z_SPD
01;0;01;000;1000;0;0;0;0;0;0;0;0;0"""  # noqa: E501
    csv_file = tmp_path / "btw21_wbz_ergebnisse.csv"
    csv_file.write_text(csv_content)
    out_path = tmp_path / "output.csv"

    feature = Election21Feature(raw_data_path=str(tmp_path), tform_data_path=str(out_path))
    with patch("geoscore_de.data_flow.features.election_21.load_election_zip"):
        with patch("geoscore_de.data_flow.features.election_21.move_extracted_file"):
            df = feature.transform(feature.load())

    # Should have NaN for proportions when total_voters is 0, not infinity
    vote_columns = [col for col in df.columns if col.startswith(("E_", "Z_"))]
    for col in vote_columns:
        assert not (df[col] == float("inf")).any(), f"{col} contains infinity values"


def test_transform_election_21_saves_to_file(tmp_path, mock_raw_election_csv_file):
    """Test that transformed data is saved to CSV file."""
    out_path = tmp_path / "output.csv"

    feature = Election21Feature(raw_data_path=str(mock_raw_election_csv_file), tform_data_path=str(out_path))
    with patch("geoscore_de.data_flow.features.election_21.load_election_zip"):
        with patch("geoscore_de.data_flow.features.election_21.move_extracted_file"):
            feature.transform(feature.load())

    # Check that output file exists
    assert out_path.exists()

    # Check that saved file can be read and has expected columns
    saved_df = pd.read_csv(out_path)
    assert "AGS" in saved_df.columns
    assert "eligible_voters" in saved_df.columns
    assert "election_participation" in saved_df.columns
