"""Tests for election 2021 data loading and transformation."""

from unittest.mock import patch

import pandas as pd
import pytest

from geoscore_de.data_flow.features.election_21 import Election21Feature


@pytest.fixture
def mock_raw_election_csv_content():
    """Create mock raw CSV content matching the expected format."""
    return """Gemeinde Name;Land;Regierungsbezirk;Kreis;Gemeinde;Wahlberechtigte (A);Wählende (B);E_Ungültige;E_Gültige;E_CDU;E_SPD;E_GRÜNE;Z_Ungültige;Z_Gültige;Z_CDU;Z_SPD;Z_GRÜNE
Flensburg;01;0;01;000;50000;40000;200;39800;15000;12000;8000;150;39850;14000;13000;9000
Kiel;01;0;01;001;30000;25000;100;24900;10000;8000;5000;80;24920;9500;8500;5500
Lübeck;02;0;02;000;40000;35000;150;34850;12000;10000;7000;120;34880;11500;11000;7500"""  # noqa: E501


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
    assert "eligible_voters" in df.columns
    assert "total_voters" in df.columns
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
    csv_content = """Gemeinde Name;Land;Regierungsbezirk;Kreis;Gemeinde;Wahlberechtigte (A);Wählende (B);E_Ungültige;E_Gültige;E_CDU;E_SPD;Z_Ungültige;Z_Gültige;Z_CDU;Z_SPD
Flensburg;01;0;01;000;1000;0;0;0;0;0;0;0;0;0"""  # noqa: E501
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


def test_transform_election_21_merges_berlin_hamburg_districts(tmp_path):
    """Test Berlin/Hamburg district AGS are merged to city-level AGS."""
    csv_content = """Land;Regierungsbezirk;Kreis;Gemeinde;Wahlberechtigte (A);Wählende (B);E_Ungültige;E_Gültige;E_CDU;E_SPD;Z_Ungültige;Z_Gültige;Z_CDU;Z_SPD
02;0;01;000;1000;800;10;790;300;250;8;792;310;240
02;0;02;000;1500;1200;12;1188;400;380;10;1190;420;360
11;1;01;000;2000;1500;15;1485;500;450;12;1488;520;430
11;2;12;000;1800;1400;14;1386;450;420;11;1389;460;410
12;0;73;032;900;700;7;693;220;210;6;694;230;200"""  # noqa: E501

    csv_file = tmp_path / "btw21_wbz_ergebnisse.csv"
    csv_file.write_text(csv_content)
    out_path = tmp_path / "output.csv"

    feature = Election21Feature(raw_data_path=str(tmp_path), tform_data_path=str(out_path), fix_missing=False)
    with patch("geoscore_de.data_flow.features.election_21.load_election_zip"):
        with patch("geoscore_de.data_flow.features.election_21.move_extracted_file"):
            transformed_df = feature.transform(feature.load())

    assert "02000000" in transformed_df["AGS"].values
    assert "11000000" in transformed_df["AGS"].values
    assert "02001000" not in transformed_df["AGS"].values
    assert "02002000" not in transformed_df["AGS"].values
    assert "11101000" not in transformed_df["AGS"].values
    assert "11212000" not in transformed_df["AGS"].values

    # Non-Berlin/Hamburg AGS must remain untouched.
    assert "12073032" in transformed_df["AGS"].values


class TestFixMissing:
    """Tests for the _fix_missing method of Election21Feature."""

    @pytest.fixture
    def mock_municipality_data_simple(self):
        """Create simple mock municipality data."""
        return pd.DataFrame(
            {"AGS": ["01001000", "01002000", "01003000"], "Municipality": ["Flensburg", "Kiel", "Lübeck"]}
        )

    @pytest.fixture
    def mock_election_data_simple(self):
        """Create simple mock election data."""
        return pd.DataFrame(
            {
                "AGS": ["01001000", "01002000"],
                "Gemeinde Name": ["Flensburg", "Kiel"],
                "eligible_voters": [100, 200],
                "total_voters": [80, 160],
                "E_CDU": [30, 60],
                "E_SPD": [20, 50],
                "Z_CDU": [32, 62],
                "Z_SPD": [22, 52],
            }
        )

    def test_fix_missing_no_missing_municipalities(self, mock_election_data_simple, mock_municipality_data_simple):
        """Test when there are no missing municipalities - data unchanged."""
        # All municipalities in election data are also in municipality data
        election_data = mock_election_data_simple.copy()
        municipality_data = mock_municipality_data_simple[
            mock_municipality_data_simple["AGS"].isin(["01001000", "01002000"])
        ]

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election_21.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = municipality_data

            result = feature._fix_missing(election_data.copy())

        # Since there are no missing municipalities, data should be unchanged
        assert len(result) == len(election_data)
        assert list(result["AGS"].values) == list(election_data["AGS"].values)

    def test_fix_missing_finds_einschl_municipality(self, mock_election_data_simple, mock_municipality_data_simple):
        """Test fixing missing municipality found in einschl. municipality."""
        # Create election data where Lübeck is missing but included in Flensburg
        election_data = pd.DataFrame(
            {
                "AGS": ["01001000", "01002000"],
                "Gemeinde Name": ["Flensburg (einschl. Lübeck)", "Kiel"],
                "eligible_voters": [150, 200],
                "total_voters": [120, 160],
                "E_CDU": [45, 60],
                "E_SPD": [30, 50],
                "Z_CDU": [48, 62],
                "Z_SPD": [32, 52],
            }
        )

        # Municipality data includes all three municipalities
        municipality_data = pd.DataFrame(
            {"AGS": ["01001000", "01002000", "01003000"], "Municipality": ["Flensburg", "Kiel", "Lübeck"]}
        )

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election_21.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = municipality_data

            result = feature._fix_missing(election_data.copy())

        # Should have one more row (Lübeck added)
        assert len(result) > len(election_data)
        # Check that Lübeck's AGS is now present
        assert "01003000" in result["AGS"].values
        # Verify the added row has data from Flensburg
        luebeck_row = result[result["AGS"] == "01003000"].iloc[0]
        assert luebeck_row["eligible_voters"] == 150
        assert luebeck_row["E_CDU"] == 45

    def test_fix_missing_preserves_original_data(self, mock_election_data_simple, mock_municipality_data_simple):
        """Test that _fix_missing preserves original data for non-missing municipalities."""
        election_data = mock_election_data_simple.copy()
        original_eligible_voters = election_data.iloc[0]["eligible_voters"]
        original_ags = election_data.iloc[0]["AGS"]

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election_21.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = mock_municipality_data_simple

            result = feature._fix_missing(election_data.copy())

        # Original data should still be present
        original_row = result[result["AGS"] == original_ags].iloc[0]
        assert original_row["eligible_voters"] == original_eligible_voters

    def test_fix_missing_copies_row_correctly(self):
        """Test that copied rows have correct values from the einschl. municipality."""
        election_data = pd.DataFrame(
            {
                "AGS": ["01001000"],
                "Gemeinde Name": ["Flensburg (einschl. Lübeck)"],
                "eligible_voters": [150],
                "total_voters": [120],
                "E_CDU": [45],
                "E_SPD": [30],
                "Z_CDU": [48],
                "Z_SPD": [32],
            }
        )

        municipality_data = pd.DataFrame({"AGS": ["01001000", "01003000"], "Municipality": ["Flensburg", "Lübeck"]})

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election_21.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = municipality_data

            result = feature._fix_missing(election_data.copy())

        # Find the added row (Lübeck)
        luebeck_row = result[result["AGS"] == "01003000"]
        assert len(luebeck_row) == 1

        # Check that it copied the values from Flensburg
        luebeck_row = luebeck_row.iloc[0]
        assert luebeck_row["eligible_voters"] == 150
        assert luebeck_row["total_voters"] == 120
        assert luebeck_row["E_CDU"] == 45
        assert luebeck_row["Z_SPD"] == 32

    def test_fix_missing_no_match_not_added(self, mock_election_data_simple):
        """Test that municipalities not found in einschl. are not added."""

        municipality_data = pd.DataFrame(
            {"AGS": ["01001000", "01002000", "01003000"], "Municipality": ["Flensburg", "Kiel", "NonExistentCity"]}
        )

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election_21.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = municipality_data

            result = feature._fix_missing(mock_election_data_simple.copy())

        # NonExistentCity should not be added (not mentioned in any einschl. name)
        assert "01003000" not in result["AGS"].values
        # Original data should remain unchanged
        assert len(result) == len(mock_election_data_simple)

    def test_fix_missing_disabled(self, mock_election_data_simple):
        """Test that fix_missing is disabled when flag is False."""
        election_data = mock_election_data_simple.copy()

        feature = Election21Feature(fix_missing=False)

        # When fix_missing is False, _fix_missing should not be called on missing data
        # The function should still work but won't add missing municipalities
        result = feature._fix_missing(election_data)

        # Data should remain unchanged
        assert len(result) == len(election_data)

    def test_fix_missing_with_nan_values(self, mock_municipality_data_simple):
        """Test that _fix_missing handles NaN values gracefully."""
        election_data = pd.DataFrame(
            {
                "AGS": ["01001000", "01002000"],
                "Gemeinde Name": ["Flensburg (einschl. Lübeck)", "Kiel"],
                "eligible_voters": [100.0, 200.0],
                "total_voters": [80.0, float("nan")],
                "E_CDU": [30.0, float("nan")],
                "E_SPD": [20.0, 50.0],
                "Z_CDU": [32.0, 62.0],
                "Z_SPD": [22.0, 52.0],
            }
        )

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election_21.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = mock_municipality_data_simple

            # Should not raise an error
            result = feature._fix_missing(election_data.copy())
            assert isinstance(result, pd.DataFrame)
