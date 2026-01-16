"""Tests for population data loading and transformation."""

import pandas as pd
import pytest

from geoscore_de.data_flow.features.population import PopulationFeature


@pytest.fixture
def mock_raw_csv_content():
    """Create mock raw CSV content matching the expected format."""
    return """;;;;
;;;;
;;;;
;;;;
;;;;
;;;;
31.12.2022;DG;Gemeinde Name;unter 3 Jahre;100;50;50
31.12.2022;DG;Gemeinde Name;3 bis unter 6 Jahre;150;75;75
31.12.2022;DG;Gemeinde Name;Insgesamt;1000;500;500
31.12.2022;01001;Flensburg;unter 3 Jahre;200;100;100
31.12.2022;01001;Flensburg;3 bis unter 6 Jahre;300;150;150
31.12.2022;01001;Flensburg;6 bis unter 10 Jahre;400;200;200
31.12.2022;01001;Flensburg;Insgesamt;2000;1000;1000
31.12.2022;05;Nordrhein-Westfalen;unter 3 Jahre;.;-;-
31.12.2022;05;Nordrhein-Westfalen;Insgesamt;.;-;-
;;;;
;;;;
;;;;
;;;;"""


@pytest.fixture
def mock_raw_csv_file(tmp_path, mock_raw_csv_content):
    """Create a temporary CSV file with mock data."""
    csv_file = tmp_path / "population.csv"
    csv_file.write_text(mock_raw_csv_content)
    return str(csv_file)


def test_load_raw_population_data(mock_raw_csv_file):
    """Test loading raw population data from CSV."""
    feature = PopulationFeature(raw_data_path=mock_raw_csv_file)
    df = feature.load()

    # Check columns exist
    assert "AGS" in df.columns
    assert "age_group" in df.columns
    assert "people_count" in df.columns
    assert "male_count" in df.columns
    assert "female_count" in df.columns
    assert "Municipality" in df.columns

    # Check data types and values
    assert df["AGS"].dtype == object
    assert len(df) > 0


def test_load_raw_population_data_ags_padding(mock_raw_csv_file):
    """Test that AGS codes are properly padded to 8 characters."""
    feature = PopulationFeature(raw_data_path=mock_raw_csv_file)
    df = feature.load()

    # Check AGS padding - should all be 8 characters
    ags_values = df["AGS"].unique()
    for ags in ags_values:
        assert len(ags) == 8, f"AGS {ags} is not 8 characters"

    # Check specific cases
    # "DG" should become "DG000000"
    # "01001" should become "01001000"
    # "05" should become "05000000"
    assert "DG000000" in ags_values
    assert "01001000" in ags_values
    assert "05000000" in ags_values


def test_load_raw_population_data_na_handling(mock_raw_csv_file):
    """Test that NA values are properly handled."""
    feature = PopulationFeature(raw_data_path=mock_raw_csv_file)
    df = feature.load()

    # Check that rows with NA values exist
    na_rows = df[df["people_count"].isna()]
    assert len(na_rows) > 0, "Should have rows with NaN values"


def test_transform_population_data_structure(tmp_path, mock_raw_csv_file):
    """Test the structure of transformed population data."""
    out_path = tmp_path / "output.csv"

    feature = PopulationFeature(raw_data_path=mock_raw_csv_file, tform_data_path=str(out_path))
    df = feature.transform(feature.load())

    # Check that AGS column exists
    assert "AGS" in df.columns

    # Check that English column names exist
    assert "age_under_3" in df.columns
    assert "age_3_to_5" in df.columns
    assert "age_6_to_9" in df.columns

    # Check that total_population column is removed
    assert "total_population" not in df.columns

    # Check that German columns are gone
    assert "unter 3 Jahre" not in df.columns
    assert "Insgesamt" not in df.columns


def test_transform_population_data_proportions(tmp_path, mock_raw_csv_file):
    """Test that values are converted to proportions."""
    out_path = tmp_path / "output.csv"

    feature = PopulationFeature(raw_data_path=mock_raw_csv_file, tform_data_path=str(out_path))
    df = feature.transform(feature.load())

    # Get a row with data (not NaN)
    valid_rows = df.dropna(subset=["age_under_3"])

    if len(valid_rows) > 0:
        row = valid_rows.iloc[0]

        # Check that values are proportions (between 0 and 1)
        for col in df.columns:
            if col != "AGS" and not pd.isna(row[col]):
                assert 0 <= row[col] <= 1, f"Column {col} value {row[col]} is not a proportion"


def test_transform_population_data_division_by_zero(tmp_path):
    """Test handling of division by zero when total_population is 0."""
    # Create test data with zero total population
    test_csv = """;;;;
;;;;
;;;;
;;;;
;;;;
31.12.2022;TEST;Test Municipality;unter 3 Jahre;0;0;0
31.12.2022;TEST;Test Municipality;Insgesamt;0;0;0
;;;;
;;;;
;;;;
;;;;"""

    csv_file = tmp_path / "test_input.csv"
    csv_file.write_text(test_csv)
    out_path = tmp_path / "test_output.csv"

    feature = PopulationFeature(raw_data_path=str(csv_file), tform_data_path=str(out_path))
    df = feature.transform(feature.load())

    # Check that we don't have inf values
    for col in df.columns:
        if col != "AGS":
            assert not (df[col] == float("inf")).any(), f"Column {col} contains inf values"
            assert not (df[col] == float("-inf")).any(), f"Column {col} contains -inf values"


def test_transform_population_data_output_file(tmp_path, mock_raw_csv_file):
    """Test that output file is created."""
    out_path = tmp_path / "output.csv"

    feature = PopulationFeature(raw_data_path=mock_raw_csv_file, tform_data_path=str(out_path))
    _ = feature.transform(feature.load())

    # Check file exists
    assert out_path.exists()

    # Check file can be read back
    loaded_df = pd.read_csv(out_path)
    assert len(loaded_df) > 0
    assert "AGS" in loaded_df.columns


def test_transform_population_data_pivot(tmp_path, mock_raw_csv_file):
    """Test that data is properly pivoted."""
    out_path = tmp_path / "output.csv"

    feature = PopulationFeature(raw_data_path=mock_raw_csv_file, tform_data_path=str(out_path))
    df = feature.transform(feature.load())

    # After pivoting, each AGS should appear only once
    assert df["AGS"].is_unique

    # Check that we have multiple age group columns
    age_columns = [col for col in df.columns if col.startswith("age_")]
    assert len(age_columns) > 0


def test_transform_population_data_nan_handling(tmp_path):
    """Test that NaN values are preserved correctly."""
    # Create test data with some NaN values
    test_csv = """;;;;
;;;;
;;;;
;;;;
;;;;
;;;;
31.12.2022;TEST1;Test Municipality 1;unter 3 Jahre;100;50;50
31.12.2022;TEST1;Test Municipality 1;Insgesamt;1000;500;500
31.12.2022;TEST2;Test Municipality 2;unter 3 Jahre;-;-;-
31.12.2022;TEST2;Test Municipality 2;Insgesamt;.;.;.
;;;;
;;;;
;;;;
;;;;"""

    csv_file = tmp_path / "test_input.csv"
    csv_file.write_text(test_csv)
    out_path = tmp_path / "test_output.csv"

    feature = PopulationFeature(raw_data_path=str(csv_file), tform_data_path=str(out_path))
    df = feature.transform(feature.load())

    # Check that we have both valid data and NaN values
    assert df["age_under_3"].notna().any(), "Should have some valid values"
    assert df["age_under_3"].isna().any(), "Should have some NaN values"


def test_transform_population_data_all_columns_renamed(tmp_path, mock_raw_csv_file):
    """Test that all age group columns are renamed to English."""
    out_path = tmp_path / "output.csv"

    feature = PopulationFeature(raw_data_path=mock_raw_csv_file, tform_data_path=str(out_path))
    df = feature.transform(feature.load())

    # Expected English column names
    expected_columns = [
        "age_under_3",
        "age_3_to_5",
        "age_6_to_9",
        "age_10_to_14",
        "age_15_to_17",
        "age_18_to_19",
        "age_20_to_24",
        "age_25_to_29",
        "age_30_to_34",
        "age_35_to_39",
        "age_40_to_44",
        "age_45_to_49",
        "age_50_to_54",
        "age_55_to_59",
        "age_60_to_64",
        "age_65_to_74",
        "age_75_and_over",
    ]

    # Check that expected columns exist (if they're in the data)
    for col in df.columns:
        if col != "AGS":
            assert col in expected_columns, f"Unexpected column: {col}"


def test_load_raw_population_data_column_types(mock_raw_csv_file):
    """Test that loaded data has correct column types."""
    feature = PopulationFeature(raw_data_path=mock_raw_csv_file)
    df = feature.load()

    # Numeric columns should be float (allowing for NaN)
    assert df["people_count"].dtype in [float, "float64"]
    assert df["male_count"].dtype in [float, "float64"]
    assert df["female_count"].dtype in [float, "float64"]

    # String columns
    assert df["AGS"].dtype == object
    assert df["age_group"].dtype == object
