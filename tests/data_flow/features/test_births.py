import pytest

from geoscore_de.data_flow.features import BirthsFeature


@pytest.fixture
def mock_raw_csv_content():
    """Create mock raw CSV content matching the expected format."""
    return """;
;
;
;
;
2023;01001;      Flensburg, kreisfreie Stadt;859
2023;01002;      Kiel, kreisfreie Stadt;2059
2023;01003;      L�beck, kreisfreie Stadt, Hansestadt;1666
2023;01004;      Neum�nster, kreisfreie Stadt;686
2023;01051;      Dithmarschen, Kreis;1021
2023;01051001;        Albersdorf;27
2023;01051002;        Arkebek;4
;;;;
;;;;
;;;;
;;;;"""


def test_load_births_data(mock_raw_csv_content, tmp_path):
    """Test loading of births data from CSV."""
    # Create a temporary CSV file with the mock content
    temp_csv_path = tmp_path / "births.csv"
    temp_csv_path.write_text(mock_raw_csv_content)

    # Load the data using the BirthsFeature class
    feature = BirthsFeature(raw_data_path=str(temp_csv_path))
    df = feature.load()

    # Check columns exist
    assert "AGS" in df.columns
    assert "births" in df.columns

    # check dtypes
    assert df["AGS"].dtype == object  # AGS should be string
    assert df["births"].dtype in [int, float]

    # check AGS values are correctly created
    df_ags = df["AGS"].tolist()
    assert "01001000" in df_ags  # Flensburg
    assert "01004000" in df_ags  # Neumünster
    assert "01051000" in df_ags  # Dithmarschen
    assert "01051001" in df_ags  # Albersdorf
    assert "01051002" in df_ags  # Arkebek
