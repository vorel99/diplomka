import pytest

from geoscore_de.data_flow.features.municipality import MunicipalityFeature


@pytest.fixture
def mock_raw_csv_content():
    """Create mock raw CSV content matching the expected format."""
    return """Tabelle: 1000A-0001
Persons: Official population and area (municipalities);;;;
Population in brief (territory on 15 May 2022);;;;
;;Persons;Area;Population density
;;number;km²;Inh/km²
2022-05-15;;;;
010010000000;Flensburg, Stadt;95015;56.73;1675
010020000000;Kiel, Landeshauptstadt;249132;118.65;2100
010030000000;Lübeck, Hansestadt;215958;214.19;1008
010040000000;Neumünster, Stadt;79625;71.66;1111
010510011011;Brunsbüttel, Stadt;12573;65.21;193
010510044044;Heide, Stadt;21487;31.97;672
010515163003;Averlak;568;9.06;63
010515163010;Brickeln;191;6.07;31
;;;;
;;;;
;;;;
;;;;"""


def test_load_municipality_data(mock_raw_csv_content, tmp_path):
    """Test loading of municipality data from CSV."""
    # Create a temporary CSV file with the mock content
    temp_csv_path = tmp_path / "municipalities_2022.csv"
    temp_csv_path.write_text(mock_raw_csv_content)

    # Load the data using the MunicipalityFeature class
    feature = MunicipalityFeature(raw_data_path=str(temp_csv_path))
    df = feature.load()

    # Check columns exist
    assert "AGS" in df.columns
    assert "Persons" in df.columns
    assert "Area" in df.columns

    # check AGS values are correctly created
    expected_ags = [
        "01001000",  # Flensburg
        "01002000",  # Kiel
        "01003000",  # Lübeck
        "01004000",  # Neumünster
        "01051011",  # Brunsbüttel
        "01051044",  # Heide
        "01051003",  # Averlak
        "01051010",  # Brickeln
    ]
    assert df["AGS"].tolist() == expected_ags
