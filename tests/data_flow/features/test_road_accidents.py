import pytest

from geoscore_de.data_flow.features import RoadAccidentsFeature


@pytest.fixture
def mock_raw_csv_content():
    """Create mock raw CSV content matching the expected format."""
    return """;
;
;
;
;
;
;
DG;Deutschland;369645;290701;78944;2770;364993
01;  Schleswig-Holstein;14645;12382;2263;86;15522
01001;      Flensburg, kreisfreie Stadt;440;390;50;2;467
01002;      Kiel, kreisfreie Stadt;1246;1117;129;4;1321
01003;      L�beck, kreisfreie Stadt, Hansestadt;1317;1167;150;1;1323
01004;      Neum�nster, kreisfreie Stadt;444;389;55;-;500
01051;      Dithmarschen, Kreis;636;508;128;6;681
01051001;        Albersdorf;12;10;2;-;20
01051002;        Arkebek;1;-;1;-;-
01051003;        Averlak;1;1;-;-;1
01051004;        Bargenstedt;5;5;-;-;5
;;;;
;;;;
;;;;
;;;;"""


def test_load_road_accidents_data(mock_raw_csv_content, tmp_path):
    """Test loading of road accidents data from CSV."""
    # Create a temporary CSV file with the mock content
    temp_csv_path = tmp_path / "road_accidents.csv"
    temp_csv_path.write_text(mock_raw_csv_content)

    # Load the data using the RoadAccidentsFeature class
    feature = RoadAccidentsFeature(raw_data_path=str(temp_csv_path))
    df = feature.load()

    # Check columns exist
    assert "AGS" in df.columns
    assert "accident_count" in df.columns
    assert "injury_accidents" in df.columns
    assert "property_damage_accidents" in df.columns
    assert "fatalities" in df.columns
    assert "injured" in df.columns

    # check dtypes
    assert df["AGS"].dtype == object  # AGS should be string
    assert df["accident_count"].dtype in [int, float]
    assert df["injury_accidents"].dtype in [int, float]
    assert df["property_damage_accidents"].dtype in [int, float]
    assert df["fatalities"].dtype in [int, float]
    assert df["injured"].dtype in [int, float]

    # check AGS values are correctly created
    df_ags = df["AGS"].tolist()
    assert "01001000" in df_ags  # Flensburg
    assert "01004000" in df_ags  # Neumünster
    assert "01051000" in df_ags  # Dithmarschen
    assert "01051001" in df_ags  # Albersdorf
