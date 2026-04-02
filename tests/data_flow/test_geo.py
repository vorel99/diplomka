"""Tests for geospatial data loading helpers."""

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from geoscore_de.data_flow.geo import load_geo_data


def test_load_geo_data_from_geojson(tmp_path):
    csv_path = tmp_path / "municipalities.csv"
    geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    geo_shape = json.dumps({"type": "Polygon", "coordinates": [list(geometry.exterior.coords)]})
    pd.DataFrame(
        {
            "Gemeinde code": ["12345.6.789"],
            "Geo Shape": [geo_shape],
            "name": ["Test Municipality"],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_geo_data(str(csv_path))

    assert isinstance(loaded, gpd.GeoDataFrame)
    assert list(loaded["AGS"]) == ["1234589"]
    assert loaded.crs.to_string() == "EPSG:4326"


def test_load_geo_data_from_csv_geo_shape(tmp_path):
    csv_path = tmp_path / "municipalities.csv"
    geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    geo_shape = json.dumps({"type": "Polygon", "coordinates": [list(geometry.exterior.coords)]})
    pd.DataFrame(
        {
            "Gemeinde code": ["01051.0.003"],
            "Geo Shape": [geo_shape],
            "name": ["Test Municipality"],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_geo_data(str(csv_path))

    assert isinstance(loaded, gpd.GeoDataFrame)
    assert loaded.loc[0, "name"] == "Test Municipality"
    assert loaded.loc[0, "AGS"] == "0105103"
    assert loaded.loc[0].geometry.geom_type == "Polygon"


def test_load_geo_data_rejects_csv_without_geometry(tmp_path):
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("Gemeinde code,name\n01051.0.003,Test Municipality\n")

    with pytest.raises(KeyError, match="Geo Shape"):
        load_geo_data(str(csv_path))
