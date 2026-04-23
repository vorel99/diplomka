"""Tests for OSM-derived feature classes: ATM, Hospitals, Tourist, University."""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from geoscore_de.data_flow.features.osm_atm import OSMATMFeature
from geoscore_de.data_flow.features.osm_hospitals import OSMHospitalsFeature
from geoscore_de.data_flow.features.osm_tourist import OSMTouristFeature
from geoscore_de.data_flow.features.osm_university import OSMUniversityFeature

# ---------------------------------------------------------------------------
# Two non-overlapping polygons used as municipality geometries
# ---------------------------------------------------------------------------
_POLY_A = Polygon([(9.9, 53.9), (10.1, 53.9), (10.1, 54.1), (9.9, 54.1)])
_POLY_B = Polygon([(10.4, 54.4), (10.6, 54.4), (10.6, 54.6), (10.4, 54.6)])

_AGS_A = "01001000"
_AGS_B = "01002000"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def muni_geo_csv(tmp_path) -> str:
    """Temporary municipality GeoShape CSV compatible with load_geo_data()."""
    rows = [
        {
            # "Gemeinde code" slice(0,5)="01001", slice(9,12)="000" → AGS "01001000"
            "Gemeinde code": "01001.00.000",
            "Geo Shape": json.dumps({"type": "Polygon", "coordinates": [list(_POLY_A.exterior.coords)]}),
            "name": "Muni A",
        },
        {
            # "Gemeinde code" slice(0,5)="01002", slice(9,12)="000" → AGS "01002000"
            "Gemeinde code": "01002.00.000",
            "Geo Shape": json.dumps({"type": "Polygon", "coordinates": [list(_POLY_B.exterior.coords)]}),
            "name": "Muni B",
        },
    ]
    path = tmp_path / "muni_geo.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def muni_data_csv(tmp_path) -> str:
    """Temporary municipality data CSV compatible with MunicipalityFeature.load().

    MU_ID "010010000000": slice(0,5)="01001", slice(9,12)="000" → AGS "01001000"
    MU_ID "010020000000": slice(0,5)="01002", slice(9,12)="000" → AGS "01002000"
    """
    content = (
        "Tabelle: 1000A-0001\n"
        "Persons: Official population and area (municipalities);;;;\n"
        "Population in brief (territory on 15 May 2022);;;;\n"
        ";;Persons;Area;Population density\n"
        ";;number;km²;Inh/km²\n"
        "2022-05-15;;;;\n"
        "010010000000;Muni A;1000;100;10\n"
        "010020000000;Muni B;2000;200;10\n"
        ";;;;\n"
        ";;;;\n"
        ";;;;\n"
        ";;;;\n"
    )
    path = tmp_path / "municipalities.csv"
    path.write_text(content)
    return str(path)


def _write_points_geojson(path, points: list[tuple[float, float]], props: list[dict] | None = None) -> str:
    """Write a Point GeoJSON file and return the path as a string."""
    if props is None:
        props = [{} for _ in points]
    gdf = gpd.GeoDataFrame(props, geometry=[Point(lon, lat) for lon, lat in points], crs="EPSG:4326")
    gdf.to_file(str(path), driver="GeoJSON")
    return str(path)


def _write_empty_geojson(path) -> str:
    """Write an empty Point GeoJSON file and return the path as a string."""
    gdf = gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], dtype="geometry")}, crs="EPSG:4326")
    gdf.to_file(str(path), driver="GeoJSON")
    return str(path)


# ===========================================================================
# OSMATMFeature
# ===========================================================================


@pytest.fixture
def atm_geojson(tmp_path) -> str:
    """Two ATMs inside Muni A, one ATM inside Muni B."""
    return _write_points_geojson(
        tmp_path / "atms.geojson",
        [(10.0, 54.0), (10.05, 54.0), (10.5, 54.5)],
    )


@pytest.fixture
def empty_atm_geojson(tmp_path) -> str:
    """Empty ATM GeoJSON (no features)."""
    return _write_empty_geojson(tmp_path / "atms_empty.geojson")


def test_osm_atm_load_returns_geodataframe(atm_geojson):
    """load() returns a GeoDataFrame with the expected number of rows."""
    feature = OSMATMFeature(raw_data_path=atm_geojson)
    gdf = feature.load()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 3


def test_osm_atm_transform_output_columns(tmp_path, atm_geojson, muni_geo_csv, muni_data_csv):
    """transform() produces exactly AGS, atms_per_1000_residents, atm_density_per_km2."""
    feature = OSMATMFeature(
        raw_data_path=atm_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        municipality_data_path=muni_data_csv,
    )
    result = feature.transform(feature.load())
    assert set(result.columns) == {"AGS", "atms_per_1000_residents", "atm_density_per_km2"}


def test_osm_atm_transform_ags_integrity(tmp_path, atm_geojson, muni_geo_csv, muni_data_csv):
    """Output contains exactly the AGS codes present in the municipality geo data."""
    feature = OSMATMFeature(
        raw_data_path=atm_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        municipality_data_path=muni_data_csv,
    )
    result = feature.transform(feature.load())
    assert set(result["AGS"]) == {_AGS_A, _AGS_B}


def test_osm_atm_transform_counts_correctly(tmp_path, atm_geojson, muni_geo_csv, muni_data_csv):
    """ATM normalized features match known counts and municipality stats."""
    feature = OSMATMFeature(
        raw_data_path=atm_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        municipality_data_path=muni_data_csv,
    )
    result = feature.transform(feature.load())
    row_a = result[result["AGS"] == _AGS_A].iloc[0]
    row_b = result[result["AGS"] == _AGS_B].iloc[0]
    # Muni A: 2 ATMs, 1000 residents, 100 km²
    assert row_a["atms_per_1000_residents"] == pytest.approx(2.0, rel=1e-3)
    assert row_a["atm_density_per_km2"] == pytest.approx(0.02, rel=1e-3)
    # Muni B: 1 ATM, 2000 residents, 200 km²
    assert row_b["atms_per_1000_residents"] == pytest.approx(0.5, rel=1e-3)
    assert row_b["atm_density_per_km2"] == pytest.approx(0.005, rel=1e-3)


def test_osm_atm_transform_empty_input(tmp_path, empty_atm_geojson, muni_geo_csv, muni_data_csv):
    """Empty ATM GeoDataFrame produces zero normalized values (not NaN) when population > 0."""
    feature = OSMATMFeature(
        raw_data_path=empty_atm_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        municipality_data_path=muni_data_csv,
    )
    result = feature.transform(feature.load())
    assert (result["atms_per_1000_residents"] == 0).all()
    assert (result["atm_density_per_km2"] == 0).all()


def test_osm_atm_transform_saves_output(tmp_path, atm_geojson, muni_geo_csv, muni_data_csv):
    """transform() writes a CSV to tform_data_path with the correct columns."""
    out_path = tmp_path / "subdir" / "atm_out.csv"
    feature = OSMATMFeature(
        raw_data_path=atm_geojson,
        tform_data_path=str(out_path),
        municipality_geo_data_path=muni_geo_csv,
        municipality_data_path=muni_data_csv,
    )
    feature.transform(feature.load())
    assert out_path.exists()
    saved = pd.read_csv(out_path)
    assert set(saved.columns) == {"AGS", "atms_per_1000_residents", "atm_density_per_km2"}
    assert len(saved) == 2


# ===========================================================================
# OSMHospitalsFeature
# ===========================================================================


@pytest.fixture
def hospital_geojson(tmp_path) -> str:
    """One hospital inside Muni A with a German country tag."""
    return _write_points_geojson(
        tmp_path / "hospitals.geojson",
        [(10.0, 54.0)],
        [{"addr:country": "DE"}],
    )


@pytest.fixture
def hospital_geojson_multi_country(tmp_path) -> str:
    """Hospitals from Germany and Poland."""
    return _write_points_geojson(
        tmp_path / "hospitals_multi.geojson",
        [(10.0, 54.0), (18.0, 52.0)],
        [{"addr:country": "DE"}, {"addr:country": "PL"}],
    )


@pytest.fixture
def empty_hospital_geojson(tmp_path) -> str:
    """Empty hospital GeoJSON (no features)."""
    return _write_empty_geojson(tmp_path / "hospitals_empty.geojson")


def test_osm_hospitals_load_returns_geodataframe(hospital_geojson):
    """load() returns a GeoDataFrame without country filtering when country_column=None."""
    feature = OSMHospitalsFeature(raw_data_path=hospital_geojson, country_column=None)
    gdf = feature.load()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1


def test_osm_hospitals_load_filters_by_country(hospital_geojson_multi_country):
    """load() retains only rows matching the configured country value."""
    feature = OSMHospitalsFeature(
        raw_data_path=hospital_geojson_multi_country,
        country_column="addr:country",
        country_value="DE",
    )
    gdf = feature.load()
    assert len(gdf) == 1
    assert gdf.iloc[0]["addr:country"] == "DE"


def test_osm_hospitals_transform_output_columns(tmp_path, hospital_geojson, muni_geo_csv):
    """transform() produces the four expected distance/count columns plus AGS."""
    feature = OSMHospitalsFeature(
        raw_data_path=hospital_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        country_column=None,
    )
    result = feature.transform(feature.load())
    assert set(result.columns) == {
        "AGS",
        "dist_nearest_hospital_km",
        "dist_mean_3_hospitals_km",
        "hospital_count_10km",
        "hospital_count_30km",
    }


def test_osm_hospitals_transform_distances_in_km(tmp_path, hospital_geojson, muni_geo_csv):
    """Distances are non-negative and within a plausible range (< 1000 km for Germany)."""
    feature = OSMHospitalsFeature(
        raw_data_path=hospital_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        country_column=None,
    )
    result = feature.transform(feature.load())
    assert (result["dist_nearest_hospital_km"] >= 0).all()
    assert (result["dist_nearest_hospital_km"] < 1000).all()
    assert (result["dist_mean_3_hospitals_km"] >= 0).all()


def test_osm_hospitals_transform_empty_input(tmp_path, empty_hospital_geojson, muni_geo_csv):
    """Empty hospitals GDF produces NaN distances and zero radius counts."""
    feature = OSMHospitalsFeature(
        raw_data_path=empty_hospital_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        country_column=None,
    )
    result = feature.transform(feature.load())
    assert result["dist_nearest_hospital_km"].isna().all()
    assert result["dist_mean_3_hospitals_km"].isna().all()
    assert (result["hospital_count_10km"] == 0).all()
    assert (result["hospital_count_30km"] == 0).all()


def test_osm_hospitals_transform_nearby_hospital_counted(tmp_path, muni_geo_csv):
    """A hospital placed inside a municipality polygon is counted in hospital_count_10km."""
    geojson_path = _write_points_geojson(
        tmp_path / "hospitals_close.geojson",
        [(10.0, 54.0)],
    )
    feature = OSMHospitalsFeature(
        raw_data_path=geojson_path,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        country_column=None,
    )
    result = feature.transform(feature.load())
    row_a = result[result["AGS"] == _AGS_A].iloc[0]
    assert row_a["hospital_count_10km"] >= 1


# ===========================================================================
# OSMTouristFeature
# ===========================================================================


@pytest.fixture
def tourist_geojson(tmp_path) -> str:
    """Two POIs inside Muni A, one POI inside Muni B."""
    return _write_points_geojson(
        tmp_path / "tourist.geojson",
        [(10.0, 54.0), (10.05, 54.0), (10.5, 54.5)],
    )


@pytest.fixture
def empty_tourist_geojson(tmp_path) -> str:
    """Empty tourist POI GeoJSON (no features)."""
    return _write_empty_geojson(tmp_path / "tourist_empty.geojson")


def test_osm_tourist_load_returns_geodataframe(tourist_geojson):
    """load() returns a GeoDataFrame with the expected number of rows."""
    feature = OSMTouristFeature(raw_data_path=tourist_geojson)
    gdf = feature.load()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 3


def test_osm_tourist_transform_output_columns(tmp_path, tourist_geojson, muni_geo_csv, muni_data_csv):
    """transform() produces exactly AGS, tourist_poi_per_1000_residents, tourist_poi_per_km2."""
    feature = OSMTouristFeature(
        raw_data_path=tourist_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        municipality_data_path=muni_data_csv,
    )
    result = feature.transform(feature.load())
    assert set(result.columns) == {"AGS", "tourist_poi_per_1000_residents", "tourist_poi_per_km2"}


def test_osm_tourist_transform_counts_correctly(tmp_path, tourist_geojson, muni_geo_csv, muni_data_csv):
    """Per-capita and per-area POI values match known counts and municipality stats."""
    feature = OSMTouristFeature(
        raw_data_path=tourist_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        municipality_data_path=muni_data_csv,
    )
    result = feature.transform(feature.load())
    row_a = result[result["AGS"] == _AGS_A].iloc[0]
    row_b = result[result["AGS"] == _AGS_B].iloc[0]
    # Muni A: 2 POIs, 1000 residents, 100 km²
    assert row_a["tourist_poi_per_1000_residents"] == pytest.approx(2.0, rel=1e-3)
    assert row_a["tourist_poi_per_km2"] == pytest.approx(0.02, rel=1e-3)
    # Muni B: 1 POI, 2000 residents, 200 km²
    assert row_b["tourist_poi_per_1000_residents"] == pytest.approx(0.5, rel=1e-3)
    assert row_b["tourist_poi_per_km2"] == pytest.approx(0.005, rel=1e-3)


def test_osm_tourist_transform_empty_input(tmp_path, empty_tourist_geojson, muni_geo_csv, muni_data_csv):
    """Empty POI GeoDataFrame produces zero normalized values when population > 0."""
    feature = OSMTouristFeature(
        raw_data_path=empty_tourist_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        municipality_data_path=muni_data_csv,
    )
    result = feature.transform(feature.load())
    assert (result["tourist_poi_per_1000_residents"] == 0).all()
    assert (result["tourist_poi_per_km2"] == 0).all()


def test_osm_tourist_transform_values_non_negative(tmp_path, muni_geo_csv, muni_data_csv):
    """Normalized POI values are always non-negative."""
    geojson_path = _write_points_geojson(tmp_path / "t.geojson", [(10.0, 54.0)])
    feature = OSMTouristFeature(
        raw_data_path=geojson_path,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        municipality_data_path=muni_data_csv,
    )
    result = feature.transform(feature.load())
    assert (result["tourist_poi_per_1000_residents"] >= 0).all()
    assert (result["tourist_poi_per_km2"] >= 0).all()


# ===========================================================================
# OSMUniversityFeature
# ===========================================================================


@pytest.fixture
def university_geojson_duplicate_names(tmp_path) -> str:
    """Two universities with the same name (one in each polygon)."""
    return _write_points_geojson(
        tmp_path / "universities_dup.geojson",
        [(10.0, 54.0), (10.5, 54.5)],
        [{"name": "Uni Test"}, {"name": "Uni Test"}],
    )


@pytest.fixture
def university_geojson_unique_names(tmp_path) -> str:
    """Two universities with distinct names."""
    return _write_points_geojson(
        tmp_path / "universities_unique.geojson",
        [(10.0, 54.0), (10.5, 54.5)],
        [{"name": "Uni A"}, {"name": "Uni B"}],
    )


@pytest.fixture
def empty_university_geojson(tmp_path) -> str:
    """Empty university GeoJSON (no features)."""
    return _write_empty_geojson(tmp_path / "universities_empty.geojson")


def test_osm_university_load_returns_geodataframe(university_geojson_unique_names):
    """load() returns a GeoDataFrame with the expected number of rows."""
    feature = OSMUniversityFeature(raw_data_path=university_geojson_unique_names)
    gdf = feature.load()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 2


def test_osm_university_transform_output_columns(tmp_path, university_geojson_unique_names, muni_geo_csv):
    """transform() produces exactly AGS and dist_nearest_university_km."""
    feature = OSMUniversityFeature(
        raw_data_path=university_geojson_unique_names,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
    )
    result = feature.transform(feature.load())
    assert set(result.columns) == {"AGS", "dist_nearest_university_km"}


def test_osm_university_transform_empty_input(tmp_path, empty_university_geojson, muni_geo_csv):
    """Empty university GDF produces NaN distances for all municipalities."""
    feature = OSMUniversityFeature(
        raw_data_path=empty_university_geojson,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
    )
    result = feature.transform(feature.load())
    assert result["dist_nearest_university_km"].isna().all()


def test_osm_university_transform_distances_in_km(tmp_path, university_geojson_unique_names, muni_geo_csv):
    """Distances are non-negative and within a plausible range (< 1000 km)."""
    feature = OSMUniversityFeature(
        raw_data_path=university_geojson_unique_names,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
    )
    result = feature.transform(feature.load())
    assert (result["dist_nearest_university_km"] >= 0).all()
    assert (result["dist_nearest_university_km"] < 1000).all()


def test_osm_university_transform_deduplication_reduces_sources(
    tmp_path, university_geojson_duplicate_names, muni_geo_csv
):
    """With deduplicate_by_name=True, duplicate-named universities collapse to one source point.

    Muni B (near (10.5, 54.5)) has a university placed at its centroid when dedup=False,
    so its nearest-university distance is near zero. After deduplication only the first
    occurrence (at (10.0, 54.0)) remains, so Muni B's distance increases.
    """
    feature_dedup = OSMUniversityFeature(
        raw_data_path=university_geojson_duplicate_names,
        tform_data_path=str(tmp_path / "out_dedup.csv"),
        municipality_geo_data_path=muni_geo_csv,
        deduplicate_by_name=True,
    )
    feature_no_dedup = OSMUniversityFeature(
        raw_data_path=university_geojson_duplicate_names,
        tform_data_path=str(tmp_path / "out_no_dedup.csv"),
        municipality_geo_data_path=muni_geo_csv,
        deduplicate_by_name=False,
    )
    result_dedup = feature_dedup.transform(feature_dedup.load())
    result_no_dedup = feature_no_dedup.transform(feature_no_dedup.load())

    dist_b_dedup = result_dedup[result_dedup["AGS"] == _AGS_B]["dist_nearest_university_km"].iloc[0]
    dist_b_no_dedup = result_no_dedup[result_no_dedup["AGS"] == _AGS_B]["dist_nearest_university_km"].iloc[0]
    assert dist_b_dedup >= dist_b_no_dedup


def test_osm_university_transform_proximity_ordering(tmp_path, muni_geo_csv):
    """Municipality closer to a university has a smaller dist_nearest_university_km."""
    geojson_path = _write_points_geojson(
        tmp_path / "uni_close.geojson",
        [(10.0, 54.0)],
    )
    feature = OSMUniversityFeature(
        raw_data_path=geojson_path,
        tform_data_path=str(tmp_path / "out.csv"),
        municipality_geo_data_path=muni_geo_csv,
        deduplicate_by_name=False,
    )
    result = feature.transform(feature.load())
    dist_a = result[result["AGS"] == _AGS_A]["dist_nearest_university_km"].iloc[0]
    dist_b = result[result["AGS"] == _AGS_B]["dist_nearest_university_km"].iloc[0]
    assert dist_a < dist_b
