"""Helpers for loading geospatial data into GeoDataFrames."""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

DEFAULT_MUNICIPALITY_GEO_DATA_PATH = "data/georef-germany-gemeinde.csv"


def load_geo_data(path: str = DEFAULT_MUNICIPALITY_GEO_DATA_PATH) -> gpd.GeoDataFrame:
    """Load geospatial data from a file into a GeoDataFrame.
    Supported format is csv from OpenDataSoft
    """
    ods_gdf_mun = gpd.read_file(path)
    ods_gdf_mun["geometry"] = ods_gdf_mun["Geo Shape"].apply(lambda x: shape(json.loads(x)) if pd.notnull(x) else None)
    ods_gdf_mun = ods_gdf_mun.drop(columns=["Geo Shape"])
    ods_gdf_mun = gpd.GeoDataFrame(ods_gdf_mun, geometry="geometry", crs="EPSG:4326")
    # add AGS column
    ods_gdf_mun["AGS"] = ods_gdf_mun["Gemeinde code"].str.slice(0, 5) + ods_gdf_mun["Gemeinde code"].str.slice(9, 12)
    return ods_gdf_mun
