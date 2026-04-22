from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.geo import load_geo_data

DEFAULT_RAW_DATA_PATH = "data/raw/osm/universities.geojson"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/osm_university.csv"
DEFAULT_MUNICIPALITY_GEO_DATA_PATH = "data/georef-germany-gemeinde.csv"


class OSMUniversityFeature(BaseFeature):
    """Create municipality-level nearest-university feature from OSM data."""

    def __init__(
        self,
        raw_data_path: str = DEFAULT_RAW_DATA_PATH,
        tform_data_path: str = DEFAULT_TFORM_DATA_PATH,
        municipality_geo_data_path: str = DEFAULT_MUNICIPALITY_GEO_DATA_PATH,
        deduplicate_by_name: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path
        self.municipality_geo_data_path = municipality_geo_data_path
        self.deduplicate_by_name = deduplicate_by_name

    def load(self) -> pd.DataFrame:
        """Load universities from OSM GeoJSON."""
        return gpd.read_file(self.raw_data_path)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute distance from municipality centroid to nearest university in kilometers.
        Create features:
        - dist_nearest_university_km: Distance to nearest university in kilometers.
        """
        gdf_uni = gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326")

        if self.deduplicate_by_name and "name" in gdf_uni.columns:
            gdf_uni = gdf_uni[gdf_uni["name"].notna()]
            gdf_uni = gdf_uni.drop_duplicates(subset="name")

        gdf_muni = load_geo_data(self.municipality_geo_data_path)[["AGS", "geometry"]]

        # Compute centroids in a metric CRS, then project to WGS84 for haversine distance.
        gdf_muni_metric = gdf_muni.to_crs(epsg=3035)
        muni_points = gpd.GeoSeries(gdf_muni_metric.geometry.centroid, crs="EPSG:3035").to_crs(epsg=4326)
        muni_coords = np.radians(np.column_stack([muni_points.y.to_numpy(), muni_points.x.to_numpy()]))

        result = pd.DataFrame({"AGS": gdf_muni["AGS"].values})

        if gdf_uni.empty:
            result["dist_nearest_university_km"] = np.nan
            self._save(result)
            return result

        gdf_uni_metric = gdf_uni.to_crs(epsg=3035)
        uni_points = gpd.GeoSeries(gdf_uni_metric.geometry.representative_point(), crs="EPSG:3035").to_crs(epsg=4326)
        uni_coords = np.radians(np.column_stack([uni_points.y.to_numpy(), uni_points.x.to_numpy()]))

        tree = BallTree(uni_coords, metric="haversine")
        dist_nearest, _ = tree.query(muni_coords, k=1)
        result["dist_nearest_university_km"] = dist_nearest[:, 0] * 6371

        self._save(result)
        return result

    def _save(self, df: pd.DataFrame) -> None:
        output_path = Path(self.tform_data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
