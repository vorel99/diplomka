from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.geo import load_geo_data

DEFAULT_RAW_DATA_PATH = "data/raw/osm/hospitals.geojson"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/osm_hospitals.csv"
DEFAULT_MUNICIPALITY_GEO_DATA_PATH = "data/georef-germany-gemeinde.csv"


class OSMHospitalsFeature(BaseFeature):
    """Create municipality-level features from OSM hospitals data."""

    def __init__(
        self,
        raw_data_path: str = DEFAULT_RAW_DATA_PATH,
        tform_data_path: str = DEFAULT_TFORM_DATA_PATH,
        municipality_geo_data_path: str = DEFAULT_MUNICIPALITY_GEO_DATA_PATH,
        country_column: str | None = "addr:country",
        country_value: str | None = "DE",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path
        self.municipality_geo_data_path = municipality_geo_data_path
        self.country_column = country_column
        self.country_value = country_value

    def load(self) -> pd.DataFrame:
        """Load hospitals from GeoJSON and optionally filter by country tag."""
        gdf_hospital = gpd.read_file(self.raw_data_path)

        if self.country_column and self.country_value and self.country_column in gdf_hospital.columns:
            gdf_hospital = gdf_hospital[gdf_hospital[self.country_column] == self.country_value]

        return gdf_hospital

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate hospitals into municipality features using centroid-based proximity."""
        gdf_hospital = gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326")

        gdf_muni = load_geo_data(self.municipality_geo_data_path)[["AGS", "geometry"]]

        # Compute centroids in a metric CRS, then project to WGS84 for haversine distance.
        gdf_muni_metric = gdf_muni.to_crs(epsg=3035)
        muni_points = gpd.GeoSeries(gdf_muni_metric.geometry.centroid, crs="EPSG:3035").to_crs(epsg=4326)
        muni_coords = np.radians(np.column_stack([muni_points.y.to_numpy(), muni_points.x.to_numpy()]))

        result = pd.DataFrame({"AGS": gdf_muni["AGS"].values})

        if gdf_hospital.empty:
            result["dist_nearest_hospital_km"] = np.nan
            result["dist_mean_3_hospitals_km"] = np.nan
            result["hospital_count_10km"] = 0
            result["hospital_count_30km"] = 0
            self._save(result)
            return result

        gdf_hospital_metric = gdf_hospital.to_crs(epsg=3035)
        hosp_points = gpd.GeoSeries(gdf_hospital_metric.geometry.representative_point(), crs="EPSG:3035").to_crs(
            epsg=4326
        )
        hosp_coords = np.radians(np.column_stack([hosp_points.y.to_numpy(), hosp_points.x.to_numpy()]))

        tree = BallTree(hosp_coords, metric="haversine")

        dist_nearest, _ = tree.query(muni_coords, k=1)
        result["dist_nearest_hospital_km"] = dist_nearest[:, 0] * 6371  # Convert radians to kilometers

        k_neighbors = min(3, len(hosp_coords))
        dist_k, _ = tree.query(muni_coords, k=k_neighbors)
        result["dist_mean_3_hospitals_km"] = dist_k.mean(axis=1) * 6371  # Convert radians to kilometers

        idx_10km = tree.query_radius(muni_coords, r=10 / 6371)
        result["hospital_count_10km"] = [len(i) for i in idx_10km]

        idx_30km = tree.query_radius(muni_coords, r=30 / 6371)
        result["hospital_count_30km"] = [len(i) for i in idx_30km]

        self._save(result)
        return result

    def _save(self, df: pd.DataFrame) -> None:
        output_path = Path(self.tform_data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
