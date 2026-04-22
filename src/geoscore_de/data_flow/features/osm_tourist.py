from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.municipality import MunicipalityFeature
from geoscore_de.data_flow.geo import load_geo_data

DEFAULT_RAW_DATA_PATH = "data/raw/osm/tourism.geojson"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/osm_tourist.csv"
DEFAULT_MUNICIPALITY_GEO_DATA_PATH = "data/georef-germany-gemeinde.csv"
DEFAULT_MUNICIPALITY_DATA_PATH = "data/raw/municipalities_2022.csv"


class OSMTouristFeature(BaseFeature):
    """Create municipality-level features from OSM tourist and historic POI data."""

    def __init__(
        self,
        raw_data_path: str = DEFAULT_RAW_DATA_PATH,
        tform_data_path: str = DEFAULT_TFORM_DATA_PATH,
        municipality_geo_data_path: str = DEFAULT_MUNICIPALITY_GEO_DATA_PATH,
        municipality_data_path: str = DEFAULT_MUNICIPALITY_DATA_PATH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_path = raw_data_path
        self.tform_data_path = tform_data_path
        self.municipality_geo_data_path = municipality_geo_data_path
        self.municipality_data_path = municipality_data_path

    def load(self) -> pd.DataFrame:
        """Load tourist/historic POIs from GeoJSON."""
        return gpd.read_file(self.raw_data_path)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate POI counts within municipality borders and derive normalized indicators.
        Create features:
        - tourist_poi_per_1000_residents: Number of tourist/historic POIs per 1000 residents.
        - tourist_poi_per_km2: Number of tourist/historic POIs per square kilometer, using municipality area data.
        """
        gdf_tourist = gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326")
        gdf_muni_geo = load_geo_data(self.municipality_geo_data_path)[["AGS", "geometry"]]

        municipality_df = MunicipalityFeature(self.municipality_data_path).load()[["AGS", "Persons", "Area"]]
        municipality_df["Persons"] = pd.to_numeric(municipality_df["Persons"], errors="coerce")
        municipality_df["Area"] = pd.to_numeric(municipality_df["Area"], errors="coerce")

        gdf_muni = gdf_muni_geo.merge(municipality_df, on="AGS", how="left")

        if gdf_tourist.crs != gdf_muni.crs:
            gdf_tourist = gdf_tourist.to_crs(gdf_muni.crs)

        # Use representative points so mixed OSM geometries (points/polygons/lines) are handled consistently.
        gdf_tourist_points = gdf_tourist[["geometry"]].copy()
        gdf_tourist_points["geometry"] = gdf_tourist_points.geometry.representative_point()

        result = gdf_muni[["AGS", "Persons", "Area"]].copy()

        if gdf_tourist_points.empty:
            result["tourist_poi_count"] = 0
        else:
            joined = gpd.sjoin(
                gdf_tourist_points,
                gdf_muni[["AGS", "geometry"]],
                how="left",
                predicate="within",
            )
            counts = joined.groupby("AGS").size().reset_index(name="tourist_poi_count")
            result = result.merge(counts, on="AGS", how="left")
            result["tourist_poi_count"] = result["tourist_poi_count"].fillna(0).astype(int)

        result["tourist_poi_per_1000_residents"] = np.where(
            result["Persons"] > 0,
            (result["tourist_poi_count"] / result["Persons"]) * 1000,
            np.nan,
        )
        result["tourist_poi_per_km2"] = np.where(
            result["Area"] > 0,
            result["tourist_poi_count"] / result["Area"],
            np.nan,
        )

        result = result[["AGS", "tourist_poi_per_1000_residents", "tourist_poi_per_km2"]]
        self._save(result)
        return result

    def _save(self, df: pd.DataFrame) -> None:
        output_path = Path(self.tform_data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
