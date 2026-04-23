from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geoscore_de.data_flow.features.base import BaseFeature
from geoscore_de.data_flow.features.municipality import MunicipalityFeature
from geoscore_de.data_flow.geo import load_geo_data

DEFAULT_RAW_DATA_PATH = "data/raw/osm/atms_full.geojson"
DEFAULT_TFORM_DATA_PATH = "data/tform/features/osm_atm.csv"
DEFAULT_MUNICIPALITY_GEO_DATA_PATH = "data/georef-germany-gemeinde.csv"
DEFAULT_MUNICIPALITY_DATA_PATH = "data/raw/municipalities_2022.csv"


class OSMATMFeature(BaseFeature):
    """Create municipality-level features from OSM ATM data."""

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
        """Load ATM points from the configured GeoJSON file."""
        gdf_atm = gpd.read_file(self.raw_data_path)

        return gdf_atm

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ATM counts by municipality polygons and derive normalized ATM indicators.
        Create features:
        - atms_per_1000_residents: Number of ATMs per 1000 residents, using municipality population data.
        - atm_density_per_km2: Number of ATMs per square kilometer, using municipality area data.
        """
        gdf_atm = gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326")
        gdf_muni_geo = load_geo_data(self.municipality_geo_data_path)[["AGS", "geometry"]]

        municipality_df = MunicipalityFeature(self.municipality_data_path).load()[["AGS", "Persons", "Area"]]
        municipality_df["Persons"] = pd.to_numeric(municipality_df["Persons"], errors="coerce")
        municipality_df["Area"] = pd.to_numeric(municipality_df["Area"], errors="coerce")

        gdf_muni = gdf_muni_geo.merge(municipality_df, on="AGS", how="left")

        if gdf_atm.crs != gdf_muni.crs:
            gdf_atm = gdf_atm.to_crs(gdf_muni.crs)

        # Use representative points to support mixed ATM geometries (points/polygons).
        gdf_atm_points = gdf_atm[["geometry"]].copy()
        gdf_atm_points["geometry"] = gdf_atm_points.geometry.representative_point()

        result = gdf_muni[["AGS", "Persons", "Area"]].copy()

        if gdf_atm_points.empty:
            result["atm_count_within"] = 0
        else:
            joined = gpd.sjoin(
                gdf_atm_points,
                gdf_muni[["AGS", "geometry"]],
                how="left",
                predicate="within",
            )
            counts = joined.groupby("AGS").size().reset_index(name="atm_count_within")
            result = result.merge(counts, on="AGS", how="left")
            result["atm_count_within"] = result["atm_count_within"].fillna(0).astype(int)

        result["atms_per_1000_residents"] = np.where(
            result["Persons"] > 0, (result["atm_count_within"] / result["Persons"]) * 1000, np.nan
        )
        result["atm_density_per_km2"] = np.where(
            result["Area"] > 0, result["atm_count_within"] / result["Area"], np.nan
        )

        result = result[["AGS", "atms_per_1000_residents", "atm_density_per_km2"]]

        self._save(result)
        return result

    def _save(self, df: pd.DataFrame) -> None:
        output_path = Path(self.tform_data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
