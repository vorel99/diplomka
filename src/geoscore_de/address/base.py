from abc import ABCMeta, abstractmethod

import geopandas as gpd

from geoscore_de.address.models import Position, StructAddress


class BaseStructAddressRetriever(metaclass=ABCMeta):
    def __init__(self, geojson_path: str = "data/gemeinden_simplify200.geojson") -> None:
        self.geojson = gpd.read_file(geojson_path)

    @abstractmethod
    def _get_struct_address(self, raw_address: str) -> StructAddress | None:
        """Get structured address from raw address string.

        Args:
            raw_address (str): Raw address string.

        Returns:
            StructAddress | None: Structured address or None if not found.
        """
        pass

    def get_struct_address(self, raw_address: str) -> StructAddress | None:
        """Get structured address from raw address string.

        Args:
            raw_address (str): Raw address string.

        Returns:
            StructAddress | None: Structured address or None if not found.
        """
        struct_address = self._get_struct_address(raw_address)
        if struct_address is None:
            return None

        ags_code = self.get_ags(struct_address.position)
        struct_address.AGS = ags_code
        return struct_address

    def get_ags(self, position: Position) -> str | None:
        """Get AGS code from raw address string.

        Args:
            position (Position): Position with latitude and longitude.
        Returns:
            str | None: AGS code or None if not found.
        """
        # based on gps coordinates of the address find the AGS code from the geojson file
        result = self.get_area_metadata(latitude=position.latitude, longitude=position.longitude)

        if result is None or result.get("AGS") is None:
            return None

        return result.get("AGS")

    def get_area_metadata(self, latitude: float, longitude: float) -> dict | None:
        """Get metadata for the area containing the given GPS coordinates.

        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate

        Returns:
            dict | None: Dictionary with area metadata or None if not found.
        """
        # Create a point from coordinates
        point = gpd.points_from_xy([longitude], [latitude])
        point_gdf = gpd.GeoDataFrame(geometry=point, crs="EPSG:4326")
        point_gdf = point_gdf.to_crs(self.geojson.crs)

        # Perform spatial join to find which polygon contains the point
        result = gpd.sjoin(point_gdf, self.geojson, how="left", predicate="within")

        if result.empty:
            return None

        # Get the first matching area and return all its properties as a dictionary
        area_data = result.iloc[0].to_dict()

        # Remove geometry and index columns that are not useful metadata
        area_data.pop("geometry", None)
        area_data.pop("index_right", None)

        return area_data
