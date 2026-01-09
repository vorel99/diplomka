import logging

import requests

from geoscore_de.address.base import BaseStructAddressRetriever
from geoscore_de.address.models import Position, StructAddress

logger = logging.getLogger(__name__)


class MapyComStructAddressRetriever(BaseStructAddressRetriever):
    """Retriever for structured addresses using the mapy.com API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mapy.com/v1/geocode"

    def get_struct_address(self, raw_address: str) -> StructAddress | None:
        params = {
            "query": raw_address,
            "lang": "en",
            "limit": 5,
            "type": "regional.address",
            "apikey": self.api_key,
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=5)
        except requests.RequestException:
            logger.error("Error while connecting to mapy.com API", exc_info=True)
            return None

        if response.status_code != 200:
            logger.error(f"mapy.com API returned status code {response.status_code}")
            return None

        data = response.json()
        items = data.get("items", [])
        if not items:
            logger.info(f"No results found for address: {raw_address}")
            return None

        item: dict = items[0]
        position_data = item.get("position", {})
        if "lat" not in position_data or "lon" not in position_data:
            logger.error("Position data is incomplete in the API response")
            return None
        position = Position(latitude=position_data["lat"], longitude=position_data["lon"])
        name = item.get("name", "")
        street = ""
        municipality = ""
        region = ""
        postal_code = item.get("zip", "")
        country = ""
        country_code = ""
        for struct in item.get("regionalStructure", []):
            struct_type = struct.get("type")
            struct_name = struct.get("name")

            if struct_type == "regional.street":
                street = struct_name
            elif struct_type == "regional.municipality":
                municipality = struct_name
            elif struct_type == "regional.region":
                region = struct_name
            elif struct["type"] == "regional.country":
                country = struct["name"]
                country_code = struct.get("isoCode", "")
        struct_address = StructAddress(
            name=name,
            street=street,
            municipality=municipality,
            region=region,
            postal_code=postal_code,
            country=country,
            country_code=country_code,
            position=position,
        )
        return struct_address
