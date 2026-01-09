import requests

from geoscore_de.address.base import BaseStructAddressRetriever
from geoscore_de.address.models import Position, StructAddress


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
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            return None

        data = response.json()
        items = data.get("items", [])
        if not items:
            return None

        item: dict = items[0]
        position = Position(latitude=item["position"]["lat"], longitude=item["position"]["lon"])
        name = item.get("name", "")
        street = ""
        municipality = ""
        region = ""
        postal_code = item.get("zip", "")
        country = ""
        country_code = ""
        for struct in item.get("regionalStructure", []):
            if struct["type"] == "regional.street":
                street = struct["name"]
            elif struct["type"] == "regional.municipality":
                municipality = struct["name"]
            elif struct["type"] == "regional.region":
                region = struct["name"]
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
