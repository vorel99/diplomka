from pydantic import BaseModel, Field


class Position(BaseModel):
    latitude: float = Field(..., description="Latitude of the address")
    longitude: float = Field(..., description="Longitude of the address")


class StructAddress(BaseModel):
    name: str = Field(..., description="Full name of the address from the geocoding service")
    street: str = Field(..., description="Street name of the address")
    municipality: str = Field(..., description="Municipality of the address")
    region: str = Field(..., description="Region of the address")
    postal_code: str = Field(..., description="Postal code of the address")
    country: str = Field(..., description="Country of the address")
    country_code: str = Field(..., description="Country code of the address")
    position: Position = Field(..., description="Geographical position of the address")
