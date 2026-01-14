"""Geocoding API endpoints."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from geoscore_de.address.mapy_com import MapyComStructAddressRetriever
from geoscore_de.address.models import StructAddress
from geoscore_de.app.config import Settings

router = APIRouter()


class GeocodeRequest(BaseModel):
    """Request model for geocoding."""

    address: str


class GeocodeResponse(BaseModel):
    """Response model for geocoding."""

    success: bool
    address: StructAddress | None = None
    error: str | None = None


@router.post("", response_model=GeocodeResponse)
async def geocode_address(request: GeocodeRequest, app_request: Request):
    """Geocode an address string and return structured address with AGS code.

    Args:
        request: Geocoding request with address string
        app_request: FastAPI request object for accessing app state

    Returns:
        Structured address with AGS code or error message
    """
    settings: Settings = app_request.app.state.settings

    try:
        retriever = MapyComStructAddressRetriever(api_key=settings.mapy_com_api_key, geojson_path=settings.geojson_path)

        result = retriever.get_struct_address(request.address)

        if result is None:
            return GeocodeResponse(success=False, error=f"Could not geocode address: {request.address}")

        return GeocodeResponse(success=True, address=result)

    except Exception as e:
        return GeocodeResponse(success=False, error=f"Error geocoding address: {str(e)}")
