"""Geocoding API endpoints."""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from geoscore_de.address.mapy_com import MapyComStructAddressRetriever
from geoscore_de.address.models import StructAddress
from geoscore_de.app.config import Settings

router = APIRouter()


class GeocodeRequest(BaseModel):
    """Request model for geocoding."""

    address: str = Field(..., min_length=1, description="Address string to geocode")


class GeocodeResponse(BaseModel):
    """Response model for geocoding."""

    success: bool = Field(..., description="Indicates if geocoding was successful")
    address: StructAddress | None = Field(
        None, description="Structured address with AGS code if geocoding was successful"
    )
    error: str | None = Field(None, description="Error message if geocoding was not successful")


@router.post("", response_model=GeocodeResponse)
async def geocode_address(request: GeocodeRequest, app_request: Request):
    """Geocode an address string and return structured address with AGS code.

    Args:
        request: Geocoding request with address string
        app_request: FastAPI request object for accessing app state

    Returns:
        Structured address with AGS code or error message
    """
    state = app_request.app.state
    settings: Settings = state.settings

    try:
        if not hasattr(state, "mapy_com_retriever"):
            state.mapy_com_retriever = MapyComStructAddressRetriever(
                api_key=settings.mapy_com_api_key,
                geojson_path=settings.geojson_path,
            )
        retriever = state.mapy_com_retriever

        result = retriever.get_struct_address(request.address)

        if result is None:
            return GeocodeResponse(success=False, error=f"Could not geocode address: {request.address}")

        return GeocodeResponse(success=True, address=result)

    except Exception as e:
        return GeocodeResponse(success=False, error=f"Error geocoding address: {str(e)}")
