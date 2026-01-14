"""Area lookup API endpoints."""

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from geoscore_de.address.mapy_com import MapyComStructAddressRetriever
from geoscore_de.address.models import Position

router = APIRouter()


class LookupResponse(BaseModel):
    """Response model for area lookup."""

    success: bool
    ags: str | None = None
    metadata: dict | None = None
    error: str | None = None


@router.get("", response_model=LookupResponse)
async def lookup_area(
    latitude: float = Query(..., description="Latitude coordinate"),
    longitude: float = Query(..., description="Longitude coordinate"),
    app_request: Request = None,
):
    """Look up area metadata for GPS coordinates.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        app_request: FastAPI request object for accessing app state

    Returns:
        AGS code and area metadata or error message
    """
    state = app_request.app.state
    settings = state.settings

    try:
        if not hasattr(state, "mapy_com_retriever"):
            state.mapy_com_retriever = MapyComStructAddressRetriever(
                api_key=settings.mapy_com_api_key,
                geojson_path=settings.geojson_path,
            )
        retriever = state.mapy_com_retriever

        position = Position(latitude=latitude, longitude=longitude)
        ags = retriever.get_ags(position)

        if ags is None:
            return LookupResponse(success=False, error=f"No area found for coordinates: {latitude}, {longitude}")

        metadata = retriever.get_area_metadata(latitude, longitude)

        return LookupResponse(success=True, ags=ags, metadata=metadata)

    except Exception as e:
        return LookupResponse(success=False, error=f"Error looking up area: {str(e)}")
