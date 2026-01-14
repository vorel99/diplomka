"""API routes package."""

from fastapi import APIRouter

from geoscore_de.app.api.routes import geocode, lookup

api_router = APIRouter(tags=["api"])

# Include sub-routers
api_router.include_router(geocode.router, prefix="/geocode", tags=["geocode"])
api_router.include_router(lookup.router, prefix="/lookup", tags=["lookup"])

__all__ = ["api_router"]
