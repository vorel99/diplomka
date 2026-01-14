"""Application middleware and startup/shutdown handlers."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from geoscore_de.address.mapy_com import MapyComStructAddressRetriever
from geoscore_de.app.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events.

    Args:
        app: FastAPI application instance

    Yields:
        Control to the application during its lifetime
    """
    # Startup: Initialize retriever
    settings: Settings = app.state.settings

    app.state.mapy_com_retriever = MapyComStructAddressRetriever(
        api_key=settings.mapy_com_api_key,
        geojson_path=settings.geojson_path,
    )

    yield
