"""FastAPI application factory."""

from fastapi import FastAPI

from geoscore_de.app.api.routes import api_router
from geoscore_de.app.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings. If None, loaded from environment.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title=settings.app_name,
        description="Geocoding and area lookup for German addresses with web interface",
        version=settings.app_version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Store settings in app state
    app.state.settings = settings

    # Include routers
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "version": settings.app_version}

    return app
