"""Main entry point for running the FastAPI application.

Run with: python -m geoscore_de.app
"""

import uvicorn

from geoscore_de.app.config import Settings


def main():
    """Run the FastAPI application."""
    settings = Settings()

    uvicorn.run(
        "geoscore_de.app.main:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()
