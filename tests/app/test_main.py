"""Tests for FastAPI application factory."""

import pytest
from fastapi.testclient import TestClient

from geoscore_de.app.config import Settings
from geoscore_de.app.main import create_app


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(
        _env_file=None,
        **{
            "mapy_com_api_key": "test-api-key",
            "geojson_path": "test/path/data.geojson",
            "app_name": "Test App",
            "app_version": "1.0.0",
        },
    )


def test_create_app_with_settings(test_settings):
    """Test creating app with provided settings."""
    app = create_app(settings=test_settings)

    assert app is not None
    assert app.title == "Test App"
    assert app.version == "1.0.0"
    assert app.state.settings == test_settings
    assert app.state.settings.mapy_com_api_key == "test-api-key"


def test_create_app_without_settings(monkeypatch):
    """Test creating app without settings loads from environment."""
    monkeypatch.setenv("MAPY_COM_API_KEY", "env-api-key")

    app = create_app(settings=None)

    assert app is not None
    assert app.state.settings is not None
    assert app.state.settings.mapy_com_api_key == "env-api-key"


def test_create_app_configuration(test_settings):
    """Test app configuration properties."""
    app = create_app(settings=test_settings)

    assert app.title == "Test App"
    assert app.version == "1.0.0"
    assert "Geocoding and area lookup" in app.description
    assert app.docs_url == "/api/docs"
    assert app.redoc_url == "/api/redoc"
    assert app.openapi_url == "/api/openapi.json"


def test_openapi_docs_available(test_settings):
    """Test that OpenAPI documentation is available."""
    app = create_app(settings=test_settings)
    client = TestClient(app)

    # Check OpenAPI JSON
    response = client.get("/api/openapi.json")
    assert response.status_code == 200
    openapi_spec = response.json()
    assert "openapi" in openapi_spec
    assert "paths" in openapi_spec

    # Check Swagger UI docs
    response = client.get("/api/docs")
    assert response.status_code == 200

    # Check ReDoc
    response = client.get("/api/redoc")
    assert response.status_code == 200


def test_app_state_settings_accessible(test_settings):
    """Test that settings are accessible through app state."""
    app = create_app(settings=test_settings)

    assert hasattr(app.state, "settings")
    assert app.state.settings.mapy_com_api_key == "test-api-key"
    assert app.state.settings.geojson_path == "test/path/data.geojson"


def test_multiple_app_instances_isolated():
    """Test that multiple app instances have isolated settings."""
    settings1 = Settings(
        _env_file=None,
        **{
            "mapy_com_api_key": "key1",
            "geojson_path": "path1.geojson",
        },
    )

    settings2 = Settings(
        _env_file=None,
        **{
            "mapy_com_api_key": "key2",
            "geojson_path": "path2.geojson",
        },
    )

    app1 = create_app(settings=settings1)
    app2 = create_app(settings=settings2)

    assert app1.state.settings.mapy_com_api_key == "key1"
    assert app2.state.settings.mapy_com_api_key == "key2"
    assert app1.state.settings != app2.state.settings
