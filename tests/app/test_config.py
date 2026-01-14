"""Tests for application configuration."""

import pytest

from geoscore_de.app.config import Settings


def test_settings_default_values(monkeypatch):
    """Test that settings have correct default values."""
    # Clear any existing env vars
    monkeypatch.delenv("MAPY_COM_API_KEY", raising=False)
    monkeypatch.delenv("GEOJSON_PATH", raising=False)
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("API_RELOAD", raising=False)

    # Set required field
    monkeypatch.setenv("MAPY_COM_API_KEY", "test-key")

    # Disable .env file loading to test only defaults and env vars
    settings = Settings(_env_file=None)

    assert settings.mapy_com_api_key == "test-key"
    assert settings.geojson_path == "data/gemeinden_simplify200.geojson"
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.api_reload is False
    assert settings.app_name == "GeoScore DE"
    assert settings.app_version == "0.1.0"


def test_settings_from_environment_variables(monkeypatch):
    """Test that settings load from environment variables."""
    monkeypatch.setenv("MAPY_COM_API_KEY", "custom-api-key")
    monkeypatch.setenv("GEOJSON_PATH", "custom/path/data.geojson")
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("API_RELOAD", "false")

    # Disable .env file loading
    settings = Settings(_env_file=None)

    assert settings.mapy_com_api_key == "custom-api-key"
    assert settings.geojson_path == "custom/path/data.geojson"
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 9000
    assert settings.api_reload is False


def test_settings_api_key_required(monkeypatch):
    """Test that MAPY_COM_API_KEY is required."""
    # Clear the API key
    monkeypatch.delenv("MAPY_COM_API_KEY", raising=False)

    with pytest.raises(Exception):  # pydantic ValidationError
        Settings(_env_file=None)


def test_settings_api_port_type_validation(monkeypatch):
    """Test that API_PORT must be an integer."""
    monkeypatch.setenv("MAPY_COM_API_KEY", "test-key")
    monkeypatch.setenv("API_PORT", "not-a-number")

    with pytest.raises(Exception):  # pydantic ValidationError
        Settings(_env_file=None)


def test_settings_api_reload_boolean_conversion(monkeypatch):
    """Test that API_RELOAD correctly converts string to boolean."""
    monkeypatch.setenv("MAPY_COM_API_KEY", "test-key")

    # Test various boolean string representations
    monkeypatch.setenv("API_RELOAD", "true")
    settings = Settings()
    assert settings.api_reload is True

    monkeypatch.setenv("API_RELOAD", "false")
    settings = Settings()
    assert settings.api_reload is False

    monkeypatch.setenv("API_RELOAD", "1")
    settings = Settings()
    assert settings.api_reload is True

    monkeypatch.setenv("API_RELOAD", "0")
    settings = Settings()
    assert settings.api_reload is False


def test_settings_loads_from_env_file(tmp_path, monkeypatch):
    """Test that settings can load from .env file."""
    # Create a temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text("MAPY_COM_API_KEY=file-api-key\nGEOJSON_PATH=file/path/data.geojson\nAPI_PORT=7000\n")

    # Change to temp directory
    monkeypatch.chdir(tmp_path)

    # Clear env vars to force loading from file
    monkeypatch.delenv("MAPY_COM_API_KEY", raising=False)
    monkeypatch.delenv("GEOJSON_PATH", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)

    settings = Settings(_env_file=str(env_file))

    assert settings.mapy_com_api_key == "file-api-key"
    assert settings.geojson_path == "file/path/data.geojson"
    assert settings.api_port == 7000


def test_settings_env_vars_override_env_file(tmp_path, monkeypatch):
    """Test that environment variables override .env file values."""
    # Create a temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text("MAPY_COM_API_KEY=file-key\n")

    # Change to temp directory
    monkeypatch.chdir(tmp_path)

    # Set env var that should override file
    monkeypatch.setenv("MAPY_COM_API_KEY", "env-var-key")

    settings = Settings()

    # Environment variable should take precedence
    assert settings.mapy_com_api_key == "env-var-key"


def test_settings_immutability(monkeypatch):
    """Test that settings are immutable after creation."""
    monkeypatch.setenv("MAPY_COM_API_KEY", "test-key")
    settings = Settings(_env_file=None)

    # This tests that we can't accidentally modify settings
    with pytest.raises(Exception):  # AttributeError or ValidationError
        settings.api_port = 9999
