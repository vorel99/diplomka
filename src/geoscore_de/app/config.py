"""Configuration management for the application."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file="configs/.env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # API Keys
    mapy_com_api_key: str = Field(
        ..., env="MAPY_COM_API_KEY", repr=False, description="API key for Mapy.cz geocoding service"
    )

    # Data paths
    geojson_path: str = Field(default="data/gemeinden_simplify200.geojson", env="GEOJSON_PATH")

    # Server settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST", description="Host address for the API server")
    api_port: int = Field(default=8000, env="API_PORT", description="Port number for the API server")
    api_reload: bool = Field(default=False, env="API_RELOAD", description="Enable or disable API server reload")

    # App metadata
    app_name: str = Field(default="GeoScore DE", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
