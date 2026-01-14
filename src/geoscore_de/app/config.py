"""Configuration management for the application."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    mapy_com_api_key: str = Field(default="", env="MAPY_COM_API_KEY")

    # Data paths
    geojson_path: str = Field(default="data/gemeinden_simplify200.geojson", env="GEOJSON_PATH")

    # App paths
    base_dir: Path = Path(__file__).parent.parent.parent.parent
    templates_dir: Path = Path(__file__).parent / "templates"
    static_dir: Path = Path(__file__).parent / "static"

    # Server settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")

    # App metadata
    app_name: str = "GeoScore DE"
    app_version: str = "0.1.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
