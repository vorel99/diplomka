"""Pydantic models for feature configuration."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class BaseComponentConfig(BaseModel):
    """Base configuration for loadable components."""

    class_name: str = Field(..., alias="class", description="Name of the component class")
    module: str = Field(..., description="Module path where the component class is located")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the component class")

    model_config = {"populate_by_name": True}


class FeatureConfig(BaseComponentConfig):
    """Configuration for a single feature."""

    name: str = Field(..., description="Unique name for the feature")
    enabled: bool = Field(default=True, description="Whether this feature is enabled")


class MunicipalitiesConfig(BaseComponentConfig):
    """Configuration for municipalities reference data."""


class MatrixConfig(BaseModel):
    """Configuration for feature matrix building."""

    join_key: str = Field(default="AGS", description="Column name to join features on")
    join_method: Literal["inner", "outer", "left", "right"] = Field(
        default="inner", description="How to join feature dataframes"
    )
    save_output: bool = Field(default=True, description="Whether to save the feature matrix to a file")
    output_path: str = Field(
        default="data/final/feature_matrix.csv", description="Path to save the final feature matrix"
    )
    missing_values: Literal["drop", "fill"] | None = Field(
        default=None, description="Strategy for handling missing values: 'drop', 'fill', or None"
    )
    fill_value: int | float | str = Field(default=0, description="Value to use when filling missing values")


class FeaturesYAMLConfig(BaseModel):
    """Root configuration model for features.yaml."""

    municipalities: MunicipalitiesConfig = Field(..., description="Configuration for municipalities reference data")
    features: list[FeatureConfig] = Field(default_factory=list, description="List of feature configurations")
    matrix: MatrixConfig = Field(default_factory=MatrixConfig, description="Matrix building configuration")
