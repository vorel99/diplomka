"""Pydantic models for feature configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from geoscore_de.data_flow.feature_engineering.config import (
    BASE_MODULE,
    ComponentConfig,
    FeatureEngineeringConfig,
)


class FeatureConfig(ComponentConfig):
    module: str = Field(
        default=f"{BASE_MODULE}.features",
        description=(
            "Module path where the feature class is located. Defaults to the features submodule of the base module."
        ),
    )
    before_transforms: list[FeatureEngineeringConfig] = Field(
        default_factory=list, description="Transformations on raw data before this feature's transformation"
    )


class MunicipalitiesConfig(FeatureConfig):
    """Configuration for municipalities reference data.
    This is a special case because the municipalities data is required for building the feature matrix,
    so it has its own configuration section separate from the other features.
    This data will be loaded first and used as the base dataframe to which other features will be joined.
    """

    name: str = Field("municipalities", description="Unique name for the municipalities reference data")


class MatrixConfig(BaseModel):
    """Configuration for feature matrix building."""

    join_key: str = Field(default="AGS", description="Column name to join features on")
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
    after_transforms: list[FeatureEngineeringConfig] = Field(
        default_factory=list, description="Transformations on transformed features (delta features)"
    )
    standalone_transforms: list[FeatureEngineeringConfig] = Field(
        default_factory=list, description="Standalone feature engineering transformations"
    )
    matrix: MatrixConfig = Field(default_factory=MatrixConfig, description="Matrix building configuration")
