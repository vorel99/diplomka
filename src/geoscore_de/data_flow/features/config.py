"""Pydantic models for feature configuration."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering transformations.
    Takes multiple columns as input and produces one column as output (e.g. municipality homogeneity).
    """

    name: str = Field(..., description="Unique name for the transformation")
    class_name: str = Field(..., alias="class", description="Name of the transformation class")
    module: str = Field(..., description="Module path where the transformation class is located")
    input_columns: list[str] = Field([], description="List of input column names to use for transformation")
    output_column: str = Field(..., description="Name of the output column")
    params: dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the transformation")


class FeatureConfig(BaseModel):
    """Configuration for a single feature."""

    name: str = Field(..., description="Unique name for the feature")
    class_name: str = Field(..., alias="class", description="Name of the feature class")
    module: str = Field(..., description="Module path where the feature class is located")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the feature class")

    model_config = {"populate_by_name": True}


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
    before_transforms: list[FeatureEngineeringConfig] = Field(
        default_factory=list, description="Transformations on raw data before transformation"
    )
    after_transforms: list[FeatureEngineeringConfig] = Field(
        default_factory=list, description="Transformations on transformed features (delta features)"
    )
    standalone_transforms: list[FeatureEngineeringConfig] = Field(
        default_factory=list, description="Standalone feature engineering transformations"
    )
    matrix: MatrixConfig = Field(default_factory=MatrixConfig, description="Matrix building configuration")
