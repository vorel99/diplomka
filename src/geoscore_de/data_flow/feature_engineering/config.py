from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Get base module path (geoscore_de.data_flow)
BASE_MODULE = ".".join(__name__.split(".")[:-2])


class ComponentConfig(BaseModel):
    """Configuration for a dynamically loaded component (e.g., feature or transformation).
    This model captures the necessary information to dynamically import and instantiate a class based on configuration.
    """

    name: str = Field(..., description="Unique name for the feature")
    class_name: str = Field(..., validation_alias="class", description="Name of the feature class")
    module: str = Field(..., description="Module path where the component class is located")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the feature class")

    model_config = {"populate_by_name": True}


class FeatureEngineeringConfig(ComponentConfig):
    """Configuration for feature engineering transformations.
    Takes multiple columns as input and produces one or more columns as output.
    """

    module: str = Field(
        default=f"{BASE_MODULE}.feature_engineering",
        description=(
            "Module path where the feature engineering class is located. "
            "Defaults to the feature_engineering submodule of the base module."
        ),
    )
    input_columns: list[str] = Field(
        default_factory=list, description="List of input column names to use for transformation"
    )
    output_column: str = Field(
        ...,
        description=(
            "Name of the output column produced by this transformation"
            "(or prefix for multiple output columns if transformation produces more than one feature)"
        ),
    )
