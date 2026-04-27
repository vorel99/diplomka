"""Pydantic models for feature configuration."""

from __future__ import annotations

from pydantic import Field

from geoscore_de.config import FeatureFilteringConfig
from geoscore_de.data_flow.feature_engineering.config import BASE_MODULE, ComponentConfig, FeatureEngineeringConfig

__all__ = ["FeatureFilteringConfig"]


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
    column_filter: FeatureFilteringConfig | None = Field(
        default=None,
        description=(
            "Optional column filtering applied to this feature's data at matrix build time. "
            "Use use_features to whitelist columns and omit_features to blacklist columns."
        ),
    )


class MunicipalitiesConfig(FeatureConfig):
    """Configuration for municipalities reference data.
    This is a special case because the municipalities data is required for building the feature matrix,
    so it has its own configuration section separate from the other features.
    This data will be loaded first and used as the base dataframe to which other features will be joined.
    """

    name: str = Field("municipalities", description="Unique name for the municipalities reference data")
