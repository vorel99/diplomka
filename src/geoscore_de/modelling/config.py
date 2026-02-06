from pydantic import BaseModel, Field


class FeatureFilteringConfig(BaseModel):
    """Configuration for feature filtering."""

    omit_features: list[str] = Field(
        default_factory=list, description="List of features to omit from the dataset during model training."
    )
    use_features: list[str] = Field(
        default_factory=list,
        description="List of features to use for model training. If empty, all features will be used.",
    )


class RowFilteringConfig(BaseModel):
    """Configuration for row filtering."""

    omit_land: list[str] = Field(
        default_factory=list, description="List of 'Land' values to omit from the dataset during model training."
    )


class ModelTrainingConfig(BaseModel):
    """Configuration for model training."""

    feature_filtering: FeatureFilteringConfig = Field(
        default_factory=FeatureFilteringConfig, description="Configuration for feature filtering."
    )
    row_filtering: RowFilteringConfig = Field(
        default_factory=RowFilteringConfig, description="Configuration for row filtering."
    )
    target_variable: str = Field(..., description="The target variable for model training.")
