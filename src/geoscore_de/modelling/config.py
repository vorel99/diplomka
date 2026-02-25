from pydantic import BaseModel, Field


class FeatureFilteringConfig(BaseModel):
    """Configuration for feature filtering.
    If both omit_features and use_features are provided, first use_features will be applied to select a
    subset of features, and then omit_features will be applied to exclude specific features from that subset.
    """

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


class ModelConfig(BaseModel):
    """Configuration for the model."""

    model_type: str = Field(
        "lightgbm", description="Type of model to use for training (e.g., 'lightgbm', 'random_forest')."
    )
    param_grid: dict = Field(
        default_factory=dict,
        description="Dictionary specifying the hyperparameters and their corresponding values for grid search.",
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    id_column: str = Field("AGS", description="Name of the column containing unique identifiers for each row.")
    target_variable: str = Field(..., description="The target variable for model training.")
    feature_filtering: FeatureFilteringConfig = Field(
        default_factory=FeatureFilteringConfig, description="Configuration for feature filtering."
    )
    row_filtering: RowFilteringConfig = Field(
        default_factory=RowFilteringConfig, description="Configuration for row filtering."
    )
    train_test_split_ratio: float = Field(
        default=0.8, description="Ratio for splitting the dataset into training and testing sets."
    )
    random_state: int = Field(default=42, description="Random state for reproducibility of train-test split.")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Configuration for the model.")
