from typing import Literal

from pydantic import BaseModel, Field, model_validator

from geoscore_de.config import FeatureFilteringConfig

__all__ = ["FeatureFilteringConfig"]


class RowFilteringConfig(BaseModel):
    """Configuration for row filtering.

    Rows are filtered by matching column values against patterns.
    Patterns support glob-style wildcards (e.g. '02*' matches '0212', '023', etc.)
    and full regular expressions.
    """

    omit_rows: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Dict mapping column names to lists of value patterns. "
            "Rows where the column value matches any pattern are omitted."
        ),
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
    split_strategy: Literal["random", "stratified_federal_state"] = Field(
        default="random",
        description=(
            "Train/test split strategy. Use 'random' for plain random split or "
            "'stratified_federal_state' to keep federal state proportions in both sets."
        ),
    )
    federal_state_column: str = Field(
        default="federal_republic_id",
        description="Column used for state-level stratification when split_strategy is 'stratified_federal_state'.",
    )
    random_state: int = Field(default=42, description="Random state for reproducibility of train-test split.")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Configuration for the model.")

    @model_validator(mode="after")
    def _validate_split_ratio(self) -> "TrainingConfig":
        if not 0 < self.train_test_split_ratio < 1:
            raise ValueError("train_test_split_ratio must be strictly between 0 and 1.")
        return self
