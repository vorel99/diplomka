"""Shared configuration models used across multiple geoscore_de subpackages."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureFilteringConfig(BaseModel):
    """Configuration for feature (column) filtering.

    If both ``use_features`` and ``omit_features`` are provided, ``use_features`` is applied first
    to select a subset of columns, and then ``omit_features`` is applied to exclude specific columns
    from that subset.  Patterns support glob-style wildcards (e.g. ``census*``) and full regular
    expressions.
    """

    omit_features: list[str] = Field(default_factory=list, description="List of features to omit from the dataset.")
    use_features: list[str] = Field(
        default_factory=list,
        description="List of features to use. If empty, all features will be used.",
    )
