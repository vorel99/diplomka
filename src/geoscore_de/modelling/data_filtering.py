import warnings

import pandas as pd

from geoscore_de.filtering import _compile_pattern, filter_features  # noqa: F401
from geoscore_de.modelling.config import TrainingConfig


def filter_rows(data: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Filter rows based on omit_rows config.

    For each column listed in row_filtering.omit_rows, rows whose column value
    matches any of the supplied patterns are dropped.  Patterns follow the same
    glob / regex rules as feature filtering (e.g. '02*' matches '0212', '023').

    Args:
        data (pd.DataFrame): Input DataFrame before row filtering.
        config (TrainingConfig): Training configuration containing row_filtering.
    Returns:
        pd.DataFrame: DataFrame with matching rows removed.
    """
    row_filter = config.row_filtering

    if not row_filter.omit_rows:
        return data

    mask = pd.Series(False, index=data.index)

    for column, patterns in row_filter.omit_rows.items():
        if column not in data.columns:
            warnings.warn(f"Column '{column}' specified in row_filtering.omit_rows not found in data.")
            continue

        col_values = data[column].astype(str)
        null_mask = data[column].isna()

        for pattern in patterns:
            compiled = _compile_pattern(pattern)
            if compiled is None:
                continue
            row_match = col_values.apply(lambda v: bool(compiled.fullmatch(v))) & ~null_mask
            mask |= row_match

    return data[~mask]
