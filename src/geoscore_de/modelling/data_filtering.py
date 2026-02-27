import re
import warnings

import pandas as pd

from geoscore_de.modelling.config import TrainingConfig


def _compile_pattern(pattern: str) -> re.Pattern | None:
    """Compile a feature pattern to regex.

    Behaviour:
    - If the pattern already looks like a regex (contains typical regex metachars
        other than '*'), it is used as-is.
    - Otherwise, we treat '*' as a wildcard over the whole column name:
        'census*' -> r'^census.*$'.
    """
    # Heuristic: if pattern has regex metachars (except '*'), treat as raw regex
    regex_meta = set(".^$[]()+?|{}\\")
    if any(ch in regex_meta for ch in pattern):
        regex = pattern
    else:
        # Glob-style: escape then turn '*' into '.*' and anchor the whole name
        escaped = re.escape(pattern)
        escaped = escaped.replace(r"\*", ".*")
        regex = f"^{escaped}$"

    try:
        return re.compile(regex)
    except re.error:
        warnings.warn(f"Invalid regex pattern in feature filter: {pattern!r}")
        return None


def _resolve_feature_patterns(columns, patterns):
    """Map use/omit patterns to concrete column names.

    Args:
        columns (list): List of column names to match against patterns.
        patterns (list): List of patterns to match against columns.
    Returns:
        list: List of column names that matched the patterns.
    """
    columns = list(columns)
    matched: list[str] = []
    seen = set()

    for pattern in patterns:
        # Exact match first to keep behaviour intuitive
        if pattern in columns and pattern not in seen:
            matched.append(pattern)
            seen.add(pattern)
            continue

        compiled = _compile_pattern(pattern)
        if compiled is None:
            continue

        for col in columns:
            if col in seen:
                continue
            if compiled.fullmatch(col):
                matched.append(col)
                seen.add(col)

    return matched


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


def filter_features(data: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Filter features based on use_features and omit_features lists in config.
    Features in use_features are selected first, then features in omit_features are dropped from the result.

    Args:
        data (pd.DataFrame): Input features DataFrame before filtering.
    Returns:
        pd.DataFrame: Filtered features DataFrame.
    """
    feature_filter = config.feature_filtering
    columns = list(data.columns)

    # 1) Select requested features (if any)
    if feature_filter.use_features:
        selected = _resolve_feature_patterns(columns, feature_filter.use_features)

        # Warn for patterns that matched nothing
        unmatched = []
        for pattern in feature_filter.use_features:
            # pattern is considered matched if it is either an exact column
            # or its compiled regex matched at least one column
            if pattern in selected:
                continue
            compiled = _compile_pattern(pattern)
            if compiled is None:
                unmatched.append(pattern)
                continue
            if not any(compiled.fullmatch(col) for col in columns):
                unmatched.append(pattern)

        if unmatched:
            warnings.warn(f"No columns matched for use_features patterns: {set(unmatched)}")

        # Preserve order based on patterns / regex matches
        data = data[selected]
        columns = list(data.columns)

    # 2) Drop omitted features (if any)
    if feature_filter.omit_features:
        to_drop = _resolve_feature_patterns(columns, feature_filter.omit_features)

        unmatched = []
        for pattern in feature_filter.omit_features:
            if pattern in to_drop:
                continue
            compiled = _compile_pattern(pattern)
            if compiled is None:
                unmatched.append(pattern)
                continue
            if not any(compiled.fullmatch(col) for col in columns):
                unmatched.append(pattern)

        if unmatched:
            warnings.warn(f"No columns matched for omit_features patterns: {set(unmatched)}")

        if to_drop:
            data = data.drop(columns=to_drop, errors="ignore")

    return data
