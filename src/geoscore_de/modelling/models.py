"""Model registry and factory for supported regression models.

Supported model types:
- ``"lightgbm"``      → :class:`lightgbm.LGBMRegressor`
- ``"random_forest"`` → :class:`sklearn.ensemble.RandomForestRegressor`
- ``"gradient_boosting"`` → :class:`sklearn.ensemble.GradientBoostingRegressor`
- ``"xgboost"``       → :class:`xgboost.XGBRegressor`  (optional – requires ``pip install xgboost``)
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

__all__ = ["SUPPORTED_MODEL_TYPES", "get_model_instance"]

# Sensible defaults for each model type,
# which can be overridden by the user via *param_overrides* in get_model_instance.
_LIGHTGBM_DEFAULTS: dict[str, Any] = {
    "verbosity": -1,
    "min_child_samples": 20,
    "min_split_gain": 0.0,
    "n_jobs": -1,
}

_RANDOM_FOREST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
}

_GRADIENT_BOOSTING_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}

_XGBOOST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "n_jobs": -1,
    "verbosity": 0,
}

# Registry: model_type → (ModelClass, default_params)
_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    "lightgbm": (LGBMRegressor, _LIGHTGBM_DEFAULTS),
    "random_forest": (RandomForestRegressor, _RANDOM_FOREST_DEFAULTS),
    "gradient_boosting": (GradientBoostingRegressor, _GRADIENT_BOOSTING_DEFAULTS),
}

# Register XGBoost only when the optional package is installed
try:
    from xgboost import XGBRegressor  # noqa: PLC0415

    _REGISTRY["xgboost"] = (XGBRegressor, _XGBOOST_DEFAULTS)
except ImportError:
    XGBRegressor = None  # type: ignore[assignment,misc]

SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(_REGISTRY)

# Cache of valid constructor parameter names per model class (avoids repeated instantiation)
_VALID_PARAMS_CACHE: dict[type, frozenset[str]] = {}


def _get_valid_params(model_class: type) -> frozenset[str]:
    """Return the set of valid constructor parameter names for *model_class*.

    Results are cached to avoid re-inspecting the signature on every call.
    """
    if model_class not in _VALID_PARAMS_CACHE:
        sig = inspect.signature(model_class.__init__)
        _VALID_PARAMS_CACHE[model_class] = frozenset(
            name for name, param in sig.parameters.items() if name not in ("self", "args", "kwargs")
        )
    return _VALID_PARAMS_CACHE[model_class]


def get_model_instance(
    model_type: str,
    random_state: int,
    param_overrides: dict[str, Any] | None = None,
) -> "BaseEstimator":
    """Instantiate a regression model with merged defaults and overrides.

    Args:
        model_type: One of the supported model type strings (e.g. ``"lightgbm"``).
        random_state: Random seed forwarded to the estimator when supported.
        param_overrides: Flat dict of hyperparameter overrides (single values, not lists).
            These values are merged on top of the defaults. Defaults to ``None`` (no overrides).

    Returns:
        A fitted-ready sklearn-compatible estimator.

    Raises:
        ValueError: If *model_type* is not in the registry (e.g. xgboost not installed).
        TypeError: If a key in *param_overrides* is not a valid constructor parameter for
            the chosen model class.
    """
    if param_overrides is None:
        param_overrides = {}

    if model_type not in _REGISTRY:
        if model_type == "xgboost" and XGBRegressor is None:
            raise ValueError("model_type='xgboost' requires the xgboost package. Install it with: pip install xgboost")
        raise ValueError(f"Unknown model_type '{model_type}'. Supported types: {sorted(SUPPORTED_MODEL_TYPES)}")

    model_class, defaults = _REGISTRY[model_type]

    # Validate override keys against the model's constructor signature
    valid_params = _get_valid_params(model_class)
    invalid_keys = set(param_overrides.keys()) - valid_params
    if invalid_keys:
        raise TypeError(
            f"Invalid hyperparameter(s) for {model_class.__name__}: {sorted(invalid_keys)}. "
            f"Valid parameters: {sorted(valid_params)}"
        )

    params = {**defaults, **param_overrides}

    # Always inject random_state where supported so it is never overridden by param_overrides
    if "random_state" in valid_params:
        params["random_state"] = random_state

    return model_class(**params)
