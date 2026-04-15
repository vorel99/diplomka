"""Tests for the model registry and factory in geoscore_de.modelling.models."""

import pytest
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from geoscore_de.modelling.config import (
    ModelConfig,
    SearchConfig,
    TrainingConfig,
)
from geoscore_de.modelling.models import SUPPORTED_MODEL_TYPES, get_model_instance


class TestGetModelInstance:
    """Tests for the get_model_instance factory function."""

    def test_lightgbm_returns_lgbm_regressor(self):
        model = get_model_instance("lightgbm", random_state=42, param_overrides={})
        assert isinstance(model, LGBMRegressor)

    def test_random_forest_returns_rf_regressor(self):
        model = get_model_instance("random_forest", random_state=42, param_overrides={})
        assert isinstance(model, RandomForestRegressor)

    def test_gradient_boosting_returns_gb_regressor(self):
        model = get_model_instance("gradient_boosting", random_state=42, param_overrides={})
        assert isinstance(model, GradientBoostingRegressor)

    def test_lightgbm_default_params_applied(self):
        model = get_model_instance("lightgbm", random_state=0, param_overrides={})
        params = model.get_params()
        assert params["verbosity"] == -1
        assert params["min_child_samples"] == 20
        assert params["n_jobs"] == -1

    def test_random_forest_default_params_applied(self):
        model = get_model_instance("random_forest", random_state=0, param_overrides={})
        params = model.get_params()
        assert params["n_estimators"] == 100
        assert params["n_jobs"] == -1

    def test_gradient_boosting_default_params_applied(self):
        model = get_model_instance("gradient_boosting", random_state=0, param_overrides={})
        params = model.get_params()
        assert params["n_estimators"] == 100
        assert params["max_depth"] == 3
        assert params["learning_rate"] == pytest.approx(0.1)

    def test_random_state_injected(self):
        model = get_model_instance("lightgbm", random_state=99, param_overrides={})
        assert model.get_params()["random_state"] == 99

    def test_random_state_injected_random_forest(self):
        model = get_model_instance("random_forest", random_state=7, param_overrides={})
        assert model.get_params()["random_state"] == 7

    def test_param_overrides_applied(self):
        model = get_model_instance("lightgbm", random_state=42, param_overrides={"n_estimators": 50, "max_depth": 3})
        params = model.get_params()
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 3

    def test_param_overrides_do_not_affect_other_defaults(self):
        model = get_model_instance("lightgbm", random_state=42, param_overrides={"n_estimators": 50})
        params = model.get_params()
        assert params["verbosity"] == -1  # Default preserved

    def test_invalid_model_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_model_instance("unsupported_model", random_state=42, param_overrides={})

    def test_invalid_param_key_raises_type_error(self):
        with pytest.raises(TypeError, match="Invalid hyperparameter"):
            get_model_instance("lightgbm", random_state=42, param_overrides={"nonexistent_param": 1})

    def test_invalid_param_key_random_forest_raises_type_error(self):
        with pytest.raises(TypeError, match="Invalid hyperparameter"):
            get_model_instance("random_forest", random_state=42, param_overrides={"num_leaves": 31})

    def test_supported_model_types_contains_expected(self):
        assert "lightgbm" in SUPPORTED_MODEL_TYPES
        assert "random_forest" in SUPPORTED_MODEL_TYPES
        assert "gradient_boosting" in SUPPORTED_MODEL_TYPES


class TestModelConfigParsing:
    """Tests for parsing and validating ModelConfig via TrainingConfig."""

    def test_lgbm_config_parsed_from_dict(self):
        cfg = TrainingConfig(
            target_variable="y",
            model={"model_type": "lightgbm"},
            search=SearchConfig(param_grid={"n_estimators": [10]}),
        )
        assert isinstance(cfg.model, ModelConfig)
        assert cfg.model.model_type == "lightgbm"
        assert cfg.search.param_grid == {"n_estimators": [10]}

    def test_random_forest_config_parsed_from_dict(self):
        cfg = TrainingConfig(
            target_variable="y",
            model={"model_type": "random_forest"},
            search=SearchConfig(param_grid={"n_estimators": [100, 200]}),
        )
        assert isinstance(cfg.model, ModelConfig)
        assert cfg.model.model_type == "random_forest"

    def test_gradient_boosting_config_parsed_from_dict(self):
        cfg = TrainingConfig(
            target_variable="y",
            model={"model_type": "gradient_boosting"},
            search=SearchConfig(param_grid={"n_estimators": [100, 200]}),
        )
        assert isinstance(cfg.model, ModelConfig)

    def test_default_model_is_lightgbm(self):
        cfg = TrainingConfig(target_variable="y")
        assert isinstance(cfg.model, ModelConfig)
        assert cfg.model.model_type == "lightgbm"

    def test_invalid_model_type_raises_validation_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TrainingConfig(
                target_variable="y",
                model={"model_type": "unsupported_model"},
            )

    def test_param_grid_defaults_to_empty(self):
        cfg = TrainingConfig(target_variable="y", model={"model_type": "random_forest"})
        assert cfg.search.param_grid == {}
