import pandas as pd
import pytest

from geoscore_de.data_flow.feature_engineering.base import StatefulFeatureEngineering
from geoscore_de.data_flow.feature_engineering.config import FeatureEngineeringConfig
from geoscore_de.modelling.config import EarlyStoppingConfig, ModelConfig, SearchConfig, TrainingConfig
from geoscore_de.modelling.train import Trainer


def _build_dataframe(rows_per_state: int) -> pd.DataFrame:
    records: list[dict] = []
    for state_id in range(1, 17):
        state_code = f"{state_id:02d}"
        for municipality_idx in range(rows_per_state):
            ags = f"{state_code}{municipality_idx:06d}"
            records.append(
                {
                    "AGS": ags,
                    "federal_republic_id": state_code,
                    "feature_one": municipality_idx,
                    "feature_two": state_id * 0.1,
                    "unemployment_unemployment_per_capita": municipality_idx * 0.01 + state_id,
                }
            )
    return pd.DataFrame(records)


def _build_config(split_strategy: str = "random") -> TrainingConfig:
    return TrainingConfig(
        target_variable="unemployment_unemployment_per_capita",
        train_test_split_ratio=0.8,
        split_strategy=split_strategy,
        federal_state_column="federal_republic_id",
        model=ModelConfig(model_type="lightgbm"),
        search=SearchConfig(param_grid={"n_estimators": [10], "max_depth": [2], "num_leaves": [10]}),
        early_stopping=EarlyStoppingConfig(early_stopping_rounds=None),
    )


def test_prepare_data_stratified_split_keeps_state_proportions():
    data = _build_dataframe(rows_per_state=60)
    trainer = Trainer(_build_config(split_strategy="stratified_federal_state"))

    _, X_test, _, _ = trainer._prepare_data(data)

    test_state_counts = data.loc[X_test.index, "federal_republic_id"].value_counts()
    assert (test_state_counts == 12).all()


def test_prepare_data_random_split_as_default_without_state_columns():
    data = pd.DataFrame(
        {
            "AGS": [f"0000{i:04d}" for i in range(100)],
            "feature_one": [float(i) for i in range(100)],
            "unemployment_unemployment_per_capita": [i * 0.01 for i in range(100)],
        }
    )
    trainer = Trainer(_build_config())

    X_train_val, X_test, y_train_val, y_test = trainer._prepare_data(data)

    assert len(X_train_val) == 80
    assert len(X_test) == 20
    assert len(y_train_val) == 80
    assert len(y_test) == 20


def test_stateful_transform_fits_on_training_split_only(monkeypatch: pytest.MonkeyPatch):
    data = pd.DataFrame(
        {
            "AGS": [f"0100{i:04d}" for i in range(8)],
            "feature_one": [1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0],
            "unemployment_unemployment_per_capita": [0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3],
        }
    )

    class SpyStatefulTransform(StatefulFeatureEngineering):
        def __init__(self, input_columns: list[str], output_column: str, **kwargs):
            super().__init__(input_columns, output_column)
            self.fit_inputs: list[pd.DataFrame] = []
            self.transform_inputs: list[pd.DataFrame] = []
            self.fitted_mean: float | None = None

        def _validate(self, df: pd.DataFrame) -> bool:
            return True

        def fit(self, X, y=None):
            self.fit_inputs.append(X.copy())
            self.fitted_mean = float(X[self.input_columns[0]].mean())
            return self

        def transform(self, X):
            self.transform_inputs.append(X.copy())
            return super().transform(X)

        def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
            df[self.output_column] = self.fitted_mean
            return df

    spy_transform = SpyStatefulTransform(["feature_one"], "feature_one_binned")
    monkeypatch.setattr(
        "geoscore_de.modelling.train.instantiate_feature_engineering_class",
        lambda config: spy_transform,
    )

    trainer = Trainer(
        _build_config().model_copy(
            update={
                "stateful_transforms": [
                    FeatureEngineeringConfig(
                        name="feature_one_bins",
                        class_name="KBinsDiscretizerBinning",
                        module="geoscore_de.data_flow.feature_engineering.kbins_binning",
                        input_columns=["feature_one"],
                        output_column="feature_one_binned",
                        params={"strategy": "quantile", "n_bins": 2},
                    )
                ]
            }
        )
    )

    X_train_val, X_test, _, _ = trainer._prepare_data(data)
    X_train_binned, X_test_binned = trainer._apply_stateful_transforms(X_train_val, X_test)

    assert "feature_one_binned" in X_train_binned.columns
    assert "feature_one_binned" in X_test_binned.columns
    # Check that fit was called only on training data and transform was called on both training and test data
    assert len(spy_transform.fit_inputs) == 1
    assert spy_transform.fit_inputs[0].equals(X_train_val)
    assert len(spy_transform.transform_inputs) == 2
    assert spy_transform.transform_inputs[0].equals(X_train_val)
    assert spy_transform.transform_inputs[1].equals(X_test)
    assert X_train_binned["feature_one_binned"].eq(spy_transform.fitted_mean).all()
    assert X_test_binned["feature_one_binned"].eq(spy_transform.fitted_mean).all()


def test_catboost_model_receives_categorical_feature_names():
    data = pd.DataFrame(
        {
            "AGS": [f"0100{i:04d}" for i in range(4)],
            "category_feature": pd.Series(["a", "b", "a", "c"], dtype="category"),
            "text_feature": ["x", "y", "x", "z"],
            "numeric_feature": [1.0, 2.0, 3.0, 4.0],
            "unemployment_unemployment_per_capita": [0.1, 0.2, 0.3, 0.4],
        }
    )
    trainer = Trainer(
        TrainingConfig(
            target_variable="unemployment_unemployment_per_capita",
            model=ModelConfig(model_type="catboost"),
            search=SearchConfig(param_grid={"depth": [2]}),
        )
    )

    X_train_val = data.drop(columns=["AGS", "unemployment_unemployment_per_capita"])
    model = trainer._get_model()

    # index 0 and 1 correspond to "category_feature" and "text_feature"
    assert trainer._get_catboost_fit_params(X_train_val)["cat_features"] == [0, 1]
    assert "cat_features" not in model.get_params()


def test_prepare_data_stratified_falls_back_to_random_on_singleton_state():
    data = pd.DataFrame(
        {
            "AGS": [f"0100{i:04d}" for i in range(50)] + ["02000000"],
            "federal_republic_id": ["01"] * 50 + ["02"],
            "feature_one": [float(i) for i in range(51)],
            "unemployment_unemployment_per_capita": [i * 0.01 for i in range(51)],
        }
    )
    trainer = Trainer(_build_config(split_strategy="stratified_federal_state"))

    with pytest.warns(UserWarning, match="Falling back to random split"):
        X_train_val, X_test, y_train_val, y_test = trainer._prepare_data(data)

    assert len(X_train_val) + len(X_test) == 51
    assert len(y_train_val) + len(y_test) == 51


def test_training_config_rejects_invalid_split_ratio():
    with pytest.raises(ValueError, match="train_test_split_ratio must be strictly between 0 and 1"):
        TrainingConfig(target_variable="target", train_test_split_ratio=1.0)
