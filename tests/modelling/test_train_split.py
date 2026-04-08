import pandas as pd
import pytest

from geoscore_de.modelling.config import ModelConfig, TrainingConfig
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
        model=ModelConfig(param_grid={"n_estimators": [10], "max_depth": [2], "num_leaves": [10]}),
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
