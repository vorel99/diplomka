import pandas as pd

from geoscore_de.modelling.training_result import TrainingResult


class _DummyEstimator:
    def predict(self, X):
        return [1.0 for _ in range(len(X))]


class _DummyGridSearch:
    best_estimator_ = _DummyEstimator()


def test_evaluate_subset_returns_prefixed_metric_keys():
    result = TrainingResult(
        grid_search=_DummyGridSearch(),
        X_train=pd.DataFrame({"x": [1.0, 2.0]}),
        X_test=pd.DataFrame({"x": [1.0, 2.0]}),
        y_train=pd.Series([1.0, 2.0]),
        y_test=pd.Series([1.0, 2.0]),
    )

    metrics = result.evaluate_subset(
        X_subset=pd.DataFrame({"x": [1.0, 2.0]}),
        y_subset=pd.Series([1.0, 2.0]),
        metric_prefix="state_01",
    )

    expected_keys = {
        "state_01_r2_score",
        "state_01_mae",
        "state_01_mse",
        "state_01_rmse",
        "state_01_mape",
        "state_01_median_ae",
        "state_01_max_error",
        "state_01_explained_variance",
    }
    assert set(metrics.keys()) == expected_keys
