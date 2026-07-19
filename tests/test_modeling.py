import numpy as np

from hospital_pricing.data import CATEGORICAL_FEATURES, load_data
from hospital_pricing.modeling import candidate_models, evaluate, subgroup_metrics


def test_pipeline_predicts_positive_cost_and_handles_unknown_category() -> None:
    X, y = load_data("data/mission_hospital.xlsx")
    model = candidate_models()["ridge"].fit(X.iloc[:200], y.iloc[:200])
    unseen = X.iloc[[200]].copy()
    unseen.loc[:, CATEGORICAL_FEATURES[0]] = "UNSEEN_VALUE"
    prediction = model.predict(unseen)
    assert prediction.shape == (1,)
    assert prediction[0] > 0


def test_evaluation_metrics_are_in_original_units() -> None:
    result = evaluate(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert result.mae == 15.0
    assert result.rmse > result.mae


def test_subgroup_metrics_suppresses_tiny_groups_and_reports_bias() -> None:
    import pandas as pd

    groups = pd.Series(["A", "A", "A", "B"])
    actual = pd.Series([100.0, 100.0, 100.0, 100.0])
    result = subgroup_metrics(
        groups, actual, np.array([110.0, 90.0, 120.0, 100.0]), minimum_size=2
    )
    assert result["group"].tolist() == ["A"]
    assert result.loc[0, "sample_size"] == 3
    assert np.isclose(result.loc[0, "mean_bias"], 20 / 3)
