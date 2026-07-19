"""Leakage-safe model construction and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 42


@dataclass(frozen=True)
class Evaluation:
    mae: float
    rmse: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="Unknown"),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _with_log_target(regressor: object) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        [("preprocessor", make_preprocessor()), ("regressor", regressor)]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=True,
    )


def candidate_models() -> dict[str, object]:
    """Return interpretable baseline and nonlinear candidate estimators."""
    ridge_pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(scale_numeric=True)),
            ("regressor", Ridge(alpha=10.0)),
        ]
    )
    return {
        "median_baseline": Pipeline(
            [("preprocessor", make_preprocessor()), ("regressor", DummyRegressor())]
        ),
        "ridge": TransformedTargetRegressor(
            regressor=ridge_pipeline, func=np.log1p, inverse_func=np.expm1
        ),
        "random_forest": _with_log_target(
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
    }


def evaluate(y_true: pd.Series | np.ndarray, predictions: np.ndarray) -> Evaluation:
    """Calculate metrics in original currency units."""
    return Evaluation(
        mae=float(mean_absolute_error(y_true, predictions)),
        rmse=float(mean_squared_error(y_true, predictions) ** 0.5),
        r2=float(r2_score(y_true, predictions)),
    )


def subgroup_metrics(
    groups: pd.Series,
    y_true: pd.Series,
    predictions: np.ndarray,
    *,
    minimum_size: int = 5,
) -> pd.DataFrame:
    """Summarize errors by group while suppressing unstable tiny slices."""
    audit = pd.DataFrame(
        {
            "group": groups.fillna("Missing").astype(str).to_numpy(),
            "actual": np.asarray(y_true),
            "predicted": np.asarray(predictions),
        }
    )
    audit["absolute_error"] = (audit["predicted"] - audit["actual"]).abs()
    audit["signed_error"] = audit["predicted"] - audit["actual"]
    summary = (
        audit.groupby("group", as_index=False)
        .agg(
            sample_size=("actual", "size"),
            mean_actual_cost=("actual", "mean"),
            mae=("absolute_error", "mean"),
            mean_bias=("signed_error", "mean"),
        )
        .sort_values("sample_size", ascending=False)
    )
    return summary.loc[summary["sample_size"] >= minimum_size].reset_index(drop=True)
