"""Train, compare, evaluate, and persist the hospital cost estimator."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split

from .data import CATEGORICAL_FEATURES, FEATURES, load_data
from .modeling import RANDOM_STATE, candidate_models, evaluate, subgroup_metrics


def train(data_path: Path, output_dir: Path) -> dict[str, object]:
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    comparison: dict[str, dict[str, float]] = {}
    models = candidate_models()
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=1,
        )
        comparison[name] = {
            "cv_mae_mean": float(-scores["test_mae"].mean()),
            "cv_mae_std": float(scores["test_mae"].std()),
            "cv_r2_mean": float(scores["test_r2"].mean()),
        }

    eligible = {name: values for name, values in comparison.items() if name != "median_baseline"}
    best_name = min(eligible, key=lambda name: eligible[name]["cv_mae_mean"])
    best_model = models[best_name]

    # Out-of-fold errors estimate a pragmatic 90% uncertainty band without
    # looking at the held-out test set.
    oof_predictions = cross_val_predict(best_model, X_train, y_train, cv=cv, n_jobs=1)
    interval_half_width = float(np.quantile(np.abs(y_train - oof_predictions), 0.90))

    best_model.fit(X_train, y_train)
    test_predictions = best_model.predict(X_test)
    test_metrics = evaluate(y_test, test_predictions)

    baseline = models["median_baseline"].fit(X_train, y_train)
    baseline_metrics = evaluate(y_test, baseline.predict(X_test))

    interval_lower = np.maximum(0, test_predictions - interval_half_width)
    interval_upper = test_predictions + interval_half_width
    interval_coverage = float(
        np.mean((y_test.to_numpy() >= interval_lower) & (y_test.to_numpy() <= interval_upper))
    )

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {"feature": FEATURES, "mae_increase": importance.importances_mean}
    ).sort_values("mae_increase", ascending=False)

    category_options = {
        feature: sorted(X[feature].dropna().astype(str).unique().tolist())
        for feature in CATEGORICAL_FEATURES
    }
    numeric_defaults = {
        feature: float(X[feature].median())
        for feature in FEATURES
        if feature not in CATEGORICAL_FEATURES
    }
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURES,
        "category_options": category_options,
        "numeric_defaults": numeric_defaults,
        "interval_half_width": interval_half_width,
        "trained_at": datetime.now(UTC).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "hospital_cost_model.joblib")
    importance_frame.to_csv(output_dir / "feature_importance.csv", index=False)
    _save_subgroup_metrics(X_test, y_test, test_predictions, output_dir)
    _save_data_quality_report(X, y, output_dir)

    metrics: dict[str, object] = {
        "dataset_rows": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "selected_model": best_name,
        "cross_validation": comparison,
        "test": test_metrics.as_dict(),
        "baseline_test": baseline_metrics.as_dict(),
        "interval_90_half_width": interval_half_width,
        "interval_test_coverage": interval_coverage,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _save_diagnostics(y_test, test_predictions, importance_frame, output_dir)
    return metrics


def _save_subgroup_metrics(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    output_dir: Path,
) -> None:
    slices = {
        "gender": X_test["GENDER"],
        "admission_type": X_test["TYPE_OF_ADMSN"],
        "implant_planned": X_test["IMPLANT_USED_Y_N"],
        "age_band": pd.cut(
            X_test["AGE"],
            bins=[0, 40, 60, np.inf],
            labels=["40 or younger", "41-60", "61 or older"],
        ).astype("object"),
    }
    reports = []
    for slice_name, groups in slices.items():
        report = subgroup_metrics(groups, y_test, predictions)
        report.insert(0, "slice", slice_name)
        reports.append(report)
    pd.concat(reports, ignore_index=True).to_csv(
        output_dir / "subgroup_metrics.csv", index=False
    )


def _save_data_quality_report(
    X: pd.DataFrame, y: pd.Series, output_dir: Path
) -> None:
    report = {
        "rows": len(X),
        "features": X.shape[1],
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "missing_values": {
            feature: {
                "count": int(X[feature].isna().sum()),
                "rate": float(X[feature].isna().mean()),
            }
            for feature in X.columns
            if X[feature].isna().any()
        },
        "target_cost_quantiles": {
            str(quantile): float(y.quantile(quantile))
            for quantile in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
        },
    }
    (output_dir / "data_quality.json").write_text(json.dumps(report, indent=2) + "\n")


def _save_diagnostics(
    y_test: pd.Series,
    predictions: np.ndarray,
    importance: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, predictions, alpha=0.75, color="#146C94")
    bounds = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    axes[0].plot(bounds, bounds, "--", color="#E45756")
    axes[0].set(xlabel="Actual cost", ylabel="Predicted cost", title="Held-out predictions")

    plot_data = importance.head(10).sort_values("mae_increase")
    axes[1].barh(plot_data["feature"], plot_data["mae_increase"], color="#2A9D8F")
    axes[1].set(xlabel="Increase in MAE when shuffled", title="Permutation importance")
    fig.tight_layout()
    fig.savefig(output_dir / "model_diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mission_hospital.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    metrics = train(args.data, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
