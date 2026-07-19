from __future__ import annotations

import sys
from pathlib import Path

import joblib
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hospital_pricing.data import CATEGORICAL_FEATURES, FEATURES  # noqa: E402

MODEL_PATH = ROOT / "outputs" / "hospital_cost_model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

st.set_page_config(page_title="Hospital Cost Estimator", page_icon="🏥", layout="wide")


@st.cache_resource
def load_bundle() -> dict[str, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. Run `PYTHONPATH=src python -m hospital_pricing.train`."
        )
    return joblib.load(MODEL_PATH)


st.title("Hospital Cost Estimator")
st.caption("A leakage-aware portfolio demo using admission-time information")
st.info(
    "This is an educational case study, not a clinical or billing tool. The dataset "
    "contains only 248 historical admissions, so uncertainty is substantial."
)

try:
    bundle = load_bundle()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

with st.sidebar:
    st.header("Portfolio snapshot")
    if metrics:
        st.metric("Held-out MAE", f"₹{metrics['test']['mae']:,.0f}")
        improvement = 1 - metrics["test"]["mae"] / metrics["baseline_test"]["mae"]
        st.metric("Improvement vs baseline", f"{improvement:.1%}")
        st.metric("Held-out R²", f"{metrics['test']['r2']:.3f}")
    st.caption("248 admissions · 18 admission-time features · 5-fold CV")

st.markdown(
    "Enter the information available when a patient arrives. Realized length of stay "
    "and final implant cost are intentionally excluded to prevent future-data leakage."
)

values: dict[str, object] = {}
with st.form("prediction_form"):
    left, right = st.columns(2)
    numeric_features = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
    for index, feature in enumerate(numeric_features):
        container = left if index % 2 == 0 else right
        default = float(bundle["numeric_defaults"][feature])
        values[feature] = container.number_input(
            feature.replace("_", " ").title(), value=default, format="%.2f"
        )

    st.subheader("Admission details")
    columns = st.columns(2)
    for index, feature in enumerate(CATEGORICAL_FEATURES):
        options = bundle["category_options"][feature]
        values[feature] = columns[index % 2].selectbox(
            feature.replace("_", " ").title(), options=options
        )

    submitted = st.form_submit_button("Estimate cost", type="primary")

if submitted:
    row = pd.DataFrame([values], columns=FEATURES)
    prediction = max(0.0, float(bundle["model"].predict(row)[0]))
    half_width = float(bundle["interval_half_width"])
    lower = max(0.0, prediction - half_width)
    upper = prediction + half_width

    st.metric("Estimated hospital cost", f"₹{prediction:,.0f}")
    st.write(f"Approximate 90% error band: **₹{lower:,.0f}–₹{upper:,.0f}**")
    st.caption(
        "The band is based on out-of-fold training errors and communicates model "
        "uncertainty; it is not a formal price guarantee."
    )

with st.expander("Model card"):
    st.write(f"Selected model: **{str(bundle['model_name']).replace('_', ' ').title()}**")
    st.write("Selection: lowest five-fold cross-validated MAE among non-baseline models")
    st.write("Target transformation: log1p, converted back to original currency units")
    if metrics:
        st.write(f"Empirical test interval coverage: **{metrics['interval_test_coverage']:.0%}**")

with st.expander("See held-out diagnostics"):
    diagnostic_path = ROOT / "outputs" / "model_diagnostics.png"
    if diagnostic_path.exists():
        st.image(str(diagnostic_path), caption="Predictions and permutation importance")
    st.caption(
        "The model underpredicts several rare high-cost admissions. The chart and "
        "metrics are retained to make that limitation visible."
    )
