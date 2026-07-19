"""Data loading and validation for the Mission Hospital case study."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TARGET = "TOTAL_COST_TO_HOSPITAL"

# Only information plausibly available at admission is used. Realized length of
# stay and implant cost are deliberately excluded because they leak future care.
NUMERIC_FEATURES = [
    "AGE",
    "BODY_WEIGHT",
    "BODY_HEIGHT",
    "HR_PULSE",
    "BP_HIGH",
    "BP_LOW",
    "RR",
    "HB",
    "UREA",
    "CREATININE",
]
CATEGORICAL_FEATURES = [
    "GENDER",
    "MARITAL_STATUS",
    "KEY_COMPLAINTS_CODE",
    "PAST_MEDICAL_HISTORY_CODE",
    "MODE_OF_ARRIVAL",
    "STATE_AT_THE_TIME_OF_ARRIVAL",
    "TYPE_OF_ADMSN",
    "IMPLANT_USED_Y_N",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def normalize_column_name(value: object) -> str:
    """Convert inconsistent spreadsheet labels to stable snake-case names."""
    text = str(value).strip().replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the raw case-study sheet."""
    frame = pd.read_excel(path, sheet_name="MH-Raw Data")
    frame.columns = [normalize_column_name(column) for column in frame.columns]

    required = set(FEATURES + [TARGET])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame.dropna(subset=[TARGET]).copy()
    if (frame[TARGET] <= 0).any():
        raise ValueError("Target costs must be positive")

    return frame[FEATURES], frame[TARGET].astype(float)
