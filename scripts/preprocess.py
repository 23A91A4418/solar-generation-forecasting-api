"""
scripts/preprocess.py
---------------------
Loads data/raw_data.csv, cleans it, performs feature engineering,
and saves data/processed_data.csv ready for model training.

Run:
    python scripts/preprocess.py
"""

import os
import sys
import numpy as np
import pandas as pd

RAW_PATH = os.path.join("data", "raw_data.csv")
PROCESSED_PATH = os.path.join("data", "processed_data.csv")


# ── Feature Engineering Functions (also imported in tests) ──────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, month, day_of_week from DATE_TIME index/column."""
    df = df.copy()
    df["hour"] = df["DATE_TIME"].dt.hour + df["DATE_TIME"].dt.minute / 60.0
    df["month"] = df["DATE_TIME"].dt.month
    df["day_of_week"] = df["DATE_TIME"].dt.dayofweek
    return df


def add_sinusoidal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour as sin/cos pair so linear models can capture the circular
    daily periodicity (23:00 ≈ 01:00 in solar terms).
    """
    df = df.copy()
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Irradiation × Ambient Temperature interaction term."""
    df = df.copy()
    df["irr_x_temp"] = df["IRRADIATION"] * df["AMBIENT_TEMPERATURE"]
    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that cannot correspond to meaningful solar generation:
      - IRRADIATION <= 0  (night-time / sensor noise)
      - AC_POWER < 0      (sensor error)
      - MODULE_TEMPERATURE < AMBIENT_TEMPERATURE - 5  (physically impossible)
    """
    df = df.copy()
    before = len(df)

    df = df[df["IRRADIATION"] > 0]
    df = df[df["AC_POWER"] >= 0]
    df = df[df["MODULE_TEMPERATURE"] >= df["AMBIENT_TEMPERATURE"] - 5]

    after = len(df)
    print(f"[preprocess] Filtered {before - after:,} anomalous rows "
          f"({100*(before - after)/before:.1f}%); {after:,} rows remain.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      - Numeric columns: fill with column median (robust to outliers).
      - Drop rows where the *target* (AC_POWER) is still NaN after fill.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians)
    df = df.dropna(subset=["AC_POWER"])
    return df


# ── Main Pipeline ────────────────────────────────────────────────────

def main():
    if not os.path.exists(RAW_PATH):
        print(f"[preprocess] ERROR: {RAW_PATH} not found.")
        print("  Run  python scripts/generate_data.py  to create it first.")
        sys.exit(1)

    print(f"[preprocess] Loading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"[preprocess] Raw shape: {df.shape}")

    # ── 1. Parse datetime ───────────────────────────────────────────
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True)

    # ── 2. Handle missing values ────────────────────────────────────
    df = handle_missing_values(df)

    # ── 3. Filter anomalous rows ────────────────────────────────────
    df = filter_data(df)

    # ── 4. Feature engineering ──────────────────────────────────────
    df = add_time_features(df)
    df = add_sinusoidal_features(df)
    df = add_interaction_features(df)

    # ── 5. Select & order output columns ────────────────────────────
    feature_cols = [
        "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "hour",
        "month",
        "day_of_week",
        "sin_hour",
        "cos_hour",
        "irr_x_temp",
        "AC_POWER",  # target — keep last
    ]
    df = df[feature_cols].reset_index(drop=True)

    # ── 6. Save ─────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[preprocess] Saved processed data -> {PROCESSED_PATH}")
    print(f"[preprocess] Final shape: {df.shape}")
    print(df.describe().round(3).to_string())


if __name__ == "__main__":
    main()
