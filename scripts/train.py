"""
scripts/train.py
----------------
Trains a Linear Regression (baseline) and a LightGBM (advanced) model
on the processed solar dataset, evaluates both, and saves artifacts.

Run:
    python scripts/train.py

Outputs:
    models/linear_regression_model.pkl
    models/advanced_model.pkl
    reports/metrics.json
"""

import os
import json
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

PROCESSED_PATH = os.path.join("data", "processed_data.csv")
LR_MODEL_PATH = os.path.join("models", "linear_regression_model.pkl")
ADV_MODEL_PATH = os.path.join("models", "advanced_model.pkl")
METRICS_PATH = os.path.join("reports", "metrics.json")

FEATURE_COLS = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "hour",
    "month",
    "day_of_week",
    "sin_hour",
    "cos_hour",
    "irr_x_temp",
]
TARGET_COL = "AC_POWER"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 6), "mae": round(mae, 6), "r2": round(r2, 6)}


def print_metrics(name: str, m: dict) -> None:
    print(f"  {name:<25} RMSE={m['rmse']:>10.4f}  MAE={m['mae']:>10.4f}  R2={m['r2']:.6f}")


def main():
    # ── Load data ────────────────────────────────────────────────────
    if not os.path.exists(PROCESSED_PATH):
        print(f"[train] ERROR: {PROCESSED_PATH} not found.")
        print("  Run  python scripts/preprocess.py  first.")
        sys.exit(1)

    print(f"[train] Loading {PROCESSED_PATH} ...")
    df = pd.read_csv(PROCESSED_PATH)
    print(f"[train] Dataset shape: {df.shape}")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[train] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Baseline: Linear Regression ──────────────────────────────────
    print("\n[train] Training Linear Regression baseline ...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_metrics = compute_metrics(y_test, lr_pred)
    print_metrics("Linear Regression", lr_metrics)

    # ── Advanced: LightGBM ───────────────────────────────────────────
    print("\n[train] Training LightGBM advanced model ...")
    lgbm_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[],
    )
    adv_pred = lgbm_model.predict(X_test)
    adv_metrics = compute_metrics(y_test, adv_pred)
    print_metrics("LightGBM", adv_metrics)

    # ── Save models ──────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_model, LR_MODEL_PATH)
    joblib.dump(lgbm_model, ADV_MODEL_PATH)
    print(f"\n[train] Saved -> {LR_MODEL_PATH}")
    print(f"[train] Saved -> {ADV_MODEL_PATH}")

    # ── Save metrics ─────────────────────────────────────────────────
    metrics = {
        "linear_regression": lr_metrics,
        "advanced_model": adv_metrics,
    }
    os.makedirs("reports", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[train] Saved -> {METRICS_PATH}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n── Performance Summary ────────────────────────────────────")
    print_metrics("Linear Regression", lr_metrics)
    print_metrics("LightGBM (advanced)", adv_metrics)
    improvement = adv_metrics["r2"] - lr_metrics["r2"]
    print(f"\n  R2 improvement (LightGBM over LR): +{improvement:.6f}")

    if lr_metrics["r2"] < 0.5:
        print("[train] WARNING: Linear Regression R2 < 0.5 -- check feature engineering.")
    if adv_metrics["r2"] <= lr_metrics["r2"]:
        print("[train] WARNING: Advanced model does NOT outperform baseline.")

    print("\n[train] Done.")


if __name__ == "__main__":
    main()
