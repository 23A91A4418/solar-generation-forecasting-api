"""
tests/test_preprocessing.py
----------------------------
Unit tests for the feature engineering functions in scripts/preprocess.py.

Run:
    pytest tests/ -v
"""

import sys
import os
import math
import numpy as np
import pandas as pd
import pytest

# Allow importing scripts/ as a module from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.preprocess import (
    add_time_features,
    add_sinusoidal_features,
    add_interaction_features,
    filter_data,
    handle_missing_values,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df():
    """Minimal DataFrame that mimics preprocessed raw data."""
    return pd.DataFrame({
        "DATE_TIME": pd.to_datetime([
            "2020-06-15 06:00", "2020-06-15 12:00", "2020-06-15 18:00",
        ]),
        "IRRADIATION": [0.1, 0.9, 0.05],
        "AMBIENT_TEMPERATURE": [22.0, 30.0, 26.0],
        "MODULE_TEMPERATURE": [25.0, 48.0, 30.0],
        "AC_POWER": [50.0, 700.0, 40.0],
    })


@pytest.fixture()
def anomalous_df():
    return pd.DataFrame({
        "DATE_TIME": pd.to_datetime(["2020-06-15 00:00", "2020-06-15 12:00",
                                      "2020-06-15 14:00"]),
        "IRRADIATION": [0.0, -0.01, 0.8],       # first two should be filtered
        "AMBIENT_TEMPERATURE": [20.0, 25.0, 28.0],
        "MODULE_TEMPERATURE": [20.0, 25.0, 40.0],
        "AC_POWER": [0.0, 0.0, 600.0],
    })


# ── add_time_features ─────────────────────────────────────────────────

def test_hour_feature_range(sample_df):
    out = add_time_features(sample_df)
    assert "hour" in out.columns
    assert out["hour"].between(0, 24).all()


def test_month_feature_range(sample_df):
    out = add_time_features(sample_df)
    assert "month" in out.columns
    assert out["month"].between(1, 12).all()


def test_hour_value_correct(sample_df):
    out = add_time_features(sample_df)
    # Row index 1 is 12:00 → hour = 12.0
    assert out.loc[1, "hour"] == pytest.approx(12.0)


def test_month_value_correct(sample_df):
    out = add_time_features(sample_df)
    assert (out["month"] == 6).all()


# ── add_sinusoidal_features ───────────────────────────────────────────

def test_sin_hour_range(sample_df):
    df = add_time_features(sample_df)
    df = add_sinusoidal_features(df)
    assert df["sin_hour"].between(-1.0, 1.0).all()


def test_cos_hour_range(sample_df):
    df = add_time_features(sample_df)
    df = add_sinusoidal_features(df)
    assert df["cos_hour"].between(-1.0, 1.0).all()


def test_sin_cos_known_values(sample_df):
    """At hour=12, sin(2π·12/24)=sin(π)≈0, cos(2π·12/24)=cos(π)=-1."""
    df = add_time_features(sample_df)
    df = add_sinusoidal_features(df)
    noon = df[df["hour"] == pytest.approx(12.0)].iloc[0]
    assert noon["sin_hour"] == pytest.approx(0.0, abs=1e-9)
    assert noon["cos_hour"] == pytest.approx(-1.0, abs=1e-9)


# ── add_interaction_features ──────────────────────────────────────────

def test_irr_x_temp_equals_product(sample_df):
    out = add_interaction_features(sample_df)
    for _, row in out.iterrows():
        expected = row["IRRADIATION"] * row["AMBIENT_TEMPERATURE"]
        assert row["irr_x_temp"] == pytest.approx(expected)


def test_irr_x_temp_column_exists(sample_df):
    out = add_interaction_features(sample_df)
    assert "irr_x_temp" in out.columns


# ── filter_data ───────────────────────────────────────────────────────

def test_filter_removes_zero_irradiation(anomalous_df):
    filtered = filter_data(anomalous_df)
    assert (filtered["IRRADIATION"] > 0).all()


def test_filter_removes_negative_irradiation(anomalous_df):
    filtered = filter_data(anomalous_df)
    assert (filtered["IRRADIATION"] > 0).all()


def test_filter_keeps_valid_rows(anomalous_df):
    filtered = filter_data(anomalous_df)
    assert len(filtered) == 1  # only row with IRRADIATION=0.8 survives


# ── handle_missing_values ─────────────────────────────────────────────

def test_missing_values_filled(sample_df):
    df = sample_df.copy()
    df.loc[0, "IRRADIATION"] = float("nan")
    out = handle_missing_values(df)
    assert not out["IRRADIATION"].isna().any()


def test_rows_dropped_when_target_missing():
    df = pd.DataFrame({
        "DATE_TIME": pd.to_datetime(["2020-06-15 12:00"]),
        "IRRADIATION": [0.9],
        "AMBIENT_TEMPERATURE": [28.0],
        "MODULE_TEMPERATURE": [42.0],
        "AC_POWER": [float("nan")],
    })
    out = handle_missing_values(df)
    # AC_POWER NaN should be dropped
    assert len(out) == 0
