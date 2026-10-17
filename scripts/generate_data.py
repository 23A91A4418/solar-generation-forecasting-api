"""
scripts/generate_data.py
------------------------
Generates a realistic synthetic solar power generation dataset
that mirrors the Kaggle "Solar Power Generation Data" schema.

Run this if you don't have the real Kaggle dataset:
    python scripts/generate_data.py

Output: data/raw_data.csv
"""

import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

# ── Configuration ────────────────────────────────────────────────────
START_DATE = "2020-05-15"
END_DATE = "2020-11-15"          # 6 months
FREQ = "15min"                   # 15-minute intervals
PEAK_AC_KW = 850.0               # rough max AC power (kW) for Plant 1
PLANT_ID = 4135001               # matches Kaggle schema
SOURCE_KEY = "1BY6WEcLGh8j5v7"  # arbitrary inverter key


def irradiation_profile(hour: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Bell-curve irradiation centred on solar noon (12:00), scaled by month."""
    peak_hour = 12.0
    half_day = 7.0  # daylight half-window
    raw = np.exp(-0.5 * ((hour - peak_hour) / half_day) ** 2)
    # Seasonal amplitude: higher in summer (month 6), lower at edges
    season = 0.75 + 0.25 * np.cos(np.pi * (month - 6) / 6)
    irr = raw * season
    irr = np.clip(irr, 0, None)
    # Night mask
    irr[(hour < 6) | (hour > 18)] = 0.0
    return irr


def main():
    os.makedirs("data", exist_ok=True)

    times = pd.date_range(START_DATE, END_DATE, freq=FREQ)
    n = len(times)

    hour = times.hour + times.minute / 60.0
    month = times.month.to_numpy()
    hour_np = hour.to_numpy()

    # ── Irradiation (0–1.2 kW/m²) ────────────────────────────────────
    base_irr = irradiation_profile(hour_np, month)
    noise_irr = rng.normal(0, 0.03, n)
    cloud_factor = 1.0 - rng.beta(0.5, 5, n) * (base_irr > 0)
    irradiation = np.clip(base_irr * cloud_factor + noise_irr, 0, 1.25)

    # ── Temperatures (°C) ─────────────────────────────────────────────
    base_ambient = 25 + 5 * np.cos(np.pi * (month - 6) / 6) + \
                   5 * np.sin(np.pi * hour_np / 12)
    ambient_temp = base_ambient + rng.normal(0, 1.5, n)

    module_temp = ambient_temp + 25 * irradiation + rng.normal(0, 1.0, n)
    module_temp = np.clip(module_temp, ambient_temp, 80)

    # ── AC Power (kW) — physically motivated ──────────────────────────
    # Power ∝ irradiation, slightly penalized by high module temp
    efficiency = 1.0 - 0.004 * np.clip(module_temp - 25, 0, None)
    ac_power = PEAK_AC_KW * irradiation * efficiency
    ac_power += rng.normal(0, 5, n)
    ac_power = np.clip(ac_power, 0, PEAK_AC_KW * 1.05)
    ac_power[irradiation == 0] = 0.0  # no generation at night

    # ── DC Power — typically ~3–5 % higher than AC ────────────────────
    dc_power = ac_power * rng.uniform(1.03, 1.06, n)

    # ── Cumulative yield fields ───────────────────────────────────────
    daily_yield = np.zeros(n)
    total_yield = np.zeros(n)
    cum_day = 0.0
    cum_total = rng.uniform(6e6, 7e6)  # starting total kWh
    current_day = times[0].date()

    for i in range(n):
        if times[i].date() != current_day:
            cum_day = 0.0
            current_day = times[i].date()
        cum_day += ac_power[i] * 0.25  # kWh per 15 min
        cum_total += ac_power[i] * 0.25
        daily_yield[i] = cum_day
        total_yield[i] = cum_total

    df = pd.DataFrame({
        "DATE_TIME": times.strftime("%d-%m-%Y %H:%M"),
        "PLANT_ID": PLANT_ID,
        "SOURCE_KEY": SOURCE_KEY,
        "DC_POWER": dc_power.round(4),
        "AC_POWER": ac_power.round(4),
        "DAILY_YIELD": daily_yield.round(4),
        "TOTAL_YIELD": total_yield.round(4),
        "AMBIENT_TEMPERATURE": ambient_temp.round(4),
        "MODULE_TEMPERATURE": module_temp.round(4),
        "IRRADIATION": irradiation.round(6),
    })

    out_path = os.path.join("data", "raw_data.csv")
    df.to_csv(out_path, index=False)
    print(f"[generate_data] Saved {len(df):,} rows -> {out_path}")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
