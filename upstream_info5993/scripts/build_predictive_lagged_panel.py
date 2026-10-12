from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("data/processed/modelling_panel/extended_all_fua_pm25_edgar_power_spatial_ghs_fua_1998_2022.csv")
OUT = Path("data/processed/modelling_panel/extended_all_fua_pm25_edgar_power_spatial_lagged_1998_2022.csv")
META = Path("data/metadata/extended_lagged_predictive_panel_1998_2022.json")
TARGET = "pm25_mean_ug_m3"


def main() -> int:
    df = pd.read_csv(SOURCE).sort_values(["eFUA_ID", "year"]).copy()
    grouped = df.groupby("eFUA_ID", group_keys=False)

    df["pm25_lag1"] = grouped[TARGET].shift(1)
    df["pm25_lag2"] = grouped[TARGET].shift(2)
    df["pm25_lag3"] = grouped[TARGET].shift(3)
    df["pm25_roll3_mean_lag1"] = grouped[TARGET].shift(1).groupby(df["eFUA_ID"]).rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
    df["pm25_roll5_mean_lag1"] = grouped[TARGET].shift(1).groupby(df["eFUA_ID"]).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    df["pm25_change_lag1"] = df["pm25_lag1"] - df["pm25_lag2"]
    df["pm25_change_lag2"] = df["pm25_lag2"] - df["pm25_lag3"]

    # Lagged emissions provide source persistence without using the target.
    emission_cols = [
        "edgar_pm25_emissions_tonnes_per_km2",
        "edgar_nox_emissions_tonnes_per_km2",
        "edgar_so2_emissions_tonnes_per_km2",
        "edgar_nh3_emissions_tonnes_per_km2",
        "edgar_nmvoc_emissions_tonnes_per_km2",
    ]
    for col in emission_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = grouped[col].shift(1)
            df[f"{col}_change_lag1"] = grouped[col].shift(1) - grouped[col].shift(2)

    lag_cols = [c for c in df.columns if "lag" in c or "roll" in c]
    before = len(df)
    modelling = df.dropna(subset=["pm25_lag1", "pm25_lag2", "pm25_roll3_mean_lag1"]).copy()
    modelling["has_predictive_lags"] = 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    modelling.to_csv(OUT, index=False, encoding="utf-8-sig")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "output": str(OUT),
        "rows_before_lag_drop": int(before),
        "rows_after_lag_drop": int(len(modelling)),
        "fua_count": int(modelling["eFUA_ID"].nunique()),
        "year_min": int(modelling["year"].min()),
        "year_max": int(modelling["year"].max()),
        "lag_columns": lag_cols,
        "leakage_note": "All PM2.5 lag and rolling features use only prior-year values within each FUA.",
    }
    META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
