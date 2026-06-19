"""
Fetch 1 year of ERA5 reanalysis from Open-Meteo archive API.
Computes monthly wind/solar statistics for any location — free, no API key.
"""
import numpy as np
import requests
import pandas as pd
from datetime import date, timedelta
from typing import Dict

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _wind_cf(v: float) -> float:
    """IEC Class II power curve — calibrated against Chinese SDWPF statistics."""
    CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0
    if v is None or np.isnan(v) or v < CUT_IN or v > CUT_OUT:
        return 0.0
    if v >= RATED:
        return 1.0
    return ((v - CUT_IN) / (RATED - CUT_IN)) ** 3


def fetch_annual_stats(lat: float, lon: float) -> Dict:
    """
    Returns monthly averages for wind speed, GHI, and capacity factors
    based on 1-year ERA5 data ending 5 days ago (archive lag).
    """
    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=365)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": "wind_speed_100m,shortwave_radiation,temperature_2m",
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_API, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()["hourly"]

    df = pd.DataFrame({
        "time":  pd.to_datetime(raw["time"]),
        "wind":  pd.to_numeric(raw.get("wind_speed_100m", []), errors="coerce"),
        "ghi":   pd.to_numeric(raw.get("shortwave_radiation", []), errors="coerce"),
        "temp":  pd.to_numeric(raw.get("temperature_2m", []), errors="coerce"),
    }).dropna(subset=["wind"])

    df["month"] = df["time"].dt.month
    df["wind_cf"]  = df["wind"].apply(_wind_cf)
    df["solar_cf"] = (df["ghi"].clip(0) / 1000.0 * 0.18).clip(0, 1)  # 18% efficiency

    monthly = df.groupby("month").agg(
        avg_wind=("wind", "mean"),
        avg_ghi=("ghi", "mean"),
        avg_temp=("temp", "mean"),
        wind_cf=("wind_cf", "mean"),
        solar_cf=("solar_cf", "mean"),
    ).round(3)

    result = []
    for m in range(1, 13):
        if m in monthly.index:
            r = monthly.loc[m]
            result.append({
                "month": MONTH_NAMES[m - 1],
                "month_num": m,
                "avg_wind_ms": round(float(r["avg_wind"]), 2),
                "avg_ghi_wm2": round(float(r["avg_ghi"]), 1),
                "avg_temp_c":  round(float(r["avg_temp"]), 1),
                "wind_capacity_factor":  round(float(r["wind_cf"]), 3),
                "solar_capacity_factor": round(float(r["solar_cf"]), 3),
            })

    annual_wind_cf  = float(df["wind_cf"].mean())
    annual_solar_cf = float(df["solar_cf"].mean())

    best_wind  = max(result, key=lambda x: x["wind_capacity_factor"])["month"]
    worst_wind = min(result, key=lambda x: x["wind_capacity_factor"])["month"]
    best_solar = max(result, key=lambda x: x["solar_capacity_factor"])["month"]

    return {
        "location": {"lat": lat, "lon": lon},
        "period": {"from": str(start_date), "to": str(end_date)},
        "monthly": result,
        "annual": {
            "avg_wind_ms": round(float(df["wind"].mean()), 2),
            "avg_ghi_wm2": round(float(df["ghi"].mean()), 1),
            "wind_capacity_factor":  round(annual_wind_cf, 3),
            "solar_capacity_factor": round(annual_solar_cf, 3),
            "best_wind_month":  best_wind,
            "worst_wind_month": worst_wind,
            "best_solar_month": best_solar,
        },
    }
