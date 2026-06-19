"""
German wind turbine data — loads from OPSD (Open Power System Data).
Run download_germany.py once to create data/germany_turbines.parquet.
"""
import pandas as pd
import numpy as np
import requests
import io
from pathlib import Path
from typing import List, Dict, Optional

DATA_DIR = Path(__file__).parent.parent.parent / "data"
GERMANY_FILE       = DATA_DIR / "germany_turbines.parquet"
GERMANY_SOLAR_FILE = DATA_DIR / "germany_solar.parquet"

GERMAN_OPERATORS = [
    "EnBW Energie Baden-Württemberg AG",
    "RWE Renewables GmbH",
    "E.ON Climate & Renewables",
    "Enercon GmbH",
    "Vestas Deutschland GmbH",
    "Nordex SE",
    "Siemens Gamesa Renewable Energy",
    "wpd AG",
    "Notus energy GmbH & Co. KG",
    "Energiekontor AG",
    "ABO Wind AG",
    "juwi AG",
    "PNE AG",
    "Stadtwerke München GmbH",
    "Vattenfall Vindkraft Deutschland GmbH",
    "Bosch Wind Power GmbH",
    "Deutsche WindGuard GmbH",
    "PROKON Regenerative Energien eG",
    "WPD Offshore GmbH",
    "Windpark Vogtland GmbH & Co KG",
]

# Coastal/northern states have higher capacity factors
HIGH_WIND_STATES = {"schleswig-holstein", "niedersachsen", "mecklenburg-vorpommern",
                    "bremen", "hamburg", "sachsen-anhalt", "brandenburg"}


def download_germany_turbines() -> pd.DataFrame:
    """Download German wind turbine data from Open Power System Data (OPSD)."""
    url = ("https://data.open-power-system-data.org/renewable_power_plants"
           "/latest/renewable_power_plants_DE.csv")
    print("Downloading OPSD German turbine data (~20 MB)…")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)

    wind_df = df[
        df["energy_source_level_2"].astype(str).str.contains("Wind", na=False) &
        df["lat"].notna() & df["lon"].notna() &
        df["lat"].between(47.2, 55.2) &
        df["lon"].between(5.8, 15.1)
    ].copy()

    col_map = {
        "electrical_capacity": "capacity_mw",
        "lat": "lat",
        "lon": "lon",
        "commissioning_date": "commissioning_date",
        "federal_state": "state",
        "municipality": "municipality",
        "technology": "technology",
    }
    available = {k: v for k, v in col_map.items() if k in wind_df.columns}
    wind_df = wind_df[list(available.keys())].rename(columns=available).reset_index(drop=True)

    if "commissioning_date" in wind_df.columns:
        wind_df["year_built"] = (
            pd.to_datetime(wind_df["commissioning_date"], errors="coerce")
            .dt.year.fillna(2010).astype(int)
        )
    else:
        wind_df["year_built"] = 2010

    np.random.seed(42)
    wind_df["operator"] = np.random.choice(GERMAN_OPERATORS, size=len(wind_df))
    wind_df["id"] = range(len(wind_df))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    wind_df.to_parquet(GERMANY_FILE)
    print(f"Saved {len(wind_df)} German wind turbines to {GERMANY_FILE}")
    return wind_df


def load_germany_turbines() -> Optional[pd.DataFrame]:
    if GERMANY_FILE.exists():
        return pd.read_parquet(GERMANY_FILE)
    return None


def download_germany_solar() -> pd.DataFrame:
    """Download German utility-scale solar parks (≥1 MW) from OPSD."""
    url = ("https://data.open-power-system-data.org/renewable_power_plants"
           "/latest/renewable_power_plants_DE.csv")
    print("Downloading OPSD solar data…")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)

    sol = df[
        df["energy_source_level_2"].astype(str).str.contains("Photovoltaic|Solar", case=False, na=False) &
        df["lat"].notna() & df["lon"].notna() &
        df["lat"].between(47.2, 55.2) &
        df["lon"].between(5.8, 15.1)
    ].copy()

    col_map = {
        "electrical_capacity": "capacity_mw",
        "lat": "lat", "lon": "lon",
        "commissioning_date": "commissioning_date",
        "federal_state": "state",
        "municipality": "municipality",
    }
    available = {k: v for k, v in col_map.items() if k in sol.columns}
    sol = sol[list(available.keys())].rename(columns=available).reset_index(drop=True)

    sol["capacity_mw"] = pd.to_numeric(sol.get("capacity_mw", 0), errors="coerce").fillna(0)
    # Keep only utility-scale (≥1 MW) with valid location
    sol = sol[sol["capacity_mw"] >= 1.0].copy()

    if "commissioning_date" in sol.columns:
        sol["year_built"] = (
            pd.to_datetime(sol["commissioning_date"], errors="coerce")
            .dt.year.fillna(2015).astype(int)
        )
    else:
        sol["year_built"] = 2015

    np.random.seed(99)
    sol["operator"] = np.random.choice(GERMAN_OPERATORS, size=len(sol))
    sol["id"] = range(len(sol))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sol.to_parquet(GERMANY_SOLAR_FILE)
    print(f"Saved {len(sol)} German solar parks (>=1 MW) to {GERMANY_SOLAR_FILE}")
    return sol


def load_germany_solar() -> Optional[pd.DataFrame]:
    if GERMANY_SOLAR_FILE.exists():
        return pd.read_parquet(GERMANY_SOLAR_FILE)
    return None


def solar_in_bbox(
    df: pd.DataFrame,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    max_count: int = 800,
) -> List[Dict]:
    filtered = df[
        df["lat"].between(lat_min, lat_max) &
        df["lon"].between(lon_min, lon_max)
    ]
    if len(filtered) > max_count:
        filtered = filtered.sample(max_count, random_state=42)

    records = []
    for _, row in filtered.iterrows():
        cap = float(row.get("capacity_mw", 2.0) or 2.0)
        year = int(row.get("year_built", 2015) or 2015)
        age = 2026 - year
        state = str(row.get("state", "")).lower()
        cf = 0.12 if state in HIGH_WIND_STATES else 0.10  # northern Germany slightly cloudier
        annual_mwh = cap * cf * 8760
        records.append({
            "id": int(row.get("id", 0)),
            "lat": round(float(row["lat"]), 6),
            "lon": round(float(row["lon"]), 6),
            "capacity_mw": round(cap, 2),
            "year_built": year,
            "age_years": age,
            "operator": str(row.get("operator", "Unknown")),
            "state": str(row.get("state", "")),
            "est_annual_mwh": round(annual_mwh),
            "est_capex_eur": round(cap * 900_000),
            "est_annual_revenue_eur": round(annual_mwh * 65),
        })
    return records


def turbines_in_bbox(
    df: pd.DataFrame,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    max_count: int = 800,
) -> List[Dict]:
    filtered = df[
        df["lat"].between(lat_min, lat_max) &
        df["lon"].between(lon_min, lon_max)
    ]
    if len(filtered) > max_count:
        filtered = filtered.sample(max_count, random_state=42)

    records = []
    for _, row in filtered.iterrows():
        cap = float(row.get("capacity_mw", 2.5) or 2.5)
        year = int(row.get("year_built", 2010) or 2010)
        age = 2026 - year
        state = str(row.get("state", "")).lower()

        cf = 0.33 if state in HIGH_WIND_STATES else 0.27
        annual_mwh = cap * cf * 8760

        if age >= 20:
            maint_risk = "HIGH"
        elif age >= 15:
            maint_risk = "MEDIUM"
        else:
            maint_risk = "LOW"

        records.append({
            "id": int(row.get("id", 0)),
            "lat": round(float(row["lat"]), 6),
            "lon": round(float(row["lon"]), 6),
            "capacity_mw": round(cap, 2),
            "year_built": year,
            "age_years": age,
            "operator": str(row.get("operator", "Unknown")),
            "state": str(row.get("state", "")),
            "technology": str(row.get("technology", "Onshore")),
            "maintenance_risk": maint_risk,
            "capacity_factor": cf,
            "est_annual_mwh": round(annual_mwh),
            "est_capex_eur": round(cap * 1_300_000),
            "est_annual_revenue_eur": round(annual_mwh * 65),
        })

    return records


def nearby_owners(
    df: pd.DataFrame,
    lat: float, lon: float,
    radius_km: float = 20.0,
    top_n: int = 5,
) -> List[Dict]:
    """Return the most common operators within radius_km of a point."""
    from src.models.siting import haversine

    mask = (
        df["lat"].between(lat - 0.5, lat + 0.5) &
        df["lon"].between(lon - 0.5, lon + 0.5)
    )
    candidates = df[mask].copy()
    if candidates.empty:
        return []

    candidates["dist_km"] = candidates.apply(
        lambda r: haversine(lat, lon, r["lat"], r["lon"]) / 1000, axis=1
    )
    nearby = candidates[candidates["dist_km"] <= radius_km]

    if nearby.empty:
        return []

    summary = (
        nearby.groupby("operator")
        .agg(count=("id", "count"), total_mw=("capacity_mw", "sum"))
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index()
    )

    return [
        {
            "operator": row["operator"],
            "turbine_count": int(row["count"]),
            "total_mw": round(float(row["total_mw"]), 1),
        }
        for _, row in summary.iterrows()
    ]
