from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.power_predictor import PowerPredictor
from src.models.clustering import TurbineClustering
from src.models.optimizer import optimize_turbine_settings
from src.models.siting import find_best_locations, wind_resource_grid
from src.models.solar_predictor import SolarPredictor
from src.models.solar_siting import find_best_solar_locations, solar_power_kw
from src.api.weather import fetch_forecast, fetch_elevation_batch
from src.api.germany import (
    load_germany_turbines, download_germany_turbines,
    turbines_in_bbox, nearby_owners,
    load_germany_solar, download_germany_solar, solar_in_bbox,
)
from src.api.maintenance import score_maintenance_risk, find_maintenance_windows
from src.api.historical import fetch_annual_stats
import requests as _requests

app = FastAPI(title="Wind Turbine Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
SCADA_FILE  = Path(__file__).parent.parent.parent / "24798654" / "SDWPF_dataset" / "sdwpf_kddcup" / "sdwpf_245days_v1.csv"

# ── State loaded at startup ────────────────────────────────────────────────────
predictor: Optional[PowerPredictor] = None
clustering: Optional[TurbineClustering] = None
turbine_stats: Optional[pd.DataFrame] = None
solar_predictor: Optional[SolarPredictor] = None
germany_df: Optional[pd.DataFrame] = None
germany_solar_df: Optional[pd.DataFrame] = None
_scada_daily: Optional[pd.DataFrame] = None  # daily stats per turbine, loaded lazily


@app.on_event("startup")
def load_models():
    global predictor, clustering, turbine_stats, solar_predictor, germany_df, germany_solar_df
    try:
        predictor = PowerPredictor.load()
        print("Power predictor loaded")
    except Exception as e:
        print(f"No predictor found: {e}")

    try:
        clustering = TurbineClustering.load()
        print("Clustering model loaded")
    except Exception as e:
        print(f"No clustering found: {e}")

    stats_path = MODELS_DIR / "turbine_stats.parquet"
    if stats_path.exists():
        turbine_stats = pd.read_parquet(stats_path)
        print(f"Turbine stats loaded: {len(turbine_stats)} turbines")

    try:
        solar_predictor = SolarPredictor.load()
        print("Solar predictor loaded")
    except Exception as e:
        print(f"No solar predictor found (run train_solar.py): {e}")

    germany_df = load_germany_turbines()
    if germany_df is not None:
        print(f"German turbines loaded: {len(germany_df)}")
    else:
        print("No German turbine data (run download_germany.py)")

    germany_solar_df = load_germany_solar()
    if germany_solar_df is not None:
        print(f"German solar parks loaded: {len(germany_solar_df)}")
    else:
        print("No German solar data — will auto-download on first request")


# ── Schemas ────────────────────────────────────────────────────────────────────

class WeatherInput(BaseModel):
    wind_speed_ms: float
    wind_direction_deg: float
    temperature_c: float


class BoundingBox(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class PredictRequest(BaseModel):
    weather: WeatherInput
    bbox: Optional[BoundingBox] = None


class OptimizeRequest(BaseModel):
    wind_speed_ms: float
    wind_direction_deg: float
    turbine_ids: Optional[List[int]] = None


class SitingRequest(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    n_turbines: int = 5
    forecast_hours: int = 48
    grid_n: int = 10  # siting grid resolution (grid_n × grid_n candidates, max 10)


class RegionRequest(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"

@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIST / "index.html"))

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "predictor_loaded": predictor is not None,
        "clustering_loaded": clustering is not None,
        "n_turbines": len(turbine_stats) if turbine_stats is not None else 0,
    }


@app.get("/api/turbines")
def get_turbines(
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
):
    if turbine_stats is None:
        raise HTTPException(503, "Models not loaded. Run train.py first.")

    df = turbine_stats.copy()

    if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]):
        df = df[
            df["lat"].between(lat_min, lat_max) &
            df["lon"].between(lon_min, lon_max)
        ]

    cluster_labels = {
        -1: "Anomalous", 0: "High Performers", 1: "Average",
        2: "Wake-Affected", 3: "Underperformers",
    }

    records = []
    for _, row in df.iterrows():
        cid = int(row.get("cluster_id", 0))
        records.append({
            "id": int(row["TurbID"]),
            "lat": round(float(row["lat"]), 6),
            "lon": round(float(row["lon"]), 6),
            "cluster_id": cid,
            "cluster_label": cluster_labels.get(cid, f"Group {cid}"),
            "performance_score": round(float(row["performance_score"]), 1),
            "mean_efficiency": round(float(row["mean_efficiency"]), 3),
            "mean_yaw_misalignment": round(float(row["mean_yaw_misalignment"]), 1),
            "availability": round(float(row["availability"]), 3),
            "mean_patv_kw": round(float(row["mean_patv"]), 1),
        })

    return {"turbines": records, "total": len(records)}


@app.get("/api/turbines/{turbine_id}")
def get_turbine(turbine_id: int):
    if turbine_stats is None:
        raise HTTPException(503, "Models not loaded.")
    row = turbine_stats[turbine_stats["TurbID"] == turbine_id]
    if row.empty:
        raise HTTPException(404, f"Turbine {turbine_id} not found")
    r = row.iloc[0].to_dict()
    r = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}
    return r


def _load_scada_daily() -> Optional[pd.DataFrame]:
    """Load SDWPF SCADA CSV, compute daily stats + fleet empirical power curve."""
    global _scada_daily
    if _scada_daily is not None:
        return _scada_daily
    if not SCADA_FILE.exists():
        return None

    df = pd.read_csv(SCADA_FILE, usecols=["TurbID", "Day", "Wspd", "Patv"])
    df["Day"]  = pd.to_numeric(df["Day"],  errors="coerce")
    df["Patv"] = pd.to_numeric(df["Patv"], errors="coerce").clip(0, 1500)
    df["Wspd"] = pd.to_numeric(df["Wspd"], errors="coerce").clip(0, 30)

    daily = df.groupby(["TurbID", "Day"]).agg(
        actual_kw   = ("Patv", "mean"),
        avg_wind_ms = ("Wspd", "mean"),
        n_samples   = ("Patv", "count"),
    ).reset_index()

    # Fleet empirical power curve: for each 1 m/s wind bin, median production
    # across ALL turbines across ALL days. Used to compute per-turbine "expected" power.
    daily["wind_bin"] = daily["avg_wind_ms"].fillna(0).clip(0, 20).round(0).astype(int)
    fleet_curve = (
        daily.groupby("wind_bin")["actual_kw"]
        .median()
        .rename("fleet_expected_kw")
        .reset_index()
    )
    daily = daily.merge(fleet_curve, on="wind_bin", how="left")

    _scada_daily = daily
    return daily


@app.get("/api/turbines/{turbine_id}/history")
def turbine_history(turbine_id: int):
    """
    Return 245-day power history (actual vs power-curve prediction) for a turbine.
    Actual data: SDWPF SCADA.
    Predicted: calibrated IEC power curve scaled by this turbine's efficiency.
    """
    daily = _load_scada_daily()
    if daily is None:
        raise HTTPException(503, "SCADA data not available")

    turb = daily[daily["TurbID"] == turbine_id].sort_values("Day").copy()
    turb = turb.dropna(subset=["actual_kw", "avg_wind_ms"])
    if turb.empty:
        raise HTTPException(404, f"No SCADA data for turbine {turbine_id}")

    # Per-turbine efficiency from training stats
    eff = 0.85
    if turbine_stats is not None:
        t = turbine_stats[turbine_stats["TurbID"] == turbine_id]
        if not t.empty:
            eff = float(t.iloc[0].get("mean_efficiency", 0.85))
            eff = min(1.2, max(0.3, eff))

    # predicted_kw = fleet median power at the same wind speed bin (empirical fleet curve).
    # A healthy turbine tracks the fleet curve → PR ≈ 100%.
    # PR < 90% signals under-performance; PR > 110% is exceptional.
    RATED_KW = 1500.0
    turb["predicted_kw"] = turb["fleet_expected_kw"].fillna(turb["actual_kw"].mean()).clip(0, RATED_KW)

    mean_actual = float(turb["actual_kw"].mean())
    mean_pred   = float(turb["predicted_kw"].mean())
    perf_ratio  = round(mean_actual / mean_pred * 100, 1) if mean_pred > 0 else 0

    return {
        "turbine_id": turbine_id,
        "n_days": len(turb),
        "days": turb[["Day", "actual_kw", "predicted_kw", "avg_wind_ms"]].round(1).to_dict("records"),
        "stats": {
            "mean_actual_kw":    round(mean_actual, 1),
            "mean_predicted_kw": round(mean_pred, 1),
            "mean_wind_ms":      round(float(turb["avg_wind_ms"].mean()), 2),
            "performance_ratio": perf_ratio,
            "efficiency":        round(eff * 100, 1),
            "rated_kw":          RATED_KW,
        },
    }


import datetime as _dt

@app.get("/api/turbines/{turbine_id}/monthly")
def turbine_monthly(turbine_id: int):
    """Monthly-aggregated performance (8 months of SDWPF data + 3-month seasonal forecast)."""
    daily = _load_scada_daily()
    if daily is None:
        raise HTTPException(503, "SCADA data not available")

    turb = (
        daily[daily["TurbID"] == turbine_id]
        .dropna(subset=["actual_kw", "avg_wind_ms", "fleet_expected_kw"])
        .sort_values("Day")
        .copy()
    )
    if turb.empty:
        raise HTTPException(404, f"No data for turbine {turbine_id}")

    # SDWPF data starts 2020-01-01 per KDD Cup paper
    start = _dt.date(2020, 1, 1)
    turb["date"]  = turb["Day"].apply(lambda d: start + _dt.timedelta(days=int(d) - 1))
    turb["month"] = turb["date"].apply(lambda d: d.month)
    turb["year"]  = turb["date"].apply(lambda d: d.year)

    agg = (
        turb.groupby(["year", "month"])
        .agg(
            actual_kw   = ("actual_kw",          "mean"),
            predicted_kw= ("fleet_expected_kw",   "mean"),
            avg_wind_ms = ("avg_wind_ms",         "mean"),
            n_days      = ("Day",                 "count"),
        )
        .reset_index()
        .sort_values(["year", "month"])
    )

    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    records = []
    for _, row in agg.iterrows():
        records.append({
            "month": int(row["month"]),
            "year":  int(row["year"]),
            "month_label": f"{MONTHS[int(row['month'])-1]} {int(row['year'])}",
            "actual_kw":    round(float(row["actual_kw"]), 1),
            "predicted_kw": round(float(row["predicted_kw"]), 1),
            "avg_wind_ms":  round(float(row["avg_wind_ms"]), 2),
            "n_days":       int(row["n_days"]),
            "is_forecast":  False,
        })

    # 3-month seasonal forecast: reuse same calendar month from data
    last_year, last_month = int(agg.iloc[-1]["year"]), int(agg.iloc[-1]["month"])
    for i in range(1, 4):
        fm = (last_month - 1 + i) % 12 + 1
        fy = last_year + (last_month + i - 1) // 12
        same = turb[turb["month"] == fm]
        if same.empty:
            same = turb
        records.append({
            "month": fm,
            "year":  fy,
            "month_label": f"{MONTHS[fm-1]} (fcst)",
            "actual_kw":    None,
            "predicted_kw": round(float(same["actual_kw"].mean()), 1),
            "avg_wind_ms":  round(float(same["avg_wind_ms"].mean()), 2),
            "n_days":       0,
            "is_forecast":  True,
        })

    mean_act  = float(turb["actual_kw"].mean())
    mean_pred = float(turb["fleet_expected_kw"].mean())
    return {
        "turbine_id": turbine_id,
        "months": records,
        "stats": {
            "mean_actual_kw":    round(mean_act, 1),
            "mean_predicted_kw": round(mean_pred, 1),
            "mean_wind_ms":      round(float(turb["avg_wind_ms"].mean()), 2),
            "performance_ratio": round(mean_act / max(mean_pred, 1e-6) * 100, 1),
        },
    }


@app.get("/api/site/performance")
def site_performance(
    lat: float = Query(...),
    lon: float = Query(...),
    capacity_mw: float = Query(2.0),
    site_type: str = Query("wind"),
):
    """ERA5 monthly stats + 3-month seasonal forecast for a lat/lon. Used for German sites."""
    try:
        stats = fetch_annual_stats(lat, lon)
    except Exception as e:
        raise HTTPException(502, f"ERA5 unavailable: {e}")

    monthly = stats.get("monthly", [])
    for m in monthly:
        cf = m.get("wind_capacity_factor" if site_type == "wind" else "solar_capacity_factor", 0)
        m["estimated_kw"] = round(capacity_mw * 1000 * float(cf))
        m["is_forecast"]  = False
        m["month_label"]  = m.get("month", "")  # already a name like "Jan"
        m["month"]        = int(m.get("month_num", 1))

    last_month = int(monthly[-1]["month"]) if monthly else 12
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    forecast = []
    for i in range(1, 4):
        fm  = (last_month - 1 + i) % 12 + 1
        ref = next((m for m in monthly if m["month"] == fm), monthly[0] if monthly else {})
        if ref:
            fcopy = dict(ref)
            fcopy["month"]       = fm
            fcopy["month_label"] = MONTHS[fm - 1] + " +"
            fcopy["is_forecast"] = True
            forecast.append(fcopy)

    return {
        "site_type":   site_type,
        "capacity_mw": capacity_mw,
        "months":      monthly + forecast,
        "annual":      stats.get("annual", {}),
    }


@app.post("/api/predict")
def predict_power(req: PredictRequest):
    if predictor is None or turbine_stats is None:
        raise HTTPException(503, "Models not loaded. Run train.py first.")

    df = turbine_stats.copy()

    if req.bbox:
        df = df[
            df["lat"].between(req.bbox.lat_min, req.bbox.lat_max) &
            df["lon"].between(req.bbox.lon_min, req.bbox.lon_max)
        ]

    result = predictor.predict_from_weather(
        df,
        wind_speed=req.weather.wind_speed_ms,
        wind_dir=req.weather.wind_direction_deg,
        temperature=req.weather.temperature_c,
    )

    total_mw = result["predicted_kw"].sum() / 1000
    return {
        "total_predicted_mw": round(total_mw, 2),
        "n_turbines": len(result),
        "turbines": result.rename(columns={"predicted_kw": "predicted_power_kw"}).to_dict("records"),
    }


@app.post("/api/optimize")
def optimize_settings(req: OptimizeRequest):
    if turbine_stats is None:
        raise HTTPException(503, "Models not loaded. Run train.py first.")

    df = turbine_stats.copy()
    if req.turbine_ids:
        df = df[df["TurbID"].isin(req.turbine_ids)]

    # Use historical mean nacelle direction as proxy for current setting
    current_ndir = {
        int(r["TurbID"]): float(r["mean_yaw_misalignment"]) + req.wind_direction_deg
        for _, r in df.iterrows()
    }
    result = optimize_turbine_settings(
        df,
        wind_speed=req.wind_speed_ms,
        wind_direction=req.wind_direction_deg,
        current_ndir=current_ndir,
    )

    total_gain = result["estimated_power_gain_kw"].sum()
    return {
        "total_power_gain_kw": round(total_gain, 1),
        "total_power_gain_mw": round(total_gain / 1000, 3),
        "turbines": result.to_dict("records"),
    }


@app.get("/api/weather")
def get_weather(lat: float = Query(...), lon: float = Query(...), hours: int = 24):
    try:
        forecast = fetch_forecast(lat, lon, hours)
        return {"forecast": forecast, "location": {"lat": lat, "lon": lon}}
    except Exception as e:
        raise HTTPException(502, f"Weather API error: {e}")


@app.get("/api/clusters")
def get_clusters():
    if turbine_stats is None or clustering is None:
        raise HTTPException(503, "Models not loaded.")

    summary = clustering.get_cluster_summary(turbine_stats)
    return {"clusters": summary.to_dict("records"), "n_clusters": int(clustering.n_clusters_)}


# ── Region analytics ───────────────────────────────────────────────────────────

@app.post("/api/region/summary")
def region_summary(req: RegionRequest):
    """Aggregate performance stats for all turbines in a bounding box."""
    if turbine_stats is None:
        raise HTTPException(503, "Models not loaded.")

    df = turbine_stats.copy()
    df = df[
        df["lat"].between(req.lat_min, req.lat_max) &
        df["lon"].between(req.lon_min, req.lon_max)
    ]

    if df.empty:
        return {"n_turbines": 0, "message": "No turbines in this region"}

    cluster_dist = df["cluster_id"].value_counts().to_dict() if "cluster_id" in df.columns else {}
    cluster_labels = {-1: "Anomalous", 0: "High Performers", 1: "Average",
                      2: "Wake-Affected", 3: "Underperformers"}

    return {
        "n_turbines": len(df),
        "avg_performance_score": round(float(df["performance_score"].mean()), 1),
        "avg_efficiency": round(float(df["mean_efficiency"].mean()), 3),
        "avg_availability": round(float(df["availability"].mean()), 3),
        "avg_yaw_misalignment_deg": round(float(df["mean_yaw_misalignment"].mean()), 1),
        "total_avg_power_kw": round(float(df["mean_patv"].sum()), 1),
        "best_turbine_id": int(df.loc[df["performance_score"].idxmax(), "TurbID"]),
        "worst_turbine_id": int(df.loc[df["performance_score"].idxmin(), "TurbID"]),
        "cluster_distribution": {
            cluster_labels.get(int(k), f"Group {k}"): int(v)
            for k, v in cluster_dist.items()
        },
        "turbines": df[["TurbID", "lat", "lon", "performance_score", "cluster_id",
                         "mean_efficiency", "availability", "mean_patv"]].to_dict("records"),
    }


# ── Wind resource heatmap ──────────────────────────────────────────────────────

@app.post("/api/region/wind-resource")
def region_wind_resource(req: RegionRequest):
    """
    Return a grid of wind resource scores for a region.
    Used to render a heatmap overlay on the map.
    """
    center_lat = (req.lat_min + req.lat_max) / 2
    center_lon = (req.lon_min + req.lon_max) / 2

    try:
        forecast = fetch_forecast(center_lat, center_lon, hours=48)
    except Exception as e:
        raise HTTPException(502, f"Weather fetch failed: {e}")

    grid = wind_resource_grid(
        req.lat_min, req.lat_max,
        req.lon_min, req.lon_max,
        forecast, grid_n=20,
    )

    avg_wind = float(np.mean([f["wind_speed_ms"] for f in forecast]))
    avg_power = float(np.mean([f.get("wind_speed_ms", 0) ** 3 for f in forecast]))

    return {
        "center": {"lat": center_lat, "lon": center_lon},
        "avg_wind_speed_ms": round(avg_wind, 2),
        "forecast_hours": len(forecast),
        "grid_points": grid,
        "note": "Wind speed assumed uniform within bbox — grid shows relative resource",
    }


# ── Turbine siting ─────────────────────────────────────────────────────────────

@app.post("/api/siting")
def turbine_siting(req: SitingRequest):
    """
    Find the best N locations for new wind turbines in a region.
    Accounts for wind resource, wake avoidance, and minimum spacing.
    """
    if turbine_stats is None:
        raise HTTPException(503, "Models not loaded.")

    center_lat = (req.lat_min + req.lat_max) / 2
    center_lon = (req.lon_min + req.lon_max) / 2

    # Fetch weather for the region
    try:
        forecast = fetch_forecast(center_lat, center_lon, hours=req.forecast_hours)
    except Exception as e:
        raise HTTPException(502, f"Weather fetch failed: {e}")

    grid_n = min(req.grid_n, 10)  # cap at 10×10=100 pts to fit Open-Meteo URL limit

    # Include turbines slightly outside bbox so wake effects propagate in
    BUFFER = 0.025  # ~2.5 km in degrees
    existing = turbine_stats[
        turbine_stats["lat"].between(req.lat_min - BUFFER, req.lat_max + BUFFER) &
        turbine_stats["lon"].between(req.lon_min - BUFFER, req.lon_max + BUFFER)
    ][["lat", "lon", "TurbID"]].to_dict("records")

    # Fetch elevation for candidate grid from Open-Meteo (free, no key)
    lats_grid = list(np.linspace(req.lat_min, req.lat_max, grid_n))
    lons_grid = list(np.linspace(req.lon_min, req.lon_max, grid_n))
    flat_lats = [lat for lat in lats_grid for _ in lons_grid]
    flat_lons = [lon for _ in lats_grid for lon in lons_grid]
    try:
        elev_flat = fetch_elevation_batch(flat_lats, flat_lons)
        elevations = np.array(elev_flat).reshape(grid_n, grid_n)
    except Exception as e:
        print(f"Elevation fetch failed ({e}), using plateau default")
        elevations = None  # siting.py will use Inner Mongolia plateau default

    # Find best locations with terrain-aware scoring
    locations = find_best_locations(
        lat_min=req.lat_min, lat_max=req.lat_max,
        lon_min=req.lon_min, lon_max=req.lon_max,
        n_turbines=req.n_turbines,
        existing_turbines=existing,
        forecast=forecast,
        grid_n=grid_n,
        elevations=elevations,
    )

    if not locations:
        return {"message": "No valid locations found — region may be too small or too crowded.",
                "locations": []}

    total_aep = sum(loc["estimated_aep_mwh_per_year"] for loc in locations)
    total_avg_kw = sum(loc["estimated_avg_power_kw"] for loc in locations)

    return {
        "n_requested": req.n_turbines,
        "n_found": len(locations),
        "existing_turbines_in_region": len(existing),
        "total_estimated_avg_power_kw": round(total_avg_kw, 1),
        "total_estimated_aep_mwh_per_year": round(total_aep, 0),
        "weather_used": {
            "avg_wind_speed_ms": round(float(np.mean([f["wind_speed_ms"] or 0 for f in forecast])), 2),
            "avg_wind_dir_deg": round(float(np.mean([f["wind_direction_deg"] or 0 for f in forecast])), 1),
        },
        "terrain_data": "Open-Meteo elevation API" if elevations is not None else "fallback (no terrain data)",
        "locations": locations,
    }


# ── Solar endpoints ───────────────────────────────────────────────────────────

class SolarWeatherInput(BaseModel):
    ghi_wm2: float           # Global Horizontal Irradiance (W/m²)
    temperature_c: float
    humidity_pct: float = 50.0
    hour: float = 12.0
    month: int = 6
    day_of_year: int = 170


class SolarSitingRequest(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    n_sites: int = 5
    site_capacity_mw: float = 10.0
    grid_n: int = 10


@app.post("/api/solar/predict")
def solar_predict(req: SolarWeatherInput):
    """Predict solar capacity factor and power for a given irradiance scenario."""
    if solar_predictor is None:
        # Fall back to physics model when ML model not yet trained
        cf = solar_power_kw(req.ghi_wm2, req.temperature_c, 1.0)
        cf = min(1.0, cf / 0.17)  # 0.17 kW per 1 kW at 1000 W/m²
    else:
        cf = solar_predictor.predict_cf(
            ghi=req.ghi_wm2,
            temperature=req.temperature_c,
            humidity=req.humidity_pct,
            hour=req.hour,
            month=req.month,
            day_of_year=req.day_of_year,
        )
    return {
        "capacity_factor": round(cf, 3),
        "estimated_power_per_mw_kw": round(cf * 1000, 1),
        "ghi_wm2": req.ghi_wm2,
        "temperature_c": req.temperature_c,
        "model": "LightGBM" if solar_predictor else "physics-fallback",
    }


@app.post("/api/solar/siting")
def solar_siting_endpoint(req: SolarSitingRequest):
    """Find best N locations for solar farms in a bounding box."""
    center_lat = (req.lat_min + req.lat_max) / 2
    center_lon = (req.lon_min + req.lon_max) / 2

    try:
        forecast = fetch_forecast(center_lat, center_lon, hours=48)
    except Exception as e:
        raise HTTPException(502, f"Weather fetch failed: {e}")

    grid_n = min(req.grid_n, 10)
    lats_grid = list(np.linspace(req.lat_min, req.lat_max, grid_n))
    lons_grid = list(np.linspace(req.lon_min, req.lon_max, grid_n))
    flat_lats = [lat for lat in lats_grid for _ in lons_grid]
    flat_lons = [lon for _ in lats_grid for lon in lons_grid]

    try:
        elev_flat  = fetch_elevation_batch(flat_lats, flat_lons)
        elevations = np.array(elev_flat).reshape(grid_n, grid_n)
    except Exception as e:
        print(f"Elevation fetch failed: {e}")
        elevations = None

    locations = find_best_solar_locations(
        lat_min=req.lat_min, lat_max=req.lat_max,
        lon_min=req.lon_min, lon_max=req.lon_max,
        n_sites=req.n_sites,
        forecast=forecast,
        grid_n=grid_n,
        elevations=elevations,
        site_capacity_mw=req.site_capacity_mw,
    )

    if not locations:
        return {"message": "No valid locations found.", "locations": []}

    avg_ghi = float(np.mean([f.get("shortwave_radiation_wm2") or 0 for f in forecast]))
    total_aep = sum(loc["estimated_aep_mwh_per_year"] for loc in locations)
    avg_cf    = sum(loc["estimated_capacity_factor"]   for loc in locations) / len(locations)

    return {
        "n_requested": req.n_sites,
        "n_found":     len(locations),
        "avg_ghi_wm2": round(avg_ghi, 1),
        "avg_capacity_factor": round(avg_cf, 3),
        "total_estimated_aep_mwh_per_year": round(total_aep, 0),
        "solar_model": "LightGBM" if solar_predictor else "physics",
        "locations": locations,
    }


# ── Maintenance endpoints ──────────────────────────────────────────────────────

@app.get("/api/maintenance/risk")
def maintenance_risk_all():
    """
    Score every Chinese turbine's maintenance urgency using SDWPF metrics.
    Returns turbines sorted by risk (worst first).
    """
    if turbine_stats is None:
        raise HTTPException(503, "Models not loaded.")

    results = []
    for _, row in turbine_stats.iterrows():
        r = score_maintenance_risk(
            turbine_id=int(row["TurbID"]),
            performance_score=float(row.get("performance_score", 50)),
            availability=float(row.get("availability", 0.9)),
            yaw_misalignment=float(row.get("mean_yaw_misalignment", 0)),
            mean_efficiency=float(row.get("mean_efficiency", 0.85)),
            age_years=0,
        )
        results.append(r)

    results.sort(key=lambda x: x["risk_score"], reverse=True)
    critical = [r for r in results if r["risk_level"] == "CRITICAL"]
    high     = [r for r in results if r["risk_level"] == "HIGH"]
    return {
        "total": len(results),
        "critical_count": len(critical),
        "high_count": len(high),
        "top_10_at_risk": results[:10],
    }


@app.get("/api/maintenance/windows")
def maintenance_windows(
    lat: float = Query(40.5),
    lon: float = Query(108.5),
    days: int = Query(16),
):
    """Find the best low-wind windows for scheduling maintenance (next 16 days)."""
    try:
        windows = find_maintenance_windows(lat, lon, days)
        return {"windows": windows, "location": {"lat": lat, "lon": lon}}
    except Exception as e:
        raise HTTPException(502, f"Forecast API error: {e}")


# ── Germany turbine endpoints ──────────────────────────────────────────────────

@app.get("/api/germany/turbines")
def get_germany_turbines(
    lat_min: float = Query(47.2),
    lat_max: float = Query(55.2),
    lon_min: float = Query(5.8),
    lon_max: float = Query(15.1),
):
    global germany_df
    if germany_df is None:
        # Try auto-download on first request
        try:
            germany_df = download_germany_turbines()
        except Exception as e:
            raise HTTPException(503, f"German turbine data not available. Run download_germany.py. Error: {e}")

    records = turbines_in_bbox(germany_df, lat_min, lat_max, lon_min, lon_max, max_count=1500)
    return {"turbines": records, "total": len(records)}


@app.get("/api/germany/nearby-owners")
def germany_nearby_owners(lat: float = Query(...), lon: float = Query(...)):
    """Return top operators within 20 km of a point (for siting location context)."""
    if germany_df is None:
        return {"owners": [], "message": "German data not loaded"}
    owners = nearby_owners(germany_df, lat, lon)
    return {"owners": owners, "location": {"lat": lat, "lon": lon}}


@app.post("/api/germany/predict")
def predict_germany(req: PredictRequest):
    """
    Predict power output for German turbines using the Chinese-trained power curve.
    Transfer: apply the LightGBM model's learned wind→power relationship,
    scaled by each turbine's rated capacity vs the Chinese fleet average (1.5 MW).
    """
    if predictor is None or germany_df is None:
        raise HTTPException(503, "Models or Germany data not loaded.")

    if req.bbox is None:
        raise HTTPException(400, "bbox required for Germany prediction")

    records = turbines_in_bbox(
        germany_df,
        req.bbox.lat_min, req.bbox.lat_max,
        req.bbox.lon_min, req.bbox.lon_max,
        max_count=200,
    )
    if not records:
        return {"total_predicted_mw": 0, "n_turbines": 0, "turbines": []}

    # Build a template turbine DF for the Chinese model (neutral yaw/efficiency)
    template = pd.DataFrame([{
        "TurbID": r["id"],
        "lat": r["lat"], "lon": r["lon"],
        "cluster_id": 1,
        "performance_score": 65.0,
        "mean_efficiency": 0.85,
        "mean_yaw_misalignment": 0.0,
        "availability": 0.95,
        "mean_patv": 700.0,
        "x_norm": 0.0, "y_norm": 0.0,
        "wake_upstream_count": 0.0,
        "wake_upstream_avg_wspd": req.weather.wind_speed_ms,
        "wake_upstream_avg_patv": 0.0,
    } for r in records])

    pred_df = predictor.predict_from_weather(
        template,
        wind_speed=req.weather.wind_speed_ms,
        wind_dir=req.weather.wind_direction_deg,
        temperature=req.weather.temperature_c,
    )

    # Scale by rated capacity (Chinese fleet avg ≈ 1.5 MW, clip CF at 1.0)
    CHINESE_RATED_KW = 1500.0
    cap_map = {r["id"]: r["capacity_mw"] * 1000 for r in records}

    result_turbines = []
    total_kw = 0.0
    for _, row in pred_df.iterrows():
        rated_kw = cap_map.get(int(row["TurbID"]), 1500)
        cf = min(1.0, max(0.0, row["predicted_kw"] / CHINESE_RATED_KW))
        pred_kw = cf * rated_kw
        total_kw += pred_kw
        result_turbines.append({
            "id": int(row["TurbID"]),
            "lat": round(float(row["lat"]), 6),
            "lon": round(float(row["lon"]), 6),
            "predicted_power_kw": round(pred_kw, 1),
        })

    return {
        "total_predicted_mw": round(total_kw / 1000, 2),
        "n_turbines": len(result_turbines),
        "turbines": result_turbines,
    }


@app.get("/api/germany/solar")
def get_germany_solar(
    lat_min: float = Query(47.2),
    lat_max: float = Query(55.2),
    lon_min: float = Query(5.8),
    lon_max: float = Query(15.1),
):
    """Return German utility-scale solar parks (≥1 MW) in bbox."""
    global germany_solar_df
    if germany_solar_df is None:
        try:
            germany_solar_df = download_germany_solar()
        except Exception as e:
            raise HTTPException(503, f"Solar data unavailable: {e}")
    records = solar_in_bbox(germany_solar_df, lat_min, lat_max, lon_min, lon_max, max_count=1000)
    return {"parks": records, "total": len(records)}


# ── Historical stats ───────────────────────────────────────────────────────────

@app.get("/api/historical/stats")
def historical_stats(lat: float = Query(...), lon: float = Query(...)):
    """
    Fetch 1-year ERA5 reanalysis data and return monthly wind+solar statistics.
    Uses Open-Meteo archive — free, no key required.
    """
    try:
        stats = fetch_annual_stats(lat, lon)
        return stats
    except Exception as e:
        raise HTTPException(502, f"Historical data fetch failed: {e}")


# ── Electricity price proxy ────────────────────────────────────────────────────

@app.get("/api/electricity/price")
def electricity_price():
    """
    Proxy current German day-ahead electricity spot price from euenergy.live.
    Returns EUR/MWh. Falls back to a recent average if the API is unavailable.
    """
    try:
        resp = _requests.get("https://api.euenergy.live/api/v1/de/current", timeout=5)
        data = resp.json()
        # euenergy returns: {"price": 68.5, "unit": "EUR/MWh", ...}
        price = float(data.get("price") or data.get("value") or 68.0)
        return {
            "price_eur_mwh": round(price, 2),
            "country": "DE",
            "unit": "EUR/MWh",
            "source": "euenergy.live",
            "live": True,
        }
    except Exception:
        # Fallback to recent German average
        return {
            "price_eur_mwh": 68.0,
            "country": "DE",
            "unit": "EUR/MWh",
            "source": "fallback",
            "live": False,
        }


# ── Serve built React frontend (must be last) ──────────────────────────────────
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        return FileResponse(str(FRONTEND_DIST / "index.html"))
