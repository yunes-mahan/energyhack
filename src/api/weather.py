import requests
from typing import List, Dict, Any

OPEN_METEO_FORECAST  = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ELEVATION = "https://api.open-meteo.com/v1/elevation"


def fetch_forecast(lat: float, lon: float, hours: int = 48) -> List[Dict[str, Any]]:
    """Fetch hourly wind + temperature + cloud cover from Open-Meteo (no API key)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "wind_speed_10m,wind_direction_10m,temperature_2m,"
            "wind_speed_100m,wind_direction_100m,cloud_cover,"
            "shortwave_radiation,direct_normal_irradiance,diffuse_radiation"
        ),
        "wind_speed_unit": "ms",
        "forecast_days": max(1, hours // 24 + 1),
        "timezone": "Asia/Shanghai",
    }
    resp = requests.get(OPEN_METEO_FORECAST, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    def safe(key, i):
        col = data.get(key, [])
        return col[i] if i < len(col) else None

    forecast = []
    for i, time in enumerate(data["time"][:hours]):
        forecast.append({
            "time": time,
            "wind_speed_ms": safe("wind_speed_10m", i),
            "wind_direction_deg": safe("wind_direction_10m", i),
            "temperature_c": safe("temperature_2m", i),
            "wind_speed_100m": safe("wind_speed_100m", i),
            "wind_direction_100m": safe("wind_direction_100m", i),
            "cloud_cover_pct":              safe("cloud_cover", i),
            "shortwave_radiation_wm2":      safe("shortwave_radiation", i),
            "direct_normal_irradiance_wm2": safe("direct_normal_irradiance", i),
            "diffuse_radiation_wm2":        safe("diffuse_radiation", i),
        })
    return forecast


def fetch_elevation_batch(lats: List[float], lons: List[float]) -> List[float]:
    """
    Fetch elevations (metres) for up to 100 coordinates from Open-Meteo
    elevation API — free, no key required.

    Important: build the URL manually so commas are NOT percent-encoded.
    Open-Meteo expects literal comma-separated values, not %2C.
    """
    if not lats:
        return []
    lat_str = ",".join(f"{v:.5f}" for v in lats)
    lon_str = ",".join(f"{v:.5f}" for v in lons)
    url = f"{OPEN_METEO_ELEVATION}?latitude={lat_str}&longitude={lon_str}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()["elevation"]
