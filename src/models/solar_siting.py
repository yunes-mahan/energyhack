"""
Solar farm siting model.

Scoring factors (weighted product):
  1. Solar resource  — GHI from Open-Meteo forecast (dominant driver)
  2. Elevation       — same suitability as wind (100-800m optimal, Inner Mongolia)
  3. Terrain slope   — flat/gentle preferred; steep penalised
  4. Aspect bonus    — south-facing slope gives up to +10% score (NH hemisphere)
  5. Spacing         — hard 300 m minimum between sites
"""
import numpy as np
from typing import List, Dict, Any, Optional
from src.models.siting import haversine, elevation_score, slope_score

PANEL_EFFICIENCY   = 0.20    # 20% for modern mono-Si panels
SYSTEM_LOSSES      = 0.85    # inverter + wiring + soiling
TEMP_COEFF         = -0.004  # -0.4 %/°C above STC 25 °C
MIN_SPACING_M      = 300.0   # solar parks can be denser than wind farms
DEFAULT_SITE_MW    = 10.0    # MW per site default


def solar_power_kw(ghi_wm2: float, temp_c: float, capacity_kw: float) -> float:
    """Simplified PVWatts-style DC→AC power estimate."""
    if ghi_wm2 <= 0 or capacity_kw <= 0:
        return 0.0
    temp_factor = 1.0 + TEMP_COEFF * (temp_c - 25.0)
    dc_kw = (ghi_wm2 / 1000.0) * capacity_kw * PANEL_EFFICIENCY * temp_factor
    return max(0.0, dc_kw * SYSTEM_LOSSES)


def aspect_score(elev_grid: np.ndarray, i: int, j: int) -> float:
    """South-facing slopes get a bonus (north neighbour higher → south slope)."""
    north_i = i - 1
    if north_i < 0 or north_i >= elev_grid.shape[0]:
        return 1.0
    dh = float(elev_grid[north_i, j]) - float(elev_grid[i, j])
    return 1.0 + float(np.clip(dh / 20.0, -0.10, 0.10))


def _score_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    elevations: np.ndarray,
    forecast: List[Dict],
    grid_spacing_m: float,
) -> np.ndarray:
    n_lat, n_lon = len(lats), len(lons)

    avg_ghi  = float(np.mean([f.get("shortwave_radiation_wm2") or 0 for f in forecast]))
    avg_temp = float(np.mean([f.get("temperature_c") or 15 for f in forecast]))
    # Normalise: at 1000 W/m² (peak sun) a 1 kW system → ~0.17 kW → solar_s ≈ 0.85
    ref_power = solar_power_kw(1000.0, 25.0, 1.0)

    scores = np.zeros((n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            elev = float(elevations[i, j])
            es   = elevation_score(elev)
            if es < 0.01:
                continue
            ss      = slope_score(elevations, i, j, grid_spacing_m)
            asp     = aspect_score(elevations, i, j)
            pwr     = solar_power_kw(avg_ghi, avg_temp, 1.0)
            solar_s = min(1.0, pwr / max(ref_power, 1e-6))
            scores[i, j] = solar_s * (es ** 0.4) * (ss ** 0.3) * asp

    return scores


def find_best_solar_locations(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    n_sites: int,
    forecast: List[Dict],
    grid_n: int = 10,
    elevations: Optional[np.ndarray] = None,
    site_capacity_mw: float = DEFAULT_SITE_MW,
) -> List[Dict]:
    """Greedy solar siting with 300 m minimum spacing."""
    lats = np.linspace(lat_min, lat_max, grid_n)
    lons = np.linspace(lon_min, lon_max, grid_n)

    lat_span_m = haversine(lat_min, lon_min, lat_max, lon_min)
    lon_span_m = haversine(lat_min, lon_min, lat_min, lon_max)
    spacing_m  = (lat_span_m + lon_span_m) / 2 / max(1, grid_n - 1)

    if elevations is None:
        elevations = np.full((grid_n, grid_n), 1100.0)

    scores = _score_grid(lats, lons, elevations, forecast, spacing_m)

    avg_ghi  = float(np.mean([f.get("shortwave_radiation_wm2") or 0 for f in forecast]))
    avg_temp = float(np.mean([f.get("temperature_c") or 15 for f in forecast]))

    selected: List[Dict] = []
    placed   = np.zeros((grid_n, grid_n), dtype=bool)

    for _ in range(n_sites):
        masked = np.where(placed, 0.0, scores)
        if masked.max() <= 0:
            break

        bi, bj     = np.unravel_index(masked.argmax(), masked.shape)
        best_lat   = float(lats[bi])
        best_lon   = float(lons[bj])
        best_elev  = float(elevations[bi, bj])
        best_score = float(masked[bi, bj])

        for i in range(grid_n):
            for j in range(grid_n):
                if haversine(float(lats[i]), float(lons[j]), best_lat, best_lon) < MIN_SPACING_M:
                    placed[i, j] = True

        capacity_kw = site_capacity_mw * 1000.0
        avg_power_kw = solar_power_kw(avg_ghi, avg_temp, capacity_kw)
        cf  = avg_power_kw / capacity_kw if capacity_kw > 0 else 0.0
        # AEP: solar irradiance is present ~8 h/day on average in Inner Mongolia
        aep = avg_power_kw * 8760 / 1000.0  # using raw average (Open-Meteo already averages over 24h)

        selected.append({
            "rank":                        len(selected) + 1,
            "lat":                         round(best_lat, 6),
            "lon":                         round(best_lon, 6),
            "siting_score":                round(best_score, 3),
            "elevation_m":                 round(best_elev, 1),
            "avg_ghi_wm2":                 round(avg_ghi, 1),
            "site_capacity_mw":            site_capacity_mw,
            "estimated_avg_power_kw":      round(avg_power_kw, 1),
            "estimated_capacity_factor":   round(cf, 3),
            "estimated_aep_mwh_per_year":  round(aep, 0),
        })

    return selected
