"""
Terrain-aware wind turbine siting model.

Scoring factors (all combined as a weighted product):
  1. Wind resource   — 100 m hub-height speed via Hellmann power-law correction
  2. Elevation       — suitable altitude (excludes water/sea, extreme peaks)
  3. Terrain slope   — flat plains preferred; steep ridges penalised
  4. Wake avoidance  — downwind shadow cone from existing turbines
  5. Spacing         — hard minimum-distance constraint (500 m)
"""
import numpy as np
from typing import List, Dict, Any, Optional

RATED_KW       = 1500.0
CUT_IN_MS      = 3.0
RATED_MS       = 11.0
CUT_OUT_MS     = 25.0
MIN_SPACING_M  = 500.0
WAKE_RADIUS_M  = 1200.0
WAKE_ANGLE_DEG = 30.0

WIND_ALPHA     = 0.143   # Hellmann exponent (open flat terrain)
REF_HEIGHT_M   = 10.0    # Open-Meteo reference height
HUB_HEIGHT_M   = 100.0   # Typical modern turbine hub height


# ── Geometry ──────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2 +
         np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(np.radians(lat2))
    y = (np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) -
         np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(dlon))
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ── Power curve ───────────────────────────────────────────────────────────────

def power_kw(wspd: float) -> float:
    if wspd < CUT_IN_MS or wspd >= CUT_OUT_MS:
        return 0.0
    if wspd >= RATED_MS:
        return RATED_KW
    t = (wspd - CUT_IN_MS) / (RATED_MS - CUT_IN_MS)
    return RATED_KW * t ** 3


# ── Terrain scoring ───────────────────────────────────────────────────────────

def elevation_score(elevation_m: float) -> float:
    """
    Suitability by altitude:
      <0 m   → water/below sea level → disqualified
      >4000 m → extreme (thin air, icing) → nearly 0
      optimal ~100–800 m (Inner Mongolia plateau)
    """
    if elevation_m < 0:
        return 0.0
    if elevation_m > 4000:
        return max(0.0, 1.0 - (elevation_m - 4000) / 500)
    return float(np.exp(-0.5 * ((elevation_m - 400) / 600) ** 2))


def slope_score(elev_grid: np.ndarray, i: int, j: int, spacing_m: float) -> float:
    """
    Penalise steep terrain.  slope = Δelevation / horizontal_distance.
    Characteristic decay: 12% slope → score 0.43; 25% → 0.12.
    """
    rows, cols = elev_grid.shape
    slopes = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            dh = abs(float(elev_grid[ni, nj]) - float(elev_grid[i, j]))
            slopes.append(dh / max(spacing_m, 1.0) * 100)  # percent
    if not slopes:
        return 1.0
    return float(np.exp(-max(slopes) / 12.0))


def wind_at_hub(wspd_10m: float, elevation_m: float) -> float:
    """
    Estimate wind speed at hub height (100 m) using:
      1. Hellmann power law:  v(100m) / v(10m) = (100/10)^α
      2. Elevation bonus:     +1.5 % per 100 m above sea level
         (higher sites have thinner boundary layer → more wind)
    """
    hub_factor  = (HUB_HEIGHT_M / REF_HEIGHT_M) ** WIND_ALPHA   # ≈ 1.39
    elev_bonus  = 1.0 + max(0.0, elevation_m) * 1.5e-4          # +0.15%/100m
    return wspd_10m * hub_factor * elev_bonus


# ── Wake exposure ─────────────────────────────────────────────────────────────

def wake_exposure_fraction(
    cand_lat: float, cand_lon: float,
    existing: List[Dict],
    forecast: List[Dict],
) -> float:
    if not existing or not forecast:
        return 0.0
    in_wake = 0
    for step in forecast:
        wind_dir = step.get("wind_direction_deg") or 0.0
        downwind = (wind_dir + 180) % 360
        for t in existing:
            dist = haversine(t["lat"], t["lon"], cand_lat, cand_lon)
            if dist > WAKE_RADIUS_M or dist < 1.0:
                continue
            brng = bearing(t["lat"], t["lon"], cand_lat, cand_lon)
            if abs(((brng - downwind + 180) % 360) - 180) <= WAKE_ANGLE_DEG:
                in_wake += 1
                break
    return in_wake / len(forecast)


# ── Grid scoring ──────────────────────────────────────────────────────────────

def _score_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    elevations: np.ndarray,   # shape [n_lat, n_lon]
    existing: List[Dict],
    forecast: List[Dict],
    grid_spacing_m: float,
) -> np.ndarray:
    n_lat, n_lon = len(lats), len(lons)
    base_wspd = float(np.mean([f.get("wind_speed_ms") or 0 for f in forecast]))
    scores = np.zeros((n_lat, n_lon))

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            elev = float(elevations[i, j])
            es = elevation_score(elev)
            if es < 0.01:
                continue
            ss   = slope_score(elevations, i, j, grid_spacing_m)
            ws   = wind_at_hub(base_wspd, elev)
            wind_s = power_kw(ws) / RATED_KW
            wake_s = 1.0 - wake_exposure_fraction(lat, lon, existing, forecast)
            # Weighted product: wind matters most, elevation/slope moderate influence
            scores[i, j] = wind_s * (es ** 0.5) * (ss ** 0.4) * (wake_s ** 0.8)

    return scores


# ── Wind resource heatmap (legacy — unchanged API) ────────────────────────────

def wind_resource_grid(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    forecast: List[Dict],
    grid_n: int = 20,
) -> List[Dict]:
    lats = np.linspace(lat_min, lat_max, grid_n)
    lons = np.linspace(lon_min, lon_max, grid_n)
    avg_power = float(np.mean([power_kw(f.get("wind_speed_ms") or 0) for f in forecast]))
    cap_factor = avg_power / RATED_KW
    return [
        {"lat": round(float(lat), 6), "lon": round(float(lon), 6),
         "avg_power_kw": round(avg_power, 1), "capacity_factor": round(cap_factor, 3)}
        for lat in lats for lon in lons
    ]


# ── Main siting function ──────────────────────────────────────────────────────

def find_best_locations(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    n_turbines: int,
    existing_turbines: List[Dict],
    forecast: List[Dict],
    grid_n: int = 15,
    elevations: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Greedy terrain-aware turbine siting.
    Each step selects the highest-scoring candidate that satisfies
    the spacing constraint; spacing mask expands as sites are chosen.

    elevations: pre-fetched [grid_n × grid_n] array from Open-Meteo elevation API.
                Falls back to Inner Mongolia plateau default (1100 m) if None.
    """
    lats = np.linspace(lat_min, lat_max, grid_n)
    lons = np.linspace(lon_min, lon_max, grid_n)

    lat_span_m = haversine(lat_min, lon_min, lat_max, lon_min)
    lon_span_m = haversine(lat_min, lon_min, lat_min, lon_max)
    spacing_m  = (lat_span_m + lon_span_m) / 2 / max(1, grid_n - 1)

    if elevations is None:
        elevations = np.full((grid_n, grid_n), 1100.0)  # Inner Mongolia plateau default

    scores = _score_grid(lats, lons, elevations, existing_turbines, forecast, spacing_m)

    selected: List[Dict] = []
    placed_mask = np.zeros((grid_n, grid_n), dtype=bool)

    for _ in range(n_turbines):
        masked = np.where(placed_mask, 0.0, scores)
        if masked.max() <= 0:
            break

        best_i, best_j = np.unravel_index(masked.argmax(), masked.shape)
        best_lat  = float(lats[best_i])
        best_lon  = float(lons[best_j])
        best_elev = float(elevations[best_i, best_j])
        best_score = float(masked[best_i, best_j])

        # Mask MIN_SPACING_M radius around chosen point
        for i in range(grid_n):
            for j in range(grid_n):
                if haversine(float(lats[i]), float(lons[j]), best_lat, best_lon) < MIN_SPACING_M:
                    placed_mask[i, j] = True

        base_wspd  = float(np.mean([f.get("wind_speed_ms") or 0 for f in forecast]))
        corr_wspd  = wind_at_hub(base_wspd, best_elev)
        wake_frac  = wake_exposure_fraction(best_lat, best_lon, existing_turbines, forecast)
        avg_kw     = power_kw(corr_wspd) * (1 - wake_frac)
        aep_mwh    = avg_kw * 8_760 / 1_000

        selected.append({
            "rank":                       len(selected) + 1,
            "lat":                        round(best_lat, 6),
            "lon":                        round(best_lon, 6),
            "siting_score":               round(best_score, 3),
            "elevation_m":                round(best_elev, 1),
            "wind_speed_100m_ms":         round(corr_wspd, 2),
            "wake_exposure_pct":          round(wake_frac * 100, 1),
            "estimated_avg_power_kw":     round(avg_kw, 1),
            "estimated_aep_mwh_per_year": round(aep_mwh, 0),
            "spacing_ok":                 True,
        })

    return selected
