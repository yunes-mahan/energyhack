"""
Maintenance risk scoring (using Chinese SDWPF performance metrics)
and low-wind maintenance window finder for the next 16 days.
"""
import numpy as np
import requests
from typing import List, Dict

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


def score_maintenance_risk(
    turbine_id: int,
    performance_score: float,
    availability: float,
    yaw_misalignment: float,
    mean_efficiency: float = 0.85,
    age_years: int = 0,
) -> Dict:
    """
    Score maintenance urgency from SDWPF-derived metrics.
    Calibrated against the Chinese fleet's failure patterns.
    """
    issues = []
    score = 0.0

    # Performance degradation — directly from Chinese KPI
    if performance_score < 40:
        issues.append("Critical power curve degradation — blade pitch or gearbox fault")
        score += 0.40
    elif performance_score < 60:
        issues.append("Below-spec performance — inspect blade pitch control system")
        score += 0.20

    # Availability (% time generating)
    if availability < 0.80:
        issues.append(f"Low availability {availability*100:.0f}% — frequent unplanned stops")
        score += 0.30
    elif availability < 0.92:
        issues.append(f"Moderate downtime ({availability*100:.0f}%) — review fault history")
        score += 0.15

    # Yaw tracking error
    if abs(yaw_misalignment) > 15:
        issues.append(f"Severe yaw error {yaw_misalignment:.1f}° — nacelle alignment service needed")
        score += 0.20
    elif abs(yaw_misalignment) > 8:
        issues.append(f"Yaw deviation {yaw_misalignment:.1f}° — recalibrate yaw sensors")
        score += 0.10

    # Power curve efficiency
    if mean_efficiency < 0.65:
        issues.append("Power curve below spec — possible icing, soiling, or pitch fault")
        score += 0.15

    # Age-based maintenance schedule
    if age_years >= 20:
        issues.append("Life extension assessment required (20+ years)")
        score += 0.30
    elif age_years >= 15:
        issues.append("Gearbox oil analysis + main bearing inspection due (15+ years)")
        score += 0.15
    elif age_years >= 10:
        issues.append("Blade inspection + bolt torque check recommended (10+ years)")
        score += 0.05

    score = min(1.0, score)

    if score >= 0.60:
        level, urgency, months = "CRITICAL", "Immediate — within 2 weeks", 0.5
    elif score >= 0.35:
        level, urgency, months = "HIGH", "Schedule within 1 month", 1
    elif score >= 0.15:
        level, urgency, months = "MEDIUM", "Schedule within 3 months", 3
    else:
        level, urgency, months = "LOW", "Routine annual inspection", 12

    return {
        "turbine_id": turbine_id,
        "risk_level": level,
        "risk_score": round(score, 2),
        "urgency": urgency,
        "next_service_months": months,
        "issues": issues or ["No significant issues — performing normally"],
    }


def find_maintenance_windows(lat: float, lon: float, days: int = 16) -> List[Dict]:
    """
    Find windows where wind speed < 5 m/s for 8+ consecutive hours.
    These are the cheapest times to take a turbine offline (minimum lost generation).
    Uses Open-Meteo 16-day forecast — free, no key.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_100m,wind_speed_10m",
        "wind_speed_unit": "ms",
        "forecast_days": min(days, 16),
        "timezone": "auto",
    }
    resp = requests.get(OPEN_METEO_FORECAST, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    times = data["time"]
    speeds = data.get("wind_speed_100m") or data.get("wind_speed_10m") or []

    LOW_WIND_MS = 5.0
    MIN_WINDOW_H = 8

    windows: List[Dict] = []
    in_window = False
    start_i = 0

    for i, s in enumerate(speeds):
        low = s is not None and s < LOW_WIND_MS
        if low and not in_window:
            in_window = True
            start_i = i
        elif not low and in_window:
            _save_window(windows, times, speeds, start_i, i, MIN_WINDOW_H, LOW_WIND_MS)
            in_window = False

    # Window reaching end of forecast
    if in_window:
        _save_window(windows, times, speeds, start_i, len(times), MIN_WINDOW_H, LOW_WIND_MS)

    windows.sort(key=lambda w: w["lost_gen_score"])
    return windows[:5]


def _save_window(windows, times, speeds, start_i, end_i, min_h, threshold):
    dur = end_i - start_i
    if dur < min_h:
        return
    window_speeds = [s for s in speeds[start_i:end_i] if s is not None]
    if not window_speeds:
        return
    avg_s = float(np.mean(window_speeds))
    windows.append({
        "start": times[start_i],
        "end": times[end_i - 1],
        "duration_hours": dur,
        "avg_wind_ms": round(avg_s, 1),
        "rating": "Optimal" if avg_s < 3 else "Good" if avg_s < 4 else "Acceptable",
        "lost_gen_score": round(avg_s / threshold, 2),
    })
