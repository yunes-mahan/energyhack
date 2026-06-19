import numpy as np
import pandas as pd
from typing import List, Dict

RATED_POWER_KW = 1500.0
RATED_WIND_SPEED = 11.0
CUT_IN_WIND_SPEED = 3.0

# Pitch angle lookup: wind speed → optimal pitch angle (degrees)
# Below rated: fine pitch (low angle, max energy capture)
# Above rated: increase pitch to limit power
PITCH_CURVE = {
    3: 0.0, 4: 0.0, 5: 0.0, 6: 0.5, 7: 1.0, 8: 1.0,
    9: 1.0, 10: 1.5, 11: 5.0, 12: 9.0, 13: 12.5,
    14: 15.5, 15: 18.0, 16: 20.0, 17: 22.0, 18: 23.5,
    19: 25.0, 20: 26.0, 25: 30.0,
}


def _optimal_pitch(wind_speed: float) -> float:
    speeds = sorted(PITCH_CURVE.keys())
    if wind_speed <= speeds[0]:
        return PITCH_CURVE[speeds[0]]
    if wind_speed >= speeds[-1]:
        return PITCH_CURVE[speeds[-1]]
    for i in range(len(speeds) - 1):
        if speeds[i] <= wind_speed <= speeds[i + 1]:
            t = (wind_speed - speeds[i]) / (speeds[i + 1] - speeds[i])
            return PITCH_CURVE[speeds[i]] * (1 - t) + PITCH_CURVE[speeds[i + 1]] * t
    return 1.0


def _yaw_power_factor(misalignment_deg: float) -> float:
    """Power loss factor due to yaw misalignment (cos^2 model)."""
    return np.cos(np.deg2rad(misalignment_deg)) ** 2


def optimize_turbine_settings(
    turbines: pd.DataFrame,
    wind_speed: float,
    wind_direction: float,
    current_ndir: Dict[int, float] = None,
    current_pab: Dict[int, float] = None,
) -> pd.DataFrame:
    """
    For each turbine, compute:
    - Recommended yaw (Ndir) = wind direction
    - Recommended pitch angles (Pab1/2/3) from pitch curve
    - Expected power gain vs current settings
    """
    results = []
    optimal_pitch = _optimal_pitch(wind_speed)

    for _, row in turbines.iterrows():
        tid = int(row["TurbID"])

        curr_ndir = (current_ndir or {}).get(tid, wind_direction)
        curr_pab = (current_pab or {}).get(tid, 1.0)

        curr_yaw_error = abs(((curr_ndir - wind_direction + 180) % 360) - 180)
        curr_power_factor = _yaw_power_factor(curr_yaw_error)

        # Optimal: yaw aligned to wind, pitch at curve value
        opt_power_factor = 1.0  # zero misalignment

        # Estimated base power from wind speed (simplified power curve)
        if wind_speed < CUT_IN_WIND_SPEED:
            base_power = 0.0
        elif wind_speed >= RATED_WIND_SPEED:
            base_power = RATED_POWER_KW
        else:
            # Cubic interpolation between cut-in and rated
            t = (wind_speed - CUT_IN_WIND_SPEED) / (RATED_WIND_SPEED - CUT_IN_WIND_SPEED)
            base_power = RATED_POWER_KW * t ** 3

        current_est_power = base_power * curr_power_factor
        optimal_est_power = base_power * opt_power_factor
        power_gain_kw = max(0.0, optimal_est_power - current_est_power)
        power_gain_pct = (power_gain_kw / (current_est_power + 1e-6)) * 100

        results.append({
            "TurbID": tid,
            "recommended_ndir": round(wind_direction % 360, 1),
            "recommended_pab1": round(optimal_pitch, 2),
            "recommended_pab2": round(optimal_pitch, 2),
            "recommended_pab3": round(optimal_pitch, 2),
            "current_yaw_error_deg": round(curr_yaw_error, 1),
            "yaw_action_needed": curr_yaw_error > 5.0,
            "pitch_action_needed": abs(curr_pab - optimal_pitch) > 1.0,
            "estimated_power_gain_kw": round(power_gain_kw, 1),
            "estimated_power_gain_pct": round(power_gain_pct, 1),
            "optimal_est_power_kw": round(optimal_est_power, 1),
        })

    return pd.DataFrame(results)


def flag_anomalies(df: pd.DataFrame, predictor) -> pd.DataFrame:
    """Flag turbines where actual power deviates significantly from predicted."""
    if predictor is None:
        return df
    df = df.copy()
    needed_cols = predictor.feature_cols
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        return df
    preds = predictor.predict(df)
    df["predicted_patv"] = preds
    df["residual"] = df["Patv"] - df["predicted_patv"]
    df["anomaly"] = df["residual"].abs() > 200  # >200 kW deviation
    return df
