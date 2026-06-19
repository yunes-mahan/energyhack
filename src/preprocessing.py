import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "24798654" / "SDWPF_dataset" / "sdwpf_kddcup"
FULL_DATA_DIR = Path(__file__).parent.parent / "24798654" / "SDWPF_dataset" / "sdwpf_full"

# Wind farm approximate center (Inner Mongolia, China)
CENTER_LAT = 40.5
CENTER_LON = 108.5
# x,y center of turbine layout (meters)
X_CENTER = 2750.0
Y_CENTER = 6050.0
# meters per degree
M_PER_DEG_LAT = 111000.0
M_PER_DEG_LON = 84400.0  # at ~40°N

RATED_POWER_KW = 1500.0
RATED_WIND_SPEED = 11.0  # m/s
CUT_IN_WIND_SPEED = 3.0  # m/s
CUT_OUT_WIND_SPEED = 25.0  # m/s


def load_locations() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sdwpf_baidukddcup2022_turb_location.csv")
    df["lat"] = CENTER_LAT + (df["y"] - Y_CENTER) / M_PER_DEG_LAT
    df["lon"] = CENTER_LON + (df["x"] - X_CENTER) / M_PER_DEG_LON
    return df


def load_scada(use_full: bool = False) -> pd.DataFrame:
    if use_full:
        df = pd.read_parquet(FULL_DATA_DIR / "sdwpf_2001_2112_full.parquet")
    else:
        df = pd.read_csv(DATA_DIR / "sdwpf_245days_v1.csv")
    return df


def _angle_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    """Signed angular difference a - b, wrapped to [-180, 180]."""
    diff = (a - b + 180) % 360 - 180
    return diff


def clean_scada(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse timestamp
    df["datetime"] = pd.to_datetime(
        df["Day"].astype(str) + " " + df["Tmstamp"].astype(str),
        format="%j %H:%M",
        errors="coerce",
    )
    # Fallback: treat Day as integer offset from a base date
    base = pd.Timestamp("2020-01-01")
    mask = df["datetime"].isna()
    if mask.any():
        df.loc[mask, "datetime"] = (
            base
            + pd.to_timedelta(df.loc[mask, "Day"] - 1, unit="D")
            + pd.to_timedelta(df.loc[mask, "Tmstamp"].str.replace(":", "h") + "m")
        )

    # Drop rows with no timestamp or no power reading
    df = df.dropna(subset=["Tmstamp", "Patv"])

    # Numeric coercion
    num_cols = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physical validity filters
    df = df[df["Wspd"].between(0, 50)]
    df = df[df["Patv"].between(-100, RATED_POWER_KW * 1.05)]
    df = df[df["Etmp"].between(-30, 50)]

    # Negative power at low wind → curtailment/idle; clip to 0
    df.loc[(df["Patv"] < 0) & (df["Wspd"] < CUT_IN_WIND_SPEED), "Patv"] = 0

    df = df.sort_values(["TurbID", "Day", "Tmstamp"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(locations[["TurbID", "x", "y", "lat", "lon"]], on="TurbID", how="left")

    # Yaw misalignment (nacelle vs wind direction)
    df["yaw_misalignment"] = _angle_diff(df["Ndir"], df["Wdir"])
    df["yaw_misalignment_abs"] = df["yaw_misalignment"].abs()

    # Cyclic encoding of directions
    for col in ["Wdir", "Ndir"]:
        rad = np.deg2rad(df[col])
        df[f"{col}_sin"] = np.sin(rad)
        df[f"{col}_cos"] = np.cos(rad)

    # Wind power physics
    df["wspd_cubed"] = df["Wspd"] ** 3
    df["wspd_sq"] = df["Wspd"] ** 2

    # Blade pitch features
    df["pab_mean"] = df[["Pab1", "Pab2", "Pab3"]].mean(axis=1)
    df["pab_std"] = df[["Pab1", "Pab2", "Pab3"]].std(axis=1)
    df["pab_imbalance"] = df[["Pab1", "Pab2", "Pab3"]].max(axis=1) - df[["Pab1", "Pab2", "Pab3"]].min(axis=1)

    # Time features
    df["hour"] = pd.to_datetime(df["Tmstamp"], format="%H:%M", errors="coerce").dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_year"] = df["Day"]
    df["month"] = ((df["Day"] - 1) // 30 + 1).clip(1, 12)

    # Rolling wind stats per turbine (last 6 steps = 1 hour)
    df = df.sort_values(["TurbID", "Day", "Tmstamp"])
    for window in [6, 18]:  # 1h, 3h
        df[f"wspd_roll{window}_mean"] = (
            df.groupby("TurbID")["Wspd"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"wspd_roll{window}_std"] = (
            df.groupby("TurbID")["Wspd"]
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )

    # Empirical power curve per turbine: bin wind speed, compute median power
    # Then efficiency = actual / median_at_that_wind_speed
    df["wspd_bin"] = pd.cut(df["Wspd"], bins=np.arange(0, 26, 1), labels=False)
    median_curve = (
        df[df["Wspd"] > CUT_IN_WIND_SPEED]
        .groupby(["TurbID", "wspd_bin"])["Patv"]
        .transform("median")
    )
    df["p_theory"] = median_curve.fillna(df["Patv"])
    mask = df["Wspd"] > CUT_IN_WIND_SPEED
    df["efficiency"] = np.where(mask, df["Patv"] / (df["p_theory"] + 1e-6), np.nan)
    df["efficiency"] = df["efficiency"].clip(0, 2.0)

    # Normalize spatial coords
    df["x_norm"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
    df["y_norm"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min())

    return df


def add_wake_features(
    df: pd.DataFrame,
    locations: pd.DataFrame,
    max_dist_m: float = 1000.0,
    wake_half_angle: float = 30.0,
) -> pd.DataFrame:
    """
    For each turbine i at each timestep:
      - Find turbines j within max_dist_m whose bearing from i ≈ wind direction (upwind)
      - Add: wake_upstream_count, wake_upstream_avg_wspd, wake_upstream_avg_patv
    Uses vectorised numpy over the turbine dimension; loops only over timesteps.
    """
    df = df.copy()
    df["time_key"] = df["Day"].astype(str) + "_" + df["Tmstamp"].astype(str)

    locs = locations.set_index("TurbID")[["x", "y"]]
    tids = sorted(df["TurbID"].unique())
    n = len(tids)
    tid_to_idx = {t: i for i, t in enumerate(tids)}

    # ── Pairwise geometry (static) ────────────────────────────────────────────
    x = np.array([locs.loc[t, "x"] for t in tids])
    y = np.array([locs.loc[t, "y"] for t in tids])
    dx = x[np.newaxis, :] - x[:, np.newaxis]   # dx[i,j] = x_j − x_i
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dist_matrix = np.sqrt(dx**2 + dy**2)
    bearing_matrix = np.degrees(np.arctan2(dx, dy)) % 360  # bearing i→j (0=N)

    in_range = (dist_matrix <= max_dist_m) & (dist_matrix > 0)  # (n, n)

    # ── Pivot to (timestep × turbine) matrices ────────────────────────────────
    wspd_piv = df.pivot_table(index="time_key", columns="TurbID", values="Wspd", aggfunc="first").reindex(columns=tids)
    patv_piv = df.pivot_table(index="time_key", columns="TurbID", values="Patv", aggfunc="first").reindex(columns=tids)
    wdir_piv = df.pivot_table(index="time_key", columns="TurbID", values="Wdir", aggfunc="first").reindex(columns=tids)

    time_keys = wspd_piv.index.tolist()
    tk_to_idx = {tk: i for i, tk in enumerate(time_keys)}
    n_times = len(time_keys)

    wspd_vals = wspd_piv.values.astype(np.float32)   # (n_times, n)
    patv_vals = patv_piv.values.astype(np.float32)
    wdir_vals = wdir_piv.values.astype(np.float32)

    wake_count = np.zeros((n_times, n), dtype=np.float32)
    wake_wspd  = np.full((n_times, n), np.nan, dtype=np.float32)
    wake_patv  = np.full((n_times, n), np.nan, dtype=np.float32)

    # ── Main loop (vectorised per timestep) ───────────────────────────────────
    print(f"   Computing wake features for {n_times} timesteps...")
    for t in range(n_times):
        # Farm-mean wind direction: good approximation, avoids per-turbine asymmetry
        wdir_t = float(np.nanmean(wdir_vals[t, :]))
        if np.isnan(wdir_t):
            continue

        angle_diff = np.abs(((bearing_matrix - wdir_t + 180) % 360) - 180)  # (n,n)
        is_upwind = in_range & (angle_diff <= wake_half_angle)               # (n,n)

        wspd_t = wspd_vals[t, :]   # (n,)
        patv_t = patv_vals[t, :]

        upwind_wspd = np.where(is_upwind, wspd_t[np.newaxis, :], np.nan)   # (n,n)
        upwind_patv = np.where(is_upwind, patv_t[np.newaxis, :], np.nan)

        wake_count[t, :] = (~np.isnan(upwind_wspd)).sum(axis=1)
        with np.errstate(all="ignore"):
            wake_wspd[t, :] = np.nanmean(upwind_wspd, axis=1)
            wake_patv[t, :] = np.nanmean(upwind_patv, axis=1)

        if t % 8000 == 0 and t > 0:
            print(f"   ... {t}/{n_times}")

    # ── Map results back to original rows ─────────────────────────────────────
    df["wake_upstream_count"]    = 0.0
    df["wake_upstream_avg_wspd"] = df["Wspd"]
    df["wake_upstream_avg_patv"] = 0.0

    t_idxs    = df["time_key"].map(tk_to_idx)
    turb_idxs = df["TurbID"].map(tid_to_idx)
    valid     = t_idxs.notna() & turb_idxs.notna()

    ti = t_idxs[valid].astype(int).values
    bi = turb_idxs[valid].astype(int).values

    df.loc[valid, "wake_upstream_count"]    = wake_count[ti, bi]
    df.loc[valid, "wake_upstream_avg_wspd"] = np.where(np.isnan(wake_wspd[ti, bi]), df.loc[valid, "Wspd"], wake_wspd[ti, bi])
    df.loc[valid, "wake_upstream_avg_patv"] = np.where(np.isnan(wake_patv[ti, bi]), 0.0, wake_patv[ti, bi])

    df = df.drop(columns=["time_key"])
    print("   Wake features done.")
    return df


def compute_turbine_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-turbine aggregated statistics for clustering and performance scoring."""
    valid = df[df["Wspd"] > CUT_IN_WIND_SPEED].copy()

    stats = valid.groupby("TurbID").agg(
        mean_efficiency=("efficiency", "mean"),
        std_efficiency=("efficiency", "std"),
        mean_patv=("Patv", "mean"),
        mean_yaw_misalignment=("yaw_misalignment_abs", "mean"),
        mean_wspd=("Wspd", "mean"),
        availability=("Patv", lambda x: (x > 0).mean()),
        x=("x", "first"),
        y=("y", "first"),
        x_norm=("x_norm", "first"),
        y_norm=("y_norm", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    ).reset_index()

    stats["performance_score"] = (stats["mean_efficiency"].clip(0, 1) * 100).round(1)
    return stats
