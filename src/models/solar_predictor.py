"""
LightGBM solar power predictor trained on Chinese State Grid 50 MW plant data.
Input:  GHI, DNI, TSI, temperature, humidity, pressure + time features
Output: capacity factor (0–1)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

PLANT_CAPACITY_MW = 50.0

FEATURE_COLS = [
    "ghi", "dni", "tsi",
    "temperature", "humidity", "pressure",
    "ghi_sq", "ghi_dni_ratio",
    "hour_sin", "hour_cos",
    "month", "day_of_year",
]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": -1,
    "n_jobs": -1,
}


class SolarPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = FEATURE_COLS
        self.train_mae = None
        self.val_mae = None
        self.plant_capacity_mw = PLANT_CAPACITY_MW

    @staticmethod
    def _rename(df: pd.DataFrame) -> pd.DataFrame:
        # Rename by position to avoid encoding issues with degree symbol
        cols = list(df.columns)
        mapping = {
            cols[0]: "time",
            cols[1]: "tsi",
            cols[2]: "dni",
            cols[3]: "ghi",
            cols[4]: "temperature",
            cols[5]: "pressure",
            cols[6]: "humidity",
            cols[7]: "power_mw",
        }
        return df.rename(columns=mapping)

    @staticmethod
    def engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ghi"]  = df["ghi"].clip(lower=0)
        df["dni"]  = df["dni"].clip(lower=0)
        df["tsi"]  = df["tsi"].clip(lower=0)

        df["time"] = pd.to_datetime(df["time"])
        hour_frac  = df["time"].dt.hour + df["time"].dt.minute / 60
        df["hour_sin"]    = np.sin(2 * np.pi * hour_frac / 24)
        df["hour_cos"]    = np.cos(2 * np.pi * hour_frac / 24)
        df["month"]       = df["time"].dt.month
        df["day_of_year"] = df["time"].dt.dayofyear

        df["ghi_sq"]        = df["ghi"] ** 2
        df["ghi_dni_ratio"] = df["ghi"] / (df["dni"] + 1.0)

        df["capacity_factor"] = (df["power_mw"] / PLANT_CAPACITY_MW).clip(0, 1)
        return df

    def prepare(self, df: pd.DataFrame):
        df = self._rename(df)
        df = self.engineer(df)
        df = df[df["ghi"] > 0].dropna(subset=self.feature_cols + ["capacity_factor"])
        return df[self.feature_cols], df["capacity_factor"]

    def fit(self, df: pd.DataFrame) -> "SolarPredictor":
        X, y = self.prepare(df)
        split = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]

        tr_ds  = lgb.Dataset(X_tr,  label=y_tr)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=tr_ds)
        self.model = lgb.train(
            LGBM_PARAMS, tr_ds, num_boost_round=800,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )
        self.train_mae = mean_absolute_error(y_tr,  self.model.predict(X_tr))
        self.val_mae   = mean_absolute_error(y_val, self.model.predict(X_val))
        print(f"Solar Train MAE: {self.train_mae:.4f} | Val MAE: {self.val_mae:.4f}  (capacity factor)")
        return self

    def predict_cf(
        self,
        ghi: float,
        temperature: float,
        dni: float | None = None,
        humidity: float = 50.0,
        pressure: float = 930.0,   # Inner Mongolia plateau ~930 hPa
        hour: float = 12.0,
        month: int = 6,
        day_of_year: int = 170,
    ) -> float:
        if ghi <= 0:
            return 0.0
        if dni is None:
            dni = ghi * 0.7
        row = {
            "ghi":           ghi,
            "dni":           max(0, dni),
            "tsi":           ghi + max(0, dni) * 0.3,
            "temperature":   temperature,
            "humidity":      humidity,
            "pressure":      pressure,
            "ghi_sq":        ghi ** 2,
            "ghi_dni_ratio": ghi / (max(0, dni) + 1.0),
            "hour_sin":      np.sin(2 * np.pi * hour / 24),
            "hour_cos":      np.cos(2 * np.pi * hour / 24),
            "month":         month,
            "day_of_year":   day_of_year,
        }
        X = pd.DataFrame([row])[self.feature_cols]
        return float(np.clip(self.model.predict(X)[0], 0, 1))

    def save(self):
        self.model.save_model(str(MODELS_DIR / "solar_predictor.txt"))
        joblib.dump({
            "train_mae": self.train_mae,
            "val_mae":   self.val_mae,
            "feature_cols": self.feature_cols,
            "plant_capacity_mw": self.plant_capacity_mw,
        }, MODELS_DIR / "solar_predictor_meta.pkl")

    @classmethod
    def load(cls) -> "SolarPredictor":
        obj = cls()
        obj.model = lgb.Booster(model_file=str(MODELS_DIR / "solar_predictor.txt"))
        meta = joblib.load(MODELS_DIR / "solar_predictor_meta.pkl")
        obj.train_mae   = meta["train_mae"]
        obj.val_mae     = meta["val_mae"]
        obj.feature_cols = meta["feature_cols"]
        obj.plant_capacity_mw = meta.get("plant_capacity_mw", PLANT_CAPACITY_MW)
        return obj
