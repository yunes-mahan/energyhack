import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "Wspd", "wspd_sq", "wspd_cubed",
    "Wdir_sin", "Wdir_cos",
    "Etmp", "Itmp",
    "Ndir_sin", "Ndir_cos",
    "yaw_misalignment", "yaw_misalignment_abs",
    "pab_mean", "pab_std", "pab_imbalance",
    "hour_sin", "hour_cos",
    "month", "day_of_year",
    "wspd_roll6_mean", "wspd_roll6_std",
    "wspd_roll18_mean", "wspd_roll18_std",
    "x_norm", "y_norm",
    "TurbID",
    # Wake effect features
    "wake_upstream_count",
    "wake_upstream_avg_wspd",
    "wake_upstream_avg_patv",
]

TARGET_COL = "Patv"

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": -1,
    "n_jobs": -1,
}


class PowerPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = FEATURE_COLS
        self.train_mae = None
        self.val_mae = None

    def prepare(self, df: pd.DataFrame):
        df = df.copy()
        # Only train on rows where wind is meaningful and power is valid
        df = df[df["Wspd"] > 0].dropna(subset=self.feature_cols + [TARGET_COL])
        X = df[self.feature_cols]
        y = df[TARGET_COL]
        return X, y

    def fit(self, df: pd.DataFrame) -> "PowerPredictor":
        X, y = self.prepare(df)

        # Temporal split: last 20% as validation
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
        self.model = lgb.train(
            LGBM_PARAMS,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        self.train_mae = mean_absolute_error(y_train, self.model.predict(X_train))
        self.val_mae = mean_absolute_error(y_val, self.model.predict(X_val))
        print(f"Train MAE: {self.train_mae:.2f} kW | Val MAE: {self.val_mae:.2f} kW")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in X.columns]
        for c in missing:
            X[c] = 0
        return self.model.predict(X[self.feature_cols]).clip(0, 1500)

    def predict_from_weather(self, turbines_df: pd.DataFrame, wind_speed: float,
                              wind_dir: float, temperature: float) -> pd.DataFrame:
        """Predict power for all turbines given a single weather snapshot."""
        df = turbines_df.copy()
        df["Wspd"] = wind_speed
        df["Wdir"] = wind_dir
        df["Etmp"] = temperature
        df["Itmp"] = temperature + 10
        # Assume nacelle follows wind (ideal case)
        df["Ndir"] = wind_dir
        df["yaw_misalignment"] = 0.0
        df["yaw_misalignment_abs"] = 0.0

        rad = np.deg2rad(wind_dir)
        df["Wdir_sin"] = np.sin(rad)
        df["Wdir_cos"] = np.cos(rad)
        df["Ndir_sin"] = np.sin(rad)
        df["Ndir_cos"] = np.cos(rad)

        df["wspd_sq"] = wind_speed ** 2
        df["wspd_cubed"] = wind_speed ** 3
        df["pab_mean"] = 1.0
        df["pab_std"] = 0.0
        df["pab_imbalance"] = 0.0
        df["hour_sin"] = 0.0
        df["hour_cos"] = 1.0
        df["month"] = 6
        df["day_of_year"] = 150
        df["wspd_roll6_mean"] = wind_speed
        df["wspd_roll6_std"] = 0.0
        df["wspd_roll18_mean"] = wind_speed
        df["wspd_roll18_std"] = 0.0

        df["predicted_kw"] = self.predict(df)
        return df[["TurbID", "lat", "lon", "predicted_kw"]]

    def feature_importance(self) -> pd.DataFrame:
        imp = pd.DataFrame({
            "feature": self.model.feature_name(),
            "importance": self.model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        return imp

    def save(self):
        self.model.save_model(str(MODELS_DIR / "power_predictor.txt"))
        joblib.dump({"train_mae": self.train_mae, "val_mae": self.val_mae,
                     "feature_cols": self.feature_cols}, MODELS_DIR / "predictor_meta.pkl")

    @classmethod
    def load(cls) -> "PowerPredictor":
        obj = cls()
        obj.model = lgb.Booster(model_file=str(MODELS_DIR / "power_predictor.txt"))
        meta = joblib.load(MODELS_DIR / "predictor_meta.pkl")
        obj.train_mae = meta["train_mae"]
        obj.val_mae = meta["val_mae"]
        obj.feature_cols = meta["feature_cols"]
        return obj
