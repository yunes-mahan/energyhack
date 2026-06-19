import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CLUSTER_FEATURES = [
    "mean_efficiency",
    "std_efficiency",
    "mean_yaw_misalignment",
    "availability",
    "x_norm",
    "y_norm",
]

CLUSTER_LABELS = {
    -1: "Anomalous",
    0: "High Performers",
    1: "Average",
    2: "Wake-Affected",
    3: "Underperformers",
}


class TurbineClustering:
    def __init__(self, method: str = "dbscan"):
        self.method = method
        self.scaler = StandardScaler()
        self.model = None
        self.labels_ = None
        self.n_clusters_ = 0

    def fit(self, stats: pd.DataFrame) -> "TurbineClustering":
        X = stats[CLUSTER_FEATURES].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        if self.method == "dbscan":
            # Tune eps so most turbines get a cluster (not noise)
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=4).fit(X_scaled)
            distances, _ = nbrs.kneighbors(X_scaled)
            eps = float(np.percentile(distances[:, -1], 90))
            self.model = DBSCAN(eps=eps, min_samples=3)
            raw_labels = self.model.fit_predict(X_scaled)
            # Fall back to KMeans if too many noise points
            noise_frac = (raw_labels == -1).mean()
            if noise_frac > 0.3:
                n = self._best_k(X_scaled)
                self.model = KMeans(n_clusters=n, random_state=42, n_init=10)
                raw_labels = self.model.fit_predict(X_scaled)
        else:
            n = self._best_k(X_scaled)
            self.model = KMeans(n_clusters=n, random_state=42, n_init=10)
            raw_labels = self.model.fit_predict(X_scaled)

        # Re-label clusters by mean efficiency (0=best, N=worst)
        self.labels_ = self._relabel_by_efficiency(raw_labels, stats["mean_efficiency"].values)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        return self

    def _best_k(self, X: np.ndarray, k_range=range(3, 8)) -> int:
        best_k, best_score = 4, -1
        for k in k_range:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
        return best_k

    def _relabel_by_efficiency(self, labels: np.ndarray, efficiency: np.ndarray) -> np.ndarray:
        unique = [l for l in np.unique(labels) if l != -1]
        cluster_means = {l: efficiency[labels == l].mean() for l in unique}
        sorted_clusters = sorted(unique, key=lambda l: -cluster_means[l])
        mapping = {old: new for new, old in enumerate(sorted_clusters)}
        mapping[-1] = -1
        return np.array([mapping[l] for l in labels])

    def get_cluster_summary(self, stats: pd.DataFrame) -> pd.DataFrame:
        stats = stats.copy()
        stats["cluster_id"] = self.labels_
        summary = stats.groupby("cluster_id").agg(
            n_turbines=("TurbID", "count"),
            mean_efficiency=("mean_efficiency", "mean"),
            mean_yaw_misalignment=("mean_yaw_misalignment", "mean"),
            availability=("availability", "mean"),
        ).reset_index()
        summary["label"] = summary["cluster_id"].map(
            lambda x: CLUSTER_LABELS.get(x, f"Group {x}")
        )
        return summary

    def save(self):
        joblib.dump({"scaler": self.scaler, "model": self.model, "labels": self.labels_,
                     "n_clusters": self.n_clusters_}, MODELS_DIR / "clustering.pkl")

    @classmethod
    def load(cls) -> "TurbineClustering":
        data = joblib.load(MODELS_DIR / "clustering.pkl")
        obj = cls()
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        obj.labels_ = data["labels"]
        obj.n_clusters_ = data.get("n_clusters", len(set(data["labels"])) - (1 if -1 in data["labels"] else 0))
        return obj
