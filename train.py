"""
Run this script once to preprocess data, train models, and save everything.
Usage: python train.py
"""
import pandas as pd
from pathlib import Path
import joblib

from src.preprocessing import load_locations, load_scada, clean_scada, engineer_features, add_wake_features, compute_turbine_stats
from src.models.power_predictor import PowerPredictor
from src.models.clustering import TurbineClustering

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("1. Loading data...")
    locations = load_locations()
    print(f"   {len(locations)} turbines loaded")

    scada_raw = load_scada(use_full=False)
    print(f"   {len(scada_raw):,} raw SCADA rows loaded")

    print("2. Cleaning...")
    scada = clean_scada(scada_raw)
    print(f"   {len(scada):,} rows after cleaning ({len(scada_raw) - len(scada):,} removed)")

    print("3. Engineering features...")
    df = engineer_features(scada, locations)

    print("3b. Computing wake effect features (vectorised, ~3-5 min)...")
    df = add_wake_features(df, locations, max_dist_m=1000.0, wake_half_angle=30.0)
    print(f"   Features: {list(df.columns)}")

    print("4. Training power predictor (LightGBM)...")
    predictor = PowerPredictor()
    predictor.fit(df)
    predictor.save()
    print(f"   Saved. Val MAE: {predictor.val_mae:.1f} kW")

    print("5. Computing per-turbine stats...")
    stats = compute_turbine_stats(df)
    print(f"   Stats for {len(stats)} turbines")

    print("6. Clustering turbines (DBSCAN)...")
    clusterer = TurbineClustering(method="kmeans")
    clusterer.fit(stats)
    stats["cluster_id"] = clusterer.labels_
    clusterer.save()
    print(f"   {clusterer.n_clusters_} clusters found")
    print(clusterer.get_cluster_summary(stats).to_string(index=False))

    print("7. Saving turbine stats...")
    stats.to_parquet(MODELS_DIR / "turbine_stats.parquet", index=False)
    print(f"   Saved to models/turbine_stats.parquet")

    print("=" * 60)
    print("Training complete. Start the API with:")
    print("  .venv\\Scripts\\uvicorn src.api.main:app --reload --port 8000")
    print("=" * 60)

    # Feature importance summary
    imp = predictor.feature_importance().head(10)
    print("\nTop 10 most important features:")
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
