"""
Download German wind turbine data from Open Power System Data (OPSD).
Run once: python download_germany.py
Creates: data/germany_turbines.parquet
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.api.germany import download_germany_turbines

if __name__ == "__main__":
    df = download_germany_turbines()
    print(f"\nDone — {len(df)} turbines saved.")
    print(f"States: {df['state'].value_counts().head(5).to_dict()}")
    print(f"Capacity range: {df['capacity_mw'].min():.2f} – {df['capacity_mw'].max():.2f} MW")
