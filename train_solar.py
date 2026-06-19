"""Train the solar power predictor on the Chinese State Grid 50 MW plant dataset."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.models.solar_predictor import SolarPredictor

EXCEL = Path("17304221/Solar station site 1 (Nominal capacity-50MW).xlsx")

if not EXCEL.exists():
    print(f"Dataset not found at {EXCEL}")
    sys.exit(1)

print(f"Loading {EXCEL} …")
df = pd.read_excel(EXCEL, sheet_name="sheet1")
print(f"  {len(df)} rows, columns: {list(df.columns)}")

predictor = SolarPredictor()
print("Training …")
predictor.fit(df)
predictor.save()
print(f"\nSaved to models/")
print(f"  Train MAE: {predictor.train_mae:.4f}  Val MAE: {predictor.val_mae:.4f}  (capacity factor, 0-1 scale)")
print(f"  = {predictor.train_mae * 50:.2f} MW train / {predictor.val_mae * 50:.2f} MW val on 50 MW plant")
