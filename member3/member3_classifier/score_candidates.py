from pathlib import Path
import sys, json
import pandas as pd
import joblib

# Usage: python score_candidates.py input.csv scores.csv
root = Path(__file__).resolve().parent
spec = json.loads((root / "inference.json").read_text())
model = joblib.load(root / spec["model_file"])
rows = pd.read_csv(sys.argv[1], dtype={"trip_id": str})
scores = rows[["trip_id", "point_id", "arc_id"]].copy()
scores["score"] = model.predict_proba(rows[spec["columns"]])[:, list(model.classes_).index(1)]
scores.to_csv(sys.argv[2], index=False)
