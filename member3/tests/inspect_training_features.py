
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT_ROOT / "train_features.csv"

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {FEATURE_FILE}"
    )

# Read a small sample first.
sample = pd.read_csv(FEATURE_FILE, nrows=5)

print("\n========== TRAINING FEATURES ==========")

print("\n1. COLUMNS")
print(sample.columns.tolist())

print("\n2. FIRST 5 ROWS")
print(sample.to_string(index=False))

print("\n3. FULL DATASET SUMMARY")
trip_stats = {}

# Read the large CSV in chunks to avoid loading all 236 MB at once.
for chunk in pd.read_csv(FEATURE_FILE, chunksize=200000):
    for trip_id, group in chunk.groupby("trip_id"):
        if trip_id not in trip_stats:
            trip_stats[trip_id] = {
                "rows": 0,
                "min_arc_id": float("inf"),
                "max_arc_id": float("-inf"),
                "min_point_id": float("inf"),
                "max_point_id": float("-inf"),
            }

        stats = trip_stats[trip_id]
        stats["rows"] += len(group)
        stats["min_arc_id"] = min(
            stats["min_arc_id"], group["arc_id"].min()
        )
        stats["max_arc_id"] = max(
            stats["max_arc_id"], group["arc_id"].max()
        )
        stats["min_point_id"] = min(
            stats["min_point_id"], group["point_id"].min()
        )
        stats["max_point_id"] = max(
            stats["max_point_id"], group["point_id"].max()
        )

for trip_id in sorted(trip_stats):
    stats = trip_stats[trip_id]
    print(
        f"Trip {trip_id}: "
        f"rows={stats['rows']}, "
        f"arc IDs={stats['min_arc_id']}..{stats['max_arc_id']}, "
        f"point IDs={stats['min_point_id']}..{stats['max_point_id']}"
    )

print("\n========== INSPECTION COMPLETE ==========")
