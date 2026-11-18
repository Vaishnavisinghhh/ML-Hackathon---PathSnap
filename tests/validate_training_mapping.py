
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT_ROOT / "train_features.csv"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"

# Collect the feature statistics for every trip.
trip_stats = {}

for chunk in pd.read_csv(FEATURE_FILE, chunksize=200000):
    for trip_id, group in chunk.groupby("trip_id"):
        trip_id = int(trip_id)

        if trip_id not in trip_stats:
            trip_stats[trip_id] = {
                "rows": 0,
                "max_arc_id": -1,
                "max_point_id": -1,
            }

        stats = trip_stats[trip_id]
        stats["rows"] += len(group)
        stats["max_arc_id"] = max(
            stats["max_arc_id"], int(group["arc_id"].max())
        )
        stats["max_point_id"] = max(
            stats["max_point_id"], int(group["point_id"].max())
        )

print("\n========== TRAINING MAPPING VALIDATION ==========")

passed = 0
failed = 0

for trip_id in sorted(trip_stats):
    trip_name = f"{trip_id:08d}"
    trip_dir = TRAIN_DIR / trip_name

    arcs_file = trip_dir / f"{trip_name}.arcs"
    track_file = trip_dir / f"{trip_name}.track"
    route_file = trip_dir / f"{trip_name}.route"

    problems = []

    if not trip_dir.exists():
        problems.append("Trip folder missing")
    elif not arcs_file.exists():
        problems.append("Arcs file missing")
    elif not track_file.exists():
        problems.append("Track file missing")
    elif not route_file.exists():
        problems.append("Route file missing")
    else:
        arcs = np.loadtxt(arcs_file, dtype=int, ndmin=2)
        track = np.loadtxt(track_file, ndmin=2)
        route = np.loadtxt(route_file, dtype=int, ndmin=1)

        stats = trip_stats[trip_id]

        if stats["max_arc_id"] >= len(arcs):
            problems.append(
                f"Candidate arc ID {stats['max_arc_id']} "
                f">= {len(arcs)} available arcs"
            )

        if stats["max_point_id"] >= len(track):
            problems.append(
                f"Point ID {stats['max_point_id']} "
                f">= {len(track)} GPS observations"
            )

        if route.min() < 0 or route.max() >= len(arcs):
            problems.append("Reference route contains invalid arc IDs")

    if problems:
        failed += 1
        print(f"\nTrip {trip_id}: FAILED")
        for problem in problems:
            print("  -", problem)
    else:
        passed += 1
        print(f"Trip {trip_id}: PASSED")

print("\n========== SUMMARY ==========")
print("Trips checked:", len(trip_stats))
print("Passed:", passed)
print("Failed:", failed)
print("================================")
