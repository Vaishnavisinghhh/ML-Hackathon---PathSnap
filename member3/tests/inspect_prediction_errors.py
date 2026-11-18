
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reconstruction.classifier_emissions import (
    calculate_classifier_emissions
)

TRIP_ID = 81
NUM_OBSERVATIONS = 30

TRIP_FOLDER = ROOT / "data" / "train" / f"{TRIP_ID:08d}"

# Load reference route.
reference = np.loadtxt(
    TRIP_FOLDER / f"{TRIP_ID:08d}.route",
    dtype=int
).reshape(-1)

reference_set = set(reference.tolist())

# Load candidate features.
features = pd.read_csv(ROOT / "train_features.csv")

trip_features = features[
    features["trip_id"] == TRIP_ID
].copy()

point_ids = sorted(
    trip_features["point_id"].unique()
)[:NUM_OBSERVATIONS]

trip_features = trip_features[
    trip_features["point_id"].isin(point_ids)
].copy()

# Calculate classifier probabilities.
scored = calculate_classifier_emissions(trip_features)

print("\nTRIP 81: FIRST 30 OBSERVATIONS")
print("Reference route length:", len(reference))

for point_id, group in scored.groupby("point_id", sort=True):

    group = group.sort_values(
        "emission_probability",
        ascending=False
    )

    print(f"\nGPS point: {point_id}")

    for _, row in group.head(5).iterrows():

        arc_id = int(row["arc_id"])

        in_reference = arc_id in reference_set

        print(
            f"Arc: {arc_id:6d} | "
            f"Probability: {row['emission_probability']:.5f} | "
            f"In reference route: {in_reference}"
        )
