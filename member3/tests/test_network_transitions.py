
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reconstruction.network_transitions import (
    project_coordinates,
    build_transition_matrices
)
from src.reconstruction.classifier_emissions import (
    calculate_classifier_emissions
)
from src.reconstruction.viterbi import viterbi


TRIP_ID = 81
TRIP_FOLDER = ROOT / "data" / "train" / f"{TRIP_ID:08d}"

print(f"\nTesting trip {TRIP_ID}")

# Load trip data.
arcs = np.loadtxt(
    TRIP_FOLDER / f"{TRIP_ID:08d}.arcs",
    dtype=int
)

nodes = np.loadtxt(
    TRIP_FOLDER / f"{TRIP_ID:08d}.nodes"
)

track = np.loadtxt(
    TRIP_FOLDER / f"{TRIP_ID:08d}.track"
)

# Load features.
features = pd.read_csv(ROOT / "train_features.csv")

trip_features = features[
    features["trip_id"] == TRIP_ID
].copy()

if trip_features.empty:
    raise ValueError(f"No features found for trip {TRIP_ID}")

# IMPORTANT: Select only the first 20 GPS observations
# BEFORE calculating classifier emissions.
first_point_ids = sorted(
    trip_features["point_id"].unique()
)[:100]

trip_features = trip_features[
    trip_features["point_id"].isin(first_point_ids)
].copy()

print("GPS point IDs:", first_point_ids)
print("Candidate rows being scored:", len(trip_features))

# Score all selected candidates in ONE model call.
start = time.perf_counter()

scored_features = calculate_classifier_emissions(
    trip_features
)

print(
    "Classifier scoring time:",
    round(time.perf_counter() - start, 2),
    "seconds"
)

# Group candidates by GPS point.
candidate_groups = [
    group.sort_values("arc_id").reset_index(drop=True)
    for _, group in scored_features.groupby("point_id", sort=True)
]

point_ids = [
    int(group["point_id"].iloc[0])
    for group in candidate_groups
]

print("Actual GPS point IDs:", point_ids)
print("Candidate counts:", [len(g) for g in candidate_groups])

log_emissions = [
    group["log_emission"].to_numpy()
    for group in candidate_groups
]

# Project coordinates.
projected_nodes, projected_track, epsg = project_coordinates(
    nodes,
    track
)

print("Projection EPSG:", epsg)

# Build network transitions.
start = time.perf_counter()

transitions = build_transition_matrices(
    arcs=arcs,
    projected_nodes=projected_nodes,
    projected_track=projected_track,
    candidate_groups=candidate_groups,
    beta=50.0
)

print(
    "Transition calculation time:",
    round(time.perf_counter() - start, 2),
    "seconds"
)

print("\nTransition matrix shapes:")
for matrix in transitions:
    print(matrix.shape)

print("\nFinite transition counts:")
for matrix in transitions:
    print(f"{np.isfinite(matrix).sum()} / {matrix.size}")

# Run Viterbi.
selected_indices = viterbi(log_emissions, transitions)

predicted_arcs = [
    int(candidate_groups[t].iloc[index]["arc_id"])
    for t, index in enumerate(selected_indices)
]

print("\nPredicted road arc IDs:", predicted_arcs)

reference_route = np.loadtxt(
    TRIP_FOLDER / f"{TRIP_ID:08d}.route",
    dtype=int
).reshape(-1)

print("Reference route first 20 arcs:", reference_route[:20])

print("\nTrip 81 test completed successfully.")
