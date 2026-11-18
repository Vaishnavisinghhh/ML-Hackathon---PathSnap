
import sys
from pathlib import Path

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
NUM_OBSERVATIONS = 100

TRIP_FOLDER = ROOT / "data" / "train" / f"{TRIP_ID:08d}"


def longest_common_subsequence(predicted, reference):
    """
    Find the longest ordered sequence shared by two arc sequences.

    Returns:
        matched_count: Number of matching arcs.
        matched_arcs: Matching arc IDs in order.
    """

    m = len(predicted)
    n = len(reference)

    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Build the LCS table.
    for i in range(1, m + 1):
        for j in range(1, n + 1):

            if predicted[i - 1] == reference[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1

            else:
                dp[i, j] = max(
                    dp[i - 1, j],
                    dp[i, j - 1]
                )

    # Recover matching arc IDs.
    matched_arcs = []

    i = m
    j = n

    while i > 0 and j > 0:

        if predicted[i - 1] == reference[j - 1]:
            matched_arcs.append(predicted[i - 1])
            i -= 1
            j -= 1

        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1

        else:
            j -= 1

    matched_arcs.reverse()

    return dp[m, n], matched_arcs


print(f"\nEvaluating Trip {TRIP_ID}")

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

reference_route = np.loadtxt(
    TRIP_FOLDER / f"{TRIP_ID:08d}.route",
    dtype=int
).reshape(-1)

# Load candidate features.
features = pd.read_csv(ROOT / "train_features.csv")

trip_features = features[
    features["trip_id"] == TRIP_ID
].copy()

if trip_features.empty:
    raise ValueError("No candidate features found for this trip.")

# Select the first 100 GPS observations.
first_point_ids = sorted(
    trip_features["point_id"].unique()
)[:NUM_OBSERVATIONS]

trip_features = trip_features[
    trip_features["point_id"].isin(first_point_ids)
].copy()

print("GPS observations:", len(first_point_ids))
print("Candidate rows:", len(trip_features))

# Calculate classifier emissions.
scored_features = calculate_classifier_emissions(
    trip_features
)

candidate_groups = [
    group.sort_values("arc_id").reset_index(drop=True)
    for _, group in scored_features.groupby(
        "point_id",
        sort=True
    )
]

log_emissions = [
    group["log_emission"].to_numpy()
    for group in candidate_groups
]

# Project GPS and road coordinates.
projected_nodes, projected_track, epsg = project_coordinates(
    nodes,
    track
)

print("Projection:", epsg)

# Build network transition matrices.
transitions = build_transition_matrices(
    arcs=arcs,
    projected_nodes=projected_nodes,
    projected_track=projected_track,
    candidate_groups=candidate_groups,
    beta=50.0
)

# Run Viterbi.
selected_indices = viterbi(
    log_emissions,
    transitions
)

predicted_arcs = [
    int(candidate_groups[t].iloc[index]["arc_id"])
    for t, index in enumerate(selected_indices)
]

# Remove consecutive duplicates.
# One road can contain several consecutive GPS observations.
collapsed_predictions = []

for arc_id in predicted_arcs:

    if (
        not collapsed_predictions
        or collapsed_predictions[-1] != arc_id
    ):
        collapsed_predictions.append(arc_id)

# Align predicted sequence with the reference route.
matched_count, matched_arcs = longest_common_subsequence(
    collapsed_predictions,
    reference_route.tolist()
)

# Calculate diagnostic metrics.
prediction_order_precision = (
    matched_count / len(collapsed_predictions)
    if collapsed_predictions else 0.0
)

reference_route_coverage = (
    matched_count / len(reference_route)
    if len(reference_route) else 0.0
)

print("\n========== ROUTE ALIGNMENT RESULTS ==========")

print("GPS observations:", len(predicted_arcs))
print("Predicted arcs before collapsing:", len(predicted_arcs))
print("Predicted arcs after collapsing:", len(collapsed_predictions))
print("Reference route length:", len(reference_route))

print("\nLCS matched arcs:", matched_count)

print(
    "Prediction order precision:",
    round(prediction_order_precision * 100, 2),
    "%"
)

print(
    "Full reference route coverage:",
    round(reference_route_coverage * 100, 2),
    "%"
)

print("\nMatched arc sequence:")
print(matched_arcs)

print("\nCollapsed predicted sequence:")
print(collapsed_predictions)

print("\nReference route first 30 arcs:")
print(reference_route[:30].tolist())

print("\n=============================================")
print("Route alignment evaluation completed.")
