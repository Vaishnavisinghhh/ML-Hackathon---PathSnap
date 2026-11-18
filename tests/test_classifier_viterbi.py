
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reconstruction.classifier_emissions import (
    calculate_classifier_emissions
)
from reconstruction.viterbi import viterbi


# -----------------------------------
# Step 1: Load actual candidate data
# -----------------------------------

data = pd.read_csv(
    PROJECT_ROOT / "student_test_features.csv"
)

print("Total candidate rows:", len(data))


# -----------------------------------
# Step 2: Calculate classifier scores
# -----------------------------------

scored_data = calculate_classifier_emissions(data)

print("Classifier scores generated successfully!")


# -----------------------------------
# Step 3: Select one trip
# -----------------------------------

trip_id = scored_data["trip_id"].iloc[0]

trip_data = scored_data[
    scored_data["trip_id"] == trip_id
].copy()

# Keep the first five complete GPS observations.
point_groups = list(
    trip_data.groupby("point_id", sort=False)
)[:5]

if len(point_groups) < 2:
    raise ValueError("At least two GPS observations are required")


# -----------------------------------
# Step 4: Prepare Viterbi emissions
# -----------------------------------

log_emissions = []
candidate_arc_ids = []

for point_id, group in point_groups:

    log_emissions.append(
        group["log_emission"].to_numpy(dtype=float)
    )

    candidate_arc_ids.append(
        group["arc_id"].tolist()
    )

print("\nCandidate counts per GPS point:")
print([len(row) for row in log_emissions])


# -----------------------------------
# Step 5: Create test transitions
# -----------------------------------

# Zero log-score means equal transition scores.
# This tests classifier-to-Viterbi integration only.
log_transitions = []

for t in range(len(log_emissions) - 1):

    matrix = np.zeros(
        (
            len(log_emissions[t]),
            len(log_emissions[t + 1])
        )
    )

    log_transitions.append(matrix)


# -----------------------------------
# Step 6: Run Viterbi
# -----------------------------------

best_path = viterbi(
    log_emissions,
    log_transitions
)


# -----------------------------------
# Step 7: Convert states to arc IDs
# -----------------------------------

predicted_arcs = []

for t, candidate_index in enumerate(best_path):

    arc_id = candidate_arc_ids[t][candidate_index]
    predicted_arcs.append(arc_id)


print("\nSelected trip:", trip_id)
print("Viterbi candidate indices:", best_path)
print("Predicted arc sequence:", predicted_arcs)

print("\nClassifier + Viterbi integration test passed!")
