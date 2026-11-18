
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reconstruction.classifier_emissions import (
    calculate_classifier_emissions
)

# Read only the first 1,000 rows for the initial test.
data = pd.read_csv(
    PROJECT_ROOT / "student_test_features.csv",
    nrows=1000
)

print("Input rows:", len(data))

# Generate classifier emission scores.
result = calculate_classifier_emissions(data)

print("\nFirst 10 candidate scores:")
print(
    result[
        [
            "trip_id",
            "point_id",
            "arc_id",
            "emission_probability",
            "log_emission"
        ]
    ].head(10).to_string(index=False)
)

print("\nProbability range:")
print(result["emission_probability"].describe())

assert result["emission_probability"].between(0, 1).all()
assert result["log_emission"].notna().all()

print("\nClassifier emission test passed!")
