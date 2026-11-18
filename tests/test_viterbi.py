
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from reconstruction.viterbi import viterbi


# GPS point 1: 2 candidate roads
# GPS point 2: 3 candidate roads
# GPS point 3: 2 candidate roads

emissions = [
    np.log([0.9, 0.1]),

    np.log([0.2, 0.7, 0.1]),

    np.log([0.1, 0.9])
]


# Matrix dimensions:
# 2 x 3, followed by 3 x 2

transitions = [
    np.array([
        [np.log(0.8), np.log(0.2), -np.inf],
        [np.log(0.1), np.log(0.8), np.log(0.1)]
    ]),

    np.array([
        [np.log(0.9), np.log(0.1)],
        [np.log(0.1), np.log(0.9)],
        [np.log(0.5), np.log(0.5)]
    ])
]

# Replace log(0) values with negative infinity.
# This explicitly marks impossible transitions.

transitions = [
    np.where(np.isneginf(matrix), -np.inf, matrix)
    for matrix in transitions
]

best_path = viterbi(emissions, transitions)

print("Variable candidate test")
print("Candidate counts:", [len(row) for row in emissions])
print("Most likely road sequence:", best_path)

assert len(best_path) == 3
assert best_path[0] < 2
assert best_path[1] < 3
assert best_path[2] < 2

print("Test passed!")
