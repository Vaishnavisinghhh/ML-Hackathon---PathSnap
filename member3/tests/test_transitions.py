
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from reconstruction.transitions import transition_probability


print(
    "Similar distances:",
    transition_probability(105, 100)
)

print(
    "Different distances:",
    transition_probability(800, 100)
)

print(
    "Unreachable road:",
    transition_probability(float("inf"), 100)
)
