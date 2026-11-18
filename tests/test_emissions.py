
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from reconstruction.emissions import emission_probability


print("Distance: 2m  =", emission_probability(2))
print("Distance: 10m =", emission_probability(10))
print("Distance: 40m =", emission_probability(40))
