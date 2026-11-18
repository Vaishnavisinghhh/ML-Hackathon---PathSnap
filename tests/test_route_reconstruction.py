
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from reconstruction.route_reconstruction import reconstruct_route


# --------------------------------
# 1. Create a synthetic road graph
# --------------------------------

graph = nx.DiGraph()

node_positions = {
    "A": (0, 0),
    "B": (100, 0),
    "C": (200, 0),
    "D": (300, 0),
    "E": (100, 100)
}

graph.add_edge("A", "B", length=100)
graph.add_edge("B", "C", length=100)
graph.add_edge("C", "D", length=100)
graph.add_edge("B", "E", length=100)


# --------------------------------
# 2. Create synthetic GPS points
# --------------------------------

observations = [
    (20, 3),
    (70, -4),
    (120, 5),
    (170, -3),
    (220, 4),
    (270, -2)
]


# --------------------------------
# 3. Reconstruct the road route
# --------------------------------

predicted_route = reconstruct_route(
    observations=observations,
    graph=graph,
    node_positions=node_positions,
    k=3,
    sigma=10,
    beta=50
)


# --------------------------------
# 4. Display results
# --------------------------------

print("\nGPS observations:", len(observations))

print("\nPredicted road sequence:")

for edge in predicted_route:
    print(edge)
