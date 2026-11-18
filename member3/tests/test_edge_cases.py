
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from reconstruction.route_reconstruction import reconstruct_route


# Create two disconnected roads.
graph = nx.DiGraph()

node_positions = {
    "A": (0, 0),
    "B": (100, 0),
    "C": (1000, 0),
    "D": (1100, 0)
}

graph.add_edge("A", "B", length=100)
graph.add_edge("C", "D", length=100)


# First GPS point is near A-B.
# Second GPS point is near C-D.
observations = [
    (25, 2),
    (1025, 2)
]


print("Testing disconnected roads...")

try:

    route = reconstruct_route(
        observations=observations,
        graph=graph,
        node_positions=node_positions,
        k=1,
        sigma=10,
        beta=50
    )

    print("Unexpected route:", route)

except ValueError as error:

    print("Correctly detected invalid route:")
    print(error)

    assert str(error) == "No valid state sequence exists"

    print("Disconnected road test passed!")


# --------------------------------
# Test 2: Single GPS observation
# --------------------------------

print("\nTesting single GPS observation...")

single_observation = [
    (25, 2)
]

single_graph = nx.DiGraph()

single_positions = {
    "A": (0, 0),
    "B": (100, 0),
    "C": (200, 0)
}

single_graph.add_edge("A", "B", length=100)
single_graph.add_edge("B", "C", length=100)

route = reconstruct_route(
    observations=single_observation,
    graph=single_graph,
    node_positions=single_positions,
    k=2,
    sigma=10,
    beta=50
)

print("Predicted route:", route)

assert len(route) == 1
assert route[0] == ("A", "B")

print("Single observation test passed!")


# --------------------------------
# Test 3: Empty GPS observations
# --------------------------------

print("\nTesting empty GPS observations...")

try:
    reconstruct_route(
        observations=[],
        graph=single_graph,
        node_positions=single_positions
    )

    raise AssertionError("Empty observations were accepted")

except ValueError as error:
    assert str(error) == "GPS observations cannot be empty"
    print("Empty observations correctly rejected!")


# --------------------------------
# Test 4: Invalid model parameters
# --------------------------------

print("\nTesting invalid parameters...")

try:
    reconstruct_route(
        observations=[(25, 2)],
        graph=single_graph,
        node_positions=single_positions,
        sigma=0
    )

    raise AssertionError("Invalid sigma was accepted")

except ValueError as error:
    assert str(error) == "Sigma must be positive"
    print("Invalid sigma correctly rejected!")


# --------------------------------
# Test 5: Invalid GPS coordinates
# --------------------------------

print("\nTesting invalid GPS coordinates...")

try:
    reconstruct_route(
        observations=[(float("nan"), 2)],
        graph=single_graph,
        node_positions=single_positions
    )

    raise AssertionError("Invalid coordinates were accepted")

except ValueError as error:
    assert str(error) == "GPS coordinates must be finite numbers"
    print("Invalid GPS coordinates correctly rejected!")
