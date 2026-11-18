
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIP_ID = 1
TRIP_NAME = f"{TRIP_ID:08d}"

trip_dir = PROJECT_ROOT / "data" / "train" / TRIP_NAME

# Load road network.
arcs = np.loadtxt(
    trip_dir / f"{TRIP_NAME}.arcs",
    dtype=int,
    ndmin=2
)

nodes = np.loadtxt(
    trip_dir / f"{TRIP_NAME}.nodes",
    ndmin=2
)

# Check whether pyproj is installed.
try:
    import pyproj
    print("pyproj available:", pyproj.__version__)
except ImportError:
    print("pyproj is NOT installed")

# Count duplicate directed connections.
unique_arcs = np.unique(arcs, axis=0)
duplicate_count = len(arcs) - len(unique_arcs)

print("\n========== ROAD GRAPH INSPECTION ==========")
print("Number of nodes:", len(nodes))
print("Number of arcs:", len(arcs))
print("Unique directed connections:", len(unique_arcs))
print("Duplicate directed connections:", duplicate_count)

# Convert longitude/latitude coordinates into approximate metres.
# This is suitable for this local inspection; we'll use a proper
# metric projection for the final implementation.
longitude = nodes[:, 0]
latitude = nodes[:, 1]

lat0 = np.radians(np.mean(latitude))
lon0 = np.radians(np.mean(longitude))

earth_radius = 6371000.0

x = earth_radius * (np.radians(longitude) - lon0) * np.cos(lat0)
y = earth_radius * (np.radians(latitude) - lat0)

# Calculate the length of every directed arc.
dx = x[arcs[:, 1]] - x[arcs[:, 0]]
dy = y[arcs[:, 1]] - y[arcs[:, 0]]
lengths = np.sqrt(dx**2 + dy**2)

print("\nARC LENGTHS IN METRES")
print("Minimum:", round(lengths.min(), 2))
print("Median:", round(float(np.median(lengths)), 2))
print("Maximum:", round(lengths.max(), 2))

# Build a directed graph.
graph = nx.DiGraph()

for arc_id, (source, destination) in enumerate(arcs):
    graph.add_edge(
        int(source),
        int(destination),
        weight=float(lengths[arc_id]),
        arc_id=arc_id
    )

print("\nDIRECTED GRAPH")
print("Graph nodes:", graph.number_of_nodes())
print("Graph edges:", graph.number_of_edges())

# Read only Trip 1 candidate features.
parts = []

for chunk in pd.read_csv(
    PROJECT_ROOT / "train_features.csv",
    chunksize=200000
):
    trip_rows = chunk[chunk["trip_id"] == TRIP_ID]
    if not trip_rows.empty:
        parts.append(trip_rows)

features = pd.concat(parts, ignore_index=True)

print("\nCANDIDATE FEATURES")
print("Rows:", len(features))
print("GPS observations:", features["point_id"].nunique())

candidate_counts = features.groupby("point_id")["arc_id"].nunique()

print("Candidates per GPS point:")
print(candidate_counts.describe().round(2))

# Inspect the first five GPS observations.
groups = list(features.groupby("point_id", sort=True))[:5]

print("\nFIRST FIVE GPS POINTS")
for point_id, group in groups:
    print(
        f"Point {point_id}: "
        f"{len(group)} candidates, "
        f"arc IDs = {group['arc_id'].tolist()}"
    )

# Check direct directed connectivity between consecutive candidate sets.
print("\nDIRECT CONNECTION CHECKS")

for i in range(len(groups) - 1):
    point_a, candidates_a = groups[i]
    point_b, candidates_b = groups[i + 1]

    connected_pairs = []

    for arc_a in candidates_a["arc_id"]:
        end_node = int(arcs[int(arc_a), 1])

        for arc_b in candidates_b["arc_id"]:
            start_node = int(arcs[int(arc_b), 0])

            if end_node == start_node:
                connected_pairs.append((int(arc_a), int(arc_b)))

    print(
        f"Point {point_a} -> {point_b}: "
        f"{len(connected_pairs)} directly connected candidate pairs"
    )

print("\n========== INSPECTION COMPLETE ==========")
