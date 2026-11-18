
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIP_ID = "00000001"

trip_dir = PROJECT_ROOT / "data" / "train" / TRIP_ID

arcs_file = trip_dir / f"{TRIP_ID}.arcs"
nodes_file = trip_dir / f"{TRIP_ID}.nodes"
track_file = trip_dir / f"{TRIP_ID}.track"
route_file = trip_dir / f"{TRIP_ID}.route"

# Load the four files.
arcs = np.loadtxt(arcs_file, dtype=int)
nodes = np.loadtxt(nodes_file)
track = np.loadtxt(track_file)
route = np.loadtxt(route_file, dtype=int, ndmin=1)

print("\n========== TRAINING TRIP INSPECTION ==========")

print("\n1. FILE SHAPES")
print("Arcs:", arcs.shape)
print("Nodes:", nodes.shape)
print("GPS track:", track.shape)
print("Reference route:", route.shape)

print("\n2. FIRST 5 ARC CONNECTIONS")
print(arcs[:5])

print("\n3. FIRST 5 NODE COORDINATES")
print(nodes[:5])

print("\n4. FIRST 5 GPS OBSERVATIONS")
print(track[:5])

print("\n5. FIRST 10 REFERENCE ROUTE ARC IDs")
print(route[:10])

print("\n6. BASIC VALIDATION")
print("Minimum arc node ID:", arcs.min())
print("Maximum arc node ID:", arcs.max())
print("Number of nodes:", len(nodes))
print("Number of arcs:", len(arcs))
print("Minimum route arc ID:", route.min())
print("Maximum route arc ID:", route.max())

print("\nAll route arc IDs within arc file:",
      route.min() >= 0 and route.max() < len(arcs))

print("\nAll arc node IDs within node file:",
      arcs.min() >= 0 and arcs.max() < len(nodes))

print("\n========== INSPECTION COMPLETE ==========")
