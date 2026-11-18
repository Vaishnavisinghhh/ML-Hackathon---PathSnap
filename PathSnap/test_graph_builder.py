
from src.data_loader import load_trip
from src.graph_builder import build_road_graph

# Load one known training trip
trip = load_trip("data/Student folder/train/00000001")

nodes = trip["nodes"]
arcs = trip["arcs"]

# Build its road graph
graph = build_road_graph(nodes, arcs)

print("========== GRAPH BUILDER TEST ==========")

print("Trip:", trip["trip_id"])
print("Graph type:", type(graph).__name__)

print("Expected nodes:", len(nodes))
print("Actual graph nodes:", graph.number_of_nodes())

print("Expected arcs:", len(arcs))
print("Actual graph edges:", graph.number_of_edges())

# Inspect the first original arc.
arc_id = 0
start_node, end_node = arcs[arc_id]

edge = graph.edges[
    int(start_node),
    int(end_node),
    arc_id
]

print("\n========== FIRST ARC ==========")

print("Original arc ID:", arc_id)
print("Start node:", start_node)
print("End node:", end_node)

print("Stored arc ID:", edge["arc_id"])
print("Length (m):", round(edge["length_m"], 3))
print("Bearing (degrees):", round(edge["bearing_deg"], 3))

print("\n========== GRAPH VALIDATION ==========")

print(
    "All original arc IDs preserved:",
    all(
        graph.has_edge(
            int(start),
            int(end),
            key=arc_id
        )
        for arc_id, (start, end) in enumerate(arcs)
    )
)

print(
    "Directed graph:",
    graph.is_directed()
)

print(
    "Multiple edges supported:",
    graph.is_multigraph()
)
