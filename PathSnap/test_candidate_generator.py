
from src.data_loader import load_trip
from src.graph_builder import build_road_graph
from src.candidate_generator import generate_candidates

# Load one training trip.
trip = load_trip("data/Student folder/train/00000001")

# Build its road network.
graph = build_road_graph(
    trip["nodes"],
    trip["arcs"]
)

# Generate candidates.
candidates = generate_candidates(
    trip["track"],
    graph,
    search_radius_m=50
)

print("========== CANDIDATE GENERATOR TEST ==========")

print("Trip:", trip["trip_id"])
print("GPS observations:", len(trip["track"]))
print("Road arcs:", len(trip["arcs"]))

print("\nTotal candidate rows:", len(candidates))

points_with_candidates = candidates["point_id"].nunique()

print(
    "GPS points with candidates:",
    points_with_candidates
)

print(
    "GPS points without candidates:",
    len(trip["track"]) - points_with_candidates
)

print(
    "Candidate coverage:",
    round(
        points_with_candidates / len(trip["track"]) * 100,
        2
    ),
    "%"
)

if not candidates.empty:

    print(
        "Average candidates per GPS point:",
        round(
            len(candidates) / points_with_candidates,
            2
        )
    )

    print(
        "Minimum candidate distance:",
        round(candidates["distance_m"].min(), 3),
        "m"
    )

    print(
        "Maximum candidate distance:",
        round(candidates["distance_m"].max(), 3),
        "m"
    )

    print("\n========== FIRST 10 CANDIDATES ==========")
    print(candidates.head(10).to_string(index=False))

    print("\n========== FIRST GPS POINT CANDIDATES ==========")

    print(
        candidates[
            candidates["point_id"] == 0
        ].head(10).to_string(index=False)
    )

else:
    print("WARNING: No candidates were generated!")
