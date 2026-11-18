
from src.data_loader import load_trip
from src.graph_builder import build_road_graph
from src.candidate_generator import generate_candidates
from src.feature_engineering import extract_features


def main():

    trip = load_trip("data/Student folder/train/00000002")

    graph = build_road_graph(
        trip["nodes"],
        trip["arcs"]
    )

    candidates = generate_candidates(
        trip["track"],
        graph,
        search_radius_m=50
    )

    features = extract_features(
        trip["track"],
        candidates,
        graph
    )

    print("=" * 65)
    print("FEATURE ENGINEERING TEST")
    print("=" * 65)

    print(f"Trip: {trip['trip_id']}")
    print(f"GPS observations: {len(trip['track'])}")
    print(f"Candidate rows: {len(candidates)}")
    print(f"Feature rows: {len(features)}")

    print("\n========== FEATURE COLUMNS ==========")
    print(features.columns.tolist())

    print("\n========== FIRST 10 FEATURE ROWS ==========")
    print(features.head(10).to_string(index=False))

    print("\n========== FEATURE QUALITY ==========")

    print(
        "Valid speed rows:",
        features["speed_valid"].sum()
    )

    print(
        "Valid heading rows:",
        features["heading_valid"].sum()
    )

    print(
        "Time-gap rows:",
        features["time_gap"].sum()
    )

    print(
        "Missing speed values:",
        features["gps_speed"].isna().sum()
    )

    print(
        "Missing bearing differences:",
        features["bearing_diff"].isna().sum()
    )

    print(
        "Missing road lengths:",
        features["road_length_m"].isna().sum()
    )

    print("\n========== FEATURE STATISTICS ==========")
    print(
        features[
            [
                "distance_m",
                "bearing_diff",
                "gps_speed",
                "road_length_m",
                "heading_consistency",
                "point_time_delta",
            ]
        ].describe()
    )

    print("\nFeature engineering test completed!")


if __name__ == "__main__":
    main()
