
from pathlib import Path
import pandas as pd

from src.data_loader import load_trip
from src.graph_builder import build_road_graph
from src.candidate_generator import generate_candidates
from src.feature_engineering import extract_features


DATASETS = {
    "train": Path("data/Student folder/train"),
    "student_test": Path("data/Student folder/student_test"),
}

OUTPUT_DIR = Path("data/processed")


def export_dataset(dataset_name, dataset_directory):

    trip_directories = sorted(
        path for path in dataset_directory.iterdir()
        if path.is_dir()
    )

    feature_output = OUTPUT_DIR / f"{dataset_name}_features.csv"
    route_output = OUTPUT_DIR / "training_routes.csv"

    # Remove previous output so repeated runs don't duplicate rows.
    if feature_output.exists():
        feature_output.unlink()

    if dataset_name == "train" and route_output.exists():
        route_output.unlink()

    total_feature_rows = 0
    total_route_rows = 0

    for index, trip_directory in enumerate(trip_directories, start=1):

        trip_id = trip_directory.name

        print(
            f"[{index}/{len(trip_directories)}] "
            f"Processing {dataset_name}: {trip_id}"
        )

        # Load trip files.
        trip = load_trip(trip_directory)

        track = trip["track"]
        nodes = trip["nodes"]
        arcs = trip["arcs"]
        route = trip["route"]

        # Build road graph.
        graph = build_road_graph(nodes, arcs)

        # Generate candidates.
        candidates = generate_candidates(
            track,
            graph,
            search_radius_m=50
        )

        # Extract features.
        features = extract_features(
            track,
            candidates,
            graph
        )

        # Add trip identifier.
        features.insert(0, "trip_id", trip_id)

        # Ensure candidate and feature counts match.
        assert len(features) == len(candidates)

        # Export features incrementally.
        features.to_csv(
            feature_output,
            mode="a",
            header=not feature_output.exists(),
            index=False
        )

        total_feature_rows += len(features)

        # Preserve ordered reference routes for training.
        if dataset_name == "train":

            route_df = pd.DataFrame({
                "trip_id": trip_id,
                "route_position": range(len(route)),
                "arc_id": route.astype(int),
            })

            route_df.to_csv(
                route_output,
                mode="a",
                header=not route_output.exists(),
                index=False
            )

            total_route_rows += len(route_df)

        print(
            f"   GPS points: {len(track)} | "
            f"Candidates/features: {len(features)}"
        )

        del graph, candidates, features, trip

    print("\n" + "-" * 65)
    print(f"{dataset_name.upper()} EXPORT COMPLETE")
    print("-" * 65)
    print(f"Trips processed: {len(trip_directories)}")
    print(f"Feature rows exported: {total_feature_rows}")
    print(f"Feature file: {feature_output}")

    if dataset_name == "train":
        print(f"Reference route entries: {total_route_rows}")
        print(f"Route file: {route_output}")


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_directory in DATASETS.items():

        if not dataset_directory.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_directory}"
            )

        export_dataset(dataset_name, dataset_directory)

    print("\nALL DATASETS EXPORTED SUCCESSFULLY!")
