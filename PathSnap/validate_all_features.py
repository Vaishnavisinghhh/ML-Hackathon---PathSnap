
from pathlib import Path
import gc
import numpy as np
import pandas as pd

from src.data_loader import load_trip
from src.graph_builder import build_road_graph
from src.candidate_generator import generate_candidates
from src.feature_engineering import extract_features


DATASETS = {
    "train": Path("data/Student folder/train"),
    "student_test": Path("data/Student folder/student_test"),
}


def validate_trip(trip_directory):

    trip_id = trip_directory.name

    # Load original trip data.
    trip = load_trip(trip_directory)

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    # Build directed road graph.
    graph = build_road_graph(nodes, arcs)

    # Generate GPS-to-road candidates.
    candidates = generate_candidates(
        track,
        graph,
        search_radius_m=50
    )

    # Extract features for every candidate.
    features = extract_features(
        track,
        candidates,
        graph
    )

    # Basic consistency checks.
    assert len(features) == len(candidates), (
        f"{trip_id}: Feature and candidate row counts differ"
    )

    assert features["distance_m"].notna().all(), (
        f"{trip_id}: Missing candidate distances"
    )

    assert features["road_length_m"].notna().all(), (
        f"{trip_id}: Missing road lengths"
    )

    # Calculate candidate coverage.
    covered_points = features["point_id"].nunique()
    total_points = len(track)

    coverage = (
        covered_points / total_points * 100
        if total_points else 0
    )

    # Verify first-point handling.
    first_point = features[
        features["point_id"] == 0
    ]

    first_point_correct = True

    if not first_point.empty:
        first_point_correct = (
            not first_point["speed_valid"].any()
            and not first_point["heading_valid"].any()
            and first_point["gps_speed"].isna().all()
            and first_point["bearing_diff"].isna().all()
        )

    assert first_point_correct, (
        f"{trip_id}: First GPS point has invalid motion features"
    )

    # Check all point and arc references.
    assert features["point_id"].between(
        0, len(track) - 1
    ).all(), f"{trip_id}: Invalid point IDs"

    assert features["arc_id"].isin(
        [int(data["arc_id"]) for _, _, _, data in
         graph.edges(keys=True, data=True)]
    ).all(), f"{trip_id}: Invalid arc IDs"

    # Check numerical feature values.
    numeric_columns = [
        "distance_m",
        "road_length_m",
        "gps_speed",
        "bearing_diff",
        "heading_consistency",
        "point_time_delta",
    ]

    for column in numeric_columns:
        values = features[column].dropna().to_numpy()

        assert np.isfinite(values).all(), (
            f"{trip_id}: Non-finite values in {column}"
        )

    return {
        "trip_id": trip_id,
        "gps_points": len(track),
        "road_nodes": len(nodes),
        "road_arcs": len(arcs),
        "candidate_rows": len(candidates),
        "feature_rows": len(features),
        "covered_points": covered_points,
        "coverage_percent": coverage,
        "valid_speed_rows": int(features["speed_valid"].sum()),
        "valid_heading_rows": int(features["heading_valid"].sum()),
        "time_gap_rows": int(features["time_gap"].sum()),
        "missing_speed_rows": int(features["gps_speed"].isna().sum()),
        "missing_heading_rows": int(features["bearing_diff"].isna().sum()),
        "missing_road_lengths": int(features["road_length_m"].isna().sum()),
        "first_point_correct": first_point_correct,
    }


def validate_dataset(dataset_name, dataset_directory):

    print("\n" + "=" * 75)
    print(f"VALIDATING DATASET: {dataset_name.upper()}")
    print("=" * 75)

    trip_directories = sorted(
        path for path in dataset_directory.iterdir()
        if path.is_dir()
    )

    results = []
    failures = []

    for index, trip_directory in enumerate(trip_directories, start=1):

        print(
            f"\n[{index}/{len(trip_directories)}] "
            f"Processing trip {trip_directory.name}..."
        )

        try:
            result = validate_trip(trip_directory)
            results.append(result)

            print(
                f"PASS | GPS: {result['gps_points']} | "
                f"Candidates: {result['candidate_rows']} | "
                f"Features: {result['feature_rows']} | "
                f"Coverage: {result['coverage_percent']:.2f}%"
            )

        except Exception as error:
            failures.append({
                "trip_id": trip_directory.name,
                "error": str(error),
            })

            print(f"FAIL | {error}")

        # Release trip-specific objects before the next trip.
        gc.collect()

    results_df = pd.DataFrame(results)
    failures_df = pd.DataFrame(failures)

    # Save validation reports.
    output_directory = Path("reports")
    output_directory.mkdir(exist_ok=True)

    results_df.to_csv(
        output_directory / f"{dataset_name}_feature_validation.csv",
        index=False
    )

    failures_df.to_csv(
        output_directory / f"{dataset_name}_feature_failures.csv",
        index=False
    )

    # Print summary.
    print("\n" + "-" * 75)
    print(f"{dataset_name.upper()} VALIDATION SUMMARY")
    print("-" * 75)

    print(f"Total trips: {len(trip_directories)}")
    print(f"Successful trips: {len(results)}")
    print(f"Failed trips: {len(failures)}")

    if not results_df.empty:
        print(
            f"Total GPS observations: "
            f"{results_df['gps_points'].sum()}"
        )

        print(
            f"Total candidate rows: "
            f"{results_df['candidate_rows'].sum()}"
        )

        print(
            f"Total feature rows: "
            f"{results_df['feature_rows'].sum()}"
        )

        print(
            f"Average candidate coverage: "
            f"{results_df['coverage_percent'].mean():.2f}%"
        )

        print(
            f"Total missing road lengths: "
            f"{results_df['missing_road_lengths'].sum()}"
        )

        print(
            f"All first-point checks passed: "
            f"{results_df['first_point_correct'].all()}"
        )

    if failures:
        print("\nFAILED TRIPS:")

        for failure in failures:
            print(
                f"{failure['trip_id']}: {failure['error']}"
            )

    print(
        f"\nReports saved in: {output_directory.resolve()}"
    )

    return results_df, failures_df


if __name__ == "__main__":

    all_results = {}

    for dataset_name, dataset_directory in DATASETS.items():

        if not dataset_directory.exists():
            print(f"Dataset directory not found: {dataset_directory}")
            continue

        results, failures = validate_dataset(
            dataset_name,
            dataset_directory
        )

        all_results[dataset_name] = {
            "results": results,
            "failures": failures,
        }

    print("\n" + "=" * 75)
    print("BATCH VALIDATION COMPLETED")
    print("=" * 75)
