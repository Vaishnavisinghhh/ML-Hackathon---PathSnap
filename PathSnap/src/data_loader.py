
from pathlib import Path
import numpy as np


def load_trip(trip_directory):
    """
    Load all four files belonging to one trip.

    Expected files:
        <trip_id>.track
        <trip_id>.nodes
        <trip_id>.arcs
        <trip_id>.route

    Returns a dictionary containing:
        trip_id, track, nodes, arcs, route
    """

    trip_directory = Path(trip_directory)
    trip_id = trip_directory.name

    track_file = trip_directory / f"{trip_id}.track"
    nodes_file = trip_directory / f"{trip_id}.nodes"
    arcs_file = trip_directory / f"{trip_id}.arcs"
    route_file = trip_directory / f"{trip_id}.route"

    required_files = [
        track_file,
        nodes_file,
        arcs_file,
        route_file,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file for trip {trip_id}: {file_path}"
            )

    # ndmin=2 ensures arrays remain two-dimensional,
    # even when a file contains only one row.
    track = np.loadtxt(track_file, dtype=float, ndmin=2)
    nodes = np.loadtxt(nodes_file, dtype=float, ndmin=2)
    arcs = np.loadtxt(arcs_file, dtype=int, ndmin=2)
    route = np.loadtxt(route_file, dtype=int, ndmin=1)

    # Validate expected column counts.
    if track.shape[1] != 3:
        raise ValueError(
            f"{trip_id}: expected 3 columns in .track, "
            f"found {track.shape[1]}"
        )

    if nodes.shape[1] != 2:
        raise ValueError(
            f"{trip_id}: expected 2 columns in .nodes, "
            f"found {nodes.shape[1]}"
        )

    if arcs.shape[1] != 2:
        raise ValueError(
            f"{trip_id}: expected 2 columns in .arcs, "
            f"found {arcs.shape[1]}"
        )

    # Validate that arc endpoints refer to existing nodes.
    if arcs.size:
        if arcs.min() < 0 or arcs.max() >= len(nodes):
            raise ValueError(
                f"{trip_id}: arc endpoint refers to a "
                "nonexistent node"
            )

    # Validate that route entries refer to existing arc rows.
    if route.size:
        if route.min() < 0 or route.max() >= len(arcs):
            raise ValueError(
                f"{trip_id}: route contains an invalid arc ID"
            )

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }


def load_all_trips(data_directory):
    """
    Load every trip folder inside a directory.

    Example:
        data/Student folder/train

    Returns:
        A list of trip dictionaries.
    """

    data_directory = Path(data_directory)

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_directory}"
        )

    trip_directories = sorted(
        path for path in data_directory.iterdir()
        if path.is_dir()
    )

    trips = []

    for trip_directory in trip_directories:
        trip = load_trip(trip_directory)
        trips.append(trip)

    return trips
