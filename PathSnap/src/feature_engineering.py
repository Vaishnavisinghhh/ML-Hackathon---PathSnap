
import math
import numpy as np
import pandas as pd

from src.graph_builder import haversine_distance, calculate_bearing


# Initial quality thresholds.
# These are configurable and should be evaluated during model validation.
MAX_TIME_GAP_SECONDS = 5.0
MAX_REASONABLE_SPEED_MPS = 50.0



def calculate_gps_motion(track):
    """
    Calculate movement information between consecutive GPS observations.

    Each interval is associated with its destination GPS point.

    For example:
        Point 0 -> Point 1: features stored at point 1.
        Point 1 -> Point 2: features stored at point 2.

    The first GPS point has no previous observation, so its
    speed and heading remain invalid.
    """

    track = np.asarray(track, dtype=float)

    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(
            "track must have three columns: longitude, latitude, timestamp"
        )

    if not np.isfinite(track).all():
        raise ValueError("Track contains non-finite values")

    number_of_points = len(track)

    gps_speed = np.full(number_of_points, np.nan)
    gps_heading = np.full(number_of_points, np.nan)
    point_time_delta = np.full(number_of_points, np.nan)

    speed_valid = np.zeros(number_of_points, dtype=bool)
    heading_valid = np.zeros(number_of_points, dtype=bool)
    time_gap = np.zeros(number_of_points, dtype=bool)

    # Start at point 1 because point 0 has no previous observation.
    for i in range(1, number_of_points):

        lon1, lat1, time1 = track[i - 1]
        lon2, lat2, time2 = track[i]

        delta_time = time2 - time1

        point_time_delta[i] = delta_time

        if delta_time <= 0:
            continue

        if delta_time > MAX_TIME_GAP_SECONDS:
            time_gap[i] = True
            continue

        distance = haversine_distance(
            lon1, lat1, lon2, lat2
        )

        speed = distance / delta_time

        if speed > MAX_REASONABLE_SPEED_MPS:
            continue

        gps_speed[i] = speed
        speed_valid[i] = True

        # Heading is undefined if the GPS position did not change.
        if distance > 0.01:
            gps_heading[i] = calculate_bearing(
                lon1, lat1, lon2, lat2
            )
            heading_valid[i] = True

    return {
        "gps_speed": gps_speed,
        "gps_heading": gps_heading,
        "point_time_delta": point_time_delta,
        "speed_valid": speed_valid,
        "heading_valid": heading_valid,
        "time_gap": time_gap,
    }



def extract_features(track, candidates, road_graph):
    """
    Extract model features for every GPS-point/road-arc candidate.

    Parameters
    ----------
    track : numpy array
        GPS observations with longitude, latitude, timestamp.

    candidates : pandas DataFrame
        Candidate rows containing point_id, arc_id, distance_m.

    road_graph : networkx.MultiDiGraph
        Directed road graph created by build_road_graph().

    Returns
    -------
    pandas DataFrame
        Candidate rows with engineered features and quality indicators.
    """

    track = np.asarray(track, dtype=float)

    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(
            "track must have three columns: longitude, latitude, timestamp"
        )

    if not np.isfinite(track).all():
        raise ValueError("Track contains non-finite values")

    required_columns = {
        "point_id",
        "arc_id",
        "distance_m",
    }

    missing_columns = required_columns - set(candidates.columns)

    if missing_columns:
        raise ValueError(
            f"Candidates are missing columns: {sorted(missing_columns)}"
        )

    if candidates.empty:
        return candidates.copy()

    result = candidates.copy()

    # Validate GPS point references.
    if (
        (result["point_id"] < 0).any()
        or (result["point_id"] >= len(track)).any()
    ):
        raise ValueError("Candidate contains an invalid point_id")

    # Build an original arc ID -> road geometry mapping.
    # Never assume graph iteration position equals original arc ID.
    road_data_by_arc_id = {}

    for _, _, _, data in road_graph.edges(
        keys=True,
        data=True
    ):
        arc_id = int(data["arc_id"])

        road_data_by_arc_id[arc_id] = {
            "road_length_m": float(data["length_m"]),
            "road_bearing_deg": float(data["bearing_deg"]),
        }

    # Retrieve road geometry using original arc IDs.
    road_information = result["arc_id"].map(
        road_data_by_arc_id
    )

    if road_information.isna().any():
        raise ValueError(
            "One or more candidate arc IDs do not exist in the road graph"
        )

    result["road_length_m"] = road_information.map(
        lambda item: item["road_length_m"]
    )

    result["_road_bearing"] = road_information.map(
        lambda item: item["road_bearing_deg"]
    )

    # Calculate GPS movement features.
    motion = calculate_gps_motion(track)

    point_ids = result["point_id"].to_numpy(dtype=int)

    result["gps_speed"] = motion["gps_speed"][point_ids]

    result["point_time_delta"] = (
        motion["point_time_delta"][point_ids]
    )

    result["speed_valid"] = (
        motion["speed_valid"][point_ids]
    )

    result["heading_valid"] = (
        motion["heading_valid"][point_ids]
    )

    result["time_gap"] = motion["time_gap"][point_ids]

    gps_headings = motion["gps_heading"][point_ids]

    road_bearings = result["_road_bearing"].to_numpy(
        dtype=float
    )

    # Calculate the smallest angular difference between
    # GPS movement direction and directed road direction.
    bearing_differences = np.full(len(result), np.nan)

    valid_headings = np.isfinite(gps_headings)

    bearing_differences[valid_headings] = np.abs(
        (
            gps_headings[valid_headings]
            - road_bearings[valid_headings]
            + 180
        ) % 360 - 180
    )

    result["bearing_diff"] = bearing_differences

    # Direction agreement:
    # +1 means aligned, 0 means perpendicular, -1 means opposite.
    result["heading_consistency"] = np.cos(
        np.radians(bearing_differences)
    )

    # Remove the temporary road bearing column.
    result = result.drop(columns=["_road_bearing"])

    # Keep a consistent column order.
    result = result[
        [
            "point_id",
            "arc_id",
            "distance_m",
            "bearing_diff",
            "gps_speed",
            "road_length_m",
            "heading_consistency",
            "point_time_delta",
            "speed_valid",
            "heading_valid",
            "time_gap",
        ]
    ]

    return result
