
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_000


def project_coordinates(lon, lat, reference_latitude):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    x = (
        EARTH_RADIUS_M
        * np.radians(lon)
        * math.cos(math.radians(reference_latitude))
    )
    y = EARTH_RADIUS_M * np.radians(lat)

    return np.column_stack((x, y))


def point_to_segment_distance(point, start, end):
    segment = end - start
    segment_length_squared = np.dot(segment, segment)

    if segment_length_squared == 0:
        return float(np.linalg.norm(point - start))

    projection = np.dot(point - start, segment)
    projection /= segment_length_squared
    projection = np.clip(projection, 0.0, 1.0)

    closest_point = start + projection * segment

    return float(np.linalg.norm(point - closest_point))


def generate_candidates(
    track,
    road_graph,
    search_radius_m=50,
    sample_spacing_m=25
):
    track = np.asarray(track, dtype=float)

    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(
            "track must have three columns: longitude, latitude, timestamp"
        )

    if not np.isfinite(track).all():
        raise ValueError("GPS track contains invalid values")

    if search_radius_m <= 0:
        raise ValueError("search_radius_m must be positive")

    if sample_spacing_m <= 0:
        raise ValueError("sample_spacing_m must be positive")

    if road_graph.number_of_edges() == 0:
        raise ValueError("Road graph contains no edges")

    reference_latitude = float(np.mean(track[:, 1]))

    gps_xy = project_coordinates(
        track[:, 0],
        track[:, 1],
        reference_latitude
    )

    arc_ids = []
    segment_starts = []
    segment_ends = []
    spatial_samples = []
    sample_arc_ids = []

    for start_node, end_node, key, data in road_graph.edges(
        keys=True,
        data=True
    ):
        start = project_coordinates(
            [data["start_lon"]],
            [data["start_lat"]],
            reference_latitude
        )[0]

        end = project_coordinates(
            [data["end_lon"]],
            [data["end_lat"]],
            reference_latitude
        )[0]

        arc_id = int(data["arc_id"])

        arc_ids.append(arc_id)
        segment_starts.append(start)
        segment_ends.append(end)

        length = float(data["length_m"])

        number_of_intervals = max(
            1,
            math.ceil(length / sample_spacing_m)
        )

        fractions = np.linspace(
            0.0,
            1.0,
            number_of_intervals + 1
        )

        samples = (
            start[None, :]
            + fractions[:, None] * (end - start)[None, :]
        )

        spatial_samples.extend(samples)
        sample_arc_ids.extend([arc_id] * len(samples))

    arc_ids = np.asarray(arc_ids, dtype=int)
    segment_starts = np.asarray(segment_starts, dtype=float)
    segment_ends = np.asarray(segment_ends, dtype=float)
    spatial_samples = np.asarray(spatial_samples, dtype=float)
    sample_arc_ids = np.asarray(sample_arc_ids, dtype=int)

    # Map each original arc ID to its position in the geometry arrays.
    # Graph iteration order is not guaranteed to match original arc order.
    arc_position_by_id = {
        int(arc_id): position
        for position, arc_id in enumerate(arc_ids)
    }

    tree = cKDTree(spatial_samples)

    query_radius = search_radius_m + sample_spacing_m / 2

    results = []

    for point_id, gps_point in enumerate(gps_xy):

        nearby_sample_indices = tree.query_ball_point(
            gps_point,
            query_radius
        )

        if not nearby_sample_indices:
            continue

        nearby_arc_ids = np.unique(
            sample_arc_ids[nearby_sample_indices]
        )

        for arc_id in nearby_arc_ids:

            # Correctly retrieve geometry using the original arc ID.
            arc_position = arc_position_by_id[int(arc_id)]

            distance = point_to_segment_distance(
                gps_point,
                segment_starts[arc_position],
                segment_ends[arc_position]
            )

            if distance <= search_radius_m:
                results.append(
                    (
                        int(point_id),
                        int(arc_id),
                        float(distance)
                    )
                )

    candidates = pd.DataFrame(
        results,
        columns=[
            "point_id",
            "arc_id",
            "distance_m"
        ]
    )

    if not candidates.empty:
        candidates = candidates.sort_values(
            ["point_id", "distance_m", "arc_id"]
        ).reset_index(drop=True)

    return candidates
