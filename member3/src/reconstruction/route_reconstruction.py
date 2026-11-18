
import math

import networkx as nx
import numpy as np

from reconstruction.emissions import emission_probability
from reconstruction.transitions import transition_probability
from reconstruction.viterbi import viterbi


def point_to_segment_distance(point, start, end):
    """Calculate the shortest distance from a point to a road segment."""

    px, py = point
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    length_squared = dx * dx + dy * dy

    if length_squared == 0:
        return math.hypot(px - x1, py - y1), 0.0

    fraction = (
        (px - x1) * dx + (py - y1) * dy
    ) / length_squared

    fraction = max(0.0, min(1.0, fraction))

    closest_x = x1 + fraction * dx
    closest_y = y1 + fraction * dy

    distance = math.hypot(
        px - closest_x,
        py - closest_y
    )

    return distance, fraction


def generate_candidates(observation, graph, node_positions, k=3):
    """Find the k nearest directed road segments."""

    candidates = []

    for u, v, data in graph.edges(data=True):

        distance, fraction = point_to_segment_distance(
            observation,
            node_positions[u],
            node_positions[v]
        )

        candidates.append({
            "edge": (u, v),
            "distance": distance,
            "fraction": fraction,
            "length": data["length"]
        })

    candidates.sort(
        key=lambda candidate: candidate["distance"]
    )

    return candidates[:k]


def gps_distance(point_a, point_b):
    """Calculate straight-line distance between GPS points."""

    return math.hypot(
        point_b[0] - point_a[0],
        point_b[1] - point_a[1]
    )


def network_distance_between_candidates(
    candidate_a,
    candidate_b,
    graph
):
    """Calculate directed road-network distance between candidates."""

    edge_a = candidate_a["edge"]
    edge_b = candidate_b["edge"]

    u1, v1 = edge_a
    u2, v2 = edge_b

    length_a = candidate_a["length"]
    length_b = candidate_b["length"]

    fraction_a = candidate_a["fraction"]
    fraction_b = candidate_b["fraction"]

    # Same directed road.
    if edge_a == edge_b:

        if fraction_b < fraction_a:
            return float("inf")

        return (fraction_b - fraction_a) * length_a

    # Remaining distance on the first road.
    remaining_a = (1.0 - fraction_a) * length_a

    # Distance travelled on the final road.
    progress_b = fraction_b * length_b

    try:
        middle_distance = nx.shortest_path_length(
            graph,
            source=v1,
            target=u2,
            weight="length"
        )

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")

    return remaining_a + middle_distance + progress_b


def reconstruct_route(
    observations,
    graph,
    node_positions,
    k=3,
    sigma=10.0,
    beta=50.0
):
    """
    Reconstruct the most likely road sequence from GPS observations.

    Supports different candidate counts at different observations.
    Coordinates and road lengths must use metres.
    """

    # --------------------------------
    # Input validation
    # --------------------------------

    if len(observations) == 0:
        raise ValueError("GPS observations cannot be empty")

    if k <= 0:
        raise ValueError("k must be positive")

    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    if beta <= 0:
        raise ValueError("Beta must be positive")

    if graph.number_of_edges() == 0:
        raise ValueError("Road graph contains no edges")

    for observation in observations:

        if len(observation) != 2:
            raise ValueError(
                "Each GPS observation must contain x and y coordinates"
            )

        if not all(math.isfinite(value) for value in observation):
            raise ValueError(
                "GPS coordinates must be finite numbers"
            )

    # --------------------------------
    # Step 1: Generate candidates
    # --------------------------------

    all_candidates = []

    for observation in observations:

        candidates = generate_candidates(
            observation,
            graph,
            node_positions,
            k
        )

        if not candidates:
            raise ValueError("No road candidates found")

        all_candidates.append(candidates)

    # --------------------------------
    # Step 2: Emission log-scores
    # --------------------------------

    log_emissions = []

    for candidates in all_candidates:

        emission_scores = []

        for candidate in candidates:

            probability = emission_probability(
                candidate["distance"],
                sigma
            )

            emission_scores.append(math.log(probability))

        log_emissions.append(
            np.asarray(emission_scores, dtype=float)
        )

    # --------------------------------
    # Step 3: Transition log-scores
    # --------------------------------

    log_transitions = []

    for t in range(1, len(observations)):

        gps_dist = gps_distance(
            observations[t - 1],
            observations[t]
        )

        previous_candidates = all_candidates[t - 1]
        current_candidates = all_candidates[t]

        transition_matrix = np.full(
            (
                len(previous_candidates),
                len(current_candidates)
            ),
            -np.inf
        )

        for i, candidate_a in enumerate(previous_candidates):

            for j, candidate_b in enumerate(current_candidates):

                network_dist = network_distance_between_candidates(
                    candidate_a,
                    candidate_b,
                    graph
                )

                probability = transition_probability(
                    network_dist,
                    gps_dist,
                    beta
                )

                if probability > 0:
                    transition_matrix[i, j] = math.log(probability)

        log_transitions.append(transition_matrix)

    # --------------------------------
    # Step 4: Viterbi algorithm
    # --------------------------------

    best_path = viterbi(
        log_emissions,
        log_transitions
    )

    # --------------------------------
    # Step 5: Reconstruct road sequence
    # --------------------------------

    predicted_route = []

    for t, candidate_index in enumerate(best_path):

        selected_candidate = all_candidates[t][candidate_index]

        predicted_route.append(
            selected_candidate["edge"]
        )

    return predicted_route
