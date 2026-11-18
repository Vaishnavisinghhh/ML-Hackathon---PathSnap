
import numpy as np
import networkx as nx
from pyproj import CRS, Transformer


def project_coordinates(nodes, track):
    """Project longitude/latitude coordinates into metres."""

    lon0 = float(np.mean(nodes[:, 0]))
    lat0 = float(np.mean(nodes[:, 1]))

    zone = int((lon0 + 180) // 6) + 1
    epsg = (32600 if lat0 >= 0 else 32700) + zone

    transformer = Transformer.from_crs(
        "EPSG:4326",
        CRS.from_epsg(epsg),
        always_xy=True
    )

    node_x, node_y = transformer.transform(nodes[:, 0], nodes[:, 1])
    track_x, track_y = transformer.transform(track[:, 0], track[:, 1])

    return (
        np.column_stack((node_x, node_y)),
        np.column_stack((track_x, track_y)),
        epsg
    )


def calculate_arc_positions(
    projected_nodes,
    projected_track,
    arcs,
    arc_ids,
    point_ids=None
):
    """Calculate GPS position fractions along candidate road segments."""

    if point_ids is None:
        point_ids = list(range(len(arc_ids)))

    if len(point_ids) != len(arc_ids):
        raise ValueError("point_ids and arc_ids must have equal lengths")

    fractions = []

    for group_index, candidate_arc_ids in enumerate(arc_ids):

        point_id = int(point_ids[group_index])

        if not 0 <= point_id < len(projected_track):
            raise ValueError(f"Invalid GPS point ID: {point_id}")

        point = projected_track[point_id]
        point_fractions = []

        for arc_id in candidate_arc_ids:

            arc_id = int(arc_id)

            if not 0 <= arc_id < len(arcs):
                raise ValueError(f"Invalid arc ID: {arc_id}")

            source, destination = arcs[arc_id]

            start = projected_nodes[source]
            end = projected_nodes[destination]

            vector = end - start
            denominator = np.dot(vector, vector)

            if denominator <= 1e-12:
                fraction = 0.0
            else:
                fraction = np.dot(point - start, vector) / denominator
                fraction = float(np.clip(fraction, 0.0, 1.0))

            point_fractions.append(fraction)

        fractions.append(np.asarray(point_fractions))

    return fractions


def build_road_graph(projected_nodes, arcs):
    """Build a directed road graph with lengths in metres."""

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(projected_nodes)))

    for source, destination in arcs:

        source = int(source)
        destination = int(destination)

        length = float(np.linalg.norm(
            projected_nodes[destination] - projected_nodes[source]
        ))

        if graph.has_edge(source, destination):
            graph[source][destination]["weight"] = min(
                graph[source][destination]["weight"],
                length
            )
        else:
            graph.add_edge(source, destination, weight=length)

    return graph


def build_transition_matrices(
    arcs,
    projected_nodes,
    projected_track,
    candidate_groups,
    beta=50.0
):
    """Build road-network log-transition matrices for Viterbi."""

    if beta <= 0:
        raise ValueError("beta must be positive")

    if len(candidate_groups) == 0:
        raise ValueError("Candidate groups cannot be empty")

    for group in candidate_groups:
        if group.empty:
            raise ValueError("Candidate groups cannot be empty")
        if "arc_id" not in group.columns or "point_id" not in group.columns:
            raise ValueError("Each group must contain arc_id and point_id")
        if group["point_id"].nunique() != 1:
            raise ValueError("Each group must contain exactly one point_id")

    point_ids = [
        int(group["point_id"].iloc[0])
        for group in candidate_groups
    ]

    if any(
        point_ids[i + 1] <= point_ids[i]
        for i in range(len(point_ids) - 1)
    ):
        raise ValueError("Candidate groups must have increasing point IDs")

    arc_ids = [
        group["arc_id"].astype(int).to_numpy()
        for group in candidate_groups
    ]

    fractions = calculate_arc_positions(
        projected_nodes,
        projected_track,
        arcs,
        arc_ids,
        point_ids
    )

    graph = build_road_graph(projected_nodes, arcs)

    distance_cache = {}

    def shortest_distance(source, destination):
        if source not in distance_cache:
            distance_cache[source] = nx.single_source_dijkstra_path_length(
                graph,
                source,
                weight="weight"
            )

        return distance_cache[source].get(destination, np.inf)

    transition_matrices = []

    for t in range(len(candidate_groups) - 1):

        previous_arcs = arc_ids[t]
        next_arcs = arc_ids[t + 1]

        matrix = np.full(
            (len(previous_arcs), len(next_arcs)),
            -np.inf,
            dtype=float
        )

        gps_distance = float(np.linalg.norm(
            projected_track[point_ids[t + 1]] -
            projected_track[point_ids[t]]
        ))

        for i, previous_arc_id in enumerate(previous_arcs):

            previous_source, previous_destination = arcs[previous_arc_id]

            previous_length = float(np.linalg.norm(
                projected_nodes[previous_destination] -
                projected_nodes[previous_source]
            ))

            for j, next_arc_id in enumerate(next_arcs):

                next_source, next_destination = arcs[next_arc_id]

                next_length = float(np.linalg.norm(
                    projected_nodes[next_destination] -
                    projected_nodes[next_source]
                ))

                f1 = fractions[t][i]
                f2 = fractions[t + 1][j]

                if previous_arc_id == next_arc_id and f2 >= f1:
                    network_distance = (f2 - f1) * previous_length
                else:
                    remaining_previous = (1.0 - f1) * previous_length
                    traveled_next = f2 * next_length

                    middle_distance = shortest_distance(
                        int(previous_destination),
                        int(next_source)
                    )

                    if not np.isfinite(middle_distance):
                        continue

                    network_distance = (
                        remaining_previous
                        + middle_distance
                        + traveled_next
                    )

                matrix[i, j] = (
                    -abs(network_distance - gps_distance) / beta
                )

        transition_matrices.append(matrix)

    return transition_matrices
