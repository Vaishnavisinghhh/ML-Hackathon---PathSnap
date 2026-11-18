
import math
import networkx as nx
import numpy as np


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two geographic coordinates
    using the Haversine formula.

    Returns distance in metres.
    """

    earth_radius = 6_371_000

    lon1, lat1, lon2, lat2 = map(
        math.radians,
        [lon1, lat1, lon2, lat2]
    )

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    return (
        2
        * earth_radius
        * math.asin(math.sqrt(a))
    )


def calculate_bearing(lon1, lat1, lon2, lat2):
    """
    Calculate the initial compass bearing from one node
    to another.

    Returns a value between 0 and 360 degrees.
    """

    lon1, lat1, lon2, lat2 = map(
        math.radians,
        [lon1, lat1, lon2, lat2]
    )

    delta_lon = lon2 - lon1

    x = (
        math.sin(delta_lon)
        * math.cos(lat2)
    )

    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1)
        * math.cos(lat2)
        * math.cos(delta_lon)
    )

    bearing = math.degrees(math.atan2(x, y))

    return (bearing + 360) % 360


def build_road_graph(nodes, arcs):
    """
    Build a directed road network from node and arc arrays.

    nodes:
        NumPy array containing [longitude, latitude].

    arcs:
        NumPy array containing [start_node_id, end_node_id].

    Returns:
        A NetworkX MultiDiGraph.

    Original arc IDs are preserved as edge keys.
    """

    nodes = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs, dtype=int)

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(
            "nodes must have exactly two columns: longitude, latitude"
        )

    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError(
            "arcs must have exactly two columns: start_node, end_node"
        )

    if not np.isfinite(nodes).all():
        raise ValueError("Node coordinates contain invalid values")

    if arcs.size:
        if arcs.min() < 0 or arcs.max() >= len(nodes):
            raise ValueError(
                "An arc references a node that does not exist"
            )

    graph = nx.MultiDiGraph()

    # Add every road node, preserving its original node ID.
    for node_id, (lon, lat) in enumerate(nodes):
        graph.add_node(
            int(node_id),
            lon=float(lon),
            lat=float(lat)
        )

    # Add every directed road arc.
    for arc_id, (start_node, end_node) in enumerate(arcs):

        start_lon, start_lat = nodes[start_node]
        end_lon, end_lat = nodes[end_node]

        length_m = haversine_distance(
            start_lon,
            start_lat,
            end_lon,
            end_lat
        )

        bearing_deg = calculate_bearing(
            start_lon,
            start_lat,
            end_lon,
            end_lat
        )

        graph.add_edge(
            int(start_node),
            int(end_node),
            key=int(arc_id),

            arc_id=int(arc_id),

            length_m=float(length_m),
            bearing_deg=float(bearing_deg),

            start_lon=float(start_lon),
            start_lat=float(start_lat),

            end_lon=float(end_lon),
            end_lat=float(end_lat)
        )

    graph.graph["node_count"] = len(nodes)
    graph.graph["arc_count"] = len(arcs)

    return graph
