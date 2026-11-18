
import math


def transition_probability(
    network_distance,
    gps_distance,
    beta=50.0
):
    """
    Calculate how likely movement between two candidate
    roads is based on road-network distance and GPS distance.

    Parameters:
        network_distance: Shortest distance through the road
                          network (metres).
        gps_distance: Straight-line distance between consecutive
                      GPS observations (metres).
        beta: Controls tolerance for distance differences.

    Returns:
        Transition probability score between 0 and 1.
    """

    if gps_distance < 0:
        raise ValueError("GPS distance cannot be negative")

    if beta <= 0:
        raise ValueError("Beta must be positive")

    # No connected route between the candidate roads
    if network_distance is None or math.isinf(network_distance):
        return 0.0

    if network_distance < 0:
        raise ValueError("Network distance cannot be negative")

    difference = abs(network_distance - gps_distance)

    probability = math.exp(-difference / beta)

    return probability
