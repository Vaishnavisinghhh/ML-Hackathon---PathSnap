
import math


def emission_probability(distance, sigma=10.0):
    """
    Calculate how likely a GPS point belongs to a candidate road.

    Parameters:
        distance: Distance between GPS point and road (metres).
        sigma: Assumed GPS measurement error (metres).

    Returns:
        Gaussian probability density.
    """

    if distance < 0:
        raise ValueError("Distance cannot be negative")

    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    probability = (
        1 / (math.sqrt(2 * math.pi) * sigma)
    ) * math.exp(
        -(distance ** 2) / (2 * sigma ** 2)
    )

    return probability
