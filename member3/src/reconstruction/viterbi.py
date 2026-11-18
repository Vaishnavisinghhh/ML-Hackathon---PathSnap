
import numpy as np


def viterbi(log_emissions, log_transitions):
    """
    Find the most likely sequence of hidden states.

    log_emissions:
        A list of arrays.
        Each array contains emission log-scores for one GPS point.
        Different observations may have different candidate counts.

    log_transitions:
        A list of matrices.
        Matrix t has shape:
        (candidates at time t, candidates at time t+1)

    Returns:
        List of selected candidate indices.
    """

    # Convert emissions into a list of arrays.
    if isinstance(log_emissions, np.ndarray):
        if log_emissions.ndim != 2:
            raise ValueError("Emissions must be a 2D array")

        emissions = [
            np.asarray(row, dtype=float)
            for row in log_emissions
        ]
    else:
        emissions = [
            np.asarray(row, dtype=float)
            for row in log_emissions
        ]

    if len(emissions) == 0:
        raise ValueError("Emissions cannot be empty")

    for row in emissions:
        if row.ndim != 1 or len(row) == 0:
            raise ValueError("Each observation needs candidates")

        if np.any(np.isnan(row)) or np.any(np.isposinf(row)):
            raise ValueError("Invalid emission log-score")

    time_steps = len(emissions)

    # Convert transition matrices into a list.
    if isinstance(log_transitions, np.ndarray):
        if time_steps == 1 and log_transitions.size == 0:
            transitions = []
        elif log_transitions.ndim == 3:
            transitions = [
                log_transitions[t]
                for t in range(len(log_transitions))
            ]
        else:
            raise ValueError("Transitions must be a 3D array")
    else:
        transitions = [
            np.asarray(matrix, dtype=float)
            for matrix in log_transitions
        ]

    if len(transitions) != time_steps - 1:
        raise ValueError(
            "There must be one transition matrix "
            "between each pair of observations"
        )

    # Validate transition matrix dimensions.
    for t, matrix in enumerate(transitions):

        expected_shape = (
            len(emissions[t]),
            len(emissions[t + 1])
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                f"Transition matrix {t} should have shape "
                f"{expected_shape}, got {matrix.shape}"
            )

        if np.any(np.isnan(matrix)) or np.any(np.isposinf(matrix)):
            raise ValueError("Invalid transition log-score")

    # Initial state probabilities: equal prior for each candidate.
    scores = [
        emissions[0] - np.log(len(emissions[0]))
    ]

    backpointers = [None]

    # Dynamic programming.
    for t in range(1, time_steps):

        previous_scores = scores[t - 1]
        transition_matrix = transitions[t - 1]

        candidate_scores = (
            previous_scores[:, None] + transition_matrix
        )

        best_previous = np.argmax(
            candidate_scores,
            axis=0
        )

        best_scores = (
            np.max(candidate_scores, axis=0)
            + emissions[t]
        )

        scores.append(best_scores)
        backpointers.append(best_previous)

    # Check whether any valid complete path exists.
    if not np.any(np.isfinite(scores[-1])):
        raise ValueError("No valid state sequence exists")

    # Backtracking.
    best_path = [0] * time_steps

    best_path[-1] = int(np.argmax(scores[-1]))

    for t in range(time_steps - 1, 0, -1):

        best_path[t - 1] = int(
            backpointers[t][best_path[t]]
        )

    return best_path
