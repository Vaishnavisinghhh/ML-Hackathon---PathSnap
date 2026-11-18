"""Metrics for candidate-road classification and per-GPS-point ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def predict_candidate_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(candidate is the aligned reference arc), robust to class ordering."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("PathSnap candidate models must provide predict_proba() for ranking/HMM emissions")
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def evaluate_candidate_table(table: pd.DataFrame, scores, *, threshold: float = 0.5) -> dict:
    """Evaluate candidate classification and the top-ranked arc for each GPS fix.

    Rows without any positive candidate are candidate misses; they contribute to
    miss-rate/coverage, but are excluded from conditional ranking accuracy and
    log-loss because no candidate in the set can be correct.
    """
    required = {"trip_id", "observation_id", "label"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Evaluation table missing columns: {sorted(missing)}")
    y = pd.to_numeric(table.label, errors="coerce").fillna(0).to_numpy(dtype=int)
    p = np.clip(np.asarray(scores, dtype=float).reshape(-1), 1e-7, 1 - 1e-7)
    if len(y) != len(p):
        raise ValueError("One candidate score is required for every candidate row")
    group_keys = [table.trip_id.astype(str), table.observation_id.astype(str)]
    work = pd.DataFrame({"trip_id": group_keys[0].to_numpy(),
                         "observation_id": group_keys[1].to_numpy(),
                         "label": y, "score": p}, index=table.index)
    if "candidate_miss" in table:
        work["marked_miss"] = pd.to_numeric(table.candidate_miss, errors="coerce").fillna(0).to_numpy(dtype=int)
    else:
        work["marked_miss"] = 0
    miss_mask = work.groupby(["trip_id", "observation_id"], sort=False).label.transform("sum") == 0
    miss_mask |= work.marked_miss.astype(bool)
    valid = work.loc[~miss_mask].copy()
    misses = work.loc[miss_mask, ["trip_id", "observation_id"]].drop_duplicates()
    by_fix = valid.groupby(["trip_id", "observation_id"], sort=False)
    point_predictions = by_fix.score.idxmax().to_numpy()
    point_correct = valid.loc[point_predictions, "label"].to_numpy()
    reference = valid.loc[valid.label.eq(1)].set_index(["trip_id", "observation_id"])
    point_nll = -np.log((reference.score / by_fix.score.sum()).clip(lower=1e-12)).to_numpy()
    if valid.empty:
        candidate_macro_f1 = candidate_precision = candidate_recall = float("nan")
    else:
        pred = (valid.score.to_numpy() >= threshold).astype(int)
        candidate_macro_f1 = float(f1_score(valid.label, pred, average="macro", zero_division=0))
        candidate_precision = float(precision_score(valid.label, pred, zero_division=0))
        candidate_recall = float(recall_score(valid.label, pred, zero_division=0))
    total_points = int(work[["trip_id", "observation_id"]].drop_duplicates().shape[0])
    miss_points = int(misses.shape[0])
    continuity = float("nan")
    if {"candidate_id", "from_node", "to_node"}.issubset(table.columns):
        selected = table.loc[point_predictions, ["trip_id", "observation_id", "candidate_id", "from_node", "to_node"]].copy()
        if "time_gap_s" in table:
            selected["time_gap_s"] = pd.to_numeric(table.loc[point_predictions, "time_gap_s"], errors="coerce").to_numpy()
        connected, pairs = 0, 0
        for _, trip in selected.groupby("trip_id", sort=False):
            trip = trip.sort_values("observation_id", kind="stable")
            prior = None
            for row in trip.itertuples(index=False):
                if prior is not None and int(row.observation_id) == int(prior.observation_id) + 1 and (not hasattr(row, "time_gap_s") or pd.isna(row.time_gap_s) or row.time_gap_s <= 2.0):
                    pairs += 1
                    connected += int(prior.candidate_id == row.candidate_id or prior.to_node == row.from_node)
                prior = row
        continuity = connected / pairs if pairs else float("nan")
    return {
        "candidate_macro_f1": candidate_macro_f1,
        "candidate_precision": candidate_precision,
        "candidate_recall": candidate_recall,
        "top1_accuracy": float(np.mean(point_correct)) if len(point_correct) else float("nan"),
        "observation_log_loss": float(np.mean(point_nll)) if len(point_nll) else float("nan"),
        "candidate_miss_rate": miss_points / total_points if total_points else float("nan"),
        "candidate_coverage": 1.0 - miss_points / total_points if total_points else float("nan"),
        "classifier_only_route_continuity": continuity,
        "gps_observations": total_points,
        "covered_observations": len(point_correct),
    }


def route_continuity(predicted_arc_ids, arc_endpoints: dict[int, tuple[int, int]]) -> float:
    """Fraction of adjacent predicted arcs whose directed endpoints connect.

    Return NaN when endpoint metadata is unavailable or fewer than two valid
    consecutive arc predictions exist. This is a diagnostic, not label accuracy.
    """
    ids = [int(x) for x in predicted_arc_ids]
    pairs = [(a, b) for a, b in zip(ids, ids[1:])
             if a in arc_endpoints and b in arc_endpoints]
    if not pairs:
        return float("nan")
    return float(np.mean([a == b or arc_endpoints[a][1] == arc_endpoints[b][0] for a, b in pairs]))
