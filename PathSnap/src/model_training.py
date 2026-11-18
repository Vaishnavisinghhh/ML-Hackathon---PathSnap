"""Vidhwath’s training pipeline: compare eight candidate classifiers and tune the CV winner.

Expected CSV: one row per (trip_id, observation_id, candidate_id), with numeric
candidate features, binary `label`, and optional `candidate_miss`. Candidate
generation and reference-route alignment are provided by Member 1. IDs and labels
are metadata and are never passed to estimators as features.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .evaluation import evaluate_candidate_table, predict_candidate_scores
except ImportError:  # direct `python src/model_training.py` use
    from evaluation import evaluate_candidate_table, predict_candidate_scores


# Explicit predictor allowlist for the original comparison CLI. The handoff
# runner supplies its nine Member 1 columns directly to the same estimators.
def get_model_features():
    return ["distance_m", "log_distance_m", "candidate_rank", "projection_fraction",
            "segment_length_m", "course_error_deg", "course_error_sin", "course_error_cos",
            "gps_speed_capped_mps", "time_gap_s", "speed_jump_flag", "from_node_degree",
            "to_node_degree", "node_degree_sum"]


SEED = 42
REQUIRED = {"trip_id", "observation_id", "candidate_id", "label"}


def validate_table(frame: pd.DataFrame, name: str) -> list[str]:
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} contains no candidate rows")
    label = pd.to_numeric(frame.label, errors="coerce")
    if label.isna().any() or not set(label.unique()).issubset({0, 1}):
        raise ValueError(f"{name}.label must contain only 0/1 values")
    features = [name for name in get_model_features() if name in frame.columns]
    if not features:
        raise ValueError("No approved model features are present in the Member 1 candidate table")
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    invalid_text = numeric.isna() & frame[features].notna()
    if invalid_text.any().any():
        raise ValueError(f"{name} has non-numeric values in selected model features")
    if numeric.isna().all().any():
        raise ValueError("At least one selected feature is entirely missing/non-numeric")
    return features


def remove_candidate_misses(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude observations whose correct arc was not generated; report miss rate separately."""
    group = ["trip_id", "observation_id"]
    covered = frame.groupby(group, sort=False).label.transform("sum") > 0
    if "candidate_miss" in frame:
        covered &= pd.to_numeric(frame.candidate_miss, errors="coerce").fillna(0).eq(0)
    return frame.loc[covered].copy()


def _pipeline(model: Any, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def model_zoo(seed: int = SEED) -> dict[str, tuple[str, Pipeline]]:
    """Eight candidate classifiers with complementary decision rules."""
    return {
        "logistic_regression": ("linear probabilistic", _pipeline(
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                               random_state=seed), scale=True)),
        "distance_weighted_knn": ("instance-based", _pipeline(
            KNeighborsClassifier(n_neighbors=11, weights="distance", p=2,
                                 n_jobs=-1), scale=True)),
        "gaussian_naive_bayes": ("probabilistic generative", _pipeline(
            GaussianNB(var_smoothing=1e-8), scale=True)),
        "shrinkage_lda": ("linear discriminant analysis", _pipeline(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), scale=True)),
        "decision_tree": ("single decision tree", _pipeline(
            DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=seed))),
        "shallow_mlp": ("neural network", _pipeline(
            MLPClassifier(hidden_layer_sizes=(24,), max_iter=160, alpha=0.01,
                          tol=0.001, random_state=seed), scale=True)),
        "random_forest": ("bagged trees", _pipeline(
            RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=3,
                                   max_features="sqrt", class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=seed))),
        "hist_gradient_boosting": ("boosting", _pipeline(
            HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08,
                                           max_leaf_nodes=15, min_samples_leaf=25,
                                           l2_regularization=1.0, class_weight="balanced",
                                           random_state=seed))),
    }


def _training_subset(frame: pd.DataFrame, max_negatives_per_point: int = 4,
                     seed: int = SEED) -> pd.DataFrame:
    """Keep all positives and nearest hard negatives to make experiments tractable."""
    covered = remove_candidate_misses(frame)
    positives = covered.loc[covered.label.eq(1)]
    negatives = covered.loc[covered.label.eq(0)]
    if "candidate_rank" in negatives:
        negatives = negatives.sort_values("candidate_rank", kind="stable")
    else:
        negatives = negatives.sample(frac=1, random_state=seed)
    negatives = negatives.groupby(["trip_id", "observation_id"], sort=False).head(max_negatives_per_point)
    return pd.concat([positives, negatives]).sort_index()


def cross_validate(model: Pipeline, frame: pd.DataFrame, features: list[str],
                   folds: int = 5, seed: int = SEED) -> dict[str, float]:
    groups = frame.trip_id.astype(str).to_numpy()
    group_count = len(np.unique(groups))
    if group_count < folds:
        raise ValueError(f"GroupKFold({folds}) needs at least {folds} trips; found {group_count}")
    splitter = GroupKFold(n_splits=folds)
    fold_rows = []
    for fold_no, (train_idx, valid_idx) in enumerate(splitter.split(frame, frame.label, groups), 1):
        train = remove_candidate_misses(frame.iloc[train_idx])
        train = _training_subset(train, seed=seed + fold_no)
        valid = frame.iloc[valid_idx]
        if train.label.nunique() != 2:
            raise ValueError(f"Fold {fold_no} training split has one label class")
        fitted = clone(model)
        fitted.fit(train[features], train.label.astype(int))
        scores = predict_candidate_scores(fitted, valid[features])
        fold_rows.append(evaluate_candidate_table(valid, scores))
    return {
        "cv_candidate_macro_f1_mean": float(np.nanmean([x["candidate_macro_f1"] for x in fold_rows])),
        "cv_candidate_macro_f1_std": float(np.nanstd([x["candidate_macro_f1"] for x in fold_rows], ddof=1)),
        "cv_top1_accuracy_mean": float(np.nanmean([x["top1_accuracy"] for x in fold_rows])),
        "cv_observation_log_loss_mean": float(np.nanmean([x["observation_log_loss"] for x in fold_rows])),
        "cv_candidate_miss_rate_mean": float(np.nanmean([x["candidate_miss_rate"] for x in fold_rows])),
        "cv_route_continuity_mean": float(np.nanmean([x["classifier_only_route_continuity"] for x in fold_rows])),
    }


def compare_models(train_table: pd.DataFrame, student_table: pd.DataFrame | None = None,
                   folds: int = 5, seed: int = SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = validate_table(train_table, "train_table")
    train_covered = remove_candidate_misses(train_table)
    if train_covered.trip_id.nunique() < folds:
        raise ValueError("Not enough covered training trips for trip-level GroupKFold")
    if student_table is not None:
        validate_table(student_table, "student_table")
    rows, fitted_models = [], {}
    for name, (family, model) in model_zoo(seed).items():
        started = time.perf_counter()
        cv_metrics = cross_validate(model, train_table, features, folds=folds, seed=seed)
        final_fit_rows = _training_subset(train_covered, seed=seed)
        fitted = clone(model).fit(final_fit_rows[features], final_fit_rows.label.astype(int))
        elapsed = time.perf_counter() - started
        student_metrics = None
        inference_seconds_per_1000 = float("nan")
        if student_table is not None:
            infer_started = time.perf_counter()
            student_scores = predict_candidate_scores(fitted, student_table[features])
            inference_duration = time.perf_counter() - infer_started
            student_metrics = evaluate_candidate_table(student_table, student_scores)
            inference_seconds_per_1000 = inference_duration * 1000 / max(len(student_table), 1)
        else:
            timing_rows = train_covered[features].head(10_000)
            infer_started = time.perf_counter()
            predict_candidate_scores(fitted, timing_rows)
            inference_seconds_per_1000 = (time.perf_counter() - infer_started) * 1000 / max(len(timing_rows), 1)
        rows.append({"name": name, "family": family, **cv_metrics,
                     "student_metrics": student_metrics, "fit_seconds_including_cv": elapsed,
                     "inference_seconds_per_1000_candidates": inference_seconds_per_1000})
        fitted_models[name] = fitted
    results = pd.DataFrame(rows).sort_values(
        "cv_candidate_macro_f1_mean", ascending=False).reset_index(drop=True)
    return results, {"features": features, "models": fitted_models,
                     "train_covered": train_covered}


def _suggest_params(trial, model_name: str) -> dict[str, Any]:
    if model_name == "logistic_regression":
        return {"model__C": trial.suggest_float("C", 0.03, 10.0, log=True)}
    if model_name == "distance_weighted_knn":
        return {"model__n_neighbors": trial.suggest_int("n_neighbors", 5, 25, step=2),
                "model__weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "model__p": trial.suggest_categorical("p", [1, 2])}
    if model_name == "gaussian_naive_bayes":
        return {"model__var_smoothing": trial.suggest_float("var_smoothing", 1e-11, 1e-6, log=True)}
    if model_name == "shrinkage_lda":
        return {"model__shrinkage": trial.suggest_float("shrinkage", 0.001, 1.0, log=True)}
    if model_name == "decision_tree":
        return {"model__max_depth": trial.suggest_int("max_depth", 3, 12),
                "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 8, 64)}
    if model_name == "shallow_mlp":
        return {"model__alpha": trial.suggest_float("alpha", 0.0001, 0.1, log=True),
                "model__learning_rate_init": trial.suggest_float("learning_rate_init", 0.0003, 0.005, log=True)}
    if model_name == "random_forest":
        return {"model__n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "model__max_depth": trial.suggest_categorical("max_depth", [10, 18, 26, None]),
                "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
                "model__max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8])}
    if model_name == "hist_gradient_boosting":
        return {"model__max_iter": trial.suggest_int("max_iter", 80, 240, step=40),
                "model__learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                "model__max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [7, 15, 23, 31]),
                "model__l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True)}
    raise ValueError(f"No tuning space for {model_name}")


def bayesian_tune(model_name: str, base_model: Pipeline, train_table: pd.DataFrame,
                  features: list[str], folds: int = 5, trials: int = 10,
                  seed: int = SEED):
    """Tune the CV-selected model with Optuna's Bayesian TPE sampler."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt, including optuna, to run Bayesian tuning") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(
                                    seed=seed, n_startup_trials=min(3, max(1, trials // 3))))
    def objective(trial):
        configured = clone(base_model).set_params(**_suggest_params(trial, model_name))
        metrics = cross_validate(configured, train_table, features, folds=folds, seed=seed)
        trial.set_user_attr("cv_candidate_macro_f1_std", metrics["cv_candidate_macro_f1_std"])
        trial.set_user_attr("cv_top1_accuracy_mean", metrics["cv_top1_accuracy_mean"])
        trial.set_user_attr("cv_observation_log_loss_mean", metrics["cv_observation_log_loss_mean"])
        return metrics["cv_candidate_macro_f1_mean"]
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_model = clone(base_model).set_params(**{
        # Optuna stores the unprefixed trial parameter names; rebuild pipeline keys.
        ("model__" + key): value for key, value in study.best_params.items()
    })
    final_rows = _training_subset(train_table, seed=seed)
    best_model.fit(final_rows[features], final_rows.label.astype(int))
    trial_rows = [{"trial": t.number + 1, "value_cv_candidate_macro_f1": t.value,
                   "params": t.params, **t.user_attrs} for t in study.trials]
    return best_model, study, pd.DataFrame(trial_rows)


def save_outputs(results: pd.DataFrame, trial_log: pd.DataFrame, final_model: Pipeline,
                 feature_columns: list[str], output_dir: str | Path,
                 tuned_comparison: dict | None = None) -> None:
    import joblib
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "model_comparison.csv", index=False)
    trial_log.to_csv(output / "bayesian_trials.csv", index=False)
    (output / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    if tuned_comparison is not None:
        (output / "tuned_vs_baseline.json").write_text(json.dumps(tuned_comparison, indent=2), encoding="utf-8")
    joblib.dump(final_model, output / "pathsnap_candidate_model.joblib")
    _plot_comparison(results, output)


def _plot_comparison(results: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt
    metrics = [("cv_candidate_macro_f1_mean", "GroupKFold candidate macro F1"),
               ("cv_top1_accuracy_mean", "GroupKFold observation top-1 accuracy"),
               ("cv_observation_log_loss_mean", "GroupKFold observation log loss")]
    for column, title in metrics:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(results.name, results[column], color="#207567")
        ax.set_title(title); ax.set_ylabel(column); ax.tick_params(axis="x", rotation=25)
        fig.tight_layout(); fig.savefig(output / f"{column}.png", dpi=160); plt.close(fig)
    if "student_metrics" in results and results.student_metrics.notna().any():
        for metric_key, title, filename in [
            ("top1_accuracy", "Student-test observation top-1 accuracy", "student_top1_accuracy.png"),
            ("candidate_macro_f1", "Student-test candidate macro F1", "student_candidate_macro_f1.png"),
            ("observation_log_loss", "Student-test observation log loss", "student_observation_log_loss.png"),
        ]:
            values = [m.get(metric_key, np.nan) if isinstance(m, dict) else np.nan
                      for m in results.student_metrics]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(results.name, values, color="#d1843d")
            ax.set_title(title); ax.set_ylabel(metric_key); ax.tick_params(axis="x", rotation=25)
            fig.tight_layout(); fig.savefig(output / filename, dpi=160); plt.close(fig)


def run_experiments(train_csv: str | Path, student_csv: str | Path | None,
                    output_dir: str | Path = "results", folds: int = 5,
                    bayes_trials: int = 10, seed: int = SEED):
    train = pd.read_csv(train_csv)
    student = pd.read_csv(student_csv) if student_csv else None
    results, bundle = compare_models(train, student, folds=folds, seed=seed)
    import joblib
    model_dir = Path(output_dir) / "baseline_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for model_name, estimator in bundle["models"].items():
        joblib.dump(estimator, model_dir / f"{model_name}.joblib", compress=3)
    winner_name = str(results.iloc[0]["name"])
    zoo = model_zoo(seed)
    winner_base = zoo[winner_name][1]
    tuned, study, trial_log = bayesian_tune(
        winner_name, winner_base, train, bundle["features"],
        folds=folds, trials=bayes_trials, seed=seed)
    tuned_comparison = None
    if student is not None:
        test_m = evaluate_candidate_table(
            student, predict_candidate_scores(tuned, student[bundle["features"]]))
        untuned = bundle["models"][winner_name]
        base_m = evaluate_candidate_table(
            student, predict_candidate_scores(untuned, student[bundle["features"]]))
        tuned_comparison = {"winner_by_training_group_cv": winner_name,
                            "untuned_student_metrics": base_m,
                            "tuned_student_metrics": test_m,
                            "best_params": study.best_params,
                            "best_cv_macro_f1": study.best_value,
                            "trials": len(study.trials)}
    save_outputs(results, trial_log, tuned, bundle["features"], output_dir, tuned_comparison)
    return results, study, tuned_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Member 1 labeled training candidate CSV")
    parser.add_argument("--student-test", help="Optional labeled student-test candidate CSV")
    parser.add_argument("--output", default="results")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bayes-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    comparison, study, tuned_comparison = run_experiments(
        args.train, args.student_test, args.output, args.folds, args.bayes_trials, args.seed)
    print(comparison[["name", "cv_candidate_macro_f1_mean", "cv_top1_accuracy_mean",
                      "cv_observation_log_loss_mean", "cv_candidate_miss_rate_mean"]].to_string(index=False))
    print("Bayesian tuning trials:", len(study.trials), "best training-CV score:", study.best_value)
    if tuned_comparison:
        print("Tuned vs. untuned student-test metrics saved under", args.output)
