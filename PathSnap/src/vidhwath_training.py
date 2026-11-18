"""Train Vidhwath's models on the team's original candidate exports.

Run from PathSnap: .venv/bin/python -m src.vidhwath_training
"""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from . import model_training as ml
from .evaluation import evaluate_candidate_table, predict_candidate_scores

RAW = Path("../../Student folder")
DATA = Path("data/member1")
OUT = Path("reports/vidhwath_models")
FEATURES = ["distance_m", "bearing_diff", "gps_speed", "road_length_m", "heading_consistency",
            "point_time_delta", "speed_valid", "heading_valid", "time_gap"]


def align(track, nodes, arcs, route):
    """Minimum total GPS-to-segment distance along nondecreasing reference-route positions."""
    scale = np.array([np.cos(np.deg2rad(np.median(track[:, 1]))), 1.]) * 111195.
    pos = (track[:, :2] - track[0, :2]) * scale
    xy = (nodes - track[0, :2]) * scale
    a, b = xy[arcs[route]].transpose(1, 0, 2)
    edge = b - a
    length_sq = np.maximum(np.sum(edge**2, axis=1), 1e-12)
    cost = np.zeros(len(route))
    previous = np.empty((len(track), len(route)), dtype=np.int32)
    order = np.arange(len(route))
    for i, fix in enumerate(pos):
        delta = fix - a
        frac = np.clip(np.sum(delta * edge, axis=1) / length_sq, 0., 1.)
        dist = np.linalg.norm(delta - frac[:, None] * edge, axis=1)
        prefix = np.minimum.accumulate(cost)
        previous[i] = np.maximum.accumulate(np.where(cost == prefix, order, 0))
        cost = prefix + dist
    j = int(np.argmin(cost))
    ids = np.empty(len(track), dtype=np.int64)
    for i in range(len(track)-1, -1, -1):
        ids[i] = route[j]
        j = previous[i,j]
    return ids


def load_split(split):
    frame = pd.read_csv(DATA / f"{split}_features.csv", dtype={"trip_id":str})
    frame.rename(columns={"point_id":"observation_id","arc_id":"candidate_id"}, inplace=True)
    assert not frame.duplicated(["trip_id","observation_id","candidate_id"]).any()
    assert set(frame.trip_id) == {d.name for d in (RAW/split).iterdir() if d.is_dir()}
    frame["label"] = np.int8(0)
    frame["from_node"] = frame["to_node"] = 0
    audit = []
    for trip, group in frame.groupby("trip_id",sort=False):
        base = RAW / split / trip / trip
        track, nodes = [np.loadtxt(base.with_suffix(ext)) for ext in (".track",".nodes")]
        arcs, route = [np.loadtxt(base.with_suffix(ext),dtype=int) for ext in (".arcs",".route")]
        points = group.observation_id.to_numpy()
        ids = group.candidate_id.to_numpy()
        assert points.min() >= 0 and points.max() < len(track)
        assert ids.min() >= 0 and ids.max() < len(arcs) and route.min() >= 0 and route.max() < len(arcs)
        truth = align(track,nodes,arcs,route)
        label = ids == truth[points]
        frame.loc[group.index,"label"] = label.astype(np.int8)
        frame.loc[group.index,["from_node","to_node"]] = arcs[ids]
        audit.append(dict(trip=trip,split=split,points=len(track),with_candidates=group.observation_id.nunique(),
                          aligned_present=int(label.sum()),candidate_rows=len(group)))
    frame["time_gap_s"] = frame.point_time_delta
    return frame, pd.DataFrame(audit)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    models = OUT / "baseline_models"
    models.mkdir(exist_ok=True)
    train,a = load_split("train")
    test,b = load_split("student_test")
    assert set(train.trip_id).isdisjoint(test.trip_id)
    pd.concat([a,b]).to_csv(OUT/"coverage.csv", index=False)
    # Limit influence of very long journeys during model selection. All candidates
    # from each sampled fix stay together, and all 80 trips enter grouped CV.
    chosen = []
    for _,group in train.groupby("trip_id"):
        ids = np.sort(group.observation_id.unique())
        selected = ids[np.linspace(0,len(ids)-1,min(320,len(ids)),dtype=int)]
        chosen.append(group[group.observation_id.isin(selected)])
    dev = pd.concat(chosen,ignore_index=True)
    fit = ml._training_subset(train)
    print(f"Full-data fit {len(fit)} candidate rows; grouped CV {len(dev)} rows",flush=True)
    rows = []
    def score(name, fitted, cv, fit_seconds):
        start = time.perf_counter()
        scores = predict_candidate_scores(fitted,test[FEATURES])
        seconds = time.perf_counter()-start
        metrics = evaluate_candidate_table(test,scores)
        correct = round(metrics["covered_observations"] * metrics["top1_accuracy"])
        return dict(name=name,**cv,**metrics,top1_correct=correct,top1_all_fixes=correct/int(b.points.sum()),
                    fit_seconds=fit_seconds,inference_seconds=seconds)
    for name,(_,base) in ml.model_zoo().items():
        cv = ml.cross_validate(base,dev,FEATURES)
        start = time.perf_counter()
        fitted = clone(base).fit(fit[FEATURES],fit.label)
        duration = time.perf_counter()-start
        rows.append(score(name,fitted,cv,duration))
        joblib.dump(fitted,models/f"{name}.joblib",compress=2)
        pd.DataFrame(rows).to_csv(OUT/"baseline_comparison.csv",index=False)
        print(name,cv["cv_candidate_macro_f1_mean"],flush=True)
    best = max(rows,key=lambda r:r["cv_candidate_macro_f1_mean"])
    name = best["name"]
    tuned,study,trials = ml.bayesian_tune(name,ml.model_zoo()[name][1],dev,FEATURES,trials=10)
    start = time.perf_counter()
    tuned.fit(fit[FEATURES],fit.label)
    row = score(name,tuned,{"best_cv_macro_f1":study.best_value},time.perf_counter()-start)
    trials.to_csv(OUT/"bayesian_trials.csv",index=False)
    joblib.dump(tuned,OUT/"tuned_model.joblib",compress=2)
    print(row,flush=True)
    (OUT/"result.json").write_text(json.dumps(dict(tuned=row,parameters=study.best_params,
        features=FEATURES, cv_fixes=dev.groupby(["trip_id","observation_id"]).ngroups,
        full_fit_rows=len(fit),train=a.sum(numeric_only=True).to_dict(),student=b.sum(numeric_only=True).to_dict()),indent=2))

if __name__=="__main__":
    main()
