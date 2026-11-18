"""Regenerate the comparison figure for Vidhwath’s models and verify the saved classifier."""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .vidhwath_training import DATA, OUT, FEATURES

def main():
    result = json.loads((OUT/"result.json").read_text())
    baselines = pd.read_csv(OUT/"baseline_comparison.csv")
    assert len(baselines) == 8 and len(pd.read_csv(OUT/"bayesian_trials.csv")) == 10
    winner = baselines.loc[baselines.cv_candidate_macro_f1_mean.idxmax()]
    tuned = result["tuned"]
    use_tuned = tuned["best_cv_macro_f1"] >= winner.cv_candidate_macro_f1_mean
    model_file = "tuned_model.joblib" if use_tuned else f"baseline_models/{winner['name']}.joblib"
    model = joblib.load(OUT/model_file)
    table = pd.read_csv(DATA/"student_test_features.csv",dtype={"trip_id":str})
    table["score"] = model.predict_proba(table[FEATURES])[:,list(model.classes_).index(1)]
    assert np.isfinite(table.score).all()
    point_rows = table.groupby(["trip_id","point_id"],sort=False).score.idxmax()
    predictions = table.loc[point_rows,["trip_id","point_id","arc_id","score"]]
    assert len(predictions) == result["student"]["with_candidates"]
    predictions.to_csv(OUT/"student_predictions.csv",index=False)
    (OUT/"inference.json").write_text(json.dumps(dict(model_file=model_file,name=winner["name"],
        variant="tuned" if use_tuned else "baseline",columns=FEATURES,positive_class=1),indent=2))
    ordered = baselines.sort_values("cv_candidate_macro_f1_mean")
    fig,axes = plt.subplots(1,2,figsize=(11,5),sharey=True,layout="constrained")
    for ax,col,title in zip(axes,["cv_candidate_macro_f1_mean","top1_all_fixes"],["Training trips · grouped CV macro F1","Student trips · correct road per GPS fix"]):
        ax.barh(ordered.name,ordered[col],color="#246e70")
        ax.set(xlim=(0,1),title=title)
        ax.bar_label(ax.containers[0],fmt="%.3f",padding=4)
    fig.suptitle("PathSnap | Eight baseline classifiers on the team feature export")
    fig.savefig(OUT/"model_comparison.png",dpi=165)
    plt.close(fig)
    print("Verified saved model and exported",len(predictions),"point predictions")
if __name__=="__main__":
    main()
