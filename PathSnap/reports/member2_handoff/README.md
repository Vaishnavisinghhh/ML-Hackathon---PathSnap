# Team 2 · Candidate road classification

Use this run for the feature exports supplied by Member 1. The earlier pilot directories contain experiments with a different candidate set and alignment method.

## Training and scoring

The candidate tables contain one row for a GPS point and a possible directed road. The two exports contain nine input features, including the three quality flags. The saved pipeline imputes missing numeric values inside each training fold and applies scaling for models that use it. Trip, point and arc IDs identify records. Reference routes supply supervision.

Neither export contains point-level labels. For each trip, the adapter uses its original ordered .route file, projects the GPS points and reference arcs into local metres and uses dynamic programming to minimize the sum of point-to-segment distances over nondecreasing positions along the reference route. It can remain on an arc or advance multiple arcs between fixes. This constructs approximate labels independently of the candidate shortlist. Revisited locations, overlapping arcs and sparse fixes can lead to ambiguous assignments. A prediction is correct when its arc ID matches the aligned reference arc.

For the five validation folds, up to 320 evenly spaced represented GPS fixes per training trip limit the influence of the longest trips. All candidates for each selected fix remain together. All 80 trips participate. On each fitting split, the classifiers retain each covered fix's positive and up to four randomly selected negatives (fixed seed). Final fits use every covered training fix. The 10 student trips enter scoring only. The selection metric is mean fold macro F1 for the two candidate classes on covered fixes. Model tuning uses ten Optuna TPE trials on the same folds.

The comparison CSV reports classifier-only metrics. The top1_all_fixes column counts a missing candidate as an incorrect road and includes all original GPS fixes in its denominator. top1_accuracy and observation_log_loss condition on the reference being in the shortlist. Continuity compares consecutive covered fixes separated by at most two seconds and counts staying on the same arc or moving to a directed successor. HMM decoding is a separate team component.

## Reproduce from the PathSnap folder

Keep the original WhatsApp exports in data/member1 and the original dataset at ../../Student folder. Run these two commands with the project environment:

    .venv/bin/python -m src.member2_handoff
    .venv/bin/python -m src.member2_results

The first command saves the eight fitted baselines, baseline metrics, per-trip coverage, tuning trials and tuned classifier. The second selects the tuned variant if its training CV score meets or exceeds the baseline, loads the selected file, writes its nine-column input contract to inference.json and exports student_predictions.csv. The score column is the model's positive-class probability estimate. Normalize scores within a GPS fix if the decoder requires candidate probabilities summing to one.
