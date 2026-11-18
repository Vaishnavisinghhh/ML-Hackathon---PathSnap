# Member 2: model choices and evaluation

## The eight models

We chose eight approaches so the comparison tests different assumptions. There is no model we can honestly call “best” before running the trip-level validation.

| Model | In plain language | Why try it for PathSnap | Main limitation |
|---|---|---|---|
| Logistic Regression | Learns a weighted, mostly linear score for whether a candidate road matches a GPS fix. | Fast, interpretable baseline; gives probabilities that can rank candidates and feed HMM emissions. | A linear boundary may miss interactions such as “small distance is good only when direction also agrees.” |
| Distance-weighted KNN | Looks at similar labeled candidate rows and lets closer examples vote more strongly. | Can learn local, non-linear patterns without assuming a straight-line decision boundary. | Sensitive to feature scaling and local data density; prediction can be expensive. |
| Gaussian Naive Bayes | Estimates match/non-match probabilities from each feature separately, then combines them. | Very fast probabilistic baseline; tests whether simple feature distributions already separate good and bad road candidates. | Assumes features are conditionally independent, which is not fully true for distance, rank, and projection features. |
| Random Forest | Averages many decision trees trained on varied samples of the data. | Captures non-linear thresholds and feature interactions; usually handles mixed-scale numeric features well. | Larger and slower than a linear model; probabilities can be less calibrated. |
| HistGradientBoosting | Builds small trees one after another, with each tree correcting errors left by earlier trees. | Captures complex feature interactions efficiently on larger candidate tables. | Needs tuning and can overfit if trees are too deep or too numerous. |
| Shrinkage LDA | Learns how features vary together within each class and uses a linear score. | Checks whether accounting for correlated distance and direction features helps. | Assumes both classes have a common covariance matrix. |
| Decision Tree | Splits candidates with a small set of learned feature thresholds. | Provides inspectable rules and a comparison to tree ensembles. | A single tree can be sensitive to the training trips. |
| Shallow MLP | Uses one hidden layer to combine the features before predicting match probability. | Learns nonlinear combinations of distance, direction and motion. | Its fit depends on scaling and regularization. |

The models are not assumed to outperform one another. We select the final candidate classifier by **mean GroupKFold candidate macro F1**, then check top-1 observation accuracy, log loss, candidate-miss rate, route continuity where graph endpoints are available, and inference time. We use the student-test partition for a separate development comparison, not to choose the hyperparameters.

## Validation and tuning

- Split by `trip_id` with 5-fold `GroupKFold`. Every candidate row from one trip stays in the same fold; this avoids inflating scores by putting nearly identical points from the same trip in both training and validation.
- For training efficiency, keep each positive candidate and up to four negative candidates per covered GPS observation. Candidate misses are left out of classifier fitting and reported as a separate miss rate.
- Run 10 Optuna TPE (Bayesian) trials on the CV winner. The objective is mean candidate macro F1 across the same trip-grouped folds. Record all trial parameters, scores, and fold diagnostics.
- Refit the tuned model on all covered training trips and save the model, feature list, comparison CSV, tuning log, and plots.

## Completed team run

The final comparison uses Member 1’s nine features and the original ordered reference routes. See [the reproducible run and metric definitions](reports/member2_handoff/README.md) and [the model comparison](reports/member2_handoff/baseline_comparison.csv). Sampling and missing-candidate counts for this run are recorded there. The 80 training trips and 10 student-test trips are disjoint.

For the decoder, extract [the classifier ZIP](reports/member2_handoff/member3_classifier.zip). With the dependencies installed, run `python score_candidates.py input.csv scores.csv` from the extracted folder. Keep the per-candidate scores and the trip, point and arc keys for sequence decoding.
