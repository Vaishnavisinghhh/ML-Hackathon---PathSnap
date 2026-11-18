# Vidhwath’s PathSnap API handoff

**Ready for integration.** Load the classifier on the FastAPI laptop. Streamlit can send the trip ID and display the decoded geometry returned by that API.

1. **Artifact.** `tuned_model.joblib` is a serialized scikit-learn Pipeline. Keep it alongside `inference.json` and `vidhwath_inference.py`. Load it once per API worker.
2. **Selected algorithm.** Shallow MLPClassifier, nine inputs, one hidden layer of 24 ReLU units, one sigmoid output. Adam minimizes binary log loss. After ten TPE trials the selected L2 alpha was 0.00132929 and initial learning rate 0.00435262.
3. **Feature order.** `distance_m, bearing_diff, gps_speed, road_length_m, heading_consistency, point_time_delta, speed_valid, heading_valid, time_gap`. Use the team's original feature generator for these quantities. Send the additional keys `trip_id` (string preserving leading zeros), `point_id`, and `arc_id` (trip-local integers). Use JSON booleans for the three flags and null for missing numeric features.
4. **Preprocessing.** The fitted median imputer and StandardScaler are the first two pipeline steps. Supply the original, unscaled features. The pipeline applies its stored training medians, means and scales on every call.
5. **Python interface.** With this directory on the Python import path, initialize `scorer = CandidateScorer()` from `vidhwath_inference`. Call `scorer.predict(candidates)`, where candidates is a list of candidate dictionaries or a pandas DataFrame. Each input row needs the three keys and nine features. No route label is required.
6. **Output and map integration.** One output dictionary per input row contains trip_id, point_id, arc_id and score (the positive-class probability estimate). Preserve all candidates when handing scores to the decoder. Within-point normalization, when required by the decoder, divides each score by the sum for that point. The decoder picks an ordered arc sequence using the graph. On the server, resolve each arc ID to its directed (from_node, to_node) row in the trip’s .arcs, then look up [longitude, latitude] in .nodes. Return continuous paths as GeoJSON LineString or split disconnected parts into a MultiLineString. A suggested API response has trip_id, predicted_arc_ids, geometry and metrics. This response belongs to the API/decoder integration. The classifier alone returns candidate scores.
7. **Measured results.** The model was selected on grouped training CV macro F1 (0.8132). On 10 student trips its first-choice arc matched the order-aligned reference at 14,943 of 19,160 fixes (77.99%). Candidate macro F1 was 0.8162 on covered fixes. The decision tree attained 78.60% top-1. These figures precede HMM decoding. Labels were inferred from ordered routes and GPS geometry.
8. **Runtime.** Training/verification used Python 3.14.7. Install the pinned packages with `python -m pip install -r requirements-inference.txt`. FastAPI and Streamlit are separate application dependencies.
9. **Runnable example.** The example files are a real candidate group from a student trip. Run the following from the handoff directory to compare every returned identifier and score with the saved output.

```python
import json
from pathlib import Path
from math import isclose
from vidhwath_inference import CandidateScorer

rows = json.loads(Path("sample_input.json").read_text())
expected = json.loads(Path("sample_output.json").read_text())
predictions = CandidateScorer().predict(rows)
for row, target in zip(predictions, expected, strict=True):
    assert all(row[k] == target[k] for k in ("trip_id", "point_id", "arc_id"))
    assert isclose(row["score"], target["score"], abs_tol=1e-7)
print(predictions)
```
