"""PathSnap candidate scorer. Load once at API startup, score one batch per trip."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


class CandidateScorer:
    def __init__(self, model_dir=None):
        root = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent
        spec = json.loads((root / "inference.json").read_text())
        self.columns = spec["columns"]
        self.model = joblib.load(root / spec["model_file"])
        if list(self.model.feature_names_in_) != self.columns:
            raise ValueError("Feature contract differs from saved model")
        self.positive = list(self.model.classes_).index(spec["positive_class"])

    def predict(self, candidates):
        """Accept a DataFrame or list of dicts. Return JSON-safe candidate scores."""
        if len(candidates) == 0:
            return []
        frame = pd.DataFrame(candidates)
        keys = ["trip_id", "point_id", "arc_id"]
        missing = set(keys + self.columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Candidate identifiers must be present and unique per row")
        features = frame[self.columns].apply(pd.to_numeric, errors="raise").astype(float)
        if np.isinf(features.to_numpy()).any():
            raise ValueError("Use null for missing features")
        out = frame[keys].copy()
        out["score"] = self.model.predict_proba(features)[:, self.positive]
        return out.to_dict(orient="records")
