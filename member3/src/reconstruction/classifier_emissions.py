
import json
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "member3_classifier"

MODEL_PATH = MODEL_DIR / "tuned_model.joblib"
CONFIG_PATH = MODEL_DIR / "inference.json"


@lru_cache(maxsize=1)
def load_classifier():
    """
    Load and cache Member 2's trained classifier and configuration.

    The model is loaded only once per Python process.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        spec = json.load(file)

    model = joblib.load(MODEL_PATH)

    return model, spec


def calculate_classifier_emissions(candidates):
    """
    Calculate classifier-based emission scores for road candidates.

    Each row represents one candidate road for one GPS observation.

    Returns:
        DataFrame containing original candidate information,
        emission_probability and log_emission.
    """

    if not isinstance(candidates, pd.DataFrame):
        raise TypeError(
            "Candidates must be a pandas DataFrame"
        )

    # Reuse the cached model and configuration.
    model, spec = load_classifier()

    feature_columns = spec["columns"]
    positive_class = spec["positive_class"]

    missing_columns = [
        column for column in feature_columns
        if column not in candidates.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    result = candidates.copy()

    X = candidates[feature_columns]

    class_index = list(model.classes_).index(positive_class)

    probabilities = model.predict_proba(X)[:, class_index]

    log_probabilities = np.log(
        np.clip(probabilities, 1e-12, 1.0)
    )

    result["emission_probability"] = probabilities
    result["log_emission"] = log_probabilities

    return result
