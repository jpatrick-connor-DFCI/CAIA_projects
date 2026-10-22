"""Score exported JSON bundles without refitting any training component."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluation import ProbabilityCalibrator
from .features import FeatureBinarizer
from .optimizer import ChecklistModel


def predict_bundle(bundle_path, frame):
    bundle = json.loads(Path(bundle_path).read_text())
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported predictive checklist bundle schema")
    features = FeatureBinarizer.from_dict(bundle["binarizer"])
    model = ChecklistModel.from_dict(bundle["model"])
    calibration = ProbabilityCalibrator.from_dict(bundle["calibrator"])
    matrix = features.transform(frame)
    score = model.predict_score(matrix)
    result = pd.DataFrame({"score": score, "flag": model.predict(matrix),
        "probability": calibration.predict(score / max(len(model.selected_indices), 1))}, index=frame.index)
    id_col = bundle["config"]["id_col"]
    if id_col in frame:
        result.insert(0, id_col, frame[id_col])
    return result
