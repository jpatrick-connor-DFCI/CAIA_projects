"""Synthetic integration fixture; never a substitute for cohort results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_demo_inputs(root, landmarks=(0,), seed=42, n=240):
    root = Path(root)
    rng = np.random.default_rng(seed)
    age = rng.normal(68, 7, n)
    psa = np.exp(rng.normal(2, 1, n))
    hemoglobin = rng.normal(12.5, 1.7, n)
    split = np.array(["train"] * int(n * .64) + ["valid"] * int(n * .16)
                     + ["test"] * (n - int(n * .64) - int(n * .16)))
    rng.shuffle(split)
    censor = rng.uniform(1200, 1900, n)
    source = np.where(rng.random(n) < .9, "adjudicated", "not_run")
    for endpoint in ("platinum", "nepc"):
        folder = root / ("prediction_inputs_adt" + ("_nepc" if endpoint == "nepc" else ""))
        folder.mkdir(parents=True, exist_ok=True)
        scale = 1250 if endpoint == "platinum" else 1700
        risk = .35 * (age - 68) / 7 + .7 * (np.log(psa) - 2) - .3 * (hemoglobin - 12.5)
        event_time = rng.exponential(scale * np.exp(-risk))
        event = event_time <= censor
        if endpoint == "nepc":
            event &= source == "adjudicated"
        duration = np.where(event, event_time, censor)
        for landmark in landmarks:
            keep = duration > landmark
            frame = pd.DataFrame({
                "DFCI_MRN": [f"SYNTHETIC_{i:04d}" for i in range(n)],
                "AGE_AT_TREATMENTSTART": age,
                "PSA__last": psa,
                "PSA__mean": psa * .9,
                "PSA__n_observations": rng.integers(1, 8, n),
                "HEMOGLOBIN__last": hemoglobin,
                "HEMOGLOBIN__n_observations": rng.integers(1, 8, n),
                f"t_{endpoint}": duration - landmark,
                endpoint.upper(): event.astype(int),
                "split": split,
                "NEPC_LABEL_SOURCE": source,
            }).loc[keep].copy()
            frame.loc[frame.index[::11], "PSA__last"] = np.nan
            frame.to_csv(folder / f"aggregated_landmark{landmark}.csv", index=False)
        manifest = {
            "data": "SYNTHETIC/longitudinal_prediction_data_adt.csv",
            "synthetic": True, "arm": "adt", "cohort": "all", "exclusion": "none",
            "anchor_col": "none", "endpoint": endpoint,
            "restrict_to_mrns": None, "exclude_mrns": None,
            "landmark_days": list(landmarks), "split_mode": "shared",
            "auc_timeline_schema_version": 3, "auc_max_time_units": 522,
            "auc_time_unit_days": 7, "time_unit_days": 7,
            "auc_horizons_by_landmark": {str(x): {endpoint: [52, 104, 156]} for x in landmarks},
            "min_patient_coverage": .2,
        }
        (folder / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    return root
