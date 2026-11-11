"""Synthetic Stage-2 ADT long frame for the local survival tree tests.

Low hemoglobin shortens time to platinum, NEPC and AVPC; PSA (log) adds a weaker signal;
albumin and age are noise. About 1/3 of patients have no PSA, to exercise item_false.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANCHOR = pd.Timestamp("2015-01-01")
MAPPING = {
    "Hemoglobin": "Hemoglobin [Mass/volume] in Blood",
    "PSA": "Prostate specific Ag [Mass/volume] in Serum or Plasma",
    "Albumin": "Albumin [Mass/volume] in Serum or Plasma",
}
HGB_FEATURE = "Hemoglobin__Mass_volume__in_Blood__mean"


def synthetic_local(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        hgb = float(rng.normal(12.5, 1.6))
        psa = float(np.exp(rng.normal(2.0, 1.2)))
        has_psa = rng.random() > 0.33
        alb = float(rng.normal(4.0, 0.4))
        age = float(rng.normal(68, 8))
        lin = 0.9 * (12.5 - hgb) + (0.3 * (np.log(psa) - 2.0) if has_psa else 0.0)
        follow = int(rng.uniform(400, 3600))

        def draw(scale):
            return int(rng.exponential(scale * np.exp(-lin))) + 1

        t_plat, t_nepc, t_avpc = draw(1500), draw(2500), draw(1800)
        base = {
            "DFCI_MRN": 50000 + pid, "AGE_AT_TREATMENTSTART": age,
            "DIAGNOSIS_DATE": ANCHOR - pd.Timedelta(days=200), "TREATMENT_ANCHOR_DATE": ANCHOR,
            "LAST_CONTACT_DATE": ANCHOR + pd.Timedelta(days=follow),
        }
        for name, t in (("PLATINUM", t_plat), ("NEPC", t_nepc), ("AVPC", t_avpc)):
            hit = t < follow
            base[name] = int(hit)
            base[f"{name}_DATE"] = (ANCHOR + pd.Timedelta(days=t if hit else follow)) if (hit or name == "PLATINUM") \
                else pd.NaT
        labs = [("Hemoglobin", hgb), ("Albumin", alb)] + ([("PSA", psa)] if has_psa else [])
        for name, value in labs:
            for day in (-60, -20):
                rows.append({**base, "LAB_DATE": ANCHOR + pd.Timedelta(days=day), "LAB_NAME": name,
                             "LAB_VALUE": value * float(rng.normal(1, 0.02))})
    return pd.DataFrame(rows)
