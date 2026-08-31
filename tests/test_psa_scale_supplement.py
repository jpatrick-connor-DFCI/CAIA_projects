from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from COMPASS.survival_analysis import compass_pipeline as cp


def test_psa_scale_supplement_reaggregates_matched_measurements():
    labs = pd.DataFrame(
        {
            "DFCI_MRN": [1, 1, 2, 2, 2, 3],
            "LAB_NAME": ["PSA", "PSA", "PSA", "PSA", "PSA", "Testosterone"],
            "LAB_VALUE": [0.0, 3.0, 1.0, 3.0, -1.0, 200.0],
            "t_lab": [-20, -10, -20, -10, -5, -10],
        }
    )
    ctx = SimpleNamespace(
        pre_treatment_lab_df=labs,
        univariate_data=pd.DataFrame(
            {"AGE": [60.0, 70.0, 80.0]}, index=pd.Index([1, 2, 3], name="DFCI_MRN")
        ),
    )

    data, feature_cols = cp.build_psa_scale_supplement_data(ctx, landmark_day=0)

    assert set(feature_cols) == {
        f"{lab}__{stat}"
        for lab in (cp.PSA_RAW_LAB_NAME, cp.PSA_LOG_LAB_NAME)
        for stat in ("mean", "min", "max", "last", "delta")
    }
    assert data.loc[1, "PSA_raw__mean"] == 1.5
    assert data.loc[1, "PSA_raw__delta"] == 3.0
    assert np.isclose(data.loc[1, "PSA_log1p__mean"], np.log(2.0))
    assert np.isclose(data.loc[1, "PSA_log1p__delta"], np.log(4.0))
    assert data.loc[1, "PSA_raw__n_observations"] == 2
    assert data.loc[1, "PSA_log1p__n_observations"] == 2
    assert data.loc[2, "PSA_raw__n_observations"] == 2
    assert data.loc[3, feature_cols].isna().all()
