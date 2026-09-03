"""Regression guards for the retired joint AVPC_NEPC model endpoint.

The Stage-1 cohort may retain AVPC_NEPC fields as audit metadata because the
shared criteria-timeline source emits them. They must not be selectable by any
modeling registry or carried as a longitudinal cause.
"""

import pandas as pd

from COMPASS.data_preprocessing import build_prediction_inputs as bpi
from COMPASS.survival_analysis import compass_pipeline as cp
from COMPASS.survival_analysis import cox_aggregated as ca
from survival_common import cohort
from survival_common.longitudinal_targets import CONFIG_ENDPOINTS, LONGITUDINAL_CONFIGS


def test_joint_endpoint_absent_from_model_registries():
    assert "avpc_nepc" not in ca.ENDPOINTS
    assert "avpc_nepc" not in cp._LONGITUDINAL_CONFIGS_BY_ENDPOINT
    assert "avpc_nepc" not in LONGITUDINAL_CONFIGS
    assert "avpc_nepc_competing" not in LONGITUDINAL_CONFIGS
    assert "avpc_nepc" not in CONFIG_ENDPOINTS.values()


def test_joint_cause_absent_from_prediction_manifest():
    assert "AVPC_NEPC" not in bpi.LONGITUDINAL_OPTIONAL_EVENT_COLS
    frame = pd.DataFrame(
        columns=[
            "PLATINUM", "DEATH", "NEPC", "t_nepc", "AVPC", "t_avpc",
            "AVPC_NEPC", "t_avpc_nepc",
        ]
    )
    events, times = bpi.longitudinal_event_columns(frame)
    assert events == ["PLATINUM", "DEATH", "NEPC", "AVPC"]
    assert times == ["t_platinum", "t_death", "t_nepc", "t_avpc"]


def test_joint_endpoint_absent_from_outcome_registry():
    assert "avpc_nepc" not in cohort.OPTIONAL_ENDPOINT_SPECS
