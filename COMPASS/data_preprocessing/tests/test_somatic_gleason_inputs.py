from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from build_somatic_gleason_inputs import (  # noqa: E402
    GLEASON_FEATURE,
    GLEASON_AVAILABLE_DATE,
    INDEX_DATE,
    INDEX_TO_ADT_DAYS,
    SEQUENCING_DATE,
    SOMATIC_AVAILABLE_DATE,
    SOMATIC_BURDEN_COVARIATE,
    build_available_case_sensitivity,
    build_indexed_feature_sets,
    closest_observation_to_adt,
    latest_available_by_landmark,
    load_somatic_features,
)


def test_latest_available_by_landmark_excludes_future_rows():
    observations = pd.DataFrame(
        {
            "DFCI_MRN": [1, 1, 2],
            "observed": pd.to_datetime(["2020-01-05", "2020-02-05", "2020-03-01"]),
            "value": [6, 9, 8],
        }
    )
    cutoffs = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2],
            "_landmark_date": pd.to_datetime(["2020-02-01", "2020-02-01"]),
        }
    )

    result = latest_available_by_landmark(
        observations, cutoffs, date_col="observed", value_cols=["value"]
    )

    assert result.loc[1, "value"] == 6
    assert 2 not in result.index


def test_latest_somatic_testing_groups_on_same_date_are_combined():
    observations = pd.DataFrame(
        {
            "DFCI_MRN": [1, 1, 1],
            "observed": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-02-01"]),
            "TP53_SNV": [1, 0, 1],
            "PTEN_DEL": [0, 1, 0],
        }
    )
    cutoffs = pd.DataFrame(
        {"DFCI_MRN": [1], "_landmark_date": pd.to_datetime(["2020-03-01"])}
    )

    result = latest_available_by_landmark(
        observations,
        cutoffs,
        date_col="observed",
        value_cols=["TP53_SNV", "PTEN_DEL"],
        combine_latest_ties_with_max=True,
    )

    assert result.loc[1, "TP53_SNV"] == 1
    assert result.loc[1, "PTEN_DEL"] == 1


def test_closest_observation_uses_absolute_distance_and_prefers_earlier_tie():
    observations = pd.DataFrame(
        {
            "DFCI_MRN": [1, 1, 1],
            "date": pd.to_datetime(["2020-01-01", "2020-01-09", "2020-01-11"]),
            "value": [6, 7, 9],
        }
    )
    anchors = pd.Series(
        pd.to_datetime(["2020-01-10"]), index=pd.Index([1], name="DFCI_MRN")
    )

    selected = closest_observation_to_adt(
        observations, anchors, date_col="date", value_cols=["value"]
    )

    assert selected.loc[1, INDEX_DATE] == pd.Timestamp("2020-01-09")
    assert selected.loc[1, INDEX_TO_ADT_DAYS] == -1
    assert selected.loc[1, "value"] == 7


def test_available_case_sensitivity_is_adt_anchored_and_excludes_future_data():
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            "split": ["train", "test", "valid"],
            "t_platinum": [100, 100, 100],
            "PLATINUM": [1, 0, 1],
            "PSA__mean": [4.0, 5.0, 6.0],
        }
    )
    anchors = pd.Series(
        pd.to_datetime(["2020-01-10"] * 3), index=pd.Index([1, 2, 3], name="DFCI_MRN")
    )
    somatic = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            SEQUENCING_DATE: pd.to_datetime(["2020-01-05", "2020-01-05", "2020-01-20"]),
            SOMATIC_AVAILABLE_DATE: pd.to_datetime(["2020-01-05", "2020-01-05", "2020-01-20"]),
            "TP53_SNV": [1, 0, 1],
        }
    )
    gleason = pd.DataFrame(
        {
            "DFCI_MRN": [1, 1, 2, 3],
            "gleason_date": pd.to_datetime(["2020-01-02", "2020-01-09", "2020-01-02", "2020-01-02"]),
            GLEASON_AVAILABLE_DATE: pd.to_datetime(["2020-01-02", "2020-01-09", "2020-01-20", "2020-01-02"]),
            GLEASON_FEATURE: [8, 9, 9, 7],
        }
    )

    result = build_available_case_sensitivity(
        base, somatic, ["TP53_SNV"], gleason, treatment_anchors=anchors
    )

    # MRN 2's score is documented after ADT; MRN 3's specimen is after ADT.
    assert result["DFCI_MRN"].tolist() == [1]
    assert result.loc[0, "split"] == "train"
    assert result.loc[0, "PSA__mean"] == 4.0
    # The Jan 9 score, not Jan 2, is closest to the Jan 10 landmark.
    assert result.loc[0, GLEASON_FEATURE] == 9
    assert "TP53_SNV" not in result

    later = build_available_case_sensitivity(
        base, somatic, ["TP53_SNV"], gleason,
        treatment_anchors=anchors, landmark_day=90,
    )
    assert later["DFCI_MRN"].tolist() == [1, 2, 3]

    somatic_only = build_available_case_sensitivity(
        base, somatic, ["TP53_SNV"], gleason,
        treatment_anchors=anchors, analysis="somatic",
    )
    assert somatic_only["DFCI_MRN"].tolist() == [1, 2]
    assert GLEASON_FEATURE not in somatic_only
    assert somatic_only.loc[0, "TP53_SNV"] == 1


def test_indexed_feature_sets_use_distinct_dates_and_followup_clocks():
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            "TREATMENT_ANCHOR_DATE": ["2020-01-10"] * 3,
            "AGE_AT_TREATMENTSTART": [70, 65, 72],
            "PLATINUM": [1, 0, 1],
            "t_platinum": [101, 143, -5],
            "t_last_contact": [143, 143, 143],
            "split": ["train", "test", "valid"],
        }
    )
    somatic = pd.DataFrame(
        {
            "DFCI_MRN": [1, 1, 2, 3],
            SEQUENCING_DATE: pd.to_datetime(
                ["2020-01-05", "2020-02-01", "2020-01-20", "2020-01-20"]
            ),
            "TP53_SNV": [1, 0, 0, 1],
        }
    )
    gleason = pd.DataFrame(
        {
            "DFCI_MRN": [1, 1, 2, 3],
            "gleason_date": pd.to_datetime(
                ["2020-01-08", "2020-03-01", "2020-02-01", "2020-01-20"]
            ),
            GLEASON_FEATURE: [7, 9, 8, 10],
        }
    )

    result = build_indexed_feature_sets(
        base, {"sequencing_snv": (somatic, ["TP53_SNV"])}, gleason
    )
    sequencing = result["sequencing_snv"].set_index("DFCI_MRN")
    gleason_out = result["gleason"].set_index("DFCI_MRN")

    assert sequencing.loc[1, INDEX_DATE] == pd.Timestamp("2020-01-05")
    assert sequencing.loc[1, "t_platinum"] == 106
    assert gleason_out.loc[1, INDEX_DATE] == pd.Timestamp("2020-01-08")
    assert gleason_out.loc[1, GLEASON_FEATURE] == 7
    assert gleason_out.loc[1, "t_platinum"] == 103
    # Censored follow-up is rebased from the selected Gleason date.
    assert gleason_out.loc[2, "t_platinum"] == 121
    # The outcome precedes patient 3's selected index dates, so that patient is
    # excluded rather than selecting a different observation using the outcome.
    assert 3 not in sequencing.index
    assert 3 not in gleason_out.index
    # The SNV run carries no burden adjustment covariate.
    assert SOMATIC_BURDEN_COVARIATE not in sequencing.columns


def test_nepc_indexed_feature_sets_ignore_pre_anchor_platinum():
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1],
            "TREATMENT_ANCHOR_DATE": ["2020-01-10"],
            "AGE_AT_TREATMENTSTART": [70],
            "PLATINUM": [1],
            "t_platinum": [-5],
            "NEPC": [1],
            "t_nepc": [100],
            "t_last_contact": [300],
            "split": ["train"],
        }
    )
    somatic = pd.DataFrame(
        {"DFCI_MRN": [1], SEQUENCING_DATE: pd.to_datetime(["2020-01-20"]), "TP53_SNV": [1]}
    )
    gleason = pd.DataFrame(
        {"DFCI_MRN": [1], "gleason_date": pd.to_datetime(["2020-01-20"]), GLEASON_FEATURE: [9]}
    )

    result = build_indexed_feature_sets(
        base, {"sequencing_snv": (somatic, ["TP53_SNV"])}, gleason, endpoint="nepc"
    )

    assert result["sequencing_snv"].loc[0, "t_nepc"] == 90
    assert result["gleason"].loc[0, "t_nepc"] == 90


@pytest.mark.parametrize(
    "endpoint, event_col, duration_col",
    [
        ("platinum", "PLATINUM", "t_platinum"),
        ("nepc", "NEPC", "t_nepc"),
        ("avpc", "AVPC", "t_avpc"),
    ],
)
def test_every_endpoint_rebases_from_dropped_date_columns(
    endpoint, event_col, duration_col
):
    """build_prediction_inputs strips every raw *_DATE column from the aggregated
    table, so each endpoint must be reconstructible from its ADT-relative
    duration alone. An endpoint added to _rebase_endpoint_from_index without a
    matching entry in _materialize_absolute_followup_dates fails here.
    """
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1],
            "TREATMENT_ANCHOR_DATE": ["2020-01-10"],
            "AGE_AT_TREATMENTSTART": [70],
            event_col: [1],
            duration_col: [100],
            "t_last_contact": [300],
            "split": ["train"],
        }
    )
    somatic = pd.DataFrame(
        {"DFCI_MRN": [1], SEQUENCING_DATE: pd.to_datetime(["2020-01-20"]), "TP53_SNV": [1]}
    )
    gleason = pd.DataFrame(
        {"DFCI_MRN": [1], "gleason_date": pd.to_datetime(["2020-01-20"]), GLEASON_FEATURE: [9]}
    )

    result = build_indexed_feature_sets(
        base, {"sequencing_snv": (somatic, ["TP53_SNV"])}, gleason, endpoint=endpoint
    )

    # Index date is 10 days after the ADT anchor, so the event clock loses 10 days.
    assert result["sequencing_snv"].loc[0, duration_col] == 90
    assert result["gleason"].loc[0, duration_col] == 90
    assert result["sequencing_snv"].loc[0, event_col] == 1


def test_censored_patient_rebases_to_last_contact_not_event_date():
    """For a censored patient the endpoint duration falls back to t_last_contact,
    so reconstruction puts a last-contact date in AVPC_DATE. The rebased clock
    must still be last contact minus the index date, and the event must stay 0.
    """
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1],
            "TREATMENT_ANCHOR_DATE": ["2020-01-10"],
            "AGE_AT_TREATMENTSTART": [70],
            "AVPC": [0],
            "t_avpc": [300],
            "t_last_contact": [300],
            "split": ["train"],
        }
    )
    somatic = pd.DataFrame(
        {"DFCI_MRN": [1], SEQUENCING_DATE: pd.to_datetime(["2020-01-20"]), "TP53_SNV": [1]}
    )
    gleason = pd.DataFrame(
        {"DFCI_MRN": [1], "gleason_date": pd.to_datetime(["2020-01-20"]), GLEASON_FEATURE: [9]}
    )

    result = build_indexed_feature_sets(
        base, {"sequencing_snv": (somatic, ["TP53_SNV"])}, gleason, endpoint="avpc"
    )

    assert result["sequencing_snv"].loc[0, "AVPC"] == 0
    assert result["sequencing_snv"].loc[0, "t_avpc"] == 290


def test_gleason_selection_uses_score_date_not_source_note_date():
    gleason = pd.DataFrame(
        {
            "DFCI_MRN": [1, 1],
            "gleason_date": pd.to_datetime(["2020-01-02", "2019-01-01"]),
            GLEASON_AVAILABLE_DATE: pd.to_datetime(["2021-01-01", "2019-01-01"]),
            GLEASON_FEATURE: [9, 7],
        }
    )
    anchors = pd.Series(
        pd.to_datetime(["2020-01-01"]), index=pd.Index([1], name="DFCI_MRN")
    )

    result = closest_observation_to_adt(
        gleason,
        anchors,
        date_col="gleason_date",
        value_cols=[GLEASON_FEATURE],
    )

    assert result.loc[1, GLEASON_FEATURE] == 9
    assert result.loc[1, INDEX_DATE] == pd.Timestamp("2020-01-02")


def test_indexed_features_can_recover_anchor_dropped_from_base_inputs():
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2],
            "AGE_AT_TREATMENTSTART": [70, 65],
            "LAST_CONTACT_DATE": ["2020-06-01", "2020-06-01"],
            "PLATINUM_DATE": ["2020-04-10", "2020-06-01"],
            "PLATINUM": [1, 0],
            "t_platinum": [100, 200],
            "split": ["train", "test"],
        }
    )
    somatic = pd.DataFrame(
        {
            "DFCI_MRN": [1],
            SEQUENCING_DATE: pd.to_datetime(["2020-01-20"]),
            "TP53_SNV": [1],
        }
    )
    gleason = pd.DataFrame(
        columns=["DFCI_MRN", "gleason_date", GLEASON_AVAILABLE_DATE, GLEASON_FEATURE]
    )
    anchors = pd.Series(
        pd.to_datetime(["2020-01-01", "2020-01-01"]),
        index=pd.Index([1, 2], name="DFCI_MRN"),
    )

    result = build_indexed_feature_sets(
        base,
        {"sequencing_snv": (somatic, ["TP53_SNV"])},
        gleason,
        treatment_anchors=anchors,
    )["sequencing_snv"].set_index("DFCI_MRN")

    assert result.loc[1, "TP53_SNV"] == 1


def _write_somatic_fixture(tmp_path):
    """Somatic matrix + manifest declaring a mix of variant classes."""
    manifest_path = tmp_path / "manifest.csv"
    somatic_path = tmp_path / "somatic.csv"
    declared = [
        "TP53_SNV", "AR_SNV", "RB1_SNV", "PTEN_DEL", "RB1_DEL", "TP53_SV",
        "AR_AMP", "TMPRSS2_SV",
    ]
    pd.DataFrame({"FEATURE": declared}).to_csv(manifest_path, index=False)
    pd.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3, 4],
            "SAMPLE_COLLECTION_DT": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
            "TP53_SNV": [1, 0, 0, np.nan],
            "AR_SNV": [1, 0, 0, 0],
            "RB1_SNV": [0, 0, 0, np.nan],
            "PTEN_DEL": [0, 1, 0, np.nan],
            "RB1_DEL": [0, 1, 0, np.nan],
            "TP53_SV": [0, 1, 0, np.nan],
            "AR_AMP": [1, 1, 1, 1],
            "TMPRSS2_SV": [1, 1, 1, 1],
        }
    ).to_csv(somatic_path, index=False)
    return somatic_path, manifest_path


def test_load_somatic_features_trio_is_any_alteration_others_snv_only(tmp_path):
    somatic_path, manifest_path = _write_somatic_fixture(tmp_path)

    frame, features = load_somatic_features(somatic_path, manifest_path)

    # TP53/PTEN/RB1 collapse every declared class into one <GENE>_ANY
    # indicator; every other gene stays SNV-only.
    assert features == ["TP53_ANY", "AR_SNV", "RB1_ANY", "PTEN_ANY"]
    for dropped in ("TP53_SNV", "RB1_SNV", "PTEN_DEL", "RB1_DEL", "TP53_SV",
                    "AR_AMP", "TMPRSS2_SV"):
        assert dropped not in frame.columns
    frame = frame.set_index("DFCI_MRN")
    # Patient 2 is altered only by DEL/SV calls -- missed by an SNV-only arm.
    assert frame.loc[1, ["TP53_ANY", "RB1_ANY", "PTEN_ANY"]].tolist() == [1, 0, 0]
    assert frame.loc[2, ["TP53_ANY", "RB1_ANY", "PTEN_ANY"]].tolist() == [1, 1, 1]
    assert frame.loc[3, ["TP53_ANY", "RB1_ANY", "PTEN_ANY"]].tolist() == [0, 0, 0]
    # No class observed -> missing, not wild-type.
    assert frame.loc[4, ["TP53_ANY", "RB1_ANY", "PTEN_ANY"]].isna().all()


def test_load_somatic_features_errors_when_manifest_has_nothing_testable(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    somatic_path = tmp_path / "somatic.csv"
    pd.DataFrame({"FEATURE": ["TMPRSS2_SV", "AR_AMP"]}).to_csv(manifest_path, index=False)
    pd.DataFrame(
        {
            "DFCI_MRN": [1],
            "SAMPLE_COLLECTION_DT": ["2020-01-01"],
            "TMPRSS2_SV": [1],
            "AR_AMP": [0],
        }
    ).to_csv(somatic_path, index=False)

    with pytest.raises(ValueError, match="none are testable"):
        load_somatic_features(somatic_path, manifest_path)


def test_load_somatic_features_snv_mode_is_snv_only_for_every_gene(tmp_path):
    somatic_path, manifest_path = _write_somatic_fixture(tmp_path)

    frame, features = load_somatic_features(
        somatic_path, manifest_path, variant_mode="snv"
    )

    # The trio is no longer special-cased: every gene is tested by its SNV call.
    assert features == ["TP53_SNV", "AR_SNV", "RB1_SNV"]
    assert not [column for column in frame.columns if column.endswith("_ANY")]
    for dropped in ("PTEN_DEL", "RB1_DEL", "TP53_SV", "AR_AMP", "TMPRSS2_SV"):
        assert dropped not in frame.columns
    frame = frame.set_index("DFCI_MRN")
    assert frame.loc[2, ["TP53_SNV", "RB1_SNV"]].tolist() == [0, 0]


def test_load_somatic_features_any_mode_collapses_every_gene(tmp_path):
    somatic_path, manifest_path = _write_somatic_fixture(tmp_path)

    frame, features = load_somatic_features(
        somatic_path, manifest_path, variant_mode="any"
    )

    assert features == ["TP53_ANY", "AR_ANY", "RB1_ANY", "PTEN_ANY", "TMPRSS2_ANY"]
    assert not [
        column for column in frame.columns
        if column.endswith(("_SNV", "_DEL", "_AMP", "_SV"))
    ]
    frame = frame.set_index("DFCI_MRN")
    # AR_ANY collapses AR_SNV and AR_AMP; TMPRSS2 enters only through its SV call.
    assert frame.loc[1, features].tolist() == [1, 1, 0, 0, 1]
    assert frame.loc[2, features].tolist() == [1, 1, 1, 1, 1]
    assert frame.loc[3, features].tolist() == [0, 1, 0, 0, 1]
    assert frame.loc[4, ["TP53_ANY", "RB1_ANY", "PTEN_ANY"]].isna().all()
    assert frame.loc[4, "AR_ANY"] == 1


def test_load_somatic_features_rejects_unknown_variant_mode(tmp_path):
    somatic_path, manifest_path = _write_somatic_fixture(tmp_path)

    with pytest.raises(ValueError, match="Unknown somatic variant_mode"):
        load_somatic_features(somatic_path, manifest_path, variant_mode="cnv")


def test_any_alteration_run_carries_burden_of_all_tested_genes():
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            "TREATMENT_ANCHOR_DATE": ["2020-01-10"] * 3,
            "AGE_AT_TREATMENTSTART": [70, 65, 72],
            "PLATINUM": [1, 0, 0],
            "t_platinum": [200, 300, 300],
            "t_last_contact": [300, 300, 300],
            "split": ["train", "train", "test"],
        }
    )
    snv = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            SEQUENCING_DATE: pd.to_datetime(["2020-01-20"] * 3),
            "TP53_SNV": [1, 0, 0],
        }
    )
    any_alteration = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            SEQUENCING_DATE: pd.to_datetime(["2020-01-20"] * 3),
            "TP53_ANY": [1, 1, 0],
            "PTEN_ANY": [1, 0, np.nan],
            "AR_ANY": [1, 0, 0],
        }
    )
    gleason = pd.DataFrame(
        {"DFCI_MRN": [1], "gleason_date": pd.to_datetime(["2020-01-20"]), GLEASON_FEATURE: [9]}
    )
    any_features = ["TP53_ANY", "PTEN_ANY", "AR_ANY"]

    result = build_indexed_feature_sets(
        base,
        {
            "sequencing_snv": (snv, ["TP53_SNV"]),
            "sequencing_any": (any_alteration, any_features),
        },
        gleason,
    )

    snv_out = result["sequencing_snv"]
    any_out = result["sequencing_any"].set_index("DFCI_MRN")
    assert SOMATIC_BURDEN_COVARIATE not in snv_out.columns
    assert SOMATIC_BURDEN_COVARIATE not in result["gleason"].columns
    # One count across every tested gene (missing calls count as unaltered),
    # identical for every per-gene fit.
    assert any_out[SOMATIC_BURDEN_COVARIATE].to_dict() == {1: 3, 2: 1, 3: 0}
    assert (any_out[SOMATIC_BURDEN_COVARIATE] == any_out[any_features].sum(axis=1)).all()


def test_indexed_feature_sets_reject_unknown_sequencing_analysis():
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1],
            "TREATMENT_ANCHOR_DATE": ["2020-01-10"],
            "AGE_AT_TREATMENTSTART": [70],
            "PLATINUM": [0],
            "t_platinum": [300],
            "t_last_contact": [300],
            "split": ["train"],
        }
    )
    somatic = pd.DataFrame(
        {"DFCI_MRN": [1], SEQUENCING_DATE: pd.to_datetime(["2020-01-20"]), "TP53_SNV": [1]}
    )
    gleason = pd.DataFrame(
        {"DFCI_MRN": [1], "gleason_date": pd.to_datetime(["2020-01-20"]), GLEASON_FEATURE: [9]}
    )

    with pytest.raises(ValueError):
        build_indexed_feature_sets(base, {"sequencing": (somatic, ["TP53_SNV"])}, gleason)
