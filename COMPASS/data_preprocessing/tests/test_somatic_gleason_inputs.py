from __future__ import annotations

import sys
import tempfile
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
    build_indexed_feature_sets,
    closest_observation_to_adt,
    latest_available_by_landmark,
    load_biomarker_prs,
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
    prs = pd.DataFrame(
        {"DFCI_MRN": [1, 2], "PROSTATE_PRS": [0.4, -0.2]}
    )

    result = build_indexed_feature_sets(
        base,
        somatic,
        ["TP53_SNV"],
        gleason,
        prs=prs,
        prs_features=["PROSTATE_PRS"],
    )
    sequencing = result["sequencing"].set_index("DFCI_MRN")
    gleason_out = result["gleason"].set_index("DFCI_MRN")
    prs_out = result["prs"].set_index("DFCI_MRN")

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
    # PRS keeps the original ADT-start clock and baseline risk set.
    assert prs_out.loc[1, INDEX_DATE] == pd.Timestamp("2020-01-10")
    assert prs_out.loc[1, INDEX_TO_ADT_DAYS] == 0
    assert prs_out.loc[1, "t_platinum"] == 101
    assert prs_out.loc[1, "PROSTATE_PRS"] == 0.4
    assert np.isnan(prs_out.loc[3, "PROSTATE_PRS"])


def test_prs_loader_bridges_sample_keyed_matrix_through_idmap():
    with tempfile.TemporaryDirectory() as directory:
        prs_path = Path(directory) / "pgs_matrix_with_avg.tsv"
        idmap_path = Path(directory) / "PROFILE_2024_idmap.csv"
        score = "PSA_PGS_CSx_Prostate_specific_antigen__PSA__levels_PGS003379"
        pd.DataFrame(
            {
                "IID": ["S1", "S2", "S3", "S_unmapped"],
                score: [0.2, 0.6, -0.1, 5.0],
            }
        ).to_csv(prs_path, sep="\t", index=False)
        # MRN 1 has two genotyped samples; MRN 2 has one. S_unmapped has no MRN.
        pd.DataFrame(
            {
                "DFCI_MRN": [1, 1, 2],
                "cbio_sample_id": ["S1", "S2", "S3"],
            }
        ).to_csv(idmap_path, index=False)

        prs, manifest = load_biomarker_prs(prs_path, idmap_path=idmap_path)

    prs = prs.set_index("DFCI_MRN")
    assert len(prs) == 2
    # Multiple samples per patient are averaged.
    assert np.isclose(prs.loc[1, score], 0.4)
    assert np.isclose(prs.loc[2, score], -0.1)
    # A sample absent from the idmap is unreachable and contributes nothing.
    assert 5.0 not in prs[score].to_numpy()
    assert {row["pgs_id"] for row in manifest} == {"PGS003379"}


def test_prs_loader_requires_idmap_for_sample_keyed_matrix():
    with tempfile.TemporaryDirectory() as directory:
        prs_path = Path(directory) / "pgs_matrix_with_avg.tsv"
        score = "PSA_PGS_CSx_Prostate_specific_antigen__PSA__levels_PGS003379"
        pd.DataFrame({"IID": ["S1"], score: [0.2]}).to_csv(
            prs_path, sep="\t", index=False
        )

        with pytest.raises(ValueError, match="needs an idmap"):
            load_biomarker_prs(prs_path, idmap_path=None)


def test_prs_loader_rejects_sample_mapped_to_multiple_mrns():
    with tempfile.TemporaryDirectory() as directory:
        prs_path = Path(directory) / "pgs_matrix_with_avg.tsv"
        idmap_path = Path(directory) / "PROFILE_2024_idmap.csv"
        score = "PSA_PGS_CSx_Prostate_specific_antigen__PSA__levels_PGS003379"
        pd.DataFrame({"IID": ["S1"], score: [0.2]}).to_csv(
            prs_path, sep="\t", index=False
        )
        pd.DataFrame(
            {"DFCI_MRN": [1, 2], "cbio_sample_id": ["S1", "S1"]}
        ).to_csv(idmap_path, index=False)

        with pytest.raises(ValueError, match="more than one"):
            load_biomarker_prs(prs_path, idmap_path=idmap_path)


def test_prs_loader_missing_ids_degrade_unless_require_all():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "complete_germline_data_df.csv.gz"
        score = "PSA_PGS_CSx_Prostate_specific_antigen__PSA__levels_PGS003379"
        pd.DataFrame({"DFCI_MRN": [1], score: [0.2]}).to_csv(path, index=False)

        prs, manifest = load_biomarker_prs(path)
        assert len(manifest) == 1
        assert list(prs.columns) == ["DFCI_MRN", score]

        with pytest.raises(ValueError, match="missing"):
            load_biomarker_prs(path, require_all=True)


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
        somatic,
        ["TP53_SNV"],
        gleason,
        treatment_anchors=anchors,
    )["sequencing"].set_index("DFCI_MRN")

    assert result.loc[1, "TP53_SNV"] == 1


def test_prs_loader_uses_supplied_ids_and_preserves_full_column_names():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "complete_germline_data_df.csv.gz"
        psa = "PSA_PGS_CSx_Prostate_specific_antigen__PSA__levels_PGS003379"
        testosterone_male = (
            "snpnet_Testosterone_male_specific_"
            "Serum_testosterone_levels_in_males_PGS000323"
        )
        testosterone_female = (
            "snpnet_Testosterone_female_specific_"
            "Serum_testosterone_levels_in_females_PGS000322"
        )
        pd.DataFrame(
            {
                "DFCI_MRN": [1, 2],
                psa: [0.1, 0.2],
                testosterone_male: [-0.1, 0.4],
                testosterone_female: [9.0, 9.0],
                "unrelated_trait_PGS999999": [8.0, 8.0],
            }
        ).to_csv(path, index=False)

        prs, manifest = load_biomarker_prs(path)

    assert set(prs.columns) == {
        "DFCI_MRN",
        psa,
        testosterone_male,
        testosterone_female,
    }
    assert {row["pgs_id"] for row in manifest} == {
        "PGS003379",
        "PGS000322",
        "PGS000323",
    }


def test_prs_loader_averages_conflicting_sample_rows_within_mrn():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "complete_germline_data_df.csv.gz"
        score = "PSA_PGS_128_Prostate_specific_antigen__PSA__levels_PGS003378"
        pd.DataFrame(
            {
                "DFCI_MRN": [1, 1, 2],
                score: [0.2, 0.6, -0.1],
            }
        ).to_csv(path, index=False)

        prs, _manifest = load_biomarker_prs(path)

    prs = prs.set_index("DFCI_MRN")
    assert len(prs) == 2
    assert np.isclose(prs.loc[1, score], 0.4)
    assert np.isclose(prs.loc[2, score], -0.1)
