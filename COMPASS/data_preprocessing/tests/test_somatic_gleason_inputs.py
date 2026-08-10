from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from build_somatic_gleason_inputs import (  # noqa: E402
    GLEASON_FEATURE,
    GLEASON_AVAILABLE_DATE,
    SOMATIC_AVAILABLE_DATE,
    build_landmark_features,
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


def test_build_landmark_features_keeps_untested_somatic_values_missing():
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2],
            "TREATMENT_ANCHOR_DATE": ["2020-01-01", "2020-01-01"],
            "AGE_AT_TREATMENTSTART": [70, 65],
            "PLATINUM": [1, 0],
            "t_platinum": [100, 200],
            "split": ["train", "test"],
        }
    )
    somatic = pd.DataFrame(
        {
            "DFCI_MRN": [1],
            SOMATIC_AVAILABLE_DATE: pd.to_datetime(["2020-01-20"]),
            "TP53_SNV": [1],
        }
    )
    gleason = pd.DataFrame(
        {
            "DFCI_MRN": [1, 1, 2],
            "gleason_date": pd.to_datetime(["2019-01-01", "2020-03-01", "2019-02-01"]),
            GLEASON_AVAILABLE_DATE: pd.to_datetime(
                ["2019-01-01", "2020-03-01", "2019-02-01"]
            ),
            GLEASON_FEATURE: [7, 9, 8],
        }
    )

    result = build_landmark_features(base, somatic, ["TP53_SNV"], gleason, 30).set_index(
        "DFCI_MRN"
    )

    assert result.loc[1, "TP53_SNV"] == 1
    assert np.isnan(result.loc[2, "TP53_SNV"])
    assert result.loc[1, GLEASON_FEATURE] == 7
    assert result.loc[2, GLEASON_FEATURE] == 8
    assert "TREATMENT_ANCHOR_DATE" in result.columns
    assert "AGE_AT_TREATMENTSTART" in result.columns


def test_historical_gleason_in_future_note_does_not_leak_backwards():
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2],
            "TREATMENT_ANCHOR_DATE": ["2020-01-01", "2020-01-01"],
            "AGE_AT_TREATMENTSTART": [70, 65],
            "PLATINUM": [1, 0],
            "t_platinum": [100, 200],
            "split": ["train", "test"],
        }
    )
    somatic = pd.DataFrame(
        columns=["DFCI_MRN", SOMATIC_AVAILABLE_DATE, "TP53_SNV"]
    )
    gleason = pd.DataFrame(
        {
            "DFCI_MRN": [1],
            "gleason_date": pd.to_datetime(["2018-01-01"]),
            # The old score was only surfaced in a note written after the landmark.
            GLEASON_AVAILABLE_DATE: pd.to_datetime(["2020-03-01"]),
            GLEASON_FEATURE: [9],
        }
    )

    result = build_landmark_features(base, somatic, ["TP53_SNV"], gleason, 30).set_index(
        "DFCI_MRN"
    )

    assert np.isnan(result.loc[1, GLEASON_FEATURE])


def test_landmark_features_can_recover_anchor_dropped_from_base_inputs():
    base = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2],
            "AGE_AT_TREATMENTSTART": [70, 65],
            "PLATINUM": [1, 0],
            "t_platinum": [100, 200],
            "split": ["train", "test"],
        }
    )
    somatic = pd.DataFrame(
        {
            "DFCI_MRN": [1],
            SOMATIC_AVAILABLE_DATE: pd.to_datetime(["2020-01-20"]),
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

    result = build_landmark_features(
        base,
        somatic,
        ["TP53_SNV"],
        gleason,
        30,
        treatment_anchors=anchors,
    ).set_index("DFCI_MRN")

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
