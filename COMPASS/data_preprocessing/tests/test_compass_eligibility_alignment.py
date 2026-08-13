from datetime import datetime

import polars as pl

from COMPASS.data_preprocessing.compile_COMPASS_cohort_data import (
    ADT_ANCHOR_MEDS,
    PARPI_MEDS,
    PLATINUM_MEDS,
    compute_eligible_mrns,
    compute_first_prostate_diagnosis,
    compute_male_mrns,
    compute_platinum_pre_diagnosis_mrns,
    compute_post_adt_exclusion_cancer_mrns,
    compute_treatment_anchor,
)
from COMPASS.data_preprocessing.longitudinal_data_processing import (
    ADT_ANCHOR_MEDS as LONGITUDINAL_ADT_MEDS,
    PARPI_MEDS as LONGITUDINAL_PARPI_MEDS,
    PLATINUM_MEDS as LONGITUDINAL_PLATINUM_MEDS,
)


def test_reference_medication_sets_are_aligned():
    assert PLATINUM_MEDS == {"CARBOPLATIN", "CISPLATIN"}
    assert PARPI_MEDS == {
        "OLAPARIB",
        "RUCAPARIB",
        "RUCAPARIB CAMSYLATE",
        "NIRAPARIB",
        "TALAZOPARIB",
        "TALAZOPARIB TOSYLATE",
    }
    assert "RELUGOLIX" not in ADT_ANCHOR_MEDS
    assert LONGITUDINAL_ADT_MEDS == ADT_ANCHOR_MEDS
    assert LONGITUDINAL_PARPI_MEDS == PARPI_MEDS
    assert LONGITUDINAL_PLATINUM_MEDS == PLATINUM_MEDS


def test_dated_diagnosis_and_post_diagnosis_adt_rules():
    icds = pl.DataFrame(
        {
            "DFCI_MRN": [1, 1, 2],
            "DIAGNOSIS_ICD10_CD": ["C61", "C61.9", "C61"],
            "START_DT": ["2022-03-01", "2022-01-01", None],
        }
    )
    diagnoses = compute_first_prostate_diagnosis(icds)
    assert diagnoses.to_dicts() == [
        {"DFCI_MRN": 1, "DIAGNOSIS_DATE": datetime(2022, 1, 1)}
    ]

    meds = pl.DataFrame(
        {
            "DFCI_MRN": [1, 1, 2],
            "NCI_PREFERRED_MED_NM": [
                "BICALUTAMIDE",
                "LEUPROLIDE ACETATE",
                "BICALUTAMIDE",
            ],
            "MED_START_DT": [
                datetime(2021, 12, 1),
                datetime(2022, 2, 1),
                datetime(2022, 2, 1),
            ],
        }
    )
    first_adt_anywhere = compute_treatment_anchor(meds, ADT_ANCHOR_MEDS)
    first_adt_post_diagnosis = compute_treatment_anchor(
        meds, ADT_ANCHOR_MEDS, diagnosis_df=diagnoses
    )

    assert first_adt_anywhere.sort("DFCI_MRN").to_dicts() == [
        {"DFCI_MRN": 1, "TREATMENT_ANCHOR_DATE": datetime(2021, 12, 1)},
        {"DFCI_MRN": 2, "TREATMENT_ANCHOR_DATE": datetime(2022, 2, 1)},
    ]
    assert first_adt_post_diagnosis.sort("DFCI_MRN").to_dicts() == [
        {"DFCI_MRN": 1, "TREATMENT_ANCHOR_DATE": datetime(2022, 2, 1)}
    ]


def test_male_prior_platinum_and_combined_eligibility_rules():
    status = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            "GENDER": ["Male", "FEMALE", "m"],
        }
    )
    assert compute_male_mrns(status) == {1, 3}

    diagnoses = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            "DIAGNOSIS_DATE": [
                datetime(2022, 1, 1),
                datetime(2022, 1, 1),
                datetime(2022, 1, 1),
            ],
        }
    )
    platinum = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            "PLATINUM_MED": ["CARBOPLATIN", "CISPLATIN", "CARBOPLATIN"],
            "PLATINUM_DATE": [
                datetime(2022, 2, 1),
                datetime(2021, 12, 1),
                datetime(2022, 3, 1),
            ],
        }
    )
    assert compute_platinum_pre_diagnosis_mrns(platinum, diagnoses) == {2}

    common = {1, 2, 3, 4, 5, 6, 7}
    eligible = compute_eligible_mrns(
        c61_mrns=common,
        dated_diagnosis_mrns=common - {7},
        male_mrns=common - {2},
        psa_eligible_mrns=common - {3},
        adt_post_diagnosis_mrns=common - {4},
        parpi_mrns={5},
        platinum_pre_diagnosis_mrns={6},
        post_adt_exclusion_cancer_mrns=set(),
    )
    assert eligible == {1}


def test_exclusion_cancers_are_strictly_after_first_adt():
    first_adt_anywhere = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3, 4],
            "TREATMENT_ANCHOR_DATE": [datetime(2022, 1, 1)] * 4,
        }
    )
    icds = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3, 4],
            "DIAGNOSIS_ICD10_CD": ["C67.9", "C34.1", "C62.9", "C18.9"],
            # Same-day and pre-ADT diagnoses do not exclude. A post-ADT
            # requested cancer does; an unrequested cancer remains descriptive.
            "START_DT": ["2022-01-01", "2022-01-02", "2021-12-31", "2022-01-02"],
        }
    )
    assert compute_post_adt_exclusion_cancer_mrns(icds, first_adt_anywhere) == {2}
