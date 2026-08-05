import pandas as pd

from COMPASS.data_preprocessing.longitudinal_data_processing import (
    ADT_ANCHOR_MEDS,
    coerce_mixed_datetime,
    compute_first_platinum,
    compute_treatment_anchor,
)


def test_coerce_mixed_datetime_parses_each_release_format():
    values = pd.Series(
        [
            "2024-01-02",
            "03/04/2024",
            "2024-05-06 07:08:09",
            "not a date",
        ]
    )

    parsed = coerce_mixed_datetime(values)

    assert parsed.iloc[:3].notna().all()
    assert pd.isna(parsed.iloc[3])


def test_mixed_medication_dates_survive_anchor_and_platinum_processing():
    medications = pd.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3, 4],
            "NCI_PREFERRED_MED_NM": [
                "LEUPROLIDE ACETATE",
                "DEGARELIX",
                "BICALUTAMIDE",
                "CARBOPLATIN",
            ],
            "MED_START_DT": [
                "2024-01-02",
                "03/04/2024",
                "2024-05-06 07:08:09",
                "07/08/2024",
            ],
        }
    )

    anchors = compute_treatment_anchor(medications, meds_set=ADT_ANCHOR_MEDS)
    platinum = compute_first_platinum(medications)

    assert set(anchors["DFCI_MRN"]) == {1, 2, 3}
    assert set(platinum["DFCI_MRN"]) == {4}

