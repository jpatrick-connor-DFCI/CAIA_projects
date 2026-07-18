"""
Script Name: build_vte_cohort_mrn_list.py

Description:
Derives the VTE-prediction-project prostate MRN list for the "vte_cohort"
comparison arm (see COMPASS_run_locally.ipynb). This restricts the current
ARPI-anchored pipeline to the older, smaller MRN universe that COMPASS's
cohort used to be built from (compile_prostate_data.py, prior to the switch
to raw EHR_DIAGNOSIS.csv-derived ICD-C61 cohort in
compile_COMPASS_cohort_data.py).

The VTE project's `med_genomics_merged_cancer_group == 'PROSTATE'` tag is
used as-is (no intersection with the current ICD-C61 cohort) -- this is the
restriction set passed to build_prediction_inputs.py's --restrict-to-mrns.
The overlap with the current ICD-C61 cohort is only reported as a diagnostic,
not applied as a filter, so it's visible how much the two cohorts diverge.

Output:
  <out-dir>/vte_cohort_prostate_mrns.csv   (single DFCI_MRN column)

Author: J. Patrick Connor
"""

import argparse
import os

import pandas as pd

DATA_PATH = '/data/gusev/USERS/jpconnor/data/'
NEPC_PROJ_PATH = os.path.join(DATA_PATH, 'CAIA/COMPASS/')

PROFILE_PATH = '/data/gusev/PROFILE/CLINICAL/'
INTAE_DATA_PATH = os.path.join(PROFILE_PATH, 'robust_VTE_pred_project_2025_03_cohort/data/')

ID_COL = "DFCI_MRN"


def load_vte_prostate_mrns(vte_cancer_types_path: str) -> pd.Series:
    """VTE-project MRNs tagged PROSTATE by med_genomics_merged_cancer_group."""
    cancer_types = pd.read_csv(vte_cancer_types_path)[[ID_COL, 'med_genomics_merged_cancer_group']]
    mrns = cancer_types.loc[
        cancer_types['med_genomics_merged_cancer_group'] == 'PROSTATE', ID_COL
    ]
    return pd.to_numeric(mrns, errors='coerce').dropna().astype(int).drop_duplicates()


def load_current_icd_c61_mrns(survival_cohort_path: str) -> pd.Series:
    """Current ICD-C61 cohort MRNs, from compile_COMPASS_cohort_data.py's output."""
    cohort = pd.read_csv(survival_cohort_path)[[ID_COL]]
    return pd.to_numeric(cohort[ID_COL], errors='coerce').dropna().astype(int).drop_duplicates()


def main():
    parser = argparse.ArgumentParser(
        description="Derive the VTE-project prostate MRN list used by the "
        "vte_cohort comparison arm's --restrict-to-mrns filter.",
    )
    parser.add_argument(
        "--vte-cancer-types",
        type=str,
        default=os.path.join(INTAE_DATA_PATH, 'first_treatments_dfci_w_inferred_cancers.csv'),
        help="VTE project's inferred-cancer-type file (DFCI_MRN, "
             "med_genomics_merged_cancer_group columns).",
    )
    parser.add_argument(
        "--survival-cohort",
        type=str,
        default=os.path.join(NEPC_PROJ_PATH, 'prostate_arpi_survival_cohort.csv'),
        help="Current ICD-C61 cohort file (compile_COMPASS_cohort_data.py output), "
             "used only for the overlap diagnostic below.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(NEPC_PROJ_PATH, 'mrn_lists'),
        help="Directory to write vte_cohort_prostate_mrns.csv to.",
    )
    args = parser.parse_args()

    vte_mrns = load_vte_prostate_mrns(args.vte_cancer_types)
    print(f"VTE-project PROSTATE-tagged MRNs: {len(vte_mrns)}")

    if os.path.exists(args.survival_cohort):
        current_mrns = load_current_icd_c61_mrns(args.survival_cohort)
        overlap = set(vte_mrns) & set(current_mrns)
        print(f"Current ICD-C61 cohort MRNs:      {len(current_mrns)}")
        print(
            f"Overlap:                          {len(overlap)} "
            f"({len(overlap) / len(vte_mrns):.1%} of VTE cohort, "
            f"{len(overlap) / len(current_mrns):.1%} of ICD-C61 cohort)"
        )
    else:
        print(
            f"  [skip] overlap diagnostic: {args.survival_cohort} not found "
            "(run compile_COMPASS_cohort_data.py first for this report)."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'vte_cohort_prostate_mrns.csv')
    pd.DataFrame({ID_COL: sorted(vte_mrns)}).to_csv(out_path, index=False)
    print(f"\nSaved VTE cohort MRN list to {out_path}")


if __name__ == "__main__":
    main()
