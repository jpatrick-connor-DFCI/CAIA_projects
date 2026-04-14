import argparse
import math
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = CURRENT_DIR.parent
DATA_PREPROCESSING_DIR = PROFILE_DIR / "data_preprocessing"
if str(DATA_PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PREPROCESSING_DIR))

from consolidate_dfci_labs import consolidate_dfci_labs  # noqa: E402

from common import load_selected_mrns, normalize_mrn_column, parse_datetime_series, safe_read_csv
from config import (
    CEA_ULN,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    HYPERCALCEMIA_THRESHOLD,
    LDH_ULN,
    SOMATIC_TARGET_PATTERNS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive v3 structured features for AVPC, biomarkers, and platinum context."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--context-path",
        type=Path,
        default=None,
        help="Optional patient context CSV. Defaults to output-dir/LLM_v3_patient_context.csv.",
    )
    parser.add_argument("--labs-path", type=Path, default=None)
    parser.add_argument("--somatic-path", type=Path, default=None)
    parser.add_argument("--mapping-csv", type=Path, default=PROFILE_DIR / "OMOP_to_DFCI_lab_ids.csv")
    parser.add_argument("--mrns", default=None, help="Comma-separated DFCI_MRN values to include.")
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=None,
        help="Optional text/CSV/TSV file containing DFCI_MRN values.",
    )
    return parser.parse_args()


def standardize_labs_input(labs_df):
    work = labs_df.copy()
    rename_map = {}
    if "TEST_NAME" not in work.columns and "TEST_TYPE_DESCR" in work.columns:
        rename_map["TEST_TYPE_DESCR"] = "TEST_NAME"
    if "TEST_NAME" not in work.columns and "TEST_TYPE_CD" in work.columns:
        rename_map["TEST_TYPE_CD"] = "TEST_NAME"
    work = work.rename(columns=rename_map)
    return work


def build_peak_measurement_summary(labs_df, measurement):
    empty = pd.DataFrame(
        columns=["DFCI_MRN", f"peak_{measurement.lower()}_value", f"peak_{measurement.lower()}_date"]
    )
    if labs_df.empty:
        return empty

    work = labs_df.loc[labs_df["collapsed_measurement"] == measurement].copy()
    if work.empty:
        return empty

    work = work.dropna(subset=["DFCI_MRN", "numeric_result_standardized", "LAB_DATE"])
    if work.empty:
        return empty

    work = work.sort_values(
        ["DFCI_MRN", "numeric_result_standardized", "LAB_DATE"],
        ascending=[True, False, False],
    )
    peak = work.drop_duplicates("DFCI_MRN", keep="first").rename(
        columns={
            "numeric_result_standardized": f"peak_{measurement.lower()}_value",
            "LAB_DATE": f"peak_{measurement.lower()}_date",
        }
    )
    return peak[["DFCI_MRN", f"peak_{measurement.lower()}_value", f"peak_{measurement.lower()}_date"]]


def value_is_positive(value):
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return False
        return value > 0

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "na", "n/a"}:
        return False
    if text in {"true", "yes", "y", "positive", "present", "detected", "mutated", "altered", "loss"}:
        return True
    if text in {"false", "no", "n", "negative", "absent", "wildtype", "wild-type", "not detected"}:
        return False
    try:
        return float(text) > 0
    except ValueError:
        return any(
            token in text
            for token in (
                "pathogenic",
                "likely pathogenic",
                "deleterious",
                "mutation",
                "mut",
                "alteration",
                "amplification",
                "deletion",
                "loss",
                "rearrangement",
            )
        ) and not any(
            token in text
            for token in ("wild", "negative", "not detected", "no mutation", "no alteration")
        )


def build_somatic_feature_table(somatic_df, patient_mrns):
    rows = []
    if patient_mrns is None:
        patient_mrns = []
    if somatic_df.empty:
        return pd.DataFrame({"DFCI_MRN": list(patient_mrns)})

    work = normalize_mrn_column(somatic_df)
    if patient_mrns:
        work = work.loc[work["DFCI_MRN"].isin(patient_mrns)].copy()
    if work.empty:
        return pd.DataFrame({"DFCI_MRN": list(patient_mrns)})

    upper_columns = {column: str(column).upper() for column in work.columns}
    feature_cols = {}
    for feature_name, patterns in SOMATIC_TARGET_PATTERNS.items():
        matched_cols = [
            column
            for column, upper_name in upper_columns.items()
            if column != "DFCI_MRN" and any(pattern in upper_name for pattern in patterns)
        ]
        feature_cols[feature_name] = matched_cols

    for mrn, mrn_df in work.groupby("DFCI_MRN"):
        row = {"DFCI_MRN": int(mrn)}
        for feature_name, matched_cols in feature_cols.items():
            row[feature_name] = False
            for column in matched_cols:
                if mrn_df[column].map(value_is_positive).any():
                    row[feature_name] = True
                    break
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_df = pd.DataFrame({"DFCI_MRN": list(patient_mrns)})

    expected = list(SOMATIC_TARGET_PATTERNS.keys())
    for column in expected:
        if column not in out_df.columns:
            out_df[column] = False

    out_df["has_any_dna_repair_alteration"] = out_df[
        [
            "has_brca1_alteration",
            "has_brca2_alteration",
            "has_atm_alteration",
            "has_cdk12_alteration",
            "has_palb2_alteration",
            "has_hrd_pathway_alteration",
            "has_ddr_pathway_alteration",
        ]
    ].any(axis=1)
    return out_df


def load_and_consolidate_labs(labs_path, mapping_csv, selected_mrns=None):
    labs_df = normalize_mrn_column(safe_read_csv(labs_path))
    if labs_df.empty:
        return pd.DataFrame()
    if selected_mrns is not None:
        labs_df = labs_df.loc[labs_df["DFCI_MRN"].isin(selected_mrns)].copy()
    if labs_df.empty:
        return pd.DataFrame()

    labs_df = standardize_labs_input(labs_df)
    labs_df["LAB_DATE"] = parse_datetime_series(
        labs_df["D_SPECIMEN_COLLECT_DT"]
        if "D_SPECIMEN_COLLECT_DT" in labs_df.columns
        else labs_df.get("SPECIMEN_COLLECT_DT")
    )
    mapping_df = pd.read_csv(mapping_csv, low_memory=False)
    consolidated = consolidate_dfci_labs(labs_df, mapping_df)
    if "LAB_DATE" in consolidated.columns:
        consolidated["LAB_DATE"] = parse_datetime_series(consolidated["LAB_DATE"])
    elif "LAB_DATE" in labs_df.columns:
        consolidated["LAB_DATE"] = parse_datetime_series(labs_df["LAB_DATE"])
    consolidated = consolidated.loc[consolidated["conversion_status"] != "unmapped_test_name"].copy()
    if "numeric_result_standardized" in consolidated.columns:
        consolidated = consolidated.dropna(subset=["numeric_result_standardized"])
    return consolidated


def build_lab_feature_table(consolidated_labs, patient_mrns):
    feature_df = pd.DataFrame({"DFCI_MRN": list(patient_mrns)})
    if consolidated_labs.empty:
        return feature_df

    peak_ldh = build_peak_measurement_summary(consolidated_labs, "LDH")
    peak_cea = build_peak_measurement_summary(consolidated_labs, "CEA")
    peak_calcium = build_peak_measurement_summary(consolidated_labs, "Calcium")

    feature_df = (
        feature_df.merge(peak_ldh, on="DFCI_MRN", how="left")
        .merge(peak_cea, on="DFCI_MRN", how="left")
        .merge(peak_calcium, on="DFCI_MRN", how="left")
    )
    feature_df["ldh_ge_2x_uln"] = feature_df["peak_ldh_value"].fillna(-1) >= (2 * LDH_ULN)
    feature_df["cea_ge_2x_uln"] = feature_df["peak_cea_value"].fillna(-1) >= (2 * CEA_ULN)
    feature_df["hypercalcemia_present"] = (
        feature_df["peak_calcium_value"].fillna(-1) >= HYPERCALCEMIA_THRESHOLD
    )
    feature_df["c6_supportive_lab_pattern_present"] = feature_df[
        ["ldh_ge_2x_uln", "cea_ge_2x_uln", "hypercalcemia_present"]
    ].any(axis=1)
    return feature_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    context_path = args.context_path or args.output_dir / "LLM_v3_patient_context.csv"
    labs_path = args.labs_path or args.data_path / "prostate_labs_data.csv"
    somatic_path = args.somatic_path or args.data_path / "prostate_somatic_data.csv"

    context_df = normalize_mrn_column(pd.read_csv(context_path, low_memory=False))
    if selected_mrns is not None:
        context_df = context_df.loc[context_df["DFCI_MRN"].isin(selected_mrns)].copy()
    if context_df.empty:
        raise ValueError("No patients remained for structured feature derivation.")

    patient_mrns = context_df["DFCI_MRN"].dropna().astype(int).tolist()

    consolidated_labs = load_and_consolidate_labs(labs_path, args.mapping_csv, selected_mrns=patient_mrns)
    lab_features = build_lab_feature_table(consolidated_labs, patient_mrns)

    somatic_df = normalize_mrn_column(safe_read_csv(somatic_path))
    somatic_features = build_somatic_feature_table(somatic_df, patient_mrns)

    feature_df = (
        pd.DataFrame({"DFCI_MRN": patient_mrns})
        .merge(lab_features, on="DFCI_MRN", how="left")
        .merge(somatic_features, on="DFCI_MRN", how="left")
    )
    for column in feature_df.columns:
        if column.startswith("has_") or column.endswith("_present"):
            feature_df[column] = feature_df[column].fillna(False).astype(bool)

    for date_col in [
        "peak_ldh_date",
        "peak_cea_date",
        "peak_calcium_date",
    ]:
        if date_col in feature_df.columns:
            feature_df[date_col] = pd.to_datetime(feature_df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    output_path = args.output_dir / "LLM_v3_derived_features.tsv"
    feature_df.to_csv(output_path, sep="\t", index=False)

    print(f"Wrote derived features: {output_path}")
    print(f"Patients with derived features: {feature_df['DFCI_MRN'].nunique()}")
    print(f"Consolidated lab rows used: {len(consolidated_labs)}")
    print(f"Somatic rows used: {len(somatic_df) if not somatic_df.empty else 0}")


if __name__ == "__main__":
    main()
