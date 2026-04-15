import argparse
from pathlib import Path

import pandas as pd

from common import normalize_mrn_column
from settings import ARM_NAMES, DEFAULT_OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Merge task-specific v3 arm outputs into final labels.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-path", type=Path, default=None)
    return parser.parse_args()


def load_arm_labels(output_dir, arm_name):
    path = output_dir / f"LLM_v3_{arm_name}_labels.tsv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["DFCI_MRN"])
    df = normalize_mrn_column(pd.read_csv(path, sep="\t", low_memory=False))
    rename_map = {column: f"{arm_name}_{column}" for column in df.columns if column != "DFCI_MRN"}
    return df.rename(columns=rename_map)


def is_true(value):
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def is_nullish(value):
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "none", "null", "nan"}


def assign_final_bucket(row):
    if is_true(row.get("nepc_has_nepc_signal")):
        return "nepc"
    if is_true(row.get("avpc_has_avpc_features")):
        return "avpc"
    if is_true(row.get("biomarker_has_biomarker_signal")):
        return "biomarker"
    return "conventional_prostate_cancer"


def needs_manual_review(row):
    if assign_final_bucket(row) != "conventional_prostate_cancer":
        return False
    return any(
        is_nullish(row.get(column))
        for column in (
            "nepc_has_nepc_signal",
            "avpc_has_avpc_features",
            "biomarker_has_biomarker_signal",
        )
    )


def main():
    args = parse_args()
    context_path = args.context_path or args.output_dir / "LLM_v3_patient_context.csv"
    context_df = normalize_mrn_column(pd.read_csv(context_path, low_memory=False))
    if "DFCI_MRN" not in context_df.columns:
        raise ValueError(f"Context file is missing DFCI_MRN: {context_path}")

    merged_df = context_df.copy()
    for arm_name in ARM_NAMES:
        arm_df = load_arm_labels(args.output_dir, arm_name)
        merged_df = merged_df.merge(arm_df, on="DFCI_MRN", how="left")

    merged_df["final_bucket"] = merged_df.apply(assign_final_bucket, axis=1)
    merged_df["needs_manual_review"] = merged_df.apply(needs_manual_review, axis=1)

    output_path = args.output_dir / "LLM_v3_merged_labels.tsv"
    merged_df.to_csv(output_path, sep="\t", index=False)

    print(f"Wrote merged labels: {output_path}")
    print(f"Patients in merged labels: {merged_df['DFCI_MRN'].nunique()}")


if __name__ == "__main__":
    main()
