import argparse
from pathlib import Path

import pandas as pd

from common import (
    load_note_text_dataframe,
    load_selected_mrns,
)
from config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_TEXT_PATHS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build v3 patient context for unified prostate phenotype extraction."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--text-source",
        choices=["compiled", "raw"],
        default="compiled",
        help="Use compiled prostate_text_data.csv or raw OncDRS JSON notes.",
    )
    parser.add_argument(
        "--raw-text-path",
        type=Path,
        action="append",
        default=None,
        help="Raw OncDRS note directory. Repeat to search multiple directories.",
    )
    parser.add_argument("--mrns", default=None, help="Comma-separated DFCI_MRN values to include.")
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=None,
        help="Optional text/CSV/TSV file containing DFCI_MRN values.",
    )
    return parser.parse_args()


def resolve_raw_text_paths(raw_text_paths_arg=None):
    if raw_text_paths_arg:
        ordered_paths = []
        seen = set()
        for path in raw_text_paths_arg:
            normalized = Path(path)
            key = str(normalized)
            if key not in seen:
                seen.add(key)
                ordered_paths.append(normalized)
        return ordered_paths
    return list(DEFAULT_RAW_TEXT_PATHS)


def build_note_counts(text_df):
    if text_df.empty:
        return pd.DataFrame(
            columns=[
                "DFCI_MRN",
                "NUM_CLINICIAN_NOTES",
                "NUM_IMAGING_NOTES",
                "NUM_PATHOLOGY_NOTES",
                "TOTAL_NUM_NOTES",
            ]
        )

    note_counts = (
        text_df.pivot_table(index="DFCI_MRN", columns="NOTE_TYPE", aggfunc="size", fill_value=0)
        .reset_index()
        .rename(
            columns={
                "Clinician": "NUM_CLINICIAN_NOTES",
                "Imaging": "NUM_IMAGING_NOTES",
                "Pathology": "NUM_PATHOLOGY_NOTES",
            }
        )
    )
    for column in ["NUM_CLINICIAN_NOTES", "NUM_IMAGING_NOTES", "NUM_PATHOLOGY_NOTES"]:
        if column not in note_counts.columns:
            note_counts[column] = 0
    note_counts["TOTAL_NUM_NOTES"] = note_counts[
        ["NUM_CLINICIAN_NOTES", "NUM_IMAGING_NOTES", "NUM_PATHOLOGY_NOTES"]
    ].sum(axis=1)
    return note_counts


def build_patient_context(text_df):
    all_mrns = sorted(set(text_df["DFCI_MRN"].dropna().tolist())) if not text_df.empty else []
    context_df = pd.DataFrame({"DFCI_MRN": all_mrns})

    note_counts = build_note_counts(text_df)
    context_df = context_df.merge(note_counts, on="DFCI_MRN", how="left")

    for column in ["NUM_CLINICIAN_NOTES", "NUM_IMAGING_NOTES", "NUM_PATHOLOGY_NOTES", "TOTAL_NUM_NOTES"]:
        if column in context_df.columns:
            context_df[column] = context_df[column].fillna(0).astype(int)
    return context_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    raw_text_paths = resolve_raw_text_paths(args.raw_text_path)

    text_df = load_note_text_dataframe(args.text_source, args.data_path, raw_text_paths, selected_mrns)
    context_df = build_patient_context(text_df)

    context_path = args.output_dir / "LLM_v3_patient_context.csv"
    context_df.to_csv(context_path, index=False)

    print(f"Wrote patient context: {context_path}")
    print(f"Patients in context: {context_df['DFCI_MRN'].nunique()}")
    print(f"Text source: {args.text_source}")
    if args.text_source == "raw":
        print(f"Raw text directories searched: {', '.join(str(path) for path in raw_text_paths)}")
    if selected_mrns is not None:
        print(f"Requested MRNs: {len(selected_mrns)}")


if __name__ == "__main__":
    main()
