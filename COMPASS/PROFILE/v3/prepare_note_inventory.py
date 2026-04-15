import argparse
from pathlib import Path

import pandas as pd

from common import load_note_text_dataframe, load_selected_mrns
from settings import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_RAW_TEXT_PATHS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the shared v3 raw-note inventory for all task-specific arms."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--text-source",
        choices=["compiled", "raw"],
        default="raw",
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
    for column in ("NUM_CLINICIAN_NOTES", "NUM_IMAGING_NOTES", "NUM_PATHOLOGY_NOTES"):
        if column not in note_counts.columns:
            note_counts[column] = 0
    note_counts["TOTAL_NUM_NOTES"] = note_counts[
        ["NUM_CLINICIAN_NOTES", "NUM_IMAGING_NOTES", "NUM_PATHOLOGY_NOTES"]
    ].sum(axis=1)
    return note_counts


def build_patient_context(text_df):
    context_df = build_note_counts(text_df)
    if context_df.empty:
        context_df = pd.DataFrame(columns=["DFCI_MRN"])
    return context_df.sort_values("DFCI_MRN").reset_index(drop=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    raw_text_paths = resolve_raw_text_paths(args.raw_text_path)

    text_df = load_note_text_dataframe(args.text_source, args.data_path, raw_text_paths, selected_mrns)
    keep_cols = [
        column
        for column in (
            "DFCI_MRN",
            "EVENT_DATE",
            "NOTE_TYPE",
            "CLINICAL_TEXT",
            "RAW_SOURCE_FILE",
            "RAW_NOTE_ID",
            "RPT_DATE",
            "RPT_TYPE",
            "SOURCE_STR",
            "PROC_DESC_STR",
            "ENCOUNTER_TYPE_DESC_STR",
        )
        if column in text_df.columns
    ]
    inventory_df = text_df[keep_cols].copy()
    inventory_df["EVENT_DATE"] = pd.to_datetime(inventory_df["EVENT_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    inventory_df = inventory_df.sort_values(["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"], na_position="last")
    context_df = build_patient_context(text_df)

    inventory_path = args.output_dir / "LLM_v3_note_inventory.csv"
    context_path = args.output_dir / "LLM_v3_patient_context.csv"
    inventory_df.to_csv(inventory_path, index=False)
    context_df.to_csv(context_path, index=False)

    print(f"Wrote note inventory: {inventory_path}")
    print(f"Wrote patient context: {context_path}")
    print(f"Patients in inventory: {inventory_df['DFCI_MRN'].nunique()}")
    print(f"Notes in inventory: {len(inventory_df)}")
    print(f"Text source: {args.text_source}")
    if args.text_source == "raw":
        print(f"Raw text directories searched: {', '.join(str(path) for path in raw_text_paths)}")
    if selected_mrns is not None:
        print(f"Requested MRNs: {len(selected_mrns)}")


if __name__ == "__main__":
    main()
