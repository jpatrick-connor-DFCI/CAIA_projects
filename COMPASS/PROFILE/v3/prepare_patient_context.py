import argparse
from pathlib import Path

import pandas as pd

from common import (
    load_note_text_dataframe,
    load_selected_mrns,
    normalize_mrn_column,
    parse_datetime_series,
    safe_read_csv,
)
from config import (
    ADT_MEDS,
    ARSI_MEDS,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_TEXT_PATHS,
    PARP_MEDS,
    PLATINUM_MEDS,
    TAXANE_MEDS,
    TOTAL_PSA_LABELS,
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


def normalize_med_names(series):
    return series.astype(str).str.upper().str.strip()


def build_first_med_summary(meds_df, med_names, date_col="MED_START_DT"):
    if meds_df.empty:
        return pd.DataFrame(columns=["DFCI_MRN", "medication", "med_start_date"])

    required_cols = {"DFCI_MRN", "NCI_PREFERRED_MED_NM", date_col}
    if not required_cols.issubset(meds_df.columns):
        return pd.DataFrame(columns=["DFCI_MRN", "medication", "med_start_date"])

    work = meds_df.loc[normalize_med_names(meds_df["NCI_PREFERRED_MED_NM"]).isin(med_names)].copy()
    if work.empty:
        return pd.DataFrame(columns=["DFCI_MRN", "medication", "med_start_date"])

    work["medication"] = normalize_med_names(work["NCI_PREFERRED_MED_NM"])
    work["med_start_date"] = parse_datetime_series(work[date_col])
    work = work.dropna(subset=["DFCI_MRN", "med_start_date"]).sort_values(["DFCI_MRN", "med_start_date"])
    return work.drop_duplicates(subset=["DFCI_MRN"], keep="first")[["DFCI_MRN", "medication", "med_start_date"]]


def summarize_psa(psa_df):
    empty = pd.DataFrame(
        columns=["DFCI_MRN", "LATEST_PSA_DATE", "LATEST_PSA_VALUE", "MAX_PSA_VALUE"]
    )
    if psa_df.empty:
        return empty

    work = psa_df.copy()
    if "TEST_TYPE_CD" in work.columns:
        work = work.loc[work["TEST_TYPE_CD"].astype(str).str.upper().isin(TOTAL_PSA_LABELS)]

    date_col = "D_SPECIMEN_COLLECT_DT" if "D_SPECIMEN_COLLECT_DT" in work.columns else "SPECIMEN_COLLECT_DT"
    value_col = "NUMERIC_RESULT" if "NUMERIC_RESULT" in work.columns else "RESULT_NBR"
    if date_col not in work.columns or value_col not in work.columns:
        return empty

    work["PSA_DATE"] = parse_datetime_series(work[date_col])
    work["PSA_VALUE"] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["DFCI_MRN", "PSA_DATE", "PSA_VALUE"])
    work = work.loc[work["PSA_VALUE"] != 9999999.0]
    if work.empty:
        return empty

    latest = work.sort_values(["DFCI_MRN", "PSA_DATE"]).drop_duplicates("DFCI_MRN", keep="last")
    max_psa = work.groupby("DFCI_MRN", as_index=False)["PSA_VALUE"].max().rename(
        columns={"PSA_VALUE": "MAX_PSA_VALUE"}
    )
    latest = latest.rename(
        columns={
            "PSA_DATE": "LATEST_PSA_DATE",
            "PSA_VALUE": "LATEST_PSA_VALUE",
        }
    )[["DFCI_MRN", "LATEST_PSA_DATE", "LATEST_PSA_VALUE"]]
    return latest.merge(max_psa, on="DFCI_MRN", how="left")


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


def compute_first_hormonal_therapy_date(context_df):
    dates = []
    for column in ("FIRST_ADT_DATE", "FIRST_ARSI_DATE"):
        if column in context_df.columns:
            dates.append(pd.to_datetime(context_df[column], errors="coerce"))
    if not dates:
        return pd.Series(pd.NaT, index=context_df.index)
    stacked = pd.concat(dates, axis=1)
    return stacked.min(axis=1)


def build_patient_context(text_df, meds_df, psa_df):
    mrn_sets = []
    for df in (text_df, meds_df, psa_df):
        if not df.empty and "DFCI_MRN" in df.columns:
            mrn_sets.append(set(df["DFCI_MRN"].dropna().tolist()))
    all_mrns = sorted(set().union(*mrn_sets)) if mrn_sets else []
    context_df = pd.DataFrame({"DFCI_MRN": all_mrns})

    note_counts = build_note_counts(text_df)
    platinum = build_first_med_summary(meds_df, PLATINUM_MEDS).rename(
        columns={"medication": "FIRST_PLATINUM_MED", "med_start_date": "FIRST_PLATINUM_DATE"}
    )
    adt = build_first_med_summary(meds_df, ADT_MEDS).rename(
        columns={"medication": "FIRST_ADT_MED", "med_start_date": "FIRST_ADT_DATE"}
    )
    arsi = build_first_med_summary(meds_df, ARSI_MEDS).rename(
        columns={"medication": "FIRST_ARSI_MED", "med_start_date": "FIRST_ARSI_DATE"}
    )
    parp = build_first_med_summary(meds_df, PARP_MEDS).rename(
        columns={"medication": "FIRST_PARP_MED", "med_start_date": "FIRST_PARP_DATE"}
    )
    taxane = build_first_med_summary(meds_df, TAXANE_MEDS).rename(
        columns={"medication": "FIRST_TAXANE_MED", "med_start_date": "FIRST_TAXANE_DATE"}
    )
    psa_summary = summarize_psa(psa_df)

    context_df = (
        context_df.merge(note_counts, on="DFCI_MRN", how="left")
        .merge(platinum, on="DFCI_MRN", how="left")
        .merge(adt, on="DFCI_MRN", how="left")
        .merge(arsi, on="DFCI_MRN", how="left")
        .merge(parp, on="DFCI_MRN", how="left")
        .merge(taxane, on="DFCI_MRN", how="left")
        .merge(psa_summary, on="DFCI_MRN", how="left")
    )

    for column in ["NUM_CLINICIAN_NOTES", "NUM_IMAGING_NOTES", "NUM_PATHOLOGY_NOTES", "TOTAL_NUM_NOTES"]:
        if column in context_df.columns:
            context_df[column] = context_df[column].fillna(0).astype(int)

    context_df["EVER_PLATINUM"] = context_df["FIRST_PLATINUM_DATE"].notna()
    context_df["EVER_ADT"] = context_df["FIRST_ADT_DATE"].notna()
    context_df["EVER_ARSI"] = context_df["FIRST_ARSI_DATE"].notna()
    context_df["EVER_PARP"] = context_df["FIRST_PARP_DATE"].notna()
    context_df["EVER_TAXANE"] = context_df["FIRST_TAXANE_DATE"].notna()
    context_df["FIRST_HORMONAL_THERAPY_DATE"] = compute_first_hormonal_therapy_date(context_df)
    return context_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    raw_text_paths = resolve_raw_text_paths(args.raw_text_path)

    meds_path = args.data_path / "prostate_medications_data.csv"
    total_psa_path = args.data_path / "total_psa_records.csv"
    fallback_labs_path = args.data_path / "prostate_labs_data.csv"

    text_df = load_note_text_dataframe(args.text_source, args.data_path, raw_text_paths, selected_mrns)
    meds_df = normalize_mrn_column(safe_read_csv(meds_path))
    psa_df = normalize_mrn_column(safe_read_csv(total_psa_path))
    if psa_df.empty:
        psa_df = normalize_mrn_column(safe_read_csv(fallback_labs_path))

    if selected_mrns is not None:
        if not meds_df.empty:
            meds_df = meds_df.loc[meds_df["DFCI_MRN"].isin(selected_mrns)].copy()
        if not psa_df.empty:
            psa_df = psa_df.loc[psa_df["DFCI_MRN"].isin(selected_mrns)].copy()

    context_df = build_patient_context(text_df, meds_df, psa_df)

    for date_col in [
        "FIRST_PLATINUM_DATE",
        "FIRST_ADT_DATE",
        "FIRST_ARSI_DATE",
        "FIRST_PARP_DATE",
        "FIRST_TAXANE_DATE",
        "FIRST_HORMONAL_THERAPY_DATE",
        "LATEST_PSA_DATE",
    ]:
        if date_col in context_df.columns:
            context_df[date_col] = pd.to_datetime(context_df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

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
