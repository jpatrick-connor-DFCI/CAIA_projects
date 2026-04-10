import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from config import (
    ADT_MEDS,
    ARSI_MEDS,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLATINUM_WINDOW_DAYS,
    NOTE_TRIGGER_REGEX,
    NOTE_TYPE_LIMITS,
    PARP_MEDS,
    PLATINUM_MEDS,
    TOTAL_PSA_LABELS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build v2 candidate notes and patient context for broad prostate event extraction."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--platinum-window-days",
        type=int,
        default=DEFAULT_PLATINUM_WINDOW_DAYS,
        help="Include clinician and imaging notes near first platinum exposure when present.",
    )
    parser.add_argument(
        "--max-clinician-notes",
        type=int,
        default=NOTE_TYPE_LIMITS["Clinician"],
    )
    parser.add_argument(
        "--max-imaging-notes",
        type=int,
        default=NOTE_TYPE_LIMITS["Imaging"],
    )
    parser.add_argument(
        "--max-pathology-notes",
        type=int,
        default=NOTE_TYPE_LIMITS["Pathology"],
    )
    return parser.parse_args()


def safe_read_csv(path, **kwargs):
    if not path.exists():
        return pd.DataFrame()
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(path, **kwargs)


def normalize_mrn_column(df):
    if df.empty or "DFCI_MRN" not in df.columns:
        return df
    work = df.copy()
    work["DFCI_MRN"] = pd.to_numeric(work["DFCI_MRN"], errors="coerce")
    work = work.dropna(subset=["DFCI_MRN"])
    work["DFCI_MRN"] = work["DFCI_MRN"].astype(int)
    return work


def parse_datetime_series(series):
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)


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
    if psa_df.empty:
        return pd.DataFrame(
            columns=["DFCI_MRN", "LATEST_PSA_DATE", "LATEST_PSA_VALUE", "MAX_PSA_VALUE"]
        )

    work = psa_df.copy()
    if "TEST_TYPE_CD" in work.columns:
        work = work.loc[work["TEST_TYPE_CD"].astype(str).str.upper().isin(TOTAL_PSA_LABELS)]

    date_col = "D_SPECIMEN_COLLECT_DT" if "D_SPECIMEN_COLLECT_DT" in work.columns else "SPECIMEN_COLLECT_DT"
    value_col = "NUMERIC_RESULT" if "NUMERIC_RESULT" in work.columns else "RESULT_NBR"
    if date_col not in work.columns or value_col not in work.columns:
        return pd.DataFrame(
            columns=["DFCI_MRN", "LATEST_PSA_DATE", "LATEST_PSA_VALUE", "MAX_PSA_VALUE"]
        )

    work["PSA_DATE"] = parse_datetime_series(work[date_col])
    work["PSA_VALUE"] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["DFCI_MRN", "PSA_DATE", "PSA_VALUE"])
    work = work.loc[work["PSA_VALUE"] != 9999999.0]
    if work.empty:
        return pd.DataFrame(
            columns=["DFCI_MRN", "LATEST_PSA_DATE", "LATEST_PSA_VALUE", "MAX_PSA_VALUE"]
        )

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


def build_patient_context(text_df, meds_df, psa_df):
    mrn_sets = []
    for df in (text_df, meds_df, psa_df):
        if not df.empty and "DFCI_MRN" in df.columns:
            mrn_sets.append(set(df["DFCI_MRN"].dropna().tolist()))
    all_mrns = sorted(set().union(*mrn_sets)) if mrn_sets else []
    context_df = pd.DataFrame({"DFCI_MRN": all_mrns})

    note_counts = build_note_counts(text_df)
    platinum = build_first_med_summary(meds_df, PLATINUM_MEDS).rename(
        columns={
            "medication": "FIRST_PLATINUM_MED",
            "med_start_date": "FIRST_PLATINUM_DATE",
        }
    )
    adt = build_first_med_summary(meds_df, ADT_MEDS).rename(
        columns={
            "medication": "FIRST_ADT_MED",
            "med_start_date": "FIRST_ADT_DATE",
        }
    )
    arsi = build_first_med_summary(meds_df, ARSI_MEDS).rename(
        columns={
            "medication": "FIRST_ARSI_MED",
            "med_start_date": "FIRST_ARSI_DATE",
        }
    )
    parp = build_first_med_summary(meds_df, PARP_MEDS).rename(
        columns={
            "medication": "FIRST_PARP_MED",
            "med_start_date": "FIRST_PARP_DATE",
        }
    )
    psa_summary = summarize_psa(psa_df)

    context_df = (
        context_df.merge(note_counts, on="DFCI_MRN", how="left")
        .merge(platinum, on="DFCI_MRN", how="left")
        .merge(adt, on="DFCI_MRN", how="left")
        .merge(arsi, on="DFCI_MRN", how="left")
        .merge(parp, on="DFCI_MRN", how="left")
        .merge(psa_summary, on="DFCI_MRN", how="left")
    )

    for column in ["NUM_CLINICIAN_NOTES", "NUM_IMAGING_NOTES", "NUM_PATHOLOGY_NOTES", "TOTAL_NUM_NOTES"]:
        if column in context_df.columns:
            context_df[column] = context_df[column].fillna(0).astype(int)

    context_df["EVER_PLATINUM"] = context_df["FIRST_PLATINUM_DATE"].notna()
    context_df["EVER_ADT"] = context_df["FIRST_ADT_DATE"].notna()
    context_df["EVER_ARSI"] = context_df["FIRST_ARSI_DATE"].notna()
    context_df["EVER_PARP"] = context_df["FIRST_PARP_DATE"].notna()
    return context_df


def annotate_text_triggers(text_df):
    work = text_df.copy()
    work["NOTE_TYPE"] = work["NOTE_TYPE"].fillna("Unknown")
    work["EVENT_DATE"] = parse_datetime_series(work["EVENT_DATE"])
    work["CLINICAL_TEXT"] = work["CLINICAL_TEXT"].fillna("").astype(str)
    normalized_text = work["CLINICAL_TEXT"].str.lower()

    for name, pattern in NOTE_TRIGGER_REGEX.items():
        work[f"HAS_{name.upper()}_TRIGGER"] = normalized_text.str.contains(pattern, regex=True, na=False)

    trigger_cols = [col for col in work.columns if col.startswith("HAS_") and col.endswith("_TRIGGER")]
    work["TRIGGER_CATEGORY_COUNT"] = work[trigger_cols].sum(axis=1)
    work["TRIGGER_SCORE"] = (
        work["HAS_HISTOLOGY_TRIGGER"].astype(int) * 3
        + work["HAS_METASTATIC_TRIGGER"].astype(int) * 3
        + work["HAS_PLATINUM_TRIGGER"].astype(int) * 3
        + work["HAS_ADT_NONRESPONSE_TRIGGER"].astype(int) * 2
        + work["HAS_BIOMARKER_TRIGGER"].astype(int)
        + work["HAS_TRIAL_TRIGGER"].astype(int)
    )
    work["NOTE_TEXT_HASH"] = work["CLINICAL_TEXT"].apply(
        lambda text: hashlib.md5(text.strip().encode("utf-8", errors="ignore")).hexdigest()
    )
    work = work.drop_duplicates(subset=["DFCI_MRN", "NOTE_TYPE", "EVENT_DATE", "NOTE_TEXT_HASH"])
    return work


def selection_reason(row):
    reasons = []
    if row["NOTE_TYPE"] == "Pathology":
        reasons.append("pathology_default")
    if row.get("HAS_HISTOLOGY_TRIGGER", False):
        reasons.append("histology_trigger")
    if row.get("HAS_METASTATIC_TRIGGER", False):
        reasons.append("metastatic_trigger")
    if row.get("HAS_PLATINUM_TRIGGER", False):
        reasons.append("platinum_trigger")
    if row.get("HAS_ADT_NONRESPONSE_TRIGGER", False):
        reasons.append("adt_trigger")
    if row.get("HAS_BIOMARKER_TRIGGER", False):
        reasons.append("biomarker_trigger")
    if row.get("HAS_TRIAL_TRIGGER", False):
        reasons.append("trial_context")
    if row.get("WITHIN_PLATINUM_WINDOW", False):
        reasons.append("platinum_window")
    if row.get("FALLBACK_INCLUDED", False):
        reasons.append("fallback_recent_note")
    return "|".join(reasons) if reasons else "unspecified"


def take_with_coverage(df, limit):
    if df.empty or len(df) <= limit:
        return df

    df = df.copy()
    df["EVENT_DATE_RANK"] = df["EVENT_DATE"].fillna(pd.Timestamp.min)
    keep_indices = []

    dated = df.loc[df["EVENT_DATE"].notna()]
    if not dated.empty:
        keep_indices.append(dated["EVENT_DATE"].idxmin())
        keep_indices.append(dated["EVENT_DATE"].idxmax())

    priority = df.sort_values(["TRIGGER_SCORE", "TRIGGER_CATEGORY_COUNT", "EVENT_DATE_RANK"], ascending=[False, False, False])
    for idx in priority.index:
        if idx not in keep_indices:
            keep_indices.append(idx)
        if len(keep_indices) >= limit:
            break

    return df.loc[keep_indices[:limit]].drop(columns=["EVENT_DATE_RANK"], errors="ignore")


def select_patient_notes(patient_df, context_row, args):
    patient_df = patient_df.copy()
    patient_df["WITHIN_PLATINUM_WINDOW"] = False
    platinum_date = context_row.get("FIRST_PLATINUM_DATE")
    if pd.notna(platinum_date):
        patient_df["DAYS_TO_PLATINUM"] = (patient_df["EVENT_DATE"] - platinum_date).dt.days
        patient_df["WITHIN_PLATINUM_WINDOW"] = patient_df["DAYS_TO_PLATINUM"].abs() <= args.platinum_window_days
    else:
        patient_df["DAYS_TO_PLATINUM"] = pd.NA

    selected_parts = []

    pathology_df = patient_df.loc[patient_df["NOTE_TYPE"] == "Pathology"].copy()
    pathology_df["FALLBACK_INCLUDED"] = False
    pathology_df["SELECTION_REASON"] = pathology_df.apply(selection_reason, axis=1)
    selected_parts.append(take_with_coverage(pathology_df, args.max_pathology_notes))

    imaging_df = patient_df.loc[
        (patient_df["NOTE_TYPE"] == "Imaging")
        & (
            patient_df["HAS_METASTATIC_TRIGGER"]
            | patient_df["HAS_HISTOLOGY_TRIGGER"]
            | patient_df["HAS_PLATINUM_TRIGGER"]
            | patient_df["WITHIN_PLATINUM_WINDOW"]
        )
    ].copy()
    imaging_df["FALLBACK_INCLUDED"] = False
    imaging_df["SELECTION_REASON"] = imaging_df.apply(selection_reason, axis=1)
    selected_parts.append(take_with_coverage(imaging_df, args.max_imaging_notes))

    clinician_df = patient_df.loc[
        (patient_df["NOTE_TYPE"] == "Clinician")
        & (
            patient_df["TRIGGER_CATEGORY_COUNT"].gt(0)
            | patient_df["WITHIN_PLATINUM_WINDOW"]
        )
    ].copy()
    clinician_df["FALLBACK_INCLUDED"] = False
    clinician_df["SELECTION_REASON"] = clinician_df.apply(selection_reason, axis=1)
    selected_parts.append(take_with_coverage(clinician_df, args.max_clinician_notes))

    selected_df = pd.concat(selected_parts, ignore_index=False)

    if selected_df.empty:
        fallback_df = patient_df.sort_values("EVENT_DATE", ascending=False).head(2).copy()
        fallback_df["FALLBACK_INCLUDED"] = True
        fallback_df["SELECTION_REASON"] = fallback_df.apply(selection_reason, axis=1)
        selected_df = fallback_df

    selected_df = selected_df.drop_duplicates(subset=["DFCI_MRN", "NOTE_TYPE", "EVENT_DATE", "NOTE_TEXT_HASH"])
    return selected_df.sort_values(["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"], na_position="last")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_path = args.data_path / "prostate_text_data.csv"
    meds_path = args.data_path / "prostate_medications_data.csv"
    total_psa_path = args.data_path / "total_psa_records.csv"
    fallback_labs_path = args.data_path / "prostate_labs_data.csv"

    text_df = normalize_mrn_column(safe_read_csv(text_path))
    meds_df = normalize_mrn_column(safe_read_csv(meds_path))
    psa_df = normalize_mrn_column(safe_read_csv(total_psa_path))
    if psa_df.empty:
        psa_df = normalize_mrn_column(safe_read_csv(fallback_labs_path))

    if text_df.empty:
        raise FileNotFoundError(f"Could not load note data from {text_path}")

    context_df = build_patient_context(text_df, meds_df, psa_df)
    annotated_text_df = annotate_text_triggers(text_df)

    selected_notes = []
    context_lookup = context_df.set_index("DFCI_MRN").to_dict(orient="index")
    for mrn, patient_df in tqdm(annotated_text_df.groupby("DFCI_MRN"), desc="Selecting v2 notes"):
        selected_notes.append(select_patient_notes(patient_df, context_lookup.get(mrn, {}), args))

    candidate_df = pd.concat(selected_notes, ignore_index=True) if selected_notes else pd.DataFrame()
    keep_cols = [
        "DFCI_MRN",
        "EVENT_DATE",
        "NOTE_TYPE",
        "CLINICAL_TEXT",
        "TRIGGER_CATEGORY_COUNT",
        "TRIGGER_SCORE",
        "SELECTION_REASON",
        "HAS_HISTOLOGY_TRIGGER",
        "HAS_METASTATIC_TRIGGER",
        "HAS_PLATINUM_TRIGGER",
        "HAS_ADT_NONRESPONSE_TRIGGER",
        "HAS_BIOMARKER_TRIGGER",
        "HAS_TRIAL_TRIGGER",
        "WITHIN_PLATINUM_WINDOW",
    ]
    if candidate_df.empty:
        candidate_df = pd.DataFrame(columns=keep_cols)
    else:
        candidate_df = candidate_df[keep_cols].sort_values(["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"])

    for date_col in ["FIRST_PLATINUM_DATE", "FIRST_ADT_DATE", "FIRST_ARSI_DATE", "FIRST_PARP_DATE", "LATEST_PSA_DATE"]:
        if date_col in context_df.columns:
            context_df[date_col] = pd.to_datetime(context_df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    candidate_path = args.output_dir / "LLM_v2_candidate_text_data.csv"
    context_path = args.output_dir / "LLM_v2_patient_context.csv"
    candidate_df.to_csv(candidate_path, index=False)
    context_df.to_csv(context_path, index=False)

    print(f"Wrote candidate notes: {candidate_path}")
    print(f"Wrote patient context: {context_path}")
    print(f"Patients in context: {context_df['DFCI_MRN'].nunique()}")
    print(f"Patients with selected notes: {candidate_df['DFCI_MRN'].nunique()}")
    print(f"Selected notes: {len(candidate_df)}")


if __name__ == "__main__":
    main()
