import json
import re
from pathlib import Path

import pandas as pd


def safe_read_csv(path, **kwargs):
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(csv_path, **kwargs)


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


def parse_mrn_values(values):
    mrns = set()
    for value in values:
        if pd.isna(value):
            continue
        tokens = re.split(r"[\s,|]+", str(value).strip())
        for token in tokens:
            if not token:
                continue
            mrn = pd.to_numeric(token, errors="coerce")
            if pd.notna(mrn):
                mrns.add(int(mrn))
    return mrns


def load_selected_mrns(mrns_arg=None, mrn_file=None):
    selected = set()
    if mrns_arg:
        selected.update(parse_mrn_values([mrns_arg]))

    if mrn_file:
        mrn_file = Path(mrn_file)
        suffix = mrn_file.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            mrn_df = pd.read_csv(mrn_file, sep=sep, low_memory=False)
            if "DFCI_MRN" in mrn_df.columns:
                selected.update(parse_mrn_values(mrn_df["DFCI_MRN"]))
            elif not mrn_df.empty:
                selected.update(parse_mrn_values(mrn_df.iloc[:, 0]))
        else:
            with open(mrn_file, "r", encoding="utf-8") as handle:
                selected.update(parse_mrn_values(handle.readlines()))

    return selected or None


def to_iso_date(value):
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def build_structured_context(row_like):
    context = {}
    for key, value in dict(row_like).items():
        if pd.isna(value):
            continue
        if isinstance(value, pd.Timestamp):
            context[key] = value.strftime("%Y-%m-%d")
        elif isinstance(value, bool):
            context[key] = bool(value)
        elif isinstance(value, (int, float, str)):
            context[key] = value
        else:
            context[key] = str(value)
    return context


def serialize_list_fields(result_row, list_fields):
    for field in list_fields:
        if isinstance(result_row.get(field), list):
            result_row[field] = " | ".join(str(item) for item in result_row[field])
    return result_row


def basic_clean_text(text):
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def deduplicate_texts(text_entries):
    seen = set()
    deduped = []
    for entry in text_entries:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def infer_note_type_from_filename(path):
    name = Path(path).name.lower()
    if "imaging" in name:
        return "Imaging"
    if "prognote" in name or "progress" in name or "clinic" in name:
        return "Clinician"
    if "pathology" in name or re.search(r"(^|[-_])path(?:[-_.]|$)", name):
        return "Pathology"
    return None


def discover_raw_text_files(raw_text_paths):
    discovered = []
    seen_files = set()
    for raw_text_path in raw_text_paths:
        raw_text_path = Path(raw_text_path)
        if not raw_text_path.exists():
            continue
        for path in sorted(raw_text_path.rglob("*.json")):
            note_type = infer_note_type_from_filename(path)
            path_key = str(path)
            if note_type is not None and path_key not in seen_files:
                seen_files.add(path_key)
                discovered.append((path, note_type))
    return discovered


def extract_raw_docs(payload):
    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, dict) and isinstance(response.get("docs"), list):
            return response["docs"]
        if isinstance(payload.get("docs"), list):
            return payload["docs"]
    if isinstance(payload, list):
        return payload
    return []


def build_raw_note_row(note, note_type, source_file):
    mrn = pd.to_numeric(note.get("DFCI_MRN"), errors="coerce")
    if pd.isna(mrn):
        return None

    text_entries = [value for key, value in note.items() if "TEXT" in str(key).upper()]
    text_to_save = basic_clean_text(" ".join(deduplicate_texts(text_entries)))
    if not text_to_save:
        return None

    event_date = note.get("EVENT_DATE") or note.get("RPT_DATE")
    return {
        "DFCI_MRN": int(mrn),
        "EVENT_DATE": event_date,
        "NOTE_TYPE": note_type,
        "CLINICAL_TEXT": text_to_save,
        "RAW_SOURCE_FILE": Path(source_file).name,
        "RAW_NOTE_ID": note.get("id"),
        "RPT_DATE": note.get("RPT_DATE"),
        "RPT_TYPE": note.get("RPT_TYPE"),
        "SOURCE_STR": note.get("SOURCE_STR"),
        "PROC_DESC_STR": note.get("PROC_DESC_STR"),
        "ENCOUNTER_TYPE_DESC_STR": note.get("ENCOUNTER_TYPE_DESC_STR"),
    }


def load_raw_text_notes(raw_text_paths, selected_mrns):
    if selected_mrns is None:
        raise ValueError("Raw text mode requires --mrns or --mrn-file.")

    raw_files = discover_raw_text_files(raw_text_paths)
    if not raw_files:
        joined_paths = ", ".join(str(path) for path in raw_text_paths)
        raise FileNotFoundError(f"No supported raw JSON note files were found under: {joined_paths}")

    rows = []
    for file_path, note_type in raw_files:
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        docs = extract_raw_docs(payload)

        for note in docs:
            mrn = pd.to_numeric(note.get("DFCI_MRN"), errors="coerce")
            if pd.isna(mrn) or int(mrn) not in selected_mrns:
                continue
            row = build_raw_note_row(note, note_type, file_path)
            if row is not None:
                rows.append(row)

    raw_df = normalize_mrn_column(pd.DataFrame(rows))
    if raw_df.empty:
        raise ValueError("No raw notes were found for the requested MRNs.")
    return raw_df


def load_note_text_dataframe(text_source, data_path, raw_text_paths, selected_mrns):
    data_path = Path(data_path)
    text_path = data_path / "prostate_text_data.csv"
    if text_source == "raw":
        return load_raw_text_notes(raw_text_paths, selected_mrns)

    text_df = normalize_mrn_column(safe_read_csv(text_path))
    if text_df.empty:
        raise FileNotFoundError(f"Could not load note data from {text_path}")

    if selected_mrns is not None:
        text_df = text_df.loc[text_df["DFCI_MRN"].isin(selected_mrns)].copy()
        if text_df.empty:
            raise ValueError("No notes remained after applying the requested MRN filter.")
    return text_df
