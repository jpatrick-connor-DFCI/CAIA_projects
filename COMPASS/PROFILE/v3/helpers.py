import gzip
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import APIError, APITimeoutError, AzureOpenAI, RateLimitError

    AZURE_IMPORT_ERROR = None
except ImportError as error:
    DefaultAzureCredential = None
    get_bearer_token_provider = None
    AzureOpenAI = None
    APIError = Exception
    APITimeoutError = Exception
    RateLimitError = Exception
    AZURE_IMPORT_ERROR = error

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print(
        "WARNING: tiktoken not installed — token estimates will use a rough word-count "
        "heuristic (1.35x multiplier). Bundle token limits may be exceeded."
    )


CURRENT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = CURRENT_DIR.parent
if str(PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILE_DIR))

from utils import clean_note  # noqa: E402


# Config
DEFAULT_DATA_PATH = Path(
    os.environ.get("CAIA_COMPASS_DATA_PATH", "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("CAIA_COMPASS_V3_OUTPUT_DIR", str(DEFAULT_DATA_PATH / "v3_outputs"))
)

_raw_text_paths_env = os.environ.get("CAIA_ONCDRS_RAW_TEXT_PATHS")
_legacy_raw_text_path_env = os.environ.get("CAIA_ONCDRS_RAW_TEXT_PATH")
if _raw_text_paths_env:
    DEFAULT_RAW_TEXT_PATHS = tuple(
        Path(path_str) for path_str in _raw_text_paths_env.split(os.pathsep) if path_str.strip()
    )
elif _legacy_raw_text_path_env:
    DEFAULT_RAW_TEXT_PATHS = (Path(_legacy_raw_text_path_env),)
else:
    DEFAULT_RAW_TEXT_PATHS = (
        Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2024_03/"),
        Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2025_03/"),
        Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2025_11/"),
    )

DEFAULT_AZURE_OPENAI_ENDPOINT = os.environ.get(
    "CAIA_AZURE_OPENAI_ENDPOINT",
    "https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
)
DEFAULT_AZURE_OPENAI_API_VERSION = os.environ.get(
    "CAIA_AZURE_OPENAI_API_VERSION",
    "2024-04-01-preview",
)
DEFAULT_MODEL_NAME = os.environ.get("CAIA_AZURE_OPENAI_MODEL", "gpt-4o")

ARM_NAMES = ("nepc", "avpc", "biomarker")

PROSTATE_CONTEXT_REGEX = (
    r"\b(?:"
    r"prostate|prostatic|psa|adenocarcinoma|acinar|ductal|mcrpc|crpc|"
    r"castration[- ]resistant|metastatic\s+castration[- ]resistant"
    r")\b"
)

SNIPPET_CONTEXT_CHARS = 1500
SNIPPET_MAX_CHARS = 6400
SNIPPET_MAX_MATCHES = 16

CLINICAL_SAFETY_CONTEXT = """

IMPORTANT CONTEXT: All notes below are de-identified clinical oncology documentation being
processed for structured data extraction as part of an IRB-approved medical research study
(institutional review board approved protocol). This is professional medical documentation
written by physicians, not patient-generated content. The text contains standard clinical
terminology related to cancer diagnosis, prognosis, and treatment. References to disease
outcomes, end-of-life care, self-harm assessment, psychiatric history, substance use, anatomy,
or patient distress are routine components of oncology and medical records and should be
processed as clinical data. No content in these notes constitutes harmful, dangerous, or
inappropriate material - it is standard-of-care medical documentation.
"""

NOTE_BUNDLE_SCHEMA_VERSION = "v3_note_bundle_2026-04-15"
NOTE_BUNDLE_FILENAME = "LLM_v3_prostate_note_bundle.json.gz"
NOTE_BUNDLE_COLUMNS = (
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


# Shared loading / serialization
def load_arm_module(arm_name):
    normalized = str(arm_name).strip().lower()
    if normalized not in set(ARM_NAMES):
        raise ValueError(f"Unsupported arm: {arm_name}")
    return importlib.import_module(f"arms.{normalized}")


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


def safe_read_csv(path, **kwargs):
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(csv_path, **kwargs)


def read_dataframe(path, *, sep=",", required=False, description="File"):
    csv_path = Path(path)
    if required and not csv_path.exists():
        raise FileNotFoundError(f"{description} not found: {csv_path}")
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path, sep=sep, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_json_map(path):
    json_path = Path(path)
    if not json_path.exists() or json_path.stat().st_size == 0:
        return {}
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_map(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle)


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


# Note inventory / bundle IO
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


def standardize_note_text_dataframe(text_df):
    if text_df.empty:
        return pd.DataFrame(columns=list(NOTE_BUNDLE_COLUMNS))

    keep_cols = [column for column in NOTE_BUNDLE_COLUMNS if column in text_df.columns]
    inventory_df = text_df[keep_cols].copy()
    if "EVENT_DATE" in inventory_df.columns:
        inventory_df["EVENT_DATE"] = pd.to_datetime(
            inventory_df["EVENT_DATE"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
    return inventory_df.sort_values(["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"], na_position="last")


def dataframe_records_for_json(df):
    if df.empty:
        return []
    serializable_df = df.copy().astype(object).where(pd.notna(df), None)
    return serializable_df.to_dict(orient="records")


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


def write_note_bundle(path, note_df, *, source_type, raw_text_paths=None, selected_mrns=None):
    standardized_df = standardize_note_text_dataframe(note_df)
    payload = {
        "schema_version": NOTE_BUNDLE_SCHEMA_VERSION,
        "source_type": source_type,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requested_mrn_count": len(selected_mrns) if selected_mrns is not None else None,
        "patient_count": int(standardized_df["DFCI_MRN"].nunique()) if not standardized_df.empty else 0,
        "note_count": int(len(standardized_df)),
        "raw_text_paths": [str(path_item) for path_item in raw_text_paths] if raw_text_paths else None,
        "notes": dataframe_records_for_json(standardized_df),
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def load_note_bundle(path, selected_mrns=None):
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Could not load compiled note bundle from {bundle_path}")

    with gzip.open(bundle_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        records = payload.get("notes", [])
    elif isinstance(payload, list):
        records = payload
    else:
        records = []

    bundle_df = normalize_mrn_column(pd.DataFrame(records))
    if bundle_df.empty:
        raise ValueError(f"No note rows were found in compiled note bundle: {bundle_path}")

    if selected_mrns is not None:
        bundle_df = bundle_df.loc[bundle_df["DFCI_MRN"].isin(selected_mrns)].copy()
        if bundle_df.empty:
            raise ValueError("No notes remained after applying the requested MRN filter.")
    return bundle_df


def load_note_text_dataframe(text_source, data_path, raw_text_paths, selected_mrns, note_bundle_path=None):
    data_path = Path(data_path)
    text_path = data_path / "prostate_text_data.csv"
    if text_source == "raw":
        return load_raw_text_notes(raw_text_paths, selected_mrns)
    if text_source == "bundle":
        if note_bundle_path is None:
            raise ValueError("Bundle mode requires a --note-bundle-path.")
        return load_note_bundle(note_bundle_path, selected_mrns)

    text_df = normalize_mrn_column(safe_read_csv(text_path))
    if text_df.empty:
        raise FileNotFoundError(f"Could not load note data from {text_path}")

    if selected_mrns is not None:
        text_df = text_df.loc[text_df["DFCI_MRN"].isin(selected_mrns)].copy()
        if text_df.empty:
            raise ValueError("No notes remained after applying the requested MRN filter.")
    return text_df


# Candidate snippet selection
def merge_windows(windows, gap_chars=80):
    if not windows:
        return []
    ordered = sorted(windows)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap_chars:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_snippet_text(
    text,
    matches,
    *,
    context_chars=SNIPPET_CONTEXT_CHARS,
    max_chars=SNIPPET_MAX_CHARS,
    max_matches=SNIPPET_MAX_MATCHES,
):
    if not matches:
        return ""

    windows = []
    for match in matches[:max_matches]:
        start = max(0, match["start"] - context_chars)
        end = min(len(text), match["end"] + context_chars)
        windows.append((start, end))

    snippets = []
    for start, end in merge_windows(windows):
        snippet = text[start:end].strip()
        if not snippet:
            continue
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        snippets.append(snippet)

    snippet_text = "\n\n...\n\n".join(snippets)
    if len(snippet_text) > max_chars:
        snippet_text = snippet_text[: max_chars - 3].rstrip() + "..."
    return snippet_text


def _trigger_flag_col(label):
    return f"HAS_TRIGGER_{str(label).upper()}"


def extract_trigger_metadata(text, note_type, arm_module):
    cleaned_text = basic_clean_text(text)
    trigger_regex = arm_module.TRIGGER_REGEX
    if not cleaned_text:
        metadata = {
            "HAS_TRIGGER_HIT": False,
            "HAS_PROSTATE_CONTEXT": False,
            "FOCUS_MATCH_COUNT": 0,
            "FOCUS_MATCH_LABELS": "",
            "FOCUS_SNIPPET_TEXT": "",
            "FOCUS_SNIPPET_CHAR_LEN": 0,
        }
        for label in trigger_regex:
            metadata[_trigger_flag_col(label)] = False
        return metadata

    matches = []
    for label, pattern in trigger_regex.items():
        for match in re.finditer(pattern, cleaned_text, flags=re.IGNORECASE):
            matches.append({"label": label, "start": match.start(), "end": match.end()})
    matches = sorted(matches, key=lambda item: (item["start"], item["end"]))

    match_labels = []
    for match in matches:
        if match["label"] not in match_labels:
            match_labels.append(match["label"])

    has_required_label = True
    required_labels = set(getattr(arm_module, "REQUIRED_TRIGGER_LABELS", set()))
    if required_labels:
        has_required_label = bool(required_labels.intersection(match_labels))

    context_regex = getattr(arm_module, "PROSTATE_CONTEXT_REGEX", None)
    require_context = bool(getattr(arm_module, "REQUIRE_PROSTATE_CONTEXT", False))
    has_prostate_context = bool(re.search(context_regex, cleaned_text, flags=re.IGNORECASE)) if context_regex else True
    bypass_note_types = set(getattr(arm_module, "ALLOW_WITHOUT_CONTEXT_NOTE_TYPES", set()))
    bypass_labels = set(getattr(arm_module, "ALLOW_WITHOUT_CONTEXT_LABELS", set()))
    context_ok = (
        not require_context
        or has_prostate_context
        or note_type in bypass_note_types
        or bool(bypass_labels.intersection(match_labels))
    )

    eligible = bool(matches) and has_required_label and context_ok
    snippet_text = build_snippet_text(cleaned_text, matches) if eligible else ""

    metadata = {
        "HAS_TRIGGER_HIT": bool(snippet_text),
        "HAS_PROSTATE_CONTEXT": has_prostate_context,
        "FOCUS_MATCH_COUNT": len(matches) if snippet_text else 0,
        "FOCUS_MATCH_LABELS": "|".join(match_labels) if snippet_text else "",
        "FOCUS_SNIPPET_TEXT": snippet_text,
        "FOCUS_SNIPPET_CHAR_LEN": len(snippet_text),
    }
    for label in trigger_regex:
        metadata[_trigger_flag_col(label)] = label in match_labels and bool(snippet_text)
    return metadata


def annotate_inventory_notes(note_df, arm_module):
    if note_df.empty:
        base_columns = [
            "DFCI_MRN",
            "EVENT_DATE",
            "NOTE_TYPE",
            "CLINICAL_TEXT",
            "FOCUS_MATCH_COUNT",
            "FOCUS_MATCH_LABELS",
            "FOCUS_SNIPPET_CHAR_LEN",
            "SELECTION_REASON",
        ]
        for label in arm_module.TRIGGER_REGEX:
            base_columns.append(_trigger_flag_col(label))
        return pd.DataFrame(columns=base_columns)

    work = note_df.copy()
    work["NOTE_TYPE"] = work["NOTE_TYPE"].fillna("Unknown")
    work["CLINICAL_TEXT"] = work["CLINICAL_TEXT"].fillna("").astype(str)
    work["EVENT_DATE"] = pd.to_datetime(work["EVENT_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")

    focus_metadata = pd.DataFrame(
        [
            extract_trigger_metadata(text, note_type, arm_module)
            for text, note_type in zip(work["CLINICAL_TEXT"], work["NOTE_TYPE"])
        ],
        index=work.index,
    )
    work = pd.concat([work, focus_metadata], axis=1)
    work = work.loc[work["HAS_TRIGGER_HIT"]].copy()
    work["ORIGINAL_NOTE_HASH"] = work["CLINICAL_TEXT"].map(
        lambda text: hashlib.md5(basic_clean_text(text).encode("utf-8", errors="ignore")).hexdigest()
    )
    work["CLINICAL_TEXT"] = work["FOCUS_SNIPPET_TEXT"]
    work["SELECTION_REASON"] = work["FOCUS_MATCH_LABELS"]
    work = work.drop_duplicates(subset=["DFCI_MRN", "NOTE_TYPE", "EVENT_DATE", "ORIGINAL_NOTE_HASH"])
    return work.sort_values(["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"], na_position="last")


# LLM runtime
def build_client():
    if AZURE_IMPORT_ERROR is not None:
        raise ImportError(
            "Azure note extraction requires azure-identity and openai in the active environment."
        ) from AZURE_IMPORT_ERROR
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        api_version=DEFAULT_AZURE_OPENAI_API_VERSION,
        azure_endpoint=DEFAULT_AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
    )


def build_token_encoder(model_name=DEFAULT_MODEL_NAME):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:  # noqa: BLE001
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001
            return None


def call_with_retry(client, model_name, messages, max_retries):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
            )
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "content_filter":
                return None, "content_filter_response"
            return response.choices[0].message.content.strip(), None
        except RateLimitError:
            time.sleep(2 ** attempt * 5)
        except APITimeoutError:
            time.sleep(2 ** attempt * 3)
        except APIError as error:
            error_body = str(error)
            if "content_filter" in error_body.lower() or "content_management" in error_body.lower():
                return None, "content_filter_input"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, f"api_error: {error_body[:200]}"
        except Exception as error:  # noqa: BLE001
            return None, f"unexpected: {type(error).__name__}: {str(error)[:200]}"
    return None, "max_retries_exceeded"


def parse_json_response(response_text):
    if response_text is None:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", response_text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def estimate_tokens(text, token_encoder=None):
    if token_encoder is not None:
        return len(token_encoder.encode(text))
    rough_words = len(re.findall(r"\S+", text))
    return max(1, math.ceil(rough_words * 1.35))


def build_cleaned_note_payloads(mrn_df, token_encoder=None):
    cleaned_notes = []
    for note_index, row in enumerate(mrn_df.to_dict(orient="records")):
        cleaned = clean_note(row["CLINICAL_TEXT"], note_type=row.get("NOTE_TYPE"))
        if not cleaned:
            continue

        payload = {
            "note_index": note_index,
            "note_date": to_iso_date(row.get("EVENT_DATE")),
            "note_type": row.get("NOTE_TYPE", "Unknown"),
            "note_text": cleaned,
        }
        token_payload = {
            "note_index": payload["note_index"],
            "note_date": payload["note_date"],
            "note_type": payload["note_type"],
            "note_text": payload["note_text"],
        }
        payload["estimated_tokens"] = estimate_tokens(json.dumps(token_payload, ensure_ascii=False), token_encoder)
        cleaned_notes.append(payload)
    return cleaned_notes


def bundle_cleaned_notes(cleaned_notes, bundle_max_tokens=None, bundle_max_notes=None):
    if not cleaned_notes:
        return []
    if bundle_max_tokens is None and bundle_max_notes is None:
        return [[note] for note in cleaned_notes]

    bundles = []
    current_bundle = []
    current_tokens = 0
    for note in cleaned_notes:
        note_tokens = note.get("estimated_tokens", 0)
        exceeds_token_limit = (
            bundle_max_tokens is not None
            and current_bundle
            and current_tokens + note_tokens > bundle_max_tokens
        )
        exceeds_note_limit = (
            bundle_max_notes is not None
            and current_bundle
            and len(current_bundle) >= bundle_max_notes
        )
        if exceeds_token_limit or exceeds_note_limit:
            bundles.append(current_bundle)
            current_bundle = []
            current_tokens = 0
        current_bundle.append(note)
        current_tokens += note_tokens

    if current_bundle:
        bundles.append(current_bundle)
    return bundles


def normalize_bundle_extractions(bundle_response, note_bundle, arm_module):
    if isinstance(bundle_response, dict) and "note_extractions" in bundle_response:
        bundle_response = bundle_response["note_extractions"]

    note_defaults = {
        note["note_index"]: arm_module.empty_note_extraction(note)
        for note in note_bundle
    }
    note_order = [note["note_index"] for note in note_bundle]
    normalized = {idx: value.copy() for idx, value in note_defaults.items()}
    returned_indices = set()
    unknown_indices = []
    non_dict_items = 0
    missing_index_items = 0

    if isinstance(bundle_response, list):
        for item in bundle_response:
            if not isinstance(item, dict):
                non_dict_items += 1
                continue
            note_index = pd.to_numeric(item.get("note_index"), errors="coerce")
            if pd.isna(note_index):
                missing_index_items += 1
                continue
            note_index = int(note_index)
            if note_index not in normalized:
                unknown_indices.append(note_index)
                continue
            returned_indices.add(note_index)
            merged = normalized[note_index].copy()
            merged.update(item)
            merged["note_date"] = merged.get("note_date") or note_defaults[note_index]["note_date"]
            merged["note_type"] = merged.get("note_type") or note_defaults[note_index]["note_type"]
            normalized[note_index] = merged

    missing_indices = [idx for idx in note_order if idx not in returned_indices]
    if unknown_indices or missing_indices or non_dict_items or missing_index_items:
        print(
            f"    [{getattr(arm_module, 'ARM_NAME', 'arm')}] bundle coverage warning: "
            f"bundle_size={len(note_bundle)} returned={len(returned_indices)} "
            f"missing={missing_indices} unknown={unknown_indices} "
            f"non_dict={non_dict_items} no_index={missing_index_items}"
        )

    ordered = []
    for note_index in note_order:
        item = normalized[note_index].copy()
        item.pop("note_index", None)
        ordered.append(item)
    return ordered


def extract_note_bundle(client, model_name, max_retries, bundled_extraction_system_message, note_bundle, arm_module):
    bundle_payload = [
        {
            "note_index": note["note_index"],
            "note_date": note["note_date"],
            "note_type": note["note_type"],
            "note_text": note["note_text"],
        }
        for note in note_bundle
    ]

    response_text, error_type = call_with_retry(
        client,
        model_name,
        [
            {"role": "system", "content": bundled_extraction_system_message},
            {"role": "user", "content": json.dumps(bundle_payload, ensure_ascii=False)},
        ],
        max_retries,
    )
    if error_type:
        return None, error_type

    extraction = parse_json_response(response_text)
    return normalize_bundle_extractions(extraction, note_bundle, arm_module), None


def log_failure(path, mrn, error_type, num_notes, stage):
    fail_row = pd.DataFrame(
        [{"DFCI_MRN": int(mrn), "error_type": error_type, "stage": stage, "num_notes": num_notes}]
    )
    fail_row.to_csv(
        path,
        mode="a",
        sep="\t",
        index=False,
        header=not path.exists() or path.stat().st_size == 0,
    )


def remove_existing_result_files(paths):
    for path in paths:
        path.unlink(missing_ok=True)


def ensure_resume_compatible(output_path, extractions_path, schema_version):
    incompatible_files = []

    if output_path.exists() and output_path.stat().st_size > 0:
        existing_df = pd.read_csv(output_path, sep="\t")
        if "schema_version" not in existing_df.columns:
            incompatible_files.append(output_path.name)
        else:
            versions = set(existing_df["schema_version"].dropna().astype(str).unique())
            if versions and versions != {schema_version}:
                incompatible_files.append(output_path.name)

    if extractions_path.exists() and extractions_path.stat().st_size > 0:
        with open(extractions_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            incompatible_files.append(extractions_path.name)
        else:
            for value in raw.values():
                if not isinstance(value, dict) or value.get("schema_version") != schema_version:
                    incompatible_files.append(extractions_path.name)
                    break

    if incompatible_files:
        joined = ", ".join(sorted(set(incompatible_files)))
        raise ValueError(
            "Existing outputs were created by an older schema "
            f"({joined}). Re-run with --overwrite-existing or use a fresh --output-dir."
        )
