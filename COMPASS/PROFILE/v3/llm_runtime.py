import json
import math
import re
import sys
import time
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

CURRENT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = CURRENT_DIR.parent
if str(PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILE_DIR))

from utils import clean_note  # noqa: E402

from common import to_iso_date  # noqa: E402
from settings import (  # noqa: E402
    DEFAULT_AZURE_OPENAI_API_VERSION,
    DEFAULT_AZURE_OPENAI_ENDPOINT,
    DEFAULT_MODEL_NAME,
)


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

    if isinstance(bundle_response, list):
        for item in bundle_response:
            if not isinstance(item, dict):
                continue
            note_index = pd.to_numeric(item.get("note_index"), errors="coerce")
            if pd.isna(note_index):
                continue
            note_index = int(note_index)
            if note_index not in normalized:
                continue
            merged = normalized[note_index].copy()
            merged.update(item)
            merged["note_date"] = merged.get("note_date") or note_defaults[note_index]["note_date"]
            merged["note_type"] = merged.get("note_type") or note_defaults[note_index]["note_type"]
            normalized[note_index] = merged

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
