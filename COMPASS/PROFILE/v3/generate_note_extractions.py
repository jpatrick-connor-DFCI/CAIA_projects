import argparse
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    TIKTOKEN_IMPORT_ERROR = None
except ImportError as error:
    tiktoken = None
    TIKTOKEN_IMPORT_ERROR = error

CURRENT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = CURRENT_DIR.parent
if str(PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILE_DIR))

from utils import clean_note  # noqa: E402

from common import load_selected_mrns, normalize_mrn_column, parse_datetime_series, to_iso_date  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_AZURE_OPENAI_API_VERSION,
    DEFAULT_AZURE_OPENAI_ENDPOINT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
)
from evidence_filters import sanitize_note_extractions  # noqa: E402
from prompts import (  # noqa: E402
    BUNDLED_EVENT_EXTRACTION_SYSTEM_PROMPT,
    CLINICAL_SAFETY_CONTEXT,
    EVENT_EXTRACTION_SYSTEM_PROMPT,
)

SCHEMA_VERSION = "v3_unified_phenotype_2026-04-14"


def parse_args():
    parser = argparse.ArgumentParser(description="Run v3 note-level extractions.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument("--context-path", type=Path, default=None)
    parser.add_argument("--mrns", default=None, help="Comma-separated DFCI_MRN values to include.")
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=None,
        help="Optional text/CSV/TSV file containing DFCI_MRN values.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit-mrns", type=int, default=None)
    parser.add_argument("--bundle-max-tokens", type=int, default=7000)
    parser.add_argument("--bundle-max-notes", type=int, default=6)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def build_client():
    if AZURE_IMPORT_ERROR is not None:
        raise ImportError(
            "generate_note_extractions.py requires azure-identity and openai in the active environment."
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


def build_token_encoder(model_name):
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
            "source_row": row,
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


def empty_note_extraction(note_payload):
    return {
        "note_date": note_payload.get("note_date"),
        "note_type": note_payload.get("note_type"),
        "nepc_scpc_mentions": [],
        "transformation_mentions": [],
        "aggressive_variant_mentions": [],
        "avpc_criteria_mentions": [],
        "biomarker_mentions": [],
        "treatment_resistance_mentions": [],
        "platinum_mentions": [],
        "overall_relevance": "low",
    }


def normalize_bundle_extractions(bundle_response, note_bundle):
    if isinstance(bundle_response, dict) and "note_extractions" in bundle_response:
        bundle_response = bundle_response["note_extractions"]

    note_defaults = {note["note_index"]: empty_note_extraction(note) for note in note_bundle}
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


def extract_note_bundle(client, model_name, max_retries, bundled_extraction_system_message, note_bundle):
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
    return normalize_bundle_extractions(extraction, note_bundle), None


def remove_existing_result_files(paths):
    for path in paths:
        path.unlink(missing_ok=True)


def ensure_resume_compatible(extractions_path):
    if not extractions_path.exists() or extractions_path.stat().st_size == 0:
        return
    with open(extractions_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Existing v3 note extractions checkpoint is not a JSON object.")
    for value in raw.values():
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Existing v3 note extractions were created by an older schema. "
                "Re-run with --overwrite-existing or use a fresh --output-dir."
            )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)

    candidate_path = args.candidate_path or args.output_dir / "LLM_v3_candidate_text_data.csv"
    context_path = args.context_path or args.output_dir / "LLM_v3_patient_context.csv"
    extractions_path = args.output_dir / "LLM_v3_note_extractions.json"
    failures_path = args.output_dir / "LLM_v3_failed_patients.tsv"

    if args.overwrite_existing:
        remove_existing_result_files([extractions_path, failures_path])
    else:
        ensure_resume_compatible(extractions_path)

    candidate_df = normalize_mrn_column(pd.read_csv(candidate_path, low_memory=False)) if candidate_path.exists() else pd.DataFrame()
    context_df = normalize_mrn_column(pd.read_csv(context_path, low_memory=False))
    if "DFCI_MRN" not in context_df.columns:
        raise ValueError(f"Context file is missing DFCI_MRN: {context_path}")

    if selected_mrns is not None:
        context_df = context_df.loc[context_df["DFCI_MRN"].isin(selected_mrns)].copy()
        if not candidate_df.empty:
            candidate_df = candidate_df.loc[candidate_df["DFCI_MRN"].isin(selected_mrns)].copy()
        if context_df.empty:
            raise ValueError("No patients remained after applying the requested MRN filter.")

    if "EVENT_DATE" in candidate_df.columns:
        candidate_df["EVENT_DATE"] = parse_datetime_series(candidate_df["EVENT_DATE"])

    unique_mrns = context_df["DFCI_MRN"].dropna().astype(int).unique().tolist()
    if args.limit_mrns is not None:
        unique_mrns = unique_mrns[: args.limit_mrns]

    extractions_by_mrn = {}
    if extractions_path.exists():
        with open(extractions_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        extractions_by_mrn = {int(key): value for key, value in raw.items()}

    if args.retry_failures and failures_path.exists():
        failed_df = pd.read_csv(failures_path, sep="\t")
        retry_mrns = set(failed_df["DFCI_MRN"].dropna().astype(int).unique())
        for mrn in retry_mrns:
            extractions_by_mrn.pop(mrn, None)
        with open(extractions_path, "w", encoding="utf-8") as handle:
            json.dump({str(key): value for key, value in extractions_by_mrn.items()}, handle)
        failures_path.unlink(missing_ok=True)
        remaining_mrns = [mrn for mrn in unique_mrns if mrn in retry_mrns]
        print(f"RETRY MODE: re-processing {len(remaining_mrns)} previously failed patients\n")
    else:
        remaining_mrns = [mrn for mrn in unique_mrns if mrn not in extractions_by_mrn]
        print(f"Processing {len(remaining_mrns)} v3 extraction patients ({len(extractions_by_mrn)} done)\n")

    candidate_groups = (
        {int(mrn): group.sort_values("EVENT_DATE") for mrn, group in candidate_df.groupby("DFCI_MRN")}
        if not candidate_df.empty
        else {}
    )

    client = build_client()
    token_encoder = build_token_encoder(args.model)
    bundled_extraction_system_message = BUNDLED_EVENT_EXTRACTION_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT
    single_extraction_system_message = EVENT_EXTRACTION_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT

    for mrn in remaining_mrns:
        mrn_df = candidate_groups.get(mrn, pd.DataFrame())
        note_extractions = []
        try:
            if not mrn_df.empty:
                cleaned_notes = build_cleaned_note_payloads(mrn_df, token_encoder)
                note_bundles = bundle_cleaned_notes(
                    cleaned_notes,
                    bundle_max_tokens=args.bundle_max_tokens,
                    bundle_max_notes=args.bundle_max_notes,
                )
                max_workers = max(1, min(args.max_workers, len(note_bundles)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            extract_note_bundle,
                            client,
                            args.model,
                            args.max_retries,
                            bundled_extraction_system_message,
                            bundle,
                        ): bundle
                        for bundle in note_bundles
                    }
                    for future in as_completed(futures):
                        submitted_bundle = futures[future]
                        try:
                            extraction, error_type = future.result()
                        except json.JSONDecodeError as error:
                            print(f"    JSON parse failed for bundled extraction of {mrn}: {error}")
                            continue
                        except Exception as error:  # noqa: BLE001
                            print(f"    Note extraction failed unexpectedly for {mrn}: {error}")
                            continue

                        if error_type:
                            if "content_filter" in error_type:
                                print(f"    Bundle filtered for {mrn} ({len(submitted_bundle)} notes)")
                                continue
                            print(f"    Note extraction failed for {mrn}: {error_type}")
                            continue
                        if extraction:
                            note_extractions.extend(extraction if isinstance(extraction, list) else [extraction])

            note_extractions = sanitize_note_extractions(note_extractions)
            note_extractions = sorted(
                note_extractions,
                key=lambda item: (item.get("note_date") is None, item.get("note_date")),
            )
            extractions_by_mrn[mrn] = {
                "schema_version": SCHEMA_VERSION,
                "num_candidate_notes": int(len(mrn_df)),
                "note_extractions": note_extractions,
            }
            with open(extractions_path, "w", encoding="utf-8") as handle:
                json.dump({str(key): value for key, value in extractions_by_mrn.items()}, handle)
        except Exception as error:  # noqa: BLE001
            print(f"  Extraction failed for {mrn}: {error}")
            log_failure(failures_path, mrn, f"extraction: {str(error)[:200]}", len(mrn_df), "extraction")
            continue

    print(f"\nCompleted v3 note extraction for {len(extractions_by_mrn)} patients")


if __name__ == "__main__":
    main()
