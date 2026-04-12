import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

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

from utils import clean_note

from config import (
    DEFAULT_AZURE_OPENAI_API_VERSION,
    DEFAULT_AZURE_OPENAI_ENDPOINT,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
)
from prompts import (
    BUNDLED_EVENT_EXTRACTION_SYSTEM_PROMPT,
    CLINICAL_SAFETY_CONTEXT,
    EVENT_EXTRACTION_SYSTEM_PROMPT,
    PATIENT_SYNTHESIS_SYSTEM_PROMPT,
)

SCHEMA_VERSION = "v2_ne_scpc_simplified_2026-04-12"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the v2 neuroendocrine/small cell prostate cancer extraction pipeline."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument("--context-path", type=Path, default=None)
    parser.add_argument(
        "--mrns",
        default=None,
        help="Comma-separated DFCI_MRN values to include.",
    )
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
    parser.add_argument(
        "--bundle-max-tokens",
        type=int,
        default=None,
        help="If set, group cleaned notes into extraction bundles up to this approximate token limit.",
    )
    parser.add_argument(
        "--bundle-max-notes",
        type=int,
        default=None,
        help="Optional cap on notes per extraction bundle.",
    )
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Ignore and replace existing label/checkpoint files in the output directory.",
    )
    return parser.parse_args()


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


def build_client():
    if AZURE_IMPORT_ERROR is not None:
        raise ImportError(
            "generate_event_labels.py requires azure-identity and openai in the active environment."
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
            wait = 2 ** attempt * 5
            time.sleep(wait)
        except APITimeoutError:
            wait = 2 ** attempt * 3
            time.sleep(wait)
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
        [
            {
                "DFCI_MRN": int(mrn),
                "error_type": error_type,
                "stage": stage,
                "num_notes": num_notes,
            }
        ]
    )
    fail_row.to_csv(
        path,
        mode="a",
        sep="\t",
        index=False,
        header=not path.exists() or path.stat().st_size == 0,
    )


def to_iso_date(value):
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def build_structured_context(context_row):
    context = {}
    for key, value in context_row.items():
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


def extract_note(client, model_name, max_retries, extraction_system_message, row):
    cleaned = clean_note(row["CLINICAL_TEXT"], note_type=row.get("NOTE_TYPE"))
    if not cleaned:
        return None, None

    user_content = (
        f"Note Type: {row.get('NOTE_TYPE', 'Unknown')}\n"
        f"Note Date: {to_iso_date(row.get('EVENT_DATE')) or row.get('EVENT_DATE')}\n\n"
        f"{cleaned}"
    )

    response_text, error_type = call_with_retry(
        client,
        model_name,
        [
            {"role": "system", "content": extraction_system_message},
            {"role": "user", "content": user_content},
        ],
        max_retries,
    )
    if error_type:
        return None, error_type

    extraction = parse_json_response(response_text)
    if isinstance(extraction, dict):
        extraction.setdefault("note_date", to_iso_date(row.get("EVENT_DATE")))
        extraction.setdefault("note_type", row.get("NOTE_TYPE", "Unknown"))
    return extraction, None


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
        "histology_mentions": [],
        "transformation_mentions": [],
        "overall_relevance": "low",
    }


def normalize_bundle_extractions(bundle_response, note_bundle):
    if isinstance(bundle_response, dict) and "note_extractions" in bundle_response:
        bundle_response = bundle_response["note_extractions"]

    note_defaults = {
        note["note_index"]: empty_note_extraction(note)
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


def extract_note_bundle(
    client,
    model_name,
    max_retries,
    bundled_extraction_system_message,
    note_bundle,
):
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


def default_patient_result(mrn, structured_context, num_notes_reviewed):
    return {
        "schema_version": SCHEMA_VERSION,
        "DFCI_MRN": int(mrn),
        "neuroendocrine_small_cell_prostate_cancer": None,
        "disease_type": None,
        "transformation_evidence": None,
        "transformation_date": None,
        "transformation_date_confidence": None,
        "supporting_quotes": [],
        "supporting_quote_dates": [],
        "confidence": "low",
        "num_notes_reviewed": num_notes_reviewed,
        "num_note_extractions": 0,
    }


def merge_patient_result(base_row, model_row):
    if not isinstance(model_row, dict):
        return base_row
    merged = base_row.copy()
    merged.update(model_row)
    merged["schema_version"] = SCHEMA_VERSION
    for field in ("supporting_quotes", "supporting_quote_dates"):
        if merged.get(field) is None:
            merged[field] = []
    return merged


def serialize_list_fields(result_row):
    list_fields = [
        "supporting_quotes",
        "supporting_quote_dates",
    ]
    for field in list_fields:
        if isinstance(result_row.get(field), list):
            result_row[field] = " | ".join(str(item) for item in result_row[field])
    return result_row


def remove_existing_result_files(paths):
    for path in paths:
        path.unlink(missing_ok=True)


def ensure_resume_compatible(output_path, extractions_path):
    incompatible_files = []

    if output_path.exists() and output_path.stat().st_size > 0:
        existing_df = pd.read_csv(output_path, sep="\t")
        if "schema_version" not in existing_df.columns:
            incompatible_files.append(output_path.name)
        else:
            versions = set(existing_df["schema_version"].dropna().astype(str).unique())
            if versions and versions != {SCHEMA_VERSION}:
                incompatible_files.append(output_path.name)

    if extractions_path.exists() and extractions_path.stat().st_size > 0:
        with open(extractions_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            incompatible_files.append(extractions_path.name)
        else:
            for value in raw.values():
                if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
                    incompatible_files.append(extractions_path.name)
                    break

    if incompatible_files:
        joined = ", ".join(sorted(set(incompatible_files)))
        raise ValueError(
            "Existing v2 outputs were created by an older schema "
            f"({joined}). Re-run with --overwrite-existing or use a fresh --output-dir."
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)

    candidate_path = args.candidate_path or args.output_dir / "LLM_v2_candidate_text_data.csv"
    context_path = args.context_path or args.output_dir / "LLM_v2_patient_context.csv"
    output_path = args.output_dir / "LLM_v2_generated_labels.tsv"
    extractions_path = args.output_dir / "LLM_v2_note_extractions.json"
    failures_path = args.output_dir / "LLM_v2_failed_patients.tsv"

    if args.overwrite_existing:
        remove_existing_result_files([output_path, extractions_path, failures_path])
    else:
        ensure_resume_compatible(output_path, extractions_path)

    candidate_df = normalize_mrn_column(pd.read_csv(candidate_path)) if candidate_path.exists() else pd.DataFrame()
    context_df = normalize_mrn_column(pd.read_csv(context_path))
    if "DFCI_MRN" not in context_df.columns:
        raise ValueError(f"Context file is missing DFCI_MRN: {context_path}")

    if selected_mrns is not None:
        context_df = context_df.loc[context_df["DFCI_MRN"].isin(selected_mrns)].copy()
        if not candidate_df.empty:
            candidate_df = candidate_df.loc[candidate_df["DFCI_MRN"].isin(selected_mrns)].copy()
        if context_df.empty:
            raise ValueError("No patients remained after applying the requested MRN filter.")

    for date_col in [
        "EVENT_DATE",
        "FIRST_PLATINUM_DATE",
        "FIRST_ADT_DATE",
        "FIRST_ARSI_DATE",
        "FIRST_PARP_DATE",
        "LATEST_PSA_DATE",
    ]:
        if date_col in candidate_df.columns:
            candidate_df[date_col] = parse_datetime_series(candidate_df[date_col])
        if date_col in context_df.columns:
            context_df[date_col] = parse_datetime_series(context_df[date_col])

    unique_mrns = context_df["DFCI_MRN"].dropna().astype(int).unique().tolist()
    if args.limit_mrns is not None:
        unique_mrns = unique_mrns[: args.limit_mrns]

    extractions_by_mrn = {}
    if extractions_path.exists():
        with open(extractions_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        extractions_by_mrn = {int(key): value for key, value in raw.items()}

    completed_mrns = set()
    if output_path.exists():
        existing_df = pd.read_csv(output_path, sep="\t")
        completed_mrns = set(existing_df["DFCI_MRN"].dropna().astype(int).unique())

    if args.overwrite_existing:
        print("OVERWRITE MODE: ignoring existing labels, checkpoints, and failures in output_dir\n")

    if args.retry_failures:
        if not failures_path.exists():
            print("No failures file found, nothing to retry.")
            return
        failed_df = pd.read_csv(failures_path, sep="\t")
        retry_mrns = set(failed_df["DFCI_MRN"].dropna().astype(int).unique())
        for mrn in retry_mrns:
            extractions_by_mrn.pop(mrn, None)
        with open(extractions_path, "w", encoding="utf-8") as handle:
            json.dump({str(key): value for key, value in extractions_by_mrn.items()}, handle)
        if output_path.exists():
            existing_df = existing_df.loc[~existing_df["DFCI_MRN"].isin(retry_mrns)]
            existing_df.to_csv(output_path, sep="\t", index=False)
            completed_mrns -= retry_mrns
        failures_path.unlink(missing_ok=True)
        remaining_mrns = [mrn for mrn in unique_mrns if mrn in retry_mrns]
        print(f"RETRY MODE: re-processing {len(remaining_mrns)} previously failed patients\n")
    else:
        remaining_mrns = [mrn for mrn in unique_mrns if mrn not in completed_mrns]
        print(f"Processing {len(remaining_mrns)} v2 patients ({len(completed_mrns)} done)\n")

    context_lookup = context_df.set_index("DFCI_MRN").to_dict(orient="index")
    candidate_groups = (
        {int(mrn): group.sort_values("EVENT_DATE") for mrn, group in candidate_df.groupby("DFCI_MRN")}
        if not candidate_df.empty
        else {}
    )

    client = build_client()
    token_encoder = build_token_encoder(args.model)
    extraction_system_message = EVENT_EXTRACTION_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT
    bundled_extraction_system_message = BUNDLED_EVENT_EXTRACTION_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT
    synthesis_system_message = PATIENT_SYNTHESIS_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT

    for mrn in tqdm(remaining_mrns, desc="V2 patients"):
        context_row = context_lookup.get(mrn, {})
        structured_context = build_structured_context(context_row)
        mrn_df = candidate_groups.get(mrn, pd.DataFrame())

        if mrn in extractions_by_mrn:
            note_extractions = extractions_by_mrn[mrn].get("note_extractions", [])
        else:
            note_extractions = []
            if not mrn_df.empty:
                cleaned_notes = build_cleaned_note_payloads(mrn_df, token_encoder)
                note_bundles = bundle_cleaned_notes(
                    cleaned_notes,
                    bundle_max_tokens=args.bundle_max_tokens,
                    bundle_max_notes=args.bundle_max_notes,
                )
                max_workers = max(1, min(args.max_workers, len(note_bundles)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    if args.bundle_max_tokens is not None or args.bundle_max_notes is not None:
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
                    else:
                        futures = {
                            executor.submit(
                                extract_note,
                                client,
                                args.model,
                                args.max_retries,
                                extraction_system_message,
                                bundle[0]["source_row"],
                            ): bundle[0]
                            for bundle in note_bundles
                        }
                    for future in as_completed(futures):
                        submitted_item = futures[future]
                        try:
                            extraction, error_type = future.result()
                        except json.JSONDecodeError as error:
                            if isinstance(submitted_item, list):
                                bundle_dates = [note.get("note_date") for note in submitted_item]
                                print(f"    JSON parse failed for bundled extraction of {mrn}: {bundle_dates} ({error})")
                            else:
                                print(
                                    f"    JSON parse failed for note {submitted_item.get('EVENT_DATE')} "
                                    f"of {mrn}: {error}"
                                )
                            continue
                        except Exception as error:  # noqa: BLE001
                            print(f"    Note extraction failed unexpectedly for {mrn}: {error}")
                            continue

                        if error_type:
                            if "content_filter" in error_type:
                                if isinstance(submitted_item, list):
                                    print(f"    Bundle filtered for {mrn} ({len(submitted_item)} notes)")
                                else:
                                    print(
                                        f"    Note filtered for {mrn} "
                                        f"({submitted_item.get('NOTE_TYPE')} {to_iso_date(submitted_item.get('EVENT_DATE'))})"
                                    )
                                continue
                            print(f"    Note extraction failed for {mrn}: {error_type}")
                            continue

                        if extraction:
                            if isinstance(extraction, list):
                                note_extractions.extend(extraction)
                            else:
                                note_extractions.append(extraction)

            note_extractions = sorted(
                note_extractions,
                key=lambda item: (item.get("note_date") is None, item.get("note_date")),
            )
            extractions_by_mrn[mrn] = {
                "schema_version": SCHEMA_VERSION,
                "structured_context": structured_context,
                "note_extractions": note_extractions,
            }
            with open(extractions_path, "w", encoding="utf-8") as handle:
                json.dump({str(key): value for key, value in extractions_by_mrn.items()}, handle)

        synthesis_payload = {
            "structured_context": structured_context,
            "note_extractions": note_extractions,
        }

        if not note_extractions:
            result_row = default_patient_result(mrn, structured_context, num_notes_reviewed=0)
        else:
            response_text, error_type = call_with_retry(
                client,
                args.model,
                [
                    {"role": "system", "content": synthesis_system_message},
                    {"role": "user", "content": json.dumps(synthesis_payload)},
                ],
                args.max_retries,
            )
            if error_type:
                print(f"  Synthesis failed for {mrn}: {error_type}")
                log_failure(failures_path, mrn, error_type, len(note_extractions), "synthesis")
                if not note_extractions:
                    result_row = default_patient_result(
                        mrn,
                        structured_context,
                        num_notes_reviewed=len(mrn_df),
                    )
                else:
                    continue
            else:
                try:
                    parsed_row = parse_json_response(response_text)
                    result_row = merge_patient_result(
                        default_patient_result(
                            mrn,
                            structured_context,
                            num_notes_reviewed=len(mrn_df),
                        ),
                        parsed_row,
                    )
                except json.JSONDecodeError as error:
                    print(f"  Synthesis JSON parse failed for {mrn}: {error}")
                    log_failure(
                        failures_path,
                        mrn,
                        f"json_parse: {str(error)[:200]}",
                        len(note_extractions),
                        "synthesis",
                    )
                    if not note_extractions:
                        result_row = default_patient_result(
                            mrn,
                            structured_context,
                            num_notes_reviewed=len(mrn_df),
                        )
                    else:
                        continue

        result_row["DFCI_MRN"] = int(mrn)
        result_row["num_notes_reviewed"] = int(len(mrn_df))
        result_row["num_note_extractions"] = int(len(note_extractions))
        result_row = serialize_list_fields(result_row)

        row_df = pd.DataFrame([result_row])
        row_df.to_csv(
            output_path,
            mode="a",
            sep="\t",
            index=False,
            header=not output_path.exists() or output_path.stat().st_size == 0,
        )

    n_success = len(pd.read_csv(output_path, sep="\t")) if output_path.exists() else 0
    print(f"\nCompleted v2 synthesis for {n_success} patients")


if __name__ == "__main__":
    main()
