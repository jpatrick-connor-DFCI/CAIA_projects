import argparse
import json
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

CURRENT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = CURRENT_DIR.parent
if str(PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILE_DIR))

from common import (  # noqa: E402
    build_structured_context,
    load_selected_mrns,
    normalize_mrn_column,
    parse_datetime_series,
    serialize_list_fields,
)
from config import (  # noqa: E402
    DEFAULT_AZURE_OPENAI_API_VERSION,
    DEFAULT_AZURE_OPENAI_ENDPOINT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
)
from evidence_filters import has_substantive_v3_evidence, sanitize_note_extractions  # noqa: E402
from prompts import CLINICAL_SAFETY_CONTEXT, PATIENT_SYNTHESIS_SYSTEM_PROMPT  # noqa: E402

SCHEMA_VERSION = "v3_unified_phenotype_2026-04-14"
LIST_FIELDS = [
    "aggressive_variant_criteria_present",
    "aggressive_variant_criteria_suspected",
    "aggressive_variant_basis",
    "secondary_platinum_factors",
    "supporting_quotes",
    "supporting_quote_dates",
    "supporting_note_types",
]


def infer_platinum_exposure_from_text(note_extractions):
    return any((extraction.get("platinum_mentions") or []) for extraction in note_extractions or [])


def infer_first_platinum_date_from_text(note_extractions):
    candidate_dates = []
    for extraction in note_extractions or []:
        for mention in extraction.get("platinum_mentions") or []:
            event_date = mention.get("event_date") or extraction.get("note_date")
            if event_date:
                candidate_dates.append(str(event_date))
    return sorted(candidate_dates)[0] if candidate_dates else None


def parse_args():
    parser = argparse.ArgumentParser(description="Run v3 patient-level synthesis.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-path", type=Path, default=None)
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument("--note-extractions-path", type=Path, default=None)
    parser.add_argument("--mrns", default=None, help="Comma-separated DFCI_MRN values to include.")
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=None,
        help="Optional text/CSV/TSV file containing DFCI_MRN values.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit-mrns", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def build_client():
    if AZURE_IMPORT_ERROR is not None:
        raise ImportError(
            "generate_patient_labels.py requires azure-identity and openai in the active environment."
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
            "Existing v3 outputs were created by an older schema "
            f"({joined}). Re-run with --overwrite-existing or use a fresh --output-dir."
        )


def default_patient_result(mrn, note_extractions, num_notes_reviewed):
    platinum_exposed = infer_platinum_exposure_from_text(note_extractions)
    return {
        "schema_version": SCHEMA_VERSION,
        "DFCI_MRN": int(mrn),
        "platinum_exposed": platinum_exposed,
        "first_platinum_date": infer_first_platinum_date_from_text(note_extractions),
        "nepc_scpc_status": "absent",
        "nepc_scpc_subtype": None,
        "nepc_scpc_evidence_level": "none",
        "nepc_scpc_date": None,
        "nepc_scpc_date_confidence": None,
        "aggressive_variant_status": "absent",
        "aggressive_variant_definition": "aparicio_2013",
        "aggressive_variant_criteria_present": [],
        "aggressive_variant_criteria_suspected": [],
        "aggressive_variant_basis": [],
        "aggressive_variant_date": None,
        "aggressive_variant_date_confidence": None,
        "conventional_prostate_cancer_status": "present",
        "dominant_disease_phenotype": "conventional_prostate_cancer",
        "transformation_status": "not_documented",
        "transformation_from": None,
        "transformation_to": None,
        "transformation_date": None,
        "transformation_date_confidence": None,
        "platinum_suggestive_phenotype": "none",
        "primary_platinum_indication": "unclear" if platinum_exposed else "not_applicable",
        "secondary_platinum_factors": [],
        "platinum_indication_detail": None,
        "platinum_indication_confidence": None,
        "supporting_quotes": [],
        "supporting_quote_dates": [],
        "supporting_note_types": [],
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
    merged["aggressive_variant_definition"] = merged.get("aggressive_variant_definition") or "aparicio_2013"
    for field in LIST_FIELDS:
        if merged.get(field) is None:
            merged[field] = []
    return merged


def should_run_synthesis(note_extractions):
    if has_substantive_v3_evidence(note_extractions):
        return True
    if infer_platinum_exposure_from_text(note_extractions):
        return True
    if any(extraction.get("overall_relevance") in {"high", "medium"} for extraction in note_extractions or []):
        return True
    return False


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)

    context_path = args.context_path or args.output_dir / "LLM_v3_patient_context.csv"
    candidate_path = args.candidate_path or args.output_dir / "LLM_v3_candidate_text_data.csv"
    extractions_path = args.note_extractions_path or args.output_dir / "LLM_v3_note_extractions.json"
    output_path = args.output_dir / "LLM_v3_generated_labels.tsv"
    failures_path = args.output_dir / "LLM_v3_failed_patients.tsv"

    if args.overwrite_existing:
        remove_existing_result_files([output_path, failures_path])
    else:
        ensure_resume_compatible(output_path, extractions_path)

    context_df = normalize_mrn_column(pd.read_csv(context_path, low_memory=False))
    candidate_df = normalize_mrn_column(pd.read_csv(candidate_path, low_memory=False)) if candidate_path.exists() else pd.DataFrame()
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

    with open(extractions_path, "r", encoding="utf-8") as handle:
        raw_extractions = json.load(handle)
    extractions_by_mrn = {int(key): value for key, value in raw_extractions.items()}

    completed_mrns = set()
    if output_path.exists():
        existing_df = pd.read_csv(output_path, sep="\t")
        completed_mrns = set(existing_df["DFCI_MRN"].dropna().astype(int).unique())

    if args.retry_failures:
        if not failures_path.exists():
            print("No failures file found, nothing to retry.")
            return
        failed_df = pd.read_csv(failures_path, sep="\t")
        retry_mrns = set(failed_df["DFCI_MRN"].dropna().astype(int).unique())
        if output_path.exists():
            existing_df = existing_df.loc[~existing_df["DFCI_MRN"].isin(retry_mrns)]
            existing_df.to_csv(output_path, sep="\t", index=False)
            completed_mrns -= retry_mrns
        failures_path.unlink(missing_ok=True)
        remaining_mrns = [mrn for mrn in unique_mrns if mrn in retry_mrns]
        print(f"RETRY MODE: re-processing {len(remaining_mrns)} previously failed patients\n")
    else:
        remaining_mrns = [mrn for mrn in unique_mrns if mrn not in completed_mrns]
        print(f"Processing {len(remaining_mrns)} v3 synthesis patients ({len(completed_mrns)} done)\n")

    context_lookup = context_df.set_index("DFCI_MRN").to_dict(orient="index")
    candidate_groups = (
        {int(mrn): group.sort_values("EVENT_DATE") for mrn, group in candidate_df.groupby("DFCI_MRN")}
        if not candidate_df.empty
        else {}
    )

    client = build_client()
    synthesis_system_message = PATIENT_SYNTHESIS_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT

    for mrn in remaining_mrns:
        context_row = context_lookup.get(mrn, {})
        structured_context = build_structured_context(context_row)
        mrn_df = candidate_groups.get(mrn, pd.DataFrame())
        extraction_entry = extractions_by_mrn.get(mrn, {})
        note_extractions = sanitize_note_extractions(extraction_entry.get("note_extractions", []))

        if not should_run_synthesis(note_extractions):
            result_row = default_patient_result(mrn, note_extractions, num_notes_reviewed=len(mrn_df))
        else:
            synthesis_payload = {
                "structured_context": structured_context,
                "note_extractions": note_extractions,
            }
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
                continue
            try:
                parsed_row = parse_json_response(response_text)
                result_row = merge_patient_result(
                    default_patient_result(mrn, note_extractions, num_notes_reviewed=len(mrn_df)),
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
                continue

        result_row["DFCI_MRN"] = int(mrn)
        result_row["platinum_exposed"] = infer_platinum_exposure_from_text(note_extractions)
        result_row["first_platinum_date"] = infer_first_platinum_date_from_text(note_extractions)
        result_row["num_notes_reviewed"] = int(len(mrn_df))
        result_row["num_note_extractions"] = int(len(note_extractions))
        result_row = serialize_list_fields(result_row, LIST_FIELDS)

        pd.DataFrame([result_row]).to_csv(
            output_path,
            mode="a",
            sep="\t",
            index=False,
            header=not output_path.exists() or output_path.stat().st_size == 0,
        )

    n_success = len(pd.read_csv(output_path, sep="\t")) if output_path.exists() else 0
    print(f"\nCompleted v3 synthesis for {n_success} patients")


if __name__ == "__main__":
    main()
