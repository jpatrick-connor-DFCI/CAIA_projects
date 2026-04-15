import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from arm_registry import load_arm_module
from common import build_structured_context, load_selected_mrns, normalize_mrn_column, parse_datetime_series, serialize_list_fields
from llm_runtime import (
    build_cleaned_note_payloads,
    build_client,
    build_token_encoder,
    bundle_cleaned_notes,
    call_with_retry,
    ensure_resume_compatible,
    extract_note_bundle,
    log_failure,
    parse_json_response,
    remove_existing_result_files,
)
from prompt_common import CLINICAL_SAFETY_CONTEXT
from settings import DEFAULT_MODEL_NAME, DEFAULT_OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Run arm-specific v3 extraction and synthesis.")
    parser.add_argument("--arm", required=True, choices=["nepc", "avpc", "biomarker"])
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


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arm_module = load_arm_module(args.arm)
    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)

    candidate_path = args.candidate_path or args.output_dir / f"LLM_v3_{arm_module.ARM_NAME}_candidate_text_data.csv"
    context_path = args.context_path or args.output_dir / "LLM_v3_patient_context.csv"
    output_path = args.output_dir / f"LLM_v3_{arm_module.ARM_NAME}_labels.tsv"
    extractions_path = args.output_dir / f"LLM_v3_{arm_module.ARM_NAME}_note_extractions.json"
    failures_path = args.output_dir / f"LLM_v3_{arm_module.ARM_NAME}_failed_patients.tsv"

    if args.overwrite_existing:
        remove_existing_result_files([output_path, extractions_path, failures_path])
    else:
        ensure_resume_compatible(output_path, extractions_path, arm_module.SCHEMA_VERSION)

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

    completed_mrns = set()
    if output_path.exists():
        existing_df = pd.read_csv(output_path, sep="\t")
        completed_mrns = set(existing_df["DFCI_MRN"].dropna().astype(int).unique())

    if args.retry_failures:
        if not failures_path.exists():
            print(f"No {arm_module.ARM_NAME} failures file found, nothing to retry.")
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
        print(f"RETRY MODE: re-processing {len(remaining_mrns)} {arm_module.ARM_NAME} patients\n")
    else:
        remaining_mrns = [mrn for mrn in unique_mrns if mrn not in completed_mrns]
        print(
            f"Processing {len(remaining_mrns)} {arm_module.ARM_NAME} patients "
            f"({len(completed_mrns)} done)\n"
        )

    context_lookup = context_df.set_index("DFCI_MRN").to_dict(orient="index")
    candidate_groups = (
        {int(mrn): group.sort_values("EVENT_DATE") for mrn, group in candidate_df.groupby("DFCI_MRN")}
        if not candidate_df.empty
        else {}
    )

    extraction_system_message = arm_module.BUNDLED_EVENT_EXTRACTION_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT
    synthesis_system_message = arm_module.PATIENT_SYNTHESIS_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT
    client = None
    token_encoder = None

    for mrn in remaining_mrns:
        context_row = context_lookup.get(mrn, {})
        structured_context = build_structured_context(context_row)
        mrn_df = candidate_groups.get(mrn, pd.DataFrame())

        if mrn in extractions_by_mrn:
            note_extractions = arm_module.sanitize_note_extractions(
                extractions_by_mrn[mrn].get("note_extractions", [])
            )
        else:
            note_extractions = []
            if not mrn_df.empty:
                if client is None:
                    client = build_client()
                if token_encoder is None:
                    token_encoder = build_token_encoder(args.model)
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
                            extraction_system_message,
                            bundle,
                            arm_module,
                        ): bundle
                        for bundle in note_bundles
                    }
                    for future in as_completed(futures):
                        submitted_bundle = futures[future]
                        try:
                            extraction, error_type = future.result()
                        except json.JSONDecodeError as error:
                            print(f"    JSON parse failed for {arm_module.ARM_NAME} bundle of {mrn}: {error}")
                            continue
                        except Exception as error:  # noqa: BLE001
                            print(f"    Extraction failed unexpectedly for {arm_module.ARM_NAME} {mrn}: {error}")
                            continue

                        if error_type:
                            if "content_filter" in error_type:
                                print(
                                    f"    Bundle filtered for {arm_module.ARM_NAME} {mrn} "
                                    f"({len(submitted_bundle)} notes)"
                                )
                                continue
                            print(f"    Extraction failed for {arm_module.ARM_NAME} {mrn}: {error_type}")
                            continue

                        if extraction:
                            note_extractions.extend(extraction if isinstance(extraction, list) else [extraction])

            note_extractions = arm_module.sanitize_note_extractions(note_extractions)
            note_extractions = sorted(
                note_extractions,
                key=lambda item: (item.get("note_date") is None, item.get("note_date")),
            )
            extractions_by_mrn[mrn] = {
                "schema_version": arm_module.SCHEMA_VERSION,
                "note_extractions": note_extractions,
            }
            with open(extractions_path, "w", encoding="utf-8") as handle:
                json.dump({str(key): value for key, value in extractions_by_mrn.items()}, handle)

        if not arm_module.has_substantive_evidence(note_extractions):
            result_row = arm_module.default_patient_result(mrn, num_notes_reviewed=len(mrn_df))
        else:
            if client is None:
                client = build_client()
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
                print(f"  Synthesis failed for {arm_module.ARM_NAME} {mrn}: {error_type}")
                log_failure(failures_path, mrn, error_type, len(note_extractions), "synthesis")
                continue
            try:
                parsed_row = parse_json_response(response_text)
                result_row = arm_module.merge_patient_result(
                    arm_module.default_patient_result(mrn, num_notes_reviewed=len(mrn_df)),
                    parsed_row,
                )
            except json.JSONDecodeError as error:
                print(f"  Synthesis JSON parse failed for {arm_module.ARM_NAME} {mrn}: {error}")
                log_failure(
                    failures_path,
                    mrn,
                    f"json_parse: {str(error)[:200]}",
                    len(note_extractions),
                    "synthesis",
                )
                continue

        result_row["DFCI_MRN"] = int(mrn)
        result_row["num_notes_reviewed"] = int(len(mrn_df))
        result_row["num_note_extractions"] = int(len(note_extractions))
        result_row = serialize_list_fields(result_row, arm_module.LIST_FIELDS)

        pd.DataFrame([result_row]).to_csv(
            output_path,
            mode="a",
            sep="\t",
            index=False,
            header=not output_path.exists() or output_path.stat().st_size == 0,
        )

    n_success = len(pd.read_csv(output_path, sep="\t")) if output_path.exists() else 0
    print(f"\nCompleted {arm_module.ARM_NAME} synthesis for {n_success} patients")


if __name__ == "__main__":
    main()
