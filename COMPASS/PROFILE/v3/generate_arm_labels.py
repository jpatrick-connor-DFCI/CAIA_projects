import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from helpers import (
    CLINICAL_SAFETY_CONTEXT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    build_cleaned_note_payloads,
    build_client,
    build_structured_context,
    build_token_encoder,
    bundle_cleaned_notes,
    call_with_retry,
    ensure_resume_compatible,
    extract_note_bundle,
    load_arm_module,
    load_json_map,
    load_selected_mrns,
    log_failure,
    normalize_mrn_column,
    parse_json_response,
    parse_datetime_series,
    read_dataframe,
    remove_existing_result_files,
    save_json_map,
    serialize_list_fields,
)


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


def load_context_dataframe(context_path):
    context_df = normalize_mrn_column(read_dataframe(context_path, required=True, description="Context file"))
    if "DFCI_MRN" not in context_df.columns:
        raise ValueError(f"Context file is missing DFCI_MRN: {context_path}")
    return context_df


def load_candidate_dataframe(candidate_path):
    candidate_df = normalize_mrn_column(
        read_dataframe(candidate_path, required=True, description="Candidate file")
    )
    if "DFCI_MRN" not in candidate_df.columns:
        raise ValueError(f"Candidate file is missing DFCI_MRN: {candidate_path}")
    if "EVENT_DATE" in candidate_df.columns:
        candidate_df["EVENT_DATE"] = parse_datetime_series(candidate_df["EVENT_DATE"])
    return candidate_df


def filter_requested_mrns(context_df, candidate_df, selected_mrns):
    if selected_mrns is None:
        return context_df, candidate_df

    context_df = context_df.loc[context_df["DFCI_MRN"].isin(selected_mrns)].copy()
    candidate_df = candidate_df.loc[candidate_df["DFCI_MRN"].isin(selected_mrns)].copy()
    if context_df.empty:
        raise ValueError("No patients remained after applying the requested MRN filter.")
    return context_df, candidate_df


def load_extractions_by_mrn(extractions_path):
    raw = load_json_map(extractions_path)
    return {int(key): value for key, value in raw.items()}


def save_extractions_by_mrn(extractions_path, extractions_by_mrn):
    save_json_map(extractions_path, {str(key): value for key, value in extractions_by_mrn.items()})


def load_completed_mrns(output_path):
    existing_df = read_dataframe(output_path, sep="\t")
    if existing_df.empty or "DFCI_MRN" not in existing_df.columns:
        return set(), existing_df
    completed_mrns = set(existing_df["DFCI_MRN"].dropna().astype(int).unique())
    return completed_mrns, existing_df


def normalize_note_extractions(arm_module, note_extractions):
    sanitized = arm_module.sanitize_note_extractions(note_extractions)
    return sorted(
        sanitized,
        key=lambda item: (item.get("note_date") is None, item.get("note_date")),
    )


def summarize_error_types(error_types, max_items=3):
    unique_errors = []
    seen = set()
    for error_type in error_types:
        if error_type and error_type not in seen:
            seen.add(error_type)
            unique_errors.append(error_type)
    if len(unique_errors) <= max_items:
        return "; ".join(unique_errors)
    return "; ".join(unique_errors[:max_items]) + f"; +{len(unique_errors) - max_items} more"


def extract_patient_note_extractions(
    mrn,
    mrn_df,
    *,
    arm_module,
    get_client,
    get_token_encoder,
    model,
    max_retries,
    max_workers,
    bundle_max_tokens,
    bundle_max_notes,
    extraction_system_message,
):
    if mrn_df.empty:
        return [], None

    cleaned_notes = build_cleaned_note_payloads(mrn_df, get_token_encoder())
    note_bundles = bundle_cleaned_notes(
        cleaned_notes,
        bundle_max_tokens=bundle_max_tokens,
        bundle_max_notes=bundle_max_notes,
    )
    if not note_bundles:
        return [], None

    note_extractions = []
    bundle_errors = []
    worker_count = max(1, min(max_workers, len(note_bundles)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                extract_note_bundle,
                get_client(),
                model,
                max_retries,
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
                error_type = f"json_parse: {str(error)[:200]}"
            except Exception as error:  # noqa: BLE001
                error_type = f"unexpected: {type(error).__name__}: {str(error)[:200]}"

            if error_type:
                if "content_filter" in error_type:
                    print(
                        f"    Bundle filtered for {arm_module.ARM_NAME} {mrn} "
                        f"({len(submitted_bundle)} notes)"
                    )
                else:
                    print(f"    Extraction failed for {arm_module.ARM_NAME} {mrn}: {error_type}")
                bundle_errors.append(error_type)
                continue

            if extraction:
                note_extractions.extend(extraction if isinstance(extraction, list) else [extraction])

    if bundle_errors:
        return None, summarize_error_types(bundle_errors)
    return normalize_note_extractions(arm_module, note_extractions), None


def synthesize_patient_result(
    mrn,
    *,
    mrn_df,
    note_extractions,
    structured_context,
    arm_module,
    get_client,
    model,
    max_retries,
    synthesis_system_message,
):
    response_text, error_type = call_with_retry(
        get_client(),
        model,
        [
            {"role": "system", "content": synthesis_system_message},
            {"role": "user", "content": json.dumps(
                {
                    "structured_context": structured_context,
                    "note_extractions": note_extractions,
                }
            )},
        ],
        max_retries,
    )
    if error_type:
        return None, error_type

    try:
        parsed_row = parse_json_response(response_text)
    except json.JSONDecodeError as error:
        return None, f"json_parse: {str(error)[:200]}"

    return (
        arm_module.merge_patient_result(
            arm_module.default_patient_result(mrn, num_notes_reviewed=len(mrn_df)),
            parsed_row,
            note_extractions=note_extractions,
        ),
        None,
    )


def append_result_row(output_path, result_row, list_fields):
    serializable_row = serialize_list_fields(result_row.copy(), list_fields)
    pd.DataFrame([serializable_row]).to_csv(
        output_path,
        mode="a",
        sep="\t",
        index=False,
        header=not output_path.exists() or output_path.stat().st_size == 0,
    )


def build_output_paths(args, arm_name):
    return {
        "candidate": args.candidate_path or args.output_dir / f"LLM_v3_{arm_name}_candidate_text_data.csv",
        "context": args.context_path or args.output_dir / "LLM_v3_patient_context.csv",
        "output": args.output_dir / f"LLM_v3_{arm_name}_labels.tsv",
        "extractions": args.output_dir / f"LLM_v3_{arm_name}_note_extractions.json",
        "failures": args.output_dir / f"LLM_v3_{arm_name}_failed_patients.tsv",
    }


def prepare_runtime_filters(args, arm_module, paths):
    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)

    candidate_df = load_candidate_dataframe(paths["candidate"])
    context_df = load_context_dataframe(paths["context"])
    context_df, candidate_df = filter_requested_mrns(context_df, candidate_df, selected_mrns)

    unique_mrns = context_df["DFCI_MRN"].dropna().astype(int).unique().tolist()
    if args.limit_mrns is not None:
        unique_mrns = unique_mrns[: args.limit_mrns]

    extractions_by_mrn = load_extractions_by_mrn(paths["extractions"])
    completed_mrns, existing_df = load_completed_mrns(paths["output"])

    if args.retry_failures:
        if not paths["failures"].exists():
            print(f"No {arm_module.ARM_NAME} failures file found, nothing to retry.")
            return None
        failed_df = read_dataframe(paths["failures"], sep="\t")
        retry_mrns = set(failed_df["DFCI_MRN"].dropna().astype(int).unique())
        for mrn in retry_mrns:
            extractions_by_mrn.pop(mrn, None)
        save_extractions_by_mrn(paths["extractions"], extractions_by_mrn)
        if not existing_df.empty:
            existing_df = existing_df.loc[~existing_df["DFCI_MRN"].isin(retry_mrns)]
            existing_df.to_csv(paths["output"], sep="\t", index=False)
        paths["failures"].unlink(missing_ok=True)
        remaining_mrns = [mrn for mrn in unique_mrns if mrn in retry_mrns]
        print(f"RETRY MODE: re-processing {len(remaining_mrns)} {arm_module.ARM_NAME} patients\n")
    else:
        remaining_mrns = [mrn for mrn in unique_mrns if mrn not in completed_mrns]
        print(
            f"Processing {len(remaining_mrns)} {arm_module.ARM_NAME} patients "
            f"({len(completed_mrns)} done)\n"
        )

    return candidate_df, context_df, extractions_by_mrn, remaining_mrns


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arm_module = load_arm_module(args.arm)
    paths = build_output_paths(args, arm_module.ARM_NAME)

    if args.overwrite_existing:
        remove_existing_result_files([paths["output"], paths["extractions"], paths["failures"]])
    else:
        ensure_resume_compatible(paths["output"], paths["extractions"], arm_module.SCHEMA_VERSION)

    runtime_state = prepare_runtime_filters(args, arm_module, paths)
    if runtime_state is None:
        return
    candidate_df, context_df, extractions_by_mrn, remaining_mrns = runtime_state

    context_lookup = context_df.set_index("DFCI_MRN").to_dict(orient="index")
    candidate_groups = (
        {int(mrn): group.sort_values("EVENT_DATE") for mrn, group in candidate_df.groupby("DFCI_MRN")}
        if not candidate_df.empty
        else {}
    )

    extraction_system_message = arm_module.BUNDLED_EVENT_EXTRACTION_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT
    synthesis_system_message = arm_module.PATIENT_SYNTHESIS_SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT
    runtime_cache = {"client": None, "token_encoder": None}

    def get_client():
        if runtime_cache["client"] is None:
            runtime_cache["client"] = build_client()
        return runtime_cache["client"]

    def get_token_encoder():
        if runtime_cache["token_encoder"] is None:
            runtime_cache["token_encoder"] = build_token_encoder(args.model)
        return runtime_cache["token_encoder"]

    for mrn in remaining_mrns:
        context_row = context_lookup.get(mrn, {})
        structured_context = build_structured_context(context_row)
        mrn_df = candidate_groups.get(mrn, pd.DataFrame())

        if mrn in extractions_by_mrn:
            note_extractions = normalize_note_extractions(
                arm_module,
                extractions_by_mrn[mrn].get("note_extractions", []),
            )
        else:
            note_extractions, extraction_error = extract_patient_note_extractions(
                mrn,
                mrn_df,
                arm_module=arm_module,
                get_client=get_client,
                get_token_encoder=get_token_encoder,
                model=args.model,
                max_retries=args.max_retries,
                max_workers=args.max_workers,
                bundle_max_tokens=args.bundle_max_tokens,
                bundle_max_notes=args.bundle_max_notes,
                extraction_system_message=extraction_system_message,
            )
            if extraction_error:
                log_failure(paths["failures"], mrn, extraction_error, len(mrn_df), "extraction")
                continue

            extractions_by_mrn[mrn] = {
                "schema_version": arm_module.SCHEMA_VERSION,
                "note_extractions": note_extractions,
            }
            save_extractions_by_mrn(paths["extractions"], extractions_by_mrn)

        if not arm_module.has_substantive_evidence(note_extractions):
            result_row = arm_module.default_patient_result(mrn, num_notes_reviewed=len(mrn_df))
        else:
            result_row, synthesis_error = synthesize_patient_result(
                mrn,
                mrn_df=mrn_df,
                note_extractions=note_extractions,
                structured_context=structured_context,
                arm_module=arm_module,
                get_client=get_client,
                model=args.model,
                max_retries=args.max_retries,
                synthesis_system_message=synthesis_system_message,
            )
            if synthesis_error:
                print(f"  Synthesis failed for {arm_module.ARM_NAME} {mrn}: {synthesis_error}")
                log_failure(paths["failures"], mrn, synthesis_error, len(note_extractions), "synthesis")
                continue

        result_row["DFCI_MRN"] = int(mrn)
        result_row["num_notes_reviewed"] = int(len(mrn_df))
        result_row["num_note_extractions"] = int(len(note_extractions))
        append_result_row(paths["output"], result_row, arm_module.LIST_FIELDS)

    n_success = len(read_dataframe(paths["output"], sep="\t"))
    print(f"\nCompleted {arm_module.ARM_NAME} synthesis for {n_success} patients")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
