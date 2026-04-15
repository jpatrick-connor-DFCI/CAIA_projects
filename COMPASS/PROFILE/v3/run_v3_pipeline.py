import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full PROFILE v3 extraction pipeline.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--text-source", choices=["compiled", "raw"], default=None)
    parser.add_argument("--raw-text-path", action="append", default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--mrn-file", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--limit-mrns", type=int, default=None)
    parser.add_argument("--bundle-max-tokens", type=int, default=None)
    parser.add_argument("--bundle-max-notes", type=int, default=None)
    parser.add_argument("--max-clinician-notes", type=int, default=None)
    parser.add_argument("--max-imaging-notes", type=int, default=None)
    parser.add_argument("--max-pathology-notes", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--label-only", action="store_true")
    return parser.parse_args()


def append_optional_arg(command, name, value):
    if value is None:
        return
    command.extend([name, str(value)])


def append_optional_args(command, name, values):
    if values is None:
        return
    for value in values:
        command.extend([name, str(value)])


def run_command(command):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def summarize_output_dir(output_dir):
    labels_path = output_dir / "LLM_v3_generated_labels.tsv"
    extractions_path = output_dir / "LLM_v3_note_extractions.json"
    failures_path = output_dir / "LLM_v3_failed_patients.tsv"

    labels_count = 0
    extraction_count = 0
    failure_count = 0

    if labels_path.exists() and labels_path.stat().st_size > 0:
        labels_df = pd.read_csv(labels_path, sep="\t")
        if "DFCI_MRN" in labels_df.columns:
            labels_count = labels_df["DFCI_MRN"].dropna().astype(int).nunique()

    if extractions_path.exists() and extractions_path.stat().st_size > 0:
        with open(extractions_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        extraction_count = len(raw)

    if failures_path.exists() and failures_path.stat().st_size > 0:
        failures_df = pd.read_csv(failures_path, sep="\t")
        if "DFCI_MRN" in failures_df.columns:
            failure_count = failures_df["DFCI_MRN"].dropna().astype(int).nunique()

    return {
        "labels_count": labels_count,
        "extractions_count": extraction_count,
        "failures_count": failure_count,
    }


def main():
    args = parse_args()
    chosen_modes = sum([args.prepare_only, args.extract_only, args.label_only])
    if chosen_modes > 1:
        raise ValueError("Use at most one of --prepare-only, --extract-only, or --label-only.")

    context_cmd = [sys.executable, str(CURRENT_DIR / "prepare_patient_context.py")]
    candidate_cmd = [sys.executable, str(CURRENT_DIR / "prepare_candidate_notes.py")]
    extraction_cmd = [sys.executable, str(CURRENT_DIR / "generate_note_extractions.py")]
    label_cmd = [sys.executable, str(CURRENT_DIR / "generate_patient_labels.py")]

    for command in (context_cmd, candidate_cmd):
        append_optional_arg(command, "--data-path", args.data_path)
    for command in (context_cmd, candidate_cmd, extraction_cmd, label_cmd):
        append_optional_arg(command, "--output-dir", args.output_dir)
        append_optional_arg(command, "--mrns", args.mrns)
        append_optional_arg(command, "--mrn-file", args.mrn_file)

    append_optional_arg(context_cmd, "--text-source", args.text_source)
    append_optional_args(context_cmd, "--raw-text-path", args.raw_text_path)

    append_optional_arg(candidate_cmd, "--text-source", args.text_source)
    append_optional_args(candidate_cmd, "--raw-text-path", args.raw_text_path)
    append_optional_arg(candidate_cmd, "--max-clinician-notes", args.max_clinician_notes)
    append_optional_arg(candidate_cmd, "--max-imaging-notes", args.max_imaging_notes)
    append_optional_arg(candidate_cmd, "--max-pathology-notes", args.max_pathology_notes)

    append_optional_arg(extraction_cmd, "--model", args.model)
    append_optional_arg(extraction_cmd, "--max-workers", args.max_workers)
    append_optional_arg(extraction_cmd, "--limit-mrns", args.limit_mrns)
    append_optional_arg(extraction_cmd, "--bundle-max-tokens", args.bundle_max_tokens)
    append_optional_arg(extraction_cmd, "--bundle-max-notes", args.bundle_max_notes)

    append_optional_arg(label_cmd, "--model", args.model)
    append_optional_arg(label_cmd, "--limit-mrns", args.limit_mrns)

    if args.retry_failures:
        extraction_cmd.append("--retry-failures")
        label_cmd.append("--retry-failures")
    if args.overwrite_existing:
        extraction_cmd.append("--overwrite-existing")
        label_cmd.append("--overwrite-existing")

    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is not None and output_dir.exists():
        summary = summarize_output_dir(output_dir)
        print(
            "Existing output-dir state:",
            f"labels_mrns={summary['labels_count']}",
            f"checkpoint_mrns={summary['extractions_count']}",
            f"failed_mrns={summary['failures_count']}",
        )

    if args.prepare_only:
        run_command(context_cmd)
        run_command(candidate_cmd)
        return
    if args.extract_only:
        run_command(extraction_cmd)
        return
    if args.label_only:
        run_command(label_cmd)
        return

    run_command(context_cmd)
    run_command(candidate_cmd)
    run_command(extraction_cmd)
    run_command(label_cmd)


if __name__ == "__main__":
    main()
