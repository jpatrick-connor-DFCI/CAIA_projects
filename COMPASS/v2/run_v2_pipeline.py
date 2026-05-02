import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full PROFILE v2 extraction pipeline.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--text-source", choices=["compiled", "raw"], default=None)
    parser.add_argument("--raw-text-path", action="append", default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--mrn-file", default=None)
    parser.add_argument("--platinum-window-days", type=int, default=None)
    parser.add_argument("--max-clinician-notes", type=int, default=None)
    parser.add_argument("--max-imaging-notes", type=int, default=None)
    parser.add_argument("--max-pathology-notes", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--limit-mrns", type=int, default=None)
    parser.add_argument("--bundle-max-tokens", type=int, default=None)
    parser.add_argument("--bundle-max-notes", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
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
    output_path = output_dir / "LLM_v2_generated_labels.tsv"
    extractions_path = output_dir / "LLM_v2_note_extractions.json"
    failures_path = output_dir / "LLM_v2_failed_patients.tsv"

    labels_count = 0
    extraction_count = 0
    failure_count = 0

    if output_path.exists() and output_path.stat().st_size > 0:
        labels_df = pd.read_csv(output_path, sep="\t")
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
        "labels_path": output_path.exists(),
        "labels_count": labels_count,
        "extractions_path": extractions_path.exists(),
        "extractions_count": extraction_count,
        "failures_path": failures_path.exists(),
        "failures_count": failure_count,
    }


def main():
    args = parse_args()
    if args.prepare_only and args.label_only:
        raise ValueError("Use at most one of --prepare-only or --label-only.")

    prepare_cmd = [sys.executable, str(CURRENT_DIR / "prepare_event_candidates.py")]
    label_cmd = [sys.executable, str(CURRENT_DIR / "generate_event_labels.py")]

    for command in (prepare_cmd, label_cmd):
        append_optional_arg(command, "--data-path", args.data_path)
        append_optional_arg(command, "--output-dir", args.output_dir)
        append_optional_arg(command, "--mrns", args.mrns)
        append_optional_arg(command, "--mrn-file", args.mrn_file)

    append_optional_arg(prepare_cmd, "--text-source", args.text_source)
    append_optional_args(prepare_cmd, "--raw-text-path", args.raw_text_path)
    append_optional_arg(prepare_cmd, "--platinum-window-days", args.platinum_window_days)
    append_optional_arg(prepare_cmd, "--max-clinician-notes", args.max_clinician_notes)
    append_optional_arg(prepare_cmd, "--max-imaging-notes", args.max_imaging_notes)
    append_optional_arg(prepare_cmd, "--max-pathology-notes", args.max_pathology_notes)

    append_optional_arg(label_cmd, "--model", args.model)
    append_optional_arg(label_cmd, "--max-workers", args.max_workers)
    append_optional_arg(label_cmd, "--limit-mrns", args.limit_mrns)
    append_optional_arg(label_cmd, "--bundle-max-tokens", args.bundle_max_tokens)
    append_optional_arg(label_cmd, "--bundle-max-notes", args.bundle_max_notes)
    if args.retry_failures:
        label_cmd.append("--retry-failures")
    if args.overwrite_existing:
        label_cmd.append("--overwrite-existing")

    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is not None:
        summary = summarize_output_dir(output_dir)
        if summary["labels_path"] or summary["extractions_path"] or summary["failures_path"]:
            mode_label = "overwrite" if args.overwrite_existing else "resume"
            print(
                "Existing output-dir state:",
                f"mode={mode_label}",
                f"labels_mrns={summary['labels_count']}",
                f"checkpoint_mrns={summary['extractions_count']}",
                f"failed_mrns={summary['failures_count']}",
            )

    if not args.label_only:
        run_command(prepare_cmd)
    if not args.prepare_only:
        run_command(label_cmd)


if __name__ == "__main__":
    main()
