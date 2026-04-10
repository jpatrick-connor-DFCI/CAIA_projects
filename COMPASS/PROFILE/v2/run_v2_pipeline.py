import argparse
import subprocess
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full PROFILE v2 extraction pipeline.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--mrn-file", default=None)
    parser.add_argument("--platinum-window-days", type=int, default=None)
    parser.add_argument("--max-clinician-notes", type=int, default=None)
    parser.add_argument("--max-imaging-notes", type=int, default=None)
    parser.add_argument("--max-pathology-notes", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--limit-mrns", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--label-only", action="store_true")
    return parser.parse_args()


def append_optional_arg(command, name, value):
    if value is None:
        return
    command.extend([name, str(value)])


def run_command(command):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


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

    append_optional_arg(prepare_cmd, "--platinum-window-days", args.platinum_window_days)
    append_optional_arg(prepare_cmd, "--max-clinician-notes", args.max_clinician_notes)
    append_optional_arg(prepare_cmd, "--max-imaging-notes", args.max_imaging_notes)
    append_optional_arg(prepare_cmd, "--max-pathology-notes", args.max_pathology_notes)

    append_optional_arg(label_cmd, "--model", args.model)
    append_optional_arg(label_cmd, "--max-workers", args.max_workers)
    append_optional_arg(label_cmd, "--limit-mrns", args.limit_mrns)
    if args.retry_failures:
        label_cmd.append("--retry-failures")

    if not args.label_only:
        run_command(prepare_cmd)
    if not args.prepare_only:
        run_command(label_cmd)


if __name__ == "__main__":
    main()
