import argparse
import subprocess
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single task-specific v3 arm.")
    parser.add_argument("--arm", required=True, choices=["nepc", "avpc", "biomarker"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--mrn-file", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--limit-mrns", type=int, default=None)
    parser.add_argument("--bundle-max-tokens", type=int, default=None)
    parser.add_argument("--bundle-max-notes", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
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
    candidate_cmd = [sys.executable, str(CURRENT_DIR / "prepare_arm_candidates.py"), "--arm", args.arm]
    label_cmd = [sys.executable, str(CURRENT_DIR / "generate_arm_labels.py"), "--arm", args.arm]

    for command in (candidate_cmd, label_cmd):
        append_optional_arg(command, "--output-dir", args.output_dir)
        append_optional_arg(command, "--mrns", args.mrns)
        append_optional_arg(command, "--mrn-file", args.mrn_file)

    append_optional_arg(label_cmd, "--model", args.model)
    append_optional_arg(label_cmd, "--max-workers", args.max_workers)
    append_optional_arg(label_cmd, "--limit-mrns", args.limit_mrns)
    append_optional_arg(label_cmd, "--bundle-max-tokens", args.bundle_max_tokens)
    append_optional_arg(label_cmd, "--bundle-max-notes", args.bundle_max_notes)
    if args.retry_failures:
        label_cmd.append("--retry-failures")
    if args.overwrite_existing:
        label_cmd.append("--overwrite-existing")

    run_command(candidate_cmd)
    run_command(label_cmd)


if __name__ == "__main__":
    main()
