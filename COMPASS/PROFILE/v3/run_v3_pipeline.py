import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from settings import ARM_NAMES, DEFAULT_OUTPUT_DIR


CURRENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fresh multi-arm PROFILE v3 pipeline.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--text-source", choices=["compiled", "raw"], default="raw")
    parser.add_argument("--raw-text-path", action="append", default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--mrn-file", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--limit-mrns", type=int, default=None)
    parser.add_argument("--bundle-max-tokens", type=int, default=None)
    parser.add_argument("--bundle-max-notes", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--arms-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
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


def run_arm_subprocess(command):
    run_command(command)
    return "ok"


def main():
    args = parse_args()
    chosen_modes = sum([args.prepare_only, args.arms_only, args.merge_only])
    if chosen_modes > 1:
        raise ValueError("Use at most one of --prepare-only, --arms-only, or --merge-only.")

    inventory_cmd = [sys.executable, str(CURRENT_DIR / "prepare_note_inventory.py")]
    merge_cmd = [sys.executable, str(CURRENT_DIR / "merge_labels.py")]

    for command in (inventory_cmd, merge_cmd):
        append_optional_arg(command, "--output-dir", args.output_dir)

    append_optional_arg(inventory_cmd, "--data-path", args.data_path)
    append_optional_arg(inventory_cmd, "--text-source", args.text_source)
    append_optional_args(inventory_cmd, "--raw-text-path", args.raw_text_path)
    append_optional_arg(inventory_cmd, "--mrns", args.mrns)
    append_optional_arg(inventory_cmd, "--mrn-file", args.mrn_file)

    arm_commands = []
    for arm_name in ARM_NAMES:
        arm_cmd = [sys.executable, str(CURRENT_DIR / "run_arm_pipeline.py"), "--arm", arm_name]
        append_optional_arg(arm_cmd, "--output-dir", args.output_dir)
        append_optional_arg(arm_cmd, "--mrns", args.mrns)
        append_optional_arg(arm_cmd, "--mrn-file", args.mrn_file)
        append_optional_arg(arm_cmd, "--model", args.model)
        append_optional_arg(arm_cmd, "--max-workers", args.max_workers)
        append_optional_arg(arm_cmd, "--limit-mrns", args.limit_mrns)
        append_optional_arg(arm_cmd, "--bundle-max-tokens", args.bundle_max_tokens)
        append_optional_arg(arm_cmd, "--bundle-max-notes", args.bundle_max_notes)
        if args.retry_failures:
            arm_cmd.append("--retry-failures")
        if args.overwrite_existing:
            arm_cmd.append("--overwrite-existing")
        arm_commands.append((arm_name, arm_cmd))

    if args.prepare_only:
        run_command(inventory_cmd)
        return
    if not args.arms_only and not args.merge_only:
        run_command(inventory_cmd)

    if not args.merge_only:
        with ThreadPoolExecutor(max_workers=len(arm_commands)) as executor:
            futures = {
                executor.submit(run_arm_subprocess, arm_cmd): arm_name
                for arm_name, arm_cmd in arm_commands
            }
            for future in as_completed(futures):
                arm_name = futures[future]
                future.result()
                print(f"Finished arm: {arm_name}")

    run_command(merge_cmd)


if __name__ == "__main__":
    main()
