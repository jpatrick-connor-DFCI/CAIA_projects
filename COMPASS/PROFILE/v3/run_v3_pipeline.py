import argparse
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from helpers import ARM_NAMES, DEFAULT_DATA_PATH, DEFAULT_MODEL_NAME, DEFAULT_OUTPUT_DIR
from generate_arm_labels import run as run_generate_arm_labels
from merge_labels import run as run_merge_labels
from prepare_arm_candidates import run as run_prepare_arm_candidates
from prepare_note_inventory import run as run_prepare_note_inventory


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fresh multi-arm PROFILE v3 pipeline.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-source", choices=["compiled", "raw", "bundle"], default="raw")
    parser.add_argument("--raw-text-path", type=Path, action="append", default=None)
    parser.add_argument("--note-bundle-path", type=Path, default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--mrn-file", type=Path, default=None)
    parser.add_argument("--arm", choices=ARM_NAMES, default=None)
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


def build_arm_args(args, arm_name):
    return Namespace(
        arm=arm_name,
        output_dir=args.output_dir,
        mrns=args.mrns,
        mrn_file=args.mrn_file,
        model=args.model,
        max_workers=args.max_workers,
        limit_mrns=args.limit_mrns,
        bundle_max_tokens=args.bundle_max_tokens,
        bundle_max_notes=args.bundle_max_notes,
        retry_failures=args.retry_failures,
        overwrite_existing=args.overwrite_existing,
    )


def run_arm(arm_args):
    run_prepare_arm_candidates(
        Namespace(
            arm=arm_args.arm,
            output_dir=arm_args.output_dir,
            inventory_path=None,
            mrns=arm_args.mrns,
            mrn_file=arm_args.mrn_file,
        )
    )
    run_generate_arm_labels(
        Namespace(
            arm=arm_args.arm,
            output_dir=arm_args.output_dir,
            candidate_path=None,
            context_path=None,
            mrns=arm_args.mrns,
            mrn_file=arm_args.mrn_file,
            model=arm_args.model or DEFAULT_MODEL_NAME,
            max_retries=3,
            max_workers=arm_args.max_workers or 4,
            limit_mrns=arm_args.limit_mrns,
            bundle_max_tokens=arm_args.bundle_max_tokens or 7000,
            bundle_max_notes=arm_args.bundle_max_notes or 6,
            retry_failures=arm_args.retry_failures,
            overwrite_existing=arm_args.overwrite_existing,
        )
    )


def run(args):
    chosen_modes = sum([args.prepare_only, args.arms_only, args.merge_only])
    if chosen_modes > 1:
        raise ValueError("Use at most one of --prepare-only, --arms-only, or --merge-only.")

    selected_arms = [args.arm] if args.arm else list(ARM_NAMES)

    if args.prepare_only or (not args.arms_only and not args.merge_only):
        run_prepare_note_inventory(
            Namespace(
                data_path=args.data_path or DEFAULT_DATA_PATH,
                output_dir=args.output_dir,
                text_source=args.text_source,
                raw_text_path=args.raw_text_path,
                note_bundle_path=args.note_bundle_path,
                mrns=args.mrns,
                mrn_file=args.mrn_file,
            )
        )
        if args.prepare_only:
            return

    if not args.merge_only:
        with ThreadPoolExecutor(max_workers=len(selected_arms)) as executor:
            futures = {
                executor.submit(run_arm, build_arm_args(args, arm_name)): arm_name
                for arm_name in selected_arms
            }
            for future in as_completed(futures):
                arm_name = futures[future]
                future.result()
                print(f"Finished arm: {arm_name}")

    if not args.arms_only:
        run_merge_labels(Namespace(output_dir=args.output_dir, context_path=None))


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
