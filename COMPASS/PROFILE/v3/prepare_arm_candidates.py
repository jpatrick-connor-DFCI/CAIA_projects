import argparse
from pathlib import Path

import pandas as pd

from arm_registry import load_arm_module
from candidate_utils import annotate_inventory_notes
from common import load_selected_mrns, normalize_mrn_column
from settings import DEFAULT_OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Build arm-specific v3 candidate note snippets.")
    parser.add_argument("--arm", required=True, choices=["nepc", "avpc", "biomarker"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inventory-path", type=Path, default=None)
    parser.add_argument("--mrns", default=None, help="Comma-separated DFCI_MRN values to include.")
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=None,
        help="Optional text/CSV/TSV file containing DFCI_MRN values.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    arm_module = load_arm_module(args.arm)
    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    inventory_path = args.inventory_path or args.output_dir / "LLM_v3_note_inventory.csv"

    inventory_df = normalize_mrn_column(pd.read_csv(inventory_path, low_memory=False))
    if selected_mrns is not None:
        inventory_df = inventory_df.loc[inventory_df["DFCI_MRN"].isin(selected_mrns)].copy()

    candidate_df = annotate_inventory_notes(inventory_df, arm_module)
    output_path = args.output_dir / f"LLM_v3_{arm_module.ARM_NAME}_candidate_text_data.csv"
    candidate_df.to_csv(output_path, index=False)

    print(f"Wrote {arm_module.ARM_NAME} candidate notes: {output_path}")
    print(f"Patients with selected notes: {candidate_df['DFCI_MRN'].nunique() if not candidate_df.empty else 0}")
    print(f"Selected notes: {len(candidate_df)}")


if __name__ == "__main__":
    main()
