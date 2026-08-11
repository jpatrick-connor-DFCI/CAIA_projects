"""COMPASS PROFILE wrapper for the shared Dynamic-DeepHit longitudinal runner."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent               # .../multivariate_longitudinal
SURVIVAL_DIR = THIS_DIR.parent                            # .../survival_analysis
SURVIVAL_PARENT = SURVIVAL_DIR.parent                      # .../COMPASS
REPO_ROOT = SURVIVAL_PARENT.parent
for _p in (str(REPO_ROOT), str(SURVIVAL_PARENT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survival_common.longitudinal_runners import build_deephit_parser, run_deephit  # noqa: E402


def main(args) -> None:
    run_deephit(args)


if __name__ == "__main__":
    parser = build_deephit_parser()
    main(parser.parse_args())
