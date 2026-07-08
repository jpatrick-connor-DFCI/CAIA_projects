"""COMPASS PROFILE wrapper for shared univariate Cox survival analysis."""

from __future__ import annotations

import sys
from pathlib import Path

SURVIVAL_DIR = Path(__file__).resolve().parent
SURVIVAL_PARENT = SURVIVAL_DIR.parent
REPO_ROOT = SURVIVAL_PARENT.parent
for _p in (str(REPO_ROOT), str(SURVIVAL_PARENT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cox_aggregated as _ca  # noqa: E402
from survival_common.cox_runners import build_univariate_parser, run_univariate  # noqa: E402
from survival_common.projects.compass_profile import CONFIG  # noqa: E402


def main(args) -> None:
    run_univariate(CONFIG, _ca, args)


if __name__ == "__main__":
    parser = build_univariate_parser(CONFIG, _ca)
    main(parser.parse_args())
