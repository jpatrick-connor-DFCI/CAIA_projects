"""COMPASS PROFILE wrapper for shared univariate Cox survival analysis."""

from __future__ import annotations

import sys
from pathlib import Path

COMPASS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPASS_DIR.parent
PIPELINE_DIR = COMPASS_DIR / "pipeline"  # cox_aggregated
for _p in (str(REPO_ROOT), str(COMPASS_DIR), str(PIPELINE_DIR)):
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
