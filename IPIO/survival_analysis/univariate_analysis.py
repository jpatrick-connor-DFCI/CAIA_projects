"""IPIO wrapper for shared univariate Cox survival analysis."""

from __future__ import annotations

import sys
from pathlib import Path

SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from _paths import ensure_survival_common_on_path  # noqa: E402

ensure_survival_common_on_path()

import cox_aggregated as _ca  # noqa: E402
from survival_common.cox_runners import build_univariate_parser, run_univariate  # noqa: E402
from survival_common.projects.ipio import CONFIG  # noqa: E402


def main(args) -> None:
    run_univariate(CONFIG, _ca, args)


if __name__ == "__main__":
    parser = build_univariate_parser(CONFIG, _ca)
    main(parser.parse_args())
