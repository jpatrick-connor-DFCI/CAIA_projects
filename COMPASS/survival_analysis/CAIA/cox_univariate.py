"""CAIA twin of PROFILE/cox_univariate.py.

Same univariate Cox code path, just injects CAIA defaults into `sys.argv`
before delegating to the PROFILE module via `runpy`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent              # .../survival_analysis/CAIA
SURVIVAL_PARENT = SCRIPT_DIR.parent                       # .../survival_analysis
PROFILE_DIR = SURVIVAL_PARENT / "PROFILE"
for _p in (str(SURVIVAL_PARENT), str(PROFILE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CAIA_RESULTS_ROOT = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/CAIA")

CAIA_DEFAULTS: dict[str, str] = {
    "--id-col":     "person_id",
    "--age-col":    "AGE_AT_DIAGNOSIS",
    "--inputs-dir": str(CAIA_RESULTS_ROOT / "prediction_inputs"),
    "--output-dir": str(CAIA_RESULTS_ROOT),
}


def _inject_defaults(defaults: dict[str, str]) -> None:
    for flag, value in defaults.items():
        # Skip if the flag is already present in either "--flag value" or "--flag=value" form.
        if any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv):
            continue
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _inject_defaults(CAIA_DEFAULTS)
    runpy.run_module("cox_univariate", run_name="__main__")
