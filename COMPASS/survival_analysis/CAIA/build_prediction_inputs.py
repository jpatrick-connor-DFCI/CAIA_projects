"""CAIA twin of PROFILE/build_prediction_inputs.py.

Same orchestration as PROFILE — coverage-based canonical labs, per-landmark
feature aggregation, person-period DeepHit frame, fixed AUC(t) horizon grid —
but with CAIA-flavoured defaults injected into `sys.argv` before delegating
to the PROFILE script.

PROFILE/build_prediction_inputs.py auto-detects `.parquet` inputs and routes
through `helpers.loaders.load_caia_parquet`, which is the only data-loading
difference.  Cohort filters (PARPi exclusion, FIRST_TREATMENT, PSA>=5) are
upstream of the parquet so they are no-ops here.

Invoke directly:

    python COMPASS/survival_analysis/CAIA/build_prediction_inputs.py \
      [--data /data/.../caia_compass_longitudinal.parquet] \
      [--output-dir /data/.../survival_analysis/CAIA/prediction_inputs] \
      [--landmark-days 0 90] [other PROFILE args]

Defaults are injected only when the flag is not already on the command line,
so any PROFILE arg can still be overridden.
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

CAIA_DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS")
CAIA_RESULTS_ROOT = CAIA_DATA_PATH / "survival_analysis" / "CAIA"

CAIA_DEFAULTS: dict[str, str] = {
    "--id-col":     "person_id",
    "--age-col":    "AGE_AT_DIAGNOSIS",
    "--data":       str(CAIA_DATA_PATH / "caia_compass_longitudinal.parquet"),
    "--output-dir": str(CAIA_RESULTS_ROOT / "prediction_inputs"),
}


def _inject_defaults(defaults: dict[str, str]) -> None:
    for flag, value in defaults.items():
        # Skip if the flag is already present in either "--flag value" or "--flag=value" form.
        if any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv):
            continue
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _inject_defaults(CAIA_DEFAULTS)
    runpy.run_module("build_prediction_inputs", run_name="__main__")
