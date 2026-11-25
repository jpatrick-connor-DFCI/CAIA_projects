"""Default input / output locations for the prostate IHC extraction (cluster layout).

Raw inputs live under PROFILE_DATA; outputs under the COMPASS data root in ``ihc_extraction/``:
``intermediate/`` (step-1 parquet files) and ``results/`` (step-2 CSVs). Every path can be
overridden with an environment variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROFILE_DATA_ROOT = Path("/data/gusev/USERS/jpconnor/data/PROFILE_DATA")
COMPASS_DATA_ROOT = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS")

# Step-1 outputs (intermediate/) and step-2 outputs (results/).
COHORT_FILE = "prostate_cohort.parquet"
LONG_FILE = "prostate_ihc_long.parquet"
SPECIMEN_ALL_FILE = "prostate_ihc_specimen_all.parquet"
ONCOPANEL_SPECIMEN_FILE = "oncopanel_prostate_specimen_ihc.csv"
ONCOPANEL_LONG_FILE = "oncopanel_prostate_ihc_long.csv"
INTERMEDIATE_FILES = (COHORT_FILE, LONG_FILE, SPECIMEN_ALL_FILE)
RESULT_FILES = (ONCOPANEL_SPECIMEN_FILE, ONCOPANEL_LONG_FILE)

@dataclass(frozen=True)
class IHCPaths:
    pathology_dir: Path      # ihc.parquet, samples.parquet (rule-based pathology extraction)
    parquet_dir: Path        # IDM_CARG_DIAGNOSIS.parquet, CDM_SPEC_CON_ids.parquet
    intermediate_dir: Path   # step-1 outputs
    results_dir: Path        # step-2 outputs


def default_paths() -> IHCPaths:
    root = Path(os.environ.get("COMPASS_IHC_EXTRACTION_ROOT",
                               str(COMPASS_DATA_ROOT / "ihc_extraction")))
    return IHCPaths(
        pathology_dir=Path(os.environ.get("PROFILE_PATHOLOGY_EXTRACTS",
                                          str(PROFILE_DATA_ROOT / "pathology_extracts"))),
        parquet_dir=Path(os.environ.get("PROFILE_PARQUET_DIR", str(PROFILE_DATA_ROOT / "parquet"))),
        intermediate_dir=root / "intermediate",
        results_dir=root / "results",
    )
