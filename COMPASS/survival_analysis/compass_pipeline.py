"""Shared driver logic for the COMPASS PROFILE survival pipeline notebooks.

Both raw-data variants (`baseline`: one-release OncDRS CSVs; `profile_data`:
merged PROFILE_data_processing parquets) and every pipeline stage (cohort
compile, longitudinal preprocessing, prediction-input build, univariate,
multivariate) go through this module, so the stage notebooks
(`01_preprocessing.ipynb` .. `03_multivariate.ipynb`) are thin wrappers over
it. See `REORGANIZATION_PLAN.md` for the motivation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SURVIVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SURVIVAL_DIR.parent.parent
DATA_PREPROCESSING_DIR = SURVIVAL_DIR.parent / "data_preprocessing"

for _p in (str(PROJECT_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_preprocessing_common.oncdrs_sources import resolve as resolve_source  # noqa: E402
from data_preprocessing_common.oncdrs_sources import scan_source  # noqa: E402

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Data variants
# ---------------------------------------------------------------------------
#
# "baseline"     -- one release, /data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03/*.csv,
#                    resolved by compile_COMPASS_cohort_data.py / longitudinal_data_processing.py's
#                    own module-level ONCDRS_PATH / NEPC_PROJ_PATH defaults (no source flags passed).
# "profile_data" -- seven merged releases, /data/gusev/USERS/jpconnor/data/PROFILE_DATA/*.parquet,
#                    resolved here via oncdrs_sources.resolve() and passed explicitly.
#
# Both variants write to disjoint data roots so they can be compared side by side
# (see compare_to_baseline below); the profile_data variant carries a baseline_root
# pointer for that comparison, and the schema audit is a profile_data-only guard
# against the upstream COLUMN_MAP silently null-filling a renamed column.

_ONCDRS_BASELINE_ROOT = Path("/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03")
_BASELINE_DATA_ROOT = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
_PROFILE_DATA_ROOT = Path("/data/gusev/USERS/jpconnor/data/PROFILE_DATA")
_PROFILE_OUTPUT_ROOT = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS_PROFILE_DATA/")

SOURCE_TABLES = [
    "EHR_DIAGNOSES",
    "MEDICATIONS",
    "LABS",
    "HEALTH_HISTORY",
    "PT_INFO_STATUS_REGISTRATION",
]

# Basenames the baseline (single-release) root uses for each logical source table.
# These match compile_COMPASS_cohort_data.py / longitudinal_data_processing.py's own
# ONCDRS_PATH-relative argparse defaults.
_BASELINE_SOURCE_BASENAMES = {
    "EHR_DIAGNOSES": "EHR_DIAGNOSIS.csv",
    "MEDICATIONS": "MEDICATIONS.csv",
    "LABS": "OUTPT_LAB_RESULTS_LABS.csv",
    "HEALTH_HISTORY": "HEALTH_HISTORY.csv",
    "PT_INFO_STATUS_REGISTRATION": "PT_INFO_STATUS_REGISTRATION.csv",
}


def _make_data_variants():
    baseline_sources = {
        t: _ONCDRS_BASELINE_ROOT / basename
        for t, basename in _BASELINE_SOURCE_BASENAMES.items()
    }
    profile_sources = {t: resolve_source(t, _PROFILE_DATA_ROOT) for t in SOURCE_TABLES}
    return {
        "baseline": dict(
            data_root=_BASELINE_DATA_ROOT,
            sources=baseline_sources,
            baseline_root=None,
        ),
        "profile_data": dict(
            data_root=_PROFILE_OUTPUT_ROOT,
            sources=profile_sources,
            baseline_root=_BASELINE_DATA_ROOT,
        ),
    }


DATA_VARIANTS = _make_data_variants()

# ---------------------------------------------------------------------------
# Module-level knobs. Notebooks override by assigning cp.<NAME> = ... before
# calling the functions below, mirroring the original notebooks' top-level cells.
# ---------------------------------------------------------------------------
N_FOLDS = 5
FORCE_RERUN = True
REBUILD_PREDICTION_INPUTS = True

# COMPASS durations (t_lab, t_platinum, ...) are measured from each arm's
# treatment anchor (time 0), so anchor_col is "none" for every arm: the
# landmark is a pure offset from the anchor with no anchor column. Arms
# differ in (a) which anchor's Stage 1 survival cohort CSV restricts them
# (--restrict-to-mrns), (b) which anchor's Stage 2 output feeds them, and
# (c) both arms use landmarks [0, 90, 180].
_ARM_SPECS = {
    "arpi": dict(anchor="arpi", landmarks=[0, 90, 180], title="ARPI"),
    "adt": dict(anchor="adt", landmarks=[0, 90, 180], title="ADT"),
}

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")


def make_runs(variant: str, arms=("adt",)) -> list[dict]:
    """Build the RUNS list for a data variant, restricted to `arms`.

    ARPI is disabled by default on profile_data: build_prediction_inputs.py
    raised "No patients have both engineered features and valid outcomes" for
    that arm on the merged-parquet root. Re-enable once that is root-caused.
    """
    if variant not in DATA_VARIANTS:
        raise ValueError(f"Unknown DATA_VARIANT: {variant!r} (expected one of {sorted(DATA_VARIANTS)})")
    spec = DATA_VARIANTS[variant]
    data_root = spec["data_root"]
    mrn_lists_dir = data_root / "mrn_lists"
    data_arpi = data_root / "longitudinal_prediction_data.csv"
    data_adt = data_root / "longitudinal_prediction_data_adt.csv"
    survival_output_root = data_root / "survival_analysis"

    runs = []
    for label in arms:
        if label not in _ARM_SPECS:
            raise ValueError(f"Unknown arm: {label!r} (expected one of {sorted(_ARM_SPECS)})")
        arm_spec = _ARM_SPECS[label]
        runs.append({
            "label": label,
            "title": arm_spec["title"],
            "variant": variant,
            "anchor_col": "none",
            "anchor": arm_spec["anchor"],
            "landmarks": arm_spec["landmarks"],
            "input_csv": data_arpi if arm_spec["anchor"] == "arpi" else data_adt,
            "restrict_to_mrns": data_root / f"prostate_{label}_survival_cohort_{label}.csv",
            "inputs_dir": survival_output_root / f"prediction_inputs_{label}",
            "output_dir": survival_output_root / f"local_runs_{label}",
            "data_root": data_root,
            "mrn_lists_dir": mrn_lists_dir,
        })

    os.chdir(PROJECT_ROOT)
    mrn_lists_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run["inputs_dir"].mkdir(parents=True, exist_ok=True)
        run["output_dir"].mkdir(parents=True, exist_ok=True)

    print("python:            ", PYTHON)
    print("cwd:               ", os.getcwd())
    print("survival_dir:      ", SURVIVAL_DIR)
    print("data_preprocessing:", DATA_PREPROCESSING_DIR)
    print("data root:         ", data_root)
    if spec["baseline_root"] is not None:
        print("baseline root:     ", spec["baseline_root"], "(read-only, comparison only)")
    for run in runs:
        print(
            f"{run['label']:30s}: anchor={run['anchor']:4s} landmarks={run['landmarks']} "
            f"inputs={run['inputs_dir']} outputs={run['output_dir']}"
        )
    return runs


# ---------------------------------------------------------------------------
# Stage 0 -- schema audit (profile_data only; no-op for baseline)
# ---------------------------------------------------------------------------

# Columns each pipeline stage actually reads. Sources:
#   LABS/HEALTH_HISTORY  -> LAB_SCAN_COLUMNS / HEALTH_SCAN_COLUMNS
#                           (longitudinal_data_processing.py)
#   MEDICATIONS          -> MEDICATION_SCAN_COLUMNS (same file)
#   EHR_DIAGNOSES        -> load_and_explode_icd (compile_COMPASS_cohort_data.py)
#   PT_INFO_...          -> load_patient_status (same file)
#
# Three tiers, because "all null" is a bug for some of these columns and
# perfectly normal for others:
#
#   REQUIRED  must be present AND have at least one non-blank value.
#             All-null here means the cohort silently comes out empty or wrong.
#   EXPECTED  must be present; all-null only warns. These are legitimately
#             sparse (HYBRID_DEATH_DT: most patients are alive; TEXT_RESULT:
#             only non-numeric lab results populate it; vitals rows have it
#             explicitly nulled in build_raw_longitudinal_data).
#   OPTIONAL  absent OR all-null only warns. The 2nd/3rd ICD-10 code slots are
#             optional (load_and_explode_icd falls back to a flat one-code-per-
#             row source) and the _NM name columns are read only by the
#             optional compile_MRNs_for_manual_review.py.
REQUIRED_COLUMNS = {
    "EHR_DIAGNOSES": ["DFCI_MRN", "START_DT", "DIAGNOSIS_ICD10_CD"],
    "MEDICATIONS": ["DFCI_MRN", "NCI_PREFERRED_MED_NM", "MED_START_DT"],
    "LABS": [
        "DFCI_MRN", "SPECIMEN_COLLECT_DT", "TEST_TYPE_CD",
        "NUMERIC_RESULT", "RESULT_UOM_NM",
    ],
    "HEALTH_HISTORY": [
        "DFCI_MRN", "CODE_TYPE", "START_DT", "HEALTH_HISTORY_TYPE", "RESULTS",
    ],
    "PT_INFO_STATUS_REGISTRATION": [
        "DFCI_MRN", "BIRTH_DT", "DERIVED_LAST_ALIVE_DATE", "GENDER_NM",
    ],
}

EXPECTED_COLUMNS = {
    # TEST_TYPE_DESCR feeds generate_new_test_name_expr(TEST_TYPE_CD,
    # TEST_TYPE_DESCR); all-null degrades lab naming rather than breaking it.
    "LABS": ["TEST_TYPE_DESCR", "TEXT_RESULT"],
    "HEALTH_HISTORY": ["UNITS_CD"],
    # All-null death dates would mean DEATH == 0 for everyone. The modelled
    # endpoint is platinum, not death, so this warns rather than raising --
    # but it is the loudest thing in this cell for a reason.
    "PT_INFO_STATUS_REGISTRATION": ["HYBRID_DEATH_DT"],
}

OPTIONAL_COLUMNS = {
    "EHR_DIAGNOSES": [
        "DIAGNOSIS_ICD10_CD2", "DIAGNOSIS_ICD10_CD3",
        "DIAGNOSIS_ICD10_NM", "DIAGNOSIS_ICD10_NM2", "DIAGNOSIS_ICD10_NM3",
    ],
}


def _nonempty_fraction_expr(name, dtype):
    """Fraction of rows where `name` is neither null nor a blank string.

    Blank matters because the null-fill hazard is not the only way a column
    arrives empty -- a source column that exists but was never populated shows
    up as "" in the CSV releases and survives the merge as an empty string.
    """
    import polars as pl

    col = pl.col(name)
    if dtype == pl.String:
        return (col.is_not_null() & (col.str.strip_chars() != "")).mean().alias(name)
    return col.is_not_null().mean().alias(name)


def audit_schema(variant: str) -> None:
    """Fail fast if a required column is absent or all-null in the variant's sources.

    No-op for "baseline": the single-release CSVs are the schema every script's
    argparse default already assumes, so there is nothing to audit against.
    """
    if DATA_VARIANTS[variant]["sources"] is None or variant == "baseline":
        print(f"[skip] schema audit: not applicable to variant {variant!r}")
        return

    import polars as pl

    sources = DATA_VARIANTS[variant]["sources"]
    problems = []
    warnings = []

    for table in SOURCE_TABLES:
        path = sources[table]
        print(f"\n===== {table} =====")
        print(f"  path: {path}")
        if not path.exists():
            problems.append(f"{table}: source file not found: {path}")
            print("  MISSING -- skipping")
            continue

        required = REQUIRED_COLUMNS[table]
        expected = EXPECTED_COLUMNS.get(table, [])
        optional = OPTIONAL_COLUMNS.get(table, [])
        tier = {c: t for t, cols in
                (("req", required), ("exp", expected), ("opt", optional))
                for c in cols}

        lf = scan_source(path)
        schema = lf.collect_schema()
        present = list(schema.names())

        # scan_source() casts parquet to all-Utf8, so read dtypes from the parquet
        # itself -- otherwise NUMERIC_RESULT would look like a string column and
        # get the blank-string treatment.
        raw_schema = (
            pl.scan_parquet(path).collect_schema()
            if path.suffix.lower() in (".parquet", ".pq")
            else schema
        )

        checkable = [c for c in required + expected + optional if c in present]
        stats = lf.select(
            [pl.len().alias("__n_rows"), pl.col("DFCI_MRN").n_unique().alias("__n_mrns")]
            + [_nonempty_fraction_expr(c, raw_schema[c]) for c in checkable]
        ).collect()

        print(f"  rows: {int(stats['__n_rows'][0]):,}   "
              f"unique DFCI_MRN: {int(stats['__n_mrns'][0]):,}")

        for group, label in ((required, "required"), (expected, "expected")):
            absent = [c for c in group if c not in present]
            if absent:
                problems.append(f"{table}: {label} columns absent: {absent}")
                print(f"  ABSENT ({label}): {absent}")
        absent_optional = [c for c in optional if c not in present]
        if absent_optional:
            warnings.append(f"{table}: optional columns absent: {absent_optional}")
            print(f"  absent (optional): {absent_optional}")

        for c in checkable:
            frac = stats[c][0]
            frac = 0.0 if frac is None else float(frac)
            flag = ""
            if frac == 0.0:
                if tier[c] == "req":
                    problems.append(f"{table}.{c}: present but 100% null/blank")
                    flag = "   <-- ALL NULL (required)"
                else:
                    warnings.append(f"{table}.{c}: present but 100% null/blank")
                    flag = f"   <-- all null ({tier[c]})"
            print(f"    {tier[c]} {c:28s} non-null {frac:7.2%}{flag}")

    print("\n" + "=" * 70)
    for w in warnings:
        print(f"WARN: {w}")
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        raise RuntimeError(
            f"{len(problems)} schema problem(s) in {_PROFILE_DATA_ROOT}. Fix COLUMN_MAP / "
            "ALIAS_MAP in PROFILE_data_processing/compile_OncDRS_data.ipynb and re-run "
            "that notebook for the affected tables before continuing."
        )
    print(f"Schema audit passed ({len(warnings)} warning(s)): every required column "
          "is present and non-empty.")


# ---------------------------------------------------------------------------
# Stage 1 -- compile COMPASS cohort data
# ---------------------------------------------------------------------------

def _run(cmd, dry_run=False):
    print("[run ] " + " ".join(str(c) for c in cmd))
    if dry_run:
        return 0
    return subprocess.call([str(c) for c in cmd])


def compile_cohort(variant: str, arms=("adt",), dry_run: bool = False) -> None:
    spec = DATA_VARIANTS[variant]
    data_root = spec["data_root"]
    cmd = [
        PYTHON, DATA_PREPROCESSING_DIR / "compile_COMPASS_cohort_data.py",
        "--icd-source", spec["sources"]["EHR_DIAGNOSES"],
        "--medications-source", spec["sources"]["MEDICATIONS"],
        "--patient-status-source", spec["sources"]["PT_INFO_STATUS_REGISTRATION"],
        "--labs-csv", spec["sources"]["LABS"],
        "--out-dir", data_root,
        "--mrn-lists-dir", data_root / "mrn_lists",
        "--survival-arms", *arms,
    ]
    rc = _run(cmd, dry_run=dry_run)
    if not dry_run and rc != 0:
        raise RuntimeError(f"compile_COMPASS_cohort_data.py failed with rc={rc}")


# ---------------------------------------------------------------------------
# Stage 2 -- preprocess raw labs once per anchor (longitudinal_data_processing.py)
# ---------------------------------------------------------------------------

def preprocess_labs(run: dict, dry_run: bool = False) -> None:
    """Full raw lab standardization for one arm's anchor. Expensive; the Parquet
    cache makes reruns cheap, but the first pass may be slow.
    """
    variant = run["variant"]
    spec = DATA_VARIANTS[variant]
    data_root = spec["data_root"]
    label = run["label"]
    output_csv = run["input_csv"]
    cache_parquet = data_root / f"consolidated_longitudinal_data_{label}.parquet"
    icd_csv = data_root / "prostate_icd_data.csv"

    cmd = [
        PYTHON, DATA_PREPROCESSING_DIR / "longitudinal_data_processing.py",
        "--health-csv", spec["sources"]["HEALTH_HISTORY"],
        "--labs-csv", spec["sources"]["LABS"],
        "--medications-csv", spec["sources"]["MEDICATIONS"],
        "--icd-csv", icd_csv,
        "--anchor-med-set", run["anchor"],
        "--survival-cohort-csv", run["restrict_to_mrns"],
        "--output-csv", output_csv,
        "--consolidated-cache-parquet", cache_parquet,
    ]
    rc = _run(cmd, dry_run=dry_run)
    if not dry_run and rc != 0:
        raise RuntimeError(f"longitudinal_data_processing.py failed for {label} with rc={rc}")


# ---------------------------------------------------------------------------
# Stage 3 -- build prediction inputs + cohort diagnostics
# ---------------------------------------------------------------------------

def clear_prediction_inputs(inputs_dir: Path) -> None:
    for pattern in (
        "aggregated_landmark*.csv",
        "pre_treatment_lab_long_landmark*.csv",
        "split_assignments_landmark*.csv",
    ):
        for p in inputs_dir.glob(pattern):
            p.unlink()
            print(f"  removed {p.name}")
    for fname in (
        "canonical_labs_train_val.csv",
        "build_manifest.json",
        "split_assignments.csv",
        "landmark_mrn_availability.csv",
        "landmark_attrition.json",
    ):
        p = inputs_dir / fname
        if p.exists():
            p.unlink()
            print(f"  removed {p.name}")


def build_prediction_inputs(run: dict, dry_run: bool = False) -> None:
    print(f"\n========== build inputs: {run['title']} ==========")
    if not dry_run:
        clear_prediction_inputs(run["inputs_dir"])
    cmd = [
        PYTHON, DATA_PREPROCESSING_DIR / "build_prediction_inputs.py",
        "--data", run["input_csv"],
        "--output-dir", run["inputs_dir"],
        "--anchor-col", run["anchor_col"],
        "--landmark-days", *[str(lm) for lm in run["landmarks"]],
        "--time-unit-days", "7",
        "--test-frac", "0.20",
        "--val-frac", "0.20",
        "--min-patient-coverage", "0.20",
    ]
    if run.get("restrict_to_mrns"):
        cmd += ["--restrict-to-mrns", run["restrict_to_mrns"]]
    rc = _run(cmd, dry_run=dry_run)
    if not dry_run and rc != 0:
        raise RuntimeError(f"build_prediction_inputs failed for {run['label']} with rc={rc}")


def cohort_diagnostics(run: dict) -> None:
    print(f"\n========== cohort diagnostics: {run['title']} ==========")
    for lm in run["landmarks"]:
        agg_path = run["inputs_dir"] / f"aggregated_landmark{lm}.csv"
        if not agg_path.exists():
            print(f"  landmark +{lm}d: aggregated CSV not found, skipping")
            continue
        agg = pd.read_csv(agg_path)

        def find_col(substr, stat):
            return next(
                (c for c in agg.columns if substr.lower() in c.lower() and c.endswith(f"__{stat}")),
                None,
            )

        n_plat = int(agg["PLATINUM"].sum())
        print(f"=== landmark +{lm}d | n_total={len(agg):,} n_PLATINUM={n_plat} ===")
        for lab_substr in ("Testosterone", "PSA", "Prostate specific Ag"):
            for stat in ("mean", "last", "max", "min"):
                col = find_col(lab_substr, stat)
                if col is None:
                    continue
                for ev in (0, 1):
                    sub = agg.loc[agg["PLATINUM"] == ev, col].dropna()
                    if sub.empty:
                        continue
                    print(
                        f"  {lab_substr:>22s} {stat:5s} PLAT={ev}: median={sub.median():>10.2f} "
                        f"max={sub.max():>12.2f} n={len(sub):>5}"
                    )
                break
        print()


# ---------------------------------------------------------------------------
# Model task specs, split by stage (was one MODEL_TASK_SPECS list in the
# monolithic notebooks; split so 02_univariate and 03_multivariate each drive
# only their own subset).
# ---------------------------------------------------------------------------

UNIVARIATE_TASK_SPECS = [
    ("univariate", "both", "cox_agg_univariate_nobs_adjusted.csv"),
]

MULTIVARIATE_TASK_SPECS = [
    ("elastic-net", "both", "cox_agg_multivariable_metrics.csv"),
    ("elastic-net", "baseline", "cox_agg_baseline_metrics.csv"),
    ("xgboost", "both", "landmark_xgboost_metrics.csv"),
    ("xgboost", "baseline", "landmark_xgboost_baseline_metrics.csv"),
]

# Output subdir for the shared-canonical-labs univariate arm. This arm is a
# SINGLE invocation over ALL landmarks (univariate_analysis.py --shared-canonical-labs
# --landmark-days <all>): the runner intersects each landmark's canonical labs and
# tests that one shared set at every landmark, so every selected landmark sees an
# identical feature list and their per-lab HRs are directly comparable. Results
# land in cox/landmark_shared/ with landmark_days as a column inside the CSV.
SHARED_UNIVARIATE_DIR = "cox/landmark_shared"
SHARED_UNIVARIATE_FILE = "cox_agg_univariate_nobs_adjusted.csv"


def tasks_for_run(run: dict, specs=MULTIVARIATE_TASK_SPECS):
    """Cross a task-spec list with this run's own landmark list.

    Both ARPI and ADT arms have landmarks=[0, 90, 180].
    """
    return [
        (model, lm, config_dir, metrics_filename)
        for model, config_dir, metrics_filename in specs
        for lm in run["landmarks"]
    ]


def model_output_dir(model: str) -> str:
    return "cox" if model in ("univariate", "elastic-net") else "xgboost"


def build_model_command(model, landmark, config_dir, row_output_dir, run):
    if model == "univariate":
        return [
            PYTHON, SURVIVAL_DIR / "univariate_analysis.py",
            "--inputs-dir", run["inputs_dir"],
            "--output-dir", row_output_dir,
            "--landmark-days", str(landmark),
            "--endpoints", "platinum",
        ]
    if model == "elastic-net":
        cmd = [
            PYTHON, SURVIVAL_DIR / "multivariate_analysis.py",
            "--model", "elastic-net",
            "--inputs-dir", run["inputs_dir"],
            "--output-dir", row_output_dir,
            "--landmark-days", str(landmark),
            "--endpoints", "platinum",
            "--n-folds", str(N_FOLDS),
        ]
        if config_dir == "baseline":
            cmd.append("--baseline")
        return cmd
    if model == "xgboost":
        cmd = [
            PYTHON, SURVIVAL_DIR / "multivariate_analysis.py",
            "--model", "xgboost",
            "--inputs-dir", run["inputs_dir"],
            "--output-dir", row_output_dir,
            "--landmark-days", str(landmark),
            "--endpoints", "platinum",
            "--n-folds", str(N_FOLDS),
        ]
        if config_dir == "baseline":
            cmd.append("--baseline")
        return cmd
    raise ValueError(f"Unknown model: {model}")


def run_shared_univariate(run: dict, dry_run: bool = False):
    """Univariate arm on ONE shared canonical lab set across all of this run's landmarks.

    Single invocation over every landmark in run["landmarks"] with
    --shared-canonical-labs, so the runner intersects each landmark's canonical
    labs and tests that shared set at every landmark. The resulting CSV carries a
    landmark_days column, so every selected landmark's rows are directly comparable.
    """
    row_output_dir = run["output_dir"] / SHARED_UNIVARIATE_DIR
    row_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = row_output_dir / SHARED_UNIVARIATE_FILE
    tag = f"{run['label']:28s} univariate  landmark_shared (labs={','.join(map(str, run['landmarks']))})"
    if metrics_path.exists() and not FORCE_RERUN:
        print(f"[skip] {tag} -> exists")
        return (tag, "skipped", 0.0)
    cmd = [
        PYTHON, SURVIVAL_DIR / "univariate_analysis.py",
        "--inputs-dir", run["inputs_dir"],
        "--output-dir", row_output_dir,
        "--landmark-days", *[str(lm) for lm in run["landmarks"]],
        "--endpoints", "platinum",
        "--shared-canonical-labs",
    ]
    print(f"[run ] {tag}")
    t0 = time.time()
    rc = _run(cmd, dry_run=dry_run)
    elapsed = time.time() - t0
    status = "ok" if rc == 0 else f"FAILED (rc={rc})"
    print(f"[done] {tag} -> {status} ({elapsed/60:.1f} min)\n")
    return (tag, status, elapsed)


def _run_tasks(run: dict, specs, dry_run: bool = False):
    print(f"\n========== run models: {run['title']} ==========")
    tasks = tasks_for_run(run, specs)
    summary = []
    for model, landmark, config_dir, metrics_filename in tasks:
        row_output_dir = run["output_dir"] / model_output_dir(model) / f"landmark_{landmark}" / config_dir
        metrics_path = row_output_dir / metrics_filename
        tag = f"{run['label']:28s} {model:11s} landmark_{landmark:<3} {config_dir}"
        if metrics_path.exists() and not FORCE_RERUN:
            print(f"[skip] {tag} -> {metrics_path.relative_to(run['output_dir'])} exists")
            summary.append((tag, "skipped", 0.0))
            continue
        if not dry_run:
            row_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_model_command(model, landmark, config_dir, row_output_dir, run)
        print(f"[run ] {tag}")
        t0 = time.time()
        rc = _run(cmd, dry_run=dry_run)
        elapsed = time.time() - t0
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[done] {tag} -> {status} ({elapsed/60:.1f} min)\n")
        summary.append((tag, status, elapsed))
    return summary


def run_univariate(run: dict, dry_run: bool = False):
    """Per-landmark univariate arm, plus the shared-canonical-labs arm."""
    summary = _run_tasks(run, UNIVARIATE_TASK_SPECS, dry_run=dry_run)
    summary.append(run_shared_univariate(run, dry_run=dry_run))
    print("\n=== run summary ===")
    for tag, status, elapsed in summary:
        print(f"  {tag} {status:>20s} {elapsed/60:6.1f} min")
    return summary


def run_multivariate(run: dict, dry_run: bool = False):
    """Elastic-net (both/baseline) and XGBoost (both/baseline) arms."""
    summary = _run_tasks(run, MULTIVARIATE_TASK_SPECS, dry_run=dry_run)
    print("\n=== run summary ===")
    for tag, status, elapsed in summary:
        print(f"  {tag} {status:>20s} {elapsed/60:6.1f} min")
    return summary


def summarize_outputs(run: dict) -> pd.DataFrame:
    rows = []
    for model, landmark, config_dir, metrics_filename in tasks_for_run(run, MULTIVARIATE_TASK_SPECS):
        metrics_path = run["output_dir"] / model_output_dir(model) / f"landmark_{landmark}" / config_dir / metrics_filename
        base = {"run": run["label"], "model": model, "landmark": landmark, "config": config_dir, "endpoint": "platinum"}
        if not metrics_path.exists():
            rows.append({**base, "n_test": None, "n_test_events": None, "c_index": None,
                         "mean_auc_t": None, "integrated_brier": None, "status": "missing"})
            continue
        df = pd.read_csv(metrics_path)
        platinum = df.loc[df["endpoint"] == "platinum"]
        if platinum.empty:
            rows.append({**base, "n_test": None, "n_test_events": None, "c_index": None,
                         "mean_auc_t": None, "integrated_brier": None, "status": "no platinum row"})
            continue
        platinum = platinum.iloc[0]
        if model == "elastic-net":
            rows.append({
                **base,
                "n_test": int(platinum["n_test"]),
                "n_test_events": int(platinum["n_events_test"]),
                "c_index": float(platinum["test_c_index"]),
                "mean_auc_t": float(platinum["test_mean_auc_t"]),
                "integrated_brier": float(platinum["test_integrated_brier"]),
                "status": "ok",
            })
        elif model == "xgboost":
            rows.append({
                **base,
                "n_test": int(platinum["n_test"]),
                "n_test_events": int(platinum["n_test_events"]),
                "c_index": float(platinum["c_index"]),
                "mean_auc_t": float(platinum["mean_auc_t"]),
                "integrated_brier": float(platinum["integrated_brier"]),
                "status": "ok",
            })
    return pd.DataFrame(rows).sort_values(["run", "landmark", "model", "config"]).reset_index(drop=True)


def summarize_univariate_shared(run: dict) -> pd.DataFrame:
    """Per-lab platinum associations at every landmark for this run, on the SHARED lab set.

    Reads the shared-arm univariate CSV (identical canonical lab set at every
    landmark, landmark_days as a column) so PSA/testosterone HRs can be read
    across landmarks side by side.
    """
    path = run["output_dir"] / SHARED_UNIVARIATE_DIR / SHARED_UNIVARIATE_FILE
    if not path.exists():
        print(f"  shared univariate output missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.loc[df["endpoint"] == "platinum"].copy()
    keep = ["landmark_days", "feature", "lab_name", "feature_stat",
            "hazard_ratio_per_sd", "p_value", "q_value", "n_patients_used", "n_events_used"]
    keep = [c for c in keep if c in df.columns]
    sort_cols = [c for c in ("lab_name", "feature_stat", "landmark_days") if c in df.columns]
    return df[keep].sort_values(sort_cols).reset_index(drop=True)


def filter_nominal(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Retain rows with nominal significance (p_value < alpha).

    Exploratory filter; q_value remains in the table so multiplicity-adjusted
    significance can be assessed separately. No stability or false-discovery-rate
    filter is applied here.
    """
    preferred_columns = [
        "cohort", "landmark_days", "endpoint", "feature", "lab_name",
        "feature_stat", "coverage", "n_patients_used", "n_events_used",
        "coef_feature", "hazard_ratio_per_sd", "ci_lower", "ci_upper",
        "p_value", "q_value", "note", "model_type", "source_path",
    ]
    filtered = (
        results.loc[results["p_value"].notna() & results["p_value"].lt(alpha)]
        .sort_values(["endpoint", "landmark_days", "p_value", "feature"])
        .reset_index(drop=True)
    )
    ordered = [c for c in preferred_columns if c in filtered.columns]
    ordered += [c for c in filtered.columns if c not in ordered]
    return filtered[ordered]


def load_univariate_results(run: dict) -> pd.DataFrame:
    """Load the shared-canonical univariate CSV (preferred) or legacy per-landmark files.

    The shared-canonical result file is preferred because it contains every
    reported landmark in one table and is the source used by Figure 3. Legacy
    per-landmark files are loaded only when the shared file is unavailable.
    """
    import re

    run_dir = run["output_dir"]
    shared_path = run_dir / SHARED_UNIVARIATE_DIR / SHARED_UNIVARIATE_FILE
    if shared_path.exists():
        paths = [shared_path]
    else:
        paths = sorted((run_dir / "cox").glob(f"landmark_*/both/{SHARED_UNIVARIATE_FILE}"))

    if not paths:
        raise FileNotFoundError(
            f"No univariate result files found under {run_dir / 'cox'}. "
            "Run the univariate models first."
        )

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "landmark_days" not in frame.columns:
            match = re.search(r"landmark_(-?\d+)", str(path.parent.parent))
            if match is None:
                raise ValueError(f"Could not infer landmark from {path}")
            frame.insert(0, "landmark_days", int(match.group(1)))
        frame["source_path"] = str(path)
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    required = {"landmark_days", "endpoint", "feature", "p_value"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Univariate results are missing columns: {sorted(missing)}")

    results["p_value"] = pd.to_numeric(results["p_value"], errors="coerce")
    results.insert(0, "cohort", run["label"])
    return results


# ---------------------------------------------------------------------------
# Cohort comparison (profile_data vs. baseline). No-op when the variant has
# no baseline_root.
# ---------------------------------------------------------------------------

def _read_landmark_attrition(root: Path, label: str):
    path = root / "survival_analysis" / f"prediction_inputs_{label}" / "landmark_attrition.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_cohort_csv_counts(root: Path, label: str) -> dict:
    """Stage 1 counts straight from that arm's survival cohort CSV."""
    path = root / f"prostate_{label}_survival_cohort_{label}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=lambda c: c in ("DFCI_MRN", "PLATINUM_DATE", "DEATH"))
    out = {"stage1_n_cohort": int(df["DFCI_MRN"].nunique())}
    if "PLATINUM_DATE" in df.columns:
        out["stage1_n_platinum_dated"] = int(df["PLATINUM_DATE"].notna().sum())
    if "DEATH" in df.columns:
        out["stage1_n_deaths"] = int(df["DEATH"].sum())
    return out


def _read_landmark_counts(root: Path, label: str, landmarks) -> dict:
    inputs_dir = root / "survival_analysis" / f"prediction_inputs_{label}"
    out = {}
    attrition = _read_landmark_attrition(root, label)
    if attrition:
        out["stage3_n_loaded_cohort"] = attrition.get("n_loaded_cohort")
        for lm, n in (attrition.get("eligible_by_landmark") or {}).items():
            out[f"lm{lm}_n_eligible"] = n
    for lm in landmarks:
        agg_path = inputs_dir / f"aggregated_landmark{lm}.csv"
        if not agg_path.exists():
            continue
        agg = pd.read_csv(agg_path, usecols=["DFCI_MRN", "PLATINUM"])
        out[f"lm{lm}_n_total"] = int(len(agg))
        out[f"lm{lm}_n_platinum"] = int(agg["PLATINUM"].sum())
    return out


def _collect_metrics(root: Path, label: str, landmarks) -> dict:
    m = {}
    m.update(_read_cohort_csv_counts(root, label))
    m.update(_read_landmark_counts(root, label, landmarks))
    return m


def compare_to_baseline(variant: str, runs: list[dict]) -> pd.DataFrame:
    """Per-stage patient counts for `runs` against the baseline (ALL_2025_03) root.

    Reads only; nothing under the baseline root is written. No-op (prints a
    note and returns an empty frame) for variants without a baseline_root,
    i.e. "baseline" itself.
    """
    baseline_root = DATA_VARIANTS[variant]["baseline_root"]
    if baseline_root is None:
        print(f"[skip] compare_to_baseline: variant {variant!r} has no baseline_root")
        return pd.DataFrame()

    new_root = DATA_VARIANTS[variant]["data_root"]
    comparison_rows = []
    for run in runs:
        label, landmarks = run["label"], run["landmarks"]
        new = _collect_metrics(new_root, label, landmarks)
        old = _collect_metrics(baseline_root, label, landmarks)
        for metric in sorted(set(new) | set(old), key=lambda k: (k.split("_")[0], k)):
            a, b = old.get(metric), new.get(metric)
            comparison_rows.append({
                "arm": label,
                "metric": metric,
                "ALL_2025_03": a,
                variant.upper(): b,
                "delta": (b - a) if (a is not None and b is not None) else None,
                "pct_change": (100.0 * (b - a) / a) if (a not in (None, 0) and b is not None) else None,
            })

    comparison_df = pd.DataFrame(comparison_rows)
    if comparison_df.empty:
        print("Nothing to compare yet -- run the stages above (and confirm the "
              f"baseline run exists at {baseline_root}).")
    else:
        missing_baseline = comparison_df["ALL_2025_03"].isna().all()
        if missing_baseline:
            print(f"WARNING: no baseline outputs found under {baseline_root}; "
                  "deltas are unavailable.")
        with pd.option_context("display.max_rows", 200, "display.width", 160):
            print(comparison_df.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    # Stage 2 attrition is written per OUTPUT ROOT, not per arm
    # (longitudinal_data_processing.py writes cohort_attrition.json next to
    # --output-csv, and both arms share that directory), so it reflects whichever
    # anchor ran last. Shown separately for that reason.
    print(f"\n=== Stage 2 cohort_attrition.json (last anchor run in each root) ===")
    for name, root in (("ALL_2025_03", baseline_root), (variant.upper(), new_root)):
        path = root / "cohort_attrition.json"
        if not path.exists():
            print(f"  {name}: not found at {path}")
            continue
        attrition = json.loads(path.read_text())
        print(f"  {name}: n_output_patients={attrition.get('n_output_patients')} "
              f"n_with_highlighted_treatment_anchor={attrition.get('n_with_highlighted_treatment_anchor')} "
              f"n_after_broad_icd_filter={attrition.get('n_after_broad_icd_filter')}")

    return comparison_df
