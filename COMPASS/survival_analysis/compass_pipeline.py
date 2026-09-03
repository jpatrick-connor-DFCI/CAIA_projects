"""Shared driver logic for the COMPASS PROFILE survival pipeline notebooks.

The pipeline uses the merged ``profile_data`` parquets for every stage (cohort
compile, longitudinal preprocessing, prediction-input build, univariate, and
multivariate), so the stage notebooks
(`01_preprocessing.ipynb` .. `03_multivariate.ipynb`) are thin wrappers over
it. See `REORGANIZATION_PLAN.md` for the motivation.
"""

from __future__ import annotations

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

from data_preprocessing_common.oncdrs_sources import TABLE_FILES  # noqa: E402
from data_preprocessing_common.oncdrs_sources import scan_source  # noqa: E402
from COMPASS.survival_analysis import cox_aggregated as _ca  # noqa: E402
from survival_common.metrics_schema import DEFAULT_COHORT  # noqa: E402

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
#
# All COMPASS runs read the merged PROFILE_data_processing parquets and write
# beneath the standard COMPASS output root. The schema audit guards
# against the upstream COLUMN_MAP silently null-filling a renamed column.

_PROFILE_DATA_ROOT = Path("/data/gusev/USERS/jpconnor/data/PROFILE_DATA")
_PROFILE_OUTPUT_ROOT = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")

SOURCE_TABLES = [
    "EHR_DIAGNOSES",
    "MEDICATIONS",
    "LABS",
    "HEALTH_HISTORY",
    "PT_INFO_STATUS_REGISTRATION",
]

PROFILE_SOURCES = {
    table: _PROFILE_DATA_ROOT / TABLE_FILES[table][0]
    for table in SOURCE_TABLES
}

# ---------------------------------------------------------------------------
# Module-level knobs. Notebooks override by assigning cp.<NAME> = ... before
# calling the functions below, mirroring the original notebooks' top-level cells.
# ---------------------------------------------------------------------------
N_FOLDS = 5
FORCE_RERUN = True
REBUILD_PREDICTION_INPUTS = True
# Which event is modeled. Must be a key of
# COMPASS.survival_analysis.cox_aggregated.ENDPOINTS. "platinum" is the
# original endpoint; "nepc" is time from the ADT anchor to the LLM-adjudicated
# NEPC diagnosis. Each endpoint belongs in its own inputs/output tree; new
# notebooks use make_endpoint_runs() to construct both trees together.
ENDPOINT = "platinum"

# multivariate_longitudinal (SurvLatent ODE + Dynamic-DeepHit) knobs. The
# editable upstream checkout is kept beside this pipeline so cluster copies of
# CAIA carry the exact SurvLatent source they run.
DEFAULT_SURVLATENT_REPO = SURVIVAL_DIR / "survlatent_ode_repo"
SURVLATENT_REPO = DEFAULT_SURVLATENT_REPO
MAX_PRED_WINDOW = 260
# These analyses use the baseline ADT cohort. Sequencing and Gleason get their
# own observation-date origins; PRS retains ADT start as time zero.
SOMATIC_GLEASON_LANDMARKS = (0,)
SOMATIC_GLEASON_INDEX_ANALYSES = ("gleason", "sequencing", "prs")

# A cohort is a patient-subset restriction applied at Stage 3 via
# --restrict-to-mrns.  "all" is the arm's full Stage-1 survival cohort; the
# other two are retrospective, medication-derived ADT-intent strata.
#
# These are deliberately not entries in _ARM_SPECS: Stage 1 and Stage 2 still
# run once per treatment anchor, and only Stage 3 onward splits.  Treating a
# cohort as an ordinary arm would incorrectly ask compile_COMPASS_cohort_data.py
# for a new treatment anchor.
#
# `label_suffix` and `title_suffix` compose onto the arm's, so the directory
# convention is unchanged: cohort "metastatic" on arm "adt" yields the label
# "adt_metastatic" and therefore prediction_inputs_adt_metastatic /
# local_runs_adt_metastatic, exactly as the dedicated ADT-intent factory built
# them before the cohorts were folded into the main cross.
COHORT_SPECS = {
    "all": {
        "label_suffix": "",
        "title_suffix": "",
        "adt_intent": None,
        "retrospective": False,
    },
    "metastatic": {
        "label_suffix": "_metastatic",
        "title_suffix": " / metastatic",
        "adt_intent": "METASTATIC",
        "retrospective": True,
    },
    "localized": {
        "label_suffix": "_localized",
        "title_suffix": " / localized-adjuvant",
        "adt_intent": "LOCALIZED_ADJUVANT",
        "retrospective": True,
    },
}
DEFAULT_COHORTS = ("all", "metastatic", "localized")

# The MRN-list-backed cohorts, as a derived view: this is what
# build_adt_intent_mrn_lists() writes and what adt_intent_mrn_list_path()
# resolves.  Keyed the same way it was before the cohort registry existed.
ADT_INTENT_MODEL_STRATA = {
    key: {
        "intent": spec["adt_intent"],
        "label": f"adt{spec['label_suffix']}",
        "title": f"ADT{spec['title_suffix']}",
    }
    for key, spec in COHORT_SPECS.items()
    if spec["retrospective"]
}
ADT_INTENT_LABELS_FILENAME = "adt_intent_labels_model_cohort.csv"
ADT_INTENT_COUNTS_FILENAME = "adt_intent_endpoint_counts.csv"


def _require_adt_index_run(run: dict) -> None:
    anchor = str(run.get("anchor", run.get("label", ""))).lower()
    if anchor != "adt":
        raise ValueError(
            "The sequencing/Gleason/PRS analyses require the ADT arm "
            f"because observations are selected relative to ADT start; got {anchor!r}."
        )

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


def endpoint_suffix(endpoint: str) -> str:
    """The directory suffix for one endpoint: "" for platinum, "_<endpoint>" else.

    Platinum keeps the un-suffixed tree for historical continuity. The rule was
    inlined in four places before this helper; keep it here so a new endpoint
    cannot pick up a different convention in one of them.
    """
    return "" if str(endpoint).lower() == "platinum" else f"_{str(endpoint).lower()}"


def run_key(run: dict) -> tuple[str, str, str]:
    """Stable identity for one run, for keying summary dicts across notebooks."""
    return (run.get("cohort", DEFAULT_COHORT), run["endpoint"], run["label"])


def stage2_runs(runs: list[dict]) -> list[dict]:
    """One run per treatment anchor -- Stage 2's unit of work.

    Stage 2 (lab preprocessing) is endpoint- and cohort-independent: it runs
    once per anchor and every endpoint/cohort run reads its output. Encoding
    that here rather than as a `if run["endpoint"] != "platinum": continue`
    guard in the notebook keeps it correct as the cross widens -- that guard
    would have run Stage 2 once per cohort once cohorts were added.
    """
    seen: set[str] = set()
    unique = []
    for run in runs:
        if run["anchor"] in seen:
            continue
        seen.add(run["anchor"])
        unique.append(run)
    return unique


def make_runs(
    arms=("adt",),
    *,
    prediction_input_dirs: dict[str, str | Path] | None = None,
    output_suffix: str = "",
    endpoint: str | None = None,
    cohort: str = "all",
) -> list[dict]:
    """Build the merged-profile RUNS list, restricted to ``arms``.

    ARPI is disabled by default: build_prediction_inputs.py
    raised "No patients have both engineered features and valid outcomes" for
    that arm on the merged-parquet root. Re-enable once that is root-caused.

    ``prediction_input_dirs`` optionally overrides the prebuilt input directory
    for individual arms. This is primarily used by notebook 03b when GPU jobs
    run on a cluster whose prediction data are mounted somewhere else.

    ``endpoint`` defaults to the legacy module-level ``ENDPOINT`` knob. New
    callers should pass it explicitly so multiple endpoints can coexist in one
    notebook process.

    ``output_suffix`` appends to BOTH the prediction-inputs and the outputs
    directory names, giving a second endpoint its own parallel tree. Both are
    suffixed, not just outputs: the NEPC cohort is gated on ``t_nepc > 0``
    (see ``--require-nepc``), so its prediction inputs are a different set of
    patients and must not overwrite the platinum build. With
    ``output_suffix="_nepc"`` a run reads/writes ``prediction_inputs_adt_nepc``
    and ``local_runs_adt_nepc``, leaving the platinum tree untouched.

    ``cohort`` selects a patient-subset restriction from ``COHORT_SPECS``,
    applied at Stage 3 via ``--restrict-to-mrns``. It composes onto the arm
    label, so ``cohort="metastatic"`` on arm ``"adt"`` yields the label
    ``adt_metastatic`` and the trees ``prediction_inputs_adt_metastatic`` /
    ``local_runs_adt_metastatic``. The default ``"all"`` restricts to the arm's
    own Stage-1 survival cohort, i.e. no additional subsetting.
    """
    selected_endpoint = ENDPOINT if endpoint is None else str(endpoint).lower()
    selected_cohort = str(cohort).lower()
    if selected_cohort not in COHORT_SPECS:
        raise ValueError(
            f"Unknown cohort: {cohort!r} (expected one of {sorted(COHORT_SPECS)})"
        )
    cohort_spec = COHORT_SPECS[selected_cohort]
    valid_endpoints = set(_ca.ENDPOINTS)
    if selected_endpoint not in valid_endpoints:
        raise ValueError(
            f"Unknown endpoint: {selected_endpoint!r} (expected one of {sorted(valid_endpoints)})"
        )
    data_root = _PROFILE_OUTPUT_ROOT
    mrn_lists_dir = data_root / "mrn_lists"
    data_arpi = data_root / "longitudinal_prediction_data.csv"
    data_adt = data_root / "longitudinal_prediction_data_adt.csv"
    survival_output_root = data_root / "survival_analysis"

    runs = []
    prediction_input_dirs = prediction_input_dirs or {}
    unknown_overrides = set(prediction_input_dirs) - set(_ARM_SPECS)
    if unknown_overrides:
        raise ValueError(
            "Unknown prediction-input override arms: "
            f"{sorted(unknown_overrides)} (expected a subset of {sorted(_ARM_SPECS)})"
        )
    for label in arms:
        if label not in _ARM_SPECS:
            raise ValueError(f"Unknown arm: {label!r} (expected one of {sorted(_ARM_SPECS)})")
        arm_spec = _ARM_SPECS[label]
        # The cohort composes onto the arm to form the run label, which in turn
        # names both directory trees. Overrides stay keyed by the ARM, not the
        # composed label, so a caller overriding "adt" reaches every cohort.
        run_label = f"{label}{cohort_spec['label_suffix']}"
        # "all" restricts to the arm's own Stage-1 survival cohort (no extra
        # subsetting); the retrospective cohorts restrict to their MRN list.
        restrict_to_mrns = (
            data_root / f"prostate_{label}_survival_cohort_{label}.csv"
            if selected_cohort == "all"
            else adt_intent_mrn_list_path(selected_cohort, data_root=data_root)
        )
        runs.append({
            "label": run_label,
            "title": f"{arm_spec['title']}{cohort_spec['title_suffix']}",
            "variant": "profile_data",
            "anchor_col": "none",
            "anchor": arm_spec["anchor"],
            "landmarks": arm_spec["landmarks"],
            "input_csv": data_arpi if arm_spec["anchor"] == "arpi" else data_adt,
            "restrict_to_mrns": restrict_to_mrns,
            "inputs_dir": Path(
                prediction_input_dirs.get(
                    label,
                    survival_output_root / f"prediction_inputs_{run_label}{output_suffix}",
                )
            ).expanduser(),
            "output_dir": survival_output_root / f"local_runs_{run_label}{output_suffix}",
            "endpoint": selected_endpoint,
            "cohort": selected_cohort,
            "intent_stratum": None if selected_cohort == "all" else selected_cohort,
            "adt_intent": cohort_spec["adt_intent"],
            "retrospective_stratification": cohort_spec["retrospective"],
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
    print("endpoint:          ", selected_endpoint)
    for run in runs:
        print(
            f"{run['label']:30s}: anchor={run['anchor']:4s} landmarks={run['landmarks']} "
            f"inputs={run['inputs_dir']} outputs={run['output_dir']}"
        )
    return runs


def make_endpoint_runs(
    arms=("adt",),
    *,
    endpoints=("platinum", "nepc"),
    cohorts=DEFAULT_COHORTS,
    prediction_input_dirs_by_endpoint: dict[str, dict[str, str | Path]] | None = None,
) -> list[dict]:
    """Create independent input/output trees across cohort x endpoint x arm.

    This is the single run factory. The ADT-intent strata used to have their
    own parallel factory that could only reach three of the four endpoints and
    never reached Stage 3b or the figure pipeline; they are now ordinary
    cohorts in this cross. Directory names are unchanged, so existing trees
    resolve exactly as before.
    """
    overrides = prediction_input_dirs_by_endpoint or {}
    unknown = set(overrides) - set(endpoints)
    if unknown:
        raise ValueError(f"Prediction-input overrides supplied for unrequested endpoints: {sorted(unknown)}")
    unknown_cohorts = {str(c).lower() for c in cohorts} - set(COHORT_SPECS)
    if unknown_cohorts:
        raise ValueError(
            f"Unknown cohorts: {sorted(unknown_cohorts)} "
            f"(expected a subset of {sorted(COHORT_SPECS)})"
        )
    runs: list[dict] = []
    for raw_cohort in cohorts:
        for raw_endpoint in endpoints:
            endpoint = str(raw_endpoint).lower()
            runs.extend(
                make_runs(
                    arms,
                    endpoint=endpoint,
                    output_suffix=endpoint_suffix(endpoint),
                    cohort=str(raw_cohort).lower(),
                    prediction_input_dirs=overrides.get(endpoint),
                )
            )
    return runs


def adt_intent_mrn_list_path(
    stratum: str,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Return the Stage-3 MRN-list path for one medication-derived stratum."""
    key = str(stratum).lower()
    if key not in ADT_INTENT_MODEL_STRATA:
        raise ValueError(
            f"Unknown ADT-intent stratum: {stratum!r} "
            f"(expected one of {sorted(ADT_INTENT_MODEL_STRATA)})"
        )
    root = _PROFILE_OUTPUT_ROOT if data_root is None else Path(data_root)
    return root / "mrn_lists" / f"{ADT_INTENT_MODEL_STRATA[key]['label']}_mrns.csv"


def build_adt_intent_mrn_lists(
    *,
    medications_path: str | Path | None = None,
    base_cohort_csv: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Path]:
    """Classify the modelled ADT cohort and write localized/metastatic MRN lists.

    The classifier uses medication history over the observed follow-up.  These
    lists therefore define *retrospective strata* and must not be interpreted
    as metastatic-status predictions available at the landmark.  Platinum is
    intentionally absent from the classifier, but later treatment escalation
    and the eventual duration/cessation of ADT contribute to the label.

    The combined audit file retains all classifier fields.  Two narrow MRN
    files feed ``build_prediction_inputs.py``; a preliminary endpoint-count
    table makes a sparse localized endpoint visible before model fitting.
    """
    import polars as pl

    from COMPASS.data_preprocessing.classify_adt_intent import classify_adt_intent
    from COMPASS.data_preprocessing.compile_COMPASS_cohort_data import (
        parse_mixed_datetime_expr,
    )

    root = _PROFILE_OUTPUT_ROOT if data_root is None else Path(data_root)
    cohort_path = (
        root / "prostate_adt_survival_cohort_adt.csv"
        if base_cohort_csv is None
        else Path(base_cohort_csv)
    )
    meds_path = (
        PROFILE_SOURCES["MEDICATIONS"]
        if medications_path is None
        else Path(medications_path)
    )
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"ADT survival cohort not found: {cohort_path}. Run Stage 1 first."
        )
    if not meds_path.exists():
        raise FileNotFoundError(f"Medication source not found: {meds_path}")

    cohort = pl.read_csv(cohort_path, infer_schema_length=0).with_columns(
        pl.col("DFCI_MRN")
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64, strict=False)
        .alias("DFCI_MRN"),
        parse_mixed_datetime_expr("FOLLOW_UP_END_DATE").alias("FOLLOW_UP_END_DATE"),
    ).filter(pl.col("DFCI_MRN").is_not_null())
    cohort_mrns = cohort["DFCI_MRN"].unique().to_list()
    if not cohort_mrns:
        raise ValueError(f"ADT survival cohort contains no usable MRNs: {cohort_path}")

    meds = (
        scan_source(meds_path)
        .select("DFCI_MRN", "NCI_PREFERRED_MED_NM", "MED_START_DT")
        .with_columns(
            pl.col("DFCI_MRN")
            .cast(pl.Float64, strict=False)
            .cast(pl.Int64, strict=False)
            .alias("DFCI_MRN")
        )
        .filter(pl.col("DFCI_MRN").is_in(cohort_mrns))
        .collect()
    )
    follow_up = cohort.select("DFCI_MRN", "FOLLOW_UP_END_DATE").unique("DFCI_MRN")
    labels = classify_adt_intent(meds, follow_up=follow_up).filter(
        pl.col("DFCI_MRN").is_in(cohort_mrns)
    )

    missing_labels = set(cohort_mrns) - set(labels["DFCI_MRN"].to_list())
    if missing_labels:
        raise ValueError(
            f"ADT-intent classification omitted {len(missing_labels)} model-cohort MRNs. "
            "Confirm that every Stage-1 ADT patient has a recognized ADT medication row."
        )

    mrn_dir = root / "mrn_lists"
    mrn_dir.mkdir(parents=True, exist_ok=True)
    labels_path = mrn_dir / ADT_INTENT_LABELS_FILENAME
    labels.sort("DFCI_MRN").write_csv(labels_path)

    outputs: dict[str, Path] = {"labels": labels_path}
    for key, spec in ADT_INTENT_MODEL_STRATA.items():
        path = adt_intent_mrn_list_path(key, data_root=root)
        subset = labels.filter(pl.col("ADT_INTENT") == spec["intent"]).select(
            "DFCI_MRN", "ADT_INTENT", "HAS_POSITIVE_METASTATIC_EVIDENCE"
        ).sort("DFCI_MRN")
        if subset.height == 0:
            raise ValueError(f"ADT-intent stratum {spec['intent']} is empty.")
        subset.write_csv(path)
        outputs[key] = path
        print(f"  {spec['title']}: {subset.height:,} patients -> {path}")

    # Preliminary counts precede the PSA/PARPi/landmark filters, but expose an
    # endpoint with too few events before expensive Cox/XGBoost fitting starts.
    joined = labels.select("DFCI_MRN", "ADT_INTENT").join(
        cohort, on="DFCI_MRN", how="inner"
    )
    # Every modeled endpoint: these counts surface an event-starved cell before
    # fitting rather than after.
    count_rows = []
    for key, spec in ADT_INTENT_MODEL_STRATA.items():
        stratum = joined.filter(pl.col("ADT_INTENT") == spec["intent"])
        for endpoint in _ca.ENDPOINTS:
            event_col = _ca.ENDPOINTS[endpoint]["event_col"]
            duration_col = "TT_" + endpoint.upper()
            if event_col not in stratum.columns or duration_col not in stratum.columns:
                count_rows.append(
                    {
                        "stratum": key,
                        "adt_intent": spec["intent"],
                        "endpoint": endpoint,
                        "n_patients": stratum.height,
                        "n_incident_eligible": None,
                        "n_incident_events": None,
                    }
                )
                continue
            durations = stratum[duration_col].cast(pl.Float64, strict=False)
            events = stratum[event_col].cast(pl.Int64, strict=False).fill_null(0)
            eligible = durations.is_not_null() & durations.gt(0)
            count_rows.append(
                {
                    "stratum": key,
                    "adt_intent": spec["intent"],
                    "endpoint": endpoint,
                    "n_patients": stratum.height,
                    "n_incident_eligible": int(eligible.sum()),
                    "n_incident_events": int((eligible & events.eq(1)).sum()),
                }
            )
    counts_path = mrn_dir / ADT_INTENT_COUNTS_FILENAME
    pl.DataFrame(count_rows).write_csv(counts_path)
    outputs["counts"] = counts_path
    print(f"  preliminary endpoint counts -> {counts_path}")
    print(
        "  NOTE: ADT_INTENT uses full observed medication history; these are "
        "retrospective strata, not landmark-available metastatic-status labels."
    )
    return outputs


# ---------------------------------------------------------------------------
# Stage 0 -- schema audit
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


def audit_schema() -> None:
    """Fail fast if a required column is absent or all-null in profile_data."""

    import polars as pl

    sources = PROFILE_SOURCES
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


def compile_cohort(
    arms=("adt",),
    dry_run: bool = False,
    *,
    avpc_nepc_labels_path: str | Path | None = None,
) -> None:
    data_root = _PROFILE_OUTPUT_ROOT
    cmd = [
        PYTHON, DATA_PREPROCESSING_DIR / "compile_COMPASS_cohort_data.py",
        "--icd-source", PROFILE_SOURCES["EHR_DIAGNOSES"],
        "--medications-source", PROFILE_SOURCES["MEDICATIONS"],
        "--patient-status-source", PROFILE_SOURCES["PT_INFO_STATUS_REGISTRATION"],
        "--labs-csv", PROFILE_SOURCES["LABS"],
        "--out-dir", data_root,
        "--mrn-lists-dir", data_root / "mrn_lists",
        "--survival-arms", *arms,
    ]
    if avpc_nepc_labels_path is not None:
        cmd += ["--avpc-nepc-labels", avpc_nepc_labels_path]
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
    data_root = _PROFILE_OUTPUT_ROOT
    label = run["label"]
    output_csv = run["input_csv"]
    cache_parquet = data_root / f"consolidated_longitudinal_data_{label}.parquet"
    icd_csv = data_root / "prostate_icd_data.csv"

    cmd = [
        PYTHON, DATA_PREPROCESSING_DIR / "longitudinal_data_processing.py",
        "--health-csv", PROFILE_SOURCES["HEALTH_HISTORY"],
        "--labs-csv", PROFILE_SOURCES["LABS"],
        "--medications-csv", PROFILE_SOURCES["MEDICATIONS"],
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
        "longitudinal_landmark*.csv",
        "longitudinal_landmark*_manifest.json",
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
    # The builder applies only this endpoint's validity gate. Thus pre-anchor
    # platinum does not remove NEPC patients, and prevalent NEPC does not remove
    # platinum patients.
    cmd += ["--endpoint", run.get("endpoint", ENDPOINT)]
    rc = _run(cmd, dry_run=dry_run)
    if not dry_run and rc != 0:
        raise RuntimeError(f"build_prediction_inputs failed for {run['label']} with rc={rc}")


def build_somatic_gleason_inputs(run: dict, dry_run: bool = False) -> None:
    """Build separate Gleason-, sequencing-, and PRS-indexed cohorts."""
    _require_adt_index_run(run)
    output_dir = run["inputs_dir"] / "somatic_gleason"
    print(f"\n========== build somatic + Gleason inputs: {run['title']} ==========")
    cmd = [
        PYTHON, DATA_PREPROCESSING_DIR / "build_somatic_gleason_inputs.py",
        "--base-inputs-dir", run["inputs_dir"],
        "--output-dir", output_dir,
        "--endpoint", run.get("endpoint", ENDPOINT),
        "--landmark-days", *[str(lm) for lm in SOMATIC_GLEASON_LANDMARKS],
    ]
    rc = _run(cmd, dry_run=dry_run)
    if not dry_run and rc != 0:
        raise RuntimeError(
            f"build_somatic_gleason_inputs failed for {run['label']} with rc={rc}"
        )


def cohort_diagnostics(run: dict) -> None:
    print(f"\n========== cohort diagnostics: {run['title']} ==========")
    endpoint = run.get("endpoint", ENDPOINT)
    event_col = _ca.ENDPOINTS[endpoint]["event_col"]
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

        if event_col not in agg.columns:
            print(f"=== landmark +{lm}d | {event_col} absent, skipping ===")
            continue
        n_events = int(agg[event_col].sum())
        print(
            f"=== landmark +{lm}d | n_total={len(agg):,} "
            f"n_{event_col}={n_events} ==="
        )
        for lab_substr in ("Testosterone", "PSA", "Prostate specific Ag"):
            for stat in ("mean", "last", "max", "min"):
                col = find_col(lab_substr, stat)
                if col is None:
                    continue
                for ev in (0, 1):
                    sub = agg.loc[agg[event_col] == ev, col].dropna()
                    if sub.empty:
                        continue
                    print(
                        f"  {lab_substr:>22s} {stat:5s} {event_col}={ev}: "
                        f"median={sub.median():>10.2f} "
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

PSA_SCALE_SUPPLEMENT_DIR = "cox_psa_scale_supplement"
PSA_SCALE_SUPPLEMENT_FILENAME = "psa_scale_univariate.csv"
PSA_RAW_LAB_NAME = "PSA_raw"
PSA_LOG_LAB_NAME = "PSA_log1p"

MULTIVARIATE_TASK_SPECS = [
    ("elastic-net", "both", "cox_agg_multivariable_metrics.csv"),
    ("elastic-net", "baseline", "cox_agg_baseline_metrics.csv"),
    ("xgboost", "both", "landmark_xgboost_metrics.csv"),
    ("xgboost", "baseline", "landmark_xgboost_baseline_metrics.csv"),
]

# Per-endpoint config pair for the longitudinal arm. config_dir doubles as
# --config for both models, reusing tasks_for_run's existing
# (model, config_dir, metrics_filename) x landmarks cross with zero changes to
# _run_tasks. metrics_filename is None for survlatent-ode because its filename
# embeds --run-id (survlatent_ode_test_metrics_{run_id}.csv); see
# longitudinal_metrics_filename() below for the resolved name.
_LONGITUDINAL_CONFIGS_BY_ENDPOINT = {
    "platinum": ("platinum", "competing"),
    "nepc": ("nepc", "nepc_competing"),
    "avpc": ("avpc", "avpc_competing"),
}

# SurvLatent ODE needs the bundled external checkout and its own conda env
# (see multivariate_longitudinal/README.md). Off by default so 03b runs with
# Dynamic-DeepHit alone; set to True once that environment is available.
RUN_SURVLATENT = False


def longitudinal_task_specs(endpoint: str = None, *, include_survlatent: bool = None):
    """(model, config_dir, metrics_filename) triples for one endpoint.

    The cause-only config is listed first so it leads the summary table; it is
    the row comparable to that endpoint's Cox/XGBoost arms.
    """
    endpoint = ENDPOINT if endpoint is None else endpoint
    if include_survlatent is None:
        include_survlatent = RUN_SURVLATENT
    if endpoint not in _LONGITUDINAL_CONFIGS_BY_ENDPOINT:
        valid = ", ".join(sorted(_LONGITUDINAL_CONFIGS_BY_ENDPOINT))
        raise ValueError(
            f"No longitudinal configs registered for endpoint {endpoint!r}. "
            f"Known: {valid}."
        )
    configs = _LONGITUDINAL_CONFIGS_BY_ENDPOINT[endpoint]
    specs = [
        ("dynamic-deephit", cfg, f"dynamic_deephit_metrics_{cfg}.csv") for cfg in configs
    ]
    if include_survlatent:
        specs.extend(("survlatent-ode", cfg, None) for cfg in configs)
    return specs


# Backwards-compatible module-level view for the default endpoint. Prefer
# longitudinal_task_specs(), which honors its explicit endpoint (or the legacy
# ENDPOINT fallback) and RUN_SURVLATENT at call time rather than at import time.
LONGITUDINAL_TASK_SPECS = longitudinal_task_specs("platinum", include_survlatent=True)


def longitudinal_run_id(config_dir: str, landmark: int) -> str:
    """Matches survlatent_ode.py's default --run-id (prostate_<config>_landmark<D>_v1)."""
    return f"prostate_{config_dir}_landmark{landmark}_v1"


def longitudinal_metrics_filename(model: str, config_dir: str, landmark: int, metrics_filename) -> str:
    if metrics_filename is not None:
        return metrics_filename
    if model == "survlatent-ode":
        return f"survlatent_ode_test_metrics_{longitudinal_run_id(config_dir, landmark)}.csv"
    raise ValueError(f"No metrics filename known for model={model!r}.")


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
    mapping = {
        "univariate": "cox",
        "elastic-net": "cox",
        "xgboost": "xgboost",
        "dynamic-deephit": "multivariate_longitudinal/dynamic_deephit",
        "survlatent-ode": "multivariate_longitudinal/survlatent_ode",
    }
    if model not in mapping:
        raise ValueError(f"Unknown model: {model}")
    return mapping[model]


def build_model_command(model, landmark, config_dir, row_output_dir, run):
    overwrite_flag = "--overwrite" if FORCE_RERUN else "--no-overwrite"
    # Stamped onto the canonical `cohort` metrics column so a metrics CSV names
    # the patient subset it was fit on. The restriction itself happens upstream
    # at Stage 3 via --restrict-to-mrns; this only records it. Runs predating
    # the cohort key fall back to the unrestricted default.
    cohort_args = ["--cohort", str(run.get("cohort") or DEFAULT_COHORT)]
    if model == "univariate":
        return [
            PYTHON, SURVIVAL_DIR / "univariate_analysis.py",
            "--inputs-dir", run["inputs_dir"],
            "--output-dir", row_output_dir,
            "--landmark-days", str(landmark),
            "--endpoints", run.get("endpoint", ENDPOINT),
            overwrite_flag,
        ]
    if model == "elastic-net":
        cmd = [
            PYTHON, SURVIVAL_DIR / "multivariate_analysis.py",
            "--model", "elastic-net",
            "--inputs-dir", run["inputs_dir"],
            "--output-dir", row_output_dir,
            "--landmark-days", str(landmark),
            "--endpoints", run.get("endpoint", ENDPOINT),
            "--n-folds", str(N_FOLDS),
            *cohort_args,
            overwrite_flag,
        ]
        if config_dir == "baseline":
            cmd.append("--baseline")
        return cmd
    if model == "dynamic-deephit":
        return [
            PYTHON, SURVIVAL_DIR / "multivariate_longitudinal" / "dynamic_deephit.py",
            "--inputs-dir", run["inputs_dir"],
            "--output-dir", row_output_dir,
            "--landmark-day", str(landmark),
            "--config", config_dir,
            "--max-pred-window", str(MAX_PRED_WINDOW),
            *cohort_args,
            overwrite_flag,
        ]
    if model == "survlatent-ode":
        if not SURVLATENT_REPO:
            raise RuntimeError(
                "compass_pipeline.SURVLATENT_REPO is not set. Set it to the path of a "
                "cloned itmoon7/survlatent_ode repo (with its conda env active) before "
                "running survlatent-ode tasks -- see multivariate_longitudinal/README.md."
            )
        cmd = [
            PYTHON, SURVIVAL_DIR / "multivariate_longitudinal" / "survlatent_ode.py",
            "--survlatent-repo", str(SURVLATENT_REPO),
            "--inputs-dir", run["inputs_dir"],
            # Absolute: survlatent_ode.py's import_survlatent() chdirs into the
            # external repo, so a relative --output-dir would resolve there instead.
            "--output-dir", str(Path(row_output_dir).resolve()),
            "--landmark-day", str(landmark),
            "--config", config_dir,
            "--run-id", longitudinal_run_id(config_dir, landmark),
            "--max-pred-window", str(MAX_PRED_WINDOW),
            overwrite_flag,
        ]
        if FORCE_RERUN:
            # A forced refit is a fresh fit: clear this run_id's checkpoints too,
            # or prepare_run_artifacts() aborts on the leftover artifacts.
            cmd.append("--overwrite-run")
        return cmd
    if model == "xgboost":
        cmd = [
            PYTHON, SURVIVAL_DIR / "multivariate_analysis.py",
            "--model", "xgboost",
            "--inputs-dir", run["inputs_dir"],
            "--output-dir", row_output_dir,
            "--landmark-days", str(landmark),
            "--endpoints", run.get("endpoint", ENDPOINT),
            "--n-folds", str(N_FOLDS),
            *cohort_args,
            overwrite_flag,
        ]
        if config_dir == "baseline":
            cmd.append("--baseline")
        return cmd
    raise ValueError(f"Unknown model: {model}")


def _run_tasks(run: dict, specs, dry_run: bool = False):
    print(f"\n========== run models: {run['title']} ==========")
    tasks = tasks_for_run(run, specs)
    summary = []
    for model, landmark, config_dir, metrics_filename in tasks:
        row_output_dir = run["output_dir"] / model_output_dir(model) / f"landmark_{landmark}" / config_dir
        metrics_path = row_output_dir / longitudinal_metrics_filename(model, config_dir, landmark, metrics_filename)
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
    """Per-landmark univariate arm."""
    summary = _run_tasks(run, UNIVARIATE_TASK_SPECS, dry_run=dry_run)
    print("\n=== run summary ===")
    for tag, status, elapsed in summary:
        print(f"  {tag} {status:>20s} {elapsed/60:6.1f} min")
    return summary


def build_psa_scale_supplement_data(
    ctx,
    *,
    landmark_day: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Recompute matched raw- and log1p-PSA summaries from row-level labs.

    Both scales use the same nonnegative PSA measurements and the same
    pre-landmark window. Recomputing before aggregation is important because
    mean(log1p(PSA)) and changes on the log scale cannot be recovered by simply
    transforming the already-aggregated mean/delta columns.
    """
    required = {"DFCI_MRN", "LAB_NAME", "LAB_VALUE", "t_lab"}
    missing = required - set(ctx.pre_treatment_lab_df.columns)
    if missing:
        raise ValueError(
            "PSA scale supplement is missing pre-landmark lab columns: "
            f"{sorted(missing)}"
        )

    psa_long = ctx.pre_treatment_lab_df.copy()
    psa_long["LAB_NAME"] = psa_long["LAB_NAME"].astype(str).str.strip()
    psa_long["LAB_VALUE"] = pd.to_numeric(psa_long["LAB_VALUE"], errors="coerce")
    psa_long = psa_long.loc[
        psa_long["LAB_NAME"].str.casefold().eq("psa")
        & psa_long["LAB_VALUE"].ge(0)
    ].copy()
    if psa_long.empty:
        raise ValueError(
            f"No nonnegative PSA observations are available at landmark +{landmark_day}d."
        )

    raw_matrix = _ca.build_feature_matrix(
        psa_long,
        landmark_offset_days=landmark_day,
        anchor_col=None,
    )
    log_long = psa_long.copy()
    log_long["LAB_VALUE"] = np.log1p(log_long["LAB_VALUE"])
    log_matrix = _ca.build_feature_matrix(
        log_long,
        landmark_offset_days=landmark_day,
        anchor_col=None,
    )

    def relabel(matrix: pd.DataFrame, lab_name: str) -> pd.DataFrame:
        return matrix.rename(
            columns={
                column: f"{lab_name}__{_ca.parse_feature_name(column)[1]}"
                for column in matrix.columns
            }
        )

    raw_matrix = relabel(raw_matrix, PSA_RAW_LAB_NAME)
    log_matrix = relabel(log_matrix, PSA_LOG_LAB_NAME)
    supplement_features = raw_matrix.join(log_matrix, how="outer")
    supplement_data = ctx.univariate_data.join(supplement_features, how="left")
    feature_cols = [
        column
        for column in supplement_features.columns
        if _ca.parse_feature_name(column)[1] != "n_observations"
    ]
    return supplement_data, feature_cols


def run_psa_scale_supplement(run: dict) -> pd.DataFrame:
    """Fit matched raw-vs-log1p PSA univariate Cox models as a supplement."""
    supplement_root = run["output_dir"] / PSA_SCALE_SUPPLEMENT_DIR
    combined_path = supplement_root / PSA_SCALE_SUPPLEMENT_FILENAME
    manifest = _ca._load_build_manifest(run["inputs_dir"])
    min_patient_coverage = float(manifest["min_patient_coverage"])

    for landmark_day in run["landmarks"]:
        output_dir = supplement_root / f"landmark_{landmark_day}" / "both"
        output_path = output_dir / PSA_SCALE_SUPPLEMENT_FILENAME
        if output_path.exists() and not FORCE_RERUN:
            print(
                f"[skip] {run['label']} PSA scale supplement landmark_{landmark_day} "
                f"-> {output_path.relative_to(run['output_dir'])} exists"
            )
            continue

        ctx = _ca.prepare_landmark_context(
            run["inputs_dir"],
            landmark_day,
            min_patient_coverage=min_patient_coverage,
        )
        supplement_data, feature_cols = build_psa_scale_supplement_data(
            ctx,
            landmark_day=landmark_day,
        )
        baseline_covariate_cols = tuple(_ca.panel_version_covariate_columns(ctx.merged))
        associations = _ca.run_univariate_nobs_adjusted_associations(
            supplement_data,
            feature_cols=feature_cols,
            endpoint=run.get("endpoint", ENDPOINT),
            min_events_per_feature=_ca.DEFAULT_MIN_EVENTS_PER_FEATURE,
            fallback_penalizer=0.05,
            baseline_covariate_cols=baseline_covariate_cols,
        )
        associations.insert(0, "landmark_days", landmark_day)
        associations.insert(1, "cohort", run.get("cohort", DEFAULT_COHORT))
        associations.insert(
            5,
            "psa_scale",
            associations["lab_name"].map(
                {PSA_RAW_LAB_NAME: "raw", PSA_LOG_LAB_NAME: "log1p"}
            ),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        associations.to_csv(output_path, index=False)
        print(
            f"[done] {run['label']} PSA scale supplement landmark_{landmark_day} "
            f"-> {output_path.relative_to(run['output_dir'])}"
        )

    frames = []
    for landmark_day in run["landmarks"]:
        path = (
            supplement_root
            / f"landmark_{landmark_day}"
            / "both"
            / PSA_SCALE_SUPPLEMENT_FILENAME
        )
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise FileNotFoundError(
            f"No PSA scale supplement results were produced under {supplement_root}."
        )
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    print(f"[done] combined PSA scale supplement -> {combined_path}")
    return combined


def run_somatic_gleason_univariate(run: dict, dry_run: bool = False):
    """Run Gleason-, sequencing-, and ADT-indexed PRS Cox analyses."""
    _require_adt_index_run(run)
    print(f"\n========== run sequencing + Gleason + PRS univariate: {run['title']} ==========")
    summary = []
    for analysis in SOMATIC_GLEASON_INDEX_ANALYSES:
        landmark = 0
        row_output_dir = (
            run["output_dir"]
            / "cox_somatic_gleason"
            / f"landmark_{landmark}"
            / analysis
            / "both"
        )
        metrics_path = row_output_dir / "cox_agg_univariate_nobs_adjusted.csv"
        tag = f"{run['label']:28s} {analysis:<10s} index-date"
        if metrics_path.exists() and not FORCE_RERUN:
            print(f"[skip] {tag} -> {metrics_path.relative_to(run['output_dir'])} exists")
            summary.append((tag, "skipped", 0.0))
            continue
        if not dry_run:
            row_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON, SURVIVAL_DIR / "univariate_analysis.py",
            "--inputs-dir", run["inputs_dir"] / "somatic_gleason" / analysis,
            "--output-dir", row_output_dir,
            "--landmark-days", str(landmark),
            "--endpoints", run.get("endpoint", ENDPOINT),
            "--feature-set", "somatic-gleason",
            "--overwrite" if FORCE_RERUN else "--no-overwrite",
        ]
        print(f"[run ] {tag}")
        t0 = time.time()
        rc = _run(cmd, dry_run=dry_run)
        elapsed = time.time() - t0
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[done] {tag} -> {status} ({elapsed/60:.1f} min)\n")
        summary.append((tag, status, elapsed))
    return summary


def run_multivariate(run: dict, dry_run: bool = False):
    """Elastic-net (both/baseline) and XGBoost (both/baseline) arms."""
    summary = _run_tasks(run, MULTIVARIATE_TASK_SPECS, dry_run=dry_run)
    print("\n=== run summary ===")
    for tag, status, elapsed in summary:
        print(f"  {tag} {status:>20s} {elapsed/60:6.1f} min")
    return summary


def run_multivariate_longitudinal(run: dict, dry_run: bool = False):
    """Dynamic-DeepHit (and optionally SurvLatent ODE) in this run's configs.

    Configs follow the run's endpoint: platinum/competing, or nepc/
    nepc_competing. SurvLatent ODE is included only when RUN_SURVLATENT is on.
    """
    specs = longitudinal_task_specs(run.get("endpoint", ENDPOINT))
    if not RUN_SURVLATENT:
        print("[survlatent-ode] disabled (compass_pipeline.RUN_SURVLATENT = False)")
    summary = _run_tasks(run, specs, dry_run=dry_run)
    print("\n=== run summary ===")
    for tag, status, elapsed in summary:
        print(f"  {tag} {status:>20s} {elapsed/60:6.1f} min")
    return summary


# summarize_outputs and summarize_longitudinal_outputs emit a stable, narrower
# shape than the on-disk metrics schema: the bare metric names below are this
# summary frame's own columns, not the canonical CSV spellings they read from.
_SUMMARY_METRIC_FIELDS = (
    "n_test",
    "n_test_events",
    "c_index",
    "mean_auc_t",
    "integrated_brier",
)


def _missing_metric_fields(status: str) -> dict:
    return {**{field: None for field in _SUMMARY_METRIC_FIELDS}, "status": status}


def _canonical_metric_fields(row) -> dict:
    """Map one canonical-schema metrics row onto the summary frame's columns."""
    return {
        "n_test": int(row["n_test"]),
        "n_test_events": int(row["n_events_test"]),
        "c_index": float(row["test_c_index"]),
        "mean_auc_t": float(row["test_mean_auc_t"]),
        "integrated_brier": float(row["test_integrated_brier"]),
        "status": "ok",
    }


def summarize_outputs(run: dict) -> pd.DataFrame:
    # Reads back whichever endpoint this run modeled; the "endpoint" column it
    # emits is what lets platinum and NEPC summaries be concatenated and
    # compared side by side (see 07_endpoint_comparison.ipynb).
    endpoint = run.get("endpoint", ENDPOINT)
    rows = []
    for model, landmark, config_dir, metrics_filename in tasks_for_run(run, MULTIVARIATE_TASK_SPECS):
        metrics_path = run["output_dir"] / model_output_dir(model) / f"landmark_{landmark}" / config_dir / metrics_filename
        base = {"run": run["label"], "model": model, "landmark": landmark, "config": config_dir, "endpoint": endpoint}
        if not metrics_path.exists():
            rows.append({**base, **_missing_metric_fields("missing")})
            continue
        df = pd.read_csv(metrics_path)
        platinum = df.loc[df["endpoint"] == endpoint]
        if platinum.empty:
            rows.append({**base, **_missing_metric_fields(f"no {endpoint} row")})
            continue
        # Elastic-net and XGBoost both write the canonical schema
        # (survival_common/metrics_schema.py), so one read serves both.
        rows.append({**base, **_canonical_metric_fields(platinum.iloc[0])})
    return pd.DataFrame(rows).sort_values(["run", "landmark", "model", "config"]).reset_index(drop=True)


def summarize_longitudinal_outputs(run: dict) -> pd.DataFrame:
    """Same schema as summarize_outputs, for the longitudinal arms.

    A separate function rather than an extension of summarize_outputs because
    of survlatent-ode, whose metrics come from the external repo's eval_model
    and are not ours to pin to the canonical schema. Dynamic-DeepHit does write
    the canonical schema and is read the same way as Cox/XGBoost. Both models
    report a row per cause -- the competing configs report death too, as a
    secondary diagnostic -- so filtering to the run's cause of interest here
    keeps the headline comparison against that endpoint's Cox/XGBoost row
    valid. Frames from both functions share columns and concat cleanly.
    """
    endpoint = run.get("endpoint", ENDPOINT)
    rows = []
    for model, landmark, config_dir, metrics_filename in tasks_for_run(run, longitudinal_task_specs(endpoint)):
        resolved_filename = longitudinal_metrics_filename(model, config_dir, landmark, metrics_filename)
        metrics_path = run["output_dir"] / model_output_dir(model) / f"landmark_{landmark}" / config_dir / resolved_filename
        base = {"run": run["label"], "model": model, "landmark": landmark, "config": config_dir, "endpoint": endpoint}
        if not metrics_path.exists():
            rows.append({**base, **_missing_metric_fields("missing")})
            continue
        df = pd.read_csv(metrics_path)
        if model == "dynamic-deephit":
            cause = df.loc[df["endpoint"] == endpoint]
            if cause.empty:
                rows.append({**base, **_missing_metric_fields(f"no {endpoint} row")})
                continue
            rows.append({**base, **_canonical_metric_fields(cause.iloc[0])})
        elif model == "survlatent-ode":
            # eval_model's exact column names live in the external repo; try
            # the conventional ones and fall back to a "present but
            # unparsed" status rather than raising, since a schema drift
            # there shouldn't take down the whole summary.
            try:
                event_col = next(c for c in ("event", "endpoint") if c in df.columns)
                cause = df.loc[df[event_col].astype(str).str.lower() == endpoint]
                row = cause.iloc[0] if not cause.empty else df.iloc[0]
                rows.append({
                    **base,
                    "n_test": int(row.get("n_test", row.get("n", np.nan))) if pd.notna(row.get("n_test", row.get("n", np.nan))) else None,
                    "n_test_events": int(row.get("n_test_events", row.get("n_events", np.nan))) if pd.notna(row.get("n_test_events", row.get("n_events", np.nan))) else None,
                    "c_index": float(row.get("c_index", np.nan)),
                    "mean_auc_t": float(row.get("mean_auc", row.get("mean_auc_t", np.nan))),
                    "integrated_brier": float(row.get("ibs", row.get("integrated_brier", np.nan))),
                    "status": "ok",
                })
            except (StopIteration, IndexError, KeyError, ValueError) as exc:
                rows.append({**base, **_missing_metric_fields(f"unparsed ({exc})")})
    return pd.DataFrame(rows).sort_values(["run", "landmark", "model", "config"]).reset_index(drop=True)


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
    """Load per-landmark univariate result CSVs for this run."""
    import re

    run_dir = run["output_dir"]
    paths = sorted((run_dir / "cox").glob("landmark_*/both/cox_agg_univariate_nobs_adjusted.csv"))

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


def load_somatic_gleason_univariate_results(run: dict) -> pd.DataFrame:
    """Load outputs from the Gleason- and sequencing-indexed runs."""
    import re

    result_root = run["output_dir"] / "cox_somatic_gleason"
    paths = sorted(
        result_root.glob(
            "landmark_*/*/both/cox_agg_univariate_nobs_adjusted.csv"
        )
    )
    if not paths:
        raise FileNotFoundError(
            f"No somatic + Gleason univariate results found under {result_root}."
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "landmark_days" not in frame.columns:
            match = re.search(r"landmark_(-?\d+)", str(path.parent.parent.parent))
            if match is None:
                raise ValueError(f"Could not infer landmark from {path}")
            frame.insert(0, "landmark_days", int(match.group(1)))
        frame.insert(0, "index_analysis", path.parent.parent.name)
        frame["source_path"] = str(path)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    results["p_value"] = pd.to_numeric(results["p_value"], errors="coerce")
    results.insert(0, "cohort", run["label"])
    return results
