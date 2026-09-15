"""Prepare immutable Polars figure tables and source fingerprints for 05_figures.

Run with --config CONFIG.json (also written automatically by the Rmd). Patient
tables stay under the data root, never under the public figure output tree.
The large longitudinal CSV is scanned once per changed treatment arm. R keeps
the established survival/GAM/statistical estimators and caches their outputs
separately. No analysis models are refit here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import polars as pl

ID = "DFCI_MRN"
PATIENT_COLUMNS = [
    ID, "AGE_AT_TREATMENTSTART", "FIRST_RECORD_DATE", "DIAGNOSIS_DATE",
    "TREATMENT_ANCHOR_DATE", "LAST_CONTACT_DATE", "DEATH", "PLATINUM_MEDICATION",
    "PLATINUM_DATE", "PLATINUM", "t_diagnosis", "t_first_treatment", "t_platinum",
    "t_last_contact", "t_death", "t_dx_to_anchor", "NEPC", "t_nepc",
]
DATES = ["FIRST_RECORD_DATE", "DIAGNOSIS_DATE", "TREATMENT_ANCHOR_DATE",
         "LAST_CONTACT_DATE", "PLATINUM_DATE"]
NUMERIC = set(PATIENT_COLUMNS) - {ID, "PLATINUM_MEDICATION", *DATES}
PRE, POST, COVERAGE_PRE, BIN_WIDTH = 365.25, 1826.25, 1826.25, 180
ARM_TABLES = ["patients", "canonical", "patient_bins", "coverage_patient", "coverage_bins"]


def canonical_lab_names() -> list[str]:
    """Read the renderer's flat category declarations so the notebook cannot drift."""
    source = Path(__file__).with_name("COMPASS_generate_figures_pipeline.R").read_text()
    names = []
    for group in ["CBC", "CMP", "LFT", "VITALS", "ANDROGEN", "OTHER"]:
        match = re.search(rf"^{group} <- c\((.*?)\)", source, re.M | re.S)
        if match is None:
            raise ValueError(f"Cannot read canonical lab group {group} from R renderer")
        names.extend(re.findall(r'"([^"\n]+)"', match.group(1)))
    return names


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def fingerprint(path: Path) -> dict:
    path = path.resolve()
    if not path.exists():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def code_version() -> str:
    here = Path(__file__).parent
    paths = sorted(here.glob("*.R")) + [here / "05_figures.Rmd", Path(__file__), here / "prepare_metastatic_figure_labels.py"]
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
    os.replace(temporary, path)


@contextmanager
def preparation_lock(root: Path):
    # Cluster workers and simultaneous knits must not publish partial caches.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def date_expr(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.String)
    return pl.coalesce(value.str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                       value.str.strptime(pl.Date, "%m/%d/%Y", strict=False))


def bin_edges(pre=PRE, post=POST) -> list[float]:
    return sorted({-pre, post, 0.0,
                   *(-float(x) for x in range(0, math.floor(pre) + 1, BIN_WIDTH)),
                   *(float(x) for x in range(0, math.floor(post) + 1, BIN_WIDTH))})


def bin_expr(edges: list[float], column="t_rel") -> pl.Expr:
    # Left-closed bins; the final right endpoint is included, as in R cut().
    result = pl.lit(None, dtype=pl.Int32)
    for i in reversed(range(len(edges) - 1)):
        upper = pl.col(column) <= edges[i + 1] if i == len(edges) - 2 else pl.col(column) < edges[i + 1]
        result = pl.when((pl.col(column) >= edges[i]) & upper).then(i).otherwise(result)
    return result.alias("bin_id")


def prepare_arm(source: Path, labs: list[str]) -> dict[str, pl.DataFrame]:
    scan = pl.scan_csv(source, infer_schema_length=0, null_values=["", "NA", "NaN", "nan"])
    names = scan.collect_schema().names()
    required = [ID, "LAB_NAME", "LAB_VALUE", "LAB_DATE", "t_lab"]
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f"{source}: missing {sorted(missing)}")
    columns = list(dict.fromkeys([*PATIENT_COLUMNS, *required]))
    collect_options = {"engine": "streaming"} if "engine" in inspect.signature(pl.LazyFrame.collect).parameters else {"streaming": True}
    data = scan.select([c for c in columns if c in names]).collect(**collect_options)
    data = data.with_columns([
        pl.col(c).cast(pl.Float64, strict=False).fill_nan(None)
        for c in [*NUMERIC, "LAB_VALUE", "t_lab"] if c in data.columns
    ])
    patients = data.select([c for c in PATIENT_COLUMNS if c in data.columns]).unique(ID, keep="first", maintain_order=True)
    patients = patients.with_columns([date_expr(c).alias(c) for c in DATES if c in patients.columns])
    if {"DIAGNOSIS_DATE", "TREATMENT_ANCHOR_DATE"} <= set(patients.columns):
        patients = patients.with_columns((pl.col("TREATMENT_ANCHOR_DATE") - pl.col("DIAGNOSIS_DATE")).dt.total_days().cast(pl.Float64).alias("t_dx_to_anchor"))
    spans = (data.filter(pl.col("LAB_NAME").is_not_null())
             .select(ID, date_expr("LAB_DATE").alias("date"))
             .group_by(ID).agg(pl.len().cast(pl.Int32).alias("lab_rows"),
                 (pl.col("date").max() - pl.col("date").min()).dt.total_days().cast(pl.Float64)
                 .fill_null(float("-inf")).alias("record_span_days")))
    join_options = {"nulls_equal": True} if "nulls_equal" in inspect.signature(pl.DataFrame.join).parameters else {"join_nulls": True}
    patients = patients.with_row_index("_order").join(spans, on=ID, how="left", **join_options).sort("_order").drop("_order")
    raw = pl.col("LAB_NAME").str.to_lowercase()
    mapping = {lab.lower(): lab for lab in labs}
    canonical = (data.with_columns(
        pl.when(raw.str.contains("prostate specific ag", literal=True)).then(pl.lit("PSA"))
        .otherwise(raw.replace_strict(mapping, default=None)).alias("LAB_GROUP"))
        .filter(pl.col("LAB_GROUP").is_in(labs))
        .select([c for c in [ID, "LAB_NAME", "LAB_VALUE", "t_lab", "PLATINUM", "t_platinum", "LAB_GROUP"] if c in data.columns or c == "LAB_GROUP"])
        .drop_nulls(["t_lab", "LAB_VALUE"])
        .with_columns(pl.col("t_lab").alias("t_rel")))
    del data
    edges = bin_edges()
    mids = {i: (a+b)/2 for i, (a,b) in enumerate(zip(edges, edges[1:]))}
    keys = [ID, "LAB_GROUP", "bin_id"]
    # Retain measurement-level platinum status to match existing R grouping.
    if "PLATINUM" in canonical.columns:
        keys.append("PLATINUM")
    binned = canonical.with_columns(bin_expr(edges)).drop_nulls("bin_id")
    bins = binned.group_by(keys).agg(
        pl.col("LAB_VALUE").mean().alias("value_raw"),
        pl.col("LAB_VALUE").filter(pl.col("LAB_VALUE").is_finite() & (pl.col("LAB_VALUE") >= 0)).log1p().mean().alias("value_log"),
        pl.len().alias("n_raw"),
        (pl.col("LAB_VALUE").is_finite() & (pl.col("LAB_VALUE") >= 0)).sum().alias("n_log"),
    ).with_columns(pl.col("bin_id").replace_strict(mids, return_dtype=pl.Float64).alias("t_mid"))
    pre = canonical.filter((pl.col("t_rel") >= -COVERAGE_PRE) & (pl.col("t_rel") < 0))
    coverage = pre.group_by(ID, "LAB_GROUP").agg(
        pl.len().alias("n_pre"), (pl.col("t_rel") >= -BIN_WIDTH).sum().alias("n_pre_180"))
    coverage_bins = pre.with_columns(bin_expr(bin_edges(COVERAGE_PRE, 0))).select(ID, "LAB_GROUP", "bin_id").unique()
    return dict(patients=patients, canonical=canonical, patient_bins=bins,
                coverage_patient=coverage, coverage_bins=coverage_bins)


def valid_bundle(directory: Path, key: str, tables=ARM_TABLES) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        return manifest["key"] == key and all(
            fingerprint(directory / f"{table}.parquet") == manifest["outputs"][table]
            for table in tables)
    except (OSError, ValueError, KeyError):
        return False


def source_files(root: Path) -> list[dict]:
    if not root.exists():
        return [fingerprint(root)]
    return [fingerprint(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in {".csv", ".json", ".tsv", ".parquet"}]


def as_list(value):
    return value if isinstance(value, list) else [value]


def prepare_metastatic(config: dict, patients: pl.DataFrame) -> pl.DataFrame:
    # Import works both as a script and as a package in tests.
    import importlib.util
    path = Path(__file__).with_name("prepare_metastatic_figure_labels.py")
    spec = importlib.util.spec_from_file_location("figure_metastatic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = config["metastatic_sources"]
    labels = module.build_labels(
        pl.read_csv(sources["intent"], infer_schema_length=0),
        pl.read_parquet(sources["stage"], columns=[ID, "EVENT_DATE", "DERIVED_STAGE_MERGED"]),
        pl.read_parquet(sources["llm"], columns=[ID, "has_metastatic_disease"]), patients)
    if Path(sources["icd"]).exists() and config.get("metastatic_extra", False):
        codes = module.normalize_id(pl.read_csv(sources["icd"], infer_schema_length=0))
        code = pl.col("DIAGNOSIS_ICD10_CD").str.strip_chars().str.to_uppercase().str.replace_all("[^A-Z0-9]", "")
        site = pl.lit("other")
        for prefix, name in {"C795": "bone", "C7931": "brain", "C7932": "brain", "C797": "adrenal",
                             "C780": "lung", "C782": "lung", "C787": "liver", "C786": "peritoneal", "C77": "node"}.items():
            site = pl.when(code.str.starts_with(prefix)).then(pl.lit(name)).otherwise(site)
        codes = (codes.join(labels.select(ID, "ANALYSIS_ANCHOR_DATE"), on=ID)
                 .filter((code.str.len_chars() >= 4) & code.str.contains("^C7[789]") & (code != "C799"))
                 .filter(module.parse_datetime("START_DT") <= pl.col("ANALYSIS_ANCHOR_DATE"))
                 .with_columns(site.alias("site")))
        groups = ["brain", "bone", "liver", "lung", "node", "adrenal", "peritoneal", "other"]
        for group in groups:
            ids = codes.filter(pl.col("site") == group)[ID].unique().implode()
            labels = labels.with_columns(pl.when(pl.col("ANALYSIS_ANCHOR_DATE").is_not_null())
                .then(pl.col(ID).is_in(ids).cast(pl.Int32)).otherwise(None).alias(f"MET_SITE_{group}"))
        labels = labels.with_columns(pl.when(pl.col("ANALYSIS_ANCHOR_DATE").is_not_null())
            .then(pl.sum_horizontal([f"MET_SITE_{g}" for g in groups])).otherwise(None).alias("N_MET_SITES"))
    return labels.with_columns(pl.col(ID).cast(pl.String), *[
        pl.col(name).dt.replace_time_zone("UTC") for name, dtype in labels.schema.items()
        if isinstance(dtype, pl.Datetime)])


def prepare(config: dict) -> dict:
    root = Path(config["data_root"])
    cache = Path(config["cache_root"])
    cohorts, endpoints, labs = map(as_list, (config["cohorts"], config["endpoints"], config["labs"]))
    supported = {arm + subset + exclusion for arm in ["adt", "arpi"]
                 for subset in ["", "_metastatic_adt", "_metastatic_llm"]
                 for exclusion in ["", "_noprecastrate"]}
    if not cohorts or set(cohorts) - supported or len(set(cohorts)) != len(cohorts):
        raise ValueError(f"Invalid or duplicate figure cohorts: {cohorts}")
    if not endpoints or set(endpoints) - {"platinum", "nepc"} or len(set(endpoints)) != len(endpoints):
        raise ValueError(f"Invalid or duplicate figure endpoints: {endpoints}")
    if config.get("scope", "all") not in {"all", "federated"}:
        raise ValueError("scope must be all or federated")
    version = code_version()
    manifest = {"version": version, "arms": {}, "cells": {}, "errors": {}, "bins": {
        "trajectory_edges": bin_edges(), "coverage_edges": bin_edges(COVERAGE_PRE, 0)}}
    arms = [] if config.get("scope") == "federated" else sorted({"arpi" if c.startswith("arpi") else "adt" for c in cohorts})
    with preparation_lock(cache):
        manifest_path = Path(config.get("manifest_path", cache / "manifest.json"))
        try:
            previous = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            previous = {}
        # Explicit force survives the next ordinary invocation, so files that
        # were replaced while preserving stat metadata cannot resurrect caches.
        manifest["force_version"] = time.time_ns() if config.get("force") else previous.get("force_version", 0)
        for arm in arms:
            source = root / ("longitudinal_prediction_data_adt.csv" if arm == "adt" else "longitudinal_prediction_data.csv")
            if not source.exists():
                manifest["errors"][arm] = f"Missing longitudinal source: {source}"
                continue
            # Table cache ignores presentation code: changing an R title never
            # rescans the large CSV. Each arm is prepared independently.
            arm_labs = labs if "platinum" in endpoints else []
            key = digest({"source": fingerprint(source), "labs": sorted(arm_labs),
                          "python": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "polars": pl.__version__})
            directory = cache / "tables" / arm / key
            started = time.monotonic()
            if config.get("force", False) or not valid_bundle(directory, key):
                before = fingerprint(source)
                try:
                    tables = prepare_arm(source, arm_labs)
                except Exception as exc:
                    manifest["errors"][arm] = f"{source}: {exc}"
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                for name, table in tables.items():
                    with tempfile.NamedTemporaryFile(dir=directory, suffix=".parquet", delete=False) as stream:
                        temporary = Path(stream.name)
                    table.write_parquet(temporary, compression="zstd")
                    os.replace(temporary, directory / f"{name}.parquet")
                if fingerprint(source) != before:
                    raise RuntimeError(f"Source changed during preparation: {source}; rerun")
                atomic_json(directory / "manifest.json", {"key": key, "outputs": {
                    name: fingerprint(directory / f"{name}.parquet") for name in tables}})
                print(f"Prepared {arm} in {time.monotonic()-started:.1f}s", flush=True)
            else:
                print(f"Reused {arm} Polars tables", flush=True)
            output_version = json.loads((directory / "manifest.json").read_text())["outputs"]
            manifest["arms"][arm] = {"source": str(source.resolve()), "key": digest([key, output_version]),
                                     "directory": str(directory.resolve())}

        if "adt" in manifest["arms"] and "adt" in cohorts and "platinum" in endpoints and config.get("metastatic", False):
            sources = config["metastatic_sources"]
            if all(Path(sources[k]).exists() for k in ["intent", "stage", "llm"]):
                key = digest({"code": [hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    hashlib.sha256(Path(__file__).with_name("prepare_metastatic_figure_labels.py").read_bytes()).hexdigest()],
                    "arm": manifest["arms"]["adt"]["key"],
                    "sources": [fingerprint(Path(p)) for p in sources.values()], "extra": config.get("metastatic_extra")})
                path = cache / "labels" / key / "metastatic.parquet"
                if config.get("force", False) or not valid_bundle(path.parent, key, ["metastatic"]):
                    patients = pl.read_parquet(Path(manifest["arms"]["adt"]["directory"]) / "patients.parquet")
                    try:
                        labels = prepare_metastatic(config, patients)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as stream:
                            temporary = Path(stream.name)
                        labels.write_parquet(temporary)
                        os.replace(temporary, path)
                        atomic_json(path.parent / "manifest.json", {"key": key, "outputs": {"metastatic": fingerprint(path)}})
                    except Exception as exc:
                        manifest["errors"]["adt__platinum"] = f"Metastatic label preparation failed: {exc}"
                if "adt__platinum" not in manifest["errors"]:
                    manifest["metastatic_labels"] = str(path.resolve())

        shared_paths = [root / "LLM_NEPC_labels" / "baca_lab_annotations.csv",
                        root / "mrn_lists" / "platinum_MRN_list.csv",
                        root / "mrn_lists" / "icd_prostate_mrn_flags.csv",
                        Path(config["classifier_path"]) / "LLM_NEPC_classifier_labels.tsv"]
        shared = [fingerprint(p) for p in shared_paths]
        for cohort in (cohorts if arms else []):
            arm = "arpi" if cohort.startswith("arpi") else "adt"
            for endpoint in endpoints:
                suffix = "" if endpoint == "platinum" else "_nepc"
                inputs = root / "survival_analysis" / f"prediction_inputs_{cohort}{suffix}"
                cell = f"{cohort}__{endpoint}"
                if arm not in manifest["arms"] or not inputs.is_dir():
                    manifest["errors"][cell] = manifest["errors"].get(arm, f"Missing prediction inputs: {inputs}")
                    continue
                sources = source_files(inputs) + source_files(root / "survival_analysis" / f"local_runs_{cohort}{suffix}")
                if cohort == "adt":
                    for forest in as_list(config.get("forest_cohorts", cohorts)):
                        sources.append(fingerprint(root / "survival_analysis" / f"local_runs_{forest}{suffix}" / "cox" /
                            f"landmark_{config.get('forest_landmark', 180)}" / "both" / "cox_agg_univariate_nobs_adjusted.csv"))
                    if config.get("adt_intent", False):
                        sources += source_files(root / "survival_analysis" / "adt_intent_comparison")
                    if endpoint == "platinum" and config.get("metastatic", False):
                        sources += [fingerprint(Path(p)) for p in config["metastatic_sources"].values()]
                cell = f"{cohort}__{endpoint}"
                manifest["cells"][cell] = digest({"sources": sources, "shared": shared,
                    "arm": manifest["arms"][arm]["key"], "code": version, "force": manifest["force_version"],
                    "settings": {k: config.get(k) for k in ["gam", "metastatic", "metastatic_extra", "adt_intent", "forest_cohorts", "forest_landmark"]}})
        manifest["federated"] = digest({"code": version, "force": manifest["force_version"],
            "file": fingerprint(Path(config["federated_path"])),
            "local": [fingerprint(root / "survival_analysis" / "local_runs_adt" / "cox" / f"landmark_{lm}" / "both" /
                                  "cox_agg_univariate_nobs_adjusted.csv") for lm in [0, 90, 180]]})
        atomic_json(Path(config.get("manifest_path", cache / "manifest.json")), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
