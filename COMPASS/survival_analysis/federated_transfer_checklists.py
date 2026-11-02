"""Locally trained checklists of sequencing-clock federated scores + clinical features.

Consumes 04/04a's transfer_km_patients.csv and the endpoint-specific base ADT
split assignments. Frozen no-DFCI scores remain fixed; the checklist is trained
on local development patients and evaluated only on local held-out patients.
No raw lab summaries or separate age term enter the five-feature candidate pool.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .predictive_checklists import collect_supplement, runner
from .predictive_checklists.config import RunConfig
from .predictive_checklists.data import LandmarkData, _validate_manifest

ID = "person_id"
SCORE = "FEDERATED_SCORE"
FEATURES = (SCORE, "GLEASON_SCORE", "TP53_ANY", "RB1_ANY", "PTEN_ANY")
FAMILIES = ("elastic_net_cox", "xgboost_cox")
POPULATION = "sequenced_with_gleason"
OUTPUT_SUBDIR = "clinical_checklists"
TABLES = ("checklist_cells.csv", "checklist_items_long.csv", "checklist_metrics.csv")
KEYS = ["model_family", "endpoint", "landmark_days"]


def transfer_config(**overrides):
    settings = dict(endpoints=("platinum", "nepc", "avpc"), horizons=(365, 1825),
                    max_items=(3, 5), min_events=5, baselines=(),
                    resolved_nepc_sensitivity=False, clinical_features=FEATURES, id_col=ID)
    settings.update(overrides)
    config = RunConfig(**settings).validate()
    if tuple(config.horizons) != (365, 1825) or tuple(config.clinical_features) != FEATURES:
        raise ValueError("Transfer checklists require 365/1825-day horizons and the five specified features")
    if config.baselines or config.resolved_nepc_sensitivity or config.id_col != ID:
        raise ValueError("Transfer checklists use person_id, no baseline refits and no label sensitivity")
    return config


def _ids(values):
    # The federated adapter normalizes DFCI MRNs to integer person_id values.
    numeric = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(numeric).all() or (numeric != np.floor(numeric)).any():
        raise ValueError("Missing or noninteger patient identifier")
    return numeric.astype("int64").astype(str)


def prepare_cells(patients_path, input_dirs, config):
    """Validate all sources before fitting; retain the transfer's outcomes in days."""
    patients_path = Path(patients_path)
    patients = pd.read_csv(patients_path)
    required = {*KEYS, ID, "clock", "bundle", "risk_score", "duration_days", "event",
                "HAS_SEQUENCING", *FEATURES[1:]}
    if missing := required - set(patients):
        raise ValueError(f"Transfer patient table is missing {sorted(missing)}")
    patients = patients.loc[patients.clock.eq("sequencing") & patients.model_family.isin(FAMILIES)
                            & patients.endpoint.isin(config.endpoints)
                            & patients.landmark_days.isin(config.landmarks)].copy()
    if patients.empty:
        raise ValueError("No sequencing-clock Cox/XGBoost scores for the requested cells")
    patients[ID] = _ids(patients[ID])
    if patients.duplicated([*KEYS, ID]).any():
        raise ValueError("Duplicate patient/model/endpoint/landmark scores; merge bundles first")
    # Identical endpoint/landmark patients must carry identical outcomes and clinical data
    # regardless of which frozen model generated the score.
    shared = patients.groupby(["endpoint", "landmark_days", ID])
    if (shared[["duration_days", "event", "HAS_SEQUENCING", *FEATURES[1:]]]
            .nunique(dropna=False).gt(1).any().any()):
        raise ValueError("Outcomes or clinical features disagree across frozen models")
    sources, splits, manifests = {}, {}, {}
    for endpoint in config.endpoints:
        folder = Path(input_dirs[endpoint])
        manifest_path = folder / "build_manifest.json"
        split_path = folder / "split_assignments.csv"
        manifest = json.loads(manifest_path.read_text())
        for landmark in config.landmarks:
            _validate_manifest(manifest, endpoint, landmark)
        split = pd.read_csv(split_path, dtype={"DFCI_MRN": "string", "split": "string"})
        if not {"DFCI_MRN", "split"}.issubset(split):
            raise ValueError(f"Invalid split columns in {split_path}")
        split[ID] = _ids(split.DFCI_MRN)
        if split[ID].duplicated().any() or not split.split.isin(["train", "valid", "test"]).all():
            raise ValueError(f"Invalid or duplicate split assignments in {split_path}")
        splits[endpoint] = split.set_index(ID)["split"]
        manifests[endpoint] = manifest
        sources[endpoint] = dict(path=str(folder.resolve()), source_manifest=manifest,
                                 manifest_sha256=runner.fingerprint(manifest_path),
                                 split_sha256=runner.fingerprint(split_path))
    pending = []
    for (family, endpoint, landmark), group in patients.groupby(KEYS, sort=True):
        if group.bundle.isna().any() or group.bundle.nunique() != 1:
            raise ValueError("Each family/endpoint/landmark must identify one frozen bundle")
        group = group.copy()
        numeric = ["risk_score", "duration_days", "event", *FEATURES[1:]]
        group[numeric] = group[numeric].apply(pd.to_numeric, errors="coerce")
        if (not np.isfinite(group.duration_days).all() or not group.duration_days.gt(0).all()
                or not group.event.isin([0, 1]).all()):
            raise ValueError("Transfer outcomes must be positive days and binary events")
        sequenced = group.HAS_SEQUENCING.astype("string").str.lower().isin(["true", "1", "1.0"])
        eligible = sequenced & group.GLEASON_SCORE.notna()
        available = group.loc[eligible].copy()
        if (not np.isfinite(available[["risk_score", *FEATURES[1:]]]).all().all()
                or not available[list(FEATURES[2:])].isin([0, 1]).all().all()):
            raise ValueError("Available-case scores/Gleason must be finite and mutation status binary")
        available["split"] = available[ID].map(splits[endpoint])
        # As in build_somatic_gleason_inputs, restrict to patients with an existing
        # base split, preserving its membership even when sequencing precedes ADT.
        audit = dict(n_scored=len(group), n_sequenced_with_gleason=len(available),
                     n_without_base_split=int(available.split.isna().sum()),
                     prediction_time_origin="sequencing date", duration_units="days_after_landmark")
        available = available.loc[available.split.notna()].copy()
        frame = available.rename(columns={"risk_score": SCORE, "duration_days": f"t_{endpoint}",
                                          "event": endpoint.upper()})
        frame = frame[[ID, "split", f"t_{endpoint}", endpoint.upper(), *FEATURES]].reset_index(drop=True)
        manifest = {**manifests[endpoint], "prediction_time_origin": "sequencing date"}
        audit.update(n_patients=len(frame), split_counts=frame.split.value_counts().to_dict(),
                     n_test_events=int(frame.loc[frame.split.eq("test"), endpoint.upper()].sum()),
                     auc_max_days=float(manifest["auc_max_time_units"]) * float(manifest["auc_time_unit_days"]))
        folder = Path(input_dirs[endpoint])
        data = LandmarkData(frame, manifest, folder / "build_manifest.json", folder, audit)
        identity = dict(model_family=family, bundle=str(group.bundle.iloc[0]), clock="sequencing",
                        cohort=POPULATION, prediction_time_origin="sequencing date",
                        analysis="locally_trained_transfer_checklist", feature_set=family)
        pending.append((family, endpoint, int(landmark), data, identity))
    return pending, sources


def run_transfer_checklists(patients_path, input_dirs, output_dir, *, config=None,
                            resume=False, overwrite=False):
    config = config or transfer_config()
    # Revalidate supplied configs through the same scope constraints.
    config = transfer_config(**config.to_dict())
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    output_dir = Path(output_dir)
    pending, sources = prepare_cells(patients_path, input_dirs, config)
    code = [Path(__file__), *sorted(Path(runner.__file__).parent.glob("*.py"))]
    signature = dict(config=config.to_dict(), inputs=sources,
                     patients_path=str(Path(patients_path).resolve()),
                     patients_sha256=runner.fingerprint(patients_path),
                     source_sha256={p.name: runner.fingerprint(p) for p in code})
    # Normalize tuple/list JSON representation for stable resume comparisons.
    signature = runner._jsonable(signature)
    manifest_path = output_dir / "run_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and resume:
        if previous["signature"] != signature:
            raise ValueError("Cannot resume: source, inputs or configuration changed")
    elif output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Run already exists: {output_dir}; use resume or overwrite")
        archive = output_dir.parent / (output_dir.name + "_previous_runs")
        archive.mkdir(exist_ok=True)
        shutil.move(str(output_dir), str(archive / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, signature=signature, status="running",
                    created_utc=(previous["created_utc"] if previous and resume else
                                 datetime.now(timezone.utc).isoformat()),
                    method="Local IPCW M-of-N checklist on frozen no-DFCI scores + Gleason + TP53/RB1/PTEN",
                    evaluation="Local held-out test; independent 365/1825-day horizon risks")
    runner.write_json(manifest_path, manifest)
    all_results, preflight, family_results = [], [], {}
    for family, endpoint, landmark, data, identity in pending:
        print(f"Transfer checklist: {family}, {endpoint}, sequencing +{landmark}d", flush=True)
        folder = output_dir / family / endpoint / f"landmark_{landmark}" / POPULATION
        preflight.append({**identity, "endpoint": endpoint, "landmark_days": landmark, **data.audit})
        rows = runner.run_cell(data, endpoint, landmark, POPULATION, config, folder,
                               resume=resume, identity_metadata=identity)
        all_results.extend(rows)
        family_results.setdefault(family, []).extend(rows)
    runner.write_json(output_dir / "preflight.json", preflight)
    runner.write_json(output_dir / "metrics.json", all_results)
    for family, rows in family_results.items():
        runner.write_json(output_dir / family / "metrics.json", rows)
    runs = {family: output_dir / family for family in family_results}
    tables = [collect_supplement.collect(runs, horizon, POPULATION) for horizon in config.horizons]
    counts = pd.DataFrame([
        dict(feature_set=family, endpoint=endpoint, landmark_days=landmark, horizon_days=horizon,
             n_events_test=int((data.frame.split.eq("test") & data.frame[endpoint.upper()].eq(1)
                                & data.frame[f"t_{endpoint}"].le(horizon)).sum()))
        for family, endpoint, landmark, data, _ in pending for horizon in config.horizons])
    for i, name in enumerate(TABLES):
        table = pd.concat([part[i] for part in tables], ignore_index=True)
        if i == 0:
            table = table.merge(counts, on=["feature_set", "endpoint", "landmark_days", "horizon_days"],
                                validate="one_to_one")
        table.to_csv(output_dir / name, index=False)
    manifest["status"] = "complete" if any(r.get("status") == "ok" for r in all_results) else "no_supported_models"
    runner.write_json(manifest_path, manifest)
    return pd.DataFrame(all_results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-dir", type=Path, required=True)
    parser.add_argument("--inputs-root", type=Path, required=True,
                        help="Directory containing endpoint-specific prediction_inputs_adt trees")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--endpoints", nargs="+", choices=["platinum", "nepc", "avpc"],
                        default=["platinum", "nepc", "avpc"])
    parser.add_argument("--landmarks", nargs="+", type=int, default=[0, 90, 180])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--solver-time-limit", type=float, default=60)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    inputs = {ep: args.inputs_root / ("prediction_inputs_adt" + ("" if ep == "platinum" else f"_{ep}"))
              for ep in args.endpoints}
    config = transfer_config(endpoints=tuple(args.endpoints), landmarks=tuple(args.landmarks),
                             n_folds=args.n_folds, seed=args.seed, bootstrap_samples=args.bootstrap_samples,
                             solver_time_limit=args.solver_time_limit)
    results = run_transfer_checklists(args.transfer_dir / "transfer_km_patients.csv", inputs,
        args.output_dir or args.transfer_dir / OUTPUT_SUBDIR, config=config,
        resume=args.resume, overwrite=args.overwrite)
    return 0 if results.status.eq("ok").any() else 2


if __name__ == "__main__":
    raise SystemExit(main())
