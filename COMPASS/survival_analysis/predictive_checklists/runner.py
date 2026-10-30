"""Training-only selection/calibration and held-out horizon evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import pickle
import platform
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .baselines import fit_baselines
from .data import load_landmark, resolved_nepc_mask
from .evaluation import evaluate_horizon, fit_calibrator
from .features import AdjusterDesign, FeatureBinarizer, schema_feature_columns
from .optimizer import fit_checklist
from .survival import CensoringDistribution, horizon_targets


class FitFailure(ValueError):
    def __init__(self, message, cv_rows):
        super().__init__(message)
        self.cv_rows = cv_rows


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outcomes(frame, endpoint):
    return frame[f"t_{endpoint}"].to_numpy(float), frame[endpoint.upper()].to_numpy(int)


def _balanced_weights(y, w):
    result = np.zeros(len(w), dtype=float)
    for label in (0, 1):
        mask = (y == label) & (w > 0)
        total = w[mask].sum()
        if total <= 0:
            raise ValueError("Horizon needs observed cases and controls")
        result[mask] = w[mask] / (2 * total)
    return result


def _counts_ok(y, config):
    return np.sum(y == 1) >= config.min_events and np.sum(y == 0) >= config.min_controls


def horizon_preflight(data, endpoint, config):
    train = data.frame.loc[data.frame["split"].isin(["train", "valid"])]
    test = data.frame.loc[data.frame["split"] == "test"]
    td, te = _outcomes(train, endpoint)
    censoring = CensoringDistribution.fit(td, te)
    cap = data.audit["auc_max_days"]
    records = []
    for horizon in config.horizons:
        record = {"horizon_days": horizon}
        try:
            if horizon >= cap:
                raise ValueError("At/beyond manifest administrative evaluation cap")
            labels, _, audit = horizon_targets(td, te, horizon, censoring, min_g=config.min_g)
            record.update(status="supported" if _counts_ok(labels, config) else "underpowered",
                          train_val=audit)
            vd, ve = _outcomes(test, endpoint)
            _, _, audit = horizon_targets(vd, ve, horizon, censoring, min_g=config.min_g)
            record["test"] = audit
        except ValueError as exc:
            record.update(status="unsupported", reason=str(exc))
        records.append(record)
    return records


def _adjust_only(config):
    return getattr(config, "adjust_only_mode", "none") == "drop"


def _binarizer(config):
    return FeatureBinarizer(min_patient_coverage=config.min_patient_coverage,
                            quantiles=config.quantiles, age_col=config.age_col, id_col=config.id_col,
                            clinical_features=config.clinical_features,
                            exclude_adjusters=_adjust_only(config))


def fit_horizon(train, endpoint, horizon, config):
    """Select checklist size in CV, calibrate OOF scores, refit on train+valid.

    Test rows are not accepted by this function. Every CV fold refits both
    preprocessing and the censoring distribution on its own training rows.
    """
    if not train["split"].isin(["train", "valid"]).all():
        raise ValueError("fit_horizon accepts only train/valid rows")
    duration, event = _outcomes(train, endpoint)
    censoring = CensoringDistribution.fit(duration, event)
    labels, weights, audit = horizon_targets(duration, event, horizon, censoring,
                                            min_g=config.min_g)
    if not _counts_ok(labels, config):
        raise ValueError("underpowered: too few known horizon cases or controls")
    # Stratify on horizon labels including the unknown-status category.
    _, frequencies = np.unique(labels, return_counts=True)
    n_folds = min(config.n_folds, int(frequencies.min()))
    if n_folds < 2:
        # An isolated early-censored patient must not prevent fitting. Endpoint
        # stratification retains all patients without manufacturing a label.
        stratification = event
        _, frequencies = np.unique(event, return_counts=True)
        n_folds = min(config.n_folds, int(frequencies.min()))
    else:
        stratification = labels
    if n_folds < 2:
        raise ValueError("underpowered: cannot form two training folds")
    folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                random_state=config.seed).split(train, stratification))
    cv_rows = []
    candidates = {}
    for max_items in config.max_items:
        oof = np.full(len(train), np.nan)
        oof_weights = np.zeros(len(train))
        errors = []
        for fold, (fit_idx, val_idx) in enumerate(folds):
            row = {"max_items": max_items, "fold": fold}
            try:
                fit_frame, val_frame = train.iloc[fit_idx], train.iloc[val_idx]
                if set(fit_frame[config.id_col]) & set(val_frame[config.id_col]):
                    raise ValueError("Patient overlap in CV")
                fd, fe = _outcomes(fit_frame, endpoint)
                vd, ve = _outcomes(val_frame, endpoint)
                fold_censoring = CensoringDistribution.fit(fd, fe)
                fy, fw, _ = horizon_targets(fd, fe, horizon, fold_censoring, min_g=config.min_g)
                vy, vw, _ = horizon_targets(vd, ve, horizon, fold_censoring, min_g=config.min_g)
                if not _counts_ok(fy, config):
                    raise ValueError("underpowered fold training risk set")
                binarizer = _binarizer(config).fit(fit_frame)
                X = binarizer.transform(fit_frame)
                known = fw > 0
                model = fit_checklist(X[known], fy[known],
                    _balanced_weights(fy, fw)[known], binarizer.groups,
                    max_items=max_items, time_limit=config.solver_time_limit,
                    solver=config.solver, seed=config.seed + fold)
                val_X = binarizer.transform(val_frame)
                score = model.predict_score(val_X)
                flag = model.predict(val_X)
                balanced = _balanced_weights(vy, vw)
                loss = float(np.sum(balanced * (flag != vy)))
                errors.append(loss)
                oof[val_idx] = score / max(len(model.selected_indices), 1)
                oof_weights[val_idx] = vw
                row.update(status="ok", balanced_error=loss,
                           n_items=len(model.selected_indices), threshold=model.threshold,
                           solver_diagnostics=model.diagnostics)
            except (ValueError, RuntimeError, ImportError) as exc:
                row.update(status="failed", reason=str(exc))
            cv_rows.append(row)
        # Never reward a candidate for skipping its difficult validation folds.
        if len(errors) == n_folds and np.isfinite(oof).all():
            candidates[max_items] = (float(np.mean(errors)), oof, oof_weights)
    if not candidates:
        raise FitFailure("No checklist size completed every CV fold; inspect cv_folds.json", cv_rows)
    selected = min(candidates, key=lambda size: (candidates[size][0], size))
    loss, oof, calibration_weights = candidates[selected]
    binarizer = _binarizer(config).fit(train)
    adjusters = None
    if _adjust_only(config):
        # The adjusters carry no outcome information, so one development-block
        # design serves the OOF calibration fit; predictions use Z = 0.
        adjusters = AdjusterDesign().fit(train, binarizer)
        calibrator = fit_calibrator(oof, labels, calibration_weights,
                                    adjusters=adjusters.transform(train),
                                    adjuster_names=adjusters.names)
    else:
        calibrator = fit_calibrator(oof, labels, calibration_weights)
    X = binarizer.transform(train)
    known = weights > 0
    model = fit_checklist(X[known], labels[known], _balanced_weights(labels, weights)[known],
                          binarizer.groups, max_items=selected,
                          time_limit=config.solver_time_limit, solver=config.solver, seed=config.seed)
    return dict(model=model, binarizer=binarizer, calibrator=calibrator, censoring=censoring,
                adjusters=adjusters, selected_max_items=selected, cv_balanced_error=loss, cv_rows=cv_rows,
                audit=audit, oof=oof, oof_weights=calibration_weights, labels=labels)


def _predict(fitted, frame):
    model = fitted["model"]
    X = fitted["binarizer"].transform(frame)
    score = model.predict_score(X)
    return score, model.predict(X), fitted["calibrator"].predict(
        score / max(len(model.selected_indices), 1))


def _metrics(frame, endpoint, horizon, censoring, score, flag, probability, config):
    duration, event = _outcomes(frame, endpoint)
    return evaluate_horizon(duration, event, score, flag, probability, horizon, censoring,
                            min_g=config.min_g, bootstrap_samples=config.bootstrap_samples,
                            seed=config.seed)


def run_cell(data, endpoint, landmark, population, config, output_dir, *, resume=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = data.frame.copy()
    if population == "resolved_nepc":
        frame = frame.loc[resolved_nepc_mask(frame)].copy()
    train = frame.loc[frame["split"].isin(["train", "valid"])].reset_index(drop=True)
    test = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    if set(train[config.id_col]) & set(test[config.id_col]):
        raise ValueError("Training/test patient overlap")
    identity = dict(endpoint=endpoint, landmark_days=landmark, population=population,
                    arm="adt", cohort="all", exclusion="none", dataset="test")
    audit = {"identity": identity, "n_patients": len(frame), "n_train_val": len(train),
             "n_test": len(test), "n_events": int(frame[endpoint.upper()].sum()),
             "low_event_warning": int(frame[endpoint.upper()].sum()) < 50,
             "missing_predictor_counts": frame.isna().sum().to_dict(), "input_audit": data.audit}
    if "NEPC_LABEL_SOURCE" in frame:
        audit["nepc_label_sources"] = frame.NEPC_LABEL_SOURCE.fillna("missing").value_counts().to_dict()
        audit["n_resolved_nepc"] = int(resolved_nepc_mask(frame).sum())
    write_json(output_dir / "cohort_audit.json", audit)
    results = []
    if len(train) == 0 or len(test) == 0:
        results = [{**identity, "model": "checklist", "status": "underpowered",
                    "reason": "Empty training or test risk set"}]
        write_json(output_dir / "metrics.json", results)
        return results
    td, te = _outcomes(train, endpoint)
    # Each horizon writes its metrics.json last, so it marks a finished horizon.
    done = {horizon for horizon in config.horizons
            if resume and (output_dir / f"horizon_{horizon}" / "metrics.json").exists()}
    baseline_results = {}
    if config.baselines and len(done) < len(config.horizons):
        columns = schema_feature_columns(train, age_col=config.age_col,
                                         clinical_features=config.clinical_features,
                                         exclude_adjusters=_adjust_only(config))
        stored = output_dir / "baselines.pkl"
        if resume and stored.exists():
            with stored.open("rb") as stream:
                baseline_results = pickle.load(stream)
        else:
            baseline_results = fit_baselines(train[columns], td, te,
                age_column=config.age_col, horizons=config.horizons,
                n_splits=config.n_folds, seed=config.seed, models=config.baselines,
                min_patient_coverage=config.min_patient_coverage, min_g=config.min_g)
            write_json(output_dir / "baselines.json", {name: dict(status=value.status,
                reason=value.reason, metadata=value.metadata) for name, value in baseline_results.items()})
            # Renamed into place so an interrupted dump is never reused on resume.
            partial = output_dir / "baselines.pkl.tmp"
            with partial.open("wb") as stream:
                pickle.dump(baseline_results, stream)
            partial.replace(stored)
    for horizon in config.horizons:
        cell = {**identity, "horizon_days": horizon}
        folder = output_dir / f"horizon_{horizon}"
        if horizon in done:
            print(f"  {endpoint} landmark={landmark} population={population} horizon={horizon}d: "
                  "reusing completed horizon", flush=True)
            results.extend(json.loads((folder / "metrics.json").read_text()))
            continue
        if resume and folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(exist_ok=True)
        print(f"  {endpoint} landmark={landmark} population={population} horizon={horizon}d", flush=True)
        start = len(results)
        cap = data.manifest["auc_max_time_units"] * data.manifest.get("auc_time_unit_days", data.manifest.get("time_unit_days", 7))
        if horizon >= cap:
            results.append({**cell, "model": "checklist", "status": "unsupported",
                            "reason": "Horizon is at/beyond the input manifest evaluation cap"})
            write_json(folder / "metrics.json", results[start:])
            continue
        try:
            fitted = fit_horizon(train, endpoint, horizon, config)
            model = fitted["model"]
            bundle = {"schema_version": 1, "identity": cell, "config": config.to_dict(),
                "model": model.to_dict(), "binarizer": fitted["binarizer"].to_dict(),
                "calibrator": fitted["calibrator"].to_dict(),
                "censoring": fitted["censoring"].to_dict(),
                "selected_max_items": fitted["selected_max_items"],
                "cv_balanced_error": fitted["cv_balanced_error"],
                "prediction_type": "independent_horizon_risk_under_existing_endpoint_censoring"}
            if fitted["adjusters"] is not None:
                bundle["adjuster_design"] = fitted["adjusters"].to_dict()
            write_json(folder / "model.json", bundle)
            write_json(folder / "cv_folds.json", fitted["cv_rows"])
            write_json(folder / "horizon_audit.json", fitted["audit"])
            items = [fitted["binarizer"].items[j] for j in model.selected_indices]
            write_json(folder / "checklist.json", {"threshold": model.threshold, "items": items})
            origin = data.manifest.get("prediction_time_origin", "ADT start")
            lines = [f"{endpoint.upper()}: event within {horizon} days after {origin} +{landmark} days",
                     f"Flag when at least {model.threshold} of {len(items)} items are checked.", ""]
            lines.extend(f"[ ] {item['name']}" for item in items)
            lines += ["", "Use the exported preprocessing rules for missing measurements.",
                      "Research model; this checklist does not estimate an exact event date."]
            (folder / "checklist.txt").write_text("\n".join(lines) + "\n")
            predictions = []
            for block, label in ((train, "train_val_in_sample"), (test, "test")):
                score, flag, probability = _predict(fitted, block)
                prediction = block[[config.id_col, "split"]].copy()
                prediction["dataset"] = label
                prediction["score"] = score
                prediction["flag"] = flag
                prediction["probability"] = probability
                predictions.append(prediction)
                if label == "test":
                    results.append({**cell, "model": "checklist", **_metrics(block, endpoint,
                        horizon, fitted["censoring"], score, flag, probability, config),
                        "n_items": len(items), "threshold": model.threshold,
                        "solver_status": model.diagnostics.get("status"),
                        "solver_optimality_gap": model.diagnostics.get("optimality_gap"),
                        "solver_solution_source": model.diagnostics.get("solution_source"),
                        "selected_max_items": fitted["selected_max_items"],
                        "cv_balanced_error": fitted["cv_balanced_error"]})
            pd.concat(predictions, ignore_index=True).to_csv(folder / "predictions.csv", index=False)
            pd.DataFrame({config.id_col: train[config.id_col], "score_fraction": fitted["oof"],
                          "ipcw": fitted["oof_weights"], "label": fitted["labels"]}).to_csv(
                              folder / "calibration_oof.csv", index=False)
            for name, baseline in baseline_results.items():
                if baseline.status != "ok":
                    results.append({**cell, "model": name, "status": baseline.status, "reason": baseline.reason})
                    continue
                try:
                    predicted = baseline.predict(test[columns], horizon)
                    results.append({**cell, "model": name, **_metrics(test, endpoint, horizon,
                        fitted["censoring"], predicted["score"], predicted["flag"], predicted["probability"], config)})
                    baseline_frame = test[[config.id_col, "split"]].copy()
                    for key, values in predicted.items():
                        baseline_frame[key] = values
                    baseline_frame.to_csv(folder / f"predictions_{name}.csv", index=False)
                except (ValueError, RuntimeError) as exc:
                    results.append({**cell, "model": name, "status": "failed", "reason": str(exc)})
        except (ValueError, RuntimeError, ImportError) as exc:
            results.append({**cell, "model": "checklist", "status": "unsupported_or_failed",
                            "reason": str(exc)})
            write_json(folder / "failure.json", results[-1])
            if isinstance(exc, FitFailure):
                write_json(folder / "cv_folds.json", exc.cv_rows)
        write_json(folder / "metrics.json", results[start:])
    write_json(output_dir / "metrics.json", results)
    return results


def _check_resumable(previous, manifest):
    """Refuse to mix completed horizons with fits from different settings or data."""
    if previous.get("schema_version") != manifest["schema_version"]:
        raise ValueError("Cannot resume: run manifest schema version differs")
    if previous.get("config") != json.loads(json.dumps(_jsonable(manifest["config"]))):
        raise ValueError("Cannot resume: run configuration differs from the existing run")
    if previous.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("Cannot resume: predictive_checklists source changed since the existing run")
    current = {(row["endpoint"], row["landmark_days"]): row for row in manifest["inputs"]}
    for row in previous.get("inputs", []):
        match = current.get((row["endpoint"], row["landmark_days"]))
        if match and any(match[key] != row[key] for key in ("manifest_sha256", "aggregated_sha256")):
            raise ValueError(f"Cannot resume: {row['endpoint']} landmark {row['landmark_days']} "
                             "inputs changed since the existing run")


def run(config, input_dirs, output_dir, *, preflight_only=False, overwrite=False, resume=False):
    output_dir = Path(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    previous = None
    if manifest_path.exists() and resume:
        previous = json.loads(manifest_path.read_text())
    elif manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Run already exists: {output_dir}; choose a new directory, "
                              "--overwrite or --resume")
    elif manifest_path.exists():
        # Preserve previous generated artifacts; a reduced rerun must not leave
        # old successful models beside new failure records.
        archive = output_dir / "previous_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        for name in ("platinum", "nepc", "run_manifest.json", "preflight.json", "metrics.json",
                     "metrics.csv", "calibration.csv"):
            source = output_dir / name
            if source.exists():
                shutil.move(str(source), str(archive / name))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": config.to_dict(), "status": "running", "inputs": [],
                "python": platform.python_version(), "packages": {},
                "method": "Zhang et al. 2021 M-of-N with local IPCW extension",
                "upstream_commit": "d205245b9f62dd2cfd977171f0e5073c13d6aebf"}
    manifest["source_sha256"] = {path.name: fingerprint(path)
        for path in sorted(Path(__file__).parent.iterdir())
        if path.is_file() and path.suffix in {".py", ".R", ".json"}}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "scikit-survival", "xgboost"):
        try:
            manifest["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            manifest["packages"][name] = None
    if previous is None:
        write_json(manifest_path, manifest)
    metrics, preflight = [], []
    split_by_endpoint = {}
    pending = []
    failed_inputs = False
    for endpoint in config.endpoints:
        for landmark in config.landmarks:
            try:
                data = load_landmark(input_dirs[endpoint], endpoint, landmark,
                                     id_col=config.id_col, age_col=config.age_col)
                assignment = split_by_endpoint.setdefault(endpoint, {})
                for patient, split in zip(data.frame[config.id_col], data.frame["split"]):
                    if patient in assignment and assignment[patient] != split:
                        raise ValueError("Patient splits differ across landmarks for this endpoint")
                    assignment[patient] = split
                audit = {"endpoint": endpoint, "landmark_days": landmark, "status": "ok",
                         "n_patients": len(data.frame),
                         "split_counts": data.frame["split"].value_counts().to_dict(),
                         "events_by_split": data.frame.groupby("split")[endpoint.upper()].sum().to_dict(),
                         "input_audit": data.audit,
                         "requested_horizons": horizon_preflight(data, endpoint, config)}
                preflight.append(audit)
                manifest["inputs"].append({"endpoint": endpoint, "landmark_days": landmark,
                    "path": str(data.input_dir), "manifest_sha256": fingerprint(data.manifest_path),
                    "aggregated_sha256": fingerprint(data.input_dir / f"aggregated_landmark{landmark}.csv"),
                    "synthetic": bool(data.manifest.get("synthetic", False)),
                    "source_manifest": data.manifest})
                pending.append((endpoint, landmark, data))
            except (ValueError, FileNotFoundError) as exc:
                failed_inputs = True
                preflight.append({"endpoint": endpoint, "landmark_days": landmark,
                                  "status": "failed", "reason": str(exc)})
    if previous is not None:
        # Checked before anything is written so a refused resume leaves the run intact.
        _check_resumable(previous, manifest)
        manifest["created_utc"] = previous["created_utc"]
        manifest["resumed_utc"] = previous.get("resumed_utc", []) + [
            datetime.now(timezone.utc).isoformat()]
        write_json(manifest_path, manifest)
    write_json(output_dir / "preflight.json", preflight)
    # Validate every input and cross-landmark split before any model sees data.
    if not failed_inputs and not preflight_only:
        for endpoint, landmark, data in pending:
            populations = ["full"]
            if endpoint == "nepc" and config.resolved_nepc_sensitivity:
                populations.append("resolved_nepc")
            for population in populations:
                metrics.extend(run_cell(data, endpoint, landmark, population, config,
                    output_dir / endpoint / f"landmark_{landmark}" / population, resume=resume))
    write_json(output_dir / "metrics.json", metrics)
    if metrics:
        pd.json_normalize(_jsonable(metrics)).to_csv(output_dir / "metrics.csv", index=False)
        calibration = []
        for row in metrics:
            identity = {key: row.get(key) for key in ("endpoint", "landmark_days", "population",
                                                    "dataset", "model", "horizon_days")}
            calibration.extend({**identity, **point} for point in row.get("calibration", []))
        if calibration:
            pd.DataFrame(calibration).to_csv(output_dir / "calibration.csv", index=False)
    successful = sum(row.get("model") == "checklist" and row.get("status") == "ok" for row in metrics)
    unsupported = sum(row.get("model") == "checklist" and row.get("status") != "ok" for row in metrics)
    status = "incomplete" if failed_inputs else ("preflight_complete" if preflight_only
             else (("complete_with_unsupported_cells" if unsupported else "complete")
                   if successful else "no_supported_models"))
    manifest.update(status=status,
                    n_successful_checklists=successful, n_unsupported_checklist_cells=unsupported,
                    preflight_only=preflight_only)
    write_json(manifest_path, manifest)
    print(f"Outputs: {output_dir}; successfully evaluated checklists: {successful}", flush=True)
    return 1 if failed_inputs else (0 if preflight_only or successful else 2)
