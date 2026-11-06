"""Federated checklist + Gleason / TP53-RB1-PTEN, post-trained on DFCI platinum (reviewer request, Figure 6).

A reviewer asked whether adding Gleason, tumor-suppressor (TSG) status, or both to the federated
checklist changes its performance. This module starts from the frozen no-DFCI generalized (rounded-threshold)
checklist that Figure 6 applies to DFCI. It keeps every federated item and adds unit-weight Gleason and/or TSG
items. It then relearns the M-of-N threshold within DFCI. Every rule is evaluated out of fold in three cohorts:

* ``gleason``: Gleason score available before ADT + landmark (ADT clock); adds ``GLEASON_SCORE >= g``.
* ``somatic``: sequenced patients on the sequencing clock (anchored on REPORT_DT); adds one TSG item.
* ``combined``: sequenced patients with a Gleason score (sequencing clock, the Figure 6 b population); adds both.

No somatic data is used on the ADT clock.

**Post-training** uses the platinum endpoint only, at the same landmark and horizon as the frozen checklist.
On the training rows it fits the reverse Kaplan-Meier censoring curve and the fixed-horizon case/control labels
(predictive_checklists.survival). The objective is the class-balanced IPCW misclassification error that the
checklist MILP minimizes (runner._balanced_weights, reimplemented here). The frozen items are fixed, so a rule is
just the frozen item count plus 0-2 new indicators against a threshold M. An exhaustive search over the new items
(the Gleason cutoff g in {7, 8, 9}; TP53, RB1, PTEN or any of the three) and M therefore finds the exact optimum
without a solver.

**Tie-breaks**, in order: lower error; the Figure 6 b defaults (Gleason >= 8, any TSG); M closest, as a fraction of
the items, to the frozen M/N; then the smaller M.

**Cross-fitting.** A single patient-level K-fold split is shared by every cohort, clock, landmark and horizon, so
the cohorts nest fold by fold. Each fold's rule is learned on the other folds' platinum rows, then applied to that
fold for platinum, AVPC and NEPC. A training fold with fewer than ``MIN_CASES`` cases or ``MIN_CONTROLS`` controls
at the horizon, or a horizon its censoring curve cannot support, fails the gate. Its patients then get no tuned
flag, so they leave the paired comparison; no other fold's rule is borrowed for them. A refit on the whole cohort
(``fold == "all"``) gives the rule one would report. It is descriptive only.

**Evaluation** is the Figure 6 b metric: horizon AUC of a yes/no rule (the IPCW balanced accuracy,
fet.fixed_horizon_mean_auc). Intervals come from a joint patient bootstrap, so the deltas against the frozen
checklist are paired. The bootstrap conditions on the out-of-fold flags, so the intervals leave out the uncertainty
of re-tuning the rules. Platinum is the tuning endpoint (still held out); AVPC and NEPC are transfer endpoints.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import federated_clinical_comparison as fcc  # noqa: E402  (puts the repo root on sys.path)
from survival_common import federated_endpoint_transfer as fet  # noqa: E402
from COMPASS.multivariable_models.predictive_checklists.survival import (  # noqa: E402
    CensoringDistribution, horizon_targets)

ID_COL = fcc.ID_COL
DURATION_COL = fcc.DURATION_COL
EVENT_COL = fcc.EVENT_COL
GLEASON_FEATURE = fcc.GLEASON_FEATURE
HAS_SEQUENCING = fcc.HAS_SEQUENCING
TRIO_FEATURES = fcc.TRIO_FEATURES
TRAINING_ENDPOINT = "platinum"

# The checklist package's Gleason/trio gates (gleason_trio_config.json).
MIN_CASES = 5
MIN_CONTROLS = 10
MIN_G = 0.05
TIE_TOLERANCE = 1e-12

GLEASON_CUTOFFS = (7, 8, 9)
DEFAULT_GLEASON_CUTOFF = fcc.GLEASON_HIGH_MIN
TSG_ANY = "any"
TSG_ITEMS = (TSG_ANY, *TRIO_FEATURES)   # the default (any) first

COHORTS = {
    "gleason": {"label": "Gleason available", "clock": "adt",
                "member": (GLEASON_FEATURE,), "gleason": True, "tsg": False},
    "somatic": {"label": "Sequenced", "clock": "sequencing",
                "member": (HAS_SEQUENCING,), "gleason": False, "tsg": True},
    "combined": {"label": "Sequenced with Gleason", "clock": "sequencing",
                 "member": (HAS_SEQUENCING, GLEASON_FEATURE), "gleason": True, "tsg": True},
}


@dataclass(frozen=True)
class RuleSpec:
    base: bool       # counts the frozen federated items
    gleason: bool    # adds a Gleason item
    tsg: bool        # adds a TSG item
    tuned: bool      # selected on the training folds (else fixed)
    label: str


# Display order. The fixed Gleason >= 8 / any-TSG rules are Figure 6 b's comparators.
RULES = {
    fcc.CHECKLIST_RULE: RuleSpec(True, False, False, False, "Checklist (frozen)"),
    "checklist_retuned": RuleSpec(True, False, False, True, "Checklist, M relearned"),
    "gleason_ge8": RuleSpec(False, True, False, False, "Gleason ≥ 8"),
    "gleason": RuleSpec(False, True, False, True, "Gleason ≥ g"),
    "tsg_any": RuleSpec(False, False, True, False, "Any TP53/RB1/PTEN"),
    "tsg": RuleSpec(False, False, True, True, "TSG item"),
    "gleason_tsg": RuleSpec(False, True, True, True, "Gleason + TSG"),
    "checklist_gleason": RuleSpec(True, True, False, True, "Checklist + Gleason"),
    "checklist_tsg": RuleSpec(True, False, True, True, "Checklist + TSG"),
    "checklist_gleason_tsg": RuleSpec(True, True, True, True, "Checklist + Gleason + TSG"),
}

EXTENSION_DIR = "checklist_extension"
COMPARISON_FILE = "transfer_checklist_extension_comparison.csv"
RULES_FILE = "transfer_checklist_extension_rules.csv"
PATIENTS_FILE = "transfer_checklist_extension_patients.csv"

CELL_KEYS = ["bundle", "landmark_days", "horizon_days"]
COMPARISON_COLUMNS = ["endpoint", "cohort", "cohort_label", "clock", "bundle", "landmark_days", "horizon_days",
                      "rule", "rule_label", "n_patients", "n_excluded", "n_events", "n_events_horizon",
                      "n_positive", "auc", "auc_lo", "auc_hi", "delta_auc_vs_checklist", "delta_auc_lo",
                      "delta_auc_hi", "refit_gleason_cutoff", "refit_tsg_item", "refit_M", "refit_N",
                      "n_folds", "n_failed_folds", "note"]
RULE_COLUMNS = ["cohort", "clock", "bundle", "landmark_days", "horizon_days", "rule", "fold", "status",
                "gleason_cutoff", "tsg_item", "M", "N", "frozen_M", "frozen_N", "train_balanced_error",
                "n_train", "n_train_cases", "n_train_controls"]
PATIENT_COLUMNS = ["cohort", "clock", "bundle", "endpoint", "landmark_days", "horizon_days", ID_COL, "fold",
                   DURATION_COL, EVENT_COL, "risk_score", GLEASON_FEATURE, *TRIO_FEATURES,
                   *[f"flag_{rule}" for rule in RULES]]


def cohort_rules(cohort: str) -> list[str]:
    """The rules a cohort can evaluate: those whose added items the cohort has."""
    spec = COHORTS[cohort]
    return [rule for rule, r in RULES.items() if (spec["gleason"] or not r.gleason) and (spec["tsg"] or not r.tsg)]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def frozen_rules(bundles: dict) -> pd.DataFrame:
    """M and N of each frozen checklist model (config ``both``), keyed by bundle stem, landmark and horizon."""
    rows = []
    for stem, bundle in bundles.items():
        if bundle.get("model_family") != fcc.CHECKLIST_FAMILY:
            continue
        for model in bundle["models"]:
            if model.get("config") != "both" or model.get("horizon_days") is None:
                continue
            payload = model["checklist"]["model"]
            rows.append({"bundle": stem, "landmark_days": int(model["landmark_days"]),
                         "horizon_days": int(model["horizon_days"]), "M": int(payload["threshold"]),
                         "N": len(payload["selected_indices"])})
    return pd.DataFrame(rows, columns=[*CELL_KEYS, "M", "N"])


def cohort_frame(scores, clinical, cohort, landmark, horizon) -> pd.DataFrame:
    """The frozen checklist's rows at one landmark and horizon (every endpoint), restricted to a cohort."""
    sc = fcc._checklist_scores(scores, landmark, horizon)
    clin = fcc._normalize_ids(clinical)[[ID_COL, "landmark_days", GLEASON_FEATURE, HAS_SEQUENCING, *TRIO_FEATURES]]
    if clin.duplicated([ID_COL, "landmark_days"]).any():
        raise ValueError("clinical features must have one row per patient and landmark")
    d = sc.merge(clin, on=[ID_COL, "landmark_days"], how="left")
    member = np.logical_and.reduce([d[col].notna().to_numpy() for col in COHORTS[cohort]["member"]])
    d = d[member].assign(horizon_days=int(horizon), cohort=cohort, clock=COHORTS[cohort]["clock"])
    if d.duplicated(["endpoint", "bundle", ID_COL]).any():
        raise ValueError(f"{cohort}: duplicate checklist rows at landmark {landmark}, horizon {horizon}")
    return d.reset_index(drop=True)


def cohort_frames(sources: dict, landmarks=fcc.CHECKLIST_GRID_LANDMARKS,
                  horizons=fcc.CHECKLIST_GRID_HORIZONS) -> dict[str, pd.DataFrame]:
    """``{cohort: (scores, clinical)}`` -> ``{cohort: rows of every landmark × horizon cell}``; missing cells skipped."""
    out = {}
    for cohort, (scores, clinical) in sources.items():
        frames = []
        for landmark in landmarks:
            for horizon in horizons:
                try:
                    frames.append(cohort_frame(scores, clinical, cohort, landmark, horizon))
                except ValueError as error:
                    print(f"checklist extension: {cohort}: skipping landmark {landmark}, horizon {horizon}: {error}")
        if frames:
            out[cohort] = pd.concat(frames, ignore_index=True)
    if not out:
        raise ValueError("no checklist scores in any cohort, landmark and horizon")
    return out


def assign_folds(frames: dict[str, pd.DataFrame], n_folds=5, seed=0) -> pd.Series:
    """One patient-level fold for every patient in any cohort (shared across cohorts, clocks and cells).

    Stratified on (in a Gleason cohort, sequenced, any platinum event), so each fold has a similar share of
    every cohort and of the training outcome. Strata with fewer than ``n_folds`` patients are merged.
    """
    from sklearn.model_selection import StratifiedKFold

    rows = []
    for cohort, frame in frames.items():
        spec = COHORTS[cohort]
        platinum = frame[EVENT_COL].where(frame["endpoint"].eq(TRAINING_ENDPOINT), 0)
        rows.append(pd.DataFrame({ID_COL: frame[ID_COL].to_numpy(), "gleason": spec["gleason"],
                                  "sequenced": spec["tsg"], "platinum": platinum.fillna(0).astype(int).to_numpy()}))
    pt = pd.concat(rows, ignore_index=True).groupby(ID_COL).max().sort_index()
    if len(pt) < n_folds:
        raise ValueError(f"{len(pt)} patients cannot fill {n_folds} folds")
    strata = (pt["gleason"].astype(int).astype(str) + pt["sequenced"].astype(int).astype(str)
              + pt["platinum"].astype(str))
    counts = strata.value_counts()
    strata = strata.where(~strata.isin(counts[counts < n_folds].index), "rare")
    if 0 < (strata == "rare").sum() < n_folds:
        strata = strata.replace("rare", strata[strata != "rare"].value_counts().idxmax())
    folds = pd.Series(-1, index=pt.index, name="fold")
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(pt)), strata.to_numpy())):
        folds.iloc[test] = k
    return folds


# --------------------------------------------------------------------------- #
# Post-training
# --------------------------------------------------------------------------- #
def candidate_items(frame: pd.DataFrame) -> tuple[dict, dict]:
    """0/1 indicators of every candidate Gleason cutoff and TSG item (an uncalled gene counts as unaltered)."""
    g = pd.to_numeric(frame[GLEASON_FEATURE], errors="coerce").to_numpy(dtype=float)
    gleason = {c: np.where(np.isnan(g), np.nan, (g >= c).astype(float)) for c in GLEASON_CUTOFFS}
    trio = frame[list(TRIO_FEATURES)].apply(pd.to_numeric, errors="coerce").eq(1)
    tsg = {feature: trio[feature].to_numpy(dtype=float) for feature in TRIO_FEATURES}
    tsg[TSG_ANY] = trio.any(axis=1).to_numpy(dtype=float)
    return gleason, tsg


def balanced_weights(labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Scale the IPCW weights of cases and of controls to 0.5 each (runner._balanced_weights)."""
    out = np.zeros(len(weights), dtype=float)
    for label in (0, 1):
        mask = (labels == label) & (weights > 0)
        total = weights[mask].sum()
        if total <= 0:
            raise ValueError("training rows need weighted cases and controls")
        out[mask] = weights[mask] / (2 * total)
    return out


def training_targets(duration, event, horizon, *, min_cases=MIN_CASES, min_controls=MIN_CONTROLS, min_g=MIN_G):
    """Platinum horizon labels and balanced weights on training rows.

    Returns ``(labels, weights, info)``. ``labels`` is None when the rows fail the gate, and ``info["status"]``
    then says why.
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    info = {"n_train": len(duration), "n_train_cases": 0, "n_train_controls": 0, "status": "ok"}
    try:
        censoring = CensoringDistribution.fit(duration, event)
        labels, weights, _ = horizon_targets(duration, event, horizon, censoring, min_g)
    except ValueError as error:   # UnsupportedHorizonError is a ValueError
        return None, None, {**info, "status": f"unsupported: {error}"}
    info.update(n_train_cases=int((labels == 1).sum()), n_train_controls=int((labels == 0).sum()))
    if info["n_train_cases"] < min_cases or info["n_train_controls"] < min_controls:
        return None, None, {**info, "status": f"too few cases/controls (need {min_cases}/{min_controls})"}
    return labels, balanced_weights(labels, weights), info


def _total(rule, count, gleason, tsg, cutoff, item):
    spec = RULES[rule]
    total = np.asarray(count, dtype=float) if spec.base else np.zeros(len(count))
    if cutoff is not None:
        total = total + gleason[cutoff]
    if item is not None:
        total = total + tsg[item]
    return total


def _fixed_selection(rule, frozen_m, frozen_n) -> dict:
    if rule == fcc.CHECKLIST_RULE:
        return {"gleason_cutoff": None, "tsg_item": None, "M": int(frozen_m), "N": int(frozen_n)}
    if rule == "gleason_ge8":
        return {"gleason_cutoff": DEFAULT_GLEASON_CUTOFF, "tsg_item": None, "M": 1, "N": 1}
    if rule == "tsg_any":
        return {"gleason_cutoff": None, "tsg_item": TSG_ANY, "M": 1, "N": 1}
    raise ValueError(f"{rule} is tuned, not fixed")


def balanced_error(flag, labels, weights) -> float:
    known = labels >= 0
    return float(weights[known][np.asarray(flag)[known].astype(bool) != (labels[known] == 1)].sum())


def fit_rule(rule, count, gleason, tsg, labels, weights, frozen_m, frozen_n) -> dict:
    """The rule's selection (Gleason cutoff, TSG item, M, N) minimizing balanced error on the training rows.

    ``count`` is the frozen item count, and ``gleason`` / ``tsg`` come from candidate_items, all on the training
    rows. A fixed rule returns its fixed selection.
    """
    spec = RULES[rule]
    if not spec.tuned:
        selection = _fixed_selection(rule, frozen_m, frozen_n)
    else:
        target = frozen_m / frozen_n
        candidates = []
        for cutoff in (GLEASON_CUTOFFS if spec.gleason else (None,)):
            for item in (TSG_ITEMS if spec.tsg else (None,)):
                total = _total(rule, count, gleason, tsg, cutoff, item)
                n = (frozen_n if spec.base else 0) + (cutoff is not None) + (item is not None)
                for m in range(1, n + 1):
                    preference = (0 if cutoff in (None, DEFAULT_GLEASON_CUTOFF) else 1,
                                  abs(cutoff - DEFAULT_GLEASON_CUTOFF) if cutoff is not None else 0,
                                  cutoff if cutoff is not None else 0,
                                  TSG_ITEMS.index(item) if item is not None else 0,
                                  abs(m / n - target), m)
                    candidates.append((balanced_error(total >= m, labels, weights), preference,
                                       {"gleason_cutoff": cutoff, "tsg_item": item, "M": m, "N": n}))
        best = min(c[0] for c in candidates)
        selection = min((c for c in candidates if c[0] <= best + TIE_TOLERANCE), key=lambda c: c[1])[2]
    flag = apply_rule(rule, selection, count, gleason, tsg)
    return {**selection, "train_balanced_error": balanced_error(flag, labels, weights)}


def apply_rule(rule, selection, count, gleason, tsg) -> np.ndarray:
    """0/1 flag of a selection (NaN where an added item is missing)."""
    total = _total(rule, count, gleason, tsg, selection["gleason_cutoff"], selection["tsg_item"])
    return np.where(np.isnan(total), np.nan, (total >= selection["M"]).astype(float))


def _subset(items: dict, mask) -> dict:
    return {key: value[mask] for key, value in items.items()}


def cross_fit(frames: dict[str, pd.DataFrame], folds: pd.Series, frozen: pd.DataFrame, n_folds=5, *,
              min_cases=MIN_CASES, min_controls=MIN_CONTROLS, min_g=MIN_G):
    """Out-of-fold flags of every rule, plus the selections of each fold and of the full-cohort refit.

    Returns ``(patients, rules)``: one patient row per cohort × cell × endpoint, with a ``flag_<rule>`` column per
    rule (NaN where the training fold failed the gate or the cohort lacks the rule's items), and one rules row per
    cohort × cell × rule × fold (``fold == "all"`` is the full-cohort refit).
    """
    frozen = frozen.set_index(CELL_KEYS)[["M", "N"]]
    patients, rules = [], []
    for cohort, frame in frames.items():
        names = cohort_rules(cohort)
        for (bundle, landmark, horizon), cell in frame.groupby(CELL_KEYS, sort=False):
            key = (bundle, int(landmark), int(horizon))
            if key not in frozen.index:
                raise ValueError(f"no frozen M/N for {key}")
            frozen_m, frozen_n = (int(v) for v in frozen.loc[key])
            cell = cell.reset_index(drop=True)
            if not cell["flag"].astype(int).eq(cell["risk_score"].ge(frozen_m).astype(int)).all():
                raise ValueError(f"{cohort} {key}: flag != (item count >= M={frozen_m})")
            cell["fold"] = cell[ID_COL].map(folds)
            if cell["fold"].isna().any():
                raise ValueError(f"{cohort} {key}: patients without a fold")
            count = cell["risk_score"].to_numpy(dtype=float)
            gleason, tsg = candidate_items(cell)
            platinum = cell["endpoint"].eq(TRAINING_ENDPOINT).to_numpy()
            fold = cell["fold"].to_numpy(dtype=int)
            flags = {rule: np.full(len(cell), np.nan) for rule in names}
            for k in [*range(n_folds), "all"]:
                train = platinum & (fold != k) if k != "all" else platinum
                labels, weights, info = training_targets(
                    cell.loc[train, DURATION_COL], cell.loc[train, EVENT_COL], horizon,
                    min_cases=min_cases, min_controls=min_controls, min_g=min_g)
                test = fold == k if k != "all" else None
                for rule in names:
                    spec = RULES[rule]
                    row = {"cohort": cohort, "clock": COHORTS[cohort]["clock"], "bundle": bundle,
                           "landmark_days": int(landmark), "horizon_days": int(horizon), "rule": rule,
                           "fold": str(k), "frozen_M": frozen_m, "frozen_N": frozen_n, **info}
                    if labels is None and spec.tuned:
                        rules.append(row)
                        continue
                    if labels is None:
                        selection = {**_fixed_selection(rule, frozen_m, frozen_n), "train_balanced_error": np.nan}
                    else:
                        selection = fit_rule(rule, count[train], _subset(gleason, train), _subset(tsg, train),
                                             labels, weights, frozen_m, frozen_n)
                    rules.append({**row, **selection})
                    if test is not None:
                        flags[rule][test] = apply_rule(rule, selection, count[test], _subset(gleason, test),
                                                       _subset(tsg, test))
            out = cell.assign(**{f"flag_{rule}": flags.get(rule, np.nan) for rule in RULES})
            patients.append(out.reindex(columns=PATIENT_COLUMNS))
    return (pd.concat(patients, ignore_index=True),
            pd.DataFrame(rules).reindex(columns=RULE_COLUMNS))


# --------------------------------------------------------------------------- #
# Held-out comparison
# --------------------------------------------------------------------------- #
def extension_comparison(patients: pd.DataFrame, rules: pd.DataFrame | None = None, *, n_bootstrap=200, seed=0,
                         min_events=5, n_folds=5) -> pd.DataFrame:
    """Horizon AUC of each rule's out-of-fold flag, with paired bootstrap deltas against the frozen checklist.

    Every rule in a cohort × cell × endpoint is scored on the same patients (those with a flag for every rule).
    Fewer than ``min_events`` events by the horizon leaves the AUCs NaN (note ``underpowered``).
    """
    refit = failed = None
    if rules is not None and not rules.empty:
        r = rules.assign(landmark_days=rules["landmark_days"].astype(int),
                         horizon_days=rules["horizon_days"].astype(int))
        refit = (r[r["fold"].astype(str).eq("all")]
                 .set_index(["cohort", *CELL_KEYS, "rule"])[["gleason_cutoff", "tsg_item", "M", "N"]])
        tuned = r[r["fold"].astype(str).ne("all") & r["rule"].map(lambda x: RULES[x].tuned)]
        failed = (tuned.assign(fail=tuned["status"].ne("ok"))
                  .groupby(["cohort", *CELL_KEYS, "fold"])["fail"].any()
                  .groupby(["cohort", *CELL_KEYS]).sum())
    rows = []
    keys = ["cohort", "clock", "bundle", "endpoint", "landmark_days", "horizon_days"]
    for (cohort, clock, bundle, endpoint, landmark, horizon), z in patients.groupby(keys, sort=False):
        names = cohort_rules(cohort)
        flag_cols = [f"flag_{rule}" for rule in names]
        ok = np.isfinite(z[flag_cols].to_numpy(dtype=float)).all(axis=1)
        n_excluded = int((~ok).sum())
        z = z[ok]
        duration = z[DURATION_COL].to_numpy(dtype=float)
        event = z[EVENT_COL].to_numpy(dtype=int)
        markers = {rule: z[f"flag_{rule}"].to_numpy(dtype=float) for rule in names}
        n_events_horizon = int(((event == 1) & (duration <= horizon)).sum())
        note = ""
        if len(z) == 0:
            note = "no patient with every out-of-fold flag"
        elif n_events_horizon < min_events:
            note = "underpowered"
        if note:
            auc = {rule: np.nan for rule in names}
            boot = {rule: np.full(1, np.nan) for rule in names}
        else:
            auc = {rule: fet.fixed_horizon_mean_auc(duration, event, marker, [horizon])
                   for rule, marker in markers.items()}
            boot = fet.bootstrap_fixed_horizon_mean_auc(duration, event, markers, [horizon], n_bootstrap, seed)
            if not np.isfinite(auc[fcc.CHECKLIST_RULE]):
                note = "horizon not estimable"
        cell = (cohort, bundle, int(landmark), int(horizon))
        for rule in names:
            lo, hi = fet._ci(boot[rule])
            delta_lo, delta_hi = fet._ci(boot[rule] - boot[fcc.CHECKLIST_RULE])
            sel = (refit.loc[(*cell, rule)] if refit is not None and (*cell, rule) in refit.index
                   else pd.Series({"gleason_cutoff": np.nan, "tsg_item": None, "M": np.nan, "N": np.nan}))
            rows.append({"endpoint": endpoint, "cohort": cohort, "cohort_label": COHORTS[cohort]["label"],
                         "clock": clock, "bundle": bundle, "landmark_days": int(landmark),
                         "horizon_days": int(horizon), "rule": rule, "rule_label": RULES[rule].label,
                         "n_patients": len(z), "n_excluded": n_excluded, "n_events": int(event.sum()),
                         "n_events_horizon": n_events_horizon, "n_positive": int(np.nansum(markers[rule])),
                         "auc": auc[rule], "auc_lo": lo, "auc_hi": hi,
                         "delta_auc_vs_checklist": auc[rule] - auc[fcc.CHECKLIST_RULE],
                         "delta_auc_lo": delta_lo, "delta_auc_hi": delta_hi,
                         "refit_gleason_cutoff": sel["gleason_cutoff"], "refit_tsg_item": sel["tsg_item"],
                         "refit_M": sel["M"], "refit_N": sel["N"], "n_folds": n_folds,
                         "n_failed_folds": int(failed.get(cell, 0)) if failed is not None else np.nan,
                         "note": note})
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def run_extension(sources: dict, frozen: pd.DataFrame, *, landmarks=fcc.CHECKLIST_GRID_LANDMARKS,
                  horizons=fcc.CHECKLIST_GRID_HORIZONS, n_folds=5, n_bootstrap=200, seed=0, min_events=5):
    """``{cohort: (scores, clinical)}`` -> ``(comparison, rules, patients)`` for every cohort × landmark × horizon."""
    frames = cohort_frames(sources, landmarks, horizons)
    folds = assign_folds(frames, n_folds, seed)
    patients, rules = cross_fit(frames, folds, frozen, n_folds)
    comparison = extension_comparison(patients, rules, n_bootstrap=n_bootstrap, seed=seed,
                                      min_events=min_events, n_folds=n_folds)
    return comparison, rules, patients


def write_extension(out_dir, comparison, rules, patients) -> Path:
    out_dir = Path(out_dir) / EXTENSION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / COMPARISON_FILE, index=False)
    rules.to_csv(out_dir / RULES_FILE, index=False)
    patients.to_csv(out_dir / PATIENTS_FILE, index=False)
    print(f"wrote checklist extension -> {out_dir}")
    return out_dir


def plot_extension(comparison: pd.DataFrame, landmark: int, endpoints=("platinum", "avpc", "nepc")):
    """Quick look: AUC (95% CI) of every rule, rows = endpoint, columns = cohort, one marker per horizon."""
    import matplotlib.pyplot as plt

    d = comparison[comparison["landmark_days"].eq(landmark)]
    cohorts = [c for c in COHORTS if c in set(d["cohort"])]
    endpoints = [e for e in endpoints if e in set(d["endpoint"])]
    horizons = sorted(d["horizon_days"].unique())
    fig, axes = plt.subplots(len(endpoints), len(cohorts), squeeze=False, sharey=True,
                             figsize=(4.2 * len(cohorts), 2.8 * len(endpoints)))
    for i, endpoint in enumerate(endpoints):
        for j, cohort in enumerate(cohorts):
            ax = axes[i][j]
            names = cohort_rules(cohort)
            for k, horizon in enumerate(horizons):
                z = d[d["endpoint"].eq(endpoint) & d["cohort"].eq(cohort) & d["horizon_days"].eq(horizon)]
                z = z.set_index("rule").reindex(names)
                x = np.arange(len(names)) + (k - (len(horizons) - 1) / 2) * 0.25
                ax.errorbar(x, z["auc"], yerr=[z["auc"] - z["auc_lo"], z["auc_hi"] - z["auc"]], fmt="o", ms=4,
                            capsize=2, label=f"{horizon // 365}-year")
            ax.axhline(0.5, ls=":", c="grey", lw=0.8)
            ax.set_xticks(range(len(names)), [RULES[r].label for r in names], rotation=60, ha="right", fontsize=7)
            if i == 0:
                ax.set_title(COHORTS[cohort]["label"], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{endpoint.upper()} AUC")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Checklist + Gleason / TSG, landmark +{landmark} days (out-of-fold, tuned on platinum)", fontsize=10)
    fig.tight_layout()
    return fig
