"""Local (DFCI) shallow survival tree, cross-fitted on time to platinum and transferred to NEPC / AVPC.

This is the single-site counterpart of ``rhino_scripts/feature_set_survival_tree``. The model code is not rewritten
here. ``_vendor/`` holds byte-identical copies of the federated app's ``survival_tree_core.py`` (the rpart
``method="exp"`` tree), ``checklist_core.py`` and ``federated_checklist.py`` (the checklist items). Their sha256
values are recorded in ``_vendor/VENDORED.json``.

**Frames.** The rows come from :mod:`survival_common.federated_endpoint_transfer`. The Stage-2 ADT long frame is
mapped to OMOP lab names, and each endpoint's date is written into the event slot. Each landmark frame is then
built by :func:`survival_common.federated_scoring.build_landmark_frame`, so the features are named and computed
exactly as in the federated bundles and are identical across endpoints. Only the at-risk set and the outcome differ
between endpoints.

**Items.** These are the federated items computed on exact local values:

- age plus the ``<lab>__{mean,min,max,last,delta}`` allowlist (``feature_group``);
- labs with coverage of at least ``min_coverage``;
- deciles of the fit rows, rounded to clinical cutoffs (``guideline_threshold``), each giving ``<=`` and ``>=`` items;
- missing values make the item false;
- items that are constant on the fit rows are dropped.

Each fit scope builds its own items from its own rows, as each federated fit scope does.

**Tree and selection** follow the federated app:

- the depth-3 tree is grown once and truncated to depths 1 and 2 (followed by the ``cp = 0`` collapse);
- the grid is ``max_depth`` x ``min_leaf_frac``, with ``min_leaf = ceil(frac * n_fit)``;
- the key is inner 5-fold mean AUC(t) when every fold has it, otherwise C;
- ties go to higher C, fewer leaves, the larger ``min_leaf_frac``, then the smaller depth.

``both`` may split on every item and ``baseline`` on age items only.

**Cross-fitting** follows ``COMPASS/federated_models/checklist_extension.py``:

- One patient-level fold split, stratified on any platinum event, is shared by every landmark and endpoint.
- Each outer fold's tree is selected and fit on the other folds' platinum rows only.
- It is applied to the held-out fold's platinum, NEPC and AVPC rows.
- The NEPC and AVPC labels are never used for fitting or selection.
- A refit on the whole platinum cohort (``fold == "all"``) gives the tree one would report. It is descriptive only,
  and it is exported as a ``survival_tree`` bundle in the federated schema.

**Scores.**

- The risk score (used for C and the Cox slope) is the leaf's shrunken rate.
- AUC(t) scores the leaf KM failure ``1 - S_leaf(t)`` on the federated week grid, as the federated controller grades
  the tree.
- Rates come from fold-specific trees, each on its own ``Lambda0`` scale, and are pooled out of fold.
  ``S_leaf(t)`` is absolute, so AUC(t) does not depend on that scale.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_VENDOR = Path(__file__).resolve().parent / "_vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import checklist_core as ccore  # noqa: E402  (vendored, byte-identical to rhino_scripts)
import federated_checklist as fcl  # noqa: E402
import survival_tree_core as stc  # noqa: E402

from survival_common import federated_endpoint_transfer as fet  # noqa: E402
from survival_common import federated_scoring as fs  # noqa: E402
from survival_common.helper import MIN_IPCW_TIMELINE_COVERAGE, MIN_IPCW_VALID_HORIZONS  # noqa: E402

ID_COL = fs.ID_COL
AGE_COL = fs.AGE_COL
DURATION_COL = fs.DURATION_COL
EVENT_COL = fs.EVENT_COL
TIME_UNIT_DAYS = 7
TRAINING_ENDPOINT = "platinum"
ENDPOINTS = ("platinum", "nepc", "avpc")
BOTH, BASELINE = "both", "baseline"
CONFIGS = (BOTH, BASELINE)
BUNDLE_STEM = "local_tree"
MODEL_FAMILY = "survival_tree"

METRICS_FILE = "local_survival_tree_metrics.csv"
AUC_FILE = "local_survival_tree_auc_t.csv"
LEAVES_FILE = "local_survival_tree_leaves.csv"
SELECTION_FILE = "local_survival_tree_selection.csv"
CV_FILE = "local_survival_tree_inner_cv.csv"
PATIENTS_FILE = "local_survival_tree_patients.csv"
BUNDLE_FILE = "local_survival_tree_model.json"


@dataclass
class TreeConfig:
    landmarks: tuple = (0, 90, 180)
    endpoints: tuple = ENDPOINTS
    max_depths: tuple = (1, 2, 3)
    min_leaf_fracs: tuple = (0.02, 0.05, 0.10)
    n_folds: int = 5
    n_inner_folds: int = 5
    min_coverage: float = 0.20
    quantiles: tuple = ccore.DEFAULT_QUANTILES
    round_thresholds: bool = True
    exclusion: str = "none"
    max_followup_days: int = fs.DEFAULT_MAX_FOLLOWUP_DAYS
    n_bootstrap: int = 200
    seed: int = 0
    horizons_days: tuple = (365, 1825)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #
def build_frames(local: pd.DataFrame, lab_mapping: dict, cfg: TreeConfig, verbose=True):
    """``{(endpoint, landmark): wide frame}`` and per-endpoint prep stats, via the federated chain."""
    frames, stats = {}, []
    for endpoint in cfg.endpoints:
        long_df, st = fet.to_federated_long(local, endpoint, lab_mapping)
        long_df, excl = fs.apply_exclusion(long_df, cfg.exclusion)
        st["exclusion"] = excl
        stats.append(st)
        for lm in cfg.landmarks:
            wide, _ = fs.build_landmark_frame(long_df, lm, max_followup_days=cfg.max_followup_days)
            frames[(endpoint, int(lm))] = wide.reset_index(drop=True)
            if verbose:
                print(f"[tree] {endpoint} landmark {lm}d: {len(wide):,} at risk, "
                      f"{int(wide[EVENT_COL].sum()) if len(wide) else 0:,} events")
    return frames, stats


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Allowlist (``checklist_executor._feature_cols``): age plus ``<lab>__<stat>``."""
    cols = sorted(c for c in frame.columns if c != AGE_COL and ccore.feature_group(c, AGE_COL) is not None)
    return ([AGE_COL] if AGE_COL in frame.columns else []) + cols


def _numeric(frame: pd.DataFrame, feature: str) -> np.ndarray:
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #
@dataclass
class ItemSpace:
    spec: dict
    binarizer: object

    @property
    def items(self) -> list[dict]:
        return self.spec["items"]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.binarizer.transform(frame).astype(np.int8)

    def allowed(self, config: str) -> np.ndarray:
        return np.array([it["op"] != "is_missing" and (config == BOTH or it["feature"] == AGE_COL)
                         for it in self.items], dtype=bool)


def item_space(rows: pd.DataFrame, cfg: TreeConfig) -> ItemSpace:
    """The federated item pipeline on one fit scope's rows, with exact order statistics."""
    presence, resolved = {}, {}
    for feature in feature_columns(rows):
        values = _numeric(rows, feature)
        observed = np.sort(values[np.isfinite(values)])
        if observed.size:
            presence[feature] = (int(observed.size), float(observed[0]), float(observed[-1]))
            resolved[feature] = observed
    n = len(rows)
    plan = fcl.plan_binarizer(presence, n, age_col=AGE_COL, min_coverage=cfg.min_coverage,
                              quantiles=cfg.quantiles)
    spec = fcl.assemble_candidates(plan, {f: resolved[f] for f in plan}, n, age_col=AGE_COL,
                                   quantiles=cfg.quantiles, round_thresholds=cfg.round_thresholds)
    kw = dict(age_col=AGE_COL, id_col=ID_COL, min_patient_coverage=cfg.min_coverage, quantiles=cfg.quantiles)
    X = ccore.binarizer_from_spec(spec, **kw).transform(rows)
    spec, _ = fcl.drop_constant_items(spec, X.sum(axis=0).astype(int).tolist())
    return ItemSpace(spec, ccore.binarizer_from_spec(spec, **kw))


# --------------------------------------------------------------------------- #
# Fitting and metrics
# --------------------------------------------------------------------------- #
def _units(duration_days) -> np.ndarray:
    return np.ceil(np.asarray(duration_days, dtype=float) / TIME_UNIT_DAYS)


def leaf_curves(tree, X, duration_days, event) -> dict:
    leaf = tree.apply(X)
    units, event = _units(duration_days), np.asarray(event, dtype=int)
    return {lf: stc.kaplan_meier(units[leaf == lf], event[leaf == lf], np.ones(int((leaf == lf).sum())))
            for lf in tree.leaves()}


def failure_matrix(curves: dict, leaf: np.ndarray, grid_units) -> np.ndarray:
    """``1 - S_leaf(t)`` per row and grid week (KM step, held past its end)."""
    grid_units = np.asarray(grid_units, dtype=float)
    table = {lf: 1.0 - stc.step_at(c, grid_units) for lf, c in curves.items()}
    if not len(leaf):
        return np.zeros((0, len(grid_units)))
    return np.vstack([table[int(lf)] for lf in leaf])


def tree_metrics(tree, curves, fit_units, fit_event, X_eval, eval_units, eval_event, grid) -> dict:
    """The controller's ``_metrics`` on row arrays (weeks; censoring reference = the fit rows)."""
    from sksurv.metrics import brier_score, cumulative_dynamic_auc
    from sksurv.util import Surv

    out = {"c_index": np.nan, "mean_auc_t": np.nan, "integrated_brier": np.nan,
           "n": int(len(eval_units)), "events": int(np.sum(eval_event))}
    if out["events"] == 0:
        return out
    leaf = tree.apply(X_eval)
    rate = np.array([tree.leaf_rate(int(lf)) for lf in leaf])
    out["c_index"] = fet.harrell_c(eval_units, eval_event, rate)
    valid = grid[(grid > eval_units.min()) & (grid < eval_units.max())]
    if len(valid) < MIN_IPCW_VALID_HORIZONS or len(valid) / max(len(grid), 1) < MIN_IPCW_TIMELINE_COVERAGE:
        return out
    train = Surv.from_arrays(event=np.asarray(fit_event, dtype=bool), time=np.asarray(fit_units, dtype=float))
    test = Surv.from_arrays(event=np.asarray(eval_event, dtype=bool), time=np.asarray(eval_units, dtype=float))
    fail = failure_matrix(curves, leaf, valid)
    try:
        out["mean_auc_t"] = float(cumulative_dynamic_auc(train, test, fail, valid)[1])
    except (ValueError, ArithmeticError):
        pass
    try:
        _, values = brier_score(train, test, 1.0 - fail, valid)
        out["integrated_brier"] = float(np.trapezoid(values, valid) / (valid[-1] - valid[0]))
    except (ValueError, ArithmeticError):
        pass
    return out


def grow(space: ItemSpace, X, duration_days, event, config: str, frac: float, max_depth: int):
    min_leaf = max(1, math.ceil(frac * len(X)))
    tree = stc.fit_pooled_tree(X, duration_days, event, max_depth=max_depth, min_leaf=min_leaf,
                               allowed=space.allowed(config), items=list(space.items), finalize=False)
    return tree, min_leaf


def _summarize(fold_df: pd.DataFrame, n_leaves: dict, n_folds: int) -> list[dict]:
    rows = []
    for (depth, frac), block in fold_df.groupby(["max_depth", "min_leaf_frac"]):
        valid = block[block["valid"]]
        auc = block["mean_auc_t"].to_numpy(dtype=float)
        ibs = block["integrated_brier"].to_numpy(dtype=float)
        rows.append({
            "max_depth": int(depth), "min_leaf_frac": float(frac),
            "cv_mean": float(valid["c_index"].mean()) if len(valid) else np.nan,
            "mean_auc_t_cv_mean": float(np.nanmean(auc)) if np.isfinite(auc).any() else np.nan,
            "integrated_brier_cv_mean": float(np.nanmean(ibs)) if np.isfinite(ibs).any() else np.nan,
            "n_valid_folds": int(len(valid)), "n_valid_auc_t_folds": int(np.isfinite(auc).sum()),
            "all_folds_valid": bool(len(valid) == n_folds),
            "n_leaves_train_val": n_leaves.get((int(depth), float(frac)), np.nan),
        })
    return rows


def select(summary: list[dict], n_folds: int) -> dict | None:
    """``tree_controller._select``: AUC if every fold has it, else C; then C, folds, fewer leaves, larger frac."""
    df = pd.DataFrame(summary)
    if df.empty:
        return None
    valid = df.loc[df["all_folds_valid"]]
    if valid.empty:
        return None
    key = "mean_auc_t_cv_mean" if valid["n_valid_auc_t_folds"].eq(n_folds).any() else "cv_mean"
    best = valid.sort_values([key, "cv_mean", "n_valid_folds", "n_leaves_train_val", "min_leaf_frac", "max_depth"],
                             ascending=[False, False, False, True, False, True],
                             na_position="last", kind="mergesort").iloc[0].to_dict()
    best["selection_key"] = key
    return best


@dataclass
class FittedTree:
    config: str
    space: ItemSpace
    tree: object
    curves: dict
    best: dict
    min_leaf: int
    grid: np.ndarray
    n_fit: int
    events_fit: int

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        leaf = self.tree.apply(self.space.transform(frame))
        return leaf, np.array([self.tree.leaf_rate(int(lf)) for lf in leaf])


def fit_selected(rows: pd.DataFrame, cfg: TreeConfig, seed: int, tag: str = "") -> tuple[dict, list[dict]]:
    """Inner-CV selection and refit on ``rows`` (platinum) for both configs.

    Returns ``({config: FittedTree}, cv_rows)``; a config with no candidate valid in every inner fold is absent.
    """
    from sklearn.model_selection import StratifiedKFold

    rows = rows.reset_index(drop=True)
    duration = rows[DURATION_COL].to_numpy(dtype=float)
    event = rows[EVENT_COL].to_numpy(dtype=int)
    units = _units(duration)
    max_depth = max(cfg.max_depths)
    if event.sum() == 0:
        return {}, []
    grid = fet.auc_horizon_grid(duration, event, max_followup_days=cfg.max_followup_days)

    fold_rows = []
    splitter = StratifiedKFold(n_splits=cfg.n_inner_folds, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(splitter.split(np.zeros(len(rows)), event)):
        space = item_space(rows.iloc[tr], cfg)
        X_tr, X_va = space.transform(rows.iloc[tr]), space.transform(rows.iloc[va])
        for config in CONFIGS:
            for frac in cfg.min_leaf_fracs:
                grown, min_leaf = grow(space, X_tr, duration[tr], event[tr], config, frac, max_depth)
                for depth in cfg.max_depths:
                    tree = stc.finalize_tree(grown.truncate(depth))
                    curves = leaf_curves(tree, X_tr, duration[tr], event[tr])
                    m = tree_metrics(tree, curves, units[tr], event[tr], X_va, units[va], event[va], grid)
                    fold_rows.append({"scope": tag, "config": config, "max_depth": depth, "min_leaf_frac": frac,
                                      "min_leaf": min_leaf, "inner_fold": k, "n_leaves": len(tree.leaves()),
                                      **m, "valid": bool(np.isfinite(m["c_index"]))})
    fold_df = pd.DataFrame(fold_rows)

    space = item_space(rows, cfg)
    X = space.transform(rows)
    fitted, cv_rows = {}, [{**r, "kind": "fold"} for r in fold_rows]
    for config in CONFIGS:
        grown = {}
        for frac in cfg.min_leaf_fracs:
            grown[frac] = grow(space, X, duration, event, config, frac, max_depth)
        n_leaves = {(d, float(f)): len(stc.finalize_tree(g.truncate(d)).leaves())
                    for f, (g, _) in grown.items() for d in cfg.max_depths}
        summary = _summarize(fold_df[fold_df["config"] == config], n_leaves, cfg.n_inner_folds)
        cv_rows += [{"scope": tag, "config": config, "kind": "summary", **r} for r in summary]
        best = select(summary, cfg.n_inner_folds)
        if best is None:
            continue
        g, min_leaf = grown[min(grown, key=lambda f: abs(f - best["min_leaf_frac"]))]
        tree = stc.finalize_tree(g.truncate(int(best["max_depth"])))
        fitted[config] = FittedTree(config, space, tree, leaf_curves(tree, X, duration, event), best,
                                    min_leaf, grid, len(rows), int(event.sum()))
    return fitted, cv_rows


# --------------------------------------------------------------------------- #
# Cross-fitting
# --------------------------------------------------------------------------- #
def assign_folds(frames: dict, n_folds: int, seed: int) -> pd.Series:
    """One patient-level fold for every patient in any frame, stratified on any platinum event."""
    from sklearn.model_selection import StratifiedKFold

    parts = []
    for (endpoint, _), wide in frames.items():
        ev = wide[EVENT_COL].to_numpy(dtype=int) if endpoint == TRAINING_ENDPOINT else np.zeros(len(wide), int)
        parts.append(pd.DataFrame({ID_COL: wide[ID_COL].to_numpy(), "platinum": ev}))
    pt = pd.concat(parts, ignore_index=True).groupby(ID_COL)["platinum"].max().sort_index()
    if len(pt) < n_folds:
        raise ValueError(f"{len(pt)} patients cannot fill {n_folds} folds")
    strata = pt.to_numpy()
    if np.bincount(strata, minlength=2).min() < n_folds:
        strata = np.zeros(len(pt), dtype=int)
    folds = pd.Series(-1, index=pt.index, name="fold")
    for k, (_, test) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed)
                                  .split(np.zeros(len(pt)), strata)):
        folds.iloc[test] = k
    return folds


def tree_rows(fitted: FittedTree, landmark: int, fold, cfg: TreeConfig) -> tuple[dict, list[dict]]:
    """Selection row and leaf rows (rule, n, events, rate, KM at fixed horizons with bands)."""
    t, best = fitted.tree, fitted.best
    sel = {"landmark_days": landmark, "fold": str(fold), "config": fitted.config,
           "max_depth": int(best["max_depth"]), "min_leaf_frac": float(best["min_leaf_frac"]),
           "min_leaf": fitted.min_leaf, "n_leaves": len(t.leaves()), "selection_key": best["selection_key"],
           "cv_mean": best["cv_mean"], "mean_auc_t_cv_mean": best["mean_auc_t_cv_mean"],
           "n_fit": fitted.n_fit, "events_fit": fitted.events_fit, "n_items": len(fitted.space.items),
           "splits": " | ".join(t.items[n.split]["name"] for _, n in sorted(t.nodes.items()) if n.split is not None)}
    leaves = []
    for lf in t.leaves():
        node, km = t.nodes[lf], fitted.curves[lf]
        row = {"landmark_days": landmark, "fold": str(fold), "config": fitted.config, "leaf": lf,
               "rule": t.rule_text(lf), "n": node.n, "events": node.d, "rate": t.leaf_rate(lf),
               "relative_rate": t.leaf_rate(lf) / t.root_rate}
        for h in cfg.horizons_days:
            for key in ("survival", "lower", "upper"):
                name = f"S_{h}d" if key == "survival" else f"S_{h}d_{key}"
                row[name] = float(stc.step_at(km, [h / TIME_UNIT_DAYS], key)[0])
        leaves.append(row)
    return sel, leaves


def cross_fit(frames: dict, cfg: TreeConfig, verbose=True) -> dict:
    """Out-of-fold leaf scores on every endpoint, the fold trees, and the full-cohort refits."""
    folds = assign_folds(frames, cfg.n_folds, cfg.seed)
    patients, selection, leaves, cv, models, refits = [], [], [], [], {}, {}
    for lm in cfg.landmarks:
        plat = frames.get((TRAINING_ENDPOINT, int(lm)))
        if plat is None or plat.empty:
            continue
        pfold = plat[ID_COL].map(folds).to_numpy()
        for k in [*range(cfg.n_folds), "all"]:
            train = plat if k == "all" else plat[pfold != k]
            fitted, cv_rows = fit_selected(train, cfg, seed=cfg.seed, tag=f"lm{lm}|fold{k}")
            cv += [{"landmark_days": lm, "outer_fold": str(k), **r} for r in cv_rows]
            for config, ft in fitted.items():
                s, lv = tree_rows(ft, lm, k, cfg)
                selection.append(s)
                leaves += lv
                if k == "all":
                    refits[(lm, config)] = ft
                    continue
                models[(lm, config, k)] = ft
                for endpoint in cfg.endpoints:
                    wide = frames.get((endpoint, int(lm)))
                    if wide is None or wide.empty:
                        continue
                    held = wide[wide[ID_COL].map(folds).to_numpy() == k]
                    leaf, rate = ft.predict(held)
                    patients.append(pd.DataFrame({
                        "endpoint": endpoint, "landmark_days": lm, "config": config, "fold": k,
                        ID_COL: held[ID_COL].to_numpy(), "leaf": leaf, "risk_score": rate,
                        DURATION_COL: held[DURATION_COL].to_numpy(dtype=float),
                        EVENT_COL: held[EVENT_COL].to_numpy(dtype=int)}))
            if verbose:
                got = {c: (f"depth {int(ft.best['max_depth'])}, {len(ft.tree.leaves())} leaves")
                       for c, ft in fitted.items()}
                print(f"[tree] landmark {lm}d fold {k}: {got or 'no valid candidate'}")
    return {"folds": folds,
            "patients": pd.concat(patients, ignore_index=True) if patients else pd.DataFrame(),
            "selection": pd.DataFrame(selection), "leaves": pd.DataFrame(leaves), "inner_cv": pd.DataFrame(cv),
            "models": models, "refits": refits}


# --------------------------------------------------------------------------- #
# Held-out evaluation
# --------------------------------------------------------------------------- #
def evaluate(result: dict, cfg: TreeConfig, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold C, mean AUC(t), Uno's C, IBS and Cox slope per endpoint x landmark x config.

    Both configs are scored on the same patients (those with an out-of-fold score for both), so the bootstrap
    deltas against ``baseline`` are paired. The grid is rebuilt on each (endpoint, landmark) cohort, as in
    :func:`fet.evaluate_transfer`, and censoring comes from that cohort. The bootstrap is conditional on the fold
    trees (it leaves out the variability of re-selecting and re-fitting them).
    """
    pts, models = result["patients"], result["models"]
    rows, auc_rows = [], []
    if pts.empty:
        return pd.DataFrame(rows), pd.DataFrame(auc_rows)
    for (endpoint, lm), z in pts.groupby(["endpoint", "landmark_days"], sort=False):
        wide = z.pivot_table(index=ID_COL, columns="config", values="risk_score", aggfunc="first")
        ids = wide.dropna().index
        if not len(ids):
            continue
        base = z[z["config"] == z["config"].iloc[0]].drop_duplicates(ID_COL).set_index(ID_COL).loc[ids]
        duration = base[DURATION_COL].to_numpy(dtype=float)
        event = base[EVENT_COL].to_numpy(dtype=int)
        grid = fet.auc_horizon_grid(duration, event, max_followup_days=cfg.max_followup_days)
        risks, fails = {}, {}
        for config in [c for c in CONFIGS if c in wide.columns]:
            zc = z[z["config"] == config].drop_duplicates(ID_COL).set_index(ID_COL).loc[ids]
            risks[config] = zc["risk_score"].to_numpy(dtype=float)
            fail = np.zeros((len(ids), len(grid)))
            for k, idx in zc.reset_index().groupby("fold").groups.items():
                ft = models[(int(lm), config, int(k))]
                fail[np.asarray(idx)] = failure_matrix(ft.curves, zc["leaf"].to_numpy()[np.asarray(idx)], grid)
            fails[config] = fail
        n_b = cfg.n_bootstrap
        boot_c = fet.bootstrap_c(duration, event, risks, n_b, cfg.seed) if n_b else {}
        boot_a = fet.bootstrap_mean_auc(duration, event, fails, grid, n_b, cfg.seed) if n_b else {}
        for config, risk in risks.items():
            row = {"endpoint": endpoint, "bundle": BUNDLE_STEM, "model_family": MODEL_FAMILY,
                   "analysis": "local_cross_fit", "trained_endpoint": TRAINING_ENDPOINT,
                   "landmark_days": int(lm), "config": config, "n_patients": len(ids),
                   "n_excluded": int(len(wide) - len(ids)), "n_events": int(event.sum()),
                   "median_followup_days": float(np.median(duration)),
                   "c_index": fet.harrell_c(duration, event, risk)}
            if config in boot_c:
                row["c_index_lo"], row["c_index_hi"] = fet._ci(boot_c[config])
            ipcw, auc_df = fet.ipcw_metrics(duration, event, risk, grid, auc_score=fails[config])
            row.update(ipcw)
            if config in boot_a:
                row["mean_auc_lo"], row["mean_auc_hi"] = fet._ci(boot_a[config])
            if config != BASELINE and BASELINE in risks:
                row["c_index_baseline"] = fet.harrell_c(duration, event, risks[BASELINE])
                row["delta_c_vs_baseline"] = row["c_index"] - row["c_index_baseline"]
                row["mean_auc_baseline"] = fet.mean_auc(duration, event, fails[BASELINE], grid)
                row["delta_auc_vs_baseline"] = row["mean_auc"] - row["mean_auc_baseline"]
                if boot_c:
                    row["delta_c_lo"], row["delta_c_hi"] = fet._ci(boot_c[config] - boot_c[BASELINE])
                if boot_a:
                    row["delta_auc_lo"], row["delta_auc_hi"] = fet._ci(boot_a[config] - boot_a[BASELINE])
            row.update(fet.cox_slope(duration, event, risk))
            rows.append(row)
            if not auc_df.empty:
                auc_rows.append(auc_df.assign(endpoint=endpoint, bundle=BUNDLE_STEM, landmark_days=int(lm),
                                              config=config))
            if verbose:
                print(f"[tree] {endpoint} {lm}d {config}: C={row['c_index']:.3f}, "
                      f"mean AUC(t)={row['mean_auc']:.3f}, delta AUC vs age={row.get('delta_auc_vs_baseline', np.nan):+.3f}")
    auc = pd.concat(auc_rows, ignore_index=True) if auc_rows else pd.DataFrame()
    return pd.DataFrame(rows), auc


# --------------------------------------------------------------------------- #
# Bundle export (federated schema)
# --------------------------------------------------------------------------- #
def bundle_model(ft: FittedTree, landmark: int) -> dict:
    """``tree_controller._record_tree``'s model dict for a local fit."""
    tree, items = ft.tree, ft.space.items
    used = sorted({n.split for n in tree.nodes.values() if n.split is not None})
    local = {item: i for i, item in enumerate(used)}
    state = ft.space.binarizer.subset(used).to_dict()
    nodes = []
    for nid in sorted(tree.nodes):
        node = tree.nodes[nid]
        nodes.append({"id": nid, "depth": node.depth, "n": node.n, "events": node.d, "z": node.z,
                      "split_item": None if node.split is None else local[node.split],
                      "item": None if node.split is None else dict(items[node.split]), "gain": node.gain,
                      "false_child": 2 * nid if node.split is not None else None,
                      "true_child": 2 * nid + 1 if node.split is not None else None})
    leaves = [{"id": lf, "rule": tree.rule_text(lf), "path": [[local[i], v] for i, v in tree.nodes[lf].path],
               "n": tree.nodes[lf].n, "events": tree.nodes[lf].d, "rate": tree.leaf_rate(lf),
               "km": ft.curves[lf]} for lf in tree.leaves()]
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in ft.best.items()}
    return {"landmark_days": int(landmark), "config": ft.config, "endpoint": TRAINING_ENDPOINT,
            "time_unit_days": TIME_UNIT_DAYS,
            "hyperparameters": {"max_depth": int(best["max_depth"]), "min_leaf_frac": float(best["min_leaf_frac"]),
                                "min_leaf": ft.min_leaf, "min_split": 2 * ft.min_leaf,
                                "split_rule": "rpart_exp_poisson", "shrink": 1.0, "min_site_cell": 0},
            "selection": best, "binarizer": state, "root_rate": tree.root_rate, "nodes": nodes, "leaves": leaves,
            "horizon_grid_units": np.asarray(ft.grid).tolist(),
            "preprocessing": {"covariate_cols": list(state["feature_columns"])},
            "missing_policy": "item_false"}


def refit_bundle(result: dict, cfg: TreeConfig, label="dfci_local") -> dict:
    """The full-cohort refits as a ``caia-federated-model-bundle`` (scorable by ``federated_inference``)."""
    return {"format": "caia-federated-model-bundle", "format_version": 1, "model_family": MODEL_FAMILY,
            "model_family_version": 1, "training_sites": ["DFCI"], "analysis_label": label,
            "endpoint": TRAINING_ENDPOINT, "local_refit": True, "run_config": cfg.to_dict(),
            "models": [bundle_model(ft, lm) for (lm, _), ft in sorted(result["refits"].items())]}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(local: pd.DataFrame, lab_mapping: dict, cfg: TreeConfig | None = None, verbose=True) -> dict:
    cfg = cfg or TreeConfig()
    frames, prep = build_frames(local, lab_mapping, cfg, verbose=verbose)
    result = cross_fit(frames, cfg, verbose=verbose)
    metrics, auc_t = evaluate(result, cfg, verbose=verbose)
    result.update(frames=frames, prep_stats=prep, metrics=metrics, auc_t=auc_t,
                  bundle=refit_bundle(result, cfg), config=cfg)
    return result


def write_outputs(result: dict, out_dir) -> Path:
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, key in ((METRICS_FILE, "metrics"), (AUC_FILE, "auc_t"), (LEAVES_FILE, "leaves"),
                      (SELECTION_FILE, "selection"), (CV_FILE, "inner_cv"), (PATIENTS_FILE, "patients")):
        result[key].to_csv(out / name, index=False)
    with open(out / BUNDLE_FILE, "w") as handle:
        json.dump(result["bundle"], handle, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[tree] wrote local survival tree outputs -> {out}")
    return out
