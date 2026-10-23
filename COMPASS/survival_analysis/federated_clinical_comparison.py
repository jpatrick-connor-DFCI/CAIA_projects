"""Transferred federated labs score vs local Gleason and TP53/RB1/PTEN Cox models.

Federated Figure 5 b-d. 04_federated_transfer.ipynb scores the frozen federated
platinum bundles on every local DFCI patient (panel a, the full ADT cohort).
This module restricts those scores to three clinical subpopulations at each
landmark and asks whether the transferred multivariable labs model ranks
patients better than a model built from what a clinician already has, and
whether it adds to that model:

* ``gleason`` -- patients with a Gleason score available before the landmark,
  compared with an unpenalized Cox of GLEASON_SCORE (continuous) + age.
* ``trio`` -- sequenced patients, on a clock anchored on the sequencing date
  (sequencing_anchored_local), compared with an unpenalized Cox of TP53_ANY +
  RB1_ANY + PTEN_ANY + age at sequencing. Each indicator is any alteration
  (SNV, DEL, AMP or SV) on the specimen whose sequencing report date is
  closest to ADT start (the one the ``sequencing_any`` screen uses).
* ``trio_gleason`` -- sequenced patients who also have a Gleason score
  available before sequencing + landmark (sequencing clock), compared with an
  unpenalized Cox of GLEASON_SCORE + the trio + age at sequencing.

In each subpopulation a third, combined Cox adds the bundle's transferred
labs score (z-scored within the subpopulation) to the clinical covariates
other than age: Gleason, the trio, or Gleason + the trio. The transferred
full model already takes age as an input, so age is not added again.

The transferred score is never refit. The clinical and combined Cox models
are fit on the same DFCI patients they are scored on, so each patient's score
comes from a stratified K-fold model that excluded them (out-of-fold); the
apparent in-sample C and mean AUC(t) are kept alongside for reference only.
All scores share the transfer frame's duration/event (landmark clock,
3650-day censoring). Mean AUC(t) is the IPCW cumulative/dynamic AUC averaged
over the horizons estimable in the scored subpopulation (censoring estimated
on it); the C-index and mean AUC(t) differences are bootstrapped over
patients with the scores resampled jointly.

Clinical data obey the transfer's strict pre-landmark rule
(``federated_scoring.filter_pre_landmark`` keeps days < landmark): a Gleason
score counts only if it was available before TREATMENT_ANCHOR_DATE + landmark.
No somatic data is used on the ADT clock. For the somatic arms the anchor is
the sequencing report date, so the transferred
labs score is recomputed from labs before sequencing + landmark, outcomes run
from sequencing + landmark, Gleason must be available before sequencing +
landmark, and the anchoring specimen defines membership at every landmark.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SURVIVAL_DIR = Path(__file__).resolve().parent
PREPROCESSING_DIR = SURVIVAL_DIR.parent / "data_preprocessing"
REPO_ROOT = SURVIVAL_DIR.parent.parent
for _p in (str(REPO_ROOT), str(SURVIVAL_DIR), str(PREPROCESSING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_somatic_gleason_inputs as sg  # noqa: E402
from risk_score_stratified_figures import GLEASON_BINS, _gleason_group  # noqa: E402
from survival_common import federated_endpoint_transfer as fet  # noqa: E402
from survival_common import federated_scoring as fs  # noqa: E402

LOCAL_ID_COL = fet.LOCAL_ID_COL
ID_COL = fs.ID_COL
DURATION_COL = fet.DURATION_COL
EVENT_COL = fet.EVENT_COL
AGE_COL = "AGE"
GLEASON_FEATURE = sg.GLEASON_FEATURE
BURDEN_COL = sg.SOMATIC_BURDEN_COVARIATE
HAS_SEQUENCING = "HAS_SEQUENCING"
TRIO_GENES = sg.CLINICAL_STRATIFIERS_TRIO_GENES
TRIO_FEATURES = tuple(f"{gene}_{sg.ANY_ALTERATION_SUFFIX}" for gene in TRIO_GENES)

# Subpopulation -> (display label, membership columns (all non-missing), clinical Cox
# covariates, covariates the combined Cox adds to the transferred score). The combined
# Cox leaves out age: the transferred full model already includes it.
SUBPOPULATIONS = {
    "gleason": {
        "label": "Gleason + age",
        "member": (GLEASON_FEATURE,),
        "covariates": (GLEASON_FEATURE, AGE_COL),
        "combined": (GLEASON_FEATURE,),
    },
    "trio": {
        "label": "TP53/RB1/PTEN + age",
        "member": (HAS_SEQUENCING,),
        "covariates": (*TRIO_FEATURES, AGE_COL),
        "combined": TRIO_FEATURES,
    },
    "trio_gleason": {
        "label": "Gleason + TP53/RB1/PTEN + age",
        "member": (HAS_SEQUENCING, GLEASON_FEATURE),
        "covariates": (GLEASON_FEATURE, *TRIO_FEATURES, AGE_COL),
        "combined": (GLEASON_FEATURE, *TRIO_FEATURES),
    },
}
TRANSFERRED_LABS = "transferred_labs"
TRANSFERRED_AGE = "transferred_age"
CLINICAL_COX = "clinical_cox"
COMBINED_COX = "combined_cox"
MODELS = (TRANSFERRED_LABS, TRANSFERRED_AGE, CLINICAL_COX, COMBINED_COX)
# Bundle config holding each transferred score.
CONFIG_FOR_MODEL = {TRANSFERRED_LABS: "both", TRANSFERRED_AGE: "baseline"}
# Combined Cox covariate: the bundle's transferred labs score, z-scored in the subpopulation.
SCORE_COVARIATE = "TRANSFERRED_SCORE"

COMPARISON_COLUMNS = [
    "endpoint", "landmark_days", "subpopulation", "subpopulation_label", "bundle",
    "model_family", "model", "n_patients", "n_events", "n_scored", "n_scored_events",
    "c_index", "c_index_lo", "c_index_hi", "delta_c_vs_transferred", "delta_lo",
    "delta_hi", "delta_c_vs_clinical", "delta_c_vs_clinical_lo", "delta_c_vs_clinical_hi",
    "c_index_apparent", "mean_auc", "mean_auc_lo", "mean_auc_hi",
    "delta_auc_vs_transferred", "delta_auc_lo", "delta_auc_hi", "delta_auc_vs_clinical",
    "delta_auc_vs_clinical_lo", "delta_auc_vs_clinical_hi", "mean_auc_apparent",
    "auc_horizons_days", "n_folds", "n_failed_folds", "n_convergence_warnings", "note",
]


# --------------------------------------------------------------------------- #
# Clinical features per landmark
# --------------------------------------------------------------------------- #
def load_clinical_sources(gleason_path, somatic_path, somatic_manifest_path):
    """Gleason timeline plus the any-alteration somatic matrix (every panel gene)."""
    gleason = sg.load_gleason(Path(gleason_path))
    somatic, features = sg.load_somatic_features(
        Path(somatic_path), Path(somatic_manifest_path), variant_mode="any"
    )
    for frame, cols in ((gleason, ["gleason_date", sg.GLEASON_AVAILABLE_DATE]),
                        (somatic, [sg.SEQUENCING_DATE, sg.SOMATIC_AVAILABLE_DATE])):
        for col in cols:
            frame[col] = sg._drop_tz(frame[col])
    return gleason, somatic, features


def _patient_anchors(local: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: ADT anchor date and age at the anchor."""
    pt = local[[LOCAL_ID_COL, "TREATMENT_ANCHOR_DATE", "AGE_AT_TREATMENTSTART"]].copy()
    pt[LOCAL_ID_COL] = pd.to_numeric(pt[LOCAL_ID_COL], errors="coerce")
    pt["TREATMENT_ANCHOR_DATE"] = sg._drop_tz(pd.to_datetime(
        pt["TREATMENT_ANCHOR_DATE"], errors="coerce", format="mixed"))
    pt = pt.dropna(subset=[LOCAL_ID_COL, "TREATMENT_ANCHOR_DATE"])
    pt[LOCAL_ID_COL] = pt[LOCAL_ID_COL].astype(int)
    pt = pt.groupby(LOCAL_ID_COL, sort=True).agg(
        anchor=("TREATMENT_ANCHOR_DATE", "first"),
        age=("AGE_AT_TREATMENTSTART", "first"),
    )
    return pt


def select_specimen(
    local: pd.DataFrame, somatic: pd.DataFrame, somatic_features: list[str]
) -> pd.DataFrame:
    """Each local patient's specimen reported closest to ADT start (sg.select_sequencing_specimen)."""
    anchors = _patient_anchors(local)["anchor"].rename_axis(sg.ca.ID_COL)
    return sg.select_sequencing_specimen(somatic, list(somatic_features), anchors)


def sequencing_anchored_local(local: pd.DataFrame, specimen: pd.DataFrame) -> pd.DataFrame:
    """Re-anchor the local long frame on each sequenced patient's sequencing date.

    Keeps patients with a specimen, sets TREATMENT_ANCHOR_DATE to SEQUENCING_DATE
    and moves AGE_AT_TREATMENTSTART to age at sequencing. The transfer derives
    every lab offset and outcome duration from the anchor date
    (fs.derive_analysis_columns), so passing this to fet.evaluate_transfer
    scores the frozen bundles on labs before sequencing + landmark with
    outcomes from sequencing + landmark.

    Score it with exclusion="none": a pre-anchor exclusion would now be judged
    against the sequencing date. Apply any cohort exclusion to ``local`` first.
    """
    sequencing = specimen[sg.SEQUENCING_DATE]
    ids = pd.to_numeric(local[LOCAL_ID_COL], errors="coerce")
    keep = ids.isin(sequencing.index)
    out = local.loc[keep].copy()
    adt = sg._drop_tz(pd.to_datetime(out["TREATMENT_ANCHOR_DATE"], errors="coerce", format="mixed"))
    new_anchor = pd.to_datetime(ids.loc[keep].map(sequencing))
    out["TREATMENT_ANCHOR_DATE"] = new_anchor
    out["AGE_AT_TREATMENTSTART"] = (
        pd.to_numeric(out["AGE_AT_TREATMENTSTART"], errors="coerce")
        + (new_anchor - adt).dt.days / fs.DAYS_PER_YEAR
    )
    return out


def build_clinical_features(
    local: pd.DataFrame,
    landmarks,
    *,
    gleason: pd.DataFrame,
    specimen: pd.DataFrame | None = None,
    somatic_features: list[str] = (),
) -> pd.DataFrame:
    """Per patient × landmark Gleason and trio features, available before the landmark.

    Gleason is the score closest to the landmark among those available by then
    (sg.closest_available_by_landmark).

    Somatic calls come only from ``specimen`` (select_specimen), with a
    sequencing-anchored ``local`` (sequencing_anchored_local): the specimen
    defines time 0, so it is available by construction at every landmark. A
    sequenced patient's uncalled genes are wild-type (0), as in
    sg.build_indexed_feature_sets. Without ``specimen`` (the ADT clock) the
    trio, burden and HAS_SEQUENCING columns are left missing: no somatic data
    is used on the ADT clock.
    """
    if specimen is not None:
        missing_trio = [f for f in TRIO_FEATURES if f not in somatic_features]
        if missing_trio:
            raise ValueError(f"Somatic matrix has no {missing_trio}; cannot build the trio model.")
        s = specimen[list(somatic_features)].apply(pd.to_numeric, errors="coerce").fillna(0)
        s[BURDEN_COL] = s.sum(axis=1).astype(int)
        s[HAS_SEQUENCING] = True
    else:
        s = pd.DataFrame(index=pd.Index([], name=sg.ca.ID_COL),
                         columns=[*TRIO_FEATURES, BURDEN_COL, HAS_SEQUENCING], dtype=float)
    pt = _patient_anchors(local)
    frames = []
    for lm in sorted({int(x) for x in landmarks}):
        # Dates are day-resolution, so "<= landmark - 1 day" is "before the landmark".
        cutoff = pt["anchor"] + pd.to_timedelta(int(lm) - 1, unit="D")
        cutoffs = pd.DataFrame({sg.ca.ID_COL: pt.index, "_landmark_date": cutoff.values})
        g = sg.closest_available_by_landmark(
            gleason, cutoffs, date_col="gleason_date",
            available_date_col=sg.GLEASON_AVAILABLE_DATE, value_cols=[GLEASON_FEATURE],
        ).dropna(subset=[GLEASON_FEATURE])
        out = pd.DataFrame({AGE_COL: pt["age"]}, index=pt.index.rename(sg.ca.ID_COL))
        out = out.join(g[[GLEASON_FEATURE]], how="left")
        # HAS_SEQUENCING stays True / NaN so membership is a notna() test like Gleason.
        out = out.join(s[[*TRIO_FEATURES, BURDEN_COL, HAS_SEQUENCING]], how="left")
        frames.append(out.reset_index().rename(columns={sg.ca.ID_COL: ID_COL})
                      .assign(landmark_days=int(lm)))
    cols = [ID_COL, "landmark_days", AGE_COL, GLEASON_FEATURE, HAS_SEQUENCING,
            *TRIO_FEATURES, BURDEN_COL]
    return pd.concat(frames, ignore_index=True)[cols]


# --------------------------------------------------------------------------- #
# Local unpenalized Cox
# --------------------------------------------------------------------------- #
def _fit_cox(frame: pd.DataFrame, covariates: list[str]):
    """Unpenalized Cox; returns (fitter, n convergence warnings)."""
    from lifelines import CoxPHFitter

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cph = CoxPHFitter(penalizer=0.0).fit(
            frame[[*covariates, DURATION_COL, EVENT_COL]], DURATION_COL, EVENT_COL
        )
    n_warn = sum("onvergence" in type(w.message).__name__ for w in caught)
    return cph, n_warn


def _linear_predictor(cph, frame: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    # x'beta, uncentered: lifelines' log partial hazard centers on each fold's
    # own training means, which would shift folds against each other.
    return frame[covariates].to_numpy(dtype=float) @ cph.params_.loc[covariates].to_numpy()


def out_of_fold_cox(frame: pd.DataFrame, covariates, *, n_folds: int = 5, seed: int = 0):
    """Stratified K-fold out-of-fold Cox linear predictor.

    Returns (predictions, n_folds used, n_failed_folds, n_convergence_warnings).
    A fold whose training split has a constant covariate or fails to fit leaves
    its patients NaN rather than borrowing another fold's model.
    """
    from sklearn.model_selection import StratifiedKFold

    covariates = list(covariates)
    event = frame[EVENT_COL].to_numpy(dtype=int)
    pred = np.full(len(frame), np.nan)
    k = int(min(n_folds, event.sum(), (1 - event).sum()))
    if k < 2:
        return pred, k, 0, 0
    failed = n_warn = 0
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for train, test in folds.split(np.zeros(len(frame)), event):
        tr = frame.iloc[train]
        if (tr[covariates].nunique() < 2).any() or tr[EVENT_COL].sum() < 1:
            failed += 1
            continue
        try:
            cph, w = _fit_cox(tr, covariates)
        except Exception:
            failed += 1
            continue
        n_warn += w
        pred[test] = _linear_predictor(cph, frame.iloc[test], covariates)
    return pred, k, failed, n_warn


def _coef_rows(cph, base: dict) -> list[dict]:
    s = cph.summary
    return [{**base, "covariate": cov,
             "coef": float(s.loc[cov, "coef"]), "hr": float(s.loc[cov, "exp(coef)"]),
             "hr_lo": float(s.loc[cov, "exp(coef) lower 95%"]),
             "hr_hi": float(s.loc[cov, "exp(coef) upper 95%"]),
             "p": float(s.loc[cov, "p"])} for cov in s.index]


def fit_local_cox(frame: pd.DataFrame, covariates, *, n_folds: int, seed: int, horizons_days) -> dict:
    """Out-of-fold Cox score plus the apparent (in-sample) fit, for reference only.

    Returns {oof, apparent_c, apparent_auc, cph (None if the full fit failed),
    fit_info, note}.
    """
    covariates = list(covariates)
    oof, k, failed, n_warn = out_of_fold_cox(frame, covariates, n_folds=n_folds, seed=seed)
    out = {"oof": oof, "apparent_c": np.nan, "apparent_auc": np.nan, "cph": None, "note": ""}
    try:
        cph, w = _fit_cox(frame, covariates)
        n_warn += w
        lp = _linear_predictor(cph, frame, covariates)
        dur, ev = frame[DURATION_COL], frame[EVENT_COL]
        out["apparent_c"] = fet.harrell_c(dur, ev, lp)
        out["apparent_auc"] = fet.mean_auc(dur, ev, lp, fet.estimable_horizons(dur, ev, horizons_days))
        out["cph"] = cph
    except Exception as exc:
        out["note"] = f"apparent fit failed: {exc}"
    out["fit_info"] = {"n_folds": k, "n_failed_folds": failed, "n_convergence_warnings": n_warn}
    return out


def fit_combined_cox(sub_df: pd.DataFrame, labs: np.ndarray, covariates, **kwargs) -> dict:
    """fit_local_cox on the transferred labs score (z-scored) + ``covariates``.

    Fit on the patients with a finite score; the others stay NaN. z-scoring
    uses no outcome, so it leaks nothing into the out-of-fold score; the
    score's HR reads per SD within the subpopulation.
    """
    has = np.isfinite(labs)
    sd = labs[has].std() if has.any() else np.nan
    if not sd > 0:
        return {"oof": np.full(len(sub_df), np.nan), "apparent_c": np.nan, "apparent_auc": np.nan,
                "cph": None, "note": "transferred score constant or missing: combined Cox not fitted",
                "fit_info": {"n_folds": 0, "n_failed_folds": 0, "n_convergence_warnings": 0}}
    frame = sub_df.loc[has].assign(**{SCORE_COVARIATE: (labs[has] - labs[has].mean()) / sd})
    fit = fit_local_cox(frame.reset_index(drop=True), [SCORE_COVARIATE, *covariates], **kwargs)
    oof = np.full(len(sub_df), np.nan)
    oof[has] = fit["oof"]
    return {**fit, "oof": oof}


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[ID_COL] = pd.to_numeric(out[ID_COL], errors="coerce")
    out = out.dropna(subset=[ID_COL])
    out[ID_COL] = out[ID_COL].astype(int)
    return out


def _endpoint_landmark_frame(scores: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    """One row per patient: outcome plus one risk column per (bundle, config)."""
    outcome = scores.drop_duplicates(ID_COL)[[ID_COL, DURATION_COL, EVENT_COL]]
    risks = scores.pivot_table(index=ID_COL, columns=["bundle", "config"],
                               values="risk_score", aggfunc="first")
    keys = list(risks.columns)
    risks.columns = [f"{b}|{c}" for b, c in keys]
    return outcome.set_index(ID_COL).join(risks).reset_index(), keys


def compare_clinical_models(
    scores: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    n_folds: int = 5,
    n_bootstrap: int = 200,
    seed: int = 0,
    min_events: int = 5,
    horizons_days=fet.DEFAULT_HORIZONS_DAYS,
    subpopulations: dict = SUBPOPULATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transferred labs/age scores vs the out-of-fold clinical and combined Cox, per subpopulation.

    Returns (comparison, coefficients). ``comparison`` has one row per
    endpoint × landmark × subpopulation × bundle × model. The clinical Cox
    does not depend on the bundle; its rows repeat for each bundle so that
    each ΔC and ΔAUC(t) (model − transferred labs, paired bootstrap over the
    same patients) sits on its own row. The combined Cox (that bundle's
    transferred labs score + the subpopulation's ``combined`` covariates, the
    clinical covariates without age) also carries its paired difference from
    the clinical Cox (``delta_*_vs_clinical``).
    """
    scores = _normalize_ids(scores)
    clinical = _normalize_ids(clinical)
    families = scores.drop_duplicates("bundle").set_index("bundle")["model_family"].to_dict()
    rows, coefs = [], []
    for (endpoint, lm), grp in scores.groupby(["endpoint", "landmark_days"], sort=True):
        frame, keys = _endpoint_landmark_frame(grp)
        frame = frame.merge(clinical[clinical["landmark_days"] == lm].drop(columns="landmark_days"),
                            on=ID_COL, how="left")
        bundles = sorted({b for b, _ in keys})
        for sub, spec in subpopulations.items():
            covariates = list(spec["covariates"])
            members = frame[frame[list(spec["member"])].notna().all(axis=1)]
            sub_df = members.dropna(subset=covariates).reset_index(drop=True)
            base = {"endpoint": endpoint, "landmark_days": int(lm), "subpopulation": sub,
                    "subpopulation_label": spec["label"], "n_patients": len(sub_df),
                    "n_events": int(sub_df[EVENT_COL].sum())}
            coef_base = {k_: base[k_] for k_ in
                         ("endpoint", "landmark_days", "subpopulation", "n_patients", "n_events")}
            note = ""
            if len(members) > len(sub_df):
                note = f"{len(members) - len(sub_df)} member(s) missing a covariate dropped; "
            fit_kw = dict(n_folds=n_folds, seed=seed, horizons_days=horizons_days)
            fitted = base["n_events"] >= min_events
            unfitted = {"oof": np.full(len(sub_df), np.nan), "apparent_c": np.nan, "apparent_auc": np.nan,
                        "cph": None, "note": "",
                        "fit_info": {"n_folds": 0, "n_failed_folds": 0, "n_convergence_warnings": 0}}
            if fitted:
                clin = fit_local_cox(sub_df, covariates, **fit_kw)
                note += clin["note"]
                if clin["cph"] is not None:
                    coefs += _coef_rows(clin["cph"], {**coef_base, "model": CLINICAL_COX, "bundle": None})
            else:
                note += f"fewer than {min_events} events: not fitted"
                clin = unfitted
            for bundle in bundles:
                risks = {m: sub_df.get(f"{bundle}|{CONFIG_FOR_MODEL[m]}", pd.Series(np.nan, index=sub_df.index))
                         .to_numpy(dtype=float) for m in CONFIG_FOR_MODEL}
                comb = (fit_combined_cox(sub_df, risks[TRANSFERRED_LABS], spec["combined"], **fit_kw)
                        if fitted else unfitted)
                if comb["cph"] is not None:
                    coefs += _coef_rows(comb["cph"], {**coef_base, "model": COMBINED_COX, "bundle": bundle})
                risks[CLINICAL_COX] = clin["oof"]
                risks[COMBINED_COX] = comb["oof"]
                fits = {CLINICAL_COX: clin, COMBINED_COX: comb}
                ok = np.all([np.isfinite(r) for r in risks.values()], axis=0)
                dur = sub_df[DURATION_COL].to_numpy(dtype=float)[ok]
                ev = sub_df[EVENT_COL].to_numpy(dtype=int)[ok]
                risks_ok = {m: r[ok] for m, r in risks.items()}
                boot = (fet.bootstrap_c(dur, ev, risks_ok, n_bootstrap, seed)
                        if n_bootstrap > 0 and ok.any() else {})
                c = {m: fet.harrell_c(dur, ev, r) if ok.any() else np.nan for m, r in risks_ok.items()}
                horizons = fet.estimable_horizons(dur, ev, horizons_days)
                auc = {m: fet.mean_auc(dur, ev, r, horizons) for m, r in risks_ok.items()}
                boot_auc = (fet.bootstrap_mean_auc(dur, ev, risks_ok, horizons, n_bootstrap, seed)
                            if n_bootstrap > 0 and horizons else {})
                for model in MODELS:
                    # The transferred scores report the clinical Cox fit, as the subpopulation's fit.
                    fit = fits.get(model, clin)
                    row = {**base, "bundle": bundle, "model_family": families.get(bundle),
                           "model": model, "n_scored": int(ok.sum()),
                           "n_scored_events": int(ev.sum()), "c_index": c[model],
                           "c_index_lo": np.nan, "c_index_hi": np.nan,
                           "delta_c_vs_transferred": np.nan, "delta_lo": np.nan, "delta_hi": np.nan,
                           "delta_c_vs_clinical": np.nan, "delta_c_vs_clinical_lo": np.nan,
                           "delta_c_vs_clinical_hi": np.nan,
                           "c_index_apparent": fits[model]["apparent_c"] if model in fits else np.nan,
                           "mean_auc": auc[model], "mean_auc_lo": np.nan, "mean_auc_hi": np.nan,
                           "delta_auc_vs_transferred": np.nan, "delta_auc_lo": np.nan,
                           "delta_auc_hi": np.nan, "delta_auc_vs_clinical": np.nan,
                           "delta_auc_vs_clinical_lo": np.nan, "delta_auc_vs_clinical_hi": np.nan,
                           "mean_auc_apparent": fits[model]["apparent_auc"] if model in fits else np.nan,
                           "auc_horizons_days": ",".join(str(int(h)) for h in horizons),
                           **fit["fit_info"],
                           "note": note + (comb["note"] if model == COMBINED_COX else "")}
                    if boot:
                        row["c_index_lo"], row["c_index_hi"] = fet._ci(boot[model])
                    if model != TRANSFERRED_LABS:
                        row["delta_c_vs_transferred"] = c[model] - c[TRANSFERRED_LABS]
                        if boot:
                            row["delta_lo"], row["delta_hi"] = fet._ci(boot[model] - boot[TRANSFERRED_LABS])
                    if boot_auc:
                        row["mean_auc_lo"], row["mean_auc_hi"] = fet._ci(boot_auc[model])
                    if model != TRANSFERRED_LABS:
                        row["delta_auc_vs_transferred"] = auc[model] - auc[TRANSFERRED_LABS]
                        if boot_auc:
                            row["delta_auc_lo"], row["delta_auc_hi"] = fet._ci(
                                boot_auc[model] - boot_auc[TRANSFERRED_LABS])
                    if model == COMBINED_COX:
                        row["delta_c_vs_clinical"] = c[model] - c[CLINICAL_COX]
                        row["delta_auc_vs_clinical"] = auc[model] - auc[CLINICAL_COX]
                        if boot:
                            row["delta_c_vs_clinical_lo"], row["delta_c_vs_clinical_hi"] = fet._ci(
                                boot[model] - boot[CLINICAL_COX])
                        if boot_auc:
                            row["delta_auc_vs_clinical_lo"], row["delta_auc_vs_clinical_hi"] = fet._ci(
                                boot_auc[model] - boot_auc[CLINICAL_COX])
                    rows.append(row)
    return (pd.DataFrame(rows, columns=COMPARISON_COLUMNS),
            pd.DataFrame(coefs))


def compare_by_subpopulation(
    inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]], **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """compare_clinical_models per subpopulation on its own (scores, clinical).

    The Gleason arm is scored on the ADT clock and the trio and trio + Gleason
    arms on the sequencing clock, so each subpopulation brings its own transfer
    scores and clinical features. Returns the concatenated (comparison,
    coefficients).
    """
    comparisons, coefs = [], []
    for sub, (scores, clinical) in inputs.items():
        comp, coef = compare_clinical_models(
            scores, clinical, subpopulations={sub: SUBPOPULATIONS[sub]}, **kwargs
        )
        comparisons.append(comp)
        coefs.append(coef)
    return (pd.concat(comparisons, ignore_index=True),
            pd.concat(coefs, ignore_index=True))


# --------------------------------------------------------------------------- #
# Risk scores inside the subpopulations
# --------------------------------------------------------------------------- #
def _describe(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return {"n": 0}
    q = v.quantile([.25, .5, .75])
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()),
            "q25": float(q[.25]), "median": float(q[.5]), "q75": float(q[.75])}


SUMMARY_SECTIONS = ("cohort", "gleason", "sequencing")


def subpopulation_score_summary(scores: pd.DataFrame, clinical: pd.DataFrame,
                                config: str = "both",
                                sections=SUMMARY_SECTIONS) -> pd.DataFrame:
    """Transferred score distribution by clinical stratum, per bundle × landmark.

    A patient's score does not depend on the endpoint (one feature frame), so
    patients are pooled across endpoints. Spearman rho of the score with
    Gleason (on 'Gleason available') and with the burden (on 'Sequenced').
    ``sections`` picks the strata: "cohort" (full cohort), "gleason" (Gleason
    available/missing and groups) and "sequencing" (sequenced and trio
    strata), so ADT- and sequencing-anchored scores can each be summarized on
    their own strata.
    """
    scores = _normalize_ids(scores)
    clinical = _normalize_ids(clinical)
    sc = (scores[scores["config"] == config]
          .drop_duplicates(["bundle", "landmark_days", ID_COL])
          .merge(clinical, on=[ID_COL, "landmark_days"], how="left"))
    sc["gleason_group"] = _gleason_group(sc)
    seq = sc[HAS_SEQUENCING].eq(True)
    any_trio = sc[list(TRIO_FEATURES)].eq(1).any(axis=1)
    rows = []
    for (bundle, lm), d in sc.groupby(["bundle", "landmark_days"], sort=True):
        r = d["risk_score"]
        g_ok = d[GLEASON_FEATURE].notna()
        s_ok = seq.loc[d.index]
        strata = []
        if "cohort" in sections:
            strata += [("Full cohort", d.index == d.index, None)]
        if "gleason" in sections:
            strata += [("Gleason available", g_ok, GLEASON_FEATURE),
                       ("Gleason missing", ~g_ok, None)]
            strata += [(label, d["gleason_group"].eq(label), None) for _, _, label in GLEASON_BINS]
        if "sequencing" in sections:
            strata += [("Sequenced", s_ok, BURDEN_COL), ("Not sequenced", ~s_ok, None),
                       ("Sequenced, Gleason available", s_ok & g_ok, GLEASON_FEATURE),
                       ("Any trio altered", s_ok & any_trio.loc[d.index], None),
                       ("No trio alteration", s_ok & ~any_trio.loc[d.index], None)]
            strata += [(f"{g} altered", s_ok & d[f].eq(1), None)
                       for g, f in zip(TRIO_GENES, TRIO_FEATURES)]
        for label, mask, rho_with in strata:
            mask = np.asarray(mask, dtype=bool)
            row = {"bundle": bundle, "model_family": d["model_family"].iloc[0],
                   "landmark_days": int(lm), "config": config, "stratum": label,
                   **_describe(r[mask])}
            if rho_with is not None and mask.sum() > 2:
                row["spearman_with"] = rho_with
                row["spearman_rho"] = float(pd.Series(r[mask].to_numpy()).corr(
                    pd.Series(d.loc[mask, rho_with].to_numpy(dtype=float)), method="spearman"))
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Notebook helpers
# --------------------------------------------------------------------------- #
KM_PATIENTS_FILE = "transfer_km_patients.csv"
KM_PATIENT_COLUMNS = ["clock", "endpoint", "bundle", "model_family", "landmark_days", ID_COL,
                      "risk_score", DURATION_COL, EVENT_COL, GLEASON_FEATURE, HAS_SEQUENCING,
                      *TRIO_FEATURES, BURDEN_COL]


def km_patients(clocks: dict, config: str = "both") -> pd.DataFrame:
    """Patient-level scores joined to clinical strata, for the R KM figures.

    ``clocks`` maps a clock name ("adt", "sequencing") to (scores, clinical)
    on that clock. Only ``config`` scores (the labs score) are kept. The R
    side assigns score groups, so cutpoints and figure styling stay in one
    place. ADT-clock rows carry no somatic columns (build_clinical_features).
    """
    frames = []
    for clock, (scores, clinical) in clocks.items():
        sc = _normalize_ids(scores)
        sc = sc[sc["config"] == config]
        clin = _normalize_ids(clinical).drop(columns=AGE_COL)
        frames.append(sc.merge(clin, on=[ID_COL, "landmark_days"], how="left").assign(clock=clock))
    return pd.concat(frames, ignore_index=True)[KM_PATIENT_COLUMNS]


def write_outputs(comparison, coefs, summary, out_dir, km=None) -> None:
    out_dir = Path(out_dir)
    comparison.to_csv(out_dir / "transfer_clinical_comparison.csv", index=False)
    coefs.to_csv(out_dir / "transfer_clinical_cox_coefs.csv", index=False)
    summary.to_csv(out_dir / "transfer_subpop_risk_summary.csv", index=False)
    if km is not None:
        km.to_csv(out_dir / KM_PATIENTS_FILE, index=False)
    print(f"wrote clinical comparison outputs -> {out_dir}")


def plot_clinical_comparison(comparison: pd.DataFrame, path=None, endpoints=fet.DEFAULT_ENDPOINTS):
    """Quick check: mean AUC(t) by landmark, rows = subpopulation, columns = endpoint."""
    import matplotlib.pyplot as plt

    d = comparison[comparison["model"] != TRANSFERRED_AGE].dropna(subset=["mean_auc"])
    subs = [s for s in SUBPOPULATIONS if s in set(d["subpopulation"])]
    eps = [e for e in endpoints if e in set(d["endpoint"])]
    if not subs or not eps:
        print("no estimable clinical comparison rows")
        return
    lms = sorted(d["landmark_days"].unique())
    bundles = sorted(d["bundle"].unique())
    series = ([(b, TRANSFERRED_LABS) for b in bundles] + [(None, CLINICAL_COX)]
              + [(b, COMBINED_COX) for b in bundles])
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e44ad"]
    width = 0.8 / len(series)
    fig, axes = plt.subplots(len(subs), len(eps), figsize=(4.2 * len(eps), 3.2 * len(subs)),
                             sharey=True, squeeze=False)
    for i, sub in enumerate(subs):
        for j, ep in enumerate(eps):
            ax = axes[i][j]
            dd = d[(d["subpopulation"] == sub) & (d["endpoint"] == ep)]
            for k, (bundle, model) in enumerate(series):
                r = dd[dd["model"] == model]
                r = (r[r["bundle"] == bundle] if bundle else r.drop_duplicates("landmark_days"))
                r = r.set_index("landmark_days").reindex(lms)
                x = np.arange(len(lms)) + (k - (len(series) - 1) / 2) * width
                label = (SUBPOPULATIONS[sub]["label"] if bundle is None
                         else f"{bundle} + clinical" if model == COMBINED_COX else bundle)
                ax.errorbar(x, r["mean_auc"], yerr=np.vstack([r["mean_auc"] - r["mean_auc_lo"],
                                                             r["mean_auc_hi"] - r["mean_auc"]]),
                            fmt={TRANSFERRED_LABS: "o", CLINICAL_COX: "D", COMBINED_COX: "^"}[model],
                            ms=6, capsize=0, color=colors[k % len(colors)], label=label)
            ax.axhline(0.5, color="#8a8a86", lw=1, ls="--", zorder=0)
            ax.set_xticks(range(len(lms)), [f"{lm}d" for lm in lms])
            n = dd.groupby("landmark_days")[["n_patients", "n_events"]].first().reindex(lms)
            ax.set_title(f"{ep.upper()} · {sub}  (n/events: "
                         + ", ".join(f"{int(a)}/{int(b)}" for a, b in n.fillna(0).to_numpy()) + ")",
                         fontsize=9)
            ax.grid(axis="y", color="#e6e5df", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[i][0].set_ylabel("Mean AUC(t)")
        axes[i][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
