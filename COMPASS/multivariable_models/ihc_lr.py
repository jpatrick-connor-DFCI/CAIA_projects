"""Logistic regression of future prostate IHC (NE-positive, NEPC-like, AR-low) on the federated risk score.

Backs ``notebooks/text_interpretability/post_landmark/11_LR_for_IHC.ipynb``. For each ADT landmark, the frozen
federated (no-DFCI) platinum-trained XGBoost score from notebook 04 (``endpoint == "nepc"`` rows: patients
with NEPC or censoring at/before the landmark are not at risk) is the only predictor, z-scored within each
fit, in four unpenalized LRs:

* ``ihc_any``   -- a future prostate-relevant specimen with >= 1 lesional stain row; all at-risk patients.
* ``ne_pos``    -- a future prostate-relevant specimen that is NE-positive (synaptophysin, chromogranin, INSM1
                   or CD56 positive); at-risk patients without a prior NE-positive one.
* ``nepc_like`` -- NE-positive, >= 1 AR-axis marker negative and none positive; same population as ``ne_pos``.
* ``ar_low``    -- AR-axis negative (>= 1 of AR, PSA, NKX3.1, PSAP, PSMA, prostein negative, none positive),
                   whatever the NE stains show; at-risk patients without a prior AR-low specimen.

**IHC source.** ``COMPASS.prostate_ihc.annotations``: specimen flags recomputed from the pre-existing step-1
intermediates (``prostate_ihc_long`` + ``prostate_ihc_specimen_all``), no raw reads. Every stained pathology
specimen with ``prostate_relevant`` (evidence tiers 1-4) can be an event, sequenced or not; other specimens are
ignored. A discordant marker call (positive and negative on the same part) counts as positive for NE markers and
as neither for AR-axis markers.
No LLM findings are used.

**Timing.** A specimen is dated by the earliest lesional ``specimen_date`` in the long table. "Future" =
ADT + landmark < date <= ADT + MAX_FOLLOWUP_DAYS (the models' 10-year cap); the label is "ever during that
window", not fixed-horizon. Prior IHC excludes no one; a prior NE-positive specimen (at or before ADT +
landmark) excludes the patient from both NE outcomes (NEPC-like is a subset of NE-positive); a prior AR-low
specimen excludes the patient from ``ar_low``. Undated specimens are never events (counted).

AUC is from out-of-fold predictions (stratified K-fold) with a patient bootstrap 95% CI, plus the AUC of
the untransformed score (``auc_raw_score``, preferred). Odds ratios (per SD), Wald CIs and LRT p-values
come from the full-population fit. The LR machinery is reused from ``nepc_subtyping_lr``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from COMPASS.multivariable_models import nepc_subtyping_lr as nsl
from COMPASS.prostate_ihc import annotations as ann

ID_COL = nsl.ID_COL                     # "person_id" (scores / LR frame)
LOCAL_ID_COL = nsl.LOCAL_ID_COL         # "DFCI_MRN" (== MRN_INT in the prostate_ihc intermediates)
SOURCE_ID_COL = "MRN_INT"
SPECIMEN_ID_COL = "SPECIMEN_ID"         # MRN|ACCESSION|PART
RISK_COL = nsl.RISK_COL
ANCHOR_SOURCE_COL = "TREATMENT_ANCHOR_DATE"
MAX_FOLLOWUP_DAYS = 3650
IHC_SOURCE_VERSION = ann.SOURCE_VERSION

# Outcome -> (event column, prior-exclusion column or None); the population is set by outcome_population().
OUTCOMES = {
    "ihc_any": "ihc_future",
    "ne_pos": "ne_pos_future",
    "nepc_like": "nepc_like_future",
    "ar_low": "ar_low_future",
}
PRIOR_EXCLUSION = {"ihc_any": None, "ne_pos": "prior_ne_pos", "nepc_like": "prior_ne_pos", "ar_low": "prior_ar_low"}
SPECIMEN_FLAGS = {"ihc_future": "ihc_any", "ne_pos_future": "ne_pos", "nepc_like_future": "nepc_like",
                  "ar_low_future": "ar_low"}
PRIOR_FLAGS = {"prior_ne_pos": "ne_pos", "prior_ar_low": "ar_low"}

MODEL_COLUMNS = [
    "cohort", "landmark_days", "outcome", "n", "events", "lrt_stat", "lrt_df", "p_lrt",
    "auc", "auc_lo", "auc_hi", "auc_raw_score", "auc_raw_score_lo", "auc_raw_score_hi",
    "converged", "n_failed_folds", "note",
]
COEF_COLUMNS = [
    "cohort", "landmark_days", "outcome", "term", "n", "events",
    "coef", "se", "odds_ratio", "or_lo", "or_hi", "p_wald", "note",
]
COUNT_COLUMNS = [
    "cohort", "landmark_days", "n_cohort_list", "n_at_risk",
    "n_ihc_future", "n_ne_pos_future", "n_nepc_like_future", "n_ar_low_future",
    "n_prior_ne_pos", "n_prior_ar_low",
    "n_ne_pos", "events_ne_pos", "n_nepc_like", "events_nepc_like", "n_ar_low", "events_ar_low",
    "n_undated_ihc_specimen",
]
AUDIT_COLUMNS = [
    LOCAL_ID_COL, SPECIMEN_ID_COL, "ACCESSION", "PART", "specimen_date", "days_from_adt",
    "prostate_relevant", "prostate_evidence", "ihc_phenotype",
    "ihc_any", "ne_pos", "nepc_like", "ar_low",
]
# Per-patient OOF predictions; ``event`` is that outcome's label (OUTCOMES[outcome]).
PREDICTION_COLUMNS = ["cohort", "landmark_days", "outcome", *nsl.PREDICTION_KEY_COLUMNS]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_ihc_specimens(long=None, spec_all=None, *, intermediate_dir=None) -> pd.DataFrame:
    """One row per stained specimen (``ann.ANNOTATION_COLUMNS`` with ``DFCI_MRN`` for ``MRN_INT``).

    Pass the long / specimen_all frames, or neither to read them from ``intermediate_dir`` (default:
    ``ihc_extraction/intermediate``). Non-relevant specimens are kept here (for audit) and dropped in
    ``build_ihc_frame``.
    """
    if (long is None) != (spec_all is None):
        raise ValueError("Pass both long and spec_all, or neither.")
    if long is None:
        t = ann.load_intermediates(intermediate_dir)
        long, spec_all = t["long"], t["spec"]
    return ann.specimen_annotations(long, spec_all).rename(
        columns={SOURCE_ID_COL: LOCAL_ID_COL})


def load_anchor_dates(path) -> pd.Series:
    """MRN -> ADT anchor date, reading only the two needed columns of the long Stage-2 frame."""
    path = str(path)
    cols = [LOCAL_ID_COL, ANCHOR_SOURCE_COL]
    local = (pd.read_parquet(path, columns=cols) if path.endswith((".parquet", ".pq"))
             else pd.read_csv(path, usecols=cols, low_memory=False))
    return anchor_dates(local)


def anchor_dates(local: pd.DataFrame) -> pd.Series:
    """MRN -> ADT anchor date from the local long frame (one value per patient)."""
    a = local[[LOCAL_ID_COL, ANCHOR_SOURCE_COL]].dropna().drop_duplicates()
    a[ANCHOR_SOURCE_COL] = pd.to_datetime(a[ANCHOR_SOURCE_COL], errors="coerce", format="mixed")
    a = a.dropna().drop_duplicates()
    a[LOCAL_ID_COL] = pd.to_numeric(a[LOCAL_ID_COL], errors="raise").astype(int)
    if a[LOCAL_ID_COL].duplicated().any():
        raise ValueError("More than one TREATMENT_ANCHOR_DATE for some patients.")
    return a.set_index(LOCAL_ID_COL)[ANCHOR_SOURCE_COL]


# --------------------------------------------------------------------------- #
# Patient x landmark frame
# --------------------------------------------------------------------------- #
def validate_anchor_coverage(scores: pd.DataFrame, anchors: pd.Series):
    """Require an ADT date for every scored patient, even those without specimens."""
    missing = scores[ID_COL].drop_duplicates().map(anchors).isna().sum()
    if missing:
        raise ValueError(f"{missing} scored patients lack an ADT anchor; reconcile scores "
                         "and longitudinal data before constructing IHC outcomes.")


def add_days_from_adt(specimens: pd.DataFrame, anchors: pd.Series) -> pd.DataFrame:
    """``days_from_adt``: pathology specimen date minus the patient's ADT anchor."""
    out = specimens.copy()
    out["days_from_adt"] = (out["specimen_date"] - out[LOCAL_ID_COL].map(anchors)).dt.days
    return out


def build_ihc_frame(scores: pd.DataFrame, specimens: pd.DataFrame, landmarks, *,
                    anchors: pd.Series, max_days: int = MAX_FOLLOWUP_DAYS) -> pd.DataFrame:
    """One row per scored patient x landmark with the IHC outcome flags.

    Only ``prostate_relevant`` specimens are used. Flags: ``ihc_future``, ``ne_pos_future``,
    ``nepc_like_future``, ``ar_low_future`` (specimen dated in (ADT + landmark, ADT + max_days]),
    ``prior_ne_pos`` / ``prior_ar_low`` (such a specimen dated at or before ADT + landmark), and
    ``undated_ihc_specimen`` (a relevant stained specimen with no date; never counted).
    """
    validate_anchor_coverage(scores, anchors)
    s = add_days_from_adt(specimens[specimens["prostate_relevant"].astype(bool)], anchors)
    undated = set(s.loc[s["ihc_any"] & s["days_from_adt"].isna(), LOCAL_ID_COL].astype(int))
    rows = []
    for lm in landmarks:
        lm = int(lm)
        sc = scores[scores["landmark_days"].eq(lm)][[ID_COL, "landmark_days", RISK_COL]].copy()
        d = s["days_from_adt"]
        fut = s[(d > lm) & (d <= max_days)]
        prior = s[d <= lm]

        def ids(df):
            return set(df[LOCAL_ID_COL].astype(int))

        flags = {name: ids(fut[fut[col]]) for name, col in SPECIMEN_FLAGS.items()}
        flags.update({name: ids(prior[prior[col]]) for name, col in PRIOR_FLAGS.items()})
        flags["undated_ihc_specimen"] = undated
        for name, members in flags.items():
            sc[name] = sc[ID_COL].isin(members).astype(int)
        rows.append(sc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# LR driver
# --------------------------------------------------------------------------- #
def outcome_population(sub: pd.DataFrame, outcome: str) -> pd.DataFrame:
    if outcome not in PRIOR_EXCLUSION:
        raise ValueError(f"Unknown outcome {outcome!r}")
    prior = PRIOR_EXCLUSION[outcome]
    return sub if prior is None else sub[sub[prior].eq(0)]


def _fit_outcome(pop, outcome, tags, *, min_events, n_folds, n_bootstrap, seed):
    y = pop[OUTCOMES[outcome]].to_numpy(dtype=int)
    n, events = len(y), int(y.sum())
    base = {**tags, "outcome": outcome, "n": n, "events": events}
    if events < min_events or n - events < min_events:
        return ({**base, "note": f"fewer than {min_events} events or non-events; not fit"}, [], None)
    X = nsl._design(pop, nsl.REFERENCE_MODEL)          # RISK_Z, z-scored within this fit
    fit = nsl.fit_logistic(y, X)
    oof, failed = nsl.oof_predictions(y, X, n_folds=n_folds, seed=seed, return_diagnostics=True)
    predictions = nsl.oof_prediction_frame(pop, y, oof, n_folds=n_folds, seed=seed,
                                           tags={**tags, "outcome": outcome})
    note = fit["note"]
    if not np.isfinite(oof).all():
        cv_note = (f"{failed} failed CV folds; AUC suppressed" if failed
                   else "fewer than 2 CV folds; AUC suppressed")
        note = "; ".join(filter(None, [note, cv_note]))
    row = {**base, "lrt_stat": fit["lrt_stat"], "lrt_df": fit["lrt_df"], "p_lrt": fit["p_lrt"],
           "auc": nsl._auc(y, oof), "converged": fit["converged"], "n_failed_folds": failed,
           "note": note}
    if n_bootstrap and np.isfinite(oof).all():
        draws = nsl.bootstrap_auc(y, {outcome: oof}, n_bootstrap=n_bootstrap, seed=seed)
        row["auc_lo"], row["auc_hi"] = nsl._ci(draws[outcome])
    # Untransformed frozen score: no fitting, so it needs no CV and is not suppressed by failed folds.
    raw = pop[nsl.RISK_COL].to_numpy(dtype=float)
    row["auc_raw_score"] = nsl._auc(y, raw)
    if n_bootstrap:
        draws = nsl.bootstrap_auc(y, {outcome: raw}, n_bootstrap=n_bootstrap, seed=seed)
        row["auc_raw_score_lo"], row["auc_raw_score_hi"] = nsl._ci(draws[outcome])
    coefs = [{**base, **r, "note": fit["note"]} for r in fit["terms"].to_dict("records")]
    return row, coefs, predictions


def _counts(sub: pd.DataFrame, tags: dict, n_list) -> dict:
    s = lambda col: int(sub[col].sum())  # noqa: E731
    out = {**tags, "n_cohort_list": n_list, "n_at_risk": len(sub),
           **{f"n_{col}": s(col) for col in (*SPECIMEN_FLAGS, *PRIOR_FLAGS)},
           "n_undated_ihc_specimen": s("undated_ihc_specimen")}
    for outcome, prior in PRIOR_EXCLUSION.items():
        if prior is None:
            continue
        pop = outcome_population(sub, outcome)
        out[f"n_{outcome}"] = len(pop)
        out[f"events_{outcome}"] = int(pop[OUTCOMES[outcome]].sum())
    return out


def run_ihc_lr(frame: pd.DataFrame, members: dict, landmarks, *, outcomes=tuple(OUTCOMES),
               min_events: int = 5, n_folds: int = 5, n_bootstrap: int = 200, seed: int = 0,
               return_predictions: bool = False):
    """Fit ``outcome ~ RISK_Z`` per cohort x landmark x outcome.

    ``members`` maps cohort -> MRN set (None = every scored patient). Rows with a
    missing risk score are dropped. Returns ``(models, coefficients, counts)``;
    with ``return_predictions``, also per-patient out-of-fold predictions
    (``PREDICTION_COLUMNS``) for every fitted outcome population.
    """
    model_rows, coef_rows, count_rows, prediction_frames = [], [], [], []
    for cohort, ids in members.items():
        for lm in landmarks:
            sub = frame[frame["landmark_days"].eq(int(lm)) & frame[RISK_COL].notna()]
            if ids is not None:
                sub = sub[sub[ID_COL].isin(ids)]
            tags = {"cohort": cohort, "landmark_days": int(lm)}
            count_rows.append(_counts(sub, tags, None if ids is None else len(ids)))
            for outcome in outcomes:
                row, coefs, predictions = _fit_outcome(
                    outcome_population(sub, outcome), outcome, tags, min_events=min_events,
                    n_folds=n_folds, n_bootstrap=n_bootstrap, seed=seed)
                model_rows.append(row)
                coef_rows += coefs
                if predictions is not None:
                    prediction_frames.append(predictions)
    out = (pd.DataFrame(model_rows).reindex(columns=MODEL_COLUMNS),
           pd.DataFrame(coef_rows).reindex(columns=COEF_COLUMNS),
           pd.DataFrame(count_rows).reindex(columns=COUNT_COLUMNS))
    if not return_predictions:
        return out
    predictions = (pd.concat(prediction_frames, ignore_index=True) if prediction_frames
                   else pd.DataFrame())
    return (*out, predictions.reindex(columns=PREDICTION_COLUMNS))


def marker_frequency(calls: pd.DataFrame, specimens: pd.DataFrame) -> pd.DataFrame:
    """Prostate-relevant specimens per marker: stained, positive, negative, discordant, stained among NE-positive."""
    return ann.marker_frequency(calls, specimens.rename(columns={LOCAL_ID_COL: SOURCE_ID_COL}))
