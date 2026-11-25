"""Logistic regression of future prostate OncoPanel IHC (and NE / NEPC-like IHC) on the federated risk score.

Backs ``notebooks/text_interpretability/post_landmark/11_LR_for_IHC.ipynb``. For each ADT landmark, the frozen
federated (no-DFCI) platinum-trained XGBoost score from notebook 04 (``endpoint == "nepc"`` rows: patients
with NEPC or censoring at/before the landmark are not at risk) is the only predictor, z-scored within each
fit, in four unpenalized LRs:

* ``ihc_any``   -- a future sequenced specimen whose matched pathology report has >=1 lesional stain call;
                   all at-risk patients.
* ``ne_pos``    -- a future sequenced specimen that is NE-positive (``NE_any_pos``: synaptophysin,
                   chromogranin, INSM1 or CD56 positive); at-risk patients without a prior NE-positive one.
* ``nepc_like`` -- a future sequenced specimen that is NEPC-like (``NEPC_ihc``: NE-positive, AR axis
                   negative, no AR-axis marker positive); same population as ``ne_pos``.
* ``ar_low``    -- a future sequenced specimen that is AR-axis low (``ARaxis_any_neg & ~ARaxis_any_pos``:
                   >=1 of AR, PSA, NKX3.1, PSAP, PSMA, prostein negative and none positive), whatever the NE
                   stains show; at-risk patients without a prior AR-low specimen.

**IHC source.** The prostate_ihc extraction (``COMPASS/prostate_ihc``), step 2:
``oncopanel_prostate_specimen_ihc.csv`` (one row per OncoPanel prostate specimen) and
``oncopanel_prostate_ihc_long.csv`` (stain mentions on the matched reports). Only specimens matched to a
pathology report (``path_matched``) can be events; unmatched ones are counted for attrition only. No LLM
findings are used. The CSV flags are used as written (a discordant marker counts as neither positive nor
negative, as in the original scripts).

**Population.** Notebook 11 restricts the scored patients to the prostate OncoPanel cohort of the step-2
output (``load_oncopanel_patients``: every ``MRN_INT`` in the specimen CSV, matched or not).

**Timing.** A specimen is dated by its pathology ``specimen_date``: the earliest lesional-stain
``specimen_date`` in the long table for that ``UNIQUE_SAMPLE_ID``. "Future" = ADT + landmark < date <=
ADT + MAX_FOLLOWUP_DAYS (the models' 10-year cap); the label is "ever during that window", not
fixed-horizon. Prior IHC excludes no one; a prior NE-positive specimen (at or before ADT + landmark)
excludes the patient from both NE outcomes (NEPC-like is a subset of NE-positive); a prior AR-low
specimen excludes the patient from ``ar_low``.

AUC is from out-of-fold predictions (stratified K-fold) with a patient bootstrap 95% CI, plus the AUC of
the untransformed score (``auc_raw_score``, preferred). Odds ratios (per SD), Wald CIs and LRT p-values
come from the full-population fit. The LR machinery is reused from ``nepc_subtyping_lr``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from COMPASS.multivariable_models import nepc_subtyping_lr as nsl

ID_COL = nsl.ID_COL                     # "person_id" (scores / LR frame)
LOCAL_ID_COL = nsl.LOCAL_ID_COL         # "DFCI_MRN" (== MRN_INT in the prostate_ihc outputs)
SOURCE_ID_COL = "MRN_INT"
SPECIMEN_ID_COL = "UNIQUE_SAMPLE_ID"
RISK_COL = nsl.RISK_COL
ANCHOR_SOURCE_COL = "TREATMENT_ANCHOR_DATE"
MAX_FOLLOWUP_DAYS = 3650
IHC_SOURCE_VERSION = "prostate-ihc-oncopanel-original-csv"  # pre-existing step-2 CSVs from the original scripts

# Marker call columns of oncopanel_prostate_specimen_ihc.csv (lesional calls; null = not stained).
MARKER_COLUMNS = ['AR', 'PSA', 'NKX3.1', 'PSAP', 'PSMA', 'Prostein/P501S', 'ERG', 'Synaptophysin', 'Chromogranin',
                  'INSM1', 'CD56', 'ASCL1', 'NEUROD1', 'DLL3', 'SSTR2', 'TTF-1', 'YAP1', 'SOX2', 'RB1', 'p53', 'PTEN',
                  'p63', 'HMWCK (34bE12)', 'CK5/6', 'PIN4 cocktail', 'AMACR', 'GATA3', 'CK7', 'CK20']
SPECIMEN_REQUIRED = [SOURCE_ID_COL, SPECIMEN_ID_COL, "path_matched", "NE_any_pos", "NEPC_ihc",
                     "ARaxis_any_pos", "ARaxis_any_neg", "collection_date"]
LONG_REQUIRED = [SPECIMEN_ID_COL, "specimen_date", "target"]

# Outcome -> event column; the population is set by outcome_population().
OUTCOMES = {
    "ihc_any": "ihc_future",
    "ne_pos": "ne_pos_future",
    "nepc_like": "nepc_like_future",
    "ar_low": "ar_low_future",
}
NE_OUTCOMES = ("ne_pos", "nepc_like")
# Outcome -> prior flag whose patients are excluded from that outcome's population.
PRIOR_EXCLUSION = {"ihc_any": None, "ne_pos": "prior_ne_pos", "nepc_like": "prior_ne_pos", "ar_low": "prior_ar_low"}
EXCLUDED_OUTCOMES = tuple(o for o, f in PRIOR_EXCLUSION.items() if f)

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
    "n_ihc_future", "n_ne_pos_future", "n_nepc_like_future", "n_prior_ne_pos",
    "n_ne_pos", "events_ne_pos", "n_nepc_like", "events_nepc_like",
    "n_ar_low_future", "n_prior_ar_low", "n_ar_low", "events_ar_low",
    "n_unmatched_specimen_future", "n_undated_ihc_specimen",
]
AUDIT_COLUMNS = [
    LOCAL_ID_COL, SPECIMEN_ID_COL, "specimen_date", "days_from_adt", "collection_date",
    "path_matched", "ihc_any", "ne_pos", "nepc_like", "ar_low", "ihc_phenotype",
]
# Per-patient OOF predictions; ``event`` is that outcome's label (OUTCOMES[outcome]).
PREDICTION_COLUMNS = ["cohort", "landmark_days", "outcome", *nsl.PREDICTION_KEY_COLUMNS]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
_TRUE = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}


def _as_bool(s: pd.Series, name: str) -> pd.Series:
    """CSV booleans (polars writes true/false) -> bool; missing -> False; anything else raises."""
    if s.dtype == bool:
        return s
    mapped = s.map(lambda v: v if isinstance(v, (bool, np.bool_)) else
                   (False if pd.isna(v) else _TRUE.get(str(v).strip().lower())))
    bad = mapped.isna()
    if bad.any():
        raise ValueError(f"Column {name!r} has non-boolean values: {sorted(s[bad].astype(str).unique())[:5]}")
    return mapped.astype(bool)


def _read(obj, what) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):            # polars
        return obj.to_pandas()
    path = Path(obj)
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}. Expected the pre-existing step-2 CSV in ihc_extraction/results/.")
    return pd.read_csv(path, low_memory=False)


def load_oncopanel_ihc(specimen, long) -> pd.DataFrame:
    """One row per OncoPanel prostate specimen with the outcome flags and its pathology date.

    ``specimen`` / ``long`` are the two step-2 CSVs (paths or frames). Returns ``DFCI_MRN,
    UNIQUE_SAMPLE_ID, path_matched, specimen_date, collection_date, ihc_any, ne_pos, nepc_like,
    ar_low, ihc_phenotype``. Flags are False for unmatched specimens.
    """
    spec = _read(specimen, "OncoPanel specimen IHC table")
    lng = _read(long, "OncoPanel IHC long table")
    for df, req, what in ((spec, SPECIMEN_REQUIRED, "specimen"), (lng, LONG_REQUIRED, "long")):
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"OncoPanel {what} table lacks columns {missing}")
    if spec[SPECIMEN_ID_COL].isna().any() or spec[SPECIMEN_ID_COL].duplicated().any():
        raise ValueError(f"Missing or duplicate {SPECIMEN_ID_COL} in the OncoPanel specimen table.")

    out = pd.DataFrame({
        LOCAL_ID_COL: pd.to_numeric(spec[SOURCE_ID_COL], errors="raise").astype(int),
        SPECIMEN_ID_COL: spec[SPECIMEN_ID_COL].astype(str),
        "path_matched": _as_bool(spec["path_matched"], "path_matched"),
    })
    # The long table also contains stains (e.g. Ki-67) without a marker-call
    # column in the specimen CSV. Any lesional stain mention counts.
    les = lng[lng["target"].astype(str).eq("lesional")].copy()
    out["ihc_any"] = out["path_matched"] & out[SPECIMEN_ID_COL].isin(les[SPECIMEN_ID_COL].astype(str))
    out["ne_pos"] = out["path_matched"] & _as_bool(spec["NE_any_pos"], "NE_any_pos").to_numpy()
    out["nepc_like"] = out["path_matched"] & _as_bool(spec["NEPC_ihc"], "NEPC_ihc").to_numpy()
    ar_pos = _as_bool(spec["ARaxis_any_pos"], "ARaxis_any_pos").to_numpy()
    ar_neg = _as_bool(spec["ARaxis_any_neg"], "ARaxis_any_neg").to_numpy()
    out["ar_low"] = out["path_matched"] & ar_neg & ~ar_pos
    out["ihc_phenotype"] = spec["ihc_phenotype"].to_numpy() if "ihc_phenotype" in spec.columns else None
    out["collection_date"] = _dates(spec["collection_date"], "collection_date")

    les["specimen_date"] = _dates(les["specimen_date"], "specimen_date")
    first = les.groupby(les[SPECIMEN_ID_COL].astype(str))["specimen_date"].min()
    out["specimen_date"] = out[SPECIMEN_ID_COL].map(first)
    out.loc[~out["path_matched"], "specimen_date"] = pd.NaT
    if ((out["ne_pos"] | out["ar_low"]) & ~out["ihc_any"]).any() or (out["nepc_like"] & ~out["ne_pos"]).any() \
            or (out["nepc_like"] & ~out["ar_low"]).any():
        raise ValueError("Inconsistent flags: NE-positive / AR-low without a stain call, or NEPC-like without "
                         "NE-positive and AR-low.")
    return out


def _dates(s: pd.Series, name: str) -> pd.Series:
    raw = s.astype("string").str.slice(0, 10)
    parsed = pd.to_datetime(raw, errors="coerce", format="%Y-%m-%d")
    bad = raw.notna() & raw.str.strip().ne("") & parsed.isna()
    if bad.any():
        raise ValueError(f"Unparseable {name} values, e.g. {raw[bad].iloc[0]!r}")
    return parsed


def load_oncopanel_patients(path=None) -> set[int]:
    """Prostate OncoPanel patients: ``MRN_INT`` of the step-2 ``oncopanel_prostate_specimen_ihc.csv``.

    That table has one row per sequenced prostate specimen (``CANCER_TYPE == 'Prostate Cancer'``), matched to a
    pathology report or not, so its patients are the original scripts' OncoPanel prostate cohort. Only the MRN
    column is read.
    """
    from COMPASS.prostate_ihc import paths as ihc_paths
    path = path or ihc_paths.default_paths().results_dir / ihc_paths.ONCOPANEL_SPECIMEN_FILE
    try:
        mrn = pd.read_csv(path, usecols=[SOURCE_ID_COL])[SOURCE_ID_COL]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing OncoPanel specimen table {path} (step-2 output).") from exc
    mrn = pd.to_numeric(mrn.dropna(), errors="raise").astype(int)
    if mrn.empty:
        raise ValueError(f"No patients in {path}.")
    return set(mrn)


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
    """``days_from_adt`` (pathology date) and ``collection_days_from_adt`` (OncoPanel date)."""
    out = specimens.copy()
    anchor = out[LOCAL_ID_COL].map(anchors)
    out["days_from_adt"] = (out["specimen_date"] - anchor).dt.days
    out["collection_days_from_adt"] = (out["collection_date"] - anchor).dt.days
    return out


def build_ihc_frame(scores: pd.DataFrame, specimens: pd.DataFrame, landmarks, *,
                    anchors: pd.Series, max_days: int = MAX_FOLLOWUP_DAYS) -> pd.DataFrame:
    """One row per scored patient x landmark with the IHC outcome flags.

    Flags: ``ihc_future``, ``ne_pos_future``, ``nepc_like_future``, ``ar_low_future`` (matched
    specimen dated in (ADT + landmark, ADT + max_days]), ``prior_ne_pos`` / ``prior_ar_low``
    (NE-positive / AR-low specimen dated at or before ADT + landmark), plus attrition: ``unmatched_specimen_future`` (an unmatched sequenced
    specimen collected in the window) and ``undated_ihc_specimen`` (a stained specimen with no
    pathology date, never counted).
    """
    validate_anchor_coverage(scores, anchors)
    s = add_days_from_adt(specimens, anchors)
    undated = set(s.loc[s["ihc_any"] & s["days_from_adt"].isna(), LOCAL_ID_COL].astype(int))
    rows = []
    for lm in landmarks:
        lm = int(lm)
        sc = scores[scores["landmark_days"].eq(lm)][[ID_COL, "landmark_days", RISK_COL]].copy()
        d = s["days_from_adt"]
        fut = s[(d > lm) & (d <= max_days)]
        prior = s[d <= lm]
        cd = s["collection_days_from_adt"]
        unmatched = s[~s["path_matched"] & (cd > lm) & (cd <= max_days)]

        def ids(df):
            return set(df[LOCAL_ID_COL].astype(int))

        flags = {
            "ihc_future": ids(fut[fut["ihc_any"]]),
            "ne_pos_future": ids(fut[fut["ne_pos"]]),
            "nepc_like_future": ids(fut[fut["nepc_like"]]),
            "ar_low_future": ids(fut[fut["ar_low"]]),
            "prior_ne_pos": ids(prior[prior["ne_pos"]]),
            "prior_ar_low": ids(prior[prior["ar_low"]]),
            "unmatched_specimen_future": ids(unmatched),
            "undated_ihc_specimen": undated,
        }
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
    flag = PRIOR_EXCLUSION[outcome]
    return sub if flag is None else sub[sub[flag].eq(0)]


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
           "n_ihc_future": s("ihc_future"), "n_ne_pos_future": s("ne_pos_future"),
           "n_nepc_like_future": s("nepc_like_future"), "n_prior_ne_pos": s("prior_ne_pos"),
           "n_ar_low_future": s("ar_low_future"), "n_prior_ar_low": s("prior_ar_low"),
           "n_unmatched_specimen_future": s("unmatched_specimen_future"),
           "n_undated_ihc_specimen": s("undated_ihc_specimen")}
    for outcome in EXCLUDED_OUTCOMES:
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


def marker_frequency(specimen) -> pd.DataFrame:
    """Matched sequenced specimens per marker: stained, positive (or retained), and stained among NE-positive."""
    spec = _read(specimen, "OncoPanel specimen IHC table")
    spec = spec[_as_bool(spec["path_matched"], "path_matched")]
    ne = _as_bool(spec["NE_any_pos"], "NE_any_pos")
    rows = []
    for m in (c for c in MARKER_COLUMNS if c in spec.columns):
        call = spec[m]
        if call.notna().sum() == 0:
            continue
        rows.append({"marker": m, "specimens_stained": int(call.notna().sum()),
                     "specimens_positive": int(call.isin(["positive", "retained", "discordant"]).sum()),
                     "stained_among_ne_pos": int((call.notna() & ne).sum())})
    if not rows:
        return pd.DataFrame(columns=["marker", "specimens_stained", "specimens_positive", "stained_among_ne_pos"])
    return pd.DataFrame(rows).sort_values("specimens_stained", ascending=False).reset_index(drop=True)
