"""Score a federated CAIA model bundle on local data.

The federated runs (caia-project-compass/rhino_scripts) fit XGBoost,
elastic-net Cox and Dynamic-DeepHit models across sites and export each fitted
model as a ``caia-federated-model-bundle`` JSON file.  This module loads such a bundle and
applies it to a local cohort, so DFCI data can be scored on the cluster without
NVFLARE and without any module from the federated repo.

Why the bundle rather than the result CSVs: the CSVs are summaries.  The
elastic-net coefficient CSV drops every zero coefficient, so it cannot rebuild
the covariate vector the fit used, and neither CSV carries the pooled
preprocessing or the Breslow baseline.

The preprocessing in the bundle is **pooled across the federation and must be
reused verbatim**.  Re-fitting an imputer or a scaler on local data would centre
the features on the DFCI cohort instead of the training cohort and shift every
prediction -- silently, since the output still looks like a plausible risk
score.

Expected input frame: one row per patient, columns named as in the federated
wide frame (``<sanitized_lab_name>__<stat>``), plus the age column.  Anything the
bundle does not ask for is ignored, and a covariate the frame lacks is treated
as missing and imputed with the pooled train mean.

``dynamic_deephit`` bundles read a *long* lab frame instead -- one row per lab
measurement, with ``person_id``, ``lab_name``, ``lab_value`` and
``days_relative_to_anchor`` -- because the model reads each patient's lab
history as a sequence.  The bundle's pooled clip bounds, means and standard
deviations are applied verbatim, the risk score is ``CIF_total`` (the
probability of the event within the model's window), and survival at a horizon
is ``1 - CIF(h)``.  Scoring runs the GRU in numpy, so torch is only needed to
get the trained ``DynamicDeepHitGRU`` back as an object.  The family's
``config="baseline"`` model is age-only: it reads one standardized-age step per
patient, so its input is a one-row-per-patient frame with the age column.

Loading the trained object in Python::

    from survival_common.federated_inference import load_fitted_model

    booster, bundle, model = load_fitted_model("xgboost_federated_model_adt.json", 90)
    booster.trees_to_dataframe()          # a real xgboost.Booster

    cox, bundle, model = load_fitted_model("cox_federated_elasticnet_model_adt.json", 90)
    cox.params_                           # coefficients by covariate name
    cox.predict(df, age_col="age_at_anchor")

    gru, bundle, model = load_fitted_model("deephit_federated_model_adt.json", 90)
    gru                                   # deephit_engine.DynamicDeepHitGRU (needs torch)
    ids, cif = predict_cif(labs_long, model)   # numpy; cif[:, -1] is the risk

CLI::

    python -m survival_common.federated_inference \\
        --bundle xgboost_federated_model_adt.json \\
        --data cohort_wide.parquet \\
        --landmark 90 \\
        --output risk_scores.csv

    # dynamic_deephit: --data is the long lab frame; --cohort lists every
    # patient to score, so one with no pre-landmark labs is still scored.
    python -m survival_common.federated_inference \\
        --bundle deephit_federated_model_adt.json \\
        --data labs_long.parquet --cohort cohort.parquet \\
        --landmark 90 --horizons 365,730 \\
        --output risk_scores.csv
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

try:
    import xgboost as xgb

    XGBOOST_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    xgb = None
    XGBOOST_IMPORT_ERROR = exc


BUNDLE_FORMAT = "caia-federated-model-bundle"
DEEPHIT_FAMILY = "dynamic_deephit"
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)


class BundleError(ValueError):
    """The bundle is not a model bundle this module can score."""


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #
def load_bundle(path) -> dict:
    """Read a bundle and check it is one this module understands.

    Validating the format up front keeps a wrong file (a results CSV renamed, a
    bundle from a future exporter) from failing later as a confusing KeyError.
    """
    with open(path) as fh:
        bundle = json.load(fh)

    if not isinstance(bundle, dict):
        raise BundleError(f"{path}: not a JSON object")
    if bundle.get("format") != BUNDLE_FORMAT:
        raise BundleError(
            f"{path}: format is {bundle.get('format')!r}, expected {BUNDLE_FORMAT!r}"
        )
    version = bundle.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise BundleError(
            f"{path}: format_version {version!r} is not supported "
            f"(this module reads {list(SUPPORTED_FORMAT_VERSIONS)})"
        )
    if not bundle.get("models"):
        raise BundleError(
            f"{path}: contains no fitted models -- the run produced none, or it "
            "aborted before the final fit"
        )
    family = bundle.get("model_family")
    allowed = {"elastic_net_cox": {1}, "xgboost_cox": {1},
               DEEPHIT_FAMILY: {1, 2, 3}, "predictive_checklist": {1}}
    if family not in allowed or version not in allowed[family]:
        raise BundleError(f"{path}: unsupported model_family/version {family!r}/{version}")
    return bundle


def model_identity(bundle: dict, model: dict) -> tuple:
    arm = (model.get("training_arm") or "landmark") if bundle.get("model_family") == DEEPHIT_FAMILY else None
    return (int(model["landmark_days"]), str(model.get("config")), arm,
            int(model["horizon_days"]) if model.get("horizon_days") is not None else None)


def select_model(bundle: dict, landmark_days: int, config: str = "both", *,
                 training_arm: str | None = None, horizon_days: int | None = None) -> dict:
    """Pick one fitted model out of a bundle.

    ``config`` is ``both`` for the full model or ``baseline`` for the age-only
    comparator, matching the federated result tables.
    """
    wanted = [
        m for m in bundle["models"]
        if int(m["landmark_days"]) == int(landmark_days)
        and str(m.get("config")) == str(config)
        and (training_arm is None or model_identity(bundle, m)[2] == training_arm)
        and (horizon_days is None or model_identity(bundle, m)[3] == int(horizon_days))
    ]
    if not wanted:
        available = sorted(
            {(int(m["landmark_days"]), str(m.get("config"))) for m in bundle["models"]}
        )
        raise BundleError(
            f"no model for landmark={landmark_days} config={config!r}; "
            f"bundle has {available}"
        )
    if len(wanted) != 1:
        raise BundleError(f"ambiguous model for landmark={landmark_days} config={config!r}; "
                          f"specify training_arm and/or horizon_days: "
                          f"{[model_identity(bundle, m) for m in wanted]}")
    return wanted[0]


# --------------------------------------------------------------------- #
# Design matrix
# --------------------------------------------------------------------- #
def build_design_matrix(
    df: pd.DataFrame,
    model: dict,
    *,
    age_col: str | None = None,
) -> pd.DataFrame:
    """Rebuild the model's design matrix from a local wide frame.

    Reproduces the federated transform in order: resolve each covariate to its
    source column, add the ``__missing`` indicators BEFORE imputing (they record
    what was absent in the raw data), impute with the pooled train means, then
    standardise with the pooled centers and scales.

    Returns a frame whose columns are exactly ``covariate_cols``, in that order.
    Column order is load-bearing: XGBoost scores by position, so a reordered
    matrix returns wrong risks without raising.
    """
    pre = model["preprocessing"]
    cols = list(pre["covariate_cols"])
    source = dict(pre.get("source_cols") or {})
    impute = dict(pre.get("impute_means") or {})
    centers = dict(pre.get("centers") or {})
    scales = dict(pre.get("scales") or {})
    missing_cols = set(pre.get("missing_indicator_cols") or [])

    out = pd.DataFrame(index=df.index)
    missing_from_frame: list[str] = []

    for col in cols:
        if col in missing_cols:
            continue  # built below, from its own source column
        src = source.get(col, col)
        if src == "age" and age_col:
            src = age_col
        if src in df.columns:
            values = pd.to_numeric(df[src], errors="coerce")
        else:
            missing_from_frame.append(col)
            values = pd.Series(np.nan, index=df.index, dtype=float)
        out[col] = values.astype(float)

    # Indicators describe the RAW data, so they must be computed before imputing.
    for col in cols:
        if col not in missing_cols:
            continue
        base = col[: -len("__missing")]
        if base in out.columns:
            out[col] = (~np.isfinite(out[base].to_numpy(dtype=float))).astype(float)
        else:
            src = source.get(base, base)
            if src in df.columns:
                raw = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=float)
                out[col] = (~np.isfinite(raw)).astype(float)
            else:
                missing_from_frame.append(col)
                out[col] = 1.0

    for col in cols:
        if col in missing_cols:
            continue
        mean = impute.get(col)
        if mean is not None:
            out[col] = out[col].fillna(float(mean))

    for col in cols:
        center = float(centers.get(col, 0.0))
        scale = float(scales.get(col, 1.0))
        out[col] = (out[col].astype(float) - center) / (scale if scale > 0 else 1.0)

    # Anything still non-finite (no pooled mean to fall back on) becomes the
    # scaled mean, i.e. zero -- the neutral value for a standardised covariate.
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if missing_from_frame:
        n = len(missing_from_frame)
        shown = ", ".join(sorted(set(missing_from_frame))[:10])
        print(
            f"[federated_inference] {n} covariate(s) absent from the input frame, "
            f"imputed with pooled train means: {shown}"
            + (" ..." if n > 10 else "")
        )

    return out[cols]


# --------------------------------------------------------------------- #
# The trained object
# --------------------------------------------------------------------- #
def _model_base_score(model_json: str) -> float | None:
    """The ``base_score`` recorded in a saved model, if it has one.

    XGBoost serialises every ``learner_model_param`` entry as a *string*, and
    for a multi-target-capable build it writes the vector form, so the value on
    disk reads ``"[2.4610445E0]"`` -- brackets included, inside the quotes.
    ``float()`` rejects that, so the brackets are stripped before parsing and
    only the first element is read (these models are single-target).  Returning
    ``None`` here disables the version guard, so a parse that quietly fails is
    the dangerous direction: it is the shape this function must get right.
    """
    try:
        raw = json.loads(model_json)
        value = raw["learner"]["learner_model_param"]["base_score"]
    except Exception:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, str):
        value = value.strip().lstrip("[").rstrip("]").split(",")[0].strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_xgboost_version(bundle: dict, model: dict) -> None:
    """Refuse to load a model this xgboost would score differently.

    XGBoost 3.x records the fitted ``base_score`` (for survival:cox, the
    intercept) in the saved model.  XGBoost 2.x ignores that field and
    substitutes its own default of 0.5, so it loads a 3.x model without error
    and returns every prediction shifted by a constant -- measured at 1.59 on
    the log-hazard scale for a Cox model, i.e. hazards scaled by ~0.2.

    Rankings survive that shift, so C-index and AUC(t) are unaffected, but
    absolute survival and any comparison against the federated run's own risk
    scores are wrong.  Nothing raises, which is why this checks explicitly
    rather than trusting the load to fail.
    """
    trained = str(bundle.get("xgboost_version") or "")
    if not trained:
        return  # written by an exporter that predates the stamp
    try:
        trained_major = int(trained.split(".")[0])
        local_major = int(str(xgb.__version__).split(".")[0])
    except (ValueError, IndexError):
        return
    if local_major >= trained_major:
        return
    base_score = _model_base_score(model.get("model_json") or "")
    if base_score is None or abs(base_score - 0.5) < 1e-9:
        return  # nothing for the older version to get wrong
    raise BundleError(
        f"this model was trained with xgboost {trained} and carries a fitted "
        f"base_score of {base_score:g}, which xgboost {xgb.__version__} ignores "
        f"(it would substitute 0.5). Every prediction would be shifted by a "
        f"constant -- rankings and C-index would still be right, but absolute "
        f"survival and risk scores would not. Install xgboost>={trained_major} "
        f"to score this bundle."
    )


def load_fitted_model(path, landmark_days: int, config: str = "both", *,
                      training_arm: str | None = None, horizon_days: int | None = None):
    """Open a bundle and hand back ``(estimator, bundle, model)`` in one call.

    The usual entry point when you want the trained object::

        booster, bundle, model = load_fitted_model("xgboost_..._adt.json", 90)
        booster.trees_to_dataframe()

    ``bundle`` and ``model`` come back too because the estimator alone cannot
    score raw data -- the preprocessing lives in ``model``.
    """
    bundle = load_bundle(path)
    model = select_model(bundle, landmark_days, config, training_arm=training_arm,
                         horizon_days=horizon_days)
    return load_model(bundle, model), bundle, model


def load_model(bundle: dict, model: dict):
    """Return the trained estimator itself, ready to use.

    For ``xgboost_cox`` this is a real :class:`xgboost.Booster` -- the same
    object the federated run trained, restored from the bundle's JSON.  Every
    Booster method works on it: ``predict``, ``save_model``, ``trees_to_dataframe``,
    ``get_score``, SHAP via ``pred_contribs=True``, ``xgboost.plot_tree``.

    For ``dynamic_deephit`` this is a :class:`deephit_engine.DynamicDeepHitGRU`
    in eval mode with the federated weights loaded -- the federated app keeps
    that class's module layout, so the state dict loads strictly.  This is the
    only part of the family that needs torch.

    For ``elastic_net_cox`` there is no third-party estimator to rebuild -- the
    federated fit is this repo's own proximal-gradient Cox, not a lifelines or
    sksurv object -- so this returns a :class:`FederatedCoxModel`, a small
    estimator exposing the same coefficients and baseline.

    Restoring from JSON rather than a pickle is deliberate.  XGBoost pickles are
    not portable across versions, and the model is trained inside the federated
    Docker image, exported here, and scored on the cluster -- three different
    environments (this repo pins xgboost 2.1.1).  The JSON model format is
    XGBoost's supported interchange format across exactly that gap.

    Scoring raw data still needs the bundle's preprocessing, so prefer
    :func:`predict_risk` for predictions and use this when you want the
    estimator: inspecting trees, computing SHAP, or re-saving the model.
    """
    family = bundle.get("model_family")

    if family == "xgboost_cox":
        if xgb is None:
            raise ModuleNotFoundError(
                "xgboost is required to load an xgboost_cox model."
            ) from XGBOOST_IMPORT_ERROR
        _check_xgboost_version(bundle, model)
        booster = xgb.Booster()
        booster.load_model(bytearray(model["model_json"], "utf-8"))
        # Restored boosters carry their own feature names, but an older writer
        # may not have; setting them keeps get_score/plot_tree labelled.
        if not booster.feature_names and model.get("feature_names"):
            booster.feature_names = list(model["feature_names"])
        return booster

    if family == "elastic_net_cox":
        return FederatedCoxModel.from_bundle(bundle, model)

    if family == DEEPHIT_FAMILY:
        return _load_deephit_module(model)

    if family == "predictive_checklist":
        return _checklist_components(model)

    raise BundleError(f"unknown model_family {family!r}")


class FederatedCoxModel:
    """The fitted federated elastic-net Cox model, as an object.

    The federated fit is this repo's own proximal-gradient solver, so there is
    no lifelines/sksurv estimator to restore -- this carries what such an
    estimator would: the coefficients, the covariate order they belong to, the
    pooled preprocessing, and the Breslow baseline.
    """

    def __init__(self, covariate_cols, coefficients, preprocessing,
                 baseline_cumhaz=None, hyperparameters=None, landmark_days=None,
                 config=None):
        self.covariate_cols = list(covariate_cols)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.preprocessing = dict(preprocessing or {})
        self.baseline_cumhaz = baseline_cumhaz
        self.hyperparameters = dict(hyperparameters or {})
        self.landmark_days = landmark_days
        self.config = config
        if len(self.coefficients) != len(self.covariate_cols):
            raise BundleError(
                f"{len(self.coefficients)} coefficients for "
                f"{len(self.covariate_cols)} covariates"
            )

    @classmethod
    def from_bundle(cls, bundle: dict, model: dict) -> "FederatedCoxModel":
        return cls(
            covariate_cols=model["covariate_cols"],
            coefficients=model["coefficients"],
            preprocessing=model["preprocessing"],
            baseline_cumhaz=model.get("baseline_cumhaz"),
            hyperparameters=model.get("hyperparameters"),
            landmark_days=model.get("landmark_days"),
            config=model.get("config"),
        )

    @property
    def params_(self) -> pd.Series:
        """Coefficients indexed by covariate name, like a lifelines ``params_``."""
        return pd.Series(self.coefficients, index=self.covariate_cols, name="coef")

    @property
    def hazard_ratios_(self) -> pd.Series:
        return np.exp(self.params_).rename("hazard_ratio")

    def _as_model_dict(self) -> dict:
        return {
            "covariate_cols": self.covariate_cols,
            "coefficients": list(self.coefficients),
            "preprocessing": self.preprocessing,
            "baseline_cumhaz": self.baseline_cumhaz,
        }

    def predict(self, df: pd.DataFrame, *, age_col: str | None = None) -> np.ndarray:
        """Linear predictor for a raw wide frame (applies the preprocessing)."""
        design = build_design_matrix(df, self._as_model_dict(), age_col=age_col)
        return design.to_numpy(dtype=float) @ self.coefficients

    def predict_survival(self, df: pd.DataFrame, horizons_days,
                         *, age_col: str | None = None) -> pd.DataFrame:
        return predict_survival(
            self.predict(df, age_col=age_col), self._as_model_dict(), horizons_days
        )

    def __repr__(self) -> str:
        nz = int(np.count_nonzero(self.coefficients))
        return (
            f"FederatedCoxModel(landmark_days={self.landmark_days}, "
            f"config={self.config!r}, n_covariates={len(self.covariate_cols)}, "
            f"n_nonzero={nz})"
        )


# --------------------------------------------------------------------- #
# Dynamic-DeepHit
# --------------------------------------------------------------------- #
def _deephit_state(model: dict) -> dict[str, np.ndarray]:
    """The bundle's ``{name: {shape, values}}`` weights as float64 arrays."""
    return {
        name: np.asarray(item["values"], dtype=np.float64).reshape(item["shape"])
        for name, item in model["state_dict"].items()
    }


def _checklist_components(model: dict):
    """Restore the frozen PROFILE binarizer, item weights and calibrator."""
    from COMPASS.survival_analysis.predictive_checklists.evaluation import ProbabilityCalibrator
    from COMPASS.survival_analysis.predictive_checklists.features import FeatureBinarizer
    from COMPASS.survival_analysis.predictive_checklists.optimizer import ChecklistModel

    payload = model.get("checklist") or {}
    if payload.get("schema_version") != 1:
        raise BundleError("unsupported predictive checklist payload schema")
    return (FeatureBinarizer.from_dict(payload["binarizer"]),
            ChecklistModel.from_dict(payload["model"]),
            ProbabilityCalibrator.from_dict(payload["calibrator"]))


def predict_checklist(frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Count selected items and apply the exported threshold and calibrator."""
    binarizer, checklist, calibrator = _checklist_components(model)
    selected = {binarizer.items[i]["feature"] for i in checklist.selected_indices}
    absent = sorted(selected - set(frame.columns))
    if absent:
        raise BundleError(f"selected checklist feature(s) absent locally: {absent}")
    matrix = binarizer.transform(frame)
    score = checklist.predict_score(matrix)
    return pd.DataFrame({
        "score": score,
        "flag": checklist.predict(matrix),
        "probability": calibrator.predict(score / max(len(checklist.selected_indices), 1)),
    }, index=frame.index)


def _load_deephit_module(model: dict):
    try:
        import torch

        from .deephit_engine import DynamicDeepHitGRU
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required to load a dynamic_deephit model as a torch module; "
            "predict_cif / predict_risk score it without torch."
        ) from exc
    arch = model["architecture"]
    net = DynamicDeepHitGRU(
        int(arch["input_dim"]), int(arch["hidden_dim"]), int(arch["n_events"]),
        int(arch["max_pred_window"]), float(arch["dropout"]),
    )
    state = {
        name: torch.from_numpy(
            np.asarray(item["values"], dtype=np.float32).reshape(item["shape"])
        )
        for name, item in model["state_dict"].items()
    }
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def _deephit_reads_age(model: dict) -> bool:
    """True for the age-only baseline; bundles from before it read labs."""
    return model["preprocessing"].get("input_kind", "labs") == "age"


def build_deephit_age_sequences(
    df: pd.DataFrame,
    model: dict,
    *,
    ids=None,
    id_col: str = "person_id",
    age_col: str | None = None,
) -> tuple[list, list[np.ndarray]]:
    """The age-only baseline's input: one ``[(age - center) / scale]`` step.

    ``df`` has one row per patient.  ``age_col`` defaults to the bundle's
    source column; ``ids`` picks and orders the patients (default: every row
    of ``df``, by ``id_col`` when present).  A missing age scores NaN -- the
    federated cohort drops those patients.
    """
    pre = model["preprocessing"]
    col = age_col or pre["source_cols"]["age"]
    if col not in df.columns:
        raise BundleError(f"the age-only model needs an age column {col!r}")
    age = pd.to_numeric(df[col], errors="coerce")
    if id_col in df.columns:
        age = pd.Series(age.to_numpy(), index=df[id_col].to_numpy())
    if ids is not None:
        age = age.reindex(pd.Index(list(ids)))
    z = (age.to_numpy(dtype=float) - float(pre["centers"]["age"])) / float(pre["scales"]["age"])
    return list(age.index), [np.array([[v]]) for v in z]


def build_deephit_sequences(
    labs: pd.DataFrame,
    model: dict,
    *,
    ids=None,
    id_col: str = "person_id",
    lab_col: str = "lab_name",
    value_col: str = "lab_value",
    days_col: str = "days_relative_to_anchor",
    ages=None,
) -> tuple[list, list[np.ndarray]]:
    """Rebuild the GRU's input sequences from a long lab frame.

    The federated app's ``build_sequences``, applied with the bundle's pooled
    preprocessing (``cohort.build_person_period_wide`` followed by
    ``deephit_engine.build_sequences``):

    1. labs strictly before the landmark, binned to
       ``REL_BIN = floor((day - landmark) / time_unit_days)`` and averaged per
       (patient, bin, lab);
    2. each patient's history depth ``-min(REL_BIN)``, and
       ``TIME = REL_BIN + depth``, so the landmark sits at ``TIME = depth``;
    3. bin means clipped to the pooled bounds, plus an all-missing anchor step
       at the landmark;
    4. per step ``[standardized values (missing -> 0) | observed mask |
       TIME / max(max_observed_time, 1)]``.

    ``ids`` defaults to every patient in ``labs`` in order of first
    appearance.  Pass the full cohort to score patients with no labs: they
    get the anchor step alone, as in training.
    """
    pre = model["preprocessing"]
    lab_names = list(pre["lab_names"])
    unit = float(pre.get("time_unit_days") or 7)
    landmark = float(model["landmark_days"])
    if ids is None:
        ids = pd.unique(labs[id_col]).tolist()
    ids = list(ids)

    sub = labs[[id_col, lab_col, value_col, days_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub[days_col] = pd.to_numeric(sub[days_col], errors="coerce")
    # The sites strip lab names before matching them, so a stray space here
    # must not drop a lab the model reads.
    sub[lab_col] = sub[lab_col].astype(str).str.strip()
    sub = sub.dropna(subset=[id_col, value_col, days_col])
    sub = sub[(sub[days_col] < landmark) & sub[lab_col].isin(lab_names)
              & sub[id_col].isin(ids)]
    sub = sub.assign(
        REL_BIN=np.floor((sub[days_col].to_numpy(dtype=float) - landmark) / unit).astype(int)
    )
    cells = sub.groupby([id_col, "REL_BIN", lab_col], sort=False)[value_col].mean().reset_index()

    depth = (-cells.groupby(id_col)["REL_BIN"].min()).reindex(pd.Index(ids)).fillna(0).astype(int)

    lo = {l: v for l, v in zip(lab_names, pre["clip_lo"]) if v is not None}
    hi = {l: v for l, v in zip(lab_names, pre["clip_hi"]) if v is not None}
    vals = cells[value_col].to_numpy(dtype=float)
    lo_arr = cells[lab_col].map(lo).to_numpy(dtype=float)
    hi_arr = cells[lab_col].map(hi).to_numpy(dtype=float)
    has = np.isfinite(lo_arr) & np.isfinite(hi_arr)
    vals[has] = np.clip(vals[has], lo_arr[has], hi_arr[has])
    cells = cells.assign(**{value_col: vals})
    cells["TIME"] = cells["REL_BIN"].to_numpy(dtype=int) + depth.reindex(cells[id_col]).to_numpy(dtype=int)

    if cells.empty:
        wide = pd.DataFrame(columns=[id_col, "TIME"] + lab_names)
    else:
        wide = cells.pivot_table(
            index=[id_col, "TIME"], columns=lab_col, values=value_col, aggfunc="mean"
        ).reset_index()
        wide.columns.name = None
    for lab in lab_names:
        if lab not in wide.columns:
            wide[lab] = np.nan
    anchors = pd.DataFrame({id_col: ids, "TIME": depth.to_numpy(dtype=int)})
    for lab in lab_names:
        anchors[lab] = np.nan
    wide = pd.concat([wide[[id_col, "TIME"] + lab_names], anchors], ignore_index=True)
    wide = wide.sort_values([id_col, "TIME"], kind="mergesort")

    raw = wide[lab_names].astype(float)
    mean = pd.Series(pre["mean"], index=lab_names, dtype=float)
    std = pd.Series(pre["std"], index=lab_names, dtype=float)
    denom = max(float(pre["max_observed_time"]), 1.0)
    x = np.hstack([
        ((raw - mean) / std).fillna(0.0).to_numpy(dtype=np.float32),
        raw.notna().to_numpy(dtype=np.float32),
        (wide["TIME"].to_numpy(dtype=float) / denom).reshape(-1, 1).astype(np.float32),
    ])

    keys = wide[id_col].to_numpy()
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    by_id = {keys[s]: x[s:e] for s, e in zip(starts, ends)}
    sequences = [by_id[i] for i in ids]
    if pre.get("include_age"):
        if ages is None:
            raise BundleError("DeepHit bundle requires age_at_anchor for every scored patient")
        age = np.asarray(ages, dtype=float)
        if len(age) != len(ids) or not np.isfinite(age).all():
            raise BundleError("DeepHit age values must align with the ordered patient IDs")
        scale = float(pre["age_std"])
        if scale <= 0:
            raise BundleError("DeepHit age_std must be positive")
        z = (age - float(pre["age_mean"])) / scale
        sequences = [np.column_stack((seq, np.full(len(seq), z[i], dtype=np.float32)))
                     for i, seq in enumerate(sequences)]
    return ids, sequences


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def deephit_cif(xs: list[np.ndarray], model: dict) -> np.ndarray:
    """``[N, max_pred_window]`` cumulative incidence, by a numpy forward pass.

    torch's single-layer GRU (gate order r, z, n) over each sequence's own
    length, the final hidden state through the head with dropout off, a
    softmax over ``max_pred_window + 1`` outputs, and the cumulative sum of
    the event pmf.  Column ``h - 1`` is CIF at ``h`` time units; the last
    column is ``CIF_total``.
    """
    arch = model["architecture"]
    window = int(arch["max_pred_window"])
    if int(arch.get("n_events", 1)) != 1:
        raise BundleError("only single-event dynamic_deephit bundles are supported")
    w = _deephit_state(model)
    W_ih, W_hh = w["gru.weight_ih_l0"], w["gru.weight_hh_l0"]
    b_ih, b_hh = w["gru.bias_ih_l0"], w["gru.bias_hh_l0"]
    H = W_hh.shape[1]
    n = len(xs)
    if n == 0:
        return np.zeros((0, window))
    if xs[0].shape[1] != W_ih.shape[1]:
        raise BundleError(
            f"sequences have {xs[0].shape[1]} features per step, the model expects "
            f"{W_ih.shape[1]}"
        )

    lengths = np.array([len(s) for s in xs])
    padded = np.zeros((n, lengths.max(), W_ih.shape[1]))
    for i, s in enumerate(xs):
        padded[i, : len(s)] = s
    h = np.zeros((n, H))
    for t in range(lengths.max()):
        gi = padded[:, t] @ W_ih.T + b_ih
        gh = h @ W_hh.T + b_hh
        r = _sigmoid(gi[:, :H] + gh[:, :H])
        z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        cand = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        step = (1.0 - z) * cand + z * h
        # A finished sequence keeps its last hidden state, as packing does.
        h = np.where((t < lengths)[:, None], step, h)

    hidden = np.maximum(h @ w["head.1.weight"].T + w["head.1.bias"], 0.0)
    logits = hidden @ w["head.4.weight"].T + w["head.4.bias"]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return np.cumsum(probs[:, :window], axis=1)


def predict_cif(
    labs: pd.DataFrame, model: dict, *, ids=None, age_col: str | None = None,
    cohort: pd.DataFrame | None = None, **cols
) -> tuple[list, np.ndarray]:
    """``(ids, CIF)`` for a long lab frame; see :func:`build_deephit_sequences`.

    For the age-only baseline ``labs`` is instead a one-row-per-patient frame
    with the age column; see :func:`build_deephit_age_sequences`.
    """
    if _deephit_reads_age(model):
        ids, xs = build_deephit_age_sequences(
            labs, model, ids=ids, id_col=cols.get("id_col", "person_id"), age_col=age_col
        )
    else:
        ages = None
        if model["preprocessing"].get("include_age"):
            if cohort is None:
                raise BundleError("DeepHit bundle with include_age needs the cohort frame")
            id_col = cols.get("id_col", "person_id")
            source = age_col or model["preprocessing"].get("age_source_col", "age_at_anchor")
            if source not in cohort or id_col not in cohort:
                raise BundleError(f"DeepHit cohort needs {id_col!r} and {source!r}")
            ordered = list(ids) if ids is not None else pd.unique(labs[id_col]).tolist()
            ages = pd.to_numeric(cohort.set_index(id_col)[source], errors="coerce").reindex(ordered).to_numpy()
        ids, xs = build_deephit_sequences(labs, model, ids=ids, ages=ages, **cols)
    return ids, deephit_cif(xs, model)


def deephit_survival(cif: np.ndarray, model: dict, horizons_days) -> pd.DataFrame:
    """``1 - CIF(h)`` at each horizon in days.

    A day horizon covers the time units it fully contains: targets are
    ``ceil(days / unit)``, so an event by day ``h`` is one in a bin up to
    ``floor(h / unit)``.  Past the window the CIF is flat at its total.
    """
    cif = np.asarray(cif, dtype=float)
    unit = float(model["preprocessing"].get("time_unit_days") or 7)
    out = {}
    for h in horizons_days:
        k = min(int(np.floor(float(h) / unit)), cif.shape[1])
        out[f"survival_{int(h)}d"] = 1.0 - cif[:, k - 1] if k >= 1 else np.ones(len(cif))
    return pd.DataFrame(out)


# --------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------- #
def predict_risk(
    df: pd.DataFrame,
    bundle: dict,
    model: dict,
    *,
    age_col: str | None = None,
    ids=None,
    cohort: pd.DataFrame | None = None,
) -> np.ndarray:
    """Linear predictor / log-hazard-ratio for each row.

    Higher is higher risk, for every family.  This is on the same scale the
    federated run used to compute its C-index, so local and federated
    discrimination are directly comparable.

    For ``dynamic_deephit`` ``df`` is the long lab frame and the score is
    ``CIF_total``, one per patient in :func:`build_deephit_sequences`'s order
    (pass ``ids`` to fix it).  Its age-only baseline reads ``df``'s age column
    instead, one row per patient.
    """
    family = bundle.get("model_family")
    if family == DEEPHIT_FAMILY:
        return predict_cif(df, model, ids=ids, age_col=age_col, cohort=cohort)[1][:, -1]

    if family == "predictive_checklist":
        return predict_checklist(df, model)["score"].to_numpy(dtype=float)

    design = build_design_matrix(df, model, age_col=age_col)

    if family == "elastic_net_cox":
        beta = np.asarray(model["coefficients"], dtype=float)
        if len(beta) != design.shape[1]:
            raise BundleError(
                f"model has {len(beta)} coefficients but {design.shape[1]} covariates"
            )
        return design.to_numpy(dtype=float) @ beta

    if family == "xgboost_cox":
        booster = load_model(bundle, model)
        feature_names = list(model.get("feature_names") or design.columns)
        dm = xgb.DMatrix(design.to_numpy(dtype=float), feature_names=feature_names)
        kwargs = {"output_margin": True}
        n_trees = model.get("n_trees_used")
        if n_trees:
            kwargs["iteration_range"] = (0, int(n_trees))
        return np.asarray(booster.predict(dm, **kwargs), dtype=float)

    raise BundleError(f"unknown model_family {family!r}")


def predict_survival(
    risk: np.ndarray,
    model: dict,
    horizons_days,
    *,
    time_unit_days: float = 1.0,
) -> pd.DataFrame:
    """S(t|x) at each horizon, for a bundle that carries a Breslow baseline.

    Only the elastic-net export carries one; without it a model gives relative
    risk only, and absolute survival cannot be recovered from the scores alone.

    A ``dynamic_deephit`` model predicts absolute risk itself: pass the CIF
    matrix from :func:`predict_cif` as ``risk``.
    """
    if "architecture" in model:
        cif = np.asarray(risk, dtype=float)
        if cif.ndim != 2:
            raise BundleError(
                "dynamic_deephit survival needs the CIF matrix from predict_cif, "
                "not the scalar risk score"
            )
        return deephit_survival(cif, model, horizons_days)
    base = model.get("baseline_cumhaz")
    if not base:
        raise BundleError(
            "this model carries no baseline_cumhaz, so only relative risk is "
            "available (use predict_risk)"
        )
    unit = float(base.get("time_unit_days") or time_unit_days) or 1.0
    times = np.asarray(base["times"], dtype=float)
    cumhaz = np.asarray(base["cumhaz"], dtype=float)

    risk = np.asarray(risk, dtype=float)
    out = {}
    for h in horizons_days:
        # The baseline timeline is in AUC time units, the horizons in days.
        h_units = float(h) / unit
        idx = np.searchsorted(times, h_units, side="right") - 1
        h0 = float(cumhaz[idx]) if idx >= 0 else 0.0
        out[f"survival_{int(h)}d"] = np.exp(-h0 * np.exp(risk))
    return pd.DataFrame(out)


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def _read_table(path: str) -> pd.DataFrame:
    if str(path).endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", required=True, help="model bundle JSON")
    ap.add_argument(
        "--data", required=True,
        help="wide frame (.csv or .parquet); for dynamic_deephit, the long lab "
        "frame, or for its age-only baseline a table with --id-col and --age-col",
    )
    ap.add_argument(
        "--cohort", default="",
        help="dynamic_deephit: table whose --id-col lists every patient to score "
        "(default: the patients in --data)",
    )
    ap.add_argument("--landmark", type=int, required=True)
    ap.add_argument("--config", default="both", choices=["both", "baseline"])
    ap.add_argument("--training-arm", choices=["landmark", "dynamic"])
    ap.add_argument("--model-horizon", type=int,
                    help="required to select a horizon-specific checklist model")
    ap.add_argument("--age-col", default="age_at_anchor")
    ap.add_argument("--id-col", default="person_id")
    ap.add_argument(
        "--horizons", default="",
        help="comma-separated days, e.g. 365,730 (elastic-net, dynamic_deephit)"
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    bundle = load_bundle(args.bundle)
    model = select_model(bundle, args.landmark, args.config,
                         training_arm=args.training_arm, horizon_days=args.model_horizon)
    df = _read_table(args.data)
    horizons = [float(h) for h in args.horizons.split(",") if h.strip()]

    if bundle.get("model_family") == DEEPHIT_FAMILY:
        cohort = _read_table(args.cohort) if args.cohort else None
        ids = (
            pd.unique(cohort[args.id_col]).tolist() if cohort is not None else None
        )
        ids, cif = predict_cif(
            df, model, ids=ids, id_col=args.id_col, age_col=args.age_col,
            cohort=cohort,
        )
        out = pd.DataFrame({args.id_col: ids, "risk_score": cif[:, -1]})
        out["landmark_days"] = args.landmark
        out["config"] = args.config
        out["model_family"] = DEEPHIT_FAMILY
        if "training_arm" in model:
            out["training_arm"] = model_identity(bundle, model)[2]
        if horizons:
            out = pd.concat([out, deephit_survival(cif, model, horizons)], axis=1)
        out.to_csv(args.output, index=False)
        print(f"[federated_inference] wrote {len(out):,} rows -> {args.output}")
        return 0

    if bundle.get("model_family") == "predictive_checklist":
        pred = predict_checklist(df, model)
        risk = pred["score"].to_numpy(dtype=float)
    else:
        pred = None
        risk = predict_risk(df, bundle, model, age_col=args.age_col)
    out = pd.DataFrame({"risk_score": risk}, index=df.index)
    # The identifier is carried through for joining only; it is never a
    # predictor, and build_design_matrix ignores it.
    if args.id_col in df.columns:
        out.insert(0, args.id_col, df[args.id_col].to_numpy())
    out["landmark_days"] = args.landmark
    out["config"] = args.config
    out["model_family"] = bundle.get("model_family")
    if model.get("horizon_days") is not None:
        out["model_horizon_days"] = model_identity(bundle, model)[3]
    if pred is not None:
        out["probability"] = pred["probability"].to_numpy()
        out["flag"] = pred["flag"].to_numpy()

    if horizons and pred is None:
        surv = predict_survival(risk, model, horizons)
        out = pd.concat([out, surv.set_index(out.index)], axis=1)

    out.to_csv(args.output, index=False)
    print(f"[federated_inference] wrote {len(out):,} rows -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
