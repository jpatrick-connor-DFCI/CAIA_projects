"""Score a federated CAIA model bundle on local data.

The federated runs (caia-project-compass/rhino_scripts) fit XGBoost and
elastic-net Cox models across sites and export each fitted model as a
``caia-federated-model-bundle`` JSON file.  This module loads such a bundle and
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

Loading the trained object in Python::

    from survival_common.federated_inference import load_fitted_model

    booster, bundle, model = load_fitted_model("xgboost_federated_model_adt.json", 90)
    booster.trees_to_dataframe()          # a real xgboost.Booster

    cox, bundle, model = load_fitted_model("cox_federated_elasticnet_model_adt.json", 90)
    cox.params_                           # coefficients by covariate name
    cox.predict(df, age_col="age_at_anchor")

CLI::

    python -m survival_common.federated_inference \\
        --bundle xgboost_federated_model_adt.json \\
        --data cohort_wide.parquet \\
        --landmark 90 \\
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
SUPPORTED_FORMAT_VERSIONS = (1,)


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
    return bundle


def select_model(bundle: dict, landmark_days: int, config: str = "both") -> dict:
    """Pick one fitted model out of a bundle.

    ``config`` is ``both`` for the full model or ``baseline`` for the age-only
    comparator, matching the federated result tables.
    """
    wanted = [
        m for m in bundle["models"]
        if int(m["landmark_days"]) == int(landmark_days)
        and str(m.get("config")) == str(config)
    ]
    if not wanted:
        available = sorted(
            {(int(m["landmark_days"]), str(m.get("config"))) for m in bundle["models"]}
        )
        raise BundleError(
            f"no model for landmark={landmark_days} config={config!r}; "
            f"bundle has {available}"
        )
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


def load_fitted_model(path, landmark_days: int, config: str = "both"):
    """Open a bundle and hand back ``(estimator, bundle, model)`` in one call.

    The usual entry point when you want the trained object::

        booster, bundle, model = load_fitted_model("xgboost_..._adt.json", 90)
        booster.trees_to_dataframe()

    ``bundle`` and ``model`` come back too because the estimator alone cannot
    score raw data -- the preprocessing lives in ``model``.
    """
    bundle = load_bundle(path)
    model = select_model(bundle, landmark_days, config)
    return load_model(bundle, model), bundle, model


def load_model(bundle: dict, model: dict):
    """Return the trained estimator itself, ready to use.

    For ``xgboost_cox`` this is a real :class:`xgboost.Booster` -- the same
    object the federated run trained, restored from the bundle's JSON.  Every
    Booster method works on it: ``predict``, ``save_model``, ``trees_to_dataframe``,
    ``get_score``, SHAP via ``pred_contribs=True``, ``xgboost.plot_tree``.

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
# Scoring
# --------------------------------------------------------------------- #
def predict_risk(
    df: pd.DataFrame,
    bundle: dict,
    model: dict,
    *,
    age_col: str | None = None,
) -> np.ndarray:
    """Linear predictor / log-hazard-ratio for each row.

    Higher is higher risk, for both families.  This is on the same scale the
    federated run used to compute its C-index, so local and federated
    discrimination are directly comparable.
    """
    design = build_design_matrix(df, model, age_col=age_col)
    family = bundle.get("model_family")

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
    """
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
    ap.add_argument("--data", required=True, help="wide frame (.csv or .parquet)")
    ap.add_argument("--landmark", type=int, required=True)
    ap.add_argument("--config", default="both", choices=["both", "baseline"])
    ap.add_argument("--age-col", default="age_at_anchor")
    ap.add_argument("--id-col", default="person_id")
    ap.add_argument(
        "--horizons", default="", help="comma-separated days, e.g. 365,730 (elastic-net)"
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    bundle = load_bundle(args.bundle)
    model = select_model(bundle, args.landmark, args.config)
    df = _read_table(args.data)

    risk = predict_risk(df, bundle, model, age_col=args.age_col)
    out = pd.DataFrame({"risk_score": risk}, index=df.index)
    # The identifier is carried through for joining only; it is never a
    # predictor, and build_design_matrix ignores it.
    if args.id_col in df.columns:
        out.insert(0, args.id_col, df[args.id_col].to_numpy())
    out["landmark_days"] = args.landmark
    out["config"] = args.config
    out["model_family"] = bundle.get("model_family")

    if args.horizons.strip():
        horizons = [float(h) for h in args.horizons.split(",") if h.strip()]
        surv = predict_survival(risk, model, horizons)
        out = pd.concat([out, surv.set_index(out.index)], axis=1)

    out.to_csv(args.output, index=False)
    print(f"[federated_inference] wrote {len(out):,} rows -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
