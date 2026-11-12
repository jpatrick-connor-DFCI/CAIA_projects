"""Local test: does restricting platinum training to the metastatic cohort help NEPC transfer?

Compares nested-CV out-of-fold (``cv_oof``) XGBoost risk scores from platinum
fits trained on the metastatic ADT cohort (arm A) and on the full ADT cohort
(arms B/C), plus optionally the frozen federated no-DFCI XGBoost (arm F) and
endpoint-trained fits (arm N, the local ceiling).  Outcomes come from the same
chain the federated transfer uses (``federated_endpoint_transfer.to_federated_long``
+ ``federated_scoring.build_landmark_frame``), so numbers are comparable to
notebook 04.  Arms in one comparison are cut to their shared patients and share
bootstrap resamples, so deltas are paired. Optional federated comparisons are
reported separately and cannot change the local comparison's patients. Only
``cv_oof`` rows are used from local risk files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from survival_common import federated_endpoint_transfer as fet
from survival_common import federated_scoring as fs

ID_COL = fet.LOCAL_ID_COL  # "DFCI_MRN" in the risk files
BASELINE_SUFFIX = "_baseline"


def load_oof_scores(path, endpoint: str, id_col: str = ID_COL) -> pd.Series:
    """Held-out nested-CV scores (``dataset == "cv_oof"``) indexed by patient id."""
    remediation = (
        f"{path}: refit this {endpoint} run with OUT_OF_FOLD_RISKS=True and "
        "FORCE_RERUN=True, then rerun the transfer evaluation."
    )
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError as exc:
        raise ValueError(f"Missing local risk file. {remediation}") from exc
    required = {"dataset", "risk_score", id_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing OOF columns {sorted(required - set(df.columns))}. {remediation}")
    if "endpoint" in df.columns:
        df = df[df["endpoint"] == endpoint]
    df = df[df["dataset"].astype(str) == "cv_oof"]
    if df.empty:
        raise ValueError(f"No cv_oof scores for {endpoint}. {remediation}")
    risk = pd.to_numeric(df["risk_score"], errors="coerce")
    if not np.isfinite(risk.to_numpy(dtype=float)).all():
        raise ValueError(f"Incomplete OOF coverage: non-finite risk scores. {remediation}")
    out = pd.Series(risk.to_numpy(), index=df[id_col].to_numpy(), name="risk")
    return out[~out.index.duplicated(keep="first")]


def load_federated_scores(path, landmark: int, endpoint: str,
                          family: str = "xgboost_cox") -> dict[str, pd.Series]:
    """Arm F: frozen federated bundle scores from notebook 04's transfer_scores.csv."""
    df = pd.read_csv(path, low_memory=False)
    df = df[(df["model_family"] == family) & (df["landmark_days"] == landmark)
            & (df["endpoint"] == endpoint)]
    out = {}
    for config, name in (("both", "F"), ("baseline", "F" + BASELINE_SUFFIX)):
        sub = df[df["config"] == config]
        if len(sub):
            out[name] = pd.Series(sub["risk_score"].to_numpy(),
                                  index=sub[fs.ID_COL].to_numpy())
    return out


def endpoint_outcomes(local: pd.DataFrame, endpoint: str, landmark: int,
                      lab_mapping: dict) -> pd.DataFrame:
    """Incident (id, duration_days, event) at one landmark, as the federated transfer builds it."""
    long_df, _ = fet.to_federated_long(local, endpoint, lab_mapping)
    wide, _ = fs.build_landmark_frame(long_df, landmark)
    return pd.DataFrame({
        "id": wide[fs.ID_COL].to_numpy(),
        "duration": wide[fs.DURATION_COL].to_numpy(dtype=float),
        "event": wide[fs.EVENT_COL].to_numpy(dtype=int),
    })


def compare(outcome: pd.DataFrame, arms: dict[str, pd.Series], *, ref: str | None = None,
            n_bootstrap: int = 200, seed: int = 0) -> pd.DataFrame:
    """Score every arm on the patients all arms share; one row per arm.

    ``<arm>_baseline`` entries are scored too and give ``delta_c_vs_baseline``;
    ``ref`` names the arm every other non-baseline arm is differenced against
    (``delta_c_vs_ref``).  Paired bootstrap CIs throughout.
    """
    ids = set(outcome["id"])
    for s in arms.values():
        ids &= set(s.index)
    ids = sorted(ids)
    cols = ["arm", "n", "events", "c_index", "c_lo", "c_hi", "mean_auc", "auc_lo", "auc_hi",
            "delta_c_vs_baseline", "delta_c_vs_baseline_lo", "delta_c_vs_baseline_hi",
            "delta_c_vs_ref", "delta_c_vs_ref_lo", "delta_c_vs_ref_hi"]
    if not ids:
        return pd.DataFrame(columns=cols)
    o = outcome.set_index("id").loc[ids]
    dur, ev = o["duration"].to_numpy(), o["event"].to_numpy()
    risks = {k: s.loc[ids].to_numpy(dtype=float) for k, s in arms.items()}
    grid = fet.auc_horizon_grid(dur, ev)
    boot_c = fet.bootstrap_c(dur, ev, risks, n_bootstrap, seed) if n_bootstrap else {}
    boot_a = fet.bootstrap_mean_auc(dur, ev, risks, grid, n_bootstrap, seed) if n_bootstrap else {}

    def diff(a, b):
        point = fet.harrell_c(dur, ev, risks[a]) - fet.harrell_c(dur, ev, risks[b])
        if a not in boot_c or b not in boot_c:
            return point, np.nan, np.nan
        lo, hi = fet._ci(boot_c[a] - boot_c[b])
        return point, lo, hi

    rows = []
    for name, r in risks.items():
        row = {"arm": name, "n": len(ids), "events": int(ev.sum()),
               "c_index": fet.harrell_c(dur, ev, r),
               "mean_auc": fet.mean_auc(dur, ev, r, grid)}
        if name in boot_c:
            row["c_lo"], row["c_hi"] = fet._ci(boot_c[name])
            row["auc_lo"], row["auc_hi"] = fet._ci(boot_a[name])
        base = name + BASELINE_SUFFIX
        if not name.endswith(BASELINE_SUFFIX) and base in risks:
            (row["delta_c_vs_baseline"], row["delta_c_vs_baseline_lo"],
             row["delta_c_vs_baseline_hi"]) = diff(name, base)
        if ref and name != ref and not name.endswith(BASELINE_SUFFIX) and ref in risks:
            (row["delta_c_vs_ref"], row["delta_c_vs_ref_lo"],
             row["delta_c_vs_ref_hi"]) = diff(name, ref)
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=cols)


def evaluate_metastatic_transfer(
    local: pd.DataFrame, lab_mapping: dict, *, landmarks, endpoints,
    met_paths: dict, all_paths: dict, federated_scores_path=None,
    ceiling_paths: dict | None = None, n_bootstrap: int = 200, seed: int = 0,
) -> pd.DataFrame:
    """Run arms A/B/C/F (and N) for every endpoint x landmark.

    ``met_paths`` / ``all_paths``: ``{(landmark, config): risk-file path}`` for the
    platinum fit on the metastatic / full cohort (config "both" or "baseline").
    ``ceiling_paths``: ``{(cohort, landmark, config): path}`` for endpoint-trained
    NEPC fits, scored on their own endpoint only.

    ``comparison`` separates local A/B and C results from optional federated
    comparisons on their own shared patients and the NEPC ceiling results.
    """
    tables = []

    def add(df, **tags):
        if len(df):
            tables.append(df.assign(**tags))

    for lm in landmarks:
        def scores(paths, ep="platinum"):
            return {("" if cfg == "both" else BASELINE_SUFFIX): load_oof_scores(p, ep)
                    for (l, cfg), p in paths.items() if l == lm}
        a = {"A" + k: v for k, v in scores(met_paths).items()}
        b = {"B" + k: v for k, v in scores(all_paths).items()}
        if "A" not in a or "B" not in b:
            raise ValueError(f"Missing required 'both' OOF risk paths for landmark {lm}.")
        c = {"C" + k.removeprefix("B"): v for k, v in b.items()}
        met_ids = set(a["A"].index)
        for ep in endpoints:
            out = endpoint_outcomes(local, ep, lm, lab_mapping)
            fed = (load_federated_scores(federated_scores_path, lm, ep)
                   if federated_scores_path else {})
            n_met = int(out["id"].isin(met_ids).sum())
            n_met_events = int(out.loc[out["id"].isin(met_ids), "event"].sum())
            tag = dict(endpoint=ep, landmark=lm, n_endpoint_cohort=len(out),
                       n_in_met_cohort=n_met, events_in_met_cohort=n_met_events)
            # Primary local comparisons must not depend on optional F coverage.
            met_out = out[out["id"].isin(met_ids)]
            primary = compare(met_out, {**a, **b}, ref="B", n_bootstrap=n_bootstrap, seed=seed)
            if primary.empty:
                raise ValueError(f"No shared local OOF patients for {ep} at landmark {lm}.")
            add(primary, population="metastatic", comparison="local", **tag)
            add(compare(out, c, n_bootstrap=n_bootstrap, seed=seed),
                population="full", comparison="local", **tag)
            if fed:
                add(compare(met_out, {**a, **b, **fed}, ref="F",
                            n_bootstrap=n_bootstrap, seed=seed),
                    population="metastatic", comparison="federated", **tag)
                add(compare(out, {**c, **fed}, ref="F",
                            n_bootstrap=n_bootstrap, seed=seed),
                    population="full", comparison="federated", **tag)
    if ceiling_paths:
        for cohort in sorted({c for c, _, _ in ceiling_paths}):
            for lm in landmarks:
                n_arms = {("N" if cfg == "both" else "N" + BASELINE_SUFFIX):
                          load_oof_scores(p, "nepc")
                          for (c, l, cfg), p in ceiling_paths.items() if c == cohort and l == lm}
                if not n_arms:
                    continue
                out = endpoint_outcomes(local, "nepc", lm, lab_mapping)
                add(compare(out, n_arms, n_bootstrap=n_bootstrap, seed=seed),
                    endpoint="nepc", landmark=lm, population=f"{cohort} (own cohort)",
                    comparison="ceiling",
                    n_endpoint_cohort=len(out), n_in_met_cohort=np.nan, events_in_met_cohort=np.nan)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
