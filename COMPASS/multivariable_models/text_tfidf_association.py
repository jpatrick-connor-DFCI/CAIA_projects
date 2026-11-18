"""Patient-level continuous TF-IDF/risk associations for text notebooks 13/14.

Weights are already fitted within a landmark, without risk labels. Resampling
conditions on that vocabulary and IDF: whole patient vectors stay together.
Only aggregate results are returned. Dense computation is bounded by term blocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, t


RESULT_COLUMNS = [
    "landmark", "term", "method", "n_patients", "n_with_term", "prevalence",
    "mean_tfidf", "risk_mean", "risk_sd", "spearman_rho", "beta_per_sd",
    "ci95_low", "ci95_high", "p_value", "q_value", "n_permutations",
    "n_bootstrap_valid", "adjustment_covariates", "status", "ci_status",
]


def patient_covariates(reports, ids, *, scope):
    """Count plus specimen indicators (pathology) or note-type fractions (pooled).

    Each patient contributes once. Fractions describe note composition rather
    than introducing one observation per note; the first sorted type is reference.
    """
    counts = reports.groupby("DFCI_MRN").size().reindex(ids)
    if counts.isna().any() or counts.le(0).any():
        raise ValueError("Every TF-IDF patient must have at least one eligible report.")
    cov = pd.DataFrame({"log_report_count": np.log1p(counts)}, index=ids)
    if scope == "pathology":
        for site in ("non-prostate", "unspecified"):
            cov["any_" + site.replace("-", "_")] = (
                reports.assign(hit=reports["specimen_site"].eq(site))
                .groupby("DFCI_MRN")["hit"].any().reindex(ids).astype(float)
            )
    elif scope == "clinical_notes":
        for kind in sorted(reports["NOTE_TYPE"].dropna().unique())[1:]:
            cov["fraction_" + kind] = (
                reports[reports["NOTE_TYPE"].eq(kind)].groupby("DFCI_MRN").size()
                .reindex(ids, fill_value=0) / counts
            )
    else:
        raise ValueError("scope must be pathology or clinical_notes")
    return cov


def _unit_ranks(values, axis=0):
    ranks = rankdata(values, axis=axis, method="average")
    centered = ranks - ranks.mean(axis=axis, keepdims=True)
    length = np.linalg.norm(centered, axis=axis, keepdims=True)
    return np.divide(centered, length, out=np.full_like(centered, np.nan), where=length > 0)


def _spearman_uncertainty(x, y, rho, *, n_permutations, n_bootstrap, seed):
    """Shared patient permutations/paired bootstrap draws across all term blocks."""
    xr, yr = _unit_ranks(x), _unit_ranks(y)
    rng = np.random.default_rng(seed)
    extreme = np.zeros(y.shape[1], dtype=np.int64)
    for start in range(0, n_permutations, 128):
        permutations = np.array([rng.permutation(xr) for _ in range(min(128, n_permutations - start))])
        null = permutations @ yr
        extreme += (np.abs(null) >= np.maximum(0, np.abs(rho) - 1e-14)).sum(axis=0)
    p = (extreme + 1) / (n_permutations + 1)

    # Re-rank after sampling: simply resampling the original ranks is not Spearman.
    rng = np.random.default_rng(seed + 1)
    boot = np.full((n_bootstrap, y.shape[1]), np.nan)
    for b in range(n_bootstrap):
        rows = rng.integers(0, len(x), size=len(x))
        boot[b] = _unit_ranks(x[rows]) @ _unit_ranks(y[rows])
    valid = np.isfinite(boot).sum(axis=0)
    lo, hi = np.full(y.shape[1], np.nan), np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        if valid[j] >= max(2, int(np.ceil(0.8 * n_bootstrap))):
            lo[j], hi[j] = np.quantile(boot[np.isfinite(boot[:, j]), j], [0.025, 0.975])
    return p, lo, hi, valid


def _adjusted_design(risk, covariates):
    """Prepare one OLS design and HC3 leverage correction per landmark."""
    covariates = covariates.loc[:, covariates.nunique() > 1]
    names = ";".join(map(str, covariates.columns))
    x = np.column_stack([np.ones(len(risk)), (risk - risk.mean()) / risk.std(ddof=0), covariates])
    if len(risk) <= x.shape[1]:
        return None, names, "skipped: insufficient residual degrees of freedom"
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return None, names, "skipped: collinear adjusted design"
    pinv = np.linalg.pinv(x)
    leverage = np.einsum("ij,ji->i", x, pinv)
    if np.any(1 - leverage <= 1e-10):
        return None, names, "skipped: unstable adjusted leverage"
    return (x, pinv, leverage), names, "ok"


def continuous_tfidf_association(
    tfidf, ids, terms, risk, covariates, landmark, *, min_patients=30,
    min_term_patients=5, n_permutations=9999, n_bootstrap=1000, seed=42, block_size=64,
):
    """Spearman independence test and adjusted OLS with HC3/t inference per term.

    Inputs: sparse patient-by-term weights, matching patient IDs and term names,
    and patient-indexed raw risk/covariates. Invalid risks are excluded; missing
    covariates are rejected. Constant covariates are omitted, collinear designs
    skipped. Bootstrap intervals are percentile intervals for Spearman rho.
    FDR is applied separately after combining landmarks via ``adjust_fdr``.
    """
    if n_permutations < 1 or n_bootstrap < 2 or block_size < 1:
        raise ValueError("Require positive permutations/block size and at least two bootstrap draws.")
    if min_patients < 3 or min_term_patients < 1:
        raise ValueError("Require min_patients >= 3 and min_term_patients >= 1.")
    ids = pd.Index(ids)
    if ids.has_duplicates or risk.index.has_duplicates or covariates.index.has_duplicates:
        raise ValueError("Expected one row per patient.")
    weights = sparse.csc_matrix(tfidf, dtype=float)
    if weights.shape != (len(ids), len(terms)):
        raise ValueError("TF-IDF shape must match patient IDs and terms.")
    if not np.isfinite(weights.data).all() or np.any(weights.data < 0):
        raise ValueError("TF-IDF weights must be finite and nonnegative.")
    scores = pd.to_numeric(risk.reindex(ids), errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(scores)
    scores, ids, weights = scores[keep], ids[keep], weights[keep].tocsc()
    cov = covariates.reindex(ids).astype(float)
    if not np.isfinite(cov.to_numpy()).all():
        raise ValueError("Missing or nonfinite adjustment covariates.")
    n = len(ids)
    mean, sd = (scores.mean(), scores.std(ddof=0)) if n else (np.nan, np.nan)
    common_reason = "skipped: too few patients" if n < min_patients else None
    if common_reason is None and (not np.isfinite(sd) or sd <= 0):
        common_reason = "skipped: constant risk score"
    design, cov_names, design_status = (None, "", common_reason)
    if common_reason is None:
        design, cov_names, design_status = _adjusted_design(scores, cov)

    rows = []
    for start in range(0, len(terms), block_size):
        y = weights[:, start:start + block_size].toarray()
        present = (y > 0).sum(axis=0)
        eligible = (present >= min_term_patients) & (np.ptp(y, axis=0) > 0) if n else np.zeros(y.shape[1], bool)
        rho = np.full(y.shape[1], np.nan)
        p, low, high = (rho.copy() for _ in range(3))
        valid = np.zeros(y.shape[1], dtype=int)
        beta, se, p_adj = (rho.copy() for _ in range(3))
        if common_reason is None and eligible.any():
            tested = y[:, eligible]
            rho[eligible] = np.clip(_unit_ranks(scores) @ _unit_ranks(tested), -1, 1)
            p[eligible], low[eligible], high[eligible], valid[eligible] = _spearman_uncertainty(
                scores, tested, rho[eligible], n_permutations=n_permutations,
                n_bootstrap=n_bootstrap, seed=seed,
            )
            if design is not None:
                x, pinv, leverage = design
                coefficients = pinv @ tested
                residual = tested - x @ coefficients
                beta[eligible] = coefficients[1]
                se[eligible] = np.sqrt(np.sum((pinv[1, :, None] * residual / (1 - leverage[:, None])) ** 2, axis=0))
                good = eligible & np.isfinite(se) & (se > 0)
                p_adj[good] = 2 * t.sf(np.abs(beta[good] / se[good]), df=n - x.shape[1])
        for j in range(y.shape[1]):
            reason = common_reason
            if reason is None and present[j] < min_term_patients:
                reason = "skipped: too few patients with term"
            if reason is None and not eligible[j]:
                reason = "skipped: constant TF-IDF weight"
            base = dict(landmark=landmark, term=terms[start + j], n_patients=n,
                        n_with_term=int(present[j]), prevalence=present[j] / n if n else np.nan,
                        mean_tfidf=y[:, j].mean() if n else np.nan, risk_mean=mean, risk_sd=sd)
            ci_status = "not estimated" if reason else (
                "ok" if np.isfinite(low[j]) else "too few nonconstant bootstrap samples")
            rows.append(dict(base, method="spearman", spearman_rho=rho[j], ci95_low=low[j], ci95_high=high[j],
                             p_value=p[j], n_permutations=n_permutations if reason is None else 0,
                             n_bootstrap_valid=int(valid[j]), status=reason or "ok", ci_status=ci_status,
                             adjustment_covariates=""))
            status = reason or design_status
            if status == "ok" and not np.isfinite(p_adj[j]):
                status = "skipped: invalid adjusted covariance"
            width = t.ppf(0.975, n - design[0].shape[1]) * se[j] if status == "ok" else np.nan
            rows.append(dict(base, method="ols_adjusted", beta_per_sd=beta[j] if status == "ok" else np.nan,
                             ci95_low=beta[j] - width, ci95_high=beta[j] + width,
                             p_value=p_adj[j] if status == "ok" else np.nan, adjustment_covariates=cov_names,
                             status=status, ci_status="ok" if status == "ok" else "not estimated"))
    return pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)


def adjust_fdr(frames):
    """BH across all terms and landmarks, separately per method; skipped rows stay NaN."""
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS)
    for _, indices in result.groupby("method").groups.items():
        p = result.loc[indices, "p_value"].to_numpy(dtype=float)
        valid = np.flatnonzero(np.isfinite(p))
        order = valid[np.argsort(p[valid])]
        q = np.full(len(p), np.nan)
        q[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
        result.loc[indices, "q_value"] = q
    return result.reindex(columns=RESULT_COLUMNS)
