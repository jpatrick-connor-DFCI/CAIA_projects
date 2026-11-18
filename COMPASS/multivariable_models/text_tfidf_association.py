"""Direct patient-level TF-IDF comparison between high- and low-risk groups.

The shared vocabulary and IDF are fitted without group labels at each landmark.
Each patient contributes once. This analysis uses no permutations or bootstrap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu


RESULT_COLUMNS = [
    "landmark", "term", "n_high", "n_low", "high_with_term", "low_with_term",
    "prevalence_high", "prevalence_low", "mean_tfidf_high", "mean_tfidf_low",
    "difference_high_minus_low", "auc_high_vs_low", "p_value", "q_value", "status",
]


def compare_tfidf_groups(tfidf, ids, terms, groups, landmark, *, min_group_patients=10, block_size=256):
    """Compare patient TF-IDF weights for groups labelled 'top' and 'bottom'.

    Return mean weights, high-minus-low difference, and Mann-Whitney probability
    of superiority (ties count half). P-values are two-sided, asymptotic, tie- and
    continuity-corrected. Small groups retain descriptive summaries but no test.
    Other group labels are excluded. Apply BH after combining landmarks.
    """
    if min_group_patients < 2 or block_size < 1:
        raise ValueError("Require min_group_patients >= 2 and positive block_size.")
    ids = pd.Index(ids)
    if ids.has_duplicates or groups.index.has_duplicates:
        raise ValueError("Expected one row per patient.")
    weights = sparse.csc_matrix(tfidf, dtype=float)
    if weights.shape != (len(ids), len(terms)):
        raise ValueError("TF-IDF shape must match patient IDs and terms.")
    if not np.isfinite(weights.data).all() or np.any(weights.data < 0):
        raise ValueError("TF-IDF weights must be finite and nonnegative.")
    labels = groups.reindex(ids)
    if labels.isna().any():
        raise ValueError("Missing risk group for a TF-IDF patient.")
    high, low = labels.eq("top").to_numpy(), labels.eq("bottom").to_numpy()
    nh, nl = int(high.sum()), int(low.sum())
    frames = []
    for start in range(0, len(terms), block_size):
        h = weights[high, start:start + block_size].toarray()
        l = weights[low, start:start + block_size].toarray()
        size = h.shape[1]
        mean_h = h.mean(axis=0) if nh else np.full(size, np.nan)
        mean_l = l.mean(axis=0) if nl else np.full(size, np.nan)
        count_h, count_l = (h > 0).sum(axis=0), (l > 0).sum(axis=0)
        auc, p = np.full(size, np.nan), np.full(size, np.nan)
        status = np.full(size, "skipped: too few patients in a group", dtype=object)
        if nh and nl:
            test = mannwhitneyu(h, l, axis=0, alternative="two-sided", method="asymptotic", use_continuity=True)
            auc = test.statistic / (nh * nl)
            if min(nh, nl) >= min_group_patients:
                p = test.pvalue
                status[:] = "ok"
        frames.append(pd.DataFrame(dict(
            landmark=landmark, term=terms[start:start + size], n_high=nh, n_low=nl,
            high_with_term=count_h, low_with_term=count_l,
            prevalence_high=count_h / nh if nh else np.full(size, np.nan),
            prevalence_low=count_l / nl if nl else np.full(size, np.nan),
            mean_tfidf_high=mean_h, mean_tfidf_low=mean_l,
            difference_high_minus_low=mean_h - mean_l, auc_high_vs_low=auc, p_value=p, status=status,
        )))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).reindex(columns=RESULT_COLUMNS)


def adjust_fdr(frames):
    """BH across all tested terms and landmarks in this notebook/timing folder."""
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS)
    p = result["p_value"].to_numpy(dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    q = np.full(len(p), np.nan)
    q[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    result["q_value"] = q
    return result.reindex(columns=RESULT_COLUMNS)
