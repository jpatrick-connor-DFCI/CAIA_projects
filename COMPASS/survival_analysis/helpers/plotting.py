"""Shared plotting utilities for COMPASS survival-analysis figures.

Used by both PROFILE and CAIA pipelines (`generate_figures.ipynb`) and by
`CAIA/summary_figures.py`. Centralises:

  * `RCPARAMS`            - matplotlib rcParams shared across all figures
  * lab clinical category sets + `assign_category()` / `parse_feature()`
  * `CATEGORY_COLORS`     - per-category hex palette
  * `irnt()`              - inverse rank-normal transform (Blom)
  * `overlay_hist()`      - PROFILE-vs-CAIA overlay histograms
  * `overlay_km()`        - PROFILE-vs-CAIA Kaplan-Meier overlay
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm


RCPARAMS: dict = {
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


CBC = {
    "WBC", "RBC", "Hemoglobin", "Hematocrit",
    "MCV", "MCH", "MCHC", "RDW", "Platelets",
    "Neutrophils absolute", "Lymphocytes absolute",
    "Monocytes absolute", "Eosinophils absolute", "Basophils absolute",
}
CMP = {"Sodium", "Potassium", "Chloride", "CO2",
       "BUN", "Creatinine", "Glucose", "Calcium"}
LFT = {"ALT", "AST", "Alkaline phosphatase",
       "Total bilirubin", "Direct bilirubin",
       "Albumin", "Globulin", "Total protein", "PT"}
VITALS = {"Body weight", "Body temperature", "Heart rate",
          "Respiratory rate", "Systolic blood pressure",
          "Diastolic blood pressure"}
ANDROGEN = {"PSA", "Testosterone"}
OTHER = {"TSH"}
DROP = {"Body height"}

CATEGORY_MAP: dict[str, str] = {}
for _label, _members in [
    ("CBC", CBC), ("CMP", CMP), ("LFT", LFT),
    ("Vitals", VITALS), ("Androgen axis", ANDROGEN), ("Other", OTHER),
]:
    for _m in _members:
        CATEGORY_MAP[_m] = _label

DRAW_ORDER = ["Other", "Vitals", "CMP", "LFT", "CBC", "Androgen axis"]
LEGEND_ORDER = ["Androgen axis", "CBC", "LFT", "CMP", "Vitals", "Other"]

CATEGORY_COLORS = {
    "Androgen axis": "#8e1c2b",
    "CBC":           "#16a085",
    "LFT":           "#e67e22",
    "CMP":           "#7d3c98",
    "Vitals":        "#5d6d7e",
    "Other":         "#95a5a6",
}
NS_COLOR = "#d5d8dc"

COHORT_COLORS = {
    "PROFILE": "#1f3a93",
    "CAIA":    "#c0392b",
}


# CAIA / LOINC-style long lab names -> short PROFILE-style names.
# Used by `canonicalize_lab_name()` so CATEGORY_MAP lookups, IRNT panels, and
# overlay comparisons across cohorts agree on a single label per analyte.
# Only includes the LOINC-style names actually observed in the CAIA parquet
# (54 unique lab names as of 2026-05-26); extend as new labs appear.
LAB_ALIASES: dict[str, str] = {
    # CBC
    "Leukocytes [#/volume] in Blood by Automated count": "WBC",
    "Erythrocytes [#/volume] in Blood by Automated count": "RBC",
    "Hemoglobin [Mass/volume] in Blood": "Hemoglobin",
    "Hematocrit [Volume Fraction] of Blood by Automated count": "Hematocrit",
    "MCV [Entitic Volume] by Automated count": "MCV",
    "MCH [Entitic Mass] by Automated count": "MCH",
    "MCHC [Entitic Mass/volume] in Red Blood Cells by Automated count": "MCHC",
    "Erythrocyte distribution width [Ratio] by Automated count": "RDW",
    "Platelets [#/volume] in Blood by Automated count": "Platelets",
    "Neutrophils [#/volume] in Blood by Automated count": "Neutrophils absolute",
    "Lymphocytes [#/volume] in Blood by Automated count": "Lymphocytes absolute",
    "Monocytes [#/volume] in Blood by Automated count": "Monocytes absolute",
    "Eosinophils [#/volume] in Blood by Automated count": "Eosinophils absolute",
    "Basophils [#/volume] in Blood by Automated count": "Basophils absolute",
    # CMP
    "Sodium [Moles/volume] in Serum or Plasma": "Sodium",
    "Potassium [Moles/volume] in Serum or Plasma": "Potassium",
    "Chloride [Moles/volume] in Serum or Plasma": "Chloride",
    "Carbon dioxide, total [Moles/volume] in Serum or Plasma": "CO2",
    "Urea nitrogen [Mass/volume] in Serum or Plasma": "BUN",
    "Creatinine [Mass/volume] in Serum or Plasma": "Creatinine",
    "Glucose [Mass/volume] in Serum or Plasma": "Glucose",
    "Calcium [Mass/volume] in Serum or Plasma": "Calcium",
    # LFT
    "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma": "ALT",
    "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma": "AST",
    "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma": "Alkaline phosphatase",
    "Bilirubin.total [Mass/volume] in Serum or Plasma": "Total bilirubin",
    "Bilirubin.direct [Mass/volume] in Serum or Plasma": "Direct bilirubin",
    "Albumin [Mass/volume] in Serum or Plasma": "Albumin",
    "Globulin [Mass/volume] in Serum": "Globulin",
    "Protein [Mass/volume] in Serum or Plasma": "Total protein",
    "Prothrombin time (PT)": "PT",
    # Vitals (already short in CAIA)
    "Body weight": "Body weight",
    "Body temperature": "Body temperature",
    "Heart rate": "Heart rate",
    "Respiratory rate": "Respiratory rate",
    "Systolic blood pressure": "Systolic blood pressure",
    "Diastolic blood pressure": "Diastolic blood pressure",
    "Body height": "Body height",
    # Other
    "Thyrotropin [Units/volume] in Serum or Plasma": "TSH",
}


def canonicalize_lab_name(lab_name: str) -> str:
    """Map a possibly-LOINC-style lab name to the short PROFILE-style name.
    Returns the input unchanged if no alias is registered.
    """
    return LAB_ALIASES.get(lab_name, lab_name)


def assign_category(lab_name: str) -> str:
    return CATEGORY_MAP.get(canonicalize_lab_name(lab_name), "Other")


def parse_feature(name: str) -> tuple[str, str]:
    """Split 'LAB_NAME__stat' into (lab_name, stat). Returns (name, '') if no '__'."""
    if "__" in name:
        lab, stat = name.split("__", 1)
        return lab, stat
    return name, ""


def irnt(series: pd.Series) -> pd.Series:
    """Inverse rank-normal transform (Blom).

    NaNs are preserved. Ties get the average rank. Output is the standard-normal
    quantile of (rank - 0.5) / n_nonnull, which puts each observation at the
    mid-quantile of its rank bucket and keeps the transformed series finite.
    """
    s = pd.Series(series, copy=True).astype(float)
    nonnull = s.notna()
    n = int(nonnull.sum())
    out = pd.Series(np.full(len(s), np.nan), index=s.index, dtype=float)
    if n == 0:
        return out
    ranks = s.rank(method="average", na_option="keep")
    out.loc[nonnull] = norm.ppf((ranks.loc[nonnull] - 0.5) / n)
    return out


def overlay_hist(
    ax,
    series_by_label: Mapping[str, pd.Series],
    *,
    bins: int | np.ndarray = 50,
    density: bool = True,
    alpha: float = 0.55,
    colors: Mapping[str, str] | None = None,
) -> None:
    """Overlay semi-transparent histograms for multiple labelled series on `ax`."""
    colors = dict(colors) if colors else dict(COHORT_COLORS)
    finite_values = []
    for s in series_by_label.values():
        v = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
        v = v[np.isfinite(v)]
        finite_values.append(v)
    if isinstance(bins, int) and finite_values:
        combined = np.concatenate(finite_values) if finite_values else np.array([])
        if combined.size:
            lo, hi = np.percentile(combined, [0.5, 99.5])
            if hi <= lo:
                lo, hi = combined.min(), combined.max() + 1.0
            bins = np.linspace(lo, hi, bins + 1)
    for (label, _s), v in zip(series_by_label.items(), finite_values):
        if v.size == 0:
            continue
        color = colors.get(label, None)
        ax.hist(
            v,
            bins=bins,
            density=density,
            alpha=alpha,
            color=color,
            label=f"{label} (n={v.size:,})",
            edgecolor="white",
            linewidth=0.3,
        )
    ax.legend(fontsize=9, loc="best")


def overlay_km(
    ax,
    survival_by_label: Mapping[str, tuple[pd.Series, pd.Series]],
    *,
    colors: Mapping[str, str] | None = None,
    title: str | None = None,
    xlabel: str = "Days",
    ylabel: str = "Survival probability",
) -> None:
    """Overlay Kaplan-Meier curves for multiple cohorts on `ax`.

    `survival_by_label` maps cohort label -> (duration_days, event_indicator).
    Requires `lifelines`; raises ModuleNotFoundError if unavailable.
    """
    from lifelines import KaplanMeierFitter

    colors = dict(colors) if colors else dict(COHORT_COLORS)
    for label, (dur, evt) in survival_by_label.items():
        dur = pd.to_numeric(dur, errors="coerce")
        evt = pd.to_numeric(evt, errors="coerce")
        mask = dur.notna() & evt.notna() & (dur >= 0)
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter(label=f"{label} (n={int(mask.sum()):,})")
        kmf.fit(durations=dur.loc[mask], event_observed=evt.loc[mask])
        kmf.plot_survival_function(ax=ax, color=colors.get(label), ci_show=True)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson 95% CI for a binomial proportion. Returns (p_hat, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)
