"""
Paired volcano plot of univariate Cox results for time-to-platinum
at landmark = 0 days and landmark = +90 days.

Each dot is one (lab × feature_stat) pair. Color encodes clinical category
(CBC, CMP, LFT, Vitals, Androgen axis, Other). Significance is FDR q < 0.05;
ns dots are faded gray, q-significant dots are bold in their category color.

Usage:
    python paired_volcano_platinum.py \
        --uni-path /data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/cox_agg_univariate_nobs_adjusted.csv \
        --out-dir  ./figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

# ----------------------- Clinical category mapping -----------------------

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
DROP = {"Body height"}  # data-quality concerns

CATEGORY_MAP: dict[str, str] = {}
for label, members in [
    ("CBC", CBC), ("CMP", CMP), ("LFT", LFT),
    ("Vitals", VITALS), ("Androgen axis", ANDROGEN), ("Other", OTHER),
]:
    for m in members:
        CATEGORY_MAP[m] = label

# draw order: last drawn sits on top → hero burgundy on top
DRAW_ORDER = ["Other", "Vitals", "CMP", "LFT", "CBC", "Androgen axis"]
LEGEND_ORDER = ["Androgen axis", "CBC", "LFT", "CMP", "Vitals", "Other"]

CATEGORY_COLORS = {
    "Androgen axis": "#8e1c2b",  # burgundy — narrative hero
    "CBC":           "#16a085",  # teal
    "LFT":           "#e67e22",  # orange
    "CMP":           "#7d3c98",  # purple
    "Vitals":        "#5d6d7e",  # slate gray
    "Other":         "#95a5a6",  # neutral gray
}
NS_COLOR = "#d5d8dc"  # ns dots — very faded


def assign_category(lab_name: str) -> str:
    return CATEGORY_MAP.get(lab_name, "Other")


def q_threshold_neglog10p(sub: pd.DataFrame) -> float | None:
    """Return the y-value (-log10 p) at which q just crosses 0.05.

    Returns None if no feature in this panel reaches q < 0.05.
    """
    sig = sub.loc[sub["q_value"] < 0.05, "p_value"]
    if sig.empty:
        return None
    cutoff_p = float(sig.max())  # largest p still surviving FDR
    return float(-np.log10(max(cutoff_p, 1e-300)))


def plot_panel(ax, sub: pd.DataFrame, title: str,
               *,
               label_testosterone: bool,
               label_psa: bool,
               bracket_general_health: bool) -> None:
    sub = sub.loc[~sub["lab_name"].isin(DROP)].copy()
    sub["category"] = sub["lab_name"].map(assign_category)
    sub["neglog10p"] = -np.log10(sub["p_value"].clip(lower=1e-300))
    sub["sig"] = sub["q_value"] < 0.05

    # ns dots underneath
    ns = sub.loc[~sub["sig"]]
    ax.scatter(ns["coef_feature"], ns["neglog10p"],
               s=20, color=NS_COLOR, alpha=0.45,
               edgecolor="none", zorder=1)

    # significant dots, drawn category-by-category so hero ends up on top
    for cat in DRAW_ORDER:
        cat_sig = sub.loc[sub["sig"] & (sub["category"] == cat)]
        if cat_sig.empty:
            continue
        is_hero = cat == "Androgen axis"
        ax.scatter(
            cat_sig["coef_feature"], cat_sig["neglog10p"],
            s=80 if is_hero else 40,
            color=CATEGORY_COLORS[cat],
            alpha=0.92,
            edgecolor="white", linewidth=0.6,
            zorder=5 if is_hero else 3,
        )

    # reference lines
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.6, zorder=0)
    q_y = q_threshold_neglog10p(sub)
    if q_y is not None:
        ax.axhline(q_y, color="black", linestyle=":", linewidth=0.9, zorder=0)

    # narrative annotations
    hero_color = CATEGORY_COLORS["Androgen axis"]

    if label_testosterone:
        t = sub.loc[(sub["lab_name"] == "Testosterone") & sub["sig"]]
        if not t.empty:
            r = t.loc[t["neglog10p"].idxmax()]
            ax.annotate(
                f"Testosterone {r['feature_stat']}",
                (r["coef_feature"], r["neglog10p"]),
                xytext=(14, 12), textcoords="offset points",
                fontsize=10.5, weight="bold", color=hero_color,
                arrowprops=dict(arrowstyle="-", lw=0.9, color=hero_color),
            )

    if label_psa:
        p = sub.loc[(sub["lab_name"] == "PSA") & sub["sig"]]
        if not p.empty:
            r = p.loc[p["neglog10p"].idxmax()]
            ax.annotate(
                f"PSA {r['feature_stat']}",
                (r["coef_feature"], r["neglog10p"]),
                xytext=(14, -22), textcoords="offset points",
                fontsize=10.5, weight="bold", color=hero_color,
                arrowprops=dict(arrowstyle="-", lw=0.9, color=hero_color),
            )

    if bracket_general_health:
        cluster = sub.loc[sub["sig"] & sub["category"].isin(["CBC", "LFT"])]
        if not cluster.empty:
            cx = float(cluster["coef_feature"].median())
            cy = float(cluster["neglog10p"].median())
            xlim = ax.get_xlim()
            xspan = xlim[1] - xlim[0] if xlim[1] > xlim[0] else 1.0
            ax.annotate(
                "CBC + LFT signals\nemerge by 90d",
                xy=(cx, cy), xycoords="data",
                xytext=(cx + 0.20 * xspan, cy + 5),
                textcoords="data",
                ha="left", va="bottom",
                fontsize=10, style="italic", color="#2c3e50",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#2c3e50",
                                connectionstyle="arc3,rad=0.18"),
            )

    # cosmetics
    ax.set_xlabel("Cox log HR per SD")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(title, fontsize=12.5, weight="bold")

    # footer: N tested · N q<0.05 · breakdown
    n_tested = len(sub)
    n_sig = int(sub["sig"].sum())
    breakdown = (sub.loc[sub["sig"], "category"]
                 .value_counts()
                 .reindex([c for c in LEGEND_ORDER if c != "Other"],
                          fill_value=0))
    short = {"Androgen axis": "Androgen", "CBC": "CBC", "LFT": "LFT",
             "CMP": "CMP", "Vitals": "Vitals"}
    bd_str = "  ".join(f"{short[c]} {int(n)}" for c, n in breakdown.items())
    ax.text(0.99, 0.02,
            f"{n_sig} / {n_tested} q<0.05   ·   {bd_str}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#5d6d7e", family="monospace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uni-path", type=Path, required=True,
                    help="Path to cox_agg_univariate_nobs_adjusted.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("./figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    uni = pd.read_csv(args.uni_path)
    uni = uni.loc[uni["endpoint"] == "platinum"].copy()
    uni = uni.dropna(subset=["coef_feature", "p_value", "q_value"])

    panels = [
        (0,  "Landmark = 0 days",
         dict(label_testosterone=True, label_psa=False,
              bracket_general_health=False)),
        (90, "Landmark = +90 days",
         dict(label_testosterone=True, label_psa=True,
              bracket_general_health=True)),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    for ax, (lm, title, kwargs) in zip(axes, panels):
        sub = uni.loc[uni["landmark_days"] == lm]
        if sub.empty:
            ax.text(0.5, 0.5, f"(no data for landmark = {lm}d)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#7f8c8d")
            ax.set_axis_off()
            continue
        plot_panel(ax, sub, title, **kwargs)

    fig.suptitle(
        "Univariate Cox (n_obs-adjusted) — time-to-platinum",
        fontsize=15, weight="bold", y=1.02,
    )

    # shared legend at the bottom
    handles = [Patch(facecolor=CATEGORY_COLORS[c], edgecolor="white",
                     label=c) for c in LEGEND_ORDER]
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=NS_COLOR, markersize=8,
                          label="ns (q ≥ 0.05)"))
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.04),
               fontsize=10)

    for ext in ("png", "pdf"):
        out = args.out_dir / f"paired_volcano_platinum.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
