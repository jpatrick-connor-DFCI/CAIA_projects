"""Per-cohort figure-generation pipeline for COMPASS, extracted from
COMPASS_generate_figures.ipynb so the notebook can call it once per cohort
in-process instead of re-executing itself via subprocess per cohort.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, norm
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

try:
    from IPython.display import display
except ImportError:
    display = print

# ---- Publication-quality rendering defaults ----
# 600-dpi rasters, white (non-transparent) backgrounds, and TrueType (type-42)
# font embedding so text in the vector PDFs stays selectable/editable in
# Illustrator rather than being outlined to paths.
SAVE_DPI = 600
plt.rcParams.update({
    "figure.dpi": 130,          # on-screen preview only
    "savefig.dpi": SAVE_DPI,    # exported raster resolution
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    # Editable-text vector output (embed TrueType, don't outline glyphs).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Typography / lines tuned for print.
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.6,
    "lines.antialiased": True,
    "patch.antialiased": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

# Colorblind-safe categorical palette (fixed assignment, never cycled), shared
# by the Figure 2 (LLM subtype / platinum enrichment) panels:
# slot 1 = blue -> platinum+, slot 8 = orange -> platinum-. Kept consistent so
# color identity means the same thing everywhere it appears.
COLOR_PLATINUM_POS = "#2a78d6"   # blue
COLOR_PLATINUM_NEG = "#eb6834"   # orange
COLOR_NEUTRAL_INK = "#52514e"    # secondary ink, for annotations/text only


def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "Accuracy":  (tp + tn) / (tp + tn + fp + fn),
        "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "Recall":    tp / (tp + fn) if (tp + fn) > 0 else 0,
        "TPR":       tp / (tp + fn) if (tp + fn) > 0 else 0,
        "FPR":       fp / (fp + tn) if (fp + tn) > 0 else 0,
        "TNR":       tn / (tn + fp) if (tn + fp) > 0 else 0,
        "FNR":       fn / (fn + tp) if (fn + tp) > 0 else 0,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }

    return pd.Series(metrics)


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (phat, lo, hi)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


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

CATEGORY_MAP = {}
for label, members in [
    ("CBC", CBC), ("CMP", CMP), ("LFT", LFT),
    ("Vitals", VITALS), ("Androgen axis", ANDROGEN), ("Other", OTHER),
]:
    for m in members:
        CATEGORY_MAP[m] = label

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


def assign_category(lab_name: str) -> str:
    return CATEGORY_MAP.get(lab_name, "Other")


def format_label(lab_name, feature_stat):
    if not feature_stat or pd.isna(feature_stat) or feature_stat == "":
        return lab_name
    return f"{lab_name} ({feature_stat})"


def parse_feature(name):
    """Split 'LAB_NAME__stat' into (lab_name, stat). Returns (name, '') if no '__'."""
    if "__" in name:
        lab, stat = name.split("__", 1)
        return lab, stat
    return name, ""


COHORTS = [
    "icd_arpi",
    "vte_arpi",
    "icd_or_vte_arpi",
    "icd_allow_other_primaries_arpi",
    "vte_allow_other_primaries_arpi",
    "icd_or_vte_allow_other_primaries_arpi",
]


def generate_figures(cohort, *, nepc_proj_path, fig_root, cohorts=COHORTS, show=False):
    """Render the full COMPASS figure set for one cohort arm.

    Mirrors the body of COMPASS_generate_figures.ipynb's per-cohort cells
    (Figures 1-7 + Table 1) so the notebook can call this once per cohort in
    the same kernel instead of re-executing itself via subprocess.
    """
    NEPC_PROJ_PATH = nepc_proj_path
    COHORT = cohort
    if COHORT not in cohorts:
        raise ValueError(f"Unknown cohort={COHORT!r}; expected one of {cohorts}")
    print(f"Generating figures for cohort: {COHORT}")

    BASE = NEPC_PROJ_PATH / "survival_analysis" / f"local_runs_{COHORT}"
    LONGITUDINAL_CSV = NEPC_PROJ_PATH / "longitudinal_prediction_data.csv"
    INPUTS_DIR = NEPC_PROJ_PATH / "survival_analysis" / f"prediction_inputs_{COHORT}"

    # Figure outputs are organized by manuscript figure, then exact panel stem.
    # Each PNG filename is prefixed with the cohort definition.
    FIG_ROOT = fig_root


    # Figure-first, panel-second layout for direct cohort comparison:
    # FIG_ROOT/<figure>/<plot-stem>/python/<cohort>_<plot-stem>.png
    FIG_LANG = "python"


    def figure_group(plot_stem):
        """Map an output stem to its manuscript figure directory."""
        if plot_stem.startswith(("figure1", "table1")):
            return "figure1"
        if plot_stem.startswith("figure2"):
            return "figure2"
        if plot_stem.startswith("figure3"):
            return "figure3"
        if plot_stem.startswith("figure4"):
            return "figure4"
        if plot_stem.startswith("km_"):
            return "KM_curves"
        if plot_stem.startswith("androgen_dist_"):
            return "androgen_distributions"
        if plot_stem.startswith("androgen_longitudinal_"):
            return "androgen_trajectories"
        raise ValueError(f"Unmapped figure output stem: {plot_stem}")


    def fig_dir(plot_stem):
        """Resolve the language directory nested under figure and panel."""
        return FIG_ROOT / figure_group(plot_stem) / plot_stem / FIG_LANG


    def figure_path(filename, *, prefix=None):
        """Return a cohort-prefixed path grouped with the same panel from every cohort."""
        filename = Path(filename)
        d = fig_dir(filename.stem)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{prefix or COHORT}_{filename.name}"


    def table_path(stem):
        """Return the cohort-prefixed extensionless base path for a table."""
        d = fig_dir(stem)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{COHORT}_{stem}"


    LANDMARKS = [0, 90]
    TOP_N = 15  # number of features shown per importance panel

    # ---- Figure 2 (LLM subtype + platinum enrichment) inputs ----
    LLM_LABEL_PATH = NEPC_PROJ_PATH / "LLM_NEPC_labels/"
    manual_annotations = pd.read_csv(LLM_LABEL_PATH / "baca_lab_annotations.csv")
    nepc_annotations = pd.read_csv(LLM_LABEL_PATH / "LLM_v3_labels.tsv", sep="\t")
    platinum_mrns = pd.read_csv(NEPC_PROJ_PATH / "mrn_lists/platinum_MRN_list.csv")

    # Platinum membership via set lookup (linear; clearer than an O(n^2) per-row
    # `mrn in ...unique()` dict comprehension).
    platinum_set = set(platinum_mrns["DFCI_MRN"].unique())
    nepc_annotations["is_platinum"] = nepc_annotations["DFCI_MRN"].isin(platinum_set)
    nepc_annotations_all = nepc_annotations.copy()
    print(f"nepc_annotations: {len(nepc_annotations):,} rows "
          f"({nepc_annotations['is_platinum'].sum():,} platinum+, "
          f"{(~nepc_annotations['is_platinum']).sum():,} platinum-)")

    COHORT_LABEL = "PROFILE"


    def load_profile_patient_and_labs(path, *, id_col="DFCI_MRN"):
        """(patient_df, labs_df) split of longitudinal_prediction_data.csv."""
        df = pd.read_csv(path, low_memory=False)
        for c in ["DIAGNOSIS_DATE", "TREATMENT_ANCHOR_DATE", "PLATINUM_DATE",
                  "LAST_CONTACT_DATE", "LAB_DATE", "FIRST_RECORD_DATE"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        if {"DIAGNOSIS_DATE", "TREATMENT_ANCHOR_DATE"} <= set(df.columns):
            df["t_dx_to_anchor"] = (df["TREATMENT_ANCHOR_DATE"] - df["DIAGNOSIS_DATE"]).dt.days.astype("Float64")

        patient_level = [
            id_col, "AGE_AT_TREATMENTSTART", "FIRST_RECORD_DATE", "DIAGNOSIS_DATE",
            "TREATMENT_ANCHOR_DATE", "LAST_CONTACT_DATE",
            "DEATH", "PLATINUM_MEDICATION", "PLATINUM_DATE", "PLATINUM",
            "t_diagnosis", "t_first_treatment", "t_platinum",
            "t_last_contact", "t_death", "t_dx_to_anchor",
        ]
        pat_cols = [c for c in patient_level if c in df.columns]
        patient_df = df[pat_cols].drop_duplicates(subset=[id_col]).reset_index(drop=True)

        lab_cols = [c for c in [id_col, "LAB_NAME", "LAB_VALUE", "LAB_UNIT", "LAB_DATE", "t_lab"]
                    if c in df.columns]
        labs_df = df.loc[df["LAB_NAME"].notna(), lab_cols].reset_index(drop=True)
        return patient_df, labs_df


    def restrict_to_base_landmark_cohort(patient_df, labs_df, inputs_dir, *, id_col, landmark):
        """Restrict Figure 1 summaries to the selected arm at its base landmark."""
        availability_path = inputs_dir / "landmark_mrn_availability.csv"
        if not availability_path.exists():
            raise FileNotFoundError(f"{availability_path} not found -- re-run build_prediction_inputs.py")
        eligible_col = f"eligible_landmark_{landmark}"
        availability = pd.read_csv(availability_path, usecols=[id_col, eligible_col])
        eligible = availability[eligible_col].astype(str).str.lower().isin(["true", "1"])
        cohort_ids = pd.to_numeric(availability.loc[eligible, id_col], errors="coerce").dropna()
        patient_ids = pd.to_numeric(patient_df[id_col], errors="coerce")
        lab_ids = pd.to_numeric(labs_df[id_col], errors="coerce")
        keep = set(cohort_ids.tolist())
        return (patient_df.loc[patient_ids.isin(keep)].copy(),
                labs_df.loc[lab_ids.isin(keep)].copy())


    def record_span_days(labs_df, *, id_col, date_col="LAB_DATE"):
        g = labs_df.groupby(id_col)[date_col]
        span = (g.max() - g.min()).dt.days.astype("Float64")
        span.name = "record_span_days"
        return span


    def load_attrition(inputs_dir):
        landmark_path = inputs_dir / "landmark_attrition.json"
        if not landmark_path.exists():
            raise FileNotFoundError(
                f"{landmark_path} not found -- re-run build_prediction_inputs.py "
                "(it now writes landmark_attrition.json alongside its CSV outputs)."
            )
        return json.loads(landmark_path.read_text())


    def render_consort_panel(ax, attrition):
        # Note: there is no separate first-treatment inclusion step -- treated status
        # is enforced downstream via the treatment anchor (see longitudinal_data_processing.py).
        # Use the cohort-specific downstream funnel written after any MRN-arm restriction.
        downstream = attrition["downstream_cohort_filters"]
        steps = [
            ("Selected MRN arm", downstream["n_before_downstream_cohort_filters"]),
            (f"+ ≥{downstream['min_psa_count']} PSA records", downstream["n_after_psa_count_filter"]),
            ("− PARPi-exposed excluded", downstream["n_after_parpi_exclusion"]),
        ]
        for landmark_str, n in sorted(attrition["eligible_by_landmark"].items(), key=lambda kv: int(kv[0])):
            sign = "+" if int(landmark_str) > 0 else ""
            steps.append((f"Eligible at landmark {sign}{landmark_str}d", n))
        steps.append(("Common cohort (all landmarks)", attrition["n_common_across_landmarks"]))

        n_steps = len(steps)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_steps + 0.4)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title("Cohort attrition (CONSORT)", fontsize=12.5, weight="bold")

        box_w, box_h = 0.86, 0.62
        for i, (label, n) in enumerate(steps):
            y = i + (1 - box_h) / 2
            ax.add_patch(FancyBboxPatch(
                ((1 - box_w) / 2, y), box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor="#eef1f5", edgecolor="#5d6d7e", linewidth=1.0,
            ))
            ax.text(0.5, y + box_h / 2, f"{label}\nn = {n:,}",
                    ha="center", va="center", fontsize=9)
            if i < n_steps - 1:
                ax.annotate(
                    "", xy=(0.5, i + 1 + (1 - box_h) / 2), xytext=(0.5, y + box_h),
                    arrowprops=dict(arrowstyle="-|>", color="#5d6d7e", lw=1.2),
                )

        split_sizes = attrition["split_sizes"]
        footer = "  ·  ".join(f"{k}: n={v:,}" for k, v in split_sizes.items())
        ax.text(0.5, n_steps + 0.15, f"Train/valid/test split — {footer}",
                ha="center", va="top", fontsize=8.5, color="#5d6d7e", style="italic")


    def platinum_km_inputs(df):
        """Cause-specific KM inputs, censoring death at its observed time."""
        t_platinum = pd.to_numeric(df["t_platinum"], errors="coerce")
        platinum = pd.to_numeric(df["PLATINUM"], errors="coerce").fillna(0).eq(1)
        t_death = pd.to_numeric(df.get("t_death", t_platinum), errors="coerce")
        death = pd.to_numeric(df.get("DEATH", pd.Series(0, index=df.index)), errors="coerce").fillna(0).eq(1)
        death_time = t_death.where(death, np.inf)
        duration = pd.concat([t_platinum, death_time], axis=1).min(axis=1)
        event = platinum & (t_platinum <= death_time)
        valid = duration.notna() & np.isfinite(duration) & (duration >= 0)
        return duration.loc[valid], event.loc[valid].astype(int)


    def render_km_panel(ax, patient_df):
        from lifelines import KaplanMeierFitter

        dur, evt = platinum_km_inputs(patient_df)
        kmf = KaplanMeierFitter(label=f"{COHORT_LABEL} {COHORT} (n={len(dur):,})")
        kmf.fit(durations=dur.to_numpy(dtype=float), event_observed=evt.to_numpy(dtype=int))
        kmf.plot_survival_function(ax=ax, color="#1f3a93", ci_show=True)
        ax.set_title("Platinum-free survival")
        ax.set_xlabel("Days from treatment anchor")
        ax.set_ylabel("Platinum-free probability")
        ax.set_ylim(0, 1.02)


    def _overlay_hist(ax, series, *, bins=50):
        v = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
            return
        if isinstance(bins, int):
            lo, hi = v.min(), v.max()
            if hi <= lo:
                lo, hi = v.min(), v.max() + 1.0
            bins = np.linspace(lo, hi, bins + 1)
        ax.hist(v, bins=bins, density=True, alpha=0.75, color="#1f3a93",
                label=f"{COHORT_LABEL} (n={v.size:,})", edgecolor="white", linewidth=0.3)
        ax.legend(fontsize=9, loc="best")


    def render_timing_panel_span(ax, labs_df, id_col):
        span = record_span_days(labs_df, id_col=id_col)
        _overlay_hist(ax, span, bins=50)
        ax.set_xlabel("Record span (days)")
        ax.set_ylabel("Density")
        ax.set_title("Per-patient lab record span")


    def render_timing_panel_dx2tx(ax, patient_df):
        _overlay_hist(ax, patient_df.get("t_dx_to_anchor", pd.Series(dtype=float)), bins=50)
        ax.set_xlabel("Days: diagnosis → treatment anchor")
        ax.set_ylabel("Density")
        ax.set_title("Diagnosis → treatment anchor")


    def render_timing_panel_tplat(ax, patient_df):
        _, event = platinum_km_inputs(patient_df)
        _overlay_hist(ax, patient_df.loc[event.index[event.eq(1)], "t_platinum"], bins=40)
        ax.set_xlabel("Days from treatment anchor to platinum (events only)")
        ax.set_ylabel("Density")
        ax.set_title("Time to platinum (event patients only)")

    def _mean_sd(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        return f"{s.mean():.1f} ± {s.std():.1f}" if not s.empty else "n/a"


    def _median_iqr(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return "n/a"
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        return f"{med:.1f} ({q1:.1f}–{q3:.1f})"


    def _count_pct(mask, total):
        n = int(mask.sum())
        return f"{n:,} ({100 * n / total:.1f}%)" if total else "n/a"


    def build_table1(patient_df):
        n = len(patient_df)
        rows = [("N", f"{n:,}")]

        age = patient_df.get("AGE_AT_TREATMENTSTART", pd.Series(dtype=float))
        rows.append(("Age at first treatment, mean ± SD", _mean_sd(age)))
        rows.append(("Age at first treatment, median (IQR)", _median_iqr(age)))

        rows.append(("Platinum exposure, n (%)",
                     _count_pct(pd.to_numeric(patient_df["PLATINUM"], errors="coerce").fillna(0).astype(bool), n)))
        rows.append(("Median follow-up from treatment anchor, days (IQR)",
                     _median_iqr(patient_df.get("t_last_contact", pd.Series(dtype=float)))))

        return pd.DataFrame(rows, columns=["Characteristic", "Value"])


    def _to_markdown_table(df):
        """Minimal GFM table renderer (no `tabulate` dependency, which isn't in
        this repo's pinned dependency list -- see README)."""
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                         for row in df.itertuples(index=False))
        return "\n".join([header, sep, body])


    def write_table1(table1, out_base):
        out_base.parent.mkdir(parents=True, exist_ok=True)
        table1.to_csv(out_base.with_suffix(".csv"), index=False)
        out_base.with_suffix(".md").write_text(_to_markdown_table(table1))
        return [out_base.with_suffix(".csv"), out_base.with_suffix(".md")]

    ID_COL = "DFCI_MRN"

    print(f"Loading cohort-specific attrition counts from {INPUTS_DIR} ...")
    attrition = load_attrition(INPUTS_DIR)

    print(f"Loading {LONGITUDINAL_CSV} ...")
    patient_df, labs_df = load_profile_patient_and_labs(LONGITUDINAL_CSV, id_col=ID_COL)
    base_landmark = min(int(lm) for lm in attrition["eligible_by_landmark"])
    patient_df, labs_df = restrict_to_base_landmark_cohort(
        patient_df, labs_df, INPUTS_DIR, id_col=ID_COL, landmark=base_landmark
    )
    assert len(patient_df) == attrition["eligible_by_landmark"][str(base_landmark)], (
        f"Figure 1 cohort mismatch: {len(patient_df)} longitudinal patients vs "
        f"{attrition['eligible_by_landmark'][str(base_landmark)]} base-landmark patients"
    )
    print(f"  selected base-landmark cohort: patients={len(patient_df):,}  labs={len(labs_df):,}")

    # --- Panel A: CONSORT ---
    n_landmarks = len(attrition["eligible_by_landmark"])
    fig, ax = plt.subplots(figsize=(7.5, 0.62 * (n_landmarks + 8) + 1))
    render_consort_panel(ax, attrition)
    fig.tight_layout()
    for ext in ("png",):
        out = figure_path(f"figure1a_consort.{ext}")
        fig.savefig(out)
        print(f"wrote {out}")
    if show:
        plt.show()
    plt.close("all")

    # --- Panel B: KM curves ---
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    render_km_panel(ax, patient_df)
    fig.tight_layout()
    for ext in ("png",):
        out = figure_path(f"figure1b_km.{ext}")
        fig.savefig(out)
        print(f"wrote {out}")
    if show:
        plt.show()
    plt.close("all")

    # --- Panel C: timing histograms (one solo panel per metric) ---
    _timing_panels = [
        ("figure1c_span", lambda ax: render_timing_panel_span(ax, labs_df, ID_COL)),
        ("figure1c_dx_to_tx", lambda ax: render_timing_panel_dx2tx(ax, patient_df)),
        ("figure1c_time_to_platinum", lambda ax: render_timing_panel_tplat(ax, patient_df)),
    ]
    for stem, render in _timing_panels:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        render(ax)
        fig.tight_layout()
        for ext in ("png",):
            out = figure_path(f"{stem}.{ext}")
            fig.savefig(out)
            print(f"wrote {out}")
        if show:
            plt.show()
        plt.close("all")

    # --- Table 1 ---
    table1 = build_table1(patient_df)
    for p in write_table1(table1, table_path("table1_baseline_characteristics")):
        print(f"wrote {p}")
    display(table1)

    # --- Compiled Figure 1: CONSORT, KM, and all three timing panels ---
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0])
    ax_a = fig.add_subplot(gs[0, :3])
    ax_b = fig.add_subplot(gs[0, 3:])
    ax_c1 = fig.add_subplot(gs[1, :2])
    ax_c2 = fig.add_subplot(gs[1, 2:4])
    ax_c3 = fig.add_subplot(gs[1, 4:])
    render_consort_panel(ax_a, attrition)
    render_km_panel(ax_b, patient_df)
    render_timing_panel_span(ax_c1, labs_df, ID_COL)
    render_timing_panel_dx2tx(ax_c2, patient_df)
    render_timing_panel_tplat(ax_c3, patient_df)
    for label, ax in zip("ABC", [ax_a, ax_b, ax_c1]):
        ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=14, weight="bold")
    fig.suptitle(f"Figure 1 — COMPASS cohort overview ({COHORT})", fontsize=14, weight="bold")
    fig.tight_layout()
    for ext in ("png",):
        out = figure_path(f"figure1_cohort_overview.{ext}")
        fig.savefig(out)
        print(f"wrote {out}")
    if show:
        plt.show()
    plt.close("all")

    SAVE_ORIGINAL_LLM = COHORT == cohorts[0]
    merged_results = (manual_annotations.drop(columns=['pathology_details', 'manual_platinum_reason'])
                      .merge(nepc_annotations.drop(columns=['has_nepc', 'has_avpc', 'has_molecular_avpc',
                                                            'avpc_criteria', 'visceral_met_pattern', 'num_snippets']),
                                                   on='DFCI_MRN', validate='one_to_one'))
    merged_results['manual_NEPC'] = (merged_results['simplified_manual_platinum_reason'].isin(['nepc', 'squamous_transformation']))
    merged_results['LLM_NEPC'] = merged_results['primary_label'] == 'nepc'

    print(f"merged_results: {len(merged_results):,} rows, "
          f"{merged_results['manual_NEPC'].sum():,} manual-NEPC positive")

    # manual = truth, LLM = pred (correct argument order for binary_metrics).
    metrics = binary_metrics(merged_results['manual_NEPC'], merged_results['LLM_NEPC'])

    # Sanity check: confusion-matrix counts must reconstruct N and the manual-NEPC total.
    assert metrics[["TP", "FP", "TN", "FN"]].sum() == len(merged_results), \
        "confusion matrix counts do not sum to len(merged_results)"
    assert metrics["TP"] + metrics["FN"] == merged_results['manual_NEPC'].sum(), \
        "TP+FN does not match manual_NEPC positive count"

    metrics

    def render_confusion_panel(ax, metrics):
        """Annotated 2x2 confusion matrix, LLM (rows) vs manual truth (cols)."""
        cm = np.array([[metrics["TN"], metrics["FN"]],
                       [metrics["FP"], metrics["TP"]]])
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        labels = ["Non-NEPC", "NEPC"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
        ax.set_xlabel("Manual annotation (truth)")
        ax.set_ylabel("LLM label (prediction)")
        ax.set_title("Panel A1 — confusion matrix", fontsize=11, weight="bold")

        thresh = cm.max() / 2 if cm.max() > 0 else 0.5
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "#0b0b0b",
                        fontsize=13, weight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        return im


    def render_metric_bar_panel(ax, metrics):
        """Compact metric bar: Accuracy, Precision, Recall, Specificity only
        (drops redundant TPR=Recall and FPR/FNR duplicates per the plan)."""
        metric_names = ["Accuracy", "Precision", "Recall", "Specificity"]
        # Specificity == TNR; reuse the existing TNR value under a clearer name.
        values = [metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["TNR"]]

        bars = ax.bar(metric_names, values, color=COLOR_PLATINUM_POS, width=0.6)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Metric value")
        ax.set_title("Panel A2 — classifier metrics", fontsize=11, weight="bold")
        ax.tick_params(axis="x", rotation=0)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10, color="#0b0b0b")
        return bars


    n_total = len(merged_results)
    n_nepc_manual = int(merged_results['manual_NEPC'].sum())
    caption_a = (f"N = {n_total:,} chart-reviewed patients; "
                 f"{n_nepc_manual:,} manual-NEPC positive.")

    # --- individual sub-panels (repo convention: always emit standalone sub-panels) ---
    fig_a1, ax_a1 = plt.subplots(figsize=(4.2, 4.2))
    render_confusion_panel(ax_a1, metrics)
    fig_a1.suptitle(caption_a, fontsize=8.5, color=COLOR_NEUTRAL_INK, y=0.01)
    fig_a1.tight_layout()
    if SAVE_ORIGINAL_LLM:
        for ext in ("png",):
            out = figure_path(f"figure2a1_confusion_matrix.{ext}", prefix="original")
            fig_a1.savefig(out)
            print(f"wrote {out}")
    if show:
        plt.show()
    plt.close("all")

    fig_a2, ax_a2 = plt.subplots(figsize=(5.0, 4.2))
    render_metric_bar_panel(ax_a2, metrics)
    fig_a2.suptitle(caption_a, fontsize=8.5, color=COLOR_NEUTRAL_INK, y=0.01)
    fig_a2.tight_layout()
    if SAVE_ORIGINAL_LLM:
        for ext in ("png",):
            out = figure_path(f"figure2a2_metric_bar.{ext}", prefix="original")
            fig_a2.savefig(out)
            print(f"wrote {out}")
    if show:
        plt.show()
    plt.close("all")

    platinum_positive_labels = nepc_annotations.loc[nepc_annotations['is_platinum'], 'primary_label'].value_counts().reset_index()
    platinum_negative_labels = nepc_annotations.loc[~nepc_annotations['is_platinum'], 'primary_label'].value_counts().reset_index()

    platinum_positive_labels['frac'] = platinum_positive_labels['count'] / platinum_positive_labels['count'].sum()
    platinum_positive_labels['platinum_status'] = 'positive'   # is_platinum == True  (n=200)

    platinum_negative_labels['frac'] = platinum_negative_labels['count'] / platinum_negative_labels['count'].sum()
    platinum_negative_labels['platinum_status'] = 'negative'    # is_platinum == False (n=1682)

    label_distributions = pd.concat([platinum_positive_labels, platinum_negative_labels], ignore_index=True)

    # --- Sanity check: guards against the swap regressing silently. ---
    expected_pos_counts = {"avpc": 118, "nepc": 36, "conventional": 30, "biomarker": 16}
    expected_neg_counts = {"conventional": 1090, "avpc": 456, "biomarker": 72, "nepc": 64}

    pos_counts = platinum_positive_labels.set_index('primary_label')['count'].to_dict()
    neg_counts = platinum_negative_labels.set_index('primary_label')['count'].to_dict()

    assert pos_counts == expected_pos_counts, (
        f"platinum-positive counts do not match expected {expected_pos_counts}; "
        f"got {pos_counts}. The platinum_status labels may be swapped again."
    )
    assert neg_counts == expected_neg_counts, (
        f"platinum-negative counts do not match expected {expected_neg_counts}; "
        f"got {neg_counts}. The platinum_status labels may be swapped again."
    )
    assert platinum_positive_labels['count'].sum() == 200
    assert platinum_negative_labels['count'].sum() == 1682

    label_distributions

    # Fixed class order (not alphabetical / not count-sorted) so bar position is
    # stable and groups the two aggressive classes together for readability.
    CLASS_ORDER = ["conventional", "avpc", "nepc", "biomarker"]

    fig_b, ax_b = plt.subplots(figsize=(7.5, 5))
    sns.barplot(
        data=label_distributions,
        x='primary_label', y='frac', hue='platinum_status',
        order=CLASS_ORDER,
        hue_order=['positive', 'negative'],
        palette={'positive': COLOR_PLATINUM_POS, 'negative': COLOR_PLATINUM_NEG},
        ax=ax_b,
    )
    ax_b.set_xlabel("LLM primary label")
    ax_b.set_ylabel("Fraction within platinum group")
    ax_b.set_ylim(0, 1.0)
    ax_b.set_title("Panel B — subtype landscape by platinum status (descriptive)",
                   fontsize=11, weight="bold")

    n_pos = int(platinum_positive_labels['count'].sum())
    n_neg = int(platinum_negative_labels['count'].sum())
    handles, _ = ax_b.get_legend_handles_labels()
    ax_b.legend(handles, [f"Platinum+ (n={n_pos:,})", f"Platinum- (n={n_neg:,})"],
                title=None, loc="upper right")

    caption_b = ("Fractions computed within each platinum group separately; groups differ "
                 "greatly in size and are not a random sample of the same population — "
                 "see Panel C for the base-rate-robust enrichment statistic.")
    fig_b.text(0.5, -0.02, caption_b, ha="center", va="top", fontsize=8.5,
                color=COLOR_NEUTRAL_INK, wrap=True)
    fig_b.tight_layout()
    if SAVE_ORIGINAL_LLM:
        for ext in ("png",):
            out = figure_path(f"figure2b_subtype_landscape.{ext}", prefix="original")
            fig_b.savefig(out)
            print(f"wrote {out}")
    if show:
        plt.show()
    plt.close("all")

    df = nepc_annotations.copy()

    # Exclude 'biomarker' (and anything outside the 3 core classes) from the
    # aggressive-vs-conventional contrast; stated here rather than silently dropped.
    n_before = len(df)
    df = df[df['primary_label'].isin(['conventional', 'avpc', 'nepc'])]
    n_excluded = n_before - len(df)
    print(f"Excluded {n_excluded:,} rows outside {{conventional, avpc, nepc}} "
          f"(e.g. 'biomarker') from the enrichment contrast; {len(df):,} rows remain.")

    df['aggressive'] = df['primary_label'].isin(['avpc', 'nepc'])

    # 2x2 table: rows = aggressive/conventional, cols = platinum True/False.
    ct = pd.crosstab(df['aggressive'], df['is_platinum'])
    ct = ct.reindex(index=[True, False], columns=[True, False])
    ct.index = ['aggressive', 'conventional']
    ct.columns = ['platinum+', 'platinum-']
    print(ct)

    # Sanity check against the plan's known margins.
    assert ct.loc['aggressive', 'platinum+'] == 154
    assert ct.loc['conventional', 'platinum+'] == 30
    assert ct.loc['aggressive', 'platinum-'] == 520
    assert ct.loc['conventional', 'platinum-'] == 1090

    OR, p_value = fisher_exact(ct.values, alternative='greater')
    print(f"\nOdds ratio (aggressive vs conventional, platinum+ vs platinum-): OR = {OR:.2f}")
    print(f"Fisher's exact p-value (one-sided, OR > 1): p = {p_value:.3g}")

    assert OR > 1, (
        "OR <= 1 -- the platinum_status labels are almost certainly swapped again; "
        "the expected direction is platinum+ enriched for aggressive variants (OR > 1)."
    )

    # P(platinum+ | aggressive) vs P(platinum+ | conventional), with Wilson CIs.
    n_aggressive = int(ct.loc['aggressive'].sum())
    n_conventional = int(ct.loc['conventional'].sum())
    platinum_given_aggressive = int(ct.loc['aggressive', 'platinum+'])
    platinum_given_conventional = int(ct.loc['conventional', 'platinum+'])

    p_agg, lo_agg, hi_agg = wilson_ci(platinum_given_aggressive, n_aggressive)
    p_conv, lo_conv, hi_conv = wilson_ci(platinum_given_conventional, n_conventional)

    print(f"P(platinum+ | aggressive)   = {platinum_given_aggressive}/{n_aggressive} = "
          f"{p_agg:.1%}  (95% CI {lo_agg:.1%}-{hi_agg:.1%})")
    print(f"P(platinum+ | conventional) = {platinum_given_conventional}/{n_conventional} = "
          f"{p_conv:.1%}  (95% CI {lo_conv:.1%}-{hi_conv:.1%})")

    def render_enrichment_panel(ax):
        groups = ["Aggressive\n(AVPC + NEPC)", "Conventional"]
        props = [p_agg, p_conv]
        lo = [p_agg - lo_agg, p_conv - lo_conv]
        hi = [hi_agg - p_agg, hi_conv - p_conv]
        # Same two-slot palette as Panel B would use for platinum+/-, but here the
        # two bars are the *aggressive* vs *conventional* groups being compared on
        # the platinum+ rate -- use a single accent color since it's one series
        # (the y-value, P(platinum+ | group)), with group identity on the x-axis.
        bar_colors = [COLOR_PLATINUM_POS, "#9a9890"]  # accent vs. muted comparison

        bars = ax.bar(groups, props, yerr=[lo, hi], capsize=6,
                      color=bar_colors, width=0.55,
                      error_kw={"ecolor": COLOR_NEUTRAL_INK, "elinewidth": 1.5})
        ax.set_ylabel("P(platinum+ | subtype group)")
        ax.set_ylim(0, max(hi_agg, hi_conv) * 1.35)
        ax.set_title("Panel C — platinum enrichment among aggressive variants",
                     fontsize=11, weight="bold")

        for bar, v, n, k in zip(bars, props, [n_aggressive, n_conventional],
                                 [platinum_given_aggressive, platinum_given_conventional]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                    f"{v:.1%}\n({k}/{n})", ha="center", va="bottom", fontsize=9.5)

        ax.text(0.5, 0.92, f"OR = {OR:.1f}, Fisher's exact p = {p_value:.1e}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, weight="bold", color=COLOR_NEUTRAL_INK)
        return bars


    fig_c, ax_c = plt.subplots(figsize=(6, 5.5))
    render_enrichment_panel(ax_c)
    caption_c = (f"Excludes 'biomarker' primary_label ({n_excluded:,} rows) from the "
                 "aggressive/conventional contrast. Error bars are 95% Wilson score "
                 "intervals. OR and p from Fisher's exact test (one-sided, OR > 1) on "
                 "the 2x2 aggressive/conventional x platinum+/- table.")
    fig_c.text(0.5, -0.05, caption_c, ha="center", va="top", fontsize=8.5,
                color=COLOR_NEUTRAL_INK, wrap=True)
    fig_c.tight_layout()
    if SAVE_ORIGINAL_LLM:
        for ext in ("png",):
            out = figure_path(f"figure2c_enrichment.{ext}", prefix="original")
            fig_c.savefig(out)
            print(f"wrote {out}")
    if show:
        plt.show()
    plt.close("all")

    fig, axd = plt.subplot_mosaic(
        [["A1", "A2", "B"],
         ["C",  "C",  "B"]],
        figsize=(15, 9),
        width_ratios=[1, 1.1, 1.3],
        height_ratios=[1, 1],
    )

    render_confusion_panel(axd["A1"], metrics)
    render_metric_bar_panel(axd["A2"], metrics)

    sns.barplot(
        data=label_distributions,
        x='primary_label', y='frac', hue='platinum_status',
        order=CLASS_ORDER, hue_order=['positive', 'negative'],
        palette={'positive': COLOR_PLATINUM_POS, 'negative': COLOR_PLATINUM_NEG},
        ax=axd["B"],
    )
    axd["B"].set_xlabel("LLM primary label")
    axd["B"].set_ylabel("Fraction within platinum group")
    axd["B"].set_ylim(0, 1.0)
    axd["B"].set_title("Panel B — subtype landscape (descriptive)", fontsize=11, weight="bold")
    handles, _ = axd["B"].get_legend_handles_labels()
    axd["B"].legend(handles, [f"Platinum+ (n={n_pos:,})", f"Platinum- (n={n_neg:,})"],
                    title=None, loc="upper right", fontsize=9)

    render_enrichment_panel(axd["C"])

    fig.suptitle("Figure 2 — LLM-extracted prostate subtypes validate against manual "
                 "annotation and reveal platinum enrichment for aggressive variants",
                 fontsize=13, weight="bold", y=1.02)

    full_caption = (
        f"(A) NEPC-vs-rest classifier, LLM vs Baca-lab manual annotation "
        f"(N={n_total:,}, {n_nepc_manual:,} manual-NEPC+). "
        f"(B) Descriptive subtype landscape, platinum+ (n={n_pos:,}) vs platinum- (n={n_neg:,}); "
        "not itself the enrichment claim. "
        f"(C) Platinum+ rate among aggressive (AVPC+NEPC) vs conventional patients "
        f"(biomarker excluded, {n_excluded:,} rows); Wilson 95% CIs; "
        f"OR = {OR:.1f}, Fisher's exact p = {p_value:.1e}."
    )
    fig.text(0.5, -0.03, full_caption, ha="center", va="top", fontsize=9,
              color=COLOR_NEUTRAL_INK, wrap=True)

    fig.tight_layout()
    if SAVE_ORIGINAL_LLM:
        for ext in ("png",):
            out = figure_path(f"figure2_llm_subtype_platinum.{ext}", prefix="original")
            fig.savefig(out, bbox_inches="tight")
            print(f"wrote {out}")
    if show:
        plt.show()
    plt.close("all")

    # Cohort-specific Figure 2 variant. The full, unfiltered LLM calls above are
    # emitted once with an `original_` prefix; each regular cohort gets the same
    # panel set after restricting labels to its base-landmark MRNs.
    cohort_mrns = set(patient_df[ID_COL].dropna().astype(str))
    nepc_annotations = nepc_annotations_all.loc[
        nepc_annotations_all["DFCI_MRN"].astype(str).isin(cohort_mrns)
    ].copy()
    print(f"Figure 2 cohort subset ({COHORT}): {len(nepc_annotations):,} / "
          f"{len(nepc_annotations_all):,} LLM-labeled MRNs")

    merged_results = (manual_annotations.drop(columns=['pathology_details', 'manual_platinum_reason'])
                      .merge(nepc_annotations.drop(columns=['has_nepc', 'has_avpc', 'has_molecular_avpc',
                                                            'avpc_criteria', 'visceral_met_pattern', 'num_snippets']),
                             on='DFCI_MRN', validate='one_to_one'))
    merged_results['manual_NEPC'] = merged_results['simplified_manual_platinum_reason'].isin(
        ['nepc', 'squamous_transformation'])
    merged_results['LLM_NEPC'] = merged_results['primary_label'] == 'nepc'
    metrics = binary_metrics(merged_results['manual_NEPC'], merged_results['LLM_NEPC'])
    n_total = len(merged_results)
    n_nepc_manual = int(merged_results['manual_NEPC'].sum())
    caption_a = (f"{COHORT} MRN subset; N={n_total:,} chart-reviewed patients; "
                 f"{n_nepc_manual:,} manual-NEPC positive.")

    fig_a1, ax_a1 = plt.subplots(figsize=(4.2, 4.2))
    render_confusion_panel(ax_a1, metrics)
    fig_a1.suptitle(caption_a, fontsize=8.5, color=COLOR_NEUTRAL_INK, y=0.01)
    fig_a1.tight_layout()
    out = figure_path("figure2a1_confusion_matrix.png")
    fig_a1.savefig(out); print(f"wrote {out}")
    plt.close(fig_a1)

    fig_a2, ax_a2 = plt.subplots(figsize=(5.0, 4.2))
    render_metric_bar_panel(ax_a2, metrics)
    fig_a2.suptitle(caption_a, fontsize=8.5, color=COLOR_NEUTRAL_INK, y=0.01)
    fig_a2.tight_layout()
    out = figure_path("figure2a2_metric_bar.png")
    fig_a2.savefig(out); print(f"wrote {out}")
    plt.close(fig_a2)

    platinum_positive_labels = (nepc_annotations.loc[nepc_annotations['is_platinum'], 'primary_label']
                                .value_counts().rename_axis('primary_label').reset_index(name='count'))
    platinum_negative_labels = (nepc_annotations.loc[~nepc_annotations['is_platinum'], 'primary_label']
                                .value_counts().rename_axis('primary_label').reset_index(name='count'))
    platinum_positive_labels['frac'] = (platinum_positive_labels['count'] /
                                        platinum_positive_labels['count'].sum())
    platinum_positive_labels['platinum_status'] = 'positive'
    platinum_negative_labels['frac'] = (platinum_negative_labels['count'] /
                                        platinum_negative_labels['count'].sum())
    platinum_negative_labels['platinum_status'] = 'negative'
    label_distributions = pd.concat([platinum_positive_labels, platinum_negative_labels], ignore_index=True)
    n_pos = int(platinum_positive_labels['count'].sum())
    n_neg = int(platinum_negative_labels['count'].sum())

    fig_b, ax_b = plt.subplots(figsize=(7.5, 5))
    sns.barplot(data=label_distributions, x='primary_label', y='frac', hue='platinum_status',
                order=CLASS_ORDER, hue_order=['positive', 'negative'],
                palette={'positive': COLOR_PLATINUM_POS, 'negative': COLOR_PLATINUM_NEG}, ax=ax_b)
    ax_b.set(xlabel="LLM primary label", ylabel="Fraction within platinum group", ylim=(0, 1.0))
    ax_b.set_title("Panel B — subtype landscape by platinum status (cohort MRN subset)",
                   fontsize=11, weight="bold")
    handles, _ = ax_b.get_legend_handles_labels()
    ax_b.legend(handles, [f"Platinum+ (n={n_pos:,})", f"Platinum- (n={n_neg:,})"],
                title=None, loc="upper right")
    fig_b.text(0.5, -0.02, f"LLM labels restricted to the {COHORT} base-landmark MRN set.",
               ha="center", va="top", fontsize=8.5, color=COLOR_NEUTRAL_INK)
    fig_b.tight_layout()
    out = figure_path("figure2b_subtype_landscape.png")
    fig_b.savefig(out); print(f"wrote {out}")
    plt.close(fig_b)

    df = nepc_annotations.loc[
        nepc_annotations['primary_label'].isin(['conventional', 'avpc', 'nepc'])
    ].copy()
    n_excluded = len(nepc_annotations) - len(df)
    df['aggressive'] = df['primary_label'].isin(['avpc', 'nepc'])
    ct = pd.crosstab(df['aggressive'], df['is_platinum']).reindex(
        index=[True, False], columns=[True, False], fill_value=0)
    ct.index = ['aggressive', 'conventional']; ct.columns = ['platinum+', 'platinum-']
    OR, p_value = fisher_exact(ct.values, alternative='greater')
    n_aggressive = int(ct.loc['aggressive'].sum())
    n_conventional = int(ct.loc['conventional'].sum())
    platinum_given_aggressive = int(ct.loc['aggressive', 'platinum+'])
    platinum_given_conventional = int(ct.loc['conventional', 'platinum+'])
    p_agg, lo_agg, hi_agg = wilson_ci(platinum_given_aggressive, n_aggressive)
    p_conv, lo_conv, hi_conv = wilson_ci(platinum_given_conventional, n_conventional)

    fig_c, ax_c = plt.subplots(figsize=(6, 5.5))
    render_enrichment_panel(ax_c)
    fig_c.text(0.5, -0.05, f"{COHORT} MRN subset; excludes biomarker labels ({n_excluded:,} rows).",
               ha="center", va="top", fontsize=8.5, color=COLOR_NEUTRAL_INK)
    fig_c.tight_layout()
    out = figure_path("figure2c_enrichment.png")
    fig_c.savefig(out); print(f"wrote {out}")
    plt.close(fig_c)

    fig, axd = plt.subplot_mosaic(
        [["A1", "A2", "B"], ["C", "C", "B"]], figsize=(15, 9),
        width_ratios=[1, 1.1, 1.3], height_ratios=[1, 1])
    render_confusion_panel(axd["A1"], metrics)
    render_metric_bar_panel(axd["A2"], metrics)
    sns.barplot(data=label_distributions, x='primary_label', y='frac', hue='platinum_status',
                order=CLASS_ORDER, hue_order=['positive', 'negative'],
                palette={'positive': COLOR_PLATINUM_POS, 'negative': COLOR_PLATINUM_NEG}, ax=axd["B"])
    axd["B"].set(xlabel="LLM primary label", ylabel="Fraction within platinum group", ylim=(0, 1.0))
    axd["B"].set_title("Panel B — subtype landscape (cohort MRN subset)", fontsize=11, weight="bold")
    handles, _ = axd["B"].get_legend_handles_labels()
    axd["B"].legend(handles, [f"Platinum+ (n={n_pos:,})", f"Platinum- (n={n_neg:,})"],
                    title=None, loc="upper right", fontsize=9)
    render_enrichment_panel(axd["C"])
    fig.suptitle(f"Figure 2 — LLM-extracted prostate subtypes ({COHORT} MRN subset)",
                 fontsize=13, weight="bold", y=1.02)
    fig.text(0.5, -0.03,
             f"LLM calls restricted to the {COHORT} base-landmark MRN set "
             f"(n={len(nepc_annotations):,} labels; platinum+ n={n_pos:,}, platinum- n={n_neg:,}).",
             ha="center", va="top", fontsize=9, color=COLOR_NEUTRAL_INK, wrap=True)
    fig.tight_layout()
    out = figure_path("figure2_llm_subtype_platinum.png")
    fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")
    if show:
        plt.show()
    plt.close(fig)

    # ----------------------------- labeling knobs ---------------------------
    TOP_K_PER_PANEL = 4
    ALWAYS_LABEL = {"Hemoglobin", "Albumin", "Alkaline phosphatase"}
    LABEL_FORMAT = "{lab} ({stat})"
    LABEL_FONTSIZE = 8.5
    MIN_LABEL_SPACING_NLP = 0.55
    PANEL_XLIM = (-4.0, 4.0)  # matches COEF_CAP so retained estimates are visible
    Y_MAX_CAP = 30  # -log10(p) ceiling; anything above this is drawn at the cap as a triangle


    def q_threshold_neglog10p(sub):
        """y-value (-log10 p) at which q just crosses 0.05; None if no q-sig features."""
        sig = sub.loc[sub["q_value"] < 0.05, "p_value"]
        if sig.empty:
            return None
        cutoff_p = float(sig.max())
        return float(-np.log10(max(cutoff_p, 1e-300)))


    def _auto_label(ax, sub, *, top_k, always_label, label_format):
        """Stack labels at the left/right panel edges with leader lines.

        Selection rules:
          - Androgen axis: every significant (lab × stat) is labeled (no dedup).
          - Other categories: dedupe to the most-significant stat per lab,
            then include `always_label` whitelist + top_k by p-value.
        """
        sig = sub.loc[sub["sig"]].copy()
        if sig.empty:
            return

        androgen_rows = sig.loc[sig["category"] == "Androgen axis"]

        non_andro = sig.loc[sig["category"] != "Androgen axis"]
        best = (non_andro.sort_values("p_value")
                         .drop_duplicates("lab_name", keep="first"))
        always_sig = best.loc[best["lab_name"].isin(always_label)]
        extra = (best.loc[~best["lab_name"].isin(always_label)].head(top_k))
        non_andro_label = pd.concat([always_sig, extra]).drop_duplicates("lab_name")

        to_label = pd.concat([androgen_rows, non_andro_label]).drop_duplicates(
            subset=["lab_name", "feature_stat"])
        if to_label.empty:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xspan = xlim[1] - xlim[0]
        yspan = ylim[1] - ylim[0]

        def _stack(side_df, side):
            if side_df.empty:
                return
            side_df = side_df.sort_values("neglog10p", ascending=False)
            if side == "left":
                label_x = xlim[0] + 0.04 * xspan
                ha = "left"
            else:
                label_x = xlim[1] - 0.04 * xspan
                ha = "right"
            last_y = None
            for _, r in side_df.iterrows():
                target_y = min(r["neglog10p"], Y_MAX_CAP)
                if last_y is not None and target_y > last_y - MIN_LABEL_SPACING_NLP:
                    target_y = last_y - MIN_LABEL_SPACING_NLP
                if target_y < ylim[0] + 0.05 * yspan:
                    continue
                last_y = target_y
                text = label_format.format(lab=r["lab_name"],
                                           stat=r["feature_stat"])
                color = CATEGORY_COLORS[r["category"]]
                ax.annotate(
                    text, xy=(r["coef_feature"], min(r["neglog10p"], Y_MAX_CAP)),
                    xytext=(label_x, target_y), textcoords="data",
                    ha=ha, va="center",
                    fontsize=LABEL_FONTSIZE, color=color, weight="semibold",
                    arrowprops=dict(arrowstyle="-", lw=0.45, color="#95a5a6"),
                    zorder=10,
                )

        left = to_label.loc[to_label["coef_feature"] < 0]
        right = to_label.loc[to_label["coef_feature"] >= 0]
        _stack(left, "left")
        _stack(right, "right")


    def plot_volcano_panel(ax, sub, title):
        sub = sub.loc[~sub["lab_name"].isin(DROP)].copy()
        sub["category"] = sub["lab_name"].map(assign_category)
        sub["neglog10p"] = -np.log10(sub["p_value"].clip(lower=1e-300))
        sub["sig"] = sub["q_value"] < 0.05
        # Cap extreme -log10(p) so a handful of huge values don't squash the rest.
        sub["capped"] = sub["neglog10p"] > Y_MAX_CAP
        sub["y"] = sub["neglog10p"].clip(upper=Y_MAX_CAP)

        ns = sub.loc[~sub["sig"]]
        ax.scatter(ns["coef_feature"], ns["y"],
                   s=20, color=NS_COLOR, alpha=0.45,
                   edgecolor="none", zorder=1)

        for cat in DRAW_ORDER:
            cat_sig = sub.loc[sub["sig"] & (sub["category"] == cat)]
            if cat_sig.empty:
                continue
            is_hero = cat == "Androgen axis"
            base_kw = dict(
                color=CATEGORY_COLORS[cat],
                alpha=0.92,
                edgecolor="white", linewidth=0.6,
            )
            in_range = cat_sig.loc[~cat_sig["capped"]]
            if not in_range.empty:
                ax.scatter(
                    in_range["coef_feature"], in_range["y"],
                    s=80 if is_hero else 40, marker="o",
                    zorder=5 if is_hero else 3, **base_kw,
                )
            over = cat_sig.loc[cat_sig["capped"]]
            if not over.empty:
                # Triangle marker only -- the exact p-value text callout was
                # dropped to keep this figure clean now that it stands alone.
                ax.scatter(
                    over["coef_feature"], over["y"],
                    s=110 if is_hero else 60, marker="^",
                    zorder=6 if is_hero else 4, **base_kw,
                )

        ax.set_xlim(*PANEL_XLIM)
        y_max = float(sub["y"].max()) if not sub.empty else 5.0
        ax.set_ylim(-0.2, max(y_max * 1.10, 5.0))

        ax.axvline(0, color="grey", linestyle="-", linewidth=0.7, zorder=0)
        for x in (-0.5, 0.5):
            ax.axvline(x, color="grey", linestyle="--", linewidth=0.6,
                       alpha=0.7, zorder=0)
        _auto_label(ax, sub,
                    top_k=TOP_K_PER_PANEL,
                    always_label=ALWAYS_LABEL,
                    label_format=LABEL_FORMAT)

        ax.set_xlabel("Cox log HR per SD")
        ax.set_ylabel(r"$-\log_{10}(p)$")
        ax.set_title(title, fontsize=12.5, weight="bold")

        n_tested = len(sub)
        n_sig = int(sub["sig"].sum())
        breakdown = (sub.loc[sub["sig"], "category"]
                     .value_counts()
                     .reindex([c for c in LEGEND_ORDER if c != "Other"], fill_value=0))
        short = {"Androgen axis": "Androgen", "CBC": "CBC", "LFT": "LFT",
                 "CMP": "CMP", "Vitals": "Vitals"}
        bd_str = "  ".join(f"{short[c]} {int(n)}" for c, n in breakdown.items())
        ax.text(0.99, 0.02,
                f"{n_sig} / {n_tested} displayed q<0.05   ·   {bd_str}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color="#5d6d7e", family="monospace")

    # The shared univariate runner writes ONE file across all landmarks, keyed by
    # an internal `landmark_days` column, to cox/landmark_shared/ (see
    # COMPASS_run_locally.ipynb's SHARED_UNIVARIATE_DIR) -- not per-landmark dirs.
    uni = pd.read_csv(BASE / "cox" / "landmark_shared" / "cox_agg_univariate_nobs_adjusted.csv")
    uni = uni.loc[uni["endpoint"] == "platinum"].copy()
    uni = uni.loc[uni["landmark_days"].isin(LANDMARKS)].copy()
    uni = uni.dropna(subset=["coef_feature", "p_value", "q_value"])
    print(f"{len(uni)} (lab × stat) rows across landmarks "
          f"{sorted(uni['landmark_days'].unique())}")

    # Filter unstable Cox estimates: |log HR| > 4 or CI spans > 2 orders of magnitude.
    COEF_CAP = 4.0
    CI_RATIO_CAP = 100
    ci_ratio = uni["ci_upper"] / uni["ci_lower"]
    mask = (uni["coef_feature"].abs() <= COEF_CAP) & (ci_ratio < CI_RATIO_CAP)
    print(f"dropping {int((~mask).sum())} / {len(uni)} unstable rows")
    uni = uni.loc[mask].copy()
    print(f"{len(uni)} rows remaining")

    # One solo volcano panel per landmark, each with its own category legend.
    panels = [(0, "0 days"), (90, "+90 days")]
    for lm, title in panels:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        sub = uni.loc[uni["landmark_days"] == lm]
        if sub.empty:
            ax.text(0.5, 0.5, f"(no data for landmark = {lm}d)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#7f8c8d")
            ax.set_axis_off()
        else:
            plot_volcano_panel(ax, sub, title)
            handles = [Patch(facecolor=CATEGORY_COLORS[c], edgecolor="white", label=c)
                       for c in LEGEND_ORDER]
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=NS_COLOR, markersize=8,
                                  label="ns (q ≥ 0.05)"))
            ax.legend(handles=handles, loc="best", fontsize=8, ncol=1)

        fig.tight_layout()
        for ext in ("png",):
            out = figure_path(f"figure3_univariate_platinum_landmark{lm}.{ext}")
            fig.savefig(out)
            print(f"wrote {out}")
        if show:
            plt.show()
        plt.close("all")

    # Full labs models vs. their age-only baseline, on the treatment-anchored cohort.
    def _read_metrics(path, auc_col, cindex_col):
        """Return (mean_auc_t, c_index) for the platinum endpoint, or (nan, nan)."""
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            return (np.nan, np.nan)
        row = df.loc[df["endpoint"] == "platinum"]
        if row.empty:
            return (np.nan, np.nan)
        row = row.iloc[0]
        return (float(row[auc_col]), float(row[cindex_col]))


    def cox_labs(lm):
        return _read_metrics(BASE / "cox" / f"landmark_{lm}" / "both" / "cox_agg_multivariable_metrics.csv",
                             "test_mean_auc_t", "test_c_index")


    def cox_baseline(lm):
        return _read_metrics(BASE / "cox" / f"landmark_{lm}" / "baseline" / "cox_agg_baseline_metrics.csv",
                             "test_mean_auc_t", "test_c_index")


    def xgb_labs(lm):
        return _read_metrics(BASE / "xgboost" / f"landmark_{lm}" / "both" / "landmark_xgboost_metrics.csv",
                             "mean_auc_t", "c_index")


    def xgb_baseline(lm):
        return _read_metrics(BASE / "xgboost" / f"landmark_{lm}" / "baseline" / "landmark_xgboost_baseline_metrics.csv",
                             "mean_auc_t", "c_index")


    # (label, loader, color, hatch). Age baselines are the hatched, lighter twins.
    DISCRIMINATION_SERIES = [
        ("Elastic-Net Cox",        cox_labs,     "#4C72B0", None),
        ("Cox baseline (age)",     cox_baseline, "#9DB3D6", "//"),
        ("XGBoost Survival",       xgb_labs,     "#B58900", None),
        ("XGBoost baseline (age)", xgb_baseline, "#E0CC8A", "//"),
    ]

    # discrimination_data[label][k] = (auc, cindex) at LANDMARKS[k]
    discrimination_data = {
        name: [loader(lm) for lm in LANDMARKS] for name, loader, _, _ in DISCRIMINATION_SERIES
    }


    def render_discrimination_panel(ax, *, ylabel, idx, data=discrimination_data,
                                    series=DISCRIMINATION_SERIES, landmarks=LANDMARKS,
                                    show_legend=False):
        """Draw one discrimination panel (AUC(t) or C-index) onto `ax`.

        Reused as-is by the standalone Figure 4a and by the compiled Figure 4.
        """
        n_series = len(series)
        bar_width = 0.19
        x_positions = np.arange(len(landmarks))
        offsets = (np.arange(n_series) - (n_series - 1) / 2) * bar_width

        for i, (name, _loader, color, hatch) in enumerate(series):
            vals = [data[name][k][idx] for k in range(len(landmarks))]
            bar_x = x_positions + offsets[i]
            ax.bar(bar_x, [v if np.isfinite(v) else 0.0 for v in vals], bar_width,
                   color=color, hatch=hatch, edgecolor="white", linewidth=0.8, label=name)
            for x, v in zip(bar_x, vals):
                if np.isfinite(v):
                    ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                            fontsize=7.5, weight="semibold", color=color)

        finite = [data[n][k][idx] for n in data for k in range(len(landmarks))
                  if np.isfinite(data[n][k][idx])]
        ax.set_ylim(0.45, min(1.0, (max(finite) if finite else 0.7) * 1.12))
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{('+' if lm > 0 else '')}{lm} days" for lm in landmarks])
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        if show_legend:
            ax.legend(loc="upper right", fontsize=8, ncol=1)


    # One solo panel per discrimination metric (legend on each).
    _disc_panels = [
        ("figure4a_discrimination_auc_platinum", "Test Mean AUC(t)", 0),
        ("figure4a_discrimination_cindex_platinum", "Test C-index", 1),
    ]
    for stem, ylabel, idx in _disc_panels:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        render_discrimination_panel(ax, ylabel=ylabel, idx=idx, show_legend=True)
        ax.set_title(f"Labs vs. age baseline — platinum ({ylabel})",
                     fontsize=11, weight="bold")
        fig.tight_layout()
        for ext in ("png",):
            out = figure_path(f"{stem}.{ext}")
            fig.savefig(out)
            print(f"wrote {out}")
        if show:
            plt.show()
        plt.close("all")

    def load_cox_coefs(landmark):
        p = BASE / "cox" / f"landmark_{landmark}" / "both" / "cox_agg_multivariable.csv"
        df = pd.read_csv(p)
        df = df.loc[df["endpoint"] == "platinum"].copy()
        df = df.loc[~df["is_age_covariate"].fillna(False).astype(bool)]
        df = df.loc[df["coef"].fillna(0) != 0]
        return df


    def load_xgb_importance(landmark):
        p = BASE / "xgboost" / f"landmark_{landmark}" / "both" / "landmark_xgboost_feature_importance.csv"
        df = pd.read_csv(p)
        df = df.loc[df["endpoint"] == "platinum"].copy()
        df = df.loc[df["feature"].str.lower() != "age"]
        df = df.loc[df["gain"].fillna(0) > 0]
        parsed = df["feature"].apply(lambda s: pd.Series(parse_feature(s),
                                                         index=["lab_name", "feature_stat"]))
        df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)],
                       axis=1)
        return df


    def render_importance_panel(ax, df, *, kind, title):
        if df.empty:
            ax.text(0.5, 0.5, "(no features to display)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#7f8c8d")
            ax.set_title(title, fontsize=11, weight="bold")
            ax.set_axis_off()
            return

        df = df.copy()
        df["category"] = df["lab_name"].map(assign_category)

        if kind == "cox":
            df = df.reindex(df["coef"].abs().sort_values(ascending=False).index).head(TOP_N)
            df = df.iloc[::-1]
            values = df["coef"].to_numpy()
            xlabel = "log HR coefficient"
        else:  # xgb
            df = df.sort_values("gain", ascending=False).head(TOP_N).iloc[::-1]
            values = df["gain"].to_numpy()
            xlabel = "XGBoost gain"

        colors = [CATEGORY_COLORS[c] for c in df["category"]]
        y = np.arange(len(df))
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.6)

        if kind == "cox":
            ax.axvline(0, color="black", linewidth=0.5, zorder=0)

        labels = [format_label(r["lab_name"], r["feature_stat"])
                  for _, r in df.iterrows()]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=11, weight="bold")


    IMPORTANCE_MODEL_ROWS = [
        ("cox", "Elastic-Net Cox", load_cox_coefs),
        ("xgb", "XGBoost Survival", load_xgb_importance),
    ]

    # One solo importance panel per (model x landmark), each with its own legend.
    for kind, model_name, loader in IMPORTANCE_MODEL_ROWS:
        for lm in LANDMARKS:
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            try:
                df = loader(lm)
            except FileNotFoundError as e:
                ax.text(0.5, 0.5, f"file not found:\n{e.filename}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="#c0392b")
                ax.set_axis_off()
            else:
                sign = "+" if lm > 0 else ""
                title = f"{model_name}  ·  {sign}{lm} days"
                render_importance_panel(ax, df, kind=kind, title=title)
                handles = [Patch(facecolor=CATEGORY_COLORS[c], edgecolor="white", label=c)
                           for c in LEGEND_ORDER]
                ax.legend(handles=handles, loc="best", fontsize=8, ncol=1)

            fig.tight_layout()
            for ext in ("png",):
                out = figure_path(f"figure4b_importance_platinum_{kind}_landmark{lm}.{ext}")
                fig.savefig(out)
                print(f"wrote {out}")
            if show:
                plt.show()
            plt.close("all")

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    render_discrimination_panel(axes[0, 0], ylabel="Test Mean AUC(t)", idx=0, show_legend=True)
    render_discrimination_panel(axes[0, 1], ylabel="Test C-index", idx=1, show_legend=False)
    axes[0, 0].set_title("AUC(t): labs vs. age baseline", fontsize=11, weight="bold")
    axes[0, 1].set_title("C-index: labs vs. age baseline", fontsize=11, weight="bold")
    for row_idx, (kind, model_name, loader) in enumerate(IMPORTANCE_MODEL_ROWS, start=1):
        for col_idx, lm in enumerate(LANDMARKS):
            ax = axes[row_idx, col_idx]
            try:
                importance_df = loader(lm)
            except FileNotFoundError as exc:
                ax.text(0.5, 0.5, f"file not found:\n{exc.filename}", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#c0392b")
                ax.set_axis_off()
            else:
                sign = "+" if lm > 0 else ""
                render_importance_panel(ax, importance_df, kind=kind, title=f"{model_name}  ·  {sign}{lm} days")
    category_handles = [Patch(facecolor=CATEGORY_COLORS[c], edgecolor="white", label=c) for c in LEGEND_ORDER]
    fig.legend(handles=category_handles, loc="lower center", ncol=len(category_handles), fontsize=9)
    fig.suptitle(f"Figure 4 — multivariate model performance ({COHORT})", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    for ext in ("png",):
        out = figure_path(f"figure4_multivariate_performance.{ext}")
        fig.savefig(out)
        print(f"wrote {out}")
    if show:
        plt.show()
    plt.close("all")

    ## ---- shared imports for Fig 5/6/7 (androgen-axis figures) ----
    import sys

    REPO_ROOT = next(
        p
        for p in [Path.cwd(), *Path.cwd().parents]
        if (p / "survival_common").exists()
    )
    sys.path.insert(0, str(REPO_ROOT))
    from survival_common.plotting import overlay_km

    try:
        from lifelines.statistics import multivariate_logrank_test
        LIFELINES_OK = True
    except ModuleNotFoundError as exc:
        multivariate_logrank_test = None
        LIFELINES_OK = False
        print(f"lifelines unavailable ({exc}); KM cells will be skipped.")

    try:
        from scipy.stats import mannwhitneyu
        SCIPY_OK = True
    except ModuleNotFoundError as exc:
        mannwhitneyu = None
        SCIPY_OK = False
        print(f"scipy unavailable ({exc}); Mann-Whitney p-values will be skipped.")


    def find_mean_col(columns, substr):
        """Case-insensitive substring match for a `<lab>__mean` column.

        Mirrors the `find_col` helper in `COMPASS_run_locally.ipynb` cell 8
        (`cohort_diagnostics`): searches for a column containing `substr`
        (case-insensitive) and ending in `__mean`. PSA may appear as `PSA__mean`
        or `Prostate specific Ag...__mean`.
        """
        return next(
            (c for c in columns if substr.lower() in c.lower() and c.endswith("__mean")),
            None,
        )


    def resolve_androgen_columns(columns):
        """Return {"Testosterone": col_or_None, "PSA": col_or_None} for a column list.

        PSA is resolved by trying both "PSA" and "Prostate specific Ag" substrings
        (whichever matches first); Testosterone by "Testosterone".
        """
        resolved = {"Testosterone": find_mean_col(columns, "Testosterone")}
        psa_col = find_mean_col(columns, "PSA")
        if psa_col is None:
            psa_col = find_mean_col(columns, "Prostate specific Ag")
        resolved["PSA"] = psa_col
        return resolved


    def quartile_split(df, col):
        """4-way categorical split of a continuous column into quartiles.

        Adapted from `IPIO/survival_analysis/IPIO_generate_figures.ipynb` cell 19.
        """
        vals = pd.to_numeric(df[col], errors="coerce")
        out = pd.Series(pd.NA, index=df.index, dtype=object)
        try:
            labels = pd.qcut(vals, 4, labels=["Low (Q1)", "Q2", "Q3", "High (Q4)"])
        except ValueError:
            return out, None
        out[vals.notna()] = labels.astype(str)[vals.notna()]
        return out, labels.cat.categories.tolist() if hasattr(labels, "cat") else None


    ANDROGEN_LABS = ["Testosterone", "PSA"]

    ## ---- KM curves: testosterone/PSA quartiles, time-to-platinum ----
    # Competing-risk note: death competes with platinum. This cause-specific KM
    # censors death at its observed time, so the y-axis reads
    # "platinum-free probability", not "survival probability". Strata are quartiles
    # of the baseline (__mean) lab value at each landmark; only the Low (Q1) and
    # High (Q4) arms are plotted (the middle quartiles are dropped for a clean
    # two-group contrast).

    FIG5_LANDMARKS = [0, 90]


    def load_aggregated_landmark(landmark):
        path = INPUTS_DIR / f"aggregated_landmark{landmark}.csv"
        if not path.exists():
            print(f"Figure 5: skipped landmark {landmark} -- {path} not found")
            return None
        return pd.read_csv(path, low_memory=False)


    def plot_km_androgen_quartile(ax, agg, lab, mean_col, landmark):
        """Overlay High (Q4) vs Low (Q1) KM curves for `mean_col` at `landmark`."""
        if not LIFELINES_OK:
            ax.text(0.5, 0.5, "(lifelines unavailable)", ha="center", va="center",
                    transform=ax.transAxes, color="#7f8c8d")
            ax.set_title(f"{lab} quartile -- landmark {landmark}d", fontsize=10, weight="bold")
            ax.set_axis_off()
            return
        if mean_col is None or "t_platinum" not in agg.columns or "PLATINUM" not in agg.columns:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="#7f8c8d")
            ax.set_title(f"{lab} quartile -- landmark {landmark}d", fontsize=10, weight="bold")
            ax.set_axis_off()
            return

        strata_col = f"_{lab}_quartile"
        d = agg.copy()
        d[strata_col], _ = quartile_split(d, mean_col)
        d = d.dropna(subset=[strata_col, "t_platinum", "PLATINUM"])
        d = d.loc[d[strata_col].isin(["Low (Q1)", "High (Q4)"])]
        km_duration, km_event = platinum_km_inputs(d)
        d = d.loc[km_duration.index].copy()
        d["_km_duration"] = km_duration
        d["_km_event"] = km_event

        if d.empty or d[strata_col].nunique() < 2:
            ax.text(0.5, 0.5, "(insufficient data after quartile split)",
                    ha="center", va="center", transform=ax.transAxes, color="#7f8c8d")
            ax.set_title(f"{lab} quartile -- landmark {landmark}d", fontsize=10, weight="bold")
            ax.set_axis_off()
            return

        survival_by_label = {
            str(level): (grp["_km_duration"], grp["_km_event"])
            for level, grp in d.groupby(strata_col)
        }
        title = f"{lab} quartile ({mean_col}) -- landmark {'+' if landmark > 0 else ''}{landmark}d"
        overlay_km(ax, survival_by_label, xlabel="Days from landmark",
                   ylabel="Platinum-free probability", title=title)

        n_low = int((d[strata_col] == "Low (Q1)").sum())
        n_high = int((d[strata_col] == "High (Q4)").sum())
        ev_low = int(d.loc[d[strata_col] == "Low (Q1)", "_km_event"].sum())
        ev_high = int(d.loc[d[strata_col] == "High (Q4)", "_km_event"].sum())
        annotation = f"Low: n={n_low}, events={ev_low}\nHigh: n={n_high}, events={ev_high}"

        if LIFELINES_OK and d[strata_col].nunique() >= 2:
            result = multivariate_logrank_test(d["_km_duration"], d[strata_col], d["_km_event"])
            annotation += f"\nlog-rank p = {result.p_value:.3g}"

        ax.text(0.98, 0.02, annotation, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="#5d6d7e", family="monospace")
        ax.legend(fontsize=7.5, loc="lower left")


    # --- solo panels: one figure per (lab x landmark) ---
    for lab in ANDROGEN_LABS:
        for landmark in FIG5_LANDMARKS:
            agg = load_aggregated_landmark(landmark)
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            if agg is None:
                ax.text(0.5, 0.5, "(aggregated CSV not found)", ha="center", va="center",
                        transform=ax.transAxes, color="#7f8c8d")
                ax.set_title(f"{lab} quartile -- landmark {landmark}d", fontsize=10, weight="bold")
                ax.set_axis_off()
            else:
                mean_cols = resolve_androgen_columns(agg.columns)
                plot_km_androgen_quartile(ax, agg, lab, mean_cols[lab], landmark)
            fig.tight_layout()
            for ext in ("png",):
                out = figure_path(f"km_{lab.lower()}_quartile_platinum_landmark{landmark}.{ext}")
                fig.savefig(out)
                print(f"wrote {out}")
            if show:
                plt.show()
            plt.close("all")

    ## ---- Figure 6: baseline androgen distribution by platinum status ----
    # Rendered as two full figure sets: one with a log1p x-scale for BOTH labs and
    # one with a raw (linear, mean-value) x-scale for both. Group medians/IQR are
    # annotated on each, plus a Mann-Whitney U p-value where scipy is available.

    FIG6_LANDMARKS = [0, 90]


    def plot_androgen_dist_by_platinum(ax, agg, lab, mean_col, landmark, *, log_scale):
        if mean_col is None or mean_col not in agg.columns or "PLATINUM" not in agg.columns:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="#7f8c8d")
            ax.set_title(f"{lab} -- landmark {landmark}d", fontsize=10, weight="bold")
            ax.set_axis_off()
            return

        vals = pd.to_numeric(agg[mean_col], errors="coerce")
        plat = pd.to_numeric(agg["PLATINUM"], errors="coerce")
        d = pd.DataFrame({"val": vals, "plat": plat}).dropna()
        if log_scale:
            d = d.loc[d["val"] >= 0]  # laboratory values: keep valid zeros for log1p

        groups = {0: d.loc[d["plat"] == 0, "val"], 1: d.loc[d["plat"] == 1, "val"]}
        if groups[0].empty and groups[1].empty:
            ax.text(0.5, 0.5, "(no data after filtering)", ha="center", va="center",
                    transform=ax.transAxes, color="#7f8c8d")
            ax.set_title(f"{lab} -- landmark {landmark}d", fontsize=10, weight="bold")
            ax.set_axis_off()
            return

        colors = {0: "#1f3a93", 1: "#8e1c2b"}
        labels = {0: "PLATINUM=0", 1: "PLATINUM=1"}
        plot_vals = {k: np.log1p(v) if log_scale else v for k, v in groups.items()}
        all_vals = pd.concat([v for v in plot_vals.values() if not v.empty])
        if all_vals.empty:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="#7f8c8d")
            ax.set_axis_off()
            return
        lo, hi = all_vals.min(), all_vals.max()
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 31)

        for k, v in plot_vals.items():
            if v.empty:
                continue
            ax.hist(v, bins=bins, density=True, alpha=0.55, color=colors[k],
                    label=f"{labels[k]} (n={len(v):,})", edgecolor="white", linewidth=0.3)

        xlabel = f"log1p({lab} {mean_col})" if log_scale else f"{lab} {mean_col}"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{lab} -- landmark {'+' if landmark > 0 else ''}{landmark}d", fontsize=10, weight="bold")
        ax.legend(fontsize=8, loc="best")

        annotation_lines = []
        for k in (0, 1):
            v = groups[k]
            if v.empty:
                continue
            med = v.median()
            q1, q3 = v.quantile([0.25, 0.75])
            annotation_lines.append(f"{labels[k]}: med={med:.2f} (IQR {q1:.2f}-{q3:.2f})")

        if SCIPY_OK and not groups[0].empty and not groups[1].empty:
            stat, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            annotation_lines.append(f"Mann-Whitney p = {p:.3g}")
        elif not SCIPY_OK:
            annotation_lines.append("Mann-Whitney p = (scipy unavailable)")

        ax.text(0.98, 0.98, "\n".join(annotation_lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="#5d6d7e", family="monospace")


    # Two full sets (log1p for both labs, raw for both), each rendered as one solo
    # panel per (lab x landmark).
    FIG6_SCALE_VARIANTS = [
        (True, "log1p", "androgen_dist_by_platinum_log"),
        (False, "raw", "androgen_dist_by_platinum_raw"),
    ]

    for use_log, scale_label, out_stem in FIG6_SCALE_VARIANTS:
        for lab in ANDROGEN_LABS:
            for landmark in FIG6_LANDMARKS:
                fig, ax = plt.subplots(figsize=(6.5, 5.0))
                agg = load_aggregated_landmark(landmark)
                if agg is None:
                    ax.text(0.5, 0.5, "(aggregated CSV not found)", ha="center", va="center",
                            transform=ax.transAxes, color="#7f8c8d")
                    ax.set_title(f"{lab} -- landmark {landmark}d", fontsize=10, weight="bold")
                    ax.set_axis_off()
                else:
                    mean_cols = resolve_androgen_columns(agg.columns)
                    plot_androgen_dist_by_platinum(
                        ax, agg, lab, mean_cols[lab], landmark, log_scale=use_log
                    )
                fig.tight_layout()
                for ext in ("png",):
                    out = figure_path(f"{out_stem}_{lab.lower()}_landmark{landmark}.{ext}")
                    fig.savefig(out)
                    print(f"wrote {out}")
                if show:
                    plt.show()
                plt.close("all")

    ## ---- Figure 7 setup: load long-format androgen-axis traces (selected cohort) ----
    # Restricts LONGITUDINAL_CSV to LAB_NAME in {Testosterone, PSA, and any
    # "Prostate specific Ag" variant}, to the selected cohort arm, and -- so the trajectory
    # panels describe exactly the modeled population -- to the patients present in
    # the aggregated landmark CSVs (union of aggregated_landmark{lm}.csv over
    # LANDMARKS). The treatment anchor
    # is now the timeline origin (time 0) of the longitudinal CSV: every t_* column,
    # t_lab included, is already measured in days FROM the treatment anchor, so t_rel is
    # just t_lab (no re-basing). Patients with no anchor have all-NaN t_* columns and
    # drop out on t_lab.notna(). Platinum events are marked at t_platinum where
    # PLATINUM == 1 (also already anchor-relative).

    RANDOM_SEED = 0
    # Full selected cohort for the group mean +/- CI panel: it plots binned group-level
    # means (not per-patient spaghetti), so there is no rendering reason to subsample
    # -- using everyone maximizes per-bin N and tightens the CIs. Set to a finite
    # integer to cap (the random subsample below only triggers when size > N_GROUP).
    N_GROUP = None


    def _aggregated_landmark_mrns(id_col="DFCI_MRN"):
        """Union of patient IDs across the aggregated landmark CSVs.

        Figure 7 should describe exactly the modeled population, so its longitudinal
        traces are restricted to patients that survived landmark construction. Each
        aggregated_landmark{lm}.csv (see build_prediction_inputs.py) holds one row
        per {id_col}; we take the union across LANDMARKS. Returns None if no
        aggregated CSV is found (in which case the figure is skipped rather than
        silently reverting to the unrestricted longitudinal population).
        """
        mrns = set()
        found = []
        for lm in LANDMARKS:
            p = INPUTS_DIR / f"aggregated_landmark{int(lm)}.csv"
            if not p.exists():
                continue
            cols = pd.read_csv(p, nrows=0).columns
            col = id_col if id_col in cols else ("DFCI_MRN" if "DFCI_MRN" in cols else None)
            if col is None:
                print(f"  [warn] {p.name} has no {id_col}/DFCI_MRN column; skipping")
                continue
            ids = pd.read_csv(p, usecols=[col])[col].dropna().unique()
            mrns.update(ids)
            found.append((p.name, len(ids)))
        if not found:
            return None
        print(f"  aggregated-landmark cohort: {len(mrns):,} unique MRNs from "
              f"{', '.join(f'{n} ({c})' for n, c in found)}")
        return mrns


    def _is_androgen_lab(name):
        n = str(name).lower()
        return ("testosterone" in n) or ("psa" in n) or ("prostate specific ag" in n)


    def load_androgen_longitudinal():
        if not LONGITUDINAL_CSV.exists():
            print(f"Figure 7: skipped -- {LONGITUDINAL_CSV} not found")
            return None

        needed = {"LAB_NAME", "LAB_VALUE", "t_lab", "DFCI_MRN"}
        usecols = None  # read all, then validate -- avoids failing on column order/extras
        df = pd.read_csv(LONGITUDINAL_CSV, low_memory=False)
        missing = needed - set(df.columns)
        if missing:
            print(f"Figure 7: skipped -- {LONGITUDINAL_CSV} missing columns {sorted(missing)}")
            return None

        # Stage-by-stage funnel so a skip reports exactly which gate emptied the frame
        # (bare "no data available" was ambiguous across four possible causes).
        print(f"Figure 7 loader: {len(df):,} rows / {df['DFCI_MRN'].nunique():,} patients in CSV")
        lab_counts = df.loc[df["LAB_NAME"].map(_is_androgen_lab), "LAB_NAME"].value_counts()
        print(f"  androgen LAB_NAME matches: {dict(lab_counts)}"
              if not lab_counts.empty else
              f"  androgen LAB_NAME matches: NONE (sample LAB_NAMEs: "
              f"{sorted(df['LAB_NAME'].dropna().unique())[:12]})")

        df = df.loc[df["LAB_NAME"].map(_is_androgen_lab)].copy()
        if df.empty:
            print("Figure 7: skipped -- no Testosterone/PSA rows in long-format CSV")
            return None

        # Restrict to patients present in the aggregated landmark CSVs so the
        # trajectory panels describe exactly the modeled (aggregated-landmark)
        # population. If no aggregated CSV is found, skip to preserve cohort identity.
        landmark_mrns = _aggregated_landmark_mrns(id_col="DFCI_MRN")
        if landmark_mrns is not None:
            n_before = df["DFCI_MRN"].nunique()
            df = df.loc[df["DFCI_MRN"].isin(landmark_mrns)].copy()
            print(f"  restricted to aggregated-landmark cohort: "
                  f"{n_before:,} -> {df['DFCI_MRN'].nunique():,} patients, {len(df):,} rows")
            if df.empty:
                print("Figure 7: skipped -- no androgen rows for aggregated-landmark patients")
                return None
        else:
            print(f"Figure 7: skipped -- no aggregated_landmark*.csv found under {INPUTS_DIR}")
            return None

        # The treatment anchor is time 0, so t_lab is already days-from-anchor. Non-anchor
        # patients have all-NaN t_* columns and are excluded by the t_lab.notna() drop.
        df["t_lab"] = pd.to_numeric(df["t_lab"], errors="coerce")
        df["t_rel"] = df["t_lab"]
        df["LAB_VALUE"] = pd.to_numeric(df["LAB_VALUE"], errors="coerce")
        n_before_drop = len(df)
        n_t_lab_ok = int(df["t_lab"].notna().sum())
        n_val_ok = int(df["LAB_VALUE"].notna().sum())
        df = df.dropna(subset=["t_rel", "LAB_VALUE"])
        print(f"  androgen rows: {n_before_drop:,} -> with non-null t_lab: {n_t_lab_ok:,}, "
              f"with non-null LAB_VALUE: {n_val_ok:,} -> kept: {len(df):,}")
        if df.empty:
            print("Figure 7: skipped -- androgen rows present but all t_lab or LAB_VALUE are NaN "
                  "(likely no treatment anchor for these patients, or non-numeric LAB_VALUE)")
            return None

        if "t_platinum" in df.columns and "PLATINUM" in df.columns:
            # t_platinum is already anchor-relative, so the event offset is t_platinum itself.
            df["t_platinum_rel"] = pd.to_numeric(df["t_platinum"], errors="coerce")
        else:
            df["t_platinum_rel"] = np.nan

        def _lab_group(name):
            n = str(name).lower()
            return "Testosterone" if "testosterone" in n else "PSA"

        df["LAB_GROUP"] = df["LAB_NAME"].map(_lab_group)
        return df


    androgen_long_df = load_androgen_longitudinal()
    if androgen_long_df is not None:
        print(f"Figure 7: {len(androgen_long_df):,} androgen-axis rows across "
              f"{androgen_long_df['DFCI_MRN'].nunique():,} {COHORT} patients")

    ## ---- Figure 7b: group mean +/- 95% CI, binned by time from treatment anchor ----
    # Uses binned means (no LOESS) to avoid a statsmodels dependency. t_rel is
    # clipped to a fixed horizon and split into fixed-width bins; within each bin,
    # mean +/- 1.96*SEM is computed separately for platinum+ and platinum-
    # patients. One panel per lab (PSA, Testosterone).

    BIN_WIDTH_DAYS = 60
    # Asymmetric window: 1 year of pre-landmark history, 2 years of follow-up.
    PRE_DAYS = 365   # clip to this many days BEFORE the treatment anchor
    POST_DAYS = 730  # clip to this many days AFTER the treatment anchor


    def _bin_group_ci(df, lab_group):
        sub = df.loc[df["LAB_GROUP"] == lab_group].copy()
        sub = sub.loc[sub["t_rel"].between(-PRE_DAYS, POST_DAYS)]
        if sub.empty:
            return None

        bin_edges = np.arange(-PRE_DAYS, POST_DAYS + BIN_WIDTH_DAYS, BIN_WIDTH_DAYS)
        sub["t_bin"] = pd.cut(sub["t_rel"], bins=bin_edges, include_lowest=True)

        if "PLATINUM" in sub.columns:
            sub["_plat_group"] = pd.to_numeric(sub["PLATINUM"], errors="coerce").fillna(0).astype(int)
        else:
            sub["_plat_group"] = 0

        # Give each patient equal weight within a bin before computing between-patient CIs.
        patient_bin = (sub.groupby(["DFCI_MRN", "t_bin", "_plat_group"], observed=True)["LAB_VALUE"]
                       .mean().reset_index())
        rows = []
        for (t_bin, plat_group), grp in patient_bin.groupby(["t_bin", "_plat_group"], observed=True):
            vals = grp["LAB_VALUE"].dropna()
            if vals.empty:
                continue
            n = grp["DFCI_MRN"].nunique()
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            rows.append({
                "t_mid": t_bin.mid, "plat_group": plat_group,
                "mean": mean, "ci_lo": mean - 1.96 * sem, "ci_hi": mean + 1.96 * sem, "n": n,
            })
        return pd.DataFrame(rows) if rows else None


    def plot_group_ci_panel(ax, df, lab_group):
        binned = _bin_group_ci(df, lab_group)
        if binned is None or binned.empty:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="#7f8c8d")
            ax.set_title(lab_group, fontsize=11, weight="bold")
            ax.set_axis_off()
            return

        colors = {0: "#1f3a93", 1: "#8e1c2b"}
        labels = {0: "Non-platinum", 1: "Platinum"}
        for plat_group, grp in binned.groupby("plat_group"):
            grp = grp.sort_values("t_mid")
            ax.plot(grp["t_mid"], grp["mean"], marker="o", markersize=3.5, linewidth=1.5,
                    color=colors.get(plat_group, "#7f8c8d"),
                    label=labels.get(plat_group, plat_group))
            ax.fill_between(grp["t_mid"], grp["ci_lo"], grp["ci_hi"],
                             color=colors.get(plat_group, "#7f8c8d"), alpha=0.2)

        ax.axvline(0, color="#2c3e50", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Days from treatment anchor (binned, 60d windows)")
        ax.set_ylabel(f"{lab_group} (mean +/- 95% CI)")
        ax.set_title(lab_group, fontsize=11, weight="bold")
        ax.legend(fontsize=8, loc="best")


    if androgen_long_df is None:
        print("Figure 7b: skipped -- no androgen longitudinal data available")
    else:
        group_patients = androgen_long_df["DFCI_MRN"].dropna().unique()
        if N_GROUP is not None and group_patients.size > N_GROUP:
            rng = np.random.default_rng(RANDOM_SEED)
            group_patients = rng.choice(group_patients, size=N_GROUP, replace=False)
        group_df = androgen_long_df.loc[androgen_long_df["DFCI_MRN"].isin(group_patients)]

        for lab_group in ["PSA", "Testosterone"]:
            n_pat = group_df.loc[group_df["LAB_GROUP"] == lab_group, "DFCI_MRN"].nunique()
            fig, ax = plt.subplots(figsize=(7.0, 5.5))
            plot_group_ci_panel(ax, group_df, lab_group)
            ax.set_title(
                f"{lab_group} -- group mean +/- 95% CI vs. days from treatment anchor "
                f"(n={n_pat:,} patients)",
                fontsize=11, weight="bold",
            )
            fig.tight_layout()
            for ext in ("png",):
                out = figure_path(f"androgen_longitudinal_group_ci_{lab_group.lower()}.{ext}")
                fig.savefig(out)
                print(f"wrote {out}")
            if show:
                plt.show()
            plt.close("all")
