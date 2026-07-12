# IPIO talk figures — implementation spec

**Audience for this doc:** a fresh coding agent with no prior conversation context.
**Goal:** extend `IPIO_generate_figures.ipynb` with figures for a 15-minute talk on the
IPIO project (time-to-irAE univariate associations + model predictive power). The
headline addition is **KM / cumulative-incidence curves** stratified by univariate
associations, plus a few supporting slides.

Everything below is achievable from **outputs already on disk — no model re-runs.**
All new plotting must reuse `survival_common.plotting` (RCPARAMS, `overlay_km`,
`CATEGORY_COLORS`) and `lifelines` (already a project dependency).

---

## 0. Orientation — read these first

- Notebook to edit: `IPIO/survival_analysis/IPIO_generate_figures.ipynb`
  - Existing sections: Imports; Paths; Clinical category mapping; Fig 1 volcano
    (labs + genomics); Fig 2 discrimination (AUC + C-index); Fig 3 importance (2×2).
  - It currently reads **only Cox/XGBoost result CSVs**, never per-patient rows.
- Cohort construction: `IPIO/survival_analysis/ipio_cohort.py`
  → `make_irae_outcome_df` builds per-patient `t_irae`, `IRAE`, `DEATH`,
    `event_type` (**0=censored, 1=irAE, 2=death**), baseline covariates.
- Shared plotting: `survival_common/plotting.py`
  → `overlay_km(ax, survival_by_label, ...)` where `survival_by_label` maps
    `label -> (duration_days, event_indicator)`. Uses lifelines, shows CI,
    auto-appends `(n=...)` to labels. Also `RCPARAMS`, `parse_feature`,
    `assign_category`, `COHORT_COLORS`.
- Multivariate driver: `IPIO/survival_analysis/multivariate_analysis.py`
  → persists per-patient **test-set** risk scores (`duration`, `event`, `risk_score`).

### Paths (mirror the notebook's existing `Paths` cell)
```python
BASE          = Path(".../IPIO/survival_analysis/local_runs")   # cox/, xgboost/, genomic/
GENOMIC_BASE  = BASE / "genomic"
INPUTS_DIR    = Path(".../IPIO/survival_analysis/prediction_inputs")   # aggregated_landmark*.csv
OUT_DIR       = Path(".../figures/CAIA/IPIO")
LANDMARKS     = [0, 90]
```
Use the notebook's own `BASE`/`OUT_DIR` values — do not hardcode new roots. Add
`INPUTS_DIR` next to them (it is `BASE.parent / "prediction_inputs"` in the standard
layout; confirm against `IPIO_run_locally.ipynb` which defines `INPUTS_DIR` and
`OUTPUT_DIR`).

### Files this spec consumes (all pre-existing)
| Purpose | Path |
|---|---|
| Per-patient outcome + features (labs arm) | `INPUTS_DIR/aggregated_landmark{0,90}.csv` |
| Per-patient outcome + features (genomic arm) | `INPUTS_DIR/genomic/genomic_aggregated_landmark{0,90}.csv` |
| Per-patient test risk scores (XGBoost) | `BASE/xgboost/landmark_{lm}/both/landmark_xgboost_patient_risks.csv` |
| Univariate Cox (to pick top lab hits) | `BASE/cox/landmark_{lm}/both/cox_agg_univariate_nobs_adjusted.csv` |
| AUC(t) time course | `BASE/{cox,xgboost}/landmark_{lm}/{both,baseline}/landmark_xgboost_auc_t.csv` (xgb) and Cox `..._auc_t` analog |
| Landmark attrition | `INPUTS_DIR/landmark_attrition.json` |

> Filenames follow the patterns in `multivariate_analysis.py` (e.g.
> `risks_path = output_dir / f"landmark_xgboost{suffix}_patient_risks.csv"`,
> `auc_path = ... f"landmark_xgboost{suffix}_auc_t.csv"`). If a given config's
> file is absent, **print what was checked and skip that panel** (match the
> notebook's existing `_first_existing_path` / graceful-skip style — never hard-crash a cell).

---

## CRITICAL correctness constraints (do not violate)

1. **Competing risk.** For time-to-irAE, death is a competing event. A naive KM
   censoring death **overestimates** cumulative irAE incidence.
   - KM panels: label the y-axis **"irAE-free probability"** (NOT "Survival
     probability") and add a footnote "death treated as censoring."
   - Also provide a **cumulative-incidence (CIF)** version using
     `lifelines.AalenJohansenFitter` with the existing `event_type` column
     (event_of_interest=1 for irAE, death=2 as competing). This is the
     appendix/backup figure.
2. **Test-set only for risk tertiles.** The `..._patient_risks.csv` file is
   already test-only; do not merge train patients into risk-stratified curves.
3. **Minimum group size guard.** Every stratified curve must drop arms with
   `< MIN_EVENTS` events (default **10**) and **print** what was dropped. Sparse
   arms (esp. genomic markers) produce meaningless KM tails.
4. **Landmark 0d is the headline.** Put +90d versions in an "appendix" markdown
   section. Do not double every figure in the main flow.
5. **Placeholders that must be replaced before the talk (flag loudly in a
   markdown cell, do not silently ship):**
   - `CATEGORY_MAP` in the "Clinical category mapping" cell is self-described as a
     placeholder ("revisit once the real canonical-lab set is confirmed").
   - `ALWAYS_LABEL = {"TSH","ALT","Creatinine"}` in the Fig 1 cell is a placeholder.
     Replace with the *actual* top univariate hits (see §2 helper) so the talk
     labels real signal.

---

## Work items (implement in this order)

### 1. Cohort-load cell  *(new cell, insert right after `Paths`)*
Load per-patient tables into a dict keyed by landmark. Provide a single helper:
```python
def load_cohort(landmark, arm="labs"):
    """Return per-patient DataFrame with t_irae, IRAE, event_type, baseline covars."""
    fn = (INPUTS_DIR / f"aggregated_landmark{landmark}.csv") if arm == "labs" \
         else (INPUTS_DIR / "genomic" / f"genomic_aggregated_landmark{landmark}.csv")
    df = pd.read_csv(fn)
    # expected cols: t_irae, IRAE, event_type, pd1pdl1, ctla4, AGE_AT_TREATMENTSTART, CANCER_TYPE_*
    return df
```
And a risk-score loader:
```python
def load_patient_risks(landmark, config="both", model="xgboost"):
    p = BASE / model / f"landmark_{landmark}" / config / f"landmark_{model}_patient_risks.csv"
    df = pd.read_csv(p)
    return df.loc[df["endpoint"] == "irae"]  # cols: duration, event, risk_score
```
Print row counts and event counts per landmark so the reader can sanity-check.

### 2. KM + log-rank + CIF helpers  *(new cell)*
Thin wrappers over shared utilities:
- `km_by_strata(ax, df, strata_col, *, title, min_events=10)`:
  build `{label: (df.t_irae, df.IRAE)}` per stratum level, drop levels with
  `< min_events` events (print drops), call `overlay_km`, set ylabel
  `"irAE-free probability"`, then compute and annotate the log-rank p-value via
  `lifelines.statistics.multivariate_logrank_test(df.t_irae, df[strata_col], df.IRAE)`.
- `cif_by_strata(ax, df, strata_col, *, event_of_interest=1, competing=2)`:
  per stratum, `AalenJohansenFitter().fit(df.t_irae, df.event_type, event_of_interest=1)`,
  plot `cumulative_density_`; ylabel `"Cumulative irAE incidence"`.
- Helper `dichotomize(df, col, cut="median")` → returns a categorical Series
  `f"{col} ≥ cut"` / `f"{col} < cut"` for continuous lab strata.
- Helper `top_univariate_labs(landmark, k=3)` → read the univariate Cox CSV
  (endpoint=="irae"), drop genomic/baseline features (reuse the notebook's
  `_is_genomic_feature` / `_is_baseline_covariate` regexes from Fig 1 cell),
  sort by p_value, return the top-k `feature` names (and their `lab_name`).

### 3. KM small-multiples cell (0d headline)  *(new cell)*
One figure, a row of KM panels at **landmark 0**, saved as
`km_strata_irae.{png,pdf}` in `OUT_DIR`. Strata:
1. **Drug class** — pd1pdl1 vs ctla4 (and combination if present descriptively;
   note combination is excluded from modeling). Derive a single categorical
   column from the `pd1pdl1`/`ctla4` flags.
2. **Top univariate lab(s)** — for each of the top ~2 labs from
   `top_univariate_labs(0)`, dichotomize at median and plot. Turns the volcano's
   top dots into curves.
3. **Model risk tertiles** — merge `load_patient_risks(0)` on the id column,
   cut `risk_score` into tertiles, plot. This is the single most persuasive
   "the model works" slide — it visualizes discrimination as curve separation and
   complements the AUC/C-index bars.
4. **Genomic marker (CONDITIONAL)** — only if some somatic indicator has
   `>= MIN_EVENTS` events in both arms (use genomic cohort table); otherwise
   **skip and print why**. Do not force it.

Every panel gets its log-rank p in the corner. Guard sparse arms per §CRITICAL-3.

### 4. CIF appendix cell  *(new cell, under an "## Appendix" markdown header)*
Re-plot the drug-class and risk-tertile strata as cumulative incidence
(`cif_by_strata`), saved as `cif_strata_irae.{png,pdf}`. This is the backup slide
that preempts the competing-risk objection.

### 5. Cohort "Table 1" + attrition cell  *(new cell, place near the top)*
Highest-value context slide. From the 0d cohort table + `landmark_attrition.json`:
n patients, irAE rate, death rate, median follow-up (median `t_irae`),
drug-class breakdown, cancer-type mix, landmark attrition counts. Render as a
styled DataFrame (or a clean matplotlib table). Save `cohort_table1_irae.{png,pdf}`.

### 6. AUC(t) time-course cell  *(new cell)*
Line plot of AUC(t) vs. horizon for **baseline vs baseline+labs** at landmark 0,
reading the existing `..._auc_t.csv` files. Richer than the single mean-AUC bar —
shows *where* labs add discrimination. Save `auc_timecourse_irae.{png,pdf}`.

### 7. (Optional) Univariate HR forest plot  *(new cell)*
Top ~8–10 labs by univariate p at 0d as a forest plot (HR, 95% CI from
`ci_lower`/`ci_upper`, colored by `CATEGORY_COLORS`). Reads from the back of a
room better than the dense volcano — a talk *alternative* to Fig 1, not a
replacement in the notebook. Save `univariate_forest_irae.{png,pdf}`.

### 8. Mark existing figures for appendix  *(markdown edits only)*
Add markdown notes: genomics volcano (Fig 1b) and genomics discrimination
(Fig 2b) → "appendix if genomic signal weak"; importance 2×2 (Fig 3) →
"show one panel (elastic-net Cox @0d) in the talk, rest appendix". Do not delete
code — just annotate so the presenter knows what to cut from slides.

---

## Acceptance checklist
- [ ] Notebook runs top-to-bottom without hard errors; missing files degrade to a
      printed "skipped: checked ..." message, not a traceback.
- [ ] KM y-axis reads "irAE-free probability" with a death-censoring footnote.
- [ ] A working CIF (Aalen–Johansen) appendix figure exists.
- [ ] Every stratified curve drops `<10`-event arms and prints the drop.
- [ ] Risk-tertile curves use test-set patients only.
- [ ] Log-rank p-value annotated on each KM panel.
- [ ] `ALWAYS_LABEL` volcano labels replaced with real top univariate hits (or a
      loud markdown flag left if the canonical-lab set is still unconfirmed).
- [ ] All new figures saved to `OUT_DIR` as both `.png` and `.pdf` (match existing
      save loop style).
- [ ] 0d is the headline; +90d variants live under an "Appendix" markdown header.

## Notes / open items for the presenter (not the agent)
- Confirm `CATEGORY_MAP` matches the real IPIO canonical-lab set before the talk;
  category colors are currently placeholder-derived.
- Confirm `INPUTS_DIR` location matches the local run (defined in `IPIO_run_locally.ipynb`).
- The "combination" drug class is excluded from modeling but may be shown
  descriptively in KM — call this out on the slide if included.
