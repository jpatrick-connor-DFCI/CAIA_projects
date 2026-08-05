# GAM-based trajectory features + nonlinear hazard testing for time-to-platinum

Implementation plan. All file paths below are relative to the repository root.

## Context

The COMPASS pipeline models time-to-first-platinum `(t_platinum, PLATINUM)` from a treatment anchor, at landmarks 0/90/180d. Per-patient lab information is currently collapsed to six summary statistics in a single `groupby().agg()` at `survival_common/cohort.py:432-448`: `mean, min, max, last, n_observations, delta`.

**`delta` (last − first, gated at ≥2 obs) is the only trend signal in the entire pipeline.** It discards every intermediate measurement, is undefined for single-observation patients, and is dominated by noise in the two endpoint draws. A per-patient OLS slope once existed (`_compute_patient_lab_slopes`) but was deleted in commit `ff0a9ac` as collateral damage in an unrelated cleanup.

Meanwhile the association-testing half is complete and well-built: `run_univariate_nobs_adjusted_associations` (`survival_common/cox_models.py:374`) fits one Cox model per feature on `[feature_z, feature_missing, n_obs_z, age, *baseline]`, emitting HR-per-SD, CI, p, and BH q-values. It assumes a **strictly linear** feature→log-hazard relationship, which has never been checked.

This plan adds the two missing halves:

1. **Hierarchical GAM trajectory features** — a shrunk smooth per patient×lab replacing the two-point `delta`, yielding level / slope / curvature / atypicality at the landmark boundary.
2. **A nonlinearity test** — penalized-spline Cox to ask whether each feature's effect on the hazard is actually linear.

Intended outcome: trend features that use all observations and are defined for nearly every patient, plus a principled answer to "is this lab's association with time-to-platinum monotone-linear, or is there a threshold / U-shape the current model is blind to?"

## Decision: mgcv for the modeling stages, Python for everything else

The two chosen designs (hierarchical smooths + a smooth term inside the Cox model) **cannot be expressed in `pygam` or `statsmodels`**:

| Requirement | pygam | statsmodels | mgcv |
| --- | --- | --- | --- |
| Hierarchical / per-patient shrunk smooths | no random effects, no factor-smooth basis | `GLMGam` has no random effects | `bs="fs"` factor-smooth interaction |
| Penalized-spline Cox | **no survival family at all** | no Cox family | `family=cox.ph()` with EDF + p-values |
| Automatic smoothness selection | grid search on λ | limited | REML / fREML |

Installing pygam/statsmodels would therefore not enable the chosen work. `mgcv` is already installed and is the reference implementation for both. Python keeps everything else — cohort construction, feature merge, the existing univariate/multivariable Cox, and all downstream artifacts.

**Constraint:** write the new R against **base R + `mgcv` + `data.table` only.** The existing `COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R` cannot even load locally (`tidyverse`, `survminer`, `broom` all missing). Do not touch that file and do not inherit its dependencies.

---

## Step 0 — Preflight

**Local: already confirmed.** R 4.4.0, `mgcv` 1.9-1, `data.table` 1.15.4, and `mgcv::cox.ph()` constructs successfully. Both Step 1 and Step 3 are locally runnable end-to-end, so development and the smoke test do not depend on cluster access.

**Cluster: still outstanding.** Before running against real data at `/data/gusev/USERS/jpconnor/`, confirm:

```bash
Rscript -e 'library(mgcv); library(data.table); cat(R.version.string, as.character(packageVersion("mgcv")), "\n")'
```

If `Rscript` or `mgcv` is unavailable there, the fallback is a hand-rolled mixed-model spline in Python — a materially different plan.

---

## Step 1 — Hierarchical trajectory smooths (new R script)

**New file:** `COMPASS/survival_analysis/gam_trajectory_features.R`

**Inputs** (all already produced by `COMPASS/data_preprocessing/build_prediction_inputs.py` into `prediction_inputs/`):

- `pre_treatment_lab_long_landmark{D}.csv` — already windowed to `t_lab < landmark` and restricted to cohort MRNs by `build_pre_treatment_lab_long` (`survival_common/cohort.py:297`). Columns: `DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab`. **Use this file directly** — it guarantees the GAM window matches the existing features exactly, with no immortal-time exposure.
- `canonical_labs_train_val.csv` — restrict to canonical labs; do not fit hundreds of junk labs.
- `split_assignments_landmark{D}.csv` — for the optional leakage sensitivity mode.

**Model, fit once per lab:**

```r
m <- bam(LAB_VALUE ~ s(t_lab, k = k_pop, bs = "tp") +
                     s(t_lab, MRN_f, bs = "fs", k = k_pat, m = 1),
         data = lab_dt, method = "fREML", discrete = TRUE, nthreads = nthreads)
```

`bs="fs"` is the hierarchical piece: all patients share one smoothing parameter, so patients with few observations shrink toward the population curve rather than producing a wild independent fit. This is what keeps coverage high.

**Scalability risk — plan for it.** `bs="fs"` with thousands of patient levels is mgcv's known bottleneck. Mitigations, in order: `bam(discrete=TRUE)` (required, not optional), small `k_pat` (4–5), `k_pop` ~10. If a lab still fails to converge in reasonable wall time, fall back to a random intercept + random slope deviation around the population smooth:

```r
s(t_lab, k = k_pop) + s(MRN_f, bs = "re") + s(MRN_f, t_lab, bs = "re")
```

Record which parameterization was used per lab in a `gam_fit_diagnostics_landmark{D}.csv` sidecar (lab, n_patients, n_obs, basis used, EDF, fit seconds, convergence flag). Do not let a silent fallback go unrecorded.

**Feature extraction.** Predict each patient's fitted curve on a dense grid; evaluate at `t* = landmark_offset_days − ε` (the right edge of the window — the pre-landmark filter is a strict `<`). Emit, using the pipeline's mandatory `<LAB>__<stat>` naming:

| Column | Meaning |
| --- | --- |
| `<LAB>__gam_level` | fitted value at `t*` — a denoised replacement for `__last` |
| `<LAB>__gam_slope` | first derivative at `t*` (central difference on the grid) — the denoised replacement for `__delta` |
| `<LAB>__gam_curvature` | second derivative at `t*` — acceleration / deceleration |
| `<LAB>__gam_auc` | mean of the fitted curve over the trailing window (default 180d) — smoothed cumulative exposure |
| `<LAB>__gam_dev` | RMS of the patient-specific deviation term over the window — "how atypical is this trajectory vs. the population" |

**Output:** `prediction_inputs/gam_trajectory_features_landmark{D}.csv`, one row per `DFCI_MRN`, plus the diagnostics sidecar.

**Leakage stance (state this explicitly in the script docstring).** The smooth is unsupervised — it never sees `t_platinum` or `PLATINUM`. Default to fitting on all cohort patients. This is a different category from the README's "canonical labs are training-block artifacts" invariant, which governs outcome-adjacent *selection* decisions. Provide `--fit-split train_val` to refit population smooths on train+valid only as a sensitivity check, and note the result in the commit message.

**CLI:** follow the repo convention — `optparse` or plain `commandArgs`, with flags for `--inputs-dir`, `--landmark-days`, `--k-pop`, `--k-pat`, `--trailing-window-days`, `--nthreads`, `--fit-split`. Hardcode the same cluster defaults used in `COMPASS/survival_analysis/cox_aggregated.py:78-79`.

## Step 2 — Merge into the aggregated table (small Python change)

The pipeline is designed so this needs almost nothing. Verified facts:

- `prepare_landmark_context` sets `raw_feature_cols = [c for c in merged.columns if c not in outcome_columns()]` (`COMPASS/survival_analysis/cox_aggregated.py:388`) — **any new column is automatically a candidate feature.**
- `matching_n_obs_feature` uses `rsplit("__", 1)` (`survival_common/cox_engine.py:71`), so `Hemoglobin__gam_slope` → `Hemoglobin__n_observations`, which already exists. The n_obs adjustment works for free — and it matters here, since measurement frequency confounds trajectory shape.
- `parse_feature` / `assign_category` (`survival_common/plotting.py:140`) and the R volcano labeller both split on `__` and need no change.

**Change:** add an optional GAM merge to `load_prebuilt_landmark` (`survival_common/cox_models.py:1082`). IPIO's adapter never passes this kwarg, so a `None` default makes the IPIO arm a no-op **by construction**, not merely by convention:

```python
gam_feature_filename: Callable[[int], str] | None = None,
```

When provided and the file exists, read it, set the index to `id_col`, and left-join onto `aggregated` **before** the split partition. Log the number of columns merged. When absent, behave exactly as today — this must be a no-op for anyone who hasn't run Step 1.

Wire it through the COMPASS adapter `_load_prebuilt_landmark` (`COMPASS/survival_analysis/cox_aggregated.py:299`), with a module-level `gam_feature_filename(landmark_day)` helper matching the existing `aggregated_filename` / `pre_treatment_lab_filename` pattern (`COMPASS/data_preprocessing/build_prediction_inputs.py:98`).

**No other Python changes are needed.** `select_feature_columns` applies its normal coverage (≥0.20) and variability gates; `run_univariate_nobs_adjusted_associations` tests the new features and BH-corrects alongside the existing ones; results land in `cox_agg_univariate_nobs_adjusted.csv` with correct `lab_name` / `feature_stat` parsing.

**Watch:** BH q-values are computed over every feature in the frame (`survival_common/cox_models.py:579`). Adding 5 stats × N canonical labs materially increases the multiple-testing burden and will shift existing q-values. Expect this, and report the before/after count of q<0.05 hits so the change is not mistaken for a regression.

## Step 3 — Nonlinear hazard test (new R script)

**New file:** `COMPASS/survival_analysis/gam_cox_nonlinearity.R`

Reads the post-merge `aggregated_landmark{D}.csv`.

**Which rows to fit on — this is load-bearing and easy to get wrong.** Trace: `survival_common/cox_runners.py:218` passes `ctx.univariate_data`, and `COMPASS/survival_analysis/cox_aggregated.py:389` sets `univariate_data = merged.copy()`. So the Python univariate pipeline **fits on all splits (train + valid + test)**, while feature *eligibility* (which columns are tested) is gated on train_val-only coverage inside `select_feature_columns`. `gam_cox_nonlinearity.R` must replicate exactly this asymmetry — fit on all rows, restrict the feature list to those `select_feature_columns` chose. If it fits on train_val only, verification item 4 below cannot match and you will chase a phantom bug.

**Adjustment set.** `survival_common/projects/compass_profile.py` does not override `static_covariates`, so it falls through to `no_static_covariates` → `()` (`survival_common/config.py:30,62`). COMPASS therefore has **no baseline covariates** — the R model is exactly `s(x) + n_obs_z + age` plus the optional `x_missing` indicator, and nothing else. Do not add gender / cancer-type terms; that is the IPIO arm's configuration, not this one.

Per selected feature, fit both a smooth and a linear model and compare:

```r
d$x <- x_imputed; d$x_missing <- as.numeric(is.na(x))   # mirror the Python treatment
mod_s <- gam(t_platinum ~ s(x, k = 10) + n_obs_z + age + x_missing,
             family = cox.ph(), weights = PLATINUM, data = d, method = "REML")
mod_l <- gam(t_platinum ~ x + n_obs_z + age + x_missing,
             family = cox.ph(), weights = PLATINUM, data = d, method = "REML")
```

Mirror the Python model's preprocessing exactly — z-scored feature, mean imputation, `feature_missing` indicator, `n_obs_z`, standardized age — so the linear fit's coefficient can be cross-checked against `coef_feature` in `cox_agg_univariate_nobs_adjusted.csv`. That cross-check is the correctness test for this step.

Apply the same guards as the Python path: skip features with `< min_events_per_feature` (10) events. `t_platinum > 0` is already guaranteed by `make_outcome_df`'s validity filter.

**Output:** `gam_cox_nonlinearity_landmark{D}.csv` with `landmark_days, endpoint, feature, lab_name, feature_stat, n_used, n_events, edf, p_smooth, p_lrt, delta_aic, coef_linear, p_linear, note`, then BH over `p_lrt` → `q_lrt` using base R's `stats::p.adjust(method="BH")` (no new dependency; matches the hand-rolled `benjamini_hochberg` at `survival_common/cox_engine.py:76`).

Interpretation: `edf ≈ 1` means the existing linear model is fine; `edf` meaningfully >1 with small `q_lrt` flags a lab whose current HR-per-SD is misleading.

**mgcv notes.** `cox.ph` takes the survival time as the response and the event indicator as `weights`; it uses Peto's correction for ties. `anova(mod_l, mod_s, test="Chisq")` on penalized GAM fits is an *approximate* LRT — the EDF-based reference distribution is conservative. Report `delta_aic` alongside `p_lrt` rather than relying on either alone, and note the approximation in the script docstring.

---

## Files touched

| File | Change |
| --- | --- |
| `COMPASS/survival_analysis/gam_trajectory_features.R` | **new** — hierarchical smooths → trajectory features |
| `COMPASS/survival_analysis/gam_cox_nonlinearity.R` | **new** — penalized-spline Cox, EDF + LRT |
| `COMPASS/survival_analysis/tests/test_gam_trajectory_features_smoke.R` | **new** — synthetic self-checking smoke test |
| `survival_common/cox_models.py` | optional `gam_feature_filename` merge in `load_prebuilt_landmark` (~15 lines) |
| `COMPASS/survival_analysis/cox_aggregated.py` | `gam_feature_filename()` helper; pass through `_load_prebuilt_landmark` |
| `README.md` | document the two new stages, their place in the pipeline order, and the leakage stance |

Deliberately unchanged: `survival_common/cohort.py` (the GAM does not run inside `build_feature_matrix` — it needs R, and keeping it a separate stage means `build_prediction_inputs.py` stays pure Python and fast), `COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R`, and the entire IPIO arm.

## Verification

1. **Preflight** — Step 0 above.

2. **Synthetic smoke test (runs anywhere, no cluster data).** Generate ~200 fake patients × 3 labs with known trajectory shapes — one planted downward slope, one planted upward, one pure noise — write them in the `pre_treatment_lab_long_landmark{D}.csv` schema, run `gam_trajectory_features.R`, and confirm: the output has one row per MRN, `__gam_slope` recovers the planted signs, and the noise lab's slope is centered on zero. This is the primary correctness gate, and local R can run it today.

   Put it at `COMPASS/survival_analysis/tests/test_gam_trajectory_features_smoke.R` — a plain Rscript, self-checking, non-zero exit on failure. The repo's only existing test (`data_preprocessing_common/tests/`) is pure-Python `unittest` with no precedent for shelling out to R, so do not try to wrap this in Python. There is no pytest config and no CI; this is meant to be run by hand.

3. **Coverage check** — on real data, compare `notna().mean()` of `<LAB>__gam_slope` vs. `<LAB>__delta` per lab. The hierarchical fit should be defined for strictly more patients; if it isn't, the shrinkage isn't working and the `fs` basis silently fell back.

4. **Cross-check against the existing pipeline** — confirm `coef_linear` from Step 3's `mod_l` matches `coef_feature` in `cox_agg_univariate_nobs_adjusted.csv` for the same feature/landmark to within numerical tolerance. A mismatch means the preprocessing was not mirrored.

5. **No-op check** — run `univariate_analysis.py` with no GAM feature file present; output must be byte-identical to the current `cox_agg_univariate_nobs_adjusted.csv`.

6. **End-to-end** — `build_prediction_inputs.py` → `gam_trajectory_features.R` → `univariate_analysis.py` → `gam_cox_nonlinearity.R` at landmarks 0/90/180. Report the top `__gam_*` hits by q-value and the labs with `edf > 1` and `q_lrt < 0.05`.

## Open risk to confirm during execution

`bs="fs"` runtime at real cohort scale is the one thing that could force the fallback parameterization. Time a single lab at landmark 0 before running the full sweep; if one lab exceeds a few minutes, switch to the random intercept+slope form for all labs and record that in the diagnostics sidecar rather than mixing parameterizations across labs.
