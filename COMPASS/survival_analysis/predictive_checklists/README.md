# Predictive checklists for the full ADT cohort

This subproject applies [Zhang et al., Learning Optimal Predictive Checklists
(NeurIPS 2021)](https://arxiv.org/html/2112.01020v1) to first platinum exposure
and first documented NEPC finding. Each model is a short, unit-weighted M-of-N
checklist: flag a patient if at least M of its N conditions are true.

The paper's objective is binary classification. Our **local IPCW extension**
learns an independent checklist for event status at each future horizon; it is
not a native continuous-time survival model. Calibrated horizon probabilities
can cross across independently fitted horizons. Do not interpolate them into
a survival curve, infer an exact event date, or interpret them as competing-risk
cumulative incidence.

## Scope and inputs

Defaults are `arm=adt`, `cohort=all`, `exclusion=none`, landmarks 0/90/180 days,
and prediction horizons 365/730/1095/1460/1825 days (1–5 years) **after each
landmark**; horizons at/beyond the input manifest evaluation cap (522 weeks =
3654 days by default, matching the 3650-day administrative censoring) are
recorded as unsupported. Day 0 is the
primary analysis. Later landmarks condition on remaining event-free and observed;
they do not restrict day-0 membership. Existing Stage-1 eligibility rules remain.

The runner reads the existing separate input trees:

```
<COMPASS data root>/survival_analysis/
    prediction_inputs_adt/
    prediction_inputs_adt_nepc/
```

Each needs `build_manifest.json` (timeline schema 3) and
`aggregated_landmark{0,90,180}.csv`. Durations are already landmark-relative days;
the adapter never rebases them. It rejects non-ADT/subcohort/exclusion inputs,
wrong endpoint provenance, duplicate patient IDs, and inconsistent splits.
It validates all requested input sets before fitting anything. Both endpoints
retain their own risk sets and existing splits; an individual's split need not
match *across* endpoints. Do not pool these endpoint predictions as independent
patients or describe cross-endpoint performance as a paired comparison.

Older input manifests may incorrectly say `endpoint=avpc` due to the builder's
former loop-variable shadowing. Rebuild those inputs using the corrected
`COMPASS/data_preprocessing/build_prediction_inputs.py` / notebook 01. Do not
simply relabel an unknown input manifest.

NEPC uses the existing any-gated-finding tier, not the stricter stated-diagnosis
tier. Current upstream preprocessing censors unresolved/absent labels and
undated positives as non-events. This convention is retained in the primary
full-cohort run and is an ascertainment limitation: absent labels are not proof
of absence. The runner audits review provenance and additionally fits a
`resolved_nepc` sensitivity analysis. Its conservative completed-status whitelist
is recorded in `input_audit`; unknown statuses are excluded from that sensitivity
only. A missing provenance column produces an explicitly empty sensitivity set.
The primary endpoint times retain existing last-contact censoring conventions;
death is not modeled as a competing event here.

## Method and leakage boundaries

1. At horizon h, an observed event at/before h is positive; follow-up strictly
   beyond h is negative. Censoring at/before h is unknown, with weight zero.
   Unknown patients remain in cohort counts and censoring estimation.
2. Reverse Kaplan–Meier estimates the censoring distribution G using training
   data only. Cases receive 1/G(event time); controls receive 1/G(h). Tied times
   follow scikit-survival's reverse-KM convention. This assumes independent
   censoring. Horizons outside training support or below `min_g` are rejected;
   weights are never silently clipped. Requested horizons must also be strictly
   below the build manifest's administrative metric cap.
3. Candidates include age and the established mean/min/max/last/delta/count
   lab summaries. Training quantiles become raw-unit `<=` / `>=` rules.
   Missing measurements use training medians, with additional missingness
   candidates. Patients without labs remain included. Only one item from each
   lab (or age) can be selected. Exported JSON contains every imputation value
   and candidate; the printable checklist names selected conditions.
4. Train+valid is the development block. Stratified CV refits feature coverage,
   imputation, candidate thresholds, and G within each training fold. It compares
   maximum lengths 3/5/8 using IPCW class-balanced misclassification, rejecting
   a candidate if any fold fails. The optimizer sequentially minimizes weighted
   error, size, and M, with an explicit numerical loss tolerance. Real-valued
   weights are summed when duplicate observations are compressed.
5. A monotone logistic probability map is fitted to development out-of-fold
   score fractions using unbalanced IPCW. Size selection and calibration use
   the same development folds (not nested CV); those CV scores are selection
   diagnostics, not unbiased performance claims. The selected checklist is
   refitted on all development patients. Test patients never fit any component.
6. Held-out metrics include time-dependent AUC on counts, weighted sensitivity,
   specificity, PPV/NPV, Brier score on calibrated probabilities, and calibration
   bins. Brier divides by all test patients, including unknown horizon outcomes.
   Patient-bootstrap intervals condition on the fitted models and training G;
   they do not include training/selection uncertainty. Unsupported metrics carry
   statuses/reasons, not substituted zeros. No integrated full-grid Brier/AUC or
   single C-index is claimed for these independent horizon classifiers.

The age-only Cox, elastic-net Cox and XGBoost survival:cox comparators are
locally refitted on these same development patients and evaluated on the same
test patients/horizons/estimand. These are matched comparator fits, not reused
published-model metrics. Their internal CV refits numeric preprocessing and
coverage. Survival probabilities use training-only Breslow baselines. Missing
optional dependencies produce explicit skipped comparator records. Comparator
binary flags use a fixed 0.5 probability cutoff, recorded in metadata; compare
AUC/Brier as the primary model comparisons, not optimized versus unoptimized
binary operating points.

## Install and run

From the repository root, in the study environment:

```bash
python -m pip install -r COMPASS/survival_analysis/predictive_checklists/requirements.txt

# Inspect both endpoints and all three landmarks before training.
python -m COMPASS.survival_analysis.predictive_checklists \
  --inputs-root /data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis \
  --output-dir /data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/predictive_checklists/preflight \
  --preflight-only

# Full approved run; all settings are captured in the output manifest.
python -m COMPASS.survival_analysis.predictive_checklists \
  --config COMPASS/survival_analysis/predictive_checklists/default_config.json \
  --output-dir /data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/predictive_checklists/full_adt
```

`--platinum-inputs` and `--nepc-inputs` override individual trees. For a baseline
pilot use `--landmarks 0 --horizons 365`; `--no-baselines` and
`--no-nepc-sensitivity` explicitly disable optional comparisons. The local
`run_predictive_checklists.ipynb` supplies the same commands. Defaults allow 60
seconds **per checklist fit**, shared across optimization stages; total runtime
depends on folds, lengths, horizons and populations. No full-cohort runtime or
event counts are assumed until preflight and a pilot have run on cluster data.

An existing output run is refused by default. `--overwrite` archives its managed
results under `previous_runs/` before writing a new run, preventing stale models
from masquerading as new fits. `--resume` instead continues an interrupted run in
place: each endpoint × landmark × population × horizon with a finished
`horizon_<days>/metrics.json` is reused (including recorded unsupported or failed
fits), partially written horizons are cleared and refitted, a cell's baselines are
reloaded from `baselines.pkl` rather than refitted, and the run-level metrics are
rebuilt from all horizons. Resuming is refused unless the configuration,
package source and input fingerprints match the existing `run_manifest.json`;
without an existing run it starts fresh. Exit code 0 indicates supported evaluated models
(or successful preflight), 1 invalid inputs, and 2 no supported evaluated models.
Inspect metric statuses even after exit 0: some requested horizons can fail.

Synthetic smoke run (explicitly labeled synthetic in every source manifest):

```bash
python -m COMPASS.survival_analysis.predictive_checklists \
  --demo --output-dir /tmp/checklists-demo --landmarks 0 --horizons 365 \
  --n-folds 2 --bootstrap-samples 20 --solver-time-limit 10
```

## Gleason + TP53/RB1/PTEN subset

`run_predictive_checklists_gleason_trio.ipynb` with `gleason_trio_config.json`
fits 5-year platinum and NEPC checklists on each endpoint's
`somatic_gleason/gleason_somatic_available_case/` inputs (passed via
`--platinum-inputs` / `--nepc-inputs`). That subset is anchored on the
sequencing report date, not ADT start: landmarks are sequencing + D, and
age is age at sequencing. The config's `clinical_features` adds
`GLEASON_SCORE` and `TP53_ANY`/`RB1_ANY`/`PTEN_ANY` to the lab and age
candidates. Each is its own group and is exempt from the lab coverage rule.
Low-cardinality columns are thresholded at every observed training value. The
matched comparators receive the same columns. A configured column that is absent
from an input is an error. `min_events` is 5 because NEPC events are sparse in
this subset, and the resolved-label NEPC sensitivity is off.

## Solvers and upstream provenance

The default HiGHS backend comes with SciPy's `milp`; CBC (`--solver cbc`) and
CPLEX (`--solver cplex`) are optional. See `optional_requirements.txt` and
`upstream.py` for the pinned authors' package and reference-loading interface.
The weighted formulation is local code, **not an unmodified upstream model**.
The original package lacks a per-patient sample-weight API. `UPSTREAM_COMMIT`
records the source reference. Upstream setup declares BSD but its pinned tree
has no license-text file; we reference/install the package rather than vendor
its source into this repository.

Every fit reports solver status, incumbent, lower bound, gap, stage diagnostics,
and whether lexicographic optimization completed. If the time budget expires
before any incumbent, a deterministic single-item fallback is explicitly marked
as heuristic. A solver gap certifies optimization of the finite candidate set
and weighted training objective; it says nothing about generalization.

## Outputs and verification

Top-level `run_manifest.json` records configuration, versions, source manifests
and SHA-256 hashes. `preflight.json`, `metrics.{csv,json}`, and `calibration.csv`
summarize all cells. Each endpoint/landmark/population has a cohort audit and
baseline metadata; each supported horizon has:

- `model.json`: portable checklist, features, calibration and censoring state.
- `checklist.{txt,json}`: readable items and decision threshold.
- `cv_folds.json`, `horizon_audit.json`, `calibration_oof.csv`: training diagnostics.
- `predictions.csv`: explicitly separated test and in-sample development scores.
- `predictions_<baseline>.csv`: matched test comparator predictions.
- `failure.json` on unsuccessful fits; metrics retain the failure reason.

Patient-level outputs belong in the study data directory, outside source control.
`baselines.pkl` is a local trusted-environment artifact; portable checklist
inference uses JSON via `inference.predict_bundle(path, frame)` without refitting.

```bash
python -m pytest -q tests/test_predictive_checklists_*.py
Rscript COMPASS/survival_analysis/predictive_checklists/render_figures.R /tmp/checklists-demo
```

### Checklist supplement

For the manuscript supplement (platinum and NEPC, 5-year horizon, each landmark, labs only
versus labs + Gleason + somatic), fit `default_config.json` and `gleason_trio_config.json` with
`"horizons": [1825]`. The cached figure workflow picks the runs up on its own: notebook `06` flattens them
(defaults `<data root>/survival_analysis/predictive_checklists/{full_adt,gleason_trio_labs}`, override with
`COMPASS_CHECKLIST_LABS_RUN` / `COMPASS_CHECKLIST_TRIO_RUN`) and `07_figures/all.R` renders the figures into
`04_multivariate/internal`. To preview two runs without the figure cache, flatten and draw them directly:

```bash
python -m COMPASS.survival_analysis.predictive_checklists.collect_supplement \
    --labs-run <labs-run> --trio-run <gleason-trio-run> --output-dir <supp-dir>
Rscript COMPASS/survival_analysis/predictive_checklists/render_supplement.R <supp-dir>
```

`collect_supplement.py` writes `checklist_cells.csv` (status, M-of-N threshold, test n),
`checklist_items_long.csv` (every item) and `checklist_metrics.csv` (checklist and baseline
held-out AUC/Brier with bootstrap intervals). The renderer writes one
`figure4s_checklist_<endpoint>_<feature_set>` figure per endpoint and feature set with a card per
landmark; cells that were not fit say why. The labs + Gleason + somatic run uses the
sequencing-anchored available-case cohort, so its metrics are not directly comparable with the
labs-only run.

The R renderer exports AUC/Brier panels and calibration plots. Tests cover
weighted exhaustive optima, duplication, time-limit status, upstream reference
parity when installed, censoring/ties against scikit-survival, fold boundaries,
endpoint provenance, synthetic two-endpoint runs and JSON inference round trips.
