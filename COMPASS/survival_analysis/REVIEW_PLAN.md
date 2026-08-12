# End-to-end code review: COMPASS survival analysis

## Context

The COMPASS survival pipeline (~9,500 LOC) produces publication-bound results for a
prostate-cancer cohort at DFCI, modeling time-to-first-platinum from ADT entry across
three landmarks (0/90/180d) and four model arms (Cox univariate, Cox elastic-net,
XGBoost, and longitudinal DeepHit/SurvLatent).

Two structural facts make a review worth doing now:

1. **The metric core is untested.** `survival_common/helper.py` (718 lines) and
   `cox_engine.py` (634) compute every AUC(t), Brier, and horizon grid the paper will
   report, and have zero test coverage. So do `cox_models.py` (1,162) and all of
   `COMPASS/survival_analysis/`. The two named leakage guards
   (`assert_no_test_leakage`, `assert_disjoint_folds`) are themselves untested.
2. **Exploration surfaced a documented-contract violation.** `cox_engine.py:196-201`
   states the evaluation split "is used only to decide which requested points are
   estimable, never to create replacement horizons," and the mean is "always sksurv's
   censoring-weighted integral … never a substitute summary." `deephit_engine.py:496`
   builds a replacement timeline and `:504` substitutes a time-constant risk. DeepHit's
   `mean_auc_t` is therefore not comparable to the Cox/XGBoost numbers it is printed
   beside.

Intended outcome: a severity-ranked findings report covering methodology, correctness,
duplication, and test gaps; a small set of unambiguous defects fixed with tests; and a
reconciliation of the README's ~15 "Known issues" against current code. Judgment calls
that would change a published number's definition are reported, not silently changed.

**Scope confirmed with user:** all four arms + shared core; re-examine (don't trust) the
README's accepted items; deliver findings *and* apply high-confidence fixes.

## Two corrections to carry into the work

These came out of exploration and are worth stating because they set the review's standard
of evidence:

- **The 10-year administrative censoring IS active.** An exploration pass reported
  `max_followup_days` defaulting to `None` (no censoring). That is the *library* default
  at `survival_common/cohort.py:100`, but the actual entry point sets
  `DEFAULT_MAX_FOLLOWUP_DAYS = 3650.0`
  ([build_prediction_inputs.py:91](COMPASS/data_preprocessing/build_prediction_inputs.py#L91))
  and passes it as the argparse default (`:918`). **Verify every claim at the entry point,
  not the library default.**
- **`survlatent_ode.py` is at
  [COMPASS/survival_analysis/multivariate_longitudinal/survlatent_ode.py](COMPASS/survival_analysis/multivariate_longitudinal/survlatent_ode.py)**,
  not in `survival_common/`.

## Stage 0 — Baseline

Record the current test baseline: **81 collected, 79 pass, 2 fail**. Both failures are
pre-existing and unrelated to survival code — `MM/DD/YYYY` parsing in
[test_mixed_date_parsing.py](COMPASS/data_preprocessing/tests/test_mixed_date_parsing.py)
(worth reporting separately: it silently drops patients from medication anchors upstream).

Work **in-place on `main`**, one commit per fix. Do not use worktree-isolated agents —
the repo carries uncommitted work and worktrees branch off stale HEAD. The 3 currently
dirty files are all in `data_preprocessing/`; the survival surface is clean.

## Stage 1 — Metric core first

Review [helper.py](survival_common/helper.py) and [cox_engine.py](survival_common/cox_engine.py)
before any arm. They are shared by every model, so a defect there multiplies across all
reported numbers, and every later finding can then be stated as "diverges from core"
rather than re-derived.

Extract the **oracle contract** from `cox_engine.py:194-207` + README:494 and audit every
arm against it:
- (a) horizons come from the manifest grid, never re-derived from eval data
- (b) admin censoring at `auc_max_time_units` applied to reference and eval alike
- (c) IPCW reference distribution = train_val only
- (d) mean = sksurv's weighted integral over valid requested horizons, with the
  ≥2-points/≥50%-coverage guard — never a substitute

## Stage 2 — Synthetic verification harness

Real data is cluster-only (`/data/gusev/...`), so everything must be verifiable
synthetically. Reading code cannot confirm a metric bug; these can.

**2a. Analytic-truth and invariant generators**
- Uninformative risk ⇒ AUC(t) ≡ 0.5 at every horizon. Catches sign flips and weight bugs.
- Perfect predictor ⇒ AUC(t) ≡ 1.0. Note `deephit_engine.py:443` passes `-risk` to
  concordance but bare `risk` to `cumulative_dynamic_auc` at `:480` — **check sign
  convention consistency**; sksurv expects higher = higher risk.
- **Censoring-rate invariance:** same event process at 20/50/70% independent censoring
  ⇒ IPCW AUC should be ~stable. This is the test that proves or kills Finding 2, since
  mislabeling competing events as censored biases the KM censoring estimator in exactly
  this dimension.

**2b. Differential testing — the highest-value technique**

Build a synthetic `pred` frame with *every* `event_1_risk_h{t}` column materialized on the
manifest grid, single (non-competing) event, then compare DeepHit's metric path against
`cox_engine.compute_ipcw_auc_t` on identical inputs. They should agree to floating-point
tolerance; they will not, and that failure **is** the proof of Finding 1. Then drop a
subset of `risk_h{t}` columns: the Cox path masks non-estimable horizons, DeepHit
substitutes constant risk. Assert disagreement before the fix and agreement after — a
genuine red/green transition, which is what makes the fix safe. Repeat for Brier
(`deephit_engine.py:573-579` vs `helper.py:391`).

**2c. Property checks on the shared core**
- `compute_horizon_grid` ([helper.py:221](survival_common/helper.py#L221)): output positive,
  sorted, deduplicated, ≤ `max_grid_points`, clamped to `admin_censor_days` — all claimed
  in the docstring, none tested.
- **Leakage-guard dtype hazard.** [helper.py:693-694](survival_common/helper.py#L693) does
  `{str(m) for m in ...}`. A float MRN (`12345.0`, a routine pandas outcome after a merge
  introduces NaN) stringifies to `"12345.0"` ≠ `"12345"`, so **the guard silently passes on
  genuine leakage**. High-value test on the repo's only safety net.

Follow the existing convention: tests in `tests/`, `sys.path` bootstrap as in
[test_admin_censoring.py:20-25](tests/test_admin_censoring.py#L20-L25). No `conftest.py`
exists; don't add one (it changes collection for the whole suite).

## Stage 3 — Arm-by-arm audit

Ordered so each stage reuses the prior verdict:
1. **Cox** — the reference implementation (`cox_models.py`, `cox_runners.py`, `cox_aggregated.py`)
2. **XGBoost** — diff its metric calls against Cox (`multivariate_analysis.py` local impl)
3. **Longitudinal** — the largest divergence (`deephit_engine.py`, `survlatent_ode.py`)
4. **Orchestration + duplication** (`compass_pipeline.py`, and the dead `xgboost_runners.py`)

## Stage 4 — Triage rule and verdicts

**FIX** requires all three: (i) violates an explicitly documented invariant or docstring
contract; (ii) correct behavior is uniquely determined — no scientific choice to make;
(iii) a failing synthetic test can demonstrate it before and after.

**REPORT** if any of: it changes a published number's *definition*; more than one
defensible answer exists; validating it requires a cluster re-run; or it's a scope decision.

| # | Finding | file:line | Verdict |
|---|---|---|---|
| 1 | DeepHit `np.arange` grid + constant-risk substitution | `deephit_engine.py:495-506`, `:573-579` | **FIX** — violates `cox_engine.py:196-201` verbatim |
| 2 | IPCW ref treats competing events as censoring | `deephit_engine.py:450-453` | REPORT + patch + evidence test |
| 3 | Tie-break prefers *least* regularization; `cv_std` computed unused | `cox_models.py:816-825`, `:780` | **FIX tie-break** / REPORT 1-SE rule |
| 4 | Selection on uncapped Harrell C vs reported IPCW/capped metrics | `cox_models.py:779`, `:816-825` | REPORT — highest-severity report item |
| 5 | Univariate runs on full cohort incl. test; BH q on train-filtered features | `cox_aggregated.py:423`, `cox_models.py:582` | REPORT |
| 6 | Early-stopping watch set == metric set | `deephit_engine.py:677-678`, `xgboost_runners.py:246-260` | REPORT |
| 7 | survlatent has no `assert_no_test_leakage` | `survlatent_ode.py:67-73`, `:479-481` | **FIX** — cannot change results |
| 8 | `xgboost_runners.py` (911 lines) entirely unused | — | REPORT (recommend delete) |
| 9 | `global ID_COL, AGE_COL` rebinds module-local copies only | `multivariate_analysis.py:638-642` | REPORT — README:502 documents it deliberately |
| 10 | Final model features ≠ any CV fold's; `min_genomic_prevalence` inconsistent | `cox_runners.py:346`, `cox_aggregated.py:447`, `xgboost_runners.py:194-201` | REPORT mismatch / FIX prevalence **if confirmed an omission** |

Net: **fix 3, report 8.** That ratio is right for a publication-bound pipeline — most of
these are science decisions, not bugs.

Reframing worth carrying into the report: Finding 7 is one instance of a pattern — the
leakage guards prove *row* disjointness but never *artifact* disjointness (features, labs,
horizons fit on the wrong block). `cox_aggregated.py:426` calls the guard, but *before*
feature selection at `:447`.

## Stage 5 — Tests to add (priority-ordered)

1. Differential DeepHit↔Cox AUC/Brier agreement — guards the Finding 1 fix, locks the invariant
2. Leakage-guard dtype tests (the float-MRN case)
3. AUC invariants (uninformative → 0.5, perfect → 1.0, censoring-rate invariance)
4. `compute_horizon_grid` docstring properties as assertions
5. Tie-break determinism — two identical `cv_mean` rows differing only in penalizer

Skip: end-to-end pipeline tests (need cluster data), XGBoost fit tests (slow, low bug density).

## Stage 6 — Deliverable

One severity-ranked report. Per finding:
`ID | Severity | Arm | file:line | Claim | Evidence | Impact on published numbers | Verdict | Recommended action`

Three sections:
- **A. Affects published numbers** (1, 2, 4, 5, 6, 10) — the section that gates publication
- **B. Correctness/robustness** (3, 7, leakage-guard dtype)
- **C. Maintainability & test gaps** (8, 9, coverage map)

Plus a **README reconciliation table**: each "Known issues" item (README:565-601) marked
still-accepted / promote-to-fix / already-fixed / mis-stated. Argue explicitly for promoting
the competing-config Brier item (README:595) and the `t_platinum > 0` landmark-depletion
item (README:579) — both are filed "Medium impact" but bear directly on interpreting results.

## Stage 7 — Safe fix order

Strictly **test first (red) → fix → test (green) → full suite → commit**, one finding per commit:

1. **Finding 7** (add leakage guard) — cannot change results
2. **Finding 3** (tie-break) — changes results only on exact ties
3. **Finding 1** (DeepHit grid) — land last; it's the only fix with real numeric blast
   radius. Flag clearly that **DeepHit numbers must be re-run** on the cluster.

After each commit, re-run the full suite and confirm **exactly** the same 2 pre-existing
failures — no more, no fewer. That's the tripwire that the review didn't break anything.

## Verification

- `python3 -m pytest -q` from repo root after every commit → expect `2 failed, N passed`
  with N growing as tests are added, and the 2 failures unchanged
  (`test_mixed_date_parsing.py`).
- New synthetic tests must show the red/green transition described in Stage 2b — a fix
  landing green on a test that never failed proves nothing.
- Import-safety check (README invariant #7): `python3 -c "import survival_common.deephit_engine"`
  and `python3 COMPASS/survival_analysis/multivariate_analysis.py --help` must both succeed
  with torch absent.
- Cluster re-run of the DeepHit arm is **out of scope here** and is the user's call after
  Finding 1 lands.

## Open questions to resolve during the work

- Is the missing `min_genomic_prevalence` at `cox_aggregated.py:447` /
  `xgboost_runners.py:194-201` an intentional asymmetry or an omission? Changes Finding 10's verdict.
- Is `xgboost_runners.py` slated for IPIO convergence or genuinely dead? Changes the
  recommendation from "delete" to "document as canonical target."
