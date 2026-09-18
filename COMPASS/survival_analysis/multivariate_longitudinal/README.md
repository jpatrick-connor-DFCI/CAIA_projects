# multivariate_longitudinal

Two longitudinal deep survival models fit on the person-period lab history
(`longitudinal_landmark{D}.csv`, built by
`COMPASS/data_preprocessing/build_prediction_inputs.py --build-longitudinal`),
each in two configs:

## Two input frames, two prediction regimes

| frame | built by | ends at | landmark recoverable as |
| --- | --- | --- | --- |
| `longitudinal_landmark{D}.csv` | `--build-longitudinal` | the landmark | each patient's `max(TIME)` |
| `longitudinal_full_landmark{D}.csv` | `--longitudinal-full-followup` | end of follow-up | the explicit `landmark_time` column |

The full frame is written **alongside** the landmark frame, never in place of
it, and its manifest carries `include_post_landmark: true`. Consumers
cross-check that flag against their own and refuse a mismatch in either
direction, because post-landmark labs are legitimate inputs only for
predictions made *after* the landmark.

> **The `max(TIME)` trap.** On the landmark frame `max(TIME) == landmark_time`,
> so code can and historically did infer the landmark that way. The full frame
> breaks that identity: `max(TIME)` becomes the patient's last follow-up visit.
> Anything anchoring to it there shifts per patient by the length of their own
> post-landmark history — silently, and with plausible-looking output. Read
> `landmark_time` explicitly on the full frame. This bit both
> `build_person_period_wide` and `add_post_landmark_horizon_columns`; both are
> now pinned by tests.

- `platinum` -- death censored, directly comparable to the existing
  univariate / elastic-net / XGBoost arms (same cohort, same AUC(t) horizon
  grid, same censoring).
- `competing` -- platinum and death as competing causes
  (`0=censored, 1=platinum, 2=death`).

See `survival_common/longitudinal_targets.py` for the shared config/target
semantics both models consume, and the root plan doc for the full design
rationale.

## `dynamic_deephit.py`

Thin wrapper around `survival_common.longitudinal_runners` /
`survival_common.deephit_engine`. Torch-gated: `--help` works without torch
installed; running the arm requires torch (see `requirements.txt`).

Training automatically uses the first CUDA GPU when PyTorch detects one and
falls back to CPU otherwise.

```bash
python dynamic_deephit.py \
  --inputs-dir <prediction_inputs_dir> --output-dir <out> \
  --landmark-day 0 --config platinum
```

### `--dynamic`: a prediction at every timepoint

Adds a second arm rather than changing the first. With `--dynamic` the unit of
prediction becomes one `(patient, prediction time)` pair instead of one patient:
the model emits a CIF at every observation time, using only history up to that
time. Requires the full-follow-up frame.

The two arms cannot collide — separate output directories
(`dynamic_deephit/` vs `dynamic_deephit_dynamic/`) and separate metrics
filename prefixes (`dynamic_deephit_` vs `dynamic_deephit_dyn_`), so each
resumes independently.

Causality is structural, not enforced by masking: the GRU is causal, so step
`t`'s output depends only on steps `<= t`. `tests/test_dynamic_no_future_leakage.py`
asserts it empirically by perturbing a lab at time `t` and requiring every
prediction before `t` to be unchanged.

Outputs, per invariant #9:

- `dynamic_deephit_dyn_metrics_{config}.csv` — the canonical block, one row per
  cause, at `prediction_time == landmark_time`. This is the row comparable to
  the Cox/XGBoost arms.
- `dynamic_deephit_dyn_metrics_by_time_{config}.csv` — one row per
  `(prediction_time, cause)`. A separate file precisely so the canonical
  metrics schema does not gain a prediction-time axis.

Prediction times whose risk set cannot support a metric get NaN metrics and a
`note` beginning `underpowered:`, rather than being dropped — a gap in the
series should say why it is missing.

```bash
python dynamic_deephit.py \
  --inputs-dir <prediction_inputs_dir> --output-dir <out> \
  --landmark-day 0 --config platinum --dynamic
```

## `incremental_risk.py`

Does accruing lab history sharpen the 6-month risk estimate? Pure
post-processing of the dynamic arm's predictions: **no refit, no torch**, so it
runs anywhere pandas does.

Two analyses, and the second is the one to lead with:

- **5a, sequential landmarks** (`incremental_risk_by_landmark.csv`) — AUC /
  Brier / C-index at a fixed horizon, by prediction time. Its trend is
  **confounded**: the risk set shrinks as the prediction time grows and is
  increasingly selected for patients who have not yet had the event, so rising
  AUC is not a clean "the model improves" claim.
- **5b, held-back-window ablation** (`incremental_risk_ablation.csv`) — at each
  prediction time, the model's full-history prediction against its own
  prediction from history truncated `delta` days earlier, on the *same*
  patients and the *same* horizon outcome. Only the input window differs, so
  `auc_gain > 0` is attributable to the newest labs.

5b needs no second fit or checkpoint because of the GRU's causality: the
prediction already emitted at step `t - delta` **is** the estimate from history
truncated there. The stale arm reads the horizon extended by `delta`
(`horizon + delta`), so both arms target the same absolute calendar point, and
a patient without a materialized step at `t - delta` is dropped rather than
back-filled.

Rows censored before the horizon are **excluded**, not counted as non-events:
counting them would bias risk downward, and bias it differently at different
prediction times — exactly the artifact the analysis exists to measure.

These files are deliberately named `incremental_risk_*.csv`, outside the
`*_metrics_*.csv` namespace invariant #9 governs: they report a *paired*
comparison indexed by prediction time, not one model's held-out performance.

```bash
python incremental_risk.py \
  --output-dir <dynamic_deephit_dynamic/landmark_0/platinum> \
  --horizon-days 182 --delta-days 90 --grid-days 0 90 180 270
```

The pipeline runs this automatically after the dynamic arm when
`RUN_DYNAMIC = True`; read the tables back with
`cp.load_incremental_risk_results(run, "ablation")`.

## `survlatent_ode.py`

Adapter around the editable in-repo checkout of
[`itmoon7/survlatent_ode`](https://github.com/itmoon7/survlatent_ode) at
`COMPASS/survival_analysis/survlatent_ode_repo/`.
Not a reimplementation -- this script imports and drives that repo's
`SurvLatentODE` class directly.

The adapter creates the upstream repository's empty runtime parents
(`model_performance/`, `surv_curves/`, and `experiments/`) before training;
fresh clones do not necessarily preserve these directories in Git.

### Prerequisites

1. **Use the bundled checkout** at
   `COMPASS/survival_analysis/survlatent_ode_repo/`. To recreate it if needed:
   ```
   git clone https://github.com/itmoon7/survlatent_ode.git \
     COMPASS/survival_analysis/survlatent_ode_repo
   ```
2. **Create and activate its conda env** from `survlatent_ode_conda.yml` in
   that repo (this pulls torch + `torchdiffeq`; see the root
   `requirements.txt` header -- torch stays commented out of this repo's own
   requirements because `multivariate_longitudinal/` is its only consumer).
3. The shared pipeline and notebook 03b default `--survlatent-repo` to that
   bundled checkout. Set `cp.SURVLATENT_REPO` in 03b only to override it.

### `os.chdir` side effect

`import_survlatent()` does `sys.path.insert(0, repo_path)` and
**`os.chdir(repo_path)`**, because the upstream `SurvLatentODE.fit()` /
`process_eval_data()` write checkpoints and performance logs to
`model_performance/<run_id>/` and `experiments/` *relative to the current
working directory*. This is why `main()` resolves `--output-dir` (and reads
`--inputs-dir` before the chdir) to an **absolute path** first -- a relative
`--output-dir` would otherwise land inside the cloned repo instead of your
intended results directory. Only `import_survlatent()` triggers the chdir,
and it is called from `main()` only, never at module import time.

### `--overwrite-run` / `--resume-run` semantics

Because checkpoints are keyed only by `run_id` (`model_performance/<run_id>/`,
`experiments/*_<run_id>.ckpt`) and that directory lives inside the repo's own
workspace rather than under `--output-dir`, reusing a `run_id` across
invocations can silently load a stale `best_model.pt` and report misleading
AUCs for what looks like a fresh run. `prepare_run_artifacts()` guards this:

- If no prior artifacts exist for `run_id`, training proceeds normally.
- If artifacts exist and neither flag is passed, it **raises** rather than
  silently reusing or silently overwriting.
- `--overwrite-run` deletes the prior `model_performance/<run_id>/` dir and
  any `experiments/*_<run_id>.ckpt` files before training.
- `--resume-run` allows training to reuse/append to the existing artifacts
  (intentional warm-start).

The default `run_id` is `prostate_<config>_landmark<D>_v1`, unique per
(config, landmark) pair so the four `compass_pipeline.py` task rows per arm
never collide by default. An observation cut appends `_cut<N>` (see below), so
each sweep point is its own fit; the no-cut spelling is unchanged, and existing
checkpoints keep their names.

```bash
python survlatent_ode.py \
  --survlatent-repo /path/to/survlatent_ode \
  --inputs-dir <prediction_inputs_dir> --output-dir <out> \
  --landmark-day 0 --config platinum
```

### Full follow-up and the `end_of_obs_idx` sweep

`--full-followup` reads `longitudinal_full_landmark{D}.csv` instead of the
landmark frame. The upstream model is already continuous-time and natively
consumes irregular observation times, so no model surgery is needed — only the
wider input and a per-cut evaluation.

`--observation-cut-days N` keeps only observations at or before
`landmark + N` and implies `--full-followup`. **That truncation is the sweep
mechanism**: `end_of_obs_idx` is not a settable parameter upstream, it is
derived as `tt[-1]`, the last retained observation time
(`survlatent_ode_repo/lib/utils.py`). Truncating the frame before the batch
builder sees it also guarantees no post-cut value reaches the encoder.

Each cut is a separate fit with its own checkpoints, so sweep by running once
per value rather than passing a grid. Patients left with no observation at or
before the cut are dropped — `variable_time_collate` fails on an empty `tt` —
and the count is reported, because a cut that quietly discards much of the
cohort otherwise looks like a clean run on a smaller, healthier sample.

Related upstream constraint: `lib/utils.py` asserts
`mask_surv.sum() > 0`, i.e. each patient's time-to-event must exceed their last
observation time. A cut that lands after some patient's event will trip it,
naming the offending ID.

From the pipeline, set `SURVLATENT_FULL_FOLLOWUP` or
`SURVLATENT_OBSERVATION_CUT_DAYS` (a single value, or `None`).

```bash
python survlatent_ode.py \
  --survlatent-repo /path/to/survlatent_ode \
  --inputs-dir <prediction_inputs_dir> --output-dir <out> \
  --landmark-day 0 --config platinum --observation-cut-days 180
```

### Risk-SD collapse warning

`write_prediction_diagnostics()` prints a `WARNING: risk variation is
essentially zero` if every test patient's predicted risk at each quantile
horizon has standard deviation `< 1e-5` -- a sign the model degenerated to
predicting the same curve for everyone (AUC(t) will read ~0.5). Treat this as
a stop condition before running a full hyperparameter grid.

The same applies to the observation-cut sweep: check this warning at one cut
point before spending fits on the rest of the grid. A collapsed run is not a
point worth comparing, and a sweep of collapsed runs produces a flat, tidy,
entirely meaningless curve.
