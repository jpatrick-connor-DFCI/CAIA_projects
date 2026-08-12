# multivariate_longitudinal

Two longitudinal deep survival models fit on the person-period lab history
(`longitudinal_landmark{D}.csv`, built by
`COMPASS/data_preprocessing/build_prediction_inputs.py --build-longitudinal`),
each in two configs:

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

```
python dynamic_deephit.py \
  --inputs-dir <prediction_inputs_dir> --output-dir <out> \
  --landmark-day 0 --config platinum
```

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
never collide by default.

```
python survlatent_ode.py \
  --survlatent-repo /path/to/survlatent_ode \
  --inputs-dir <prediction_inputs_dir> --output-dir <out> \
  --landmark-day 0 --config platinum
```

### Risk-SD collapse warning

`write_prediction_diagnostics()` prints a `WARNING: risk variation is
essentially zero` if every test patient's predicted risk at each quantile
horizon has standard deviation `< 1e-5` -- a sign the model degenerated to
predicting the same curve for everyone (AUC(t) will read ~0.5). Treat this as
a stop condition before running a full hyperparameter grid.
