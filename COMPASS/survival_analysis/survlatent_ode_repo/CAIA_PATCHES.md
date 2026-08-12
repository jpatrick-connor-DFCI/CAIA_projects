# CAIA vendoring notes

Upstream: `https://github.com/itmoon7/survlatent_ode`

Pinned source commit: see `UPSTREAM_COMMIT`.

Local compatibility changes:

- `lib/utils.py` creates the `model_performance/` and `surv_curves/` parent
  directories before creating a run-specific directory. Upstream assumes
  those empty parents survived cloning, which is not guaranteed by Git.

Runtime checkpoints, logs, plots, and survival curves are intentionally
ignored by the parent CAIA repository.
