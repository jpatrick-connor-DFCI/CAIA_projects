# CAIA vendoring notes

Upstream: `https://github.com/itmoon7/survlatent_ode`

Pinned source commit: see `UPSTREAM_COMMIT`.

Local compatibility changes:

- `lib/utils.py` creates the `model_performance/` and `surv_curves/` parent
  directories before creating a run-specific directory. Upstream assumes
  those empty parents survived cloning, which is not guaranteed by Git.

Runtime checkpoints, logs, plots, and survival curves are intentionally
ignored by the parent CAIA repository.

This checkout is intentionally runtime-only. Upstream demonstration data,
example notebooks, the architecture image, README, generated artifacts, and
the unused standalone `lib/ode_rnn.py` model were removed. The retained Python
modules are the complete transitive import set used by `lib.neural_ode_surv`
and `lib.utils` in the CAIA adapter. The upstream license and conda environment
specification are retained.
