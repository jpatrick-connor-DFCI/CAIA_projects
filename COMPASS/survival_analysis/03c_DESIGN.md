# 03c: clinical text as a predictor of time-to-platinum

**Status: built.** This was written as a design before implementation and is kept as the
rationale record; the *What was built* section at the end states where the implementation
differs from the plan above it.

Design for `03c_multivariate_text.ipynb`, which sits alongside `03_multivariate.ipynb`
(Cox/XGBoost on labs) and `03b_multivariate_longitudinal.ipynb` (Dynamic-DeepHit), using
pooled clinical-note embeddings from the `clinical_text_embedding_project` as predictors of
ADT -> platinum.

Decisions taken: three feature arms (`text`, `labs`, `labs_text`); full 3x768 pooled
embeddings, unreduced; elastic-net Cox + XGBoost only (no DeepHit arm for now).

## What already exists, and what that buys us

Two findings determine the whole shape of this work.

**1. The embedding project's pooling is already landmark-aware and anchor-parameterized.**
`survival/preprocessing.py:generate_survival_embedding_df` takes `note_timing_col` and
`max_note_window`, filters to notes strictly before the landmark, re-centers note times on
it, and shifts every `tt_*` outcome column by the same amount
(`survival/preprocessing.py:388-392`). It then asserts `max(note_time) <= 0`. That is the
same landmark contract COMPASS's `build_prediction_inputs.py` enforces.

Consequence: we do **not** reimplement pooling or leakage control. We compute an
ADT-relative note-time column and pass it in as `note_timing_col`. `anchors.py` is the
registry to extend — it already carries `treatment` and `sequencing`, and a third `adt`
anchor is the idiomatic addition.

**2. COMPASS's `feature_set` switch is an existing, tested extension point.**
`cox_aggregated.py:prepare_landmark_context` accepts `feature_set` in
`{"labs", "somatic_gleason", "genomic"}`. The `somatic_gleason` branch
(`cox_aggregated.py:484-522`) is an exact template for what we need:

- a CSV manifest in `inputs_dir` declares the feature columns,
- those columns become `raw_feature_cols`,
- `always_include_feature_cols` exempts them from the canonical-lab gate,
- `restrict_to_labs=[]` and `canonical_labs=[]` for the text-only arm.

That last point is load-bearing. `parse_feature_name` (`survival_common/cox_engine.py:70`)
splits on `__`, so a column named `Clinician__emb0042` parses as lab `Clinician`, stat
`emb0042`, and `select_feature_columns`'s `restrict_to_labs` gate would drop all 2304
embedding columns silently. Registering a `text` feature set that bypasses the gate is the
fix; renaming columns is not.

## Shape of the work

### Stage 1 — an `adt` anchor in the embedding project

Add to `anchors.py`:

```python
"adt": {
    "date_col": "adt_start_date",
    "note_time_col": "NOTE_TIME_REL_ADT",
    "age_col": "AGE_AT_ADT_START",
},
```

`anchor_suffix("adt")` returns `__adt`, so embedding files and results directories are
namespaced automatically and nothing existing moves. The ADT start dates come from the
COMPASS cohort build, joined on `DFCI_MRN` — the shared key on both sides.

### Stage 2 — build text features per landmark

A new `COMPASS/data_preprocessing/build_text_embedding_inputs.py`, modeled directly on
`build_somatic_gleason_inputs.py`. For each landmark in {0, 90, 180}:

1. Read the COMPASS landmark frame (MRNs, split labels, ADT-relative outcome clock).
2. Pool notes with `generate_survival_embedding_df(..., note_timing_col="NOTE_TIME_REL_ADT",
   max_note_window=landmark_day, survival_df=None)` — `None` because COMPASS owns the
   outcome clock; we want embeddings only and must not let the embedding project's
   `tt_*` shifting touch COMPASS's columns.
3. Left-join pooled embeddings onto the COMPASS frame on `DFCI_MRN`.
4. Write one self-contained inputs tree per arm under
   `<inputs_dir>/text_embedding/<feature_set>/`, each holding the standard
   `aggregated_landmark{D}.csv` / `pre_treatment_lab_long_landmark{D}.csv` filenames plus a
   `text_embedding_features.csv` manifest (`feature`, `feature_kind="text_embedding"`) and
   its own `build_manifest.json`.

   *Built differently from the plan:* one flat `text_embedding_landmark{D}.csv` would have
   forced `multivariate_analysis.py` to learn a new filename. Reusing the standard names
   inside a per-arm directory means `--inputs-dir` alone selects the arm and nothing
   downstream changes. The AUC horizon grid is also recomputed per arm on the matched
   cohort rather than inherited, since the subset's event-time quantiles differ.

Complete-case policy: the embedding project requires all three pre-anchor note modalities
(Clinician, Imaging, Pathology) and errors out otherwise
(`generate_embedding_prediction_datasets.py:442`). COMPASS patients missing a modality at a
given landmark therefore drop out. **This makes the text cohort a subset, which is why the
`labs` arm must be refit on the matched subset rather than compared against
`03_multivariate.ipynb`'s numbers** — the same available-case logic
`run_multivariate_available_case_sensitivity` already implements. The manifest should record
the per-landmark cohort size and attrition so the notebook can report it.

### Stage 3 — register the `text` feature set

In `cox_aggregated.py:prepare_landmark_context`, extend the allowed set to include `text`
and `labs_text`:

| feature_set | raw_feature_cols | restrict_to_labs | canonical_labs |
|---|---|---|---|
| `text` | manifest columns only | `[]` | `[]` |
| `labs_text` | lab features + manifest columns | canonical labs | selected |
| `labs` | unchanged | canonical labs | selected |

For `labs_text`, embedding columns go in `always_include_feature_cols` so they survive the
lab gate while lab features are still gated normally. Leave `genomic_feature_cols` empty —
the 2.5% prevalence floor is for binary mutation indicators and would be meaningless
against dense continuous embedding dimensions.

Coverage/variability filters still apply to embedding columns and should: a dimension that
is constant or all-NaN on a fold is correctly dropped.

### Stage 4 — pipeline plumbing

In `compass_pipeline.py`:

```python
TEXT_TASK_SPECS = [
    ("elastic-net", "text",      "cox_agg_text_metrics.csv"),
    ("elastic-net", "labs",      "cox_agg_multivariable_metrics.csv"),
    ("elastic-net", "labs_text", "cox_agg_labs_text_metrics.csv"),
    ("xgboost",     "text",      "landmark_xgboost_text_metrics.csv"),
    ("xgboost",     "labs",      "landmark_xgboost_metrics.csv"),
    ("xgboost",     "labs_text", "landmark_xgboost_labs_text_metrics.csv"),
]
```

plus a `run_multivariate_text(run)` following `run_multivariate_available_case_sensitivity`'s
structure, writing to
`local_runs_adt/text_embedding/<arm>/<model_dir>/landmark_{D}/<feature_set>/`, and a
`summarize_text_outputs(run)` that emits one row per (model, landmark, feature_set) with
C-index / mean AUC(t) / integrated Brier, plus paired `delta_cindex` against that cell's
`labs` arm.

Distinct metrics filenames per arm mean the three arms cannot overwrite each other and each
resumes independently under `OVERWRITE = False`, matching 03/03b behavior.

### Stage 5 — the notebook

`03c_multivariate_text.ipynb`, following 03b's structure exactly: markdown header stating
what it runs after -> config cell -> preconditions -> run -> summary.

The preconditions cell matters most here, and mirrors 03b's `RUN_DYNAMIC` check: it counts
the missing `text_embedding/<arm>/aggregated_landmark{D}.csv` frames across all three arms
and prints a warning naming `BUILD_TEXT_INPUTS` and
`cp.build_text_embedding_inputs(run)`, rather than failing deep inside a fit.

## Cost, and what to run first

Elastic-net on 2304 dense columns x 5 folds x 3 landmarks is the expensive part — far more
than the lab arms, which carry tens of columns. XGBoost on 2304 dense features is also
substantially slower than on labs.

Recommended smoke test before committing to the grid: landmark 0 only, elastic-net only,
`text` and `labs` arms. That is 2 fits and answers the question that gates everything else —
whether text carries any signal for this endpoint on the matched cohort.

```python
RUNS[0]["landmarks"] = [0]
```

## Open questions, as resolved

1. **Pooling strategy** — *resolved.* `time_decay_mean` with `decay_param=0.01`, which is
   what every pooling call in the embedding project uses
   (`generate_embedding_prediction_datasets.py`, `generate_mortality_trajectories.py`,
   `ICI_generate_embeddings.py`). Matched rather than defaulted, so this arm's features are
   the manuscript models' features. Both values are module constants in
   `build_text_embedding_inputs.py`.
2. **ADT start date provenance** — *partly open.* The builder joins
   `TREATMENT_ANCHOR_DATE` from the COMPASS cohort build on `DFCI_MRN`, via
   `build_somatic_gleason_inputs.load_treatment_anchors`, so it is the same anchor date the
   rest of the ADT arm uses. How `pre_adt_castrate` interacts with it is untested here
   because the headline config sets `EXCLUSIONS = ("none",)`; check before running that
   exclusion against this arm.
3. **Cohort attrition size** — *deferred to first run, by decision.* Not measured before
   building. The builder prints per-landmark retention and records `n_patients_by_landmark`
   and `split_sizes_by_landmark` in each arm's manifest, and the notebook's last cell reads
   them back as a retention table. Read it before reading any `delta_c_index`: severe
   attrition weakens the matched comparator, and that is a property of the cohort, not of
   the models.

## What was built

| piece | where |
|---|---|
| `adt` anchor | `clinical_text_embedding_project/.../anchors.py` |
| builder | `COMPASS/data_preprocessing/build_text_embedding_inputs.py` |
| `text` / `labs_text` feature sets | `cox_aggregated.py:prepare_landmark_context` |
| CLI choices | `survival_common/projects/compass_profile.py` |
| lab-gate exemptions | `multivariate_analysis.py` (x2), `survival_common/cox_models.py` |
| pipeline entry points | `compass_pipeline.py`: `build_text_embedding_inputs`, `run_multivariate_text`, `summarize_text_outputs` |
| notebook | `03c_multivariate_text.ipynb` |

Three lab gates had to be exempted, not one: the per-fold CV gate and the final
full-train_val gate in `multivariate_analysis.py`, plus a third inside
`tune_multivariable_model` in shared `survival_common/cox_models.py`. That third one is
shared with IPIO, so it took a `restrict_to_canonical_labs` parameter defaulting to the
old behavior, with the flag derived in `cox_runners.py` from whether `ctx.canonical_labs`
is empty rather than from any COMPASS feature-set name.

`genomic_feature_cols` is left explicitly empty for both text arms. In
`tune_multivariable_model` it defaults to `always_include_feature_cols` when None, which
would have applied the 2.5% mutation-prevalence floor to 2304 dense continuous dimensions
and dropped essentially all of them.
