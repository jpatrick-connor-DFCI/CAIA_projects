# `avpc_nepc` — a third survival endpoint from the longitudinal criteria timeline

**Status:** design, not yet implemented. This document is the implementation spec.

## Context

The COMPASS survival pipeline currently models two endpoints: `platinum` (first
carboplatin/cisplatin exposure) and `nepc` (strict, veto-gated LLM NEPC diagnosis from
`nepc_dx_labels.parquet`). The `nepc` endpoint is precision-biased by design and, after the
`t_nepc > 0` incident gate, is likely event-poor — the README flags this as a high-impact known
issue and names `avpc_nepc_timeline.parquet` as the fallback broader label source, "a different
label [that] would need its own endpoint."

This is that endpoint. The timeline artifact already exists and is currently unused by this repo:
`tasks/longitudinal_NEPC/build_nepc_timeline.py` in `LLM_clinical_annotations` emits one dated row
per (patient, criterion), deduped to earliest onset. What is missing is (a) a patient-level label
derived from it carrying explicit AVPC and NEPC event times, and (b) the wiring to make it a
first-class endpoint alongside `platinum` and `nepc`.

**Terminology.** In this repo an *arm* is the treatment anchor (`adt`, `arpi`; `_ARM_SPECS` in
`compass_pipeline.py`) and an *endpoint* is the modeled event. What is added here is a third
**endpoint**. It still gets a fully parallel input and output tree
(`prediction_inputs_adt_avpc_nepc/`, `local_runs_adt_avpc_nepc/`) via the existing `output_suffix`
mechanism.

## Label definitions

Source: `avpc_nepc_timeline.parquet`, columns `DFCI_MRN`, `event_date`, `date_source`,
`date_precision`, `criterion_added`, `cumulative_criteria`, `num_criteria_to_date`,
`supporting_quote`, `confidence`, `source_note_date`.

Two criterion families live in `criterion_added`: Aparicio `C1`–`C7`, and four NEPC features
(`NEPC:small_cell_dx`, `NEPC:histologic_transformation`, `NEPC:ne_features`,
`NEPC:positive_ne_ihc`). A synthetic `conventional` row (no date) marks screened-negative patients.

| Label | Positive when | Event date |
| --- | --- | --- |
| `AVPC` | ≥ 3 distinct `C1`–`C7` criteria **and** zero `NEPC:*` criteria ever | earliest `event_date` at which the C-only cumulative count reaches 3 |
| `NEPC_TIMELINE` | any `NEPC:*` criterion | `event_date` of the first `NEPC:*` criterion |
| `AVPC_NEPC` (modeled) | `AVPC` or `NEPC_TIMELINE` | **NEPC precedence:** `NEPC_TIMELINE_DATE` if `NEPC_TIMELINE == 1`, else `AVPC_DATE` |

`AVPC` and `NEPC_TIMELINE` are generated and carried separately for audit and stratification; the
**union** `AVPC_NEPC` is the only thing modeled.

Three load-bearing decisions:

1. **NEPC precedence applies to timing, not just category.** A patient who crosses 3 Aparicio
   criteria at day 100 and gains a first NEPC feature at day 300 has `t_avpc_nepc = 300`, not 100.
   This mirrors the classifier's "PRECEDENCE IS STRICT" rule, extended to the time axis.
2. **The 3-criteria count is C1–C7 only.** Do **not** read the timeline's `num_criteria_to_date` /
   `cumulative_criteria` — those mix in the four `NEPC:*` keys. Recompute a C-only running count.
3. **Unscreened patients are censored negatives**, matching the strict endpoint's precedent, with
   `label_source` carried so a sensitivity analysis can restrict on it.

---

## Part 1 — the label builder (in `LLM_clinical_annotations`)

New file: `tasks/longitudinal_NEPC/build_avpc_nepc_labels.py`

Reads `$LLM_ANNOTATIONS_DATA_PATH/LLM_avpc_nepc_timeline/avpc_nepc_timeline.parquet`, writes
`.../LLM_avpc_nepc_timeline/avpc_nepc_labels.parquet` — one row per patient. Model the module
shape, CLI, and atomic-write style on the sibling `tasks/nepc_diagnosis/build_nepc_dx_labels.py`,
and reuse `CRITERION_LABELS` from `build_nepc_timeline.py` rather than re-listing the criterion keys.

Output schema, deliberately parallel to `nepc_dx_labels.parquet` so the survival-side loader is a
near-copy of `load_nepc_dx_labels`:

```
DFCI_MRN
has_avpc, avpc_date
has_nepc_timeline, nepc_timeline_date
has_avpc_nepc, avpc_nepc_date          # the modeled pair
date_source, date_precision            # from the row that DEFINED avpc_nepc_date
n_avpc_criteria                        # max C-only cumulative count reached
avpc_criteria, nepc_criteria           # sorted key lists, for audit
supporting_quote, confidence, source_note_date
label_source
```

Construction notes:

- Derive the criterion sets with explicit `AVPC_KEYS = {"C1".."C7"}` / `NEPC_KEYS = {"NEPC:*"}` —
  a `startswith("C")` test would also match `conventional`.
- The timeline is already deduped to earliest onset per (patient, criterion) by `build_timeline`'s
  `_prefer_onset`, so no re-deduplication is needed. Sort by `(event_date, criterion_added)` and
  walk, counting distinct C keys; `avpc_date` is the `event_date` of the block at which that count
  first reaches 3. Same-date blocks can push the count from 1 to 3 at once — that date is correct.
- **Undated rows** (`event_date` null) cannot be timed. Exclude them from the running count and
  from date selection, and demote any patient who would be positive only on undated evidence to
  `has_avpc_nepc = 0`, printing the count. Mirrors the existing demotion at
  `COMPASS/data_preprocessing/compile_COMPASS_cohort_data.py:755`.
- `label_source` values: `timeline_positive`; `timeline_negative` (dated criteria present but below
  threshold); `conventional` (the synthetic screened-negative row); `auto_negative_no_evidence`
  (absent from the timeline entirely — filled in downstream by the join, not here).
- Print a summary: n patients, n AVPC-positive, n NEPC-positive, n union, the `n_avpc_criteria`
  distribution, and the `date_source` / `date_precision` breakdown among events. `date_source =
  "note_date"` means earliest *documentation*, not onset — the same caveat the README carries for
  the strict endpoint, and it will be more common here.

---

## Part 2 — wiring the endpoint into `PROFILE-testing`

The endpoint machinery is registry-driven: `univariate_analysis.py` and `multivariate_analysis.py`
bind `--endpoints` to `choices=list(ENDPOINTS)`, and `build_prediction_inputs.py` to
`choices=sorted(ENDPOINTS)`, so those CLIs need no edit. The work is eleven registry/derivation
points, in dependency order.

### 2.1 Cohort compile — `COMPASS/data_preprocessing/compile_COMPASS_cohort_data.py`

- Add `AVPC_NEPC_LABELS_PATH` beside `NEPC_DX_LABELS_PATH` (line 114), pointing at
  `LLM_annotations/LLM_avpc_nepc_timeline/avpc_nepc_labels.parquet`.
- Add `load_avpc_nepc_labels(path)` modeled directly on `load_nepc_dx_labels` (line 673): same
  missing-file-is-non-fatal empty-frame contract, same `parse_mixed_datetime_expr` for the ISO
  string dates, same undated-positive demotion. Renames: `has_avpc_nepc → AVPC_NEPC`,
  `avpc_nepc_date → AVPC_NEPC_DATE`, `date_source → AVPC_NEPC_DATE_SOURCE`, `date_precision →
  AVPC_NEPC_DATE_PRECISION`, `label_source → AVPC_NEPC_LABEL_SOURCE`, plus the carried components
  `has_avpc → AVPC`, `avpc_date → AVPC_DATE`, `has_nepc_timeline → NEPC_TIMELINE`,
  `nepc_timeline_date → NEPC_TIMELINE_DATE`, `n_avpc_criteria → AVPC_N_CRITERIA`.
- `--avpc-nepc-labels` CLI arg alongside `--nepc-labels` (line ~1063).
- Left-join, `fill_null(0)` on `AVPC_NEPC` (mirroring line 902), and set
  `AVPC_NEPC_LABEL_SOURCE = "auto_negative_no_evidence"` where the join missed.
- `TT_AVPC_NEPC` derivation copying the NEPC block at lines 929–942 (event date when the event
  occurred, else `FOLLOW_UP_END_DATE`; null when no anchor).
- Add the columns to the cohort select list (line ~961) and a summary block mirroring lines
  992–1030, including the prevalent (`TT_AVPC_NEPC <= 0`) note.

### 2.2 Stage 2 — `COMPASS/data_preprocessing/longitudinal_data_processing.py`

Mirror lines 632–644 and the `nepc_cols` passthrough at line 676: derive
`t_avpc_nepc = (AVPC_NEPC_DATE - anchor).days` for positives, else `t_last_contact`; append the
AVPC / NEPC-timeline columns only when present.

### 2.3 `make_outcome_df` — `survival_common/cohort.py`

This is the one place where copy-paste would be a mistake. The NEPC endpoint is threaded through
five separate `if has_nepc:` blocks (derivation ~line 244, `rebased_duration_cols` ~line 266, admin
censoring ~line 292, endpoint validation ~line 313, validity conditions ~line 340). A third
endpoint added the same way triples the drift surface.

Refactor to a registry at module scope:

```python
# Endpoints whose columns the upstream cohort may or may not carry. Each maps
# the endpoint key to (event_col, duration_col, event_date_col).
OPTIONAL_ENDPOINT_SPECS = {
    "nepc":      ("NEPC",      "t_nepc",      "NEPC_DATE"),
    "avpc_nepc": ("AVPC_NEPC", "t_avpc_nepc", "AVPC_NEPC_DATE"),
}
```

and drive all five blocks from a loop over the specs present in `pat.columns`. `platinum` keeps its
bespoke path — it is always required and uniquely feeds the `t_either` / `EITHER` derivation.
Replace the hardcoded `if endpoint not in {"platinum", "nepc"}` check with membership in
`{"platinum", *OPTIONAL_ENDPOINT_SPECS}`, and keep `require_nepc` as the existing deprecated alias
for `endpoint="nepc"`. The validity gate for `avpc_nepc` is the incident pair `t_avpc_nepc notna` +
`t_avpc_nepc > 0`, exactly as for `nepc`.

### 2.4 Endpoint registry — `COMPASS/survival_analysis/cox_aggregated.py`

Add to `ENDPOINTS` (line 101):

```python
"avpc_nepc": {
    "duration_col": "t_avpc_nepc",
    "event_col": "AVPC_NEPC",
    "description": "Time from the treatment anchor (first ADT exposure = time 0) to the "
                   "first documented AVPC (>=3 Aparicio criteria) or NEPC criterion, "
                   "from the longitudinal criteria timeline",
},
```

Then add every one of these to `OUTCOME_METADATA_COLUMNS` (line 119): `AVPC_NEPC`, `t_avpc_nepc`,
`t_avpc_nepc_from_first_record`, `AVPC_NEPC_DATE`, `AVPC_NEPC_DATE_SOURCE`,
`AVPC_NEPC_DATE_PRECISION`, `AVPC_NEPC_LABEL_SOURCE`, `AVPC`, `AVPC_DATE`, `AVPC_N_CRITERIA`,
`NEPC_TIMELINE`, `NEPC_TIMELINE_DATE`.

> **Highest-risk step in the change.** `OUTCOME_METADATA_COLUMNS` is what keeps outcome columns out
> of the feature matrix *for every endpoint*, and this endpoint contributes three near-perfect
> leakage columns (`AVPC`, `NEPC_TIMELINE`, `AVPC_N_CRITERIA`) with no analogue among the existing
> endpoints. Missing one produces a model that predicts its own label at ~1.0 AUC on the platinum
> and nepc runs too. The existing comment at line 119 already says exactly this about the NEPC
> columns — extend it.

### 2.5 Prediction inputs — `COMPASS/data_preprocessing/build_prediction_inputs.py`

- `LONGITUDINAL_OPTIONAL_EVENT_COLS` (line 85): add `"AVPC_NEPC": "t_avpc_nepc"`.
- `AGGREGATED_DROP_COLUMNS` (line 361): add `AVPC_NEPC_DATE`, `AVPC_DATE`, `NEPC_TIMELINE_DATE`,
  `t_avpc_nepc_from_first_record`.
- `--endpoint` choices and the manifest's `auc_horizons_by_landmark` loop (line 703) are already
  driven by `ENDPOINTS`; the "skip endpoints this cohort doesn't carry" guard (lines 704–715)
  handles a cohort built without the labels mounted with no change.
- The `endpoint == "nepc"` / else message branch at lines 538–549 needs a third case, or better a
  generic message driven by `ENDPOINTS[endpoint]["description"]`.

### 2.6 Longitudinal configs — `survival_common/longitudinal_targets.py`

Add `avpc_nepc` (`["AVPC_NEPC"]` / `["t_avpc_nepc"]`) and `avpc_nepc_competing`
(`["AVPC_NEPC", "DEATH"]` / `["t_avpc_nepc", "t_death"]`) to `LONGITUDINAL_CONFIGS`, matching
entries in `CONFIG_ENDPOINTS`, and a third cause-ordering assert alongside the existing two. Update
the module docstring, which currently says "Four configs."

### 2.7 SurvLatent mirror — `COMPASS/survival_analysis/multivariate_longitudinal/survlatent_ode.py`

Add the two matching `EVENT_CONFIGS` entries (line 53), honoring the string-vs-list dispatch
convention documented at lines 50–52. The import-time asserts at lines 71–91 catch any mismatch.

### 2.8 Pipeline driver — `COMPASS/survival_analysis/compass_pipeline.py`

- `make_runs` (line 137): replace the hardcoded `{"platinum", "nepc"}` validation with
  `set(cox_aggregated.ENDPOINTS)`, removing a drift point permanently.
- `_LONGITUDINAL_CONFIGS_BY_ENDPOINT` (line 572): add
  `"avpc_nepc": ("avpc_nepc", "avpc_nepc_competing")`.
- `make_endpoint_runs`' suffix rule (`"" if platinum else f"_{endpoint}"`) already yields
  `_avpc_nepc` — no change.

### 2.9 R figures — `COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R`

`SUPPORTED_ENDPOINTS` (line 374) → add `"avpc_nepc"`; `ENDPOINT_SUFFIXES` (line 375) → add
`avpc_nepc = "_avpc_nepc"`. Everything downstream (`FIG_ROOT` nesting,
`read_endpoint_performance`'s filter, the `EMIT_ENDPOINT_INDEPENDENT` guard) is already generic
over the suffix.

### 2.10 Notebooks

Parameter cell of `01_preprocessing`, `02_univariate`, `03_multivariate`,
`03b_multivariate_longitudinal`: `ENDPOINTS = ("platinum", "nepc", "avpc_nepc")`. `01` also needs
`--avpc-nepc-labels` threaded into the Stage 1 compile call. `07_endpoint_comparison`: extend its
`ENDPOINTS` tuple (line 18) so the count table and the <50-event underpowered warning cover the new
endpoint.

### 2.11 README

Extend the "Arms and endpoints" tables with the new endpoint and the two new configs, and update
two high-impact known issues: the event-poor-NEPC item now has its answer, and the "two different
NEPC definitions are in play" warning becomes **three** — strict `nepc_dx_labels` (the `nepc`
endpoint), the broad binary classifier (Figure 2 strata), and the criteria timeline (this
endpoint). Numbers from the three will not agree; the "always name which one" rule matters more,
not less.

---

## Tests

New `tests/test_avpc_nepc_labels.py` — label construction, against a small in-memory timeline frame
rather than the real parquet:

- 3rd C criterion sets `avpc_date`; 2 criteria → negative.
- A same-date block pushing the count 1 → 3 dates the event at that block.
- NEPC precedence: C-threshold at day 100 + NEPC feature at day 300 → `has_avpc == 0`,
  `has_nepc_timeline == 1`, `avpc_nepc_date == 300`.
- `NEPC:*` keys never contribute to the 3-criteria count (2 C + 2 NEPC → NEPC-positive by the NEPC
  rule, and `n_avpc_criteria == 2`).
- An undated-only positive is demoted to `has_avpc_nepc = 0`.
- `conventional` row → negative, `label_source == "conventional"`.

New `tests/test_avpc_nepc_endpoint.py` — modeled on `tests/test_nepc_endpoint.py`, whose central
invariant (the incident gate must never fire on another endpoint's run) is exactly the one at risk:

- `t_avpc_nepc > 0` drops prevalent cases **only** under `endpoint="avpc_nepc"`; the platinum and
  nepc cohorts are identical with and without the AVPC_NEPC columns present.
- **Leakage guard:** `AVPC`, `NEPC_TIMELINE`, `AVPC_N_CRITERIA` and all date/provenance columns are
  absent from the feature matrix for all three endpoints. Assert against
  `cox_aggregated.outcome_columns()`, not a hand-written list.
- A cohort with no AVPC_NEPC columns raises the clear "rebuild with the labels" error rather than a
  KeyError, and `longitudinal_event_columns` omits them from the manifest.

Extend `tests/test_nepc_longitudinal.py` for the two new configs' cause ordering, and
`tests/test_figure_pipeline_endpoints.py` for the `_avpc_nepc` suffix agreement between R and
`make_runs`.

## Verification

```bash
# 1. Unit tests — full suite, since cohort.py and cox_aggregated.py are shared by every endpoint.
python -m pytest tests/ COMPASS/data_preprocessing/tests/ -q

# 2. Labels: build and eyeball the summary (event counts, date_source mix, criteria distribution).
python tasks/longitudinal_NEPC/build_avpc_nepc_labels.py     # in LLM_clinical_annotations

# 3. Stage 1 — confirm AVPC_NEPC counts and the prevalent-case note appear in the compile summary.
#    Then Stage 3 for the new endpoint; check landmark_attrition.json and that build_manifest.json
#    carries endpoint="avpc_nepc" plus its auc_horizons_by_landmark entry.

# 4. Notebooks 01 -> 02 -> 03 with ENDPOINTS including "avpc_nepc".
#    prediction_inputs_adt/ and local_runs_adt/ must be untouched (compare mtimes) — the suffixed
#    tree is the only thing written.

# 5. THE LEAKAGE CHECK. In 03's output, confirm no AVPC*/NEPC* column appears in
#    landmark_xgboost_feature_selection.csv for ANY endpoint. A test-set C-index near 1.0 on any
#    arm means a column was missed in OUTCOME_METADATA_COLUMNS — stop and fix before reading
#    anything else.

# 6. 07_endpoint_comparison.ipynb across all three endpoints: event counts side by side, and the
#    underpowered warning. Expect avpc_nepc >> nepc in events — if not, the timeline join failed.

# 7. 05_figures.Rmd with endpoint="avpc_nepc"; panels land under FIG_ROOT/ADT/avpc_nepc/.
```

## Sequencing

Part 1 (labels) is independent of Part 2 (wiring) and can land first — the survival repo treats a
missing label file as non-fatal throughout, so the wiring can be merged and tested against an
absent parquet before real labels exist. Within Part 2, §2.3's `make_outcome_df` refactor should
land as its own commit with the existing suite green *before* the new endpoint is added on top, so
a regression in the shared path stays unambiguously separable from the new endpoint.

## Known consequences

- **The `avpc_nepc` cohort is a third distinct patient set.** The `t_avpc_nepc > 0` incident gate
  drops prevalent cases, as `--require-nepc` does for `nepc`. Cross-endpoint metric comparisons are
  confounded by cohort composition, not just by event definition.
- **Event dates skew toward documentation, not onset.** `date_source = "note_date"` is more common
  in the timeline than in the veto-gated diagnosis labels, because per-criterion mentions rarely
  carry a stated onset date. `AVPC_NEPC_DATE_SOURCE` and `AVPC_NEPC_DATE_PRECISION` ride through to
  the prediction inputs so a sensitivity analysis can restrict to `stated` / `day` precision.
- **This is a lower-precision label than the `nepc` endpoint by construction.** It trades precision
  for events. Name which definition produced any number that leaves this repo.
