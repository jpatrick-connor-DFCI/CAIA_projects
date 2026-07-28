# Add ADT-anchored COMPASS cohorts alongside the existing ARPI arms

**Status:** planned, not implemented. This document is the implementation spec.

## Context

The COMPASS time-to-platinum analysis currently anchors every patient at **first
ARPI/taxane/radium-223 exposure** (`TREATMENT_ANCHOR_MEDS`, 7 drugs). That anchor
is time 0 for every duration in the pipeline, and cohort entry is simply "has a
non-null anchor date". Two arms are analyzed — `with_other_primaries` and
`without_other_primaries` — differing only in whether competing non-prostate
primary malignancies are excluded.

ARPI initiation is late in the prostate treatment sequence. Anchoring instead at
**first ADT (androgen deprivation) exposure** moves time 0 years earlier, giving a
larger, earlier-stage cohort and a longer observable runway to platinum. The goal
is two new cohorts (`adt_with_other_primaries`, `adt_without_other_primaries`)
carried through the identical pipeline and producing the identical figure set,
run side-by-side with the ARPI arms.

**Hard constraint: the existing ARPI outputs must not change.**
`build_prediction_inputs.py:411` intersects MRNs across all requested landmarks
and derives the train/test split on the first landmark, so any change to the ARPI
landmark list would shrink those cohorts and re-derive their splits. Every change
below is therefore additive and gated.

### Decisions made

| Question | Decision |
| --- | --- |
| ADT drug set | GnRH agonists + antagonists + 1st-gen antiandrogens (10 drugs) |
| Cohort entry | Any ADT start (mirrors current ARPI logic: anchor date not-null) |
| Endpoint | `platinum`, unchanged |
| Landmarks | ARPI arms stay `[0, 90]`; **ADT arms use `[0, 90, 365]`** |
| Stage 2 | Separate ADT output file via a new `--anchor-med-set` flag |
| Layout | Extend the existing cohort loops to 4 keys — one Run All does everything |

---

## Stage 1 — `COMPASS/data_preprocessing/compile_COMPASS_cohort_data.py`

**1. Add the ADT anchor set** next to `TREATMENT_ANCHOR_MEDS` (`:87-97`). All 10
strings verified present in `unique_meds.csv` (the `NCI_PREFERRED_MED_NM`
vocabulary dump); matching is exact string equality after
`.str.to_uppercase().str.strip_chars()`, so the spellings are load-bearing —
note `TRIPTORELIN` and `TRIPTORELIN PAMOATE` are separate vocabulary entries.

```python
ADT_ANCHOR_MEDS = {
    # GnRH agonists
    "LEUPROLIDE ACETATE", "GOSERELIN ACETATE", "TRIPTORELIN",
    "TRIPTORELIN PAMOATE", "HISTRELIN ACETATE",
    # GnRH antagonists
    "DEGARELIX", "RELUGOLIX",
    # First-generation antiandrogens
    "BICALUTAMIDE", "FLUTAMIDE", "NILUTAMIDE",
}
```

Rename the existing set to `ARPI_ANCHOR_MEDS` with
`TREATMENT_ANCHOR_MEDS = ARPI_ANCHOR_MEDS` retained as an alias, and add
`ANCHOR_MED_SETS = {"arpi": ARPI_ANCHOR_MEDS, "adt": ADT_ANCHOR_MEDS}`.

**2. Parameterize the anchor computation.** `compute_treatment_anchor`
(`:430-438`) and `load_medications_for_survival` (`:412-427`) both close over the
module constant. Add a `meds_set` parameter (defaulting to `ARPI_ANCHOR_MEDS` so
existing callers are unaffected); `load_medications_for_survival` must keep the
union `ARPI ∪ ADT ∪ PLATINUM` so one meds scan serves both anchors.

**3. Make the writer loop a product over `{primaries} × {anchor}`.** `main()`
(`:674-695`) currently loops over the 2-key `cohorts` dict and hardcodes the
`_arpi` suffix at `:686-693` — where that suffix means exactly one thing,
`TREATMENT_ANCHOR_DATE.is_not_null()`. Compute `adt_anchor_df` alongside
`anchor_df` (`:636`) — both reuse the same `meds_for_survival`, since anchor
dates don't depend on cohort membership — then loop over both anchors, writing:

- `prostate_adt_survival_cohort_{cohort_key}_adt.csv`
- `mrn_lists/{cohort_key}_adt_mrns.csv`

The bare (unanchored) `prostate_arpi_survival_cohort_{key}.csv` and
`{key}_mrns.csv` outputs stay as-is — they are anchor-independent. Keep the
existing ARPI filenames byte-identical.

**4. Add an `ADT_EXPOSED` flag** to `build_icd_prostate_mrn_flags` (`:388-405`),
mirroring `ARPI_DOCETAXEL_EXPOSED`. The R CONSORT panel needs it for the ADT
arms' Figure 1A.

---

## Stage 2 — `COMPASS/data_preprocessing/longitudinal_data_processing.py`

Stage 2 re-derives the anchor from medications and **overwrites** Stage 1's copy
(`:446-454`), so it is authoritative and must know which anchor is in play.

- Mirror `ADT_ANCHOR_MEDS` / `ANCHOR_MED_SETS` next to `TREATMENT_ANCHOR_MEDS`
  (`:98-112`), matching the existing duplication convention between the two stages.
- Add `--anchor-med-set {arpi,adt}` (default `arpi`) to `parse_args` (`:791-919`);
  thread the resolved set into `compute_treatment_anchor` (`:340-358`).
- Default `--survival-cohort-csv` and `--output-csv` off the chosen anchor:
  `arpi` → current defaults unchanged; `adt` →
  `prostate_adt_survival_cohort_with_other_primaries_adt.csv` and
  `longitudinal_prediction_data_adt.csv`.
- **Critical:** add `"anchor_med_set": args.anchor_med_set` to
  `build_cache_provenance` (`:695-712`). Provenance currently tracks only file
  signatures plus the PSA/PARPi prefilter settings, so without this the
  `consolidated_longitudinal_data.parquet` cache would silently serve rows built
  under the other anchor. Also give the ADT run a distinct cache path so the two
  anchors don't thrash one cache file.

Everything downstream of the anchor (`t_lab`, `t_platinum`, `t_last_contact`,
`FIRST_RECORD_DATE`, `:496-542`) is already expressed relative to
`TREATMENT_ANCHOR_DATE` and needs no change.

Stage 3 (`build_prediction_inputs.py`) needs **no code change** — it already
accepts `--input-csv`, `--restrict-to-mrns`, `--landmark-days`, and
`--output-dir`.

---

## Run notebook — `COMPASS/survival_analysis/COMPASS_run_locally.ipynb`

**Cell 2** — expand the cohort registry from 2 to 4 keys and attach per-cohort
landmarks and input CSV:

```python
COHORT_SPECS = {
    "without_other_primaries":     dict(anchor="arpi", landmarks=[0, 90]),
    "with_other_primaries":        dict(anchor="arpi", landmarks=[0, 90]),
    "adt_without_other_primaries": dict(anchor="adt",  landmarks=[0, 90, 365]),
    "adt_with_other_primaries":    dict(anchor="adt",  landmarks=[0, 90, 365]),
}
```

Each `RUNS` entry gains `landmarks` and `input_csv` (ARPI →
`longitudinal_prediction_data.csv`, ADT → `..._adt.csv`); `anchor_col` stays
`"none"` for all four (the anchor *is* time 0). The ARPI entries must keep their
existing `label`, `inputs_dir`, and `output_dir` values so their outputs land in
the same directories as today.

**Cell 6** — currently one hardcoded Stage 2 call. Make it two: the existing ARPI
invocation unchanged, plus `--anchor-med-set adt` pointing at the widest ADT
cohort. Note this is the expensive step (full raw lab standardization); the
parquet cache makes reruns cheap but the first ADT pass will be slow.

**Cell 8** — `build_prediction_inputs(run)` must pass `--landmark-days` from the
run spec and `--input-csv`; `TASKS` (currently a flat 10-row list with landmarks
0/90 baked in) becomes generated per-run from `run["landmarks"]`, yielding 10
tasks for ARPI arms and 15 for ADT arms. `build_model_command` must likewise take
the landmark from the task row. `--endpoints platinum` stays everywhere.

---

## R figure pipeline — `COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R`

**1. Cohort registry (`:144-148`)** — extend `COHORTS` and `COHORT_LABELS` to the
same 4 keys ("ADT — with other primaries", etc.). The figures notebook (cell 4)
already loops `for (COHORT in COHORTS)` and needs no edit.

**2. Per-cohort landmarks.** `LANDMARKS <- c(0, 90)` (`:198`) is duplicated at
`FIG5_LANDMARKS` (`:1378`) and `FIG6_LANDMARKS` (`:1429`). Replace all three with
a lookup: `LANDMARKS <- COHORT_LANDMARKS[[cohort]]`, and have Fig 5/6 reference
`LANDMARKS` directly. Two places currently assume exactly two landmarks and must
become data-driven over `LANDMARKS`:

- `panels <- list(list(0, "0 days"), list(90, "+90 days"))` (`:1105`) — Figure 3
  volcano panels
- `disc_panels` (`:1219`) and the Fig 4a x-axis factor levels (`:1183`)

This yields a third volcano/importance/KM/androgen panel per ADT arm
automatically, since the downstream loops (`:1290`, `:1413`, `:1487`, `:1522`)
already iterate `LANDMARKS`.

**3. CONSORT panel (`:325-341`)** — the
`if (identical(COHORT, "without_other_primaries"))` branch would silently
mislabel all three other arms via its `else`. Rewrite to branch on two
independent axes derived from the cohort key: primaries
(`grepl("without_other_primaries", COHORT)`) and anchor
(`startsWith(COHORT, "adt_")`). For ADT arms the exposure step becomes
`ADT_EXPOSED == 1` with label `"ADT exposure"` instead of
`ARPI_DOCETAXEL_EXPOSED == 1` / `"ARPI/docetaxel exposure"`. Add `"ADT_EXPOSED"`
to the `required` column check at `:298-302`.

**4. Anchor wording in rendered labels.** These strings say "treatment anchor"
generically and read correctly for both anchors, but should name the anchor for
clarity. Introduce
`ANCHOR_LABEL <- if (is_adt) "ADT initiation" else "ARPI/chemo initiation"` and
interpolate at: Fig 1B xlab (`:386`), Fig 1C x labels (`:502`, `:504`), Table 1
row label (`:447`), Fig 7b title/xlab (`:1652`, `:1635`).

**5. Figure 2 hardcoded assertions (`:634-655`, `:720-723`)** — nine `stopifnot`
calls on absolute patient counts (200/1682/154/30/520/1090/...) that run
unconditionally on *every* cohort call. They are computed on the full
cohort-independent LLM label set, so they will still pass — but they are a live
tripwire: any change to `LLM_v3_labels.tsv` or `platinum_MRN_list.csv` aborts all
four arms. Leave the values alone (out of scope) but confirm they pass on the
first ADT run.

No new figure stems are introduced, so `figure_group()` (`:182-191`) needs no
change — the ADT arms reuse every existing stem, distinguished by the `COHORT`
filename prefix in `save_fig` (`:218`). Output layout stays
`FIG_ROOT/<group>/<stem>/<cohort>_<stem>.png`.

---

## Files to modify

| File | Change |
| --- | --- |
| `COMPASS/data_preprocessing/compile_COMPASS_cohort_data.py` | `ADT_ANCHOR_MEDS`, parameterized anchor fns, anchor×primaries writer loop, `ADT_EXPOSED` flag |
| `COMPASS/data_preprocessing/longitudinal_data_processing.py` | `ADT_ANCHOR_MEDS`, `--anchor-med-set`, anchor in cache provenance, ADT default paths |
| `COMPASS/survival_analysis/COMPASS_run_locally.ipynb` | 4-key `COHORT_SPECS`, second Stage 2 call, per-run landmarks/input CSV in `TASKS` |
| `COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R` | 4-key `COHORTS`, per-cohort landmarks, generalized CONSORT, anchor-aware labels |
| `README.md` | Document the ADT arms, the new artifacts, and `--anchor-med-set` |

`COMPASS_generate_figures.ipynb` needs no edit (it reads `COHORTS` from the
sourced `.R`). `build_prediction_inputs.py`, `cox_aggregated.py`, and
`survival_common/*` need no edit.

---

## Verification

1. **Stage 1** — run `compile_COMPASS_cohort_data.py`. Confirm 4 anchored cohort
   CSVs + 4 MRN lists exist; confirm the two ARPI files are unchanged vs. their
   current versions (`md5sum` before/after); confirm
   `adt_with_other_primaries ⊇ adt_without_other_primaries` (the existing
   `assert_cohort_set_invariants` at `:211-219` covers this) and that each ADT arm
   is larger than its ARPI counterpart — if not, the drug-name match is failing
   and should be debugged against `unique_meds.csv`.
2. **Stage 2** — run both anchor passes. Confirm two distinct output CSVs, and
   that the ADT one has a materially more negative `t_diagnosis` distribution and
   larger `t_platinum` values (time 0 moved earlier). Verify the parquet cache
   manifest records `anchor_med_set` and that flipping the flag forces a rebuild
   rather than a silent cache hit.
3. **Stage 3 / models** — Run All on the run notebook. Check
   `landmark_attrition.json` and the `[debug] merged landmark` lines for each ADT
   arm, especially how many patients survive the 365d landmark; because
   `common_mrns` is intersected across landmarks, severe attrition at 365d
   shrinks the whole ADT arm and is the signal to drop back to `[0, 90]`. Confirm
   `prediction_inputs_without_other_primaries/` and `..._with_other_primaries/`
   are byte-identical to their pre-change state.
4. **Figures** — Run All on `COMPASS_generate_figures.ipynb`. Expect 4 cohort
   prefixes in each panel directory; ADT arms should have 3
   volcano/importance/KM/androgen panels vs. 2 for ARPI. Spot-check
   `figure1a_consort` for both ADT arms (correct "ADT exposure" step and correct
   primaries step) and `figure1b_km` axis wording.
5. **Regression** — `python -m pytest tests/`.
