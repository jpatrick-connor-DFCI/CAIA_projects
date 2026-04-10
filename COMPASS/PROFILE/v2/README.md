# PROFILE LLM Extraction v2

This directory contains a parallel v2 arm of the LLM extraction workflow. The original platinum-focused scripts under `COMPASS/PROFILE/` are unchanged.

## What v2 changes

- Broadens the target cohort to all prostate cancer patients present in the local prostate bundle
- Builds a patient context table from notes, meds, and PSA data
- Can load notes either from the compiled `prostate_text_data.csv` bundle or directly from raw OncDRS JSON files
- Selects candidate notes with trigger-based retrieval instead of only using a `+/-90` day platinum window
- Extracts patient-level events instead of only asking why platinum was used
- Treats `clinical_trial` as context rather than a primary platinum-reason label

## Files

- `prepare_event_candidates.py`: builds v2 candidate notes and patient context
- `generate_event_labels.py`: runs the two-stage LLM extraction and synthesis workflow
- `run_v2_pipeline.py`: convenience wrapper that runs both steps in sequence
- `prompts.py`: v2 extraction and synthesis prompts
- `config.py`: default paths, medication groups, and trigger rules

## Default outputs

By default, v2 writes to `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/v2_outputs/`.

- `LLM_v2_candidate_text_data.csv`
- `LLM_v2_patient_context.csv`
- `LLM_v2_generated_labels.tsv`
- `LLM_v2_note_extractions.json`
- `LLM_v2_failed_patients.tsv`

You can override the base data or output locations with:

- `CAIA_COMPASS_DATA_PATH`
- `CAIA_COMPASS_V2_OUTPUT_DIR`
- `CAIA_ONCDRS_RAW_TEXT_PATH`
- `CAIA_ONCDRS_RAW_TEXT_PATHS`

## Run

```bash
python COMPASS/PROFILE/v2/prepare_event_candidates.py
python COMPASS/PROFILE/v2/generate_event_labels.py --max-workers 4
```

One-command version:

```bash
python COMPASS/PROFILE/v2/run_v2_pipeline.py --max-workers 4
```

Raw OncDRS mode with a predefined MRN list:

```bash
python COMPASS/PROFILE/v2/run_v2_pipeline.py \
  --text-source raw \
  --mrn-file path/to/mrns.txt \
  --output-dir /data/gusev/USERS/jpconnor/data/CAIA/COMPASS/LLM_v2 \
  --max-workers 4
```

By default, raw mode searches the union of:

- `/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2024_03`
- `/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2025_03`
- `/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2025_11`

`--mrn-file` can be a plain-text list, CSV, or TSV. In raw mode, an MRN list is required.
If you want to override the defaults, repeat `--raw-text-path`:

```bash
python COMPASS/PROFILE/v2/run_v2_pipeline.py \
  --text-source raw \
  --raw-text-path /path/to/dir1 \
  --raw-text-path /path/to/dir2 \
  --mrn-file path/to/mrns.txt
```

Useful debug options:

```bash
python COMPASS/PROFILE/v2/generate_event_labels.py --limit-mrns 25
python COMPASS/PROFILE/v2/generate_event_labels.py --retry-failures
python COMPASS/PROFILE/v2/generate_event_labels.py --overwrite-existing
python COMPASS/PROFILE/v2/run_v2_pipeline.py --mrns 12345,67890 --max-workers 4
python COMPASS/PROFILE/v2/run_v2_pipeline.py --mrn-file path/to/mrns.txt --max-workers 4
python COMPASS/PROFILE/v2/prepare_event_candidates.py --text-source raw --mrn-file path/to/mrns.txt
```

By default, `generate_event_labels.py` resumes from any existing `LLM_v2_generated_labels.tsv` and `LLM_v2_note_extractions.json` in the output directory. Use `--overwrite-existing` to ignore and replace those files.

## Comparison framing

v1 remains the platinum-rationale pipeline:

- note selection anchored to unlabeled platinum-treated patients
- notes limited to `+/-90` days around platinum start
- patient-level output focused on platinum indication

v2 is the broader event pipeline:

- patient context built for the broader prostate cohort
- note selection based on histology, metastasis, platinum, ADT-resistance, biomarker, and trial-context triggers
- patient-level output focused on metastatic date, platinum date, transformation date, histology, and ADT nonresponse evidence
