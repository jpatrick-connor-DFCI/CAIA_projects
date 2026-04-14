# PROFILE LLM Extraction v3

This directory contains the first unified `v3` iteration of the PROFILE extraction workflow.

`v3` broadens the target beyond the `v2` NEPC/SCPC event workflow and adds:

- unified phenotype labeling for all prostate patients
- explicit `NEPC/SCPC` status
- explicit `aggressive variant prostate cancer (AVPC)` status using Aparicio-style criteria
- structured biomarker and lab-derived context
- platinum-indication labeling for platinum-exposed patients

## Pipeline

1. `prepare_patient_context.py`
   - builds one-row-per-patient context with note counts, PSA summaries, and first med dates
2. `derive_structured_features.py`
   - derives lab and somatic features used in AVPC and biomarker synthesis
3. `prepare_candidate_notes.py`
   - selects focused note snippets for NEPC, AVPC, resistance, biomarker, and platinum rationale
4. `generate_note_extractions.py`
   - runs note-level LLM extraction
5. `generate_patient_labels.py`
   - synthesizes patient-level phenotype and platinum indication labels

One-command runner:

```bash
python COMPASS/PROFILE/v3/run_v3_pipeline.py --max-workers 4
```

Stage-specific runs:

```bash
python COMPASS/PROFILE/v3/run_v3_pipeline.py --prepare-only
python COMPASS/PROFILE/v3/run_v3_pipeline.py --extract-only --max-workers 4
python COMPASS/PROFILE/v3/run_v3_pipeline.py --label-only
```

## Outputs

By default, `v3` writes to `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/v3_outputs/`.

- `LLM_v3_patient_context.csv`
- `LLM_v3_derived_features.tsv`
- `LLM_v3_candidate_text_data.csv`
- `LLM_v3_note_extractions.json`
- `LLM_v3_generated_labels.tsv`
- `LLM_v3_failed_patients.tsv`

## Notes

- `AVPC` is modeled as a phenotype layer, not just a histology enum.
- `C1` is primarily note/pathology-derived.
- `C5-C7` use a hybrid approach with structured context plus note evidence.
- The current `v3` implementation treats `cisplatin` and `carboplatin` as the platinum exposure definition.
