# PROFILE LLM Extraction v3

This directory contains the first unified `v3` iteration of the PROFILE extraction workflow.

`v3` broadens the target beyond the `v2` NEPC/SCPC event workflow and adds:

- unified phenotype labeling for all prostate patients
- explicit `NEPC/SCPC` status
- explicit `aggressive variant prostate cancer (AVPC)` status using Aparicio-style criteria
- text-only biomarker and lab-pattern capture when those details are explicitly documented in notes
- platinum-indication labeling for platinum-exposed patients

## Pipeline

1. `prepare_patient_context.py`
   - builds one-row-per-patient context with note counts only
2. `prepare_candidate_notes.py`
   - selects focused note snippets for NEPC, AVPC, resistance, biomarker, and platinum rationale
3. `generate_note_extractions.py`
   - runs note-level LLM extraction
4. `generate_patient_labels.py`
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
- `LLM_v3_candidate_text_data.csv`
- `LLM_v3_note_extractions.json`
- `LLM_v3_generated_labels.tsv`
- `LLM_v3_failed_patients.tsv`

## Notes

- `AVPC` is modeled as a phenotype layer, not just a histology enum.
- `C1-C7` are assessed from note text only in the current implementation.
- The current `v3` implementation treats `cisplatin` and `carboplatin` as the platinum exposure definition.
