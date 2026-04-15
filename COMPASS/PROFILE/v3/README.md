# PROFILE LLM Extraction v3

This `v3` redesign uses three separate, task-specific text-only pipelines that are merged deterministically at the end.

The guiding idea is to preserve the focused behavior that made `v2` work well for NEPC extraction:

- `NEPC arm`: neuroendocrine / small-cell prostate cancer and transformation
- `AVPC arm`: aggressive-variant / anaplastic language plus Aparicio-style features
- `Biomarker arm`: platinum-relevant biomarker or molecular rationale
- `Merge step`: deterministic precedence `NEPC -> AVPC -> biomarker -> conventional`

## Architecture

1. `compile_prostate_note_bundle.py`
   - loads all raw OncDRS notes for the supplied prostate MRN list
   - writes `LLM_v3_prostate_note_bundle.json.gz`
2. `run_v3_pipeline.py`
   - prepares the shared note inventory
   - runs one or more arm pipelines in-process
   - merges the arm outputs into the final `v3` bucket

`prepare_note_inventory.py`, `prepare_arm_candidates.py`, `generate_arm_labels.py`,
and `merge_labels.py` still exist as stage-level entrypoints for debugging or backwards
compatibility, but `run_v3_pipeline.py` is the primary entrypoint.

The task-specific arm definitions live under `COMPASS/PROFILE/v3/arms/`.

## Recommended Two-Step Run

```bash
python COMPASS/PROFILE/v3/compile_prostate_note_bundle.py --mrn-file path/to/prostate_mrns.txt
python COMPASS/PROFILE/v3/run_v3_pipeline.py --text-source bundle --max-workers 4
```

If the bundle lives somewhere else, pass `--note-bundle-path path/to/LLM_v3_prostate_note_bundle.json.gz`.

`v3` still defaults to raw OncDRS note loading for backwards compatibility. Use `--text-source compiled` only if you intentionally want the precompiled `prostate_text_data.csv` path.

## One-Command Raw Run

```bash
python COMPASS/PROFILE/v3/run_v3_pipeline.py --mrn-file path/to/mrns.txt --max-workers 4
```

## Stage Modes

```bash
python COMPASS/PROFILE/v3/run_v3_pipeline.py --prepare-only --mrn-file path/to/mrns.txt
python COMPASS/PROFILE/v3/run_v3_pipeline.py --arms-only --mrn-file path/to/mrns.txt --max-workers 4
python COMPASS/PROFILE/v3/run_v3_pipeline.py --merge-only
python COMPASS/PROFILE/v3/run_v3_pipeline.py --arm nepc --arms-only --mrn-file path/to/mrns.txt
```

## Outputs

By default, `v3` writes to `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/v3_outputs/`.

Shared files:

- `LLM_v3_prostate_note_bundle.json.gz`
- `LLM_v3_note_inventory.csv`
- `LLM_v3_patient_context.csv`
- `LLM_v3_merged_labels.tsv`

Per-arm files:

- `LLM_v3_nepc_candidate_text_data.csv`
- `LLM_v3_nepc_note_extractions.json`
- `LLM_v3_nepc_labels.tsv`
- `LLM_v3_nepc_failed_patients.tsv`
- `LLM_v3_avpc_candidate_text_data.csv`
- `LLM_v3_avpc_note_extractions.json`
- `LLM_v3_avpc_labels.tsv`
- `LLM_v3_avpc_failed_patients.tsv`
- `LLM_v3_biomarker_candidate_text_data.csv`
- `LLM_v3_biomarker_note_extractions.json`
- `LLM_v3_biomarker_labels.tsv`
- `LLM_v3_biomarker_failed_patients.tsv`

## Notes

- All three arms follow the same retrieval pattern as the NEPC approach: any trigger hit makes the note eligible, and the note text is replaced by large merged snippet windows around the trigger hits.
- The shared snippet window is intentionally much larger in the current implementation so the LLM sees more complete surrounding context for trial, workup, and suspicion language.
- No structured labs, genomics tables, medication tables, or PSA tables are used in this `v3` design.
- `cisplatin` and `carboplatin` are still the platinum terms recognized in note text.
