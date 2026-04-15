# PROFILE LLM Extraction v3

This `v3` redesign uses three separate, task-specific text-only pipelines that are merged deterministically at the end.

The guiding idea is to preserve the focused behavior that made `v2` work well for NEPC extraction:

- `NEPC arm`: neuroendocrine / small-cell prostate cancer and transformation
- `AVPC arm`: aggressive-variant / anaplastic language plus Aparicio-style features
- `Biomarker arm`: platinum-relevant biomarker or molecular rationale
- `Merge step`: deterministic precedence `NEPC -> AVPC -> biomarker -> conventional`

## Architecture

1. `prepare_note_inventory.py`
   - loads the shared raw note inventory for the requested MRNs
   - writes note-count metadata to `LLM_v3_patient_context.csv`
2. `run_arm_pipeline.py --arm nepc`
   - builds trigger-centered snippets for the NEPC arm
   - extracts note-level NEPC evidence
   - synthesizes one NEPC label row per MRN
3. `run_arm_pipeline.py --arm avpc`
   - builds trigger-centered snippets for the AVPC arm
   - extracts note-level AVPC-feature evidence
   - synthesizes one AVPC label row per MRN
4. `run_arm_pipeline.py --arm biomarker`
   - builds trigger-centered snippets for the biomarker arm
   - extracts note-level biomarker evidence
   - synthesizes one biomarker label row per MRN
5. `merge_labels.py`
   - merges the three arm outputs on `DFCI_MRN`
   - assigns the final `v3` bucket

The task-specific arm definitions live under `COMPASS/PROFILE/v3/arms/`.

## One-Command Run

```bash
python COMPASS/PROFILE/v3/run_v3_pipeline.py --mrn-file path/to/mrns.txt --max-workers 4
```

`v3` defaults to raw OncDRS note loading. Use `--text-source compiled` only if you intentionally want the precompiled `prostate_text_data.csv` path.

## Stage Modes

```bash
python COMPASS/PROFILE/v3/run_v3_pipeline.py --prepare-only --mrn-file path/to/mrns.txt
python COMPASS/PROFILE/v3/run_v3_pipeline.py --arms-only --mrn-file path/to/mrns.txt --max-workers 4
python COMPASS/PROFILE/v3/run_v3_pipeline.py --merge-only
```

## Outputs

By default, `v3` writes to `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/v3_outputs/`.

Shared files:

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
- No structured labs, genomics tables, medication tables, or PSA tables are used in this `v3` design.
- `cisplatin` and `carboplatin` are still the platinum terms recognized in note text.
