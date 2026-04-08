# CAIA

Current repository for the CAIA prostate profiling workflow. The active code in this repo is focused on assembling a prostate cancer cohort from local DFCI/Profile exports, preparing note text for manual or LLM review, and generating patient-level labels about why platinum chemotherapy was used.

This is no longer the older OMOP/PySpark preprocessing repository described in prior versions of the README. The current checked-in code is mostly pandas-based and lives under `COMPASS/PROFILE/`.

## Current Repository Structure

```text
CAIA/
├── COMPASS/
│   ├── COMPASS_Cohort_Report.docx
│   ├── COMPASS_Exclusion_and_Threshold_Analysis.docx
│   └── PROFILE/
│       ├── data_preprocessing/
│       │   ├── compile_prostate_data.py
│       │   ├── compile_MRNs_for_manual_review.py
│       │   └── compile_text_for_LLM_review.py
│       ├── regex_generation/
│       │   ├── generate_regex_rules.py
│       │   └── regex_prompts.py
│       ├── generate_LLM_labels.py
│       └── utils.py
├── IPIO/
│   └── IPIO_Cohort_Report.docx
└── README.md
```

## What Is In Scope Right Now

The active workflow supports retrospective review of prostate cancer patients who received platinum chemotherapy, with an emphasis on identifying likely neuroendocrine or small-cell transformation and other clinical reasons for platinum use.

The main pieces are:

- Cohort and source-data compilation from local Profile / OncDRS / embedding-project exports
- Heuristic filtering for prostate patients, PSA records, and platinum exposures
- Candidate-patient selection for manual review
- Note-window extraction for LLM review around platinum start
- Regex-based note cleaning for clinician, imaging, and pathology notes
- Two-stage LLM labeling: per-note extraction followed by patient-level synthesis

## Active Workflow

### 1. Build the prostate cohort data bundle

`COMPASS/PROFILE/data_preprocessing/compile_prostate_data.py`

This script:

- pulls prostate MRNs from `first_treatments_dfci_w_inferred_cancers.csv`
- loads note metadata and batch JSON note files from the clinical text embedding project
- writes a prostate-only note table to `prostate_text_data.csv`
- filters related ICD, health history, medication, lab, and somatic datasets to the same MRNs
- creates convenience outputs for total PSA records and first platinum exposure records

Key implementation detail: paths are hard-coded to shared filesystem locations under `/data/gusev/...`.

### 2. Prepare manual review candidates

`COMPASS/PROFILE/data_preprocessing/compile_MRNs_for_manual_review.py`

This script combines the derived prostate datasets to create a review table for platinum-treated patients. It adds:

- note counts by note type
- whether platinum drug names appear in notes
- non-prostate primary malignancy flags from ICD-10 codes
- PARP inhibitor exposure
- BRCA2-related somatic columns from the somatic data table

Primary output:

- `prostate_pxs_for_review_v2.csv`

### 3. Prepare note text for LLM review

`COMPASS/PROFILE/data_preprocessing/compile_text_for_LLM_review.py`

This script:

- reads existing annotation data from `baca_lab_patient_annotations.tsv`
- keeps patients whose platinum indication is still unlabeled
- joins those patients to the full prostate note table
- keeps notes within +/-90 days of platinum start
- writes the result to `LLM_candidate_text_data.csv`

### 4. Generate regex cleaning rules

`COMPASS/PROFILE/regex_generation/generate_regex_rules.py`

This script samples notes by `NOTE_TYPE`, sends the samples to an Azure OpenAI deployment, and asks the model to propose:

- boilerplate-removal regexes
- regex patterns for extracting structured note elements

Outputs are written under the data directory configured in the script, including:

- per-note-type GPT responses
- a synthesized `generated_rules.py`

The prompt templates for this step live in `COMPASS/PROFILE/regex_generation/regex_prompts.py`.

### 5. Clean notes and run LLM labeling

`COMPASS/PROFILE/generate_LLM_labels.py`

This is the main labeling script. It:

- reads `LLM_candidate_text_data.csv`
- cleans note text with `clean_note(...)` from `COMPASS/PROFILE/utils.py`
- runs per-note extraction prompts against an Azure OpenAI `gpt-4o` deployment
- checkpoints note-level JSON extractions
- runs a second synthesis prompt per patient to infer the primary reason platinum was used
- appends patient-level results to a TSV output
- tracks failures separately and supports `--retry-failures`

Configured outputs:

- `LLM_generated_labels.tsv`
- `LLM_note_extractions.json`
- `LLM_failed_patients.tsv`

## `utils.py`

`COMPASS/PROFILE/utils.py` contains:

- regex cleaning rules shared across note types plus note-type-specific rules
- helper function `clean_note(text, note_type=None)`
- the per-note extraction prompt
- the patient-level platinum-classification synthesis prompt

The current prompts are designed for de-identified oncology notes and classify platinum rationale into categories such as:

- neuroendocrine transformation
- de novo neuroendocrine disease
- clinical trial
- CRPC
- disease progression
- non-prostate primary
- biomarker-driven use

## Dependencies and Runtime Assumptions

There is no packaged environment definition in this repo at the moment. The checked-in scripts assume access to:

- Python with `pandas`, `numpy`, and `tqdm`
- `openai`
- `azure-identity`
- local/shared CSV and JSON data files under `/data/gusev/...`
- valid Azure credentials for `DefaultAzureCredential`
- access to the Azure OpenAI endpoint hard-coded in the LLM scripts

Because paths and endpoints are currently embedded directly in the scripts, this repo is best understood as a research workflow snapshot rather than a portable package.

## Current State of `IPIO`

`IPIO/` currently contains only `IPIO_Cohort_Report.docx`. The older IPIO preprocessing code referenced by earlier README versions is not present in this checkout.

## Notes

- Hidden `.ipynb_checkpoints` and `.DS_Store` files exist in the tree but are not part of the intended workflow.
- Several scripts read and write data outside the repository root.
- If this repo is going to be used by others, the next cleanup step would be to externalize paths, add an environment file, and document the expected input tables explicitly.
