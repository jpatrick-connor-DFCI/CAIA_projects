# IHC calls for prostate OncoPanel specimens (NEPC project)

**Question:** among OncoPanel specimens with CANCER_TYPE "Prostate Cancer", how many have a pathology report for that exact specimen, how many are neuroendocrine (NE) positive by IHC, and how many look like NEPC?

Built 2026-11-17. Synthetic data with synthetic identifiers.

## Results

These are historical counts from the original extraction, before the corrections below. Rerun both
extraction steps and the LR notebook to regenerate results; these counts have not been recomputed.

| Step | Specimens | Patients |
|---|---|---|
| OncoPanel, `CDM_SPEC_CON` CANCER_TYPE = Prostate Cancer | 1,095 | 1,063 |
| Pathology report found for the sequenced block (**retained**) | **1,004** | **979** |
| Any NE stain done (synaptophysin, chromogranin, INSM1 or CD56) | 116 | 115 |
| **NE positive** (one or more of those four positive) | **55** | **54** |
| **NEPC-like** (NE positive, AR axis negative, no AR-axis marker positive) | **17** | **17** |
| NE positive, AR axis positive (amphicrine) | 27 | 27 |
| NE positive, AR axis not stained | 11 | 11 |

"AR axis" means AR, PSA, NKX3.1, PSAP, PSMA or prostein (P501S).

**Agreement with the OncoPanel diagnosis label:** 25 matched specimens are labelled small-cell or NE in `ORIGINAL_PATH_DIAGNOSIS`. 21 of them are NE positive by IHC and 12 are NEPC-like. The rest usually lack an AR-axis stain or have no stain recorded. Separately, 5 specimens labelled adenocarcinoma come out NEPC-like by IHC.

**The NEPC count depends on the definition:**
- Strict IHC definition: 17.
- Adding the genomics small-cell / NE label: about 22.
- NE positive by IHC or small-cell / NE histology on the report: 61 patients.

The specimen table carries every flag needed to apply any of these.

## Files
| File | One row per | Contents |
|---|---|---|
| `oncopanel_prostate_specimen_ihc.csv` | sequenced specimen (1,095) | SAMPLE_ID, block and matched pathology accession, `path_matched`, OncoPanel diagnosis and `NE_label_oncopanel`, pathology organ / histology, `ihc_phenotype`, flags (`NEPC_ihc`, `amphicrine_ihc`, `NE_stained`, `NE_any_pos`, `ARaxis_any_pos`, `ARaxis_any_neg`, `NE_histology`), Ki-67 %, and one call per marker (see below) |
| `oncopanel_prostate_ihc_long.csv` | stain mention on a matched report (1,013) | marker, result, qualifier, %, lesional / background target, and `RAW_TEXT`, the source sentence, for checking each call |
| `extract_prostate_ihc.py` | | Step 1: extracts prostate and NE IHC for patients with registry C61, malignant prostate pathology, or prostate OncoPanel specimens, and writes `intermediate/` |
| `oncopanel_ihc.py` | | Step 2: matches OncoPanel specimens to their pathology report and builds the two CSVs in `results/` |
| `annotations.py` | | Specimen outcome flags (incl. AR-low) rebuilt from `intermediate/`; used by the IHC LR |
| `paths.py` | | Default input / output locations (cluster layout) and their environment overrides |
| `intermediate/` | | Step 1 output for all prostate cancer patients and all stained specimens, kept for re-use |

**Locations (cluster).** Outputs: `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/ihc_extraction/`, with
`intermediate/` (`prostate_cohort.parquet`, `prostate_ihc_long.parquet`, `prostate_ihc_specimen_all.parquet`) and
`results/` (the two CSVs). Inputs: `/data/gusev/USERS/jpconnor/data/PROFILE_DATA/pathology_extracts/` (`ihc`, `samples`)
and `PROFILE_DATA/parquet/` (`CDM_SPEC_CON_ids`, `IDM_CARG_DIAGNOSIS`). Override with `COMPASS_IHC_EXTRACTION_ROOT`,
`PROFILE_PATHOLOGY_EXTRACTS` and `PROFILE_PARQUET_DIR`.

**Pipeline use.** The COMPASS notebooks do not run the extraction code or read any raw data. `annotations.py`
rebuilds the LR outcomes from the three pre-existing step-1 intermediates in `ihc_extraction/intermediate/`
and testing notebook `COMPASS/notebooks/testing/11_prostate_ihc_extraction.ipynb`
summarises them. `ihc_lr.py` (notebook `text_interpretability/post_landmark/11_LR_for_IHC.ipynb`) uses them:
- **Specimen** = pathology part (MRN, accession, part) with >=1 lesional stain row, restricted to `prostate_relevant`
  (evidence tiers 1-4), sequenced or not. The step-2 OncoPanel CSVs are no longer used.
- **Calls** are recomputed per part from the lesional long rows. A discordant NE marker counts as positive; a
  discordant AR-axis marker counts as neither positive nor negative. `prostate_ihc_specimen_all`'s
  `NE_any_pos` / `ARaxis_any_pos` must agree with the recomputed calls except where a marker is discordant (either
  reading is accepted there), or loading raises. The annotations are therefore the same whichever extraction built the
  intermediates, as long as the long rows are the same.
- **Outcomes:** `ihc_any` (Ki-67-only workups count), `ne_pos`, `nepc_like` (NE+, AR axis negative, none positive), and
  `ar_low` (>=1 AR-axis marker negative, none positive, whatever the NE stains show), dated by the earliest lesional
  `specimen_date`.

The modules here also contain later corrections (patient-keyed joins, discordant calls as both positive and negative,
OncoPanel patients in the cohort, supplement-polarity fixes). The extractor also assigns
supplemental markers their own tissue target and handles negated NE histology in both extraction steps.
Mixed or nonlocal tissue cues give an `unknown` target and do not contribute lesional calls.
These corrections are **not reflected in
the pre-existing intermediates or CSVs**: in those, a discordant NE marker is not NE-positive and supplement calls use the original
polarity rules. To use them, rerun from the repository root (needs polars >= 1.24), which overwrites `intermediate/`
and `results/`:
```bash
python -m COMPASS.prostate_ihc.extract_prostate_ihc && python -m COMPASS.prostate_ihc.oncopanel_ihc
```

**Marker calls:** positive, negative, discordant (both on the same report) or equivocal; RB1 and PTEN use retained / lost. Only tumour (lesional) IHC stains count, and all parts of the matched report are combined. The markers are:
- **AR axis:** AR, PSA, NKX3.1, PSAP, PSMA, prostein, ERG
- **NE:** synaptophysin, chromogranin, INSM1, CD56, ASCL1, NEUROD1, DLL3, SSTR2, TTF-1, YAP1, SOX2
- **Tumour suppressors:** RB1, p53, PTEN
- **Basal:** p63, 34βE12, CK5/6, PIN4
- **Other:** AMACR, GATA3, CK7, CK20, Ki-67

## Methods
- **Specimen matching.** `BLOCK_ACCESSION_NBR` is normalised (dashes removed, leading zeros stripped) and joined on accession and patient. The pathology `ACCESSION` or `OUTSIDE_SPECIMEN_ID` can match. The letter after the year is ignored, because newer pathology reports omit it (BS-23-**W**46905 is written BS-23-46905); without this, 311 more specimens would go unmatched. Of the 91 still unmatched, most are blocks with no report in the pathology extract, such as 2014–2016 cases and outside slides.
- **IHC source.** Calls come from the rule-based pathology extraction (`PROFILE_DATA/pathology_extracts/ihc.parquet`). It was about 93% precise with about 80% line recall on development review, not against a hand-labelled gold standard. Flow cytometry rows are excluded. AMACR, PSMA, PTEN and spelled-out PSA are not named by the extractor; they are recovered by re-parsing the same source sentences (`source == 'supplement'`).

## Caveats
- **Calls are per report, not per tissue core.** A specimen counts as NE positive if any part of the matched report has an NE-positive stain. That part may not be the exact block that was sequenced.
- **Weak staining.** Focal or weak staining counts as positive; see `qualifier` and `RAW_TEXT`.
- **No NE stain means not tested.** Most specimens (888) had no NE stain at all, so "not NE positive" usually means not tested, not negative.
