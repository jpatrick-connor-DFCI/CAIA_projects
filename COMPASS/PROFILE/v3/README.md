# PROFILE LLM Extraction v3

Classifies each prostate cancer patient into one of four buckets with a single LLM call:

- **nepc** — neuroendocrine / small-cell prostate cancer or documented histologic transformation
- **avpc** — aggressive-variant / anaplastic language or one or more Aparicio C1–C7 features
- **biomarker** — platinum-relevant biomarker (BRCA1/2, ATM, CDK12, PALB2, HRD/HRR/DDR, MSI-H, MMR, TMB)
- **conventional** — none of the above

Precedence: `nepc > avpc > biomarker > conventional` (the LLM applies it in the same call).

Independently, every patient is also flagged for `has_non_prostate_primary` (e.g., NSCLC, colorectal, urothelial, RCC, lymphoma) — this annotation can co-occur with any primary label.

## Files

```text
v3/
  helpers.py                       # config, triggers, prompt, snippet builder, LLM client
  run_v3_pipeline.py               # main entrypoint
  compile_prostate_note_bundle.py  # optional: pre-compile raw OncDRS notes into a gzip bundle
```

## How it works

1. Load notes for the requested MRNs (from a pre-compiled gzip bundle if present, otherwise raw OncDRS JSONs).
2. Per note, scan for any of the combined NEPC + AVPC + biomarker trigger regexes and build a snippet window around the matches.
3. Per patient, rank triggered notes by `(# trigger categories, # triggers, recency)` and keep the top N (default 25).
4. Send all selected snippets to the LLM in a single call. The model returns the patient's primary label, per-category booleans, supporting quotes, and a rationale.
5. Patients with no triggered notes are written out as `conventional` without an LLM call.

## Recommended run

```bash
# (one-time) compile a gzip note bundle so re-runs don't re-scan raw OncDRS JSON
python COMPASS/PROFILE/v3/compile_prostate_note_bundle.py --mrn-file path/to/prostate_mrns.txt

# classify
python COMPASS/PROFILE/v3/run_v3_pipeline.py --mrn-file path/to/prostate_mrns.txt --max-workers 4
```

If the bundle lives elsewhere: `--note-bundle-path path/to/LLM_v3_prostate_note_bundle.json.gz`.

## One-command raw run

```bash
python COMPASS/PROFILE/v3/run_v3_pipeline.py --mrn-file path/to/mrns.txt --max-workers 4
```

When no bundle exists at the expected path, the pipeline falls back to scanning raw OncDRS JSONs directly.

## Outputs

By default, `v3` writes to `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/v3_outputs/`.

- `LLM_v3_prostate_note_bundle.json.gz` — optional pre-compiled note bundle
- `LLM_v3_labels.tsv` — one row per patient with the final classification, supporting quotes, confidence, and rationale
- `LLM_v3_failed_patients.tsv` — appended for any patient whose LLM call errored

`LLM_v3_labels.tsv` columns:

```text
DFCI_MRN, primary_label,
has_nepc, has_avpc, has_biomarker, has_non_prostate_primary,
biomarker_genes, avpc_criteria, non_prostate_primary_types,
supporting_quotes, supporting_quote_dates,
confidence, rationale, num_snippets
```

The pipeline is resumable: re-running skips MRNs already present in `LLM_v3_labels.tsv`. Use `--overwrite` to start fresh.

## Useful flags

```text
--max-notes-per-patient N    # cap selected snippets per patient (default 25)
--max-workers N              # concurrent patient classifications (default 4)
--limit-mrns N               # cap how many MRNs to process this run
--model NAME                 # override the Azure OpenAI deployment (default gpt-4o)
--overwrite                  # delete prior labels/failures and start over
```

## Notes

- All triggers are matched on `clean_note`-cleaned text. Each note's snippet is capped at ~8000 chars; per-patient snippet counts are capped (default 25), so a single LLM call typically sees ~50k tokens of focused context (~40% of gpt-4o's 128k window).
- No structured labs, genomics tables, medication tables, or PSA tables are used in this `v3` design — all signal comes from note text.
- `cisplatin` and `carboplatin` are no longer used as triggers; biomarker selection is driven by the molecular terms above.
