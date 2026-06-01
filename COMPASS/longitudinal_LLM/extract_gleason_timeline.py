"""Pipeline 1 — Longitudinal Gleason score extraction.

For every prostate patient, any note mentioning a Gleason score / Grade Group /
ISUP grade is collected, and one LLM call per note extracts each documented
Gleason score with the date the grade was assigned. The per-note extractions are
aggregated and de-duplicated into a per-patient timeline: every distinct Gleason
score the patient received, with its date.

Outputs (under <output-dir>):
  gleason_timeline.tsv          deduped timeline (every score + date per patient)
  gleason_extractions_raw.tsv   per-note extractions (provenance, pre-dedup)
  gleason_processed_notes.tsv   processed-note log (resumability + failures)
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from helpers import (
    CLINICAL_SAFETY_CONTEXT,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_NAME,
    PROSTATE_TEXT_CSV,
    build_client,
    call_with_retry,
    derive_grade_group,
    filter_note_types,
    iter_note_snippets,
    load_notes,
    load_selected_mrns,
    parse_json_response,
    resolve_date,
)

DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_PATH) / "LLM_gleason_timeline"

# Any mention of Gleason / Grade Group / ISUP grading collects the note.
TRIGGER_REGEX = {
    "gleason": r"\b(?:gleason|grade\s+group|isup(?:\s+grade)?)\b",
}

RAW_COLUMNS = [
    "note_uid",
    "DFCI_MRN",
    "note_date",
    "note_type",
    "gleason_primary",
    "gleason_secondary",
    "gleason_total",
    "grade_group",
    "specimen_type",
    "scoring_date",
    "is_historical_reference",
    "quote",
]

TIMELINE_COLUMNS = [
    "DFCI_MRN",
    "gleason_date",
    "date_source",
    "gleason_primary",
    "gleason_secondary",
    "gleason_total",
    "grade_group",
    "specimen_type",
    "is_historical_reference",
    "supporting_quote",
    "note_date",
    "note_type",
]

PROCESSED_COLUMNS = ["note_uid", "DFCI_MRN", "num_findings", "status"]

SYSTEM_PROMPT = """
You are a clinical data extraction system for an IRB-approved prostate cancer research study.

You will receive a JSON payload with ONE de-identified clinical note snippet for a
single patient. The snippet was selected because it mentions a Gleason score,
Grade Group, or ISUP grade.

## TASK
Extract EVERY distinct Gleason score documented in the snippet. A single note may
report more than one (e.g., a current biopsy result plus a historical
prostatectomy score). For each distinct score, report:
- primary: primary Gleason pattern as an integer 1-5 (null if only a grade group is given).
- secondary: secondary Gleason pattern as an integer 1-5 (null if only a grade group is given).
- total: total Gleason sum as an integer 2-10 (null if not derivable from the text).
- grade_group: ISUP Grade Group 1-5 if explicitly stated; otherwise null (it will be derived).
- specimen_type: one of "biopsy", "prostatectomy", "TURP", "metastasis", "unknown".
- scoring_date: the date the specimen was obtained / the grade was originally assigned,
  AS STATED in the text (YYYY-MM-DD; for partial dates use the first of the month/year).
  If the snippet states no date for this score, return null.
- is_historical_reference: true if the score is quoted from a prior/outside report;
  false if it is the result being newly reported in this note.
- quote: a verbatim excerpt (~20-60 words) containing the score.

## RULES
- Extract only scores explicitly documented. Never infer or compute a score that is not written.
- Treat separate specimens or separate dates as separate entries; do not merge them.
- If the identical score is restated several times in the snippet, report it once.
- Planned, pending, or "awaiting" pathology is NOT a score.

## OUTPUT FORMAT
Return ONLY valid JSON:
{
  "gleason_findings": [
    {"primary": 4, "secondary": 3, "total": 7, "grade_group": 3,
     "specimen_type": "biopsy", "scoring_date": "2019-03-01",
     "is_historical_reference": false, "quote": "<verbatim>"}
  ]
}
If no actual Gleason score is documented, return {"gleason_findings": []}.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a longitudinal Gleason-score timeline per prostate patient via the LLM."
    )
    parser.add_argument("--mrn-file", type=Path, default=None)
    parser.add_argument("--mrns", default=None)
    parser.add_argument("--notes-csv", type=Path, default=PROSTATE_TEXT_CSV)
    parser.add_argument("--note-bundle-path", type=Path, default=None)
    parser.add_argument("--raw-text-path", type=Path, action="append", default=None)
    parser.add_argument(
        "--note-types",
        nargs="+",
        default=None,
        help="Restrict to these NOTE_TYPE values (e.g. Pathology). Default: all notes. "
        "Gleason is authoritatively assigned in pathology, so 'Pathology' is far "
        "cheaper and higher-fidelity.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=600,
        help="Chars of context kept on each side of a Gleason match. Smaller windows "
        "raise the copy-forward dedup hit-rate and shrink per-call tokens.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit-notes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def append_rows(path, rows, columns):
    if not rows:
        return
    pd.DataFrame(rows, columns=columns).to_csv(
        path,
        mode="a",
        sep="\t",
        index=False,
        header=not path.exists() or path.stat().st_size == 0,
    )


def extract_note(client, model, max_retries, item):
    payload = {
        "patient_mrn": int(item["DFCI_MRN"]),
        "note_date": item["note_date"],
        "note_type": item["note_type"],
        "note_text": item["snippet"],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + CLINICAL_SAFETY_CONTEXT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    response_text, error = call_with_retry(client, model, messages, max_retries)
    if error:
        return None, error
    try:
        result = parse_json_response(response_text)
    except json.JSONDecodeError as exc:
        return None, f"json_parse: {exc}"
    if not isinstance(result, dict):
        return None, f"non_dict_response: {type(result).__name__}"
    findings = result.get("gleason_findings")
    if not isinstance(findings, list):
        return None, "missing_gleason_findings"
    return findings, None


def raw_rows_from_findings(item, findings):
    rows = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        rows.append({
            "note_uid": item["note_uid"],
            "DFCI_MRN": int(item["DFCI_MRN"]),
            "note_date": item["note_date"],
            "note_type": item["note_type"],
            "gleason_primary": finding.get("primary"),
            "gleason_secondary": finding.get("secondary"),
            "gleason_total": finding.get("total"),
            "grade_group": finding.get("grade_group"),
            "specimen_type": finding.get("specimen_type"),
            "scoring_date": finding.get("scoring_date"),
            "is_historical_reference": finding.get("is_historical_reference"),
            "quote": finding.get("quote"),
        })
    return rows


def _to_int(value):
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed)


def build_timeline(raw_path, timeline_path):
    """Resolve dates, validate, and de-duplicate raw extractions into the timeline."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        pd.DataFrame(columns=TIMELINE_COLUMNS).to_csv(timeline_path, sep="\t", index=False)
        return 0

    raw = pd.read_csv(raw_path, sep="\t")
    seen = set()
    rows = []
    for r in raw.itertuples(index=False):
        primary = _to_int(getattr(r, "gleason_primary", None))
        secondary = _to_int(getattr(r, "gleason_secondary", None))
        total = _to_int(getattr(r, "gleason_total", None))
        # Gleason total is primary + secondary by definition; recompute it when
        # both patterns are present so an LLM arithmetic slip can't propagate.
        if primary is not None and secondary is not None:
            total = primary + secondary
        # Require a usable total; drop grade-group-only or malformed extractions.
        if total is None or not (2 <= total <= 10):
            continue
        if primary is not None and not (1 <= primary <= 5):
            continue
        if secondary is not None and not (1 <= secondary <= 5):
            continue

        grade_group = _to_int(getattr(r, "grade_group", None))
        if grade_group is None or not (1 <= grade_group <= 5):
            grade_group = derive_grade_group(primary, secondary)

        gleason_date, date_source = resolve_date(
            getattr(r, "scoring_date", None), getattr(r, "note_date", None)
        )
        specimen_type = getattr(r, "specimen_type", None)
        mrn = int(getattr(r, "DFCI_MRN"))

        key = (mrn, primary, secondary, total, gleason_date, specimen_type)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "DFCI_MRN": mrn,
            "gleason_date": gleason_date,
            "date_source": date_source,
            "gleason_primary": primary,
            "gleason_secondary": secondary,
            "gleason_total": total,
            "grade_group": grade_group,
            "specimen_type": specimen_type,
            "is_historical_reference": getattr(r, "is_historical_reference", None),
            "supporting_quote": getattr(r, "quote", None),
            "note_date": getattr(r, "note_date", None),
            "note_type": getattr(r, "note_type", None),
        })

    timeline = pd.DataFrame(rows, columns=TIMELINE_COLUMNS)
    if not timeline.empty:
        # Nullable Int64 so integer grades render as "3"/"<NA>", not "3.0"/"NaN".
        for col in ("gleason_primary", "gleason_secondary", "gleason_total", "grade_group"):
            timeline[col] = timeline[col].astype("Int64")
        timeline = timeline.sort_values(
            ["DFCI_MRN", "gleason_date"], na_position="last"
        )
    timeline.to_csv(timeline_path, sep="\t", index=False)
    return len(timeline)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "gleason_extractions_raw.tsv"
    processed_path = args.output_dir / "gleason_processed_notes.tsv"
    timeline_path = args.output_dir / "gleason_timeline.tsv"

    if args.overwrite:
        for path in (raw_path, processed_path, timeline_path):
            path.unlink(missing_ok=True)

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    notes_df = load_notes(
        csv_path=args.notes_csv,
        bundle_path=args.note_bundle_path,
        raw_text_paths=args.raw_text_path,
        selected_mrns=selected_mrns,
    )
    print(
        f"Loaded notes: {len(notes_df)} rows for "
        f"{notes_df['DFCI_MRN'].nunique()} patients"
    )

    if args.note_types:
        notes_df = filter_note_types(notes_df, args.note_types)
        print(f"After note-type filter {args.note_types}: {len(notes_df)} rows")

    items = list(
        iter_note_snippets(notes_df, TRIGGER_REGEX, context_chars=args.context_chars)
    )
    print(f"Notes mentioning Gleason (deduped): {len(items)}")

    completed = set()
    if processed_path.exists() and processed_path.stat().st_size > 0:
        completed = set(pd.read_csv(processed_path, sep="\t")["note_uid"].astype(str))
    print(f"Already processed notes: {len(completed)}")

    todo = [it for it in items if it["note_uid"] not in completed]
    if args.limit_notes is not None:
        todo = todo[: args.limit_notes]
    print(f"Notes to extract with LLM: {len(todo)}")

    if todo:
        client = build_client()

        def worker(item):
            findings, error = extract_note(client, args.model, args.max_retries, item)
            return item, findings, error

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(worker, it): it for it in todo}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Notes", unit="note"
            ):
                item, findings, error = future.result()
                if error or findings is None:
                    append_rows(
                        processed_path,
                        [{
                            "note_uid": item["note_uid"],
                            "DFCI_MRN": int(item["DFCI_MRN"]),
                            "num_findings": 0,
                            "status": error or "no_result",
                        }],
                        PROCESSED_COLUMNS,
                    )
                    continue
                rows = raw_rows_from_findings(item, findings)
                append_rows(raw_path, rows, RAW_COLUMNS)
                append_rows(
                    processed_path,
                    [{
                        "note_uid": item["note_uid"],
                        "DFCI_MRN": int(item["DFCI_MRN"]),
                        "num_findings": len(rows),
                        "status": "ok",
                    }],
                    PROCESSED_COLUMNS,
                )

    n = build_timeline(raw_path, timeline_path)
    print(f"Wrote Gleason timeline ({n} rows): {timeline_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
