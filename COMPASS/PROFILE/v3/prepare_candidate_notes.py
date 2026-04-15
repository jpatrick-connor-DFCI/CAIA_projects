import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

from common import (
    basic_clean_text,
    load_note_text_dataframe,
    load_selected_mrns,
    normalize_mrn_column,
    parse_datetime_series,
)
from config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_TEXT_PATHS,
    FALLBACK_NOTE_MAX_CHARS,
    NOTE_TYPE_LIMITS,
    PLATINUM_FALLBACK_LIMITS,
    PLATINUM_WINDOW_AFTER_DAYS,
    PLATINUM_WINDOW_BEFORE_DAYS,
    PROSTATE_CONTEXT_REGEX,
    SNIPPET_CONTEXT_CHARS,
    SNIPPET_MAX_CHARS,
    SNIPPET_MAX_MATCHES,
    TRIGGER_REGEX,
    TRIGGER_SCORE_WEIGHTS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build v3 candidate notes for unified NEPC/AVPC/platinum extraction."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--context-path",
        type=Path,
        default=None,
        help="Optional patient context CSV. Defaults to output-dir/LLM_v3_patient_context.csv.",
    )
    parser.add_argument(
        "--text-source",
        choices=["compiled", "raw"],
        default="compiled",
        help="Use compiled prostate_text_data.csv or raw OncDRS JSON notes.",
    )
    parser.add_argument(
        "--raw-text-path",
        type=Path,
        action="append",
        default=None,
        help="Raw OncDRS note directory. Repeat to search multiple directories.",
    )
    parser.add_argument("--mrns", default=None, help="Comma-separated DFCI_MRN values to include.")
    parser.add_argument(
        "--mrn-file",
        type=Path,
        default=None,
        help="Optional text/CSV/TSV file containing DFCI_MRN values.",
    )
    parser.add_argument("--max-clinician-notes", type=int, default=NOTE_TYPE_LIMITS["Clinician"])
    parser.add_argument("--max-imaging-notes", type=int, default=NOTE_TYPE_LIMITS["Imaging"])
    parser.add_argument("--max-pathology-notes", type=int, default=NOTE_TYPE_LIMITS["Pathology"])
    parser.add_argument("--platinum-window-before-days", type=int, default=PLATINUM_WINDOW_BEFORE_DAYS)
    parser.add_argument("--platinum-window-after-days", type=int, default=PLATINUM_WINDOW_AFTER_DAYS)
    return parser.parse_args()


def resolve_raw_text_paths(raw_text_paths_arg=None):
    if raw_text_paths_arg:
        ordered_paths = []
        seen = set()
        for path in raw_text_paths_arg:
            normalized = Path(path)
            key = str(normalized)
            if key not in seen:
                seen.add(key)
                ordered_paths.append(normalized)
        return ordered_paths
    return list(DEFAULT_RAW_TEXT_PATHS)


def merge_windows(windows, gap_chars=80):
    if not windows:
        return []
    ordered = sorted(windows)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap_chars:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_snippet_text(text, matches):
    if not matches:
        return ""

    windows = []
    for match in matches[:SNIPPET_MAX_MATCHES]:
        start = max(0, match["start"] - SNIPPET_CONTEXT_CHARS)
        end = min(len(text), match["end"] + SNIPPET_CONTEXT_CHARS)
        windows.append((start, end))

    snippets = []
    for start, end in merge_windows(windows):
        snippet = text[start:end].strip()
        if not snippet:
            continue
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        snippets.append(snippet)

    snippet_text = "\n\n...\n\n".join(snippets)
    if len(snippet_text) > SNIPPET_MAX_CHARS:
        snippet_text = snippet_text[: SNIPPET_MAX_CHARS - 3].rstrip() + "..."
    return snippet_text


def build_fallback_text(text):
    cleaned = basic_clean_text(text)
    if len(cleaned) <= FALLBACK_NOTE_MAX_CHARS:
        return cleaned
    return cleaned[: FALLBACK_NOTE_MAX_CHARS - 3].rstrip() + "..."


def trigger_flag_col(label):
    return f"HAS_TRIGGER_{label.upper()}"


def extract_focus_grep_metadata(text, note_type):
    cleaned_text = basic_clean_text(text)
    if not cleaned_text:
        empty = {
            "HAS_FOCUSED_TRIGGER": False,
            "HAS_PROSTATE_CONTEXT": False,
            "FOCUS_MATCH_COUNT": 0,
            "FOCUS_MATCH_LABELS": "",
            "FOCUS_SNIPPET_TEXT": "",
            "FOCUS_SNIPPET_CHAR_LEN": 0,
        }
        for label in TRIGGER_REGEX:
            empty[trigger_flag_col(label)] = False
        return empty

    matches = []
    match_labels = []
    for label, pattern in TRIGGER_REGEX.items():
        for match in re.finditer(pattern, cleaned_text, flags=re.IGNORECASE):
            matches.append({"label": label, "start": match.start(), "end": match.end()})
            if label not in match_labels:
                match_labels.append(label)

    has_prostate_context = bool(re.search(PROSTATE_CONTEXT_REGEX, cleaned_text, flags=re.IGNORECASE))
    eligible = bool(matches) and (
        has_prostate_context
        or note_type == "Pathology"
        or "platinum" in match_labels
        or "biomarker" in match_labels
    )
    snippet_text = build_snippet_text(cleaned_text, matches) if eligible else ""

    metadata = {
        "HAS_FOCUSED_TRIGGER": bool(snippet_text),
        "HAS_PROSTATE_CONTEXT": has_prostate_context,
        "FOCUS_MATCH_COUNT": len(matches) if snippet_text else 0,
        "FOCUS_MATCH_LABELS": "|".join(match_labels) if snippet_text else "",
        "FOCUS_SNIPPET_TEXT": snippet_text,
        "FOCUS_SNIPPET_CHAR_LEN": len(snippet_text),
    }
    for label in TRIGGER_REGEX:
        metadata[trigger_flag_col(label)] = label in match_labels and bool(snippet_text)
    return metadata


def annotate_text_triggers(text_df):
    work = text_df.copy()
    work["NOTE_TYPE"] = work["NOTE_TYPE"].fillna("Unknown")
    work["EVENT_DATE"] = parse_datetime_series(work["EVENT_DATE"])
    work["CLINICAL_TEXT"] = work["CLINICAL_TEXT"].fillna("").astype(str)

    focus_metadata = pd.DataFrame(
        [
            extract_focus_grep_metadata(text, note_type)
            for text, note_type in zip(work["CLINICAL_TEXT"], work["NOTE_TYPE"])
        ],
        index=work.index,
    )
    work = pd.concat([work, focus_metadata], axis=1)
    work["CLEANED_TEXT"] = work["CLINICAL_TEXT"].map(basic_clean_text)
    work["NOTE_TEXT_HASH"] = work["CLEANED_TEXT"].apply(
        lambda text: hashlib.md5(text.strip().encode("utf-8", errors="ignore")).hexdigest()
    )

    category_cols = [trigger_flag_col(label) for label in TRIGGER_REGEX]
    work["TRIGGER_CATEGORY_COUNT"] = work[category_cols].sum(axis=1)
    work["TRIGGER_SCORE"] = 0
    for label, weight in TRIGGER_SCORE_WEIGHTS.items():
        work["TRIGGER_SCORE"] += work[trigger_flag_col(label)].astype(int) * weight
    work["TRIGGER_SCORE"] += work["HAS_PROSTATE_CONTEXT"].astype(int)
    work["TRIGGER_SCORE"] += work["FOCUS_MATCH_COUNT"].clip(upper=6)
    return work


def selection_reason(row):
    reasons = []
    for label in TRIGGER_REGEX:
        if row.get(trigger_flag_col(label), False):
            reasons.append(f"grep_{label}")
    if row.get("HAS_PROSTATE_CONTEXT", False):
        reasons.append("prostate_context")
    if row.get("FALLBACK_INCLUDED", False):
        reasons.append("platinum_window_fallback")
    return "|".join(reasons) if reasons else "unspecified"


def take_with_coverage(df, limit):
    if df.empty or len(df) <= limit:
        return df

    work = df.copy()
    work["EVENT_DATE_RANK"] = work["EVENT_DATE"].fillna(pd.Timestamp.min)
    keep_indices = []

    dated = work.loc[work["EVENT_DATE"].notna()]
    if not dated.empty:
        keep_indices.append(dated["EVENT_DATE"].idxmin())
        keep_indices.append(dated["EVENT_DATE"].idxmax())

    priority = work.sort_values(
        ["TRIGGER_SCORE", "TRIGGER_CATEGORY_COUNT", "EVENT_DATE_RANK"],
        ascending=[False, False, False],
    )
    for idx in priority.index:
        if idx not in keep_indices:
            keep_indices.append(idx)
        if len(keep_indices) >= limit:
            break

    return work.loc[keep_indices[:limit]].drop(columns=["EVENT_DATE_RANK"], errors="ignore")


def compute_platinum_window_flags(work, context_df, before_days, after_days):
    work = work.copy()
    work["WITHIN_PLATINUM_WINDOW"] = False
    work["DAYS_TO_PLATINUM"] = pd.NA
    return work


def build_selected_note_text(df):
    work = df.copy()
    work["SELECTED_TEXT"] = work["FOCUS_SNIPPET_TEXT"]
    fallback_mask = work["SELECTED_TEXT"].fillna("").eq("")
    if fallback_mask.any():
        work.loc[fallback_mask, "SELECTED_TEXT"] = work.loc[fallback_mask, "CLEANED_TEXT"].map(build_fallback_text)
    return work


def select_patient_notes(patient_df, args):
    patient_df = patient_df.copy()
    patient_df["FALLBACK_INCLUDED"] = False

    selected_parts = []
    for note_type, limit in [
        ("Pathology", args.max_pathology_notes),
        ("Imaging", args.max_imaging_notes),
        ("Clinician", args.max_clinician_notes),
    ]:
        note_df = patient_df.loc[
            (patient_df["NOTE_TYPE"] == note_type) & patient_df["HAS_FOCUSED_TRIGGER"]
        ].copy()
        note_df["SELECTION_REASON"] = note_df.apply(selection_reason, axis=1)
        selected_parts.append(take_with_coverage(note_df, limit))

    selected_df = (
        pd.concat(selected_parts, ignore_index=False) if selected_parts else patient_df.iloc[0:0].copy()
    )
    selected_keys = set(selected_df.index.tolist())

    if patient_df["WITHIN_PLATINUM_WINDOW"].any():
        fallback_parts = []
        for note_type, limit in PLATINUM_FALLBACK_LIMITS.items():
            fallback_df = patient_df.loc[
                (patient_df["NOTE_TYPE"] == note_type)
                & patient_df["WITHIN_PLATINUM_WINDOW"]
                & (~patient_df.index.isin(selected_keys))
            ].copy()
            if fallback_df.empty:
                continue
            fallback_df["PLATINUM_DISTANCE"] = fallback_df["DAYS_TO_PLATINUM"].abs()
            fallback_df = fallback_df.sort_values(
                ["HAS_FOCUSED_TRIGGER", "PLATINUM_DISTANCE", "EVENT_DATE"],
                ascending=[False, True, False],
            ).head(limit)
            fallback_df["FALLBACK_INCLUDED"] = True
            fallback_df["SELECTION_REASON"] = fallback_df.apply(selection_reason, axis=1)
            fallback_parts.append(fallback_df.drop(columns=["PLATINUM_DISTANCE"], errors="ignore"))

        if fallback_parts:
            selected_df = pd.concat([selected_df] + fallback_parts, ignore_index=False)

    selected_df = selected_df.drop_duplicates(subset=["DFCI_MRN", "NOTE_TYPE", "EVENT_DATE", "NOTE_TEXT_HASH"])
    selected_df = build_selected_note_text(selected_df)
    return selected_df.sort_values(["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"], na_position="last")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_mrns = load_selected_mrns(args.mrns, args.mrn_file)
    raw_text_paths = resolve_raw_text_paths(args.raw_text_path)
    context_path = args.context_path or args.output_dir / "LLM_v3_patient_context.csv"
    context_df = normalize_mrn_column(pd.read_csv(context_path, low_memory=False))

    if "DFCI_MRN" not in context_df.columns:
        raise ValueError(f"Context file is missing DFCI_MRN: {context_path}")

    if selected_mrns is None:
        selected_mrns = set(context_df["DFCI_MRN"].dropna().astype(int).unique().tolist())
    else:
        context_df = context_df.loc[context_df["DFCI_MRN"].isin(selected_mrns)].copy()

    text_df = load_note_text_dataframe(args.text_source, args.data_path, raw_text_paths, selected_mrns)
    text_df = text_df.loc[text_df["DFCI_MRN"].isin(context_df["DFCI_MRN"])].copy()

    annotated_text_df = annotate_text_triggers(text_df)
    annotated_text_df = compute_platinum_window_flags(
        annotated_text_df,
        context_df,
        args.platinum_window_before_days,
        args.platinum_window_after_days,
    )

    selected_notes = []
    for _, patient_df in annotated_text_df.groupby("DFCI_MRN"):
        selected_notes.append(select_patient_notes(patient_df, args))

    candidate_df = pd.concat(selected_notes, ignore_index=True) if selected_notes else pd.DataFrame()
    trigger_cols = [trigger_flag_col(label) for label in TRIGGER_REGEX]
    keep_cols = [
        "DFCI_MRN",
        "EVENT_DATE",
        "NOTE_TYPE",
        "SELECTED_TEXT",
        "FOCUS_SNIPPET_CHAR_LEN",
        "FOCUS_MATCH_COUNT",
        "FOCUS_MATCH_LABELS",
        "TRIGGER_CATEGORY_COUNT",
        "TRIGGER_SCORE",
        "SELECTION_REASON",
        "HAS_FOCUSED_TRIGGER",
        "HAS_PROSTATE_CONTEXT",
        "WITHIN_PLATINUM_WINDOW",
        "DAYS_TO_PLATINUM",
        "FALLBACK_INCLUDED",
    ] + trigger_cols

    if candidate_df.empty:
        candidate_df = pd.DataFrame(columns=keep_cols)
    else:
        candidate_df = (
            candidate_df[keep_cols]
            .rename(columns={"SELECTED_TEXT": "CLINICAL_TEXT"})
            .sort_values(["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"])
        )

    if "EVENT_DATE" in candidate_df.columns:
        candidate_df["EVENT_DATE"] = pd.to_datetime(candidate_df["EVENT_DATE"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )

    candidate_path = args.output_dir / "LLM_v3_candidate_text_data.csv"
    candidate_df.to_csv(candidate_path, index=False)

    print(f"Wrote candidate notes: {candidate_path}")
    print(f"Patients with selected notes: {candidate_df['DFCI_MRN'].nunique() if not candidate_df.empty else 0}")
    print(f"Selected notes: {len(candidate_df)}")
    print(f"Text source: {args.text_source}")
    if args.text_source == "raw":
        print(f"Raw text directories searched: {', '.join(str(path) for path in raw_text_paths)}")


if __name__ == "__main__":
    main()
