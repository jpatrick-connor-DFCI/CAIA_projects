import hashlib
import re

import pandas as pd

from common import basic_clean_text
from settings import SNIPPET_CONTEXT_CHARS, SNIPPET_MAX_CHARS, SNIPPET_MAX_MATCHES


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


def build_snippet_text(
    text,
    matches,
    *,
    context_chars=SNIPPET_CONTEXT_CHARS,
    max_chars=SNIPPET_MAX_CHARS,
    max_matches=SNIPPET_MAX_MATCHES,
):
    if not matches:
        return ""

    windows = []
    for match in matches[:max_matches]:
        start = max(0, match["start"] - context_chars)
        end = min(len(text), match["end"] + context_chars)
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
    if len(snippet_text) > max_chars:
        snippet_text = snippet_text[: max_chars - 3].rstrip() + "..."
    return snippet_text


def _trigger_flag_col(label):
    return f"HAS_TRIGGER_{str(label).upper()}"


def extract_trigger_metadata(text, note_type, arm_module):
    cleaned_text = basic_clean_text(text)
    trigger_regex = arm_module.TRIGGER_REGEX
    if not cleaned_text:
        metadata = {
            "HAS_TRIGGER_HIT": False,
            "HAS_PROSTATE_CONTEXT": False,
            "FOCUS_MATCH_COUNT": 0,
            "FOCUS_MATCH_LABELS": "",
            "FOCUS_SNIPPET_TEXT": "",
            "FOCUS_SNIPPET_CHAR_LEN": 0,
        }
        for label in trigger_regex:
            metadata[_trigger_flag_col(label)] = False
        return metadata

    matches = []
    for label, pattern in trigger_regex.items():
        for match in re.finditer(pattern, cleaned_text, flags=re.IGNORECASE):
            matches.append({"label": label, "start": match.start(), "end": match.end()})
    matches = sorted(matches, key=lambda item: (item["start"], item["end"]))

    match_labels = []
    for match in matches:
        if match["label"] not in match_labels:
            match_labels.append(match["label"])

    has_required_label = True
    required_labels = set(getattr(arm_module, "REQUIRED_TRIGGER_LABELS", set()))
    if required_labels:
        has_required_label = bool(required_labels.intersection(match_labels))

    context_regex = getattr(arm_module, "PROSTATE_CONTEXT_REGEX", None)
    require_context = bool(getattr(arm_module, "REQUIRE_PROSTATE_CONTEXT", False))
    has_prostate_context = bool(re.search(context_regex, cleaned_text, flags=re.IGNORECASE)) if context_regex else True
    bypass_note_types = set(getattr(arm_module, "ALLOW_WITHOUT_CONTEXT_NOTE_TYPES", set()))
    bypass_labels = set(getattr(arm_module, "ALLOW_WITHOUT_CONTEXT_LABELS", set()))
    context_ok = (
        not require_context
        or has_prostate_context
        or note_type in bypass_note_types
        or bool(bypass_labels.intersection(match_labels))
    )

    eligible = bool(matches) and has_required_label and context_ok
    snippet_text = build_snippet_text(cleaned_text, matches) if eligible else ""

    metadata = {
        "HAS_TRIGGER_HIT": bool(snippet_text),
        "HAS_PROSTATE_CONTEXT": has_prostate_context,
        "FOCUS_MATCH_COUNT": len(matches) if snippet_text else 0,
        "FOCUS_MATCH_LABELS": "|".join(match_labels) if snippet_text else "",
        "FOCUS_SNIPPET_TEXT": snippet_text,
        "FOCUS_SNIPPET_CHAR_LEN": len(snippet_text),
    }
    for label in trigger_regex:
        metadata[_trigger_flag_col(label)] = label in match_labels and bool(snippet_text)
    return metadata


def annotate_inventory_notes(note_df, arm_module):
    if note_df.empty:
        base_columns = [
            "DFCI_MRN",
            "EVENT_DATE",
            "NOTE_TYPE",
            "CLINICAL_TEXT",
            "FOCUS_MATCH_COUNT",
            "FOCUS_MATCH_LABELS",
            "FOCUS_SNIPPET_CHAR_LEN",
            "SELECTION_REASON",
        ]
        for label in arm_module.TRIGGER_REGEX:
            base_columns.append(_trigger_flag_col(label))
        return pd.DataFrame(columns=base_columns)

    work = note_df.copy()
    work["NOTE_TYPE"] = work["NOTE_TYPE"].fillna("Unknown")
    work["CLINICAL_TEXT"] = work["CLINICAL_TEXT"].fillna("").astype(str)
    work["EVENT_DATE"] = pd.to_datetime(work["EVENT_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")

    focus_metadata = pd.DataFrame(
        [
            extract_trigger_metadata(text, note_type, arm_module)
            for text, note_type in zip(work["CLINICAL_TEXT"], work["NOTE_TYPE"])
        ],
        index=work.index,
    )
    work = pd.concat([work, focus_metadata], axis=1)
    work = work.loc[work["HAS_TRIGGER_HIT"]].copy()
    work["ORIGINAL_NOTE_HASH"] = work["CLINICAL_TEXT"].map(
        lambda text: hashlib.md5(basic_clean_text(text).encode("utf-8", errors="ignore")).hexdigest()
    )
    work["CLINICAL_TEXT"] = work["FOCUS_SNIPPET_TEXT"]
    work["SELECTION_REASON"] = work["FOCUS_MATCH_LABELS"]
    work = work.drop_duplicates(subset=["DFCI_MRN", "NOTE_TYPE", "EVENT_DATE", "ORIGINAL_NOTE_HASH"])
    return work.sort_values(["DFCI_MRN", "EVENT_DATE", "NOTE_TYPE"], na_position="last")
