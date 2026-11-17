"""Gated vs ungated NE-positive IHC notes: an audit of the LLM NE gate.

Backs ``notebooks/testing/12_review_false_pos.ipynb``. Notebook 11 calls a qualifying
IHC note NE-positive only when its report has an LLM NEPC finding that passes
LCA's ``gate_findings`` (trial-context screen, then ``ne_gate.screen_ne_quote``).
This module re-runs that per-finding screen, so every finding carries the step
and reason that kept or dropped it, and sorts each qualifying IHC note of a scored
patient into one review group:

* ``gated``             -- >=1 finding passes the gate (nb11 NE-positive).
* ``gate_rejected``     -- >=1 finding passes the trial-context screen, but none passes
                           ne_gate ("ungated-only" positive).
* ``no_finding``        -- NE marker in the report; the LLM read it and returned no finding.
* ``not_run_or_failed`` -- NE marker; in the LLM evidence, but the final attempt is
                           missing or failed.
* ``not_in_evidence``   -- NE marker; never selected into the LLM evidence set.
* ``trial_context_only``-- every finding was dropped by the trial-context screen
                           (counted, not put on the review sheet).

"Ungated" removes ne_gate only. The trial-context screen and final-attempt
compaction stay on (``load_label_inputs`` already applies both). The review sheet
is for manual calls (``true_ne`` / ``not_ne`` / ``unclear``), and the read-back
functions turn a filled sheet into gate precision, missed-positive rates and the
event counts each definition would give notebook 11. Nothing here refits a model.
LCA is imported read-only.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from COMPASS.multivariable_models import ihc_lr as il

LOCAL_ID_COL = il.LOCAL_ID_COL
ID_COL = il.ID_COL

REVIEW_GROUPS = ("gated", "gate_rejected", "no_finding", "not_run_or_failed", "not_in_evidence")
UNREVIEWED_GROUPS = ("trial_context_only",)
REVIEWER_CALLS = ("true_ne", "not_ne", "unclear")
STAGE_PASSED, STAGE_NE_GATE, STAGE_TRIAL = "passed", "ne_gate", "trial_context"
DEFAULT_WINDOW_CHARS = 250
DEFAULT_MAX_TEXT_CHARS = 4000

FINDING_COLUMNS = [LOCAL_ID_COL, "RPT_ID", "note_uid", "finding_index", "attempt_id",
                   "evidence_type", "specimen_site", "confidence", "quote",
                   "trial_reason", "gate_stage", "gate_reason"]
SUMMARY_COLUMNS = ["review_group", "notes", "reviewed", "true_ne", "not_ne", "unclear",
                   "frac_true_ne", "frac_true_ne_lo", "frac_true_ne_hi"]


# --------------------------------------------------------------------------- #
# LLM inputs
# --------------------------------------------------------------------------- #
def load_llm_inputs(lca, llm_nepc_dir):
    """``load_label_inputs`` (final attempts, trial reasons attached) plus its final-attempt ledger."""
    ne = lca["note_extraction"]
    inputs = ne.load_label_inputs(lca["nepc"].SPEC, Path(llm_nepc_dir))
    ledger = ne._final_attempts(ne._read(inputs["paths"]["processed"], ne.LEDGER_SCHEMA))
    return inputs, ledger


def _to_pandas(frame) -> pd.DataFrame:
    return frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)


def _pathology_rpt_id(note_uid):
    note_type, rpt_id = il.parse_note_uid(note_uid)
    if note_type != "Pathology" or rpt_id in (None, "", "None"):
        return None
    if re.search(r"#\d+$", rpt_id):
        raise ValueError("Ambiguous pathology note UID in LCA evidence; resolve report "
                         "identity before reviewing.")
    return rpt_id


def finding_audit(lca, inputs, *, mrns=None) -> pd.DataFrame:
    """One row per final-attempt pathology finding with the gate step that kept or dropped it.

    Re-runs ``gate_findings``' per-row logic (trial-context reason first, then
    ``screen_ne_quote``) and fails unless the passing rows equal ``gate_findings``' own.
    """
    selected = None if mrns is None else {int(m) for m in mrns}
    f = _to_pandas(inputs["findings"])
    if "_trial_context_reason" not in f.columns:
        raise ValueError("Findings lack _trial_context_reason; load them with load_label_inputs.")
    if f.empty:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    f = f[f[LOCAL_ID_COL].notna()].copy()
    f[LOCAL_ID_COL] = f[LOCAL_ID_COL].astype(int)
    if selected is not None:
        f = f[f[LOCAL_ID_COL].isin(selected)]
    f["RPT_ID"] = f["note_uid"].map(_pathology_rpt_id)
    f = f[f["RPT_ID"].notna()].copy()

    screen = lca["nepc"].screen_ne_quote
    stages, reasons = [], []
    for trial, quote, etype in zip(f["_trial_context_reason"], f["quote"], f["evidence_type"]):
        if isinstance(trial, str) and trial:
            stages.append(STAGE_TRIAL)
            reasons.append(trial)
            continue
        ok, reason = screen(quote, etype)
        stages.append(STAGE_PASSED if ok else STAGE_NE_GATE)
        reasons.append(None if ok else reason)
    f["gate_stage"], f["gate_reason"] = stages, reasons
    f = f.rename(columns={"_trial_context_reason": "trial_reason"})

    gated, _ = lca["nepc"].gate_findings(inputs["findings"])
    theirs = {(int(m), r["note_uid"], int(r["finding_index"]))
              for m, rows in gated.items() if selected is None or int(m) in selected
              for r in rows if il.parse_note_uid(r["note_uid"])[0] == "Pathology"}
    mine = set(zip(f.loc[f["gate_stage"].eq(STAGE_PASSED), LOCAL_ID_COL],
                   f.loc[f["gate_stage"].eq(STAGE_PASSED), "note_uid"],
                   f.loc[f["gate_stage"].eq(STAGE_PASSED), "finding_index"].astype(int)))
    if mine != theirs:
        raise RuntimeError(f"Re-run gate disagrees with gate_findings ({len(mine ^ theirs)} "
                           "findings); the LCA gate logic has changed.")
    return f.reindex(columns=FINDING_COLUMNS).reset_index(drop=True)


def llm_note_table(inputs, ledger, *, mrns=None) -> pd.DataFrame:
    """One row per pathology evidence note: final-attempt status and the snippet the LLM read."""
    selected = None if mrns is None else {int(m) for m in mrns}
    led = _to_pandas(ledger)
    status = {(int(m), u): s for m, u, s in zip(led.get(LOCAL_ID_COL, []), led.get("note_uid", []),
                                                led.get("status", []))}
    rows = []
    for (mrn, uid), note in inputs["notes"].items():
        if selected is not None and int(mrn) not in selected:
            continue
        rpt_id = _pathology_rpt_id(uid)
        if rpt_id is None:
            continue
        rows.append({LOCAL_ID_COL: int(mrn), "RPT_ID": rpt_id, "in_evidence": True,
                     "llm_status": status.get((int(mrn), uid)) or "not_run",
                     "snippet": (note or {}).get("snippet", "")})
    cols = [LOCAL_ID_COL, "RPT_ID", "in_evidence", "llm_status", "snippet"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def _finding_summary(findings: pd.DataFrame) -> pd.DataFrame:
    keys = [LOCAL_ID_COL, "RPT_ID"]
    cols = [*keys, "n_findings", "n_gated", "n_ne_gate_rejected", "n_trial_rejected",
            "evidence_types", "gate_reasons", "trial_reasons", "quotes"]
    if findings.empty:
        return pd.DataFrame(columns=cols)
    f = findings.sort_values([*keys, "finding_index"]).copy()
    stage, etype = f["gate_stage"], f["evidence_type"].astype(str)
    f["_pass"] = stage.eq(STAGE_PASSED).astype(int)
    f["_rej"] = stage.eq(STAGE_NE_GATE).astype(int)
    f["_trial"] = stage.eq(STAGE_TRIAL).astype(int)
    f["_gate_reason"] = (etype + ":" + f["gate_reason"].astype(str)).where(stage.eq(STAGE_NE_GATE))
    f["_trial_reason"] = f["gate_reason"].astype(str).where(stage.eq(STAGE_TRIAL))
    f["_quote"] = "[" + etype + "|" + stage + "] " + f["quote"].fillna("").astype(str)
    uniq = lambda s: ";".join(sorted(set(s.dropna())))  # noqa: E731
    out = f.groupby(keys, sort=False).agg(
        n_findings=("_pass", "size"), n_gated=("_pass", "sum"), n_ne_gate_rejected=("_rej", "sum"),
        n_trial_rejected=("_trial", "sum"), evidence_types=("evidence_type", uniq),
        gate_reasons=("_gate_reason", uniq), trial_reasons=("_trial_reason", uniq),
        quotes=("_quote", " || ".join))
    return out.reset_index().reindex(columns=cols)


def classify_notes(ihc_notes: pd.DataFrame, findings: pd.DataFrame, llm_notes: pd.DataFrame,
                   pats: il.IHCPatterns, *, ok_statuses) -> pd.DataFrame:
    """Qualifying IHC notes with their LLM layers and ``review_group`` (None = nothing to review)."""
    keys = [LOCAL_ID_COL, "RPT_ID"]
    q = ihc_notes[ihc_notes["qualifies"].astype(bool)].copy()
    q["RPT_ID"] = q["RPT_ID"].astype(str)
    out = q.merge(_finding_summary(findings), on=keys, how="left", validate="one_to_one")
    ne_rx = pats.markers[il.NE_PANEL][il.NE_PANEL]
    llm = llm_notes.assign(ne_marker_in_snippet=llm_notes["snippet"].fillna("").map(
        lambda s: bool(ne_rx.search(s)))).drop(columns="snippet")
    out = out.merge(llm, on=keys, how="left", validate="one_to_one")
    for c in ("n_findings", "n_gated", "n_ne_gate_rejected", "n_trial_rejected"):
        out[c] = out[c].fillna(0).astype(int)
    for c in ("evidence_types", "gate_reasons", "trial_reasons", "quotes"):
        out[c] = out[c].fillna("")
    out["in_evidence"] = out["in_evidence"].astype("boolean").fillna(False).astype(bool)
    out["llm_status"] = out["llm_status"].where(out["in_evidence"], "not_in_evidence")
    out["ne_marker_in_snippet"] = out["ne_marker_in_snippet"].astype("boolean")
    out["ne_markers"] = out["ihc_markers"].fillna("").map(
        lambda s: ";".join(m for m in s.split(";") if m in dict(il.NE_MARKER_LABELS)))
    has_ne_marker = out["ihc_panels"].fillna("").str.split(";").map(lambda p: il.NE_PANEL in p)

    group = pd.Series(None, index=out.index, dtype=object)
    screened = out["n_gated"] + out["n_ne_gate_rejected"]
    group[out["n_gated"].gt(0)] = "gated"
    group[group.isna() & screened.gt(0)] = "gate_rejected"
    rest = group.isna() & has_ne_marker
    group[rest & out["n_trial_rejected"].gt(0)] = "trial_context_only"
    rest = group.isna() & has_ne_marker
    group[rest & ~out["in_evidence"]] = "not_in_evidence"
    group[rest & out["in_evidence"] & ~out["llm_status"].isin(ok_statuses)] = "not_run_or_failed"
    group[group.isna() & has_ne_marker] = "no_finding"
    out["review_group"] = group
    return out


def group_counts(classified: pd.DataFrame) -> pd.DataFrame:
    order = [*REVIEW_GROUPS, *UNREVIEWED_GROUPS]
    c = classified.dropna(subset=["review_group"])
    t = c.groupby("review_group").agg(notes=("RPT_ID", "size"), patients=(LOCAL_ID_COL, "nunique"))
    return t.reindex(order).fillna(0).astype(int).reset_index()


def reason_table(findings: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame:
    """Findings on qualifying IHC notes: evidence type x gate step:reason code (cue dropped)."""
    keys = [LOCAL_ID_COL, "RPT_ID"]
    f = findings.merge(classified[keys], on=keys, how="inner")
    if f.empty:
        return pd.DataFrame()
    code = f["gate_reason"].fillna("").str.split(":").str[0]
    f = f.assign(step=np.where(f["gate_stage"].eq(STAGE_PASSED), STAGE_PASSED,
                               f["gate_stage"] + ":" + code))
    return pd.crosstab(f["step"], f["evidence_type"], margins=True, margins_name="total")


# --------------------------------------------------------------------------- #
# Review sheet
# --------------------------------------------------------------------------- #
def review_pattern(ne_gate) -> re.Pattern:
    """NE terms (ne_gate.NE_TERM) or NE markers (ne_gate.NE_MARKER): the text windows' anchors."""
    return re.compile(f"(?:{ne_gate.NE_TERM.pattern})|(?:{ne_gate.NE_MARKER.pattern})", re.IGNORECASE)


def text_window(text: str, rx: re.Pattern, *, window: int = DEFAULT_WINDOW_CHARS,
                max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    """Report text within ``window`` characters of each ``rx`` match, merged, whitespace flattened."""
    text = text or ""
    spans = []
    for m in rx.finditer(text):
        lo, hi = max(0, m.start() - window), min(len(text), m.end() + window)
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    if not spans:
        spans = [[0, min(len(text), 2 * window)]]
    parts = [("…" if lo > 0 else "") + text[lo:hi] + ("…" if hi < len(text) else "") for lo, hi in spans]
    out = re.sub(r"\s+", " ", " | ".join(parts)).strip()
    return out if len(out) <= max_chars else out[:max_chars] + " [truncated]"


def in_window_columns(landmarks) -> list:
    return [f"in_window_{int(lm)}" for lm in landmarks]


def sheet_columns(landmarks) -> list:
    return ["review_id", "review_group", LOCAL_ID_COL, "RPT_ID", "note_date", "days_from_adt",
            *in_window_columns(landmarks), "ihc_markers", "ne_markers", "llm_status",
            "ne_marker_in_snippet", "evidence_types", "gate_reasons", "trial_reasons", "quotes",
            "PROC_DESC", "text_window", "reviewer_call", "reviewer_note"]


def build_review_sheet(classified: pd.DataFrame, notes_text: pd.DataFrame, scores: pd.DataFrame,
                       anchors: pd.Series, landmarks, rx: re.Pattern, *,
                       window: int = DEFAULT_WINDOW_CHARS, max_chars: int = DEFAULT_MAX_TEXT_CHARS,
                       max_days: int = il.MAX_FOLLOWUP_DAYS) -> pd.DataFrame:
    """One row per reviewable note (``REVIEW_GROUPS``) with a text window and blank reviewer columns.

    ``in_window_<lm>`` = the patient is scored at that landmark and the note falls in
    nb11's future window (ADT + lm < note date <= ADT + max_days).
    """
    keys = [LOCAL_ID_COL, "RPT_ID"]
    s = classified[classified["review_group"].isin(REVIEW_GROUPS)].copy()
    s = il.add_days_from_adt(s, anchors)
    text = notes_text[[*keys, "PROC_DESC", "RPT_TEXT"]].assign(RPT_ID=lambda d: d["RPT_ID"].astype(str))
    s = s.merge(text, on=keys, how="left", validate="one_to_one")
    s["text_window"] = s["RPT_TEXT"].fillna("").map(
        lambda t: text_window(t, rx, window=window, max_chars=max_chars))
    s["PROC_DESC"] = s["PROC_DESC"].fillna("").map(lambda t: re.sub(r"\s+", " ", t).strip())
    s["quotes"] = s["quotes"].map(lambda t: re.sub(r"\s+", " ", t).strip())
    for lm, col in zip(landmarks, in_window_columns(landmarks)):
        at_lm = set(scores.loc[scores["landmark_days"].eq(int(lm)), ID_COL].astype(int))
        d = s["days_from_adt"]
        s[col] = (s[LOCAL_ID_COL].isin(at_lm) & d.gt(int(lm)) & d.le(max_days)).astype(int)
    s["review_id"] = s[LOCAL_ID_COL].astype(str) + ":" + s["RPT_ID"].astype(str)
    s["note_date"] = pd.to_datetime(s["note_date"]).dt.strftime("%Y-%m-%d")
    s["reviewer_call"], s["reviewer_note"] = "", ""
    order = {g: i for i, g in enumerate(REVIEW_GROUPS)}
    s = s.sort_values(["review_group", LOCAL_ID_COL, "note_date"],
                      key=lambda c: c.map(order) if c.name == "review_group" else c)
    return s.reindex(columns=sheet_columns(landmarks)).reset_index(drop=True)


def write_review_sheet(sheet: pd.DataFrame, path) -> Path:
    """Write the blank sheet; refuse to overwrite a sheet that already holds reviewer calls."""
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if "reviewer_call" in old and old["reviewer_call"].str.strip().ne("").any():
            raise FileExistsError(f"{path} already has reviewer calls. Save reviewed copies under "
                                  "the reviewed-sheet name, or move this file before regenerating.")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# --------------------------------------------------------------------------- #
# Read-back
# --------------------------------------------------------------------------- #
def read_reviewed(path, sheet: pd.DataFrame):
    """Join a filled sheet's calls onto the current sheet -> (reviewed sheet, stale review_ids)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = {"review_id", "reviewer_call"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    df["reviewer_call"] = df["reviewer_call"].str.strip().str.lower()
    bad = sorted(set(df["reviewer_call"]) - {"", *REVIEWER_CALLS})
    if bad:
        raise ValueError(f"Unknown reviewer_call values {bad}; use one of {list(REVIEWER_CALLS)} or blank.")
    if df["review_id"].duplicated().any():
        raise ValueError("Duplicate review_id rows in the reviewed sheet.")
    if "reviewer_note" not in df:
        df["reviewer_note"] = ""
    stale = sorted(set(df["review_id"]) - set(sheet["review_id"]))
    out = (sheet.drop(columns=["reviewer_call", "reviewer_note"])
                .merge(df[["review_id", "reviewer_call", "reviewer_note"]], on="review_id", how="left"))
    out[["reviewer_call", "reviewer_note"]] = out[["reviewer_call", "reviewer_note"]].fillna("")
    return out, stale


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return math.nan, math.nan
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return centre - half, centre + half


def _call_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=False):
        calls = g["reviewer_call"]
        t, f = int(calls.eq("true_ne").sum()), int(calls.eq("not_ne").sum())
        lo, hi = wilson(t, t + f)
        rows.append({by: key, "notes": len(g), "reviewed": int(calls.ne("").sum()), "true_ne": t,
                     "not_ne": f, "unclear": int(calls.eq("unclear").sum()),
                     "frac_true_ne": t / (t + f) if t + f else math.nan,
                     "frac_true_ne_lo": lo, "frac_true_ne_hi": hi})
    return pd.DataFrame(rows, columns=[by, *SUMMARY_COLUMNS[1:]])


def summarize_review(reviewed: pd.DataFrame) -> pd.DataFrame:
    """Per review group: calls and the true-NE fraction among decided calls (Wilson 95% CI).

    For ``gated`` that fraction is the gate's precision; for every other group it is
    the fraction of nb11-negative notes that were really NE-positive.
    """
    order = {g: i for i, g in enumerate(REVIEW_GROUPS)}
    t = _call_table(reviewed, "review_group")
    return t.sort_values("review_group", key=lambda c: c.map(order)).reset_index(drop=True)


def summarize_by_reason(reviewed: pd.DataFrame) -> pd.DataFrame:
    """``gate_rejected`` notes by ne_gate reason code (a note with several codes counts under each)."""
    g = reviewed[reviewed["review_group"].eq("gate_rejected")].copy()
    if g.empty:
        return pd.DataFrame(columns=["reason", *SUMMARY_COLUMNS[1:]])
    g["reason"] = g["gate_reasons"].map(
        lambda s: sorted({p.split(":")[1] for p in s.split(";") if p.count(":") >= 1}) or ["(none)"])
    t = _call_table(g.explode("reason"), "reason")
    return t.sort_values("notes", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Definitions -> nb11 event counts (no refit)
# --------------------------------------------------------------------------- #
def ne_note_keys(findings: pd.DataFrame, stages) -> pd.DataFrame:
    """Reports with >=1 finding in ``stages``, shaped like ``load_gated_ne_notes``' first output."""
    f = findings[findings["gate_stage"].isin(stages)]
    cols = [LOCAL_ID_COL, "RPT_ID", "gated_evidence_types"]
    if f.empty:
        return pd.DataFrame(columns=cols)
    return (f.groupby([LOCAL_ID_COL, "RPT_ID"])["evidence_type"]
              .agg(lambda s: ";".join(sorted(set(s)))).rename("gated_evidence_types").reset_index())


def definition_keys(findings: pd.DataFrame, reviewed: pd.DataFrame | None = None) -> dict:
    """``gated`` (nb11), ``ungated`` (ne_gate removed) and, given calls, ``reviewed``.

    ``reviewed`` = gated reports not called ``not_ne`` plus any reviewed note called
    ``true_ne``; blank and ``unclear`` calls keep the gated status.
    """
    keys = [LOCAL_ID_COL, "RPT_ID"]
    out = {"gated": ne_note_keys(findings, {STAGE_PASSED}),
           "ungated": ne_note_keys(findings, {STAGE_PASSED, STAGE_NE_GATE})}
    if reviewed is not None:
        calls = reviewed[keys + ["reviewer_call"]].astype({LOCAL_ID_COL: int, "RPT_ID": str})
        drop = calls[calls["reviewer_call"].eq("not_ne")][keys]
        add = calls[calls["reviewer_call"].eq("true_ne")][keys]
        g = out["gated"].astype({LOCAL_ID_COL: int, "RPT_ID": str})
        g = g.merge(drop.assign(_drop=True), on=keys, how="left")
        g = g[g["_drop"].isna()].drop(columns="_drop")
        add = add.merge(g[keys], on=keys, how="left", indicator=True)
        add = add[add["_merge"].eq("left_only")][keys].assign(gated_evidence_types="reviewer")
        out["reviewed"] = pd.concat([g, add], ignore_index=True)
    return out


def event_counts(scores, ihc_notes, definitions: dict, review_status: dict, anchors, landmarks,
                 *, outcomes=("ne_pos_all", "ne_pos_given_ihc")) -> pd.DataFrame:
    """n and events of nb11's NE outcomes per definition x landmark (cohort ``all``; no fitting)."""
    rows = []
    for name, keys in definitions.items():
        merged = il.merge_ne_positive(ihc_notes, keys, review_status)
        frame = il.build_ihc_frame(scores, merged, landmarks, anchors=anchors, review_status=review_status)
        for lm in landmarks:
            sub = frame[frame["landmark_days"].eq(int(lm))]
            for outcome in outcomes:
                pop = il.outcome_population(sub, outcome)
                rows.append({"definition": name, "landmark_days": int(lm), "outcome": outcome,
                             "n": len(pop), "events": int(pop[il.OUTCOMES[outcome]].sum())})
    return pd.DataFrame(rows, columns=["definition", "landmark_days", "outcome", "n", "events"])


# --------------------------------------------------------------------------- #
# Patient-level LLM NEPC labels re-compiled without the gate vs manual review
# --------------------------------------------------------------------------- #
# Each definition = the gate stages whose findings still make a patient "nepc".
LABEL_DEFINITIONS = {
    "gated": (STAGE_PASSED,),                                   # LCA nepc_labels (build_labels)
    "no_ne_gate": (STAGE_PASSED, STAGE_NE_GATE),                # ne_gate removed, trial screen kept
    "no_gate": (STAGE_PASSED, STAGE_NE_GATE, STAGE_TRIAL),      # every final-attempt finding counts
}
MANUAL_LABEL_COL = "updated_manual_low_confidence"              # Figure 1's chart-review column
MANUAL_LABEL_LEVELS = ("conventional", "avpc", "nepc")
LLM_LABEL_LEVELS = ("nepc", "non_nepc", "unlabeled")
METRIC_COLUMNS = ["definition", "scope", "n_annotated", "n_unlabeled", "n", "tp", "fp", "fn", "tn",
                  "sensitivity", "sensitivity_lo", "sensitivity_hi", "specificity", "specificity_lo",
                  "specificity_hi", "ppv", "ppv_lo", "ppv_hi", "npv", "npv_lo", "npv_hi"]


def _gate_stage(trial, quote, etype, screen):
    if isinstance(trial, str) and trial:
        return STAGE_TRIAL
    return STAGE_PASSED if screen(quote, etype)[0] else STAGE_NE_GATE


def patient_label_definitions(lca, inputs) -> pd.DataFrame:
    """One row per LLM_nepc cohort patient: its nepc_status under each of ``LABEL_DEFINITIONS``.

    Uses every final-attempt finding (all note types, not only pathology), with
    ``build_labels``' rule: >=1 counted finding -> ``nepc``; otherwise ``unlabeled``
    when review status is in ``UNLABELED_STATUSES`` (failed / not_run), else
    ``non_nepc``. Raises unless ``gated`` matches ``gate_findings`` patient for patient.
    """
    f = _to_pandas(inputs["findings"])
    if "_trial_context_reason" not in f.columns:
        raise ValueError("Findings lack _trial_context_reason; load them with load_label_inputs.")
    f = f[f[LOCAL_ID_COL].notna()].copy()
    f[LOCAL_ID_COL] = f[LOCAL_ID_COL].astype(int)
    screen = lca["nepc"].screen_ne_quote
    f["gate_stage"] = [_gate_stage(t, q, e, screen) for t, q, e in
                       zip(f["_trial_context_reason"], f["quote"], f["evidence_type"])]

    gated, _ = lca["nepc"].gate_findings(inputs["findings"])
    theirs = {int(m) for m, rows in gated.items() if rows}
    mine = set(f.loc[f["gate_stage"].eq(STAGE_PASSED), LOCAL_ID_COL])
    if mine != theirs:
        raise RuntimeError(f"Re-run gate disagrees with gate_findings on {len(mine ^ theirs)} "
                           "patients; the LCA gate logic has changed.")

    unlabeled = set(lca["nepc"].UNLABELED_STATUSES)
    status = {int(m): s for m, s in inputs["status"].items()}
    counts = f.groupby([LOCAL_ID_COL, "gate_stage"]).size().unstack(fill_value=0)
    out = pd.DataFrame({LOCAL_ID_COL: sorted(status)})
    out["review_status"] = out[LOCAL_ID_COL].map(status)
    for stage in (STAGE_PASSED, STAGE_NE_GATE, STAGE_TRIAL):
        col = counts[stage] if stage in counts else pd.Series(dtype=int)
        out[f"n_findings_{stage}"] = out[LOCAL_ID_COL].map(col).fillna(0).astype(int)
    for name, stages in LABEL_DEFINITIONS.items():
        pos = sum(out[f"n_findings_{s}"] for s in stages).gt(0)
        out[f"nepc_status_{name}"] = np.where(pos, "nepc", np.where(
            out["review_status"].isin(unlabeled), "unlabeled", "non_nepc"))
    return out


def stored_label_drift(labels: pd.DataFrame, stored) -> int:
    """Patients whose stored ``nepc_labels.parquet`` status differs from the re-compiled ``gated`` one."""
    s = _to_pandas(stored)[[LOCAL_ID_COL, "nepc_status"]].copy()
    s[LOCAL_ID_COL] = s[LOCAL_ID_COL].astype(int)
    s["nepc_status"] = s["nepc_status"].fillna("unlabeled")
    m = labels[[LOCAL_ID_COL, "nepc_status_gated"]].merge(s, on=LOCAL_ID_COL, how="outer")
    return int(m["nepc_status_gated"].ne(m["nepc_status"]).sum())


def load_manual_annotations(path) -> pd.DataFrame:
    """Chart-review truth (one row per patient, ``manual_label`` in ``MANUAL_LABEL_LEVELS``, blanks dropped)."""
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {LOCAL_ID_COL, MANUAL_LABEL_COL} - set(m.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    m = pd.DataFrame({
        LOCAL_ID_COL: pd.to_numeric(m[LOCAL_ID_COL].str.strip().str.replace(r"\.0$", "", regex=True),
                                    errors="coerce"),
        "manual_label": m[MANUAL_LABEL_COL].str.strip().str.lower()})
    m = m[~m["manual_label"].isin(["", "nan", "na", "null", "none"])]
    bad = sorted(set(m["manual_label"]) - set(MANUAL_LABEL_LEVELS))
    if bad:
        raise ValueError(f"Unexpected {MANUAL_LABEL_COL} values {bad}; expected {list(MANUAL_LABEL_LEVELS)}.")
    if m[LOCAL_ID_COL].isna().any() or m[LOCAL_ID_COL].duplicated().any():
        raise ValueError(f"{path} must have one row per non-missing {LOCAL_ID_COL}.")
    return m.astype({LOCAL_ID_COL: int}).reset_index(drop=True)


def join_manual(labels: pd.DataFrame, manual: pd.DataFrame, scored_ids=None) -> pd.DataFrame:
    """Annotated patients with their LLM labels; annotated patients outside the LLM cohort -> ``unlabeled``."""
    j = manual.merge(labels, on=LOCAL_ID_COL, how="left", validate="one_to_one")
    j["in_llm_cohort"] = j["review_status"].notna()
    for name in LABEL_DEFINITIONS:
        j[f"nepc_status_{name}"] = j[f"nepc_status_{name}"].fillna("unlabeled")
    if scored_ids is not None:
        j["scored"] = j[LOCAL_ID_COL].isin({int(m) for m in scored_ids})
    return j


def _scopes(joined):
    yield "annotated", joined
    if "scored" in joined:
        yield "annotated_scored", joined[joined["scored"]]


def manual_confusion(joined: pd.DataFrame) -> pd.DataFrame:
    """Long confusion table: definition x scope x manual_label (rows) x LLM label (cols), counts."""
    rows = []
    for scope, d in _scopes(joined):
        for name in LABEL_DEFINITIONS:
            ct = pd.crosstab(d["manual_label"], d[f"nepc_status_{name}"]).reindex(
                index=list(MANUAL_LABEL_LEVELS), columns=list(LLM_LABEL_LEVELS), fill_value=0)
            for man, row in ct.iterrows():
                rows.append({"definition": name, "scope": scope, "manual_label": man, **row.to_dict()})
    return pd.DataFrame(rows, columns=["definition", "scope", "manual_label", *LLM_LABEL_LEVELS])


def manual_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    """Manual nepc vs (conventional + avpc) against each LLM label; unlabeled patients left out. Wilson CIs."""
    rows = []
    for scope, d in _scopes(joined):
        for name in LABEL_DEFINITIONS:
            lab = d[f"nepc_status_{name}"]
            k = d[lab.ne("unlabeled")]
            truth, pred = k["manual_label"].eq("nepc"), k[f"nepc_status_{name}"].eq("nepc")
            tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
            fn, tn = int((truth & ~pred).sum()), int((~truth & ~pred).sum())
            r = {"definition": name, "scope": scope, "n_annotated": len(d),
                 "n_unlabeled": int(lab.eq("unlabeled").sum()), "n": len(k),
                 "tp": tp, "fp": fp, "fn": fn, "tn": tn}
            for metric, (a, b) in {"sensitivity": (tp, fn), "specificity": (tn, fp),
                                   "ppv": (tp, fp), "npv": (tn, fn)}.items():
                r[metric] = a / (a + b) if a + b else math.nan
                r[f"{metric}_lo"], r[f"{metric}_hi"] = wilson(a, a + b)
            rows.append(r)
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def label_flips(joined: pd.DataFrame) -> pd.DataFrame:
    """Annotated patients whose LLM label changes when the gate is removed, with the manual call."""
    cols = [f"nepc_status_{n}" for n in LABEL_DEFINITIONS]
    changed = joined[cols].nunique(axis=1).gt(1)
    keep = [LOCAL_ID_COL, "manual_label", "review_status", *cols,
            *[c for c in joined.columns if c.startswith("n_findings_")],
            *(["scored"] if "scored" in joined else [])]
    return joined.loc[changed, keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Patient-level review sheet for gate flips (gated non_nepc -> nepc ungated)
# --------------------------------------------------------------------------- #
FLIP_TYPES = ("ne_gate", "trial_context")   # the screen whose removal makes the patient nepc
FLIP_SHEET_COLUMNS = ["review_id", "flip_type", LOCAL_ID_COL, "review_status", "scored", "manual_label",
                      "n_findings", "n_notes", "note_types", "evidence_types", "reason_codes",
                      "earliest_event_date", "findings", "context", "reviewer_call", "reviewer_note"]


def _reason_code(stage, reason):
    code = str(reason or "").split(":")[0] or "(none)"
    return code if stage == STAGE_NE_GATE else f"trial:{code}"


def flipped_patients(labels: pd.DataFrame) -> pd.DataFrame:
    """Patients ``non_nepc`` when gated but ``nepc`` with no gate, with the screen that flips them."""
    f = labels[labels["nepc_status_gated"].ne("nepc") & labels["nepc_status_no_gate"].eq("nepc")].copy()
    f["flip_type"] = np.where(f["nepc_status_no_ne_gate"].eq("nepc"), "ne_gate", "trial_context")
    return f.reset_index(drop=True)


def flip_findings(lca, inputs, mrns) -> pd.DataFrame:
    """Every final-attempt finding (all note types) of ``mrns`` with its gate step and reason."""
    selected = {int(m) for m in mrns}
    f = _to_pandas(inputs["findings"])
    f = f[f[LOCAL_ID_COL].notna()].copy()
    f[LOCAL_ID_COL] = f[LOCAL_ID_COL].astype(int)
    f = f[f[LOCAL_ID_COL].isin(selected)].copy()
    screen = lca["nepc"].screen_ne_quote
    stages, reasons = [], []
    for trial, quote, etype in zip(f["_trial_context_reason"], f["quote"], f["evidence_type"]):
        if isinstance(trial, str) and trial:
            stages.append(STAGE_TRIAL)
            reasons.append(trial)
            continue
        ok, reason = screen(quote, etype)
        stages.append(STAGE_PASSED if ok else STAGE_NE_GATE)
        reasons.append(None if ok else reason)
    f["gate_stage"], f["gate_reason"] = stages, reasons
    if f["gate_stage"].eq(STAGE_PASSED).any():
        raise RuntimeError("A flipped patient has a finding that passes the gate; the labels are stale.")
    return f.reset_index(drop=True)


def quote_window(snippet: str, quote: str, fallback: re.Pattern, *, window: int = DEFAULT_WINDOW_CHARS) -> str:
    """Snippet text within ``window`` characters of the quote (whitespace-insensitive), else of NE terms."""
    snippet, words = snippet or "", (quote or "").split()
    m = re.search(r"\s+".join(map(re.escape, words)), snippet, re.IGNORECASE) if words else None
    if m is None:
        return "[quote not found in snippet] " + text_window(snippet, fallback, window=window,
                                                             max_chars=2 * DEFAULT_MAX_TEXT_CHARS)
    lo, hi = max(0, m.start() - window), min(len(snippet), m.end() + window)
    body = snippet[lo:m.start()] + ">>" + snippet[m.start():m.end()] + "<<" + snippet[m.end():hi]
    return re.sub(r"\s+", " ", ("…" if lo else "") + body + ("…" if hi < len(snippet) else "")).strip()


def build_flip_sheet(flips: pd.DataFrame, findings: pd.DataFrame, notes: dict, rx: re.Pattern, *,
                     manual: pd.DataFrame | None = None, scored_ids=None,
                     window: int = DEFAULT_WINDOW_CHARS, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> pd.DataFrame:
    """One row per flipped patient: every rejected finding, its step:reason, and the snippet around its quote.

    ``findings`` entries are numbered ``[k]`` and the matching ``context`` entry carries the
    same number; the quote is marked ``>>…<<``. ``context`` is the evidence snippet the
    LLM read, not the full note.
    """
    man = {} if manual is None else dict(zip(manual[LOCAL_ID_COL].astype(int), manual["manual_label"]))
    scored = None if scored_ids is None else {int(m) for m in scored_ids}
    by_mrn = {int(m): g for m, g in findings.groupby(LOCAL_ID_COL)}
    rows = []
    for p in flips.itertuples(index=False):
        mrn = int(getattr(p, LOCAL_ID_COL))
        g = by_mrn.get(mrn, findings.iloc[0:0])
        g = g[g["gate_stage"].ne(STAGE_PASSED)].sort_values(["source_note_date", "note_uid", "finding_index"],
                                                            na_position="last")
        f_parts, c_parts = [], []
        for k, r in enumerate(g.itertuples(index=False), start=1):
            head = f"[{k}] {r.note_type} {r.source_note_date or '?'} | {r.evidence_type} | " \
                   f"{r.gate_stage}:{r.gate_reason or ''}"
            f_parts.append(f"{head} | event {r.event_date or '?'} | {re.sub(chr(10), ' ', str(r.quote or ''))}")
            snippet = (notes.get((mrn, r.note_uid)) or {}).get("snippet", "")
            c_parts.append(f"[{k}] " + quote_window(snippet, r.quote, rx, window=window))
        context = " || ".join(c_parts)
        if len(context) > max_chars:
            context = context[:max_chars] + " [truncated]"
        uniq = lambda s: ";".join(sorted({str(x) for x in s if pd.notna(x) and str(x)}))  # noqa: E731
        dates = pd.to_datetime(g["event_date"], errors="coerce").dropna()
        rows.append({
            "review_id": str(mrn), "flip_type": p.flip_type, LOCAL_ID_COL: mrn,
            "review_status": p.review_status,
            "scored": "" if scored is None else int(mrn in scored), "manual_label": man.get(mrn, ""),
            "n_findings": len(g), "n_notes": g["note_uid"].nunique(), "note_types": uniq(g["note_type"]),
            "evidence_types": uniq(g["evidence_type"]),
            "reason_codes": uniq(_reason_code(s, r) for s, r in zip(g["gate_stage"], g["gate_reason"])),
            "earliest_event_date": dates.min().strftime("%Y-%m-%d") if len(dates) else "",
            "findings": " || ".join(f_parts), "context": context, "reviewer_call": "", "reviewer_note": ""})
    order = {t: i for i, t in enumerate(FLIP_TYPES)}
    out = pd.DataFrame(rows, columns=FLIP_SHEET_COLUMNS)
    return out.sort_values(["flip_type", LOCAL_ID_COL], key=lambda c: c.map(order) if c.name == "flip_type" else c
                           ).reset_index(drop=True)


def summarize_flip_review(reviewed: pd.DataFrame) -> dict:
    """Calls by flip type, by reason code (a patient with several codes counts under each) and overall."""
    order = {t: i for i, t in enumerate(FLIP_TYPES)}
    by_type = _call_table(reviewed, "flip_type").sort_values(
        "flip_type", key=lambda c: c.map(order)).reset_index(drop=True)
    r = reviewed.assign(reason=reviewed["reason_codes"].map(lambda s: s.split(";") if s else ["(none)"]))
    by_reason = _call_table(r.explode("reason"), "reason").sort_values("notes", ascending=False)
    overall = _call_table(reviewed.assign(all="all"), "all")
    rename = {"notes": "patients"}
    return {"by_flip_type": by_type.rename(columns=rename), "by_reason": by_reason.reset_index(drop=True)
            .rename(columns=rename), "overall": overall.rename(columns=rename)}


def flip_label_counts(labels: pd.DataFrame, reviewed: pd.DataFrame) -> pd.DataFrame:
    """Cohort NEPC counts: gated, no_ne_gate, no_gate, and ``reviewed`` = gated + flips called ``true_ne``.

    ``reviewed_with_unclear`` also adds the ``unclear`` calls (an upper bound).
    """
    calls = reviewed["reviewer_call"]
    gated = int(labels["nepc_status_gated"].eq("nepc").sum())
    counts = {name: int(labels[f"nepc_status_{name}"].eq("nepc").sum()) for name in LABEL_DEFINITIONS}
    counts["reviewed"] = gated + int(calls.eq("true_ne").sum())
    counts["reviewed_with_unclear"] = counts["reviewed"] + int(calls.eq("unclear").sum())
    counts["not_yet_reviewed"] = int(calls.eq("").sum())
    return pd.DataFrame([{"definition": k, "nepc_patients": v} for k, v in counts.items()])
