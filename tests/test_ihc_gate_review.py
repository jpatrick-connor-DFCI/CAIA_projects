import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.multivariable_models import ihc_gate_review as gr  # noqa: E402
from COMPASS.multivariable_models import ihc_lr as il  # noqa: E402

NE_MARKER = re.compile(r"\b(?:synaptophysin|chromogranin(?:\s+a)?|cd\s?56|insm-?1|"
                       r"neuron[\s-]+specific\s+enolase|nse)\b", re.I)
NE_TERM = re.compile(r"\b(?:neuro[\s-]?endocrine|small[\s-]?cell)\b", re.I)
PATS = il.build_patterns(NE_MARKER.pattern)
OK = {"ok", "ok_with_rejections"}
PROSTATE = "Prostate, needle biopsy"

# (mrn, rpt, date, proc, text)
NOTES = [
    (1, "10", "2021-01-01", PROSTATE, "Adenocarcinoma. Synaptophysin positive in tumor cells."),
    (2, "20", "2021-01-01", PROSTATE, "Adenocarcinoma. Chromogranin stain performed, see below."),
    (2, "21", "2020-02-01", PROSTATE, "Adenocarcinoma. CD56 immunostain."),
    (3, "30", "2021-01-01", PROSTATE, "Adenocarcinoma. INSM1 immunostain."),
    (3, "31", "2021-01-01", PROSTATE, "Adenocarcinoma. Synaptophysin immunostain."),
    (3, "32", "2021-01-01", PROSTATE, "Adenocarcinoma. p63 immunostain shows loss of basal cells."),
    (4, "40", "2021-01-01", "Lung, wedge", "Small cell carcinoma of lung. Synaptophysin positive."),
    (5, "50", "2021-01-01", PROSTATE, "Adenocarcinoma. Synaptophysin negative."),
]
# (mrn, rpt, finding_index, evidence_type, quote, trial_reason)
FINDINGS = [
    (1, "10", 0, "positive_ne_ihc", "Synaptophysin positive in tumor cells", None),
    (5, "50", 0, "positive_ne_ihc", "Synaptophysin negative", None),
    (3, "31", 0, "positive_ne_ihc", "Synaptophysin positive", "boilerplate:eligibility"),
    (4, "40", 0, "nepc_diagnosis", "Small cell carcinoma positive", None),
]
EVIDENCE = {(1, "10"): "ok", (5, "50"): "ok", (3, "31"): "ok", (4, "40"): "ok",
            (2, "20"): "ok", (2, "21"): "failed"}


def _screen(quote, etype):
    return ("negative" not in quote, None if "negative" not in quote else "negative_ihc")


def _gate(findings):
    gated = {}
    for r in findings.to_dict("records"):
        ok = not r["_trial_context_reason"] and _screen(r["quote"], r["evidence_type"])[0]
        if ok:
            gated.setdefault(int(r["DFCI_MRN"]), []).append(r)
    return gated, {}


def _lca(findings=None, gate=_gate, notes=None):
    findings = findings if findings is not None else pd.DataFrame([
        {"DFCI_MRN": m, "note_uid": f"Pathology:{r}", "finding_index": i, "attempt_id": "a",
         "evidence_type": t, "specimen_site": None, "confidence": "high", "quote": q,
         "_trial_context_reason": tr} for m, r, i, t, q, tr in FINDINGS])
    notes = notes if notes is not None else {
        (m, f"Pathology:{r}"): {"snippet": next(t for mm, rr, _, _, t in NOTES if (mm, rr) == (m, r))}
        for m, r in EVIDENCE}
    ledger = pd.DataFrame([{"DFCI_MRN": m, "note_uid": f"Pathology:{r}", "status": s}
                           for (m, r), s in EVIDENCE.items() if s != "missing"])

    class NoteExtraction:
        LEDGER_SCHEMA, OK_STATUSES = {}, OK
        load_label_inputs = staticmethod(lambda spec, d: {
            "findings": findings, "notes": notes, "status": {m: "ok" for m in range(1, 6)},
            "paths": {"processed": "P"}})
        _read = staticmethod(lambda path, schema: ledger)
        _final_attempts = staticmethod(lambda led: led)

    class Nepc:
        SPEC = "SPEC"
        UNLABELED_STATUSES = {"failed", "not_run"}
        screen_ne_quote = staticmethod(_screen)
        gate_findings = staticmethod(gate)

    class NeGate:
        NE_MARKER, NE_TERM = NE_MARKER, NE_TERM

    return {"note_extraction": NoteExtraction, "nepc": Nepc, "ne_gate": NeGate}


def _notes_text():
    return pd.DataFrame([{"DFCI_MRN": m, "RPT_ID": r, "note_date": pd.Timestamp(d), "PROC_DESC": p,
                          "RPT_TEXT": t} for m, r, d, p, t in NOTES])


def _classified(lca=None):
    lca = lca or _lca()
    inputs, ledger = gr.load_llm_inputs(lca, "dir")
    findings = gr.finding_audit(lca, inputs)
    llm = gr.llm_note_table(inputs, ledger)
    ihc = il.tag_ihc_notes(_notes_text(), PATS)
    return findings, gr.classify_notes(ihc, findings, llm, PATS, ok_statuses=OK), ihc, inputs


def test_finding_audit_records_step_and_reason():
    findings, *_ = _classified()
    f = findings.set_index("RPT_ID")
    assert f.loc["10", "gate_stage"] == "passed" and pd.isna(f.loc["10", "gate_reason"])
    assert (f.loc["50", "gate_stage"], f.loc["50", "gate_reason"]) == ("ne_gate", "negative_ihc")
    assert (f.loc["31", "gate_stage"], f.loc["31", "gate_reason"]) == ("trial_context", "boilerplate:eligibility")


def test_finding_audit_requires_parity_with_gate_findings():
    def extra(findings):
        g, r = _gate(findings)
        g.setdefault(5, []).append(findings.iloc[1].to_dict())
        return g, r
    lca = _lca(gate=extra)
    inputs, _ = gr.load_llm_inputs(lca, "dir")
    with pytest.raises(RuntimeError, match="disagrees"):
        gr.finding_audit(lca, inputs)


def test_collision_uid_is_rejected():
    lca = _lca(notes={(1, "Pathology:10#2"): {"snippet": "x"}})
    inputs, ledger = gr.load_llm_inputs(lca, "dir")
    with pytest.raises(ValueError, match="Ambiguous"):
        gr.llm_note_table(inputs, ledger)


def test_review_groups():
    _, c, *_ = _classified()
    groups = dict(zip(c["RPT_ID"], c["review_group"].where(c["review_group"].notna(), None)))
    assert groups == {"10": "gated", "50": "gate_rejected", "20": "no_finding",
                      "21": "not_run_or_failed", "30": "not_in_evidence",
                      "31": "trial_context_only", "32": None}          # 40 fails the specimen rule
    row = c.set_index("RPT_ID")
    assert row.loc["20", "ne_marker_in_snippet"] and row.loc["30", "llm_status"] == "not_in_evidence"
    assert row.loc["50", "gate_reasons"] == "positive_ne_ihc:negative_ihc"
    assert row.loc["10", "ne_markers"] == "synaptophysin"
    counts = gr.group_counts(c).set_index("review_group")["notes"]
    assert counts.to_dict() == {g: 1 for g in [*gr.REVIEW_GROUPS, *gr.UNREVIEWED_GROUPS]}


def _scores():
    return pd.DataFrame([{"person_id": m, "landmark_days": lm, "risk_score": float(m)}
                         for m in range(1, 6) for lm in (0, 90)])


ANCHORS = pd.Series(pd.Timestamp("2020-01-01"), index=range(1, 6))


def _sheet():
    _, c, *_ = _classified()
    return gr.build_review_sheet(c, _notes_text(), _scores(), ANCHORS, (0, 90),
                                 gr.review_pattern(_lca()["ne_gate"]), window=20)


def test_review_sheet_rows_windows_and_blank_calls():
    s = _sheet()
    assert list(s.columns) == gr.sheet_columns((0, 90))
    assert list(s["review_group"]) == list(gr.REVIEW_GROUPS)
    r = s.set_index("RPT_ID")
    assert (r.loc["21", "in_window_0"], r.loc["21", "in_window_90"]) == (1, 0)   # day 31
    assert (r.loc["10", "in_window_0"], r.loc["10", "in_window_90"]) == (1, 1)
    assert "Synaptophysin" in r.loc["10", "text_window"] and r.loc["10", "review_id"] == "1:10"
    assert s["reviewer_call"].eq("").all()


def test_text_window_merges_truncates_and_falls_back():
    rx = re.compile("NE")
    assert gr.text_window("aaNEbbNEcc", rx, window=2) == "aaNEbbNEcc"
    assert gr.text_window("xxxxNExxxxxxxxxxNExxxx", rx, window=1) == "…xNEx… | …xNEx…"
    assert gr.text_window("abcdef", rx, window=2) == "abcd…"
    assert gr.text_window("NE" + "x" * 50, rx, window=50, max_chars=10).endswith("[truncated]")


def test_write_guard_and_read_back(tmp_path):
    s = _sheet()
    path = gr.write_review_sheet(s, tmp_path / "sheet.csv")
    gr.write_review_sheet(s, path)                                 # blank sheet: overwrite is fine
    filled = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    filled.loc[filled["RPT_ID"].eq("10"), "reviewer_call"] = "not_ne"
    filled.loc[filled["RPT_ID"].eq("50"), "reviewer_call"] = " TRUE_NE "
    filled.loc[filled["RPT_ID"].eq("20"), "reviewer_call"] = "true_ne"
    filled.loc[filled["RPT_ID"].eq("30"), "reviewer_call"] = "unclear"
    filled.to_csv(path, index=False)
    with pytest.raises(FileExistsError):
        gr.write_review_sheet(s, path)

    reviewed, stale = gr.read_reviewed(path, s)
    assert stale == []
    summary = gr.summarize_review(reviewed).set_index("review_group")
    assert summary.loc["gated", ["reviewed", "not_ne", "frac_true_ne"]].tolist() == [1, 1, 0.0]
    assert summary.loc["gate_rejected", "true_ne"] == 1
    assert summary.loc["not_in_evidence", "unclear"] == 1 and np.isnan(summary.loc["not_in_evidence", "frac_true_ne"])
    by_reason = gr.summarize_by_reason(reviewed)
    assert by_reason.iloc[0][["reason", "true_ne"]].tolist() == ["negative_ihc", 1]

    bad = filled.assign(reviewer_call="maybe")
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="Unknown reviewer_call"):
        gr.read_reviewed(tmp_path / "bad.csv", s)
    pd.concat([filled, filled.iloc[[0]].assign(review_id="9:99")]).to_csv(tmp_path / "stale.csv", index=False)
    assert gr.read_reviewed(tmp_path / "stale.csv", s)[1] == ["9:99"]


def test_definitions_change_nb11_event_counts(tmp_path):
    findings, c, ihc, inputs = _classified()
    s = _sheet()
    s.loc[s["RPT_ID"].eq("10"), "reviewer_call"] = "not_ne"
    s.loc[s["RPT_ID"].eq("50"), "reviewer_call"] = "true_ne"
    s.loc[s["RPT_ID"].eq("20"), "reviewer_call"] = "true_ne"
    defs = gr.definition_keys(findings, s)
    assert set(defs["gated"]["RPT_ID"]) == {"10", "40"}
    assert set(defs["ungated"]["RPT_ID"]) == {"10", "40", "50"}
    assert set(defs["reviewed"]["RPT_ID"]) == {"40", "50", "20"}
    ev = gr.event_counts(_scores(), ihc, defs, inputs["status"], ANCHORS, (90,))
    e = ev.set_index(["definition", "outcome"])["events"]
    assert e[("gated", "ne_pos_all")] == 1                       # patient 1 (40 fails the specimen rule)
    assert e[("ungated", "ne_pos_all")] == 2                     # + patient 5
    assert e[("reviewed", "ne_pos_all")] == 2                    # patients 5 and 2; patient 1 dropped
    assert ev.set_index(["definition", "outcome"]).loc[("gated", "ne_pos_all"), "n"] == 5


def _label_inputs(status=None):
    lca = _lca()
    inputs, _ = gr.load_llm_inputs(lca, "dir")
    if status is not None:
        inputs = {**inputs, "status": status}
    return lca, inputs


def test_patient_label_definitions_drop_each_screen():
    lca, inputs = _label_inputs({1: "ok", 3: "ok", 4: "ok", 5: "ok", 6: "failed", 7: "ok"})
    lab = gr.patient_label_definitions(lca, inputs).set_index("DFCI_MRN")
    # 1 passes; 5 rejected by ne_gate; 3 dropped by the trial screen; 4 passes (non-prostate, still a label)
    assert lab["nepc_status_gated"].to_dict() == {1: "nepc", 3: "non_nepc", 4: "nepc", 5: "non_nepc",
                                                 6: "unlabeled", 7: "non_nepc"}
    assert lab.loc[5, "nepc_status_no_ne_gate"] == "nepc" and lab.loc[3, "nepc_status_no_ne_gate"] == "non_nepc"
    assert lab.loc[3, "nepc_status_no_gate"] == "nepc"
    assert lab.loc[5, "n_findings_ne_gate"] == 1 and lab.loc[3, "n_findings_trial_context"] == 1


def test_patient_label_definitions_require_gate_parity():
    lca = _lca(gate=lambda f: ({5: [f.iloc[1].to_dict()]}, {}))
    inputs, _ = gr.load_llm_inputs(lca, "dir")
    with pytest.raises(RuntimeError, match="disagrees"):
        gr.patient_label_definitions(lca, inputs)


def test_stored_label_drift():
    lca, inputs = _label_inputs()
    lab = gr.patient_label_definitions(lca, inputs)
    stored = lab[["DFCI_MRN"]].assign(nepc_status=lab["nepc_status_gated"])
    assert gr.stored_label_drift(lab, stored) == 0
    stored.loc[stored["DFCI_MRN"].eq(5), "nepc_status"] = "nepc"
    assert gr.stored_label_drift(lab, stored) == 1


def test_manual_comparison(tmp_path):
    lca, inputs = _label_inputs({1: "ok", 3: "ok", 4: "ok", 5: "ok", 6: "failed"})
    lab = gr.patient_label_definitions(lca, inputs)
    path = tmp_path / "manual.csv"
    pd.DataFrame({"DFCI_MRN": ["1", "3.0", "4", "5", "6", "8", "9"],
                  "updated_manual_low_confidence": ["NEPC", "nepc", "avpc", "conventional",
                                                    "nepc", "conventional", ""]}).to_csv(path, index=False)
    manual = gr.load_manual_annotations(path)
    assert len(manual) == 6                                              # blank dropped
    j = gr.join_manual(lab, manual, scored_ids={1, 5})
    assert not j.set_index("DFCI_MRN").loc[8, "in_llm_cohort"]

    m = gr.manual_metrics(j).set_index(["definition", "scope"])
    g = m.loc[("gated", "annotated")]
    # labeled: 1 nepc/nepc (tp), 3 nepc/non (fn), 4 avpc/nepc (fp), 5 conv/non (tn); 6 and 8 unlabeled
    assert (g["tp"], g["fn"], g["fp"], g["tn"], g["n"], g["n_unlabeled"]) == (1, 1, 1, 1, 4, 2)
    assert m.loc[("no_ne_gate", "annotated"), "fp"] == 2               # 5 flips to nepc
    assert m.loc[("no_gate", "annotated"), "sensitivity"] == 1.0       # 3 flips to nepc
    assert m.loc[("gated", "annotated_scored"), "n"] == 2

    c = gr.manual_confusion(j).set_index(["definition", "scope", "manual_label"])
    assert c.loc[("gated", "annotated", "nepc")].to_dict() == {"nepc": 1, "non_nepc": 1, "unlabeled": 1}
    assert c.loc[("gated", "annotated", "conventional"), "unlabeled"] == 1

    flips = gr.label_flips(j)
    assert set(flips["DFCI_MRN"]) == {3, 5}


def test_manual_annotations_validate(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame({"DFCI_MRN": ["1"], "updated_manual_low_confidence": ["squamous"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected"):
        gr.load_manual_annotations(path)
    pd.DataFrame({"DFCI_MRN": ["1", "1"], "updated_manual_low_confidence": ["nepc", "avpc"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one row"):
        gr.load_manual_annotations(path)
