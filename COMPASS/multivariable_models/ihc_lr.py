"""Logistic regression of future prostate IHC (and NE-positive IHC) on the federated risk score.

Backs ``notebooks/testing/11_LR_for_IHC.ipynb``. For each ADT landmark, the frozen
federated (no-DFCI) platinum-trained XGBoost score from notebook 04 (``endpoint ==
"nepc"`` rows: patients with NEPC or censoring at/before the landmark are not at
risk) is the only predictor, z-scored within each fit, in three unpenalized LRs:

* ``ihc_any``          -- any qualifying IHC note after ADT + landmark; all at-risk patients.
* ``ne_pos_all``       -- any NE-positive qualifying IHC note after ADT + landmark;
                          at-risk patients without an NE-positive IHC note at/before it.
* ``ne_pos_given_ihc`` -- same event, among patients with >=1 qualifying future IHC note
                          (and no prior NE-positive one).

**IHC notes** come from a regex over raw pathology ``RPT_TEXT``: a generic IHC term
or a named marker (NE, prostate lineage, basal-cell/diagnostic, other). A note
*qualifies* when it has IHC and the specimen is prostate cancer: the first site named
in ``PROC_DESC`` + the report's opening is prostate, or the text states prostatic
origin; a note naming a non-prostate primary is excluded unless it also states
prostatic origin.

**NE-positive** = a qualifying IHC note whose report (``DFCI_MRN`` + ``RPT_ID``) has at
least one LLM NEPC finding (any of the four evidence types) that passes the same gate
``nepc_labels.parquet`` uses (ne_gate + trial-context screen, final attempts only). The
gate is imported read-only from the LLM_clinical_annotations repo. A note without a
gated finding is NE-negative only with complete patient-level LLM review (or a
validated no-trigger status). Incomplete or absent review excludes the patient
from both NE outcomes, including when some positive findings are already known.

**Timing.** Notes are dated by their report ``EVENT_DATE``. "Future" = ADT + landmark <
note date <= ADT + MAX_FOLLOWUP_DAYS (the models' 10-year cap); the label is "ever
during that window", not fixed-horizon. Prior IHC notes do not exclude anyone; prior
NE-positive IHC notes exclude the patient from both NE outcomes.

AUC is from out-of-fold predictions (stratified K-fold) with a patient bootstrap 95%
CI; odds ratios (per SD), Wald CIs and LRT p-values come from the full-population fit.
The LR machinery is reused from ``nepc_subtyping_lr``.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from COMPASS.multivariable_models import nepc_subtyping_lr as nsl

ID_COL = nsl.ID_COL                     # "person_id" (scores / LR frame)
LOCAL_ID_COL = nsl.LOCAL_ID_COL         # "DFCI_MRN" (notes, LLM findings, local frame)
RISK_COL = nsl.RISK_COL
ANCHOR_SOURCE_COL = "TREATMENT_ANCHOR_DATE"
MAX_FOLLOWUP_DAYS = 3650
SPECIMEN_HEADER_CHARS = 300
IHC_REGEX_VERSION = "ihc-regex-v2"
NE_REVIEW_COMPLETE = {"ok", "no_trigger_notes"}

NOTE_COLUMNS = ["RPT_ID", LOCAL_ID_COL, "EVENT_DATE", "PROC_DESC", "RPT_TEXT"]

# Outcome -> event column; the population is set by outcome_population().
OUTCOMES = {
    "ihc_any": "ihc_future",
    "ne_pos_all": "ne_pos_future",
    "ne_pos_given_ihc": "ne_pos_future",
}

MODEL_COLUMNS = [
    "cohort", "landmark_days", "outcome", "n", "events", "lrt_stat", "lrt_df", "p_lrt",
    "auc", "auc_lo", "auc_hi", "converged", "n_failed_folds", "note",
]
COEF_COLUMNS = [
    "cohort", "landmark_days", "outcome", "term", "n", "events",
    "coef", "se", "odds_ratio", "or_lo", "or_hi", "p_wald", "note",
]
COUNT_COLUMNS = [
    "cohort", "landmark_days", "n_cohort_list", "n_at_risk",
    "n_ne_review_complete", "n_ne_review_excluded",
    "n_any_ihc_mention_future", "n_ihc_future", "n_ne_pos_future", "n_prior_ne_pos",
    "n_ne_pos_all", "events_ne_pos_all", "n_ne_pos_given_ihc", "events_ne_pos_given_ihc",
    "n_ne_gated_ihc_lost_to_specimen", "n_ne_gated_without_ihc_future",
]
AUDIT_COLUMNS = [
    LOCAL_ID_COL, "RPT_ID", "note_date", "days_from_adt", "ihc_any", "ihc_generic",
    "ihc_markers", "ihc_panels", "specimen_site", "prostate_site", "prostatic_origin",
    "non_prostate_primary", "qualifies", "gated_evidence_types", "ne_gated", "ne_positive",
    "ne_review_status",
]


# --------------------------------------------------------------------------- #
# Regex
# --------------------------------------------------------------------------- #
_FLAGS = re.IGNORECASE
_CLAUSE_BREAK = re.compile(r"[.;\n]")
_WINDOW = 40

IHC_GENERIC = r"\b(?:immunohistochem\w*|ihc|immuno-?stain\w*|immunoperoxidase|immunoreactiv\w*)\b"
# A generic term that says IHC was not done / not resulted / only suggested.
GENERIC_NEGATED_BEFORE = (r"\b(?:no|pending|awaiting|planned|recommend\w*|suggest\w*|consider\w*|could|may|might|can|if)"
                          r"\s+(?:\w+\s+){0,2}$")
NOT_RESULTED_AFTER = (r"^\W*(?:\w+\W+){0,4}?(?:not\s+(?:been\s+)?(?:performed|done|obtained|available|"
                      r"indicated|required|attempted)|pending|to\s+follow|deferred|cancel\w*|"
                      r"will\s+be\s+(?:reported|issued|performed)|(?:is|are)\s+recommended|"
                      r"may\s+be\s+(?:performed|considered|helpful))\b")

# Apply prospective cues to marker lists as well as generic IHC mentions.
PROSPECTIVE_BEFORE = re.compile(
    r"\b(?:recommend\w*|suggest\w*|consider\w*|pending|awaiting|planned)\b[^.;\n]*$", _FLAGS)
NO_BEFORE = re.compile(r"\bno\s+(?:\w+\s+){0,2}$", _FLAGS)
PERFORMED_AFTER = re.compile(
    r"^\s*(?:stains?|staining|testing)?\s*(?:was|were|is|are)?\s*(?:performed|done|ordered)\b", _FLAGS)
PSMA_IMAGING = re.compile(r"\b(?:pet|imaging|scan|radioligand)\b", _FLAGS)


# Panel -> {marker label: pattern}. The NE panel is ne_gate's NE_MARKER, filled in by
# build_patterns so it stays identical to the gate the labels use.
NE_PANEL = "neuroendocrine"
MARKERS = {
    "lineage": {
        "NKX3.1": r"\bnkx\s?3[.\s-]?1\b",
        "AR": r"\b(?:androgen\s+receptor|ar)\b",
        "PSA": r"\b(?:psa|prostate[\s-]specific\s+antigen)\b",
        "PSMA": r"\b(?:psma|prostate[\s-]specific\s+membrane\s+antigen)\b",
        "PSAP": r"\b(?:psap|prostatic\s+acid\s+phosphatase)\b",
        "prostein": r"\b(?:prostein|p501s)\b",
    },
    "basal_diagnostic": {
        "p63": r"\bp\s?63\b",
        "HMWCK": (r"\b(?:hmw[\s-]?ck|high[\s-]molecular[\s-]weight\s+(?:cyto)?keratins?|"
                  r"34\s?(?:β|beta|b)\s?e\s?12|ck\s?903)\b"),
        "CK5/6": r"\b(?:ck|cytokeratin)\s?5\s?/\s?6\b",
        "AMACR": r"\b(?:amacr|p504s|racemase)\b",
        "PIN4": r"\bpin[\s-]?4\b",
    },
    "other": {
        "ERG": r"\berg\b",
        "Ki-67": r"\b(?:ki[\s-]?67|mib[\s-]?1)\b",
        "TTF-1": r"\bttf[\s-]?1\b",
        "GATA3": r"\bgata[\s-]?3\b",
        "CK7": r"\b(?:ck|cytokeratin)\s?7\b",
        "CK20": r"\b(?:ck|cytokeratin)\s?20\b",
        "p53": r"\bp\s?53\b",
        "RB1": r"\b(?:rb1?|retinoblastoma(?:\s+protein)?)\b",
        "PTEN": r"\bpten\b",
    },
}
# Short or dual-use tokens count only with a stain cue in the same clause (+/-40 chars).
CONTEXT_REQUIRED = {"AR", "PSA", "PSMA", "ERG", "RB1", "PTEN"}
STAIN_CUE = (r"(?:\b(?:stain\w*|immuno\w*|ihc|positiv\w*|negativ\w*|pos|neg|reactiv\w*|express\w*|"
             r"retained|intact|loss|lost|patchy|diffuse\w*|focal\w*|labell?\w*|highlight\w*)\b|"
             r"\(\s*[+-]\s*\)|(?<=\w)\+)")
# Per-marker mentions that are not stains: serum PSA values, PSMA imaging.
MARKER_EXCLUDE_BEFORE = {"PSA": r"\b(?:serum|rising|nadir|doubling)\s+$"}
MARKER_EXCLUDE_AFTER = {
    "PSA": r"^\s*(?:level|value|velocity|doubling|nadir|of|was|is|=|:|\(ng)?\s*[<>≤≥]?\s*\d",
    "PSMA": r"^\s*-?\s*(?:pet|scan|avid|imaging|ct|targeted|radioligand|lu)\b",
}

# Specimen. The prostate/non-prostate site patterns are the ones in
# notebooks/patient_extraction/01_high_vs_low_risk_transfer_pxs.ipynb (cell 13).
SPECIMEN_SITES = (
    ("prostate", r"\b(?:prostat\w*|turp|transurethral|seminal\s+vesicles?)\b"),
    ("non-prostate", r"\b(?:bone|marrow|liver|hepatic|lymph|nodes?|nodal|lung|pulmonary|pleura\w*|adrenal"
                     r"|brain|dura\w*|spine|spinal|vertebra\w*|soft\s+tissue|peritone\w*|omentum|omental)\b"),
)
PROSTATIC_ORIGIN = (
    r"\bmetasta\w*\s+(?:[\w-]+\s+){0,3}?prostat\w*\s+(?:adeno)?carcinoma\b"
    r"|\bprostat\w*\s+(?:origin|primary)\b"
    r"|\b(?:origin|primary)\s+(?:in|from)\s+(?:the\s+)?prostate\b"
    r"|\b(?:consistent|compatible)\s+with\s+(?:[\w-]+\s+){0,3}?prostat\w*\s+(?:adeno)?carcinoma\b"
)
# Non-prostate primaries. Built from ne_gate's NON_PROSTATE_NE terms, except that bare
# "urothelial" (which also matches benign urothelial mucosa in TURP / prostatic-urethra
# specimens) is narrowed to urothelial carcinoma.
NON_PROSTATE_PRIMARY = (
    r"\b(?:sclc|small[\s-]?cell\s+lung|merkel|carcinoid|pulmonary\s+(?:origin|primary|small)|"
    r"lung\s+(?:origin|primary)|urothelial\s+(?:[\w-]+\s+){0,2}?carcinoma|"
    r"bladder\s+(?:primary|origin|cancer|carcinoma|tumou?r)|pancreatic\s+(?:net|neuroendocrine)|pnet|"
    r"gep[\s-]?nets?|well[\s-]differentiated\s+neuroendocrine\s+tumou?r|"
    r"renal\s+cell\s+carcinoma|rcc|colorectal\s+(?:adeno)?carcinoma|colonic\s+adenocarcinoma|"
    r"colon\s+(?:adeno)?carcinoma|(?:lung|pulmonary)\s+(?:adenocarcinoma|squamous)|"
    r"non[\s-]small[\s-]cell|melanoma|lymphoma|hepatocellular|hcc)\b"
)
# A specimen phrase preceded by one of these (same clause, 60 chars) is not asserted.
# "History of ..." vetoes a non-prostate primary (a prior other cancer is not this
# specimen) but not prostatic origin ("history of prostate cancer: metastatic ...").
_NEGATION_CUES = (r"no|not|negative|without|rule\s+out|r/o|exclude\w*|versus|vs|differential"
                  r"|unlikely|against")
ORIGIN_NOT_ASSERTED_BEFORE = rf"\b(?:{_NEGATION_CUES})\b[^.;:\n]{{0,60}}$"
PRIMARY_NOT_ASSERTED_BEFORE = (rf"\b(?:{_NEGATION_CUES}|history|hx|h/o|prior|previous|remote)\b"
                               r"[^.;:\n]{0,60}$")
# ne_gate.NE_MARKER alternatives -> per-marker labels (applied to the matched text).
NE_MARKER_LABELS = (
    ("synaptophysin", r"synaptophysin"), ("chromogranin", r"chromogranin"), ("CD56", r"cd\s?56"),
    ("INSM1", r"insm"), ("NSE", r"enolase|nse"),
)


@dataclass(frozen=True)
class IHCPatterns:
    generic: re.Pattern
    generic_negated_before: re.Pattern
    not_resulted_after: re.Pattern
    markers: dict                 # panel -> {label: compiled}
    stain_cue: re.Pattern
    exclude_before: dict
    exclude_after: dict
    sites: tuple
    prostatic_origin: re.Pattern
    non_prostate_primary: re.Pattern
    origin_not_asserted_before: re.Pattern
    primary_not_asserted_before: re.Pattern
    ne_labels: tuple


def build_patterns(ne_marker: str) -> IHCPatterns:
    """Compile every pattern; ``ne_marker`` is ne_gate.NE_MARKER's source pattern."""
    c = lambda p: re.compile(p, _FLAGS)  # noqa: E731
    markers = {NE_PANEL: {NE_PANEL: c(ne_marker)}}
    markers.update({panel: {k: c(v) for k, v in pats.items()} for panel, pats in MARKERS.items()})
    return IHCPatterns(
        generic=c(IHC_GENERIC),
        generic_negated_before=c(GENERIC_NEGATED_BEFORE),
        not_resulted_after=c(NOT_RESULTED_AFTER),
        markers=markers,
        stain_cue=c(STAIN_CUE),
        exclude_before={k: c(v) for k, v in MARKER_EXCLUDE_BEFORE.items()},
        exclude_after={k: c(v) for k, v in MARKER_EXCLUDE_AFTER.items()},
        sites=tuple((site, c(p)) for site, p in SPECIMEN_SITES),
        prostatic_origin=c(PROSTATIC_ORIGIN),
        non_prostate_primary=c(NON_PROSTATE_PRIMARY),
        origin_not_asserted_before=c(ORIGIN_NOT_ASSERTED_BEFORE),
        primary_not_asserted_before=c(PRIMARY_NOT_ASSERTED_BEFORE),
        ne_labels=tuple((label, c(p)) for label, p in NE_MARKER_LABELS),
    )


def _clause_before(text: str, start: int, window: int = _WINDOW) -> str:
    seg = text[max(0, start - window):start]
    return _CLAUSE_BREAK.split(seg)[-1]


def _clause_after(text: str, end: int, window: int = _WINDOW) -> str:
    seg = text[end:end + window]
    return _CLAUSE_BREAK.split(seg)[0]


def _not_resulted(text, m, pats: IHCPatterns) -> bool:
    return bool(pats.not_resulted_after.search(_clause_after(text, m.end(), 60)))


def generic_ihc(text: str, pats: IHCPatterns) -> bool:
    """A generic IHC term that is not negated, not pending, and not only suggested."""
    for m in pats.generic.finditer(text):
        if pats.generic_negated_before.search(_clause_before(text, m.start())):
            continue
        if PROSPECTIVE_BEFORE.search(_clause_before(text, m.start(), 100)):
            continue
        if _not_resulted(text, m, pats):
            continue
        return True
    return False


def marker_hits(text: str, pats: IHCPatterns) -> dict:
    """Panel -> sorted marker labels with at least one stain mention (negative results count)."""
    hits = {}
    for panel, markers in pats.markers.items():
        for label, rx in markers.items():
            for m in rx.finditer(text):
                before, after = _clause_before(text, m.start()), _clause_after(text, m.end())
                if PROSPECTIVE_BEFORE.search(_clause_before(text, m.start(), 100)):
                    continue
                # "No ... expression" is a completed negative stain; "No ...
                # stain was performed" is not a completed workup.
                if NO_BEFORE.search(before) and PERFORMED_AFTER.search(after):
                    continue
                if label == "PSMA" and PSMA_IMAGING.search(before + " " + after):
                    continue
                if label in pats.exclude_before and pats.exclude_before[label].search(before):
                    continue
                if label in pats.exclude_after and pats.exclude_after[label].search(after):
                    continue
                if label in CONTEXT_REQUIRED and not (pats.stain_cue.search(before)
                                                      or pats.stain_cue.search(after)):
                    continue
                if _not_resulted(text, m, pats):
                    continue
                if panel == NE_PANEL:
                    label_m = next((lab for lab, lrx in pats.ne_labels if lrx.search(m.group(0))),
                                   NE_PANEL)
                    hits.setdefault(panel, []).append(label_m)
                    continue                      # keep scanning: several NE markers per note
                hits.setdefault(panel, []).append(label)
                break
    return {p: sorted(set(v)) for p, v in hits.items()}


def specimen_site(header: str, pats: IHCPatterns) -> str:
    """'prostate', 'non-prostate' or 'unspecified': whichever site is named first."""
    found = [(m.start(), site) for site, rx in pats.sites for m in [rx.search(header)] if m]
    return min(found)[1] if found else "unspecified"


def _asserted(rx: re.Pattern, veto: re.Pattern, text: str) -> bool:
    """Any match of ``rx`` whose preceding 60 characters (same clause) lack a ``veto`` cue."""
    return any(not veto.search(text[max(0, m.start() - 60):m.start()]) for m in rx.finditer(text))


def tag_note(text: str, proc_desc: str, pats: IHCPatterns) -> dict:
    text = text or ""
    header = f"{proc_desc or ''} {text[:SPECIMEN_HEADER_CHARS]}"
    generic = generic_ihc(text, pats)
    markers = marker_hits(text, pats)
    site = specimen_site(header, pats)
    origin = _asserted(pats.prostatic_origin, pats.origin_not_asserted_before, text)
    other_primary = _asserted(pats.non_prostate_primary, pats.primary_not_asserted_before, text)
    ihc_any = generic or bool(markers)
    prostate_site = site == "prostate"
    qualifies = ihc_any and (prostate_site or origin) and not (other_primary and not origin)
    return {
        "ihc_any": ihc_any, "ihc_generic": generic,
        "ihc_markers": ";".join(sorted({m for v in markers.values() for m in v})),
        "ihc_panels": ";".join(sorted(markers)),
        "specimen_site": site, "prostate_site": prostate_site, "prostatic_origin": origin,
        "non_prostate_primary": other_primary, "qualifies": qualifies,
    }


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_pathology_notes(path, mrns=None) -> pd.DataFrame:
    """Pathology reports (``RPT_ID`` as Utf8, the same cast LCA uses for note_uid)."""
    import polars as pl
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing pathology notes: {path} (set PROFILE_PATHOLOGY_NOTES).")
    lazy = pl.scan_parquet(path)
    names = lazy.collect_schema().names()
    missing = [c for c in NOTE_COLUMNS if c not in names and c != "PROC_DESC"]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    lazy = lazy.select(
        pl.col("RPT_ID").cast(pl.Utf8),
        pl.col(LOCAL_ID_COL).cast(pl.Int64, strict=False),
        pl.col("EVENT_DATE").cast(pl.Utf8).str.extract(r"^(\d{4}-\d{2}-\d{2})", 1).alias("note_date"),
        (pl.col("PROC_DESC").cast(pl.Utf8) if "PROC_DESC" in names else pl.lit(None, dtype=pl.Utf8))
        .alias("PROC_DESC"),
        pl.col("RPT_TEXT").cast(pl.Utf8).fill_null(""),
    ).filter(pl.col(LOCAL_ID_COL).is_not_null())
    if mrns is not None:
        lazy = lazy.filter(pl.col(LOCAL_ID_COL).is_in([int(m) for m in mrns]))
    out = lazy.collect().to_pandas()
    out[LOCAL_ID_COL] = out[LOCAL_ID_COL].astype(int)
    out["note_date"] = pd.to_datetime(out["note_date"], errors="coerce")
    return out


def tag_ihc_notes(notes: pd.DataFrame, pats: IHCPatterns) -> pd.DataFrame:
    """One row per note with the regex tags (no text)."""
    tags = pd.DataFrame([tag_note(t, p, pats) for t, p in
                         zip(notes["RPT_TEXT"].to_numpy(), notes["PROC_DESC"].to_numpy())],
                        index=notes.index)
    if tags.empty:
        tags = pd.DataFrame(columns=list(tag_note("", "", pats)), index=notes.index)
    return pd.concat([notes[[LOCAL_ID_COL, "RPT_ID", "note_date"]], tags], axis=1)


def import_lca(lca_root):
    """Read-only import of the LLM_clinical_annotations modules the gate needs."""
    lca_root = Path(lca_root)
    if not (lca_root / "tasks" / "nepc" / "ne_gate.py").exists():
        raise FileNotFoundError(
            f"LLM_clinical_annotations not found at {lca_root}; set COMPASS_LCA_ROOT.")
    if str(lca_root) not in sys.path:
        sys.path.insert(0, str(lca_root))
    from tasks import note_extraction
    from tasks.nepc import ne_gate, run_nepc_extraction
    return {"note_extraction": note_extraction, "ne_gate": ne_gate, "nepc": run_nepc_extraction}


def lca_commit(lca_root) -> str:
    try:
        return subprocess.run(["git", "-C", str(lca_root), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_note_uid(note_uid):
    """Split a note UID without discarding LCA's collision suffix."""
    if not isinstance(note_uid, str) or ":" not in note_uid:
        return None, None
    note_type, rest = note_uid.split(":", 1)
    return note_type, rest


def gated_rows_to_notes(gated: dict) -> pd.DataFrame:
    """``gate_findings`` output -> one row per gated pathology report with its evidence types."""
    rows = []
    for mrn, findings in gated.items():
        for f in findings:
            note_type, rpt_id = parse_note_uid(f.get("note_uid"))
            if note_type != "Pathology" or rpt_id in (None, "", "None"):
                continue
            if re.search(r"#\d+$", rpt_id):
                raise ValueError("Ambiguous pathology note UID: collision suffix cannot be mapped "
                                 "to RPT_ID safely. Resolve report identity before fitting.")
            rows.append({LOCAL_ID_COL: int(mrn), "RPT_ID": rpt_id, "evidence_type": f["evidence_type"]})
    if not rows:
        return pd.DataFrame(columns=[LOCAL_ID_COL, "RPT_ID", "gated_evidence_types"])
    df = pd.DataFrame(rows)
    return (df.groupby([LOCAL_ID_COL, "RPT_ID"])["evidence_type"]
              .agg(lambda s: ";".join(sorted(set(s)))).rename("gated_evidence_types").reset_index())


def load_gated_ne_notes(lca, llm_nepc_dir, *, mrns=None):
    """Return gated reports and patient review status from the same LCA snapshot.

    Fail on collision-suffixed pathology UIDs even when that note has no gated
    finding: otherwise an unsuffixed positive could be assigned to its sibling.
    ``mrns`` scopes validation to the patients included in this analysis.
    """
    inputs = lca["note_extraction"].load_label_inputs(lca["nepc"].SPEC, Path(llm_nepc_dir))
    selected = None if mrns is None else set(map(int, mrns))
    for mrn, uid in inputs["notes"]:
        if selected is not None and int(mrn) not in selected:
            continue
        note_type, report_id = parse_note_uid(uid)
        if note_type == "Pathology" and re.search(r"#\d+$", report_id):
            raise ValueError("Ambiguous pathology note UID in LCA evidence; resolve report "
                             "identity before fitting.")
    gated, _ = lca["nepc"].gate_findings(inputs["findings"])
    if selected is not None:
        gated = {m: f for m, f in gated.items() if int(m) in selected}
    status = {int(m): s for m, s in inputs["status"].items()
              if selected is None or int(m) in selected}
    return gated_rows_to_notes(gated), status


def merge_ne_positive(ihc_notes: pd.DataFrame, gated_notes: pd.DataFrame,
                      review_status: dict) -> pd.DataFrame:
    keys = [LOCAL_ID_COL, "RPT_ID"]
    if ihc_notes[keys].isna().any().any() or ihc_notes.duplicated(keys).any():
        raise ValueError("Missing or ambiguous pathology report IDs; resolve report identity "
                         "before attaching LLM findings.")
    out = ihc_notes.merge(gated_notes, on=keys, how="left", validate="one_to_one")
    out["ne_review_status"] = out[LOCAL_ID_COL].map(review_status).fillna("unknown")
    out["gated_evidence_types"] = out["gated_evidence_types"].fillna("")
    out["ne_gated"] = out["gated_evidence_types"].ne("")
    out["ne_positive"] = (out["qualifies"].astype(bool) & out["ne_gated"]).astype("boolean")
    unknown = (out["qualifies"].astype(bool) & ~out["ne_gated"]
               & ~out["ne_review_status"].isin(NE_REVIEW_COMPLETE))
    out.loc[unknown, "ne_positive"] = pd.NA
    return out


def load_anchor_dates(path) -> pd.Series:
    """MRN -> ADT anchor date, reading only the two needed columns of the long Stage-2 frame."""
    path = str(path)
    cols = [LOCAL_ID_COL, ANCHOR_SOURCE_COL]
    local = (pd.read_parquet(path, columns=cols) if path.endswith((".parquet", ".pq"))
             else pd.read_csv(path, usecols=cols, low_memory=False))
    return anchor_dates(local)


def anchor_dates(local: pd.DataFrame) -> pd.Series:
    """MRN -> ADT anchor date from the local long frame (one value per patient)."""
    a = local[[LOCAL_ID_COL, ANCHOR_SOURCE_COL]].dropna().drop_duplicates()
    a[ANCHOR_SOURCE_COL] = pd.to_datetime(a[ANCHOR_SOURCE_COL], errors="coerce", format="mixed")
    a = a.dropna().drop_duplicates()
    a[LOCAL_ID_COL] = pd.to_numeric(a[LOCAL_ID_COL], errors="raise").astype(int)
    if a[LOCAL_ID_COL].duplicated().any():
        raise ValueError("More than one TREATMENT_ANCHOR_DATE for some patients.")
    return a.set_index(LOCAL_ID_COL)[ANCHOR_SOURCE_COL]


# --------------------------------------------------------------------------- #
# Patient x landmark frame
# --------------------------------------------------------------------------- #
def validate_anchor_coverage(scores: pd.DataFrame, anchors: pd.Series):
    """Require an ADT date for every scored patient, even those without notes."""
    missing = scores[ID_COL].drop_duplicates().map(anchors).isna().sum()
    if missing:
        raise ValueError(f"{missing} scored patients lack an ADT anchor; reconcile scores "
                         "and longitudinal data before constructing IHC outcomes.")


def add_days_from_adt(notes: pd.DataFrame, anchors: pd.Series) -> pd.DataFrame:
    out = notes.copy()
    out["days_from_adt"] = (out["note_date"] - out[LOCAL_ID_COL].map(anchors)).dt.days
    return out


def build_ihc_frame(scores: pd.DataFrame, notes: pd.DataFrame, landmarks, *,
                    anchors: pd.Series, review_status: dict, max_days: int = MAX_FOLLOWUP_DAYS) -> pd.DataFrame:
    """One row per scored patient x landmark with the IHC outcome flags.

    ``notes`` is the merged note table; offsets are recomputed from the validated
    ``anchors``. Incomplete review makes both NE flags missing and excludes the
    patient from NE fits, while IHC occurrence remains evaluable. Patient flags:
    ``any_ihc_mention_future`` (IHC regardless of specimen), ``ihc_future``
    (qualifying), ``ne_pos_future``, ``prior_ne_pos`` (NE-positive note dated at or
    before ADT + landmark), ``ne_gated_ihc_lost_to_specimen`` (future IHC note with a
    gated finding that fails the specimen rule, and no NE-positive future note) and
    ``ne_gated_without_ihc_future`` (future gated report with no IHC regex hit).
    """
    validate_anchor_coverage(scores, anchors)
    notes = add_days_from_adt(notes, anchors)
    dated = notes.dropna(subset=["days_from_adt"])
    rows = []
    for lm in landmarks:
        sc = scores[scores["landmark_days"].eq(int(lm))][[ID_COL, "landmark_days", RISK_COL]]
        d = dated["days_from_adt"]
        fut = dated[(d > int(lm)) & (d <= max_days)]
        prior = dated[d <= int(lm)]

        def ids(mask_df):
            return set(mask_df[LOCAL_ID_COL].astype(int))

        flags = {
            "any_ihc_mention_future": ids(fut[fut["ihc_any"].astype(bool)]),
            "ihc_future": ids(fut[fut["qualifies"].astype(bool)]),
            "ne_pos_future": ids(fut[fut["ne_positive"].fillna(False)]),
            "prior_ne_pos": ids(prior[prior["ne_positive"].fillna(False)]),
            "ne_gated_without_ihc_future": ids(fut[fut["ne_gated"] & ~fut["ihc_any"].astype(bool)]),
        }
        lost = ids(fut[fut["ne_gated"] & fut["ihc_any"].astype(bool) & ~fut["qualifies"].astype(bool)])
        flags["ne_gated_ihc_lost_to_specimen"] = lost - flags["ne_pos_future"]
        sc = sc.copy()
        for name, s in flags.items():
            sc[name] = sc[ID_COL].isin(s).astype(int)
        sc["ne_review_status"] = sc[ID_COL].map(review_status).fillna("unknown")
        sc["ne_review_complete"] = sc["ne_review_status"].isin(NE_REVIEW_COMPLETE)
        for col in ("ne_pos_future", "prior_ne_pos"):
            sc[col] = sc[col].astype(float).where(sc["ne_review_complete"])
        rows.append(sc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# LR driver
# --------------------------------------------------------------------------- #
def outcome_population(sub: pd.DataFrame, outcome: str) -> pd.DataFrame:
    if outcome == "ihc_any":
        return sub
    if outcome == "ne_pos_all":
        return sub[sub["ne_review_complete"] & sub["prior_ne_pos"].eq(0)]
    if outcome == "ne_pos_given_ihc":
        return sub[sub["ne_review_complete"] & sub["prior_ne_pos"].eq(0) & sub["ihc_future"].eq(1)]
    raise ValueError(f"Unknown outcome {outcome!r}")


def _fit_outcome(pop, outcome, tags, *, min_events, n_folds, n_bootstrap, seed):
    y = pop[OUTCOMES[outcome]].to_numpy(dtype=int)
    n, events = len(y), int(y.sum())
    base = {**tags, "outcome": outcome, "n": n, "events": events}
    if events < min_events or n - events < min_events:
        return ({**base, "note": f"fewer than {min_events} events or non-events; not fit"}, [])
    X = nsl._design(pop, nsl.REFERENCE_MODEL)          # RISK_Z, z-scored within this fit
    fit = nsl.fit_logistic(y, X)
    oof, failed = nsl.oof_predictions(y, X, n_folds=n_folds, seed=seed, return_diagnostics=True)
    note = fit["note"]
    if not np.isfinite(oof).all():
        cv_note = (f"{failed} failed CV folds; AUC suppressed" if failed
                   else "fewer than 2 CV folds; AUC suppressed")
        note = "; ".join(filter(None, [note, cv_note]))
    row = {**base, "lrt_stat": fit["lrt_stat"], "lrt_df": fit["lrt_df"], "p_lrt": fit["p_lrt"],
           "auc": nsl._auc(y, oof), "converged": fit["converged"], "n_failed_folds": failed,
           "note": note}
    if n_bootstrap and np.isfinite(oof).all():
        draws = nsl.bootstrap_auc(y, {outcome: oof}, n_bootstrap=n_bootstrap, seed=seed)
        row["auc_lo"], row["auc_hi"] = nsl._ci(draws[outcome])
    coefs = [{**base, **r, "note": fit["note"]} for r in fit["terms"].to_dict("records")]
    return row, coefs


def _counts(sub: pd.DataFrame, tags: dict, n_list) -> dict:
    ne_all = outcome_population(sub, "ne_pos_all")
    ne_ihc = outcome_population(sub, "ne_pos_given_ihc")
    s = lambda col: int(sub[col].sum())  # noqa: E731
    return {**tags, "n_cohort_list": n_list, "n_at_risk": len(sub),
            "n_ne_review_complete": int(sub["ne_review_complete"].sum()),
            "n_ne_review_excluded": int((~sub["ne_review_complete"]).sum()),
            "n_any_ihc_mention_future": s("any_ihc_mention_future"), "n_ihc_future": s("ihc_future"),
            "n_ne_pos_future": s("ne_pos_future"), "n_prior_ne_pos": s("prior_ne_pos"),
            "n_ne_pos_all": len(ne_all), "events_ne_pos_all": int(ne_all["ne_pos_future"].sum()),
            "n_ne_pos_given_ihc": len(ne_ihc),
            "events_ne_pos_given_ihc": int(ne_ihc["ne_pos_future"].sum()),
            "n_ne_gated_ihc_lost_to_specimen": s("ne_gated_ihc_lost_to_specimen"),
            "n_ne_gated_without_ihc_future": s("ne_gated_without_ihc_future")}


def run_ihc_lr(frame: pd.DataFrame, members: dict, landmarks, *, outcomes=tuple(OUTCOMES),
               min_events: int = 5, n_folds: int = 5, n_bootstrap: int = 200, seed: int = 0):
    """Fit ``outcome ~ RISK_Z`` per cohort x landmark x outcome.

    ``members`` maps cohort -> MRN set (None = every scored patient). Rows with a
    missing risk score are dropped. Returns ``(models, coefficients, counts)``.
    """
    model_rows, coef_rows, count_rows = [], [], []
    for cohort, ids in members.items():
        for lm in landmarks:
            sub = frame[frame["landmark_days"].eq(int(lm)) & frame[RISK_COL].notna()]
            if ids is not None:
                sub = sub[sub[ID_COL].isin(ids)]
            tags = {"cohort": cohort, "landmark_days": int(lm)}
            count_rows.append(_counts(sub, tags, None if ids is None else len(ids)))
            for outcome in outcomes:
                row, coefs = _fit_outcome(
                    outcome_population(sub, outcome), outcome, tags, min_events=min_events,
                    n_folds=n_folds, n_bootstrap=n_bootstrap, seed=seed)
                model_rows.append(row)
                coef_rows += coefs
    return (pd.DataFrame(model_rows).reindex(columns=MODEL_COLUMNS),
            pd.DataFrame(coef_rows).reindex(columns=COEF_COLUMNS),
            pd.DataFrame(count_rows).reindex(columns=COUNT_COLUMNS))


def marker_frequency(notes: pd.DataFrame) -> pd.DataFrame:
    """Qualifying IHC notes per marker, overall and among NE-positive notes."""
    q = notes[notes["qualifies"].astype(bool)]
    rows = []
    for markers, ne in zip(q["ihc_markers"], q["ne_positive"].fillna(False)):
        for m in filter(None, str(markers).split(";")):
            rows.append({"marker": m, "ne_positive": bool(ne)})
    if not rows:
        return pd.DataFrame(columns=["marker", "notes", "ne_positive_notes"])
    df = pd.DataFrame(rows)
    return (df.groupby("marker").agg(notes=("ne_positive", "size"), ne_positive_notes=("ne_positive", "sum"))
              .sort_values("notes", ascending=False).reset_index())
