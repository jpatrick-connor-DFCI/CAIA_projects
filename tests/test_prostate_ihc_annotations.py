import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.prostate_ihc import annotations as ann  # noqa: E402
from COMPASS.prostate_ihc import paths as ihc_paths  # noqa: E402

ANCHOR = pd.Timestamp("2020-01-01")

# kind -> lesional (marker, result) rows
KINDS = {
    "stain": [("p63", "negative")],
    "ki67": [("Ki-67", None)],
    "ne": [("Synaptophysin", "positive")],
    "amphi": [("Synaptophysin", "positive"), ("AR", "positive")],
    "nepc": [("Synaptophysin", "positive"), ("NKX3.1", "negative")],
    "arlow": [("AR", "negative"), ("PSA", "negative")],
    "ar_mixed": [("AR", "negative"), ("PSA", "positive")],
    "disc_ne": [("Synaptophysin", "positive"), ("Synaptophysin", "negative"), ("NKX3.1", "negative")],
    "disc_ar": [("AR", "positive"), ("AR", "negative")],
    "lost": [("PTEN", "lost")],
}


def _day(d):
    return None if d is None else (ANCHOR + np.timedelta64(int(d), "D")).strftime("%Y-%m-%d")


def tables(specs, discordant="neither"):
    """specs: (mrn, part, day or None, kind[, relevant]) -> (long, spec_all) frames like the intermediates.

    ``discordant`` sets how spec_all's NE_any_pos / ARaxis_any_pos treat a marker called both ways: "neither" (the
    original intermediates) or "both" (what a rerun of the current extraction writes)."""
    if discordant not in ("neither", "both"):
        raise ValueError(discordant)
    lrows, srows = [], []
    for spec in specs:
        mrn, part, day, kind = spec[:4]
        relevant = spec[4] if len(spec) > 4 else True
        acc = f"BS20-{mrn}"
        for marker, result in KINDS[kind]:
            lrows.append({"MRN_INT": mrn, "ACCESSION": acc, "PART": part, "target": "lesional",
                          "marker_norm": marker, "group": "g", "result": result, "source": "extractor",
                          "specimen_date": _day(day)})
        # background rows never call or date a specimen
        lrows.append({"MRN_INT": mrn, "ACCESSION": acc, "PART": part, "target": "background",
                      "marker_norm": "Synaptophysin", "group": "g", "result": "positive", "source": "extractor",
                      "specimen_date": _day(-5000)})
        calls = {}
        for marker, result in KINDS[kind]:
            calls.setdefault(marker, set()).add(result)
        pos = lambda ms: any(m in ms and {"positive", "retained"} & calls[m]  # noqa: E731
                             and (discordant == "both" or not {"negative", "lost"} & calls[m])
                             for m in calls if m in ms)
        srows.append({"MRN_INT": mrn, "ACCESSION": acc, "PART": part,
                      "prostate_evidence": "2 prostate site" if relevant else "other primary",
                      "prostate_relevant": relevant, "ihc_phenotype": "x", "NE_histology": False,
                      "NE_any_pos": bool(pos(ann.NE_MARKERS)), "ARaxis_any_pos": bool(pos(ann.AR_AXIS_MARKERS))})
    return pd.DataFrame(lrows), pd.DataFrame(srows)


def test_marker_sets_match_extraction():
    ex = pytest.importorskip("COMPASS.prostate_ihc.extract_prostate_ihc")
    assert list(ann.NE_MARKERS) == ex.NE and list(ann.AR_AXIS_MARKERS) == ex.ARX
    assert list(ann.POSITIVE_RESULTS) == ex.POSV


def test_flags_per_kind():
    kinds = list(KINDS)
    long, spec = tables([(i, "A", 100, k) for i, k in enumerate(kinds)])
    out = ann.specimen_annotations(long, spec).set_index("MRN_INT")
    got = {kinds[i]: tuple(out.loc[i, ["ihc_any", "ne_pos", "nepc_like", "ar_low"]]) for i in out.index}
    assert got == {
        "stain": (True, False, False, False),
        "ki67": (True, False, False, False),
        "ne": (True, True, False, False),
        "amphi": (True, True, False, False),
        "nepc": (True, True, True, True),
        "arlow": (True, False, False, True),
        "ar_mixed": (True, False, False, False),
        "disc_ne": (True, True, True, True),      # discordant NE counts as positive
        "disc_ar": (True, False, False, False),   # discordant AR is neither positive nor negative
        "lost": (True, False, False, False),
    }
    assert out.loc[0, "specimen_date"] == pd.Timestamp(_day(100))      # lesional, not background
    calls = ann.marker_calls(long).set_index(["MRN_INT", "marker_norm"])["call"]
    assert calls[(kinds.index("disc_ne"), "Synaptophysin")] == "discordant"
    assert calls[(kinds.index("ki67"), "Ki-67")] == "equivocal"
    assert calls[(kinds.index("lost"), "PTEN")] == "negative"


def test_specimen_is_part_level_and_keeps_nonrelevant():
    long, spec = tables([(1, "A", 100, "ne"), (1, "B", 100, "stain", False), (2, None, None, "arlow")])
    out = ann.specimen_annotations(long, spec)
    assert len(out) == 3 and out["SPECIMEN_ID"].is_unique
    assert out.set_index("PART").loc["B", "prostate_relevant"] == False  # noqa: E712
    assert out.set_index("MRN_INT").loc[2, "PART"] == "" and pd.isna(out.set_index("MRN_INT").loc[2, "specimen_date"])


def test_rejects_inconsistent_intermediates():
    long, spec = tables([(1, "A", 100, "disc_ne")])
    for kind, col, value in (("stain", "NE_any_pos", True), ("ne", "NE_any_pos", False),
                             ("arlow", "ARaxis_any_pos", True), ("amphi", "ARaxis_any_pos", False)):
        lg, sp = tables([(1, "A", 100, kind)])
        with pytest.raises(ValueError, match=f"disagree on {col}"):
            ann.specimen_annotations(lg, sp.assign(**{col: value}))
    with pytest.raises(ValueError, match="do not line up"):
        ann.specimen_annotations(long, pd.concat([spec, spec.assign(PART="Z")]))
    with pytest.raises(ValueError, match="Duplicate"):
        ann.specimen_annotations(long, pd.concat([spec, spec]))
    with pytest.raises(ValueError, match="lacks columns"):
        ann.specimen_annotations(long, spec.drop(columns="prostate_relevant"))
    bad = long.assign(specimen_date="soon")
    with pytest.raises(ValueError, match="Unparseable"):
        ann.specimen_annotations(bad, spec)


def test_either_discordant_reading_of_spec_all_gives_same_flags():
    specs = [(i, "A", 100, k) for i, k in enumerate(KINDS)]
    legacy = ann.specimen_annotations(*tables(specs, "neither"))
    rerun = ann.specimen_annotations(*tables(specs, "both"))
    pd.testing.assert_frame_equal(legacy, rerun)


def test_load_intermediates_reads_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    long, spec = tables([(1, "A", 100, "nepc")])
    pd.DataFrame({"MRN_INT": [1]}).to_parquet(tmp_path / ihc_paths.COHORT_FILE)
    long.assign(RAW_TEXT="x").to_parquet(tmp_path / ihc_paths.LONG_FILE)
    spec.assign(diagnosis_text="x").to_parquet(tmp_path / ihc_paths.SPECIMEN_ALL_FILE)
    out = ann.build(tmp_path)
    assert out["specimens"]["nepc_like"].tolist() == [True]
    with pytest.raises(FileNotFoundError):
        ann.load_intermediates(tmp_path / "missing")


def test_summary_and_marker_frequency():
    long, spec = tables([(1, "A", 100, "nepc"), (2, "A", 100, "arlow"), (3, "A", 100, "ne", False)])
    s = ann.specimen_annotations(long, spec)
    summ = ann.summary(s).set_index("flag")
    assert summ.loc["NE-positive", ["specimens_all", "specimens_relevant"]].tolist() == [2, 1]
    assert summ.loc["AR-low (AR-axis negative)", "patients_relevant"] == 2
    mf = ann.marker_frequency(ann.marker_calls(long), s).set_index("marker")
    assert mf.loc["Synaptophysin"].tolist() == [1, 1, 0, 0, 1]      # the non-relevant NE+ one is dropped
    assert mf.loc["NKX3.1"].tolist() == [1, 0, 1, 0, 1]
    assert mf.loc["AR"].tolist() == [1, 0, 1, 0, 0]
