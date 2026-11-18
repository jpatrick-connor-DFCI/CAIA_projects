"""Exercise the separate extraction and exploratory notebooks with synthetic data only."""
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from survival_common import federated_endpoint_transfer as fet
from survival_common import federated_scoring as fs


NOTEBOOK = Path(__file__).resolve().parents[1] / (
    "COMPASS/notebooks/patient_extraction/01_high_vs_low_risk_transfer_pxs.ipynb"
)


TEXT_DIR = NOTEBOOK.parents[1] / "text_interpretability"
ANALYSIS_DIR = TEXT_DIR / "post_landmark"   # 13-16 on notes after each landmark
PRE_DIR = TEXT_DIR / "pre_landmark"         # the same four notebooks on notes up to each landmark
ANALYSIS_NOTEBOOK = ANALYSIS_DIR / "13_pathology_topics_vs_risk.ipynb"
TOPIC_NOTEBOOK = ANALYSIS_DIR / "15_pathology_bertopic_vs_risk.ipynb"   # BERTopic, split out of 13


def _cell_source(notebooks, cell_id):
    """Source of the first cell with this id, searching the notebooks in order."""
    for notebook in notebooks:
        for cell in json.loads(notebook.read_text())["cells"]:
            if cell.get("id") == cell_id:
                return notebook, "".join(cell["source"])
    raise KeyError(cell_id)


def _code_cell_ids(notebook):
    return [c["id"] for c in json.loads(notebook.read_text())["cells"] if c["cell_type"] == "code" and c["id"] != "config"]


def _analysis_source(cell_id):
    return _cell_source((ANALYSIS_NOTEBOOK, TOPIC_NOTEBOOK), cell_id)[1]


def _run_analysis(env, *cell_ids, notebook=None):
    for cell_id in cell_ids:
        path, src = _cell_source((notebook,) if notebook else (ANALYSIS_NOTEBOOK, TOPIC_NOTEBOOK), cell_id)
        exec(compile(src, f"{path}:{cell_id}", "exec"), env)


def _run_cells(env, *indices):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for i in indices:
        exec(compile("".join(cells[i]["source"]), f"{NOTEBOOK}:cell_{i}", "exec"), env)


@pytest.fixture
def extraction(tmp_path):
    scores = pd.DataFrame([
        {"model_family": "xgboost_cox", "config": "both", "endpoint": "platinum",
         "landmark_days": lm, fs.ID_COL: 100001 + i, "risk_score": float(i)}
        for lm in (0, 90, 180) for i in range(20)
    ])
    env = dict(np=np, pd=pd, fet=fet, fs=fs, SCORES_PATH=tmp_path / "scores.csv",
               DATA_PATH=tmp_path / "local.csv", PATHOLOGY_PATH=tmp_path / "notes.parquet",
               OUTPUT_DIR=tmp_path / "out", MODEL_FAMILY="xgboost_cox",
               SCORE_ENDPOINT="platinum", LANDMARKS=(0, 90, 180), DECILE=0.1,
               ENDPOINTS=("avpc", "platinum", "nepc"),
               WINDOW_MAX_DAYS=3650)   # 13-16 use post-landmark notes only
    return env, scores


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, "bad"])
def test_invalid_score_is_not_scored_at_one_landmark(extraction, invalid):
    env, scores = extraction
    scores["risk_score"] = scores["risk_score"].astype(object)
    scores.loc[(scores["landmark_days"] == 90) & (scores[fs.ID_COL] == 100001), "risk_score"] = invalid
    scores.to_csv(env["SCORES_PATH"], index=False)
    _run_cells(env, 4, 5)
    patient = env["cohort"].set_index("DFCI_MRN").loc[100001]
    assert patient["bottom_decile_lm0"]
    assert patient["decile_group_lm90"] == "not scored"
    assert not patient["bottom_decile_lm90"] and not patient["top_decile_lm90"]
    assert pd.isna(patient["risk_score_lm90"]) and pd.isna(patient["risk_percentile_lm90"])
    assert env["cutoffs"].loc[90, "n_scored"] == 19
    assert env["cutoffs"].loc[90, "n_not_scored"] == 1
    assert env["cutoffs"].loc[90, "bottom_cutoff"] == pytest.approx(2.8)
    assert env["cutoffs"].loc[90, "top_cutoff"] == pytest.approx(17.2)


@pytest.mark.parametrize("values, message", [
    ([1.0] * 20, "cutoffs coincide"),
    ([0.0] + [1.0] * 18 + [2.0], "cutoffs coincide"),
    ([np.nan] * 20, "no finite numeric risk scores"),
])
def test_unusable_landmark_stops_extraction(extraction, values, message):
    env, scores = extraction
    scores.loc[scores["landmark_days"] == 90, "risk_score"] = values
    scores.to_csv(env["SCORES_PATH"], index=False)
    with pytest.raises(ValueError, match=f"Landmark 90:.*{message}"):
        _run_cells(env, 4, 5)
    assert not env["OUTPUT_DIR"].exists()


def test_extract_writes_manifest_and_multiline_reports(extraction, capsys):
    env, scores = extraction
    scores.to_csv(env["SCORES_PATH"], index=False)
    ids = list(range(100001, 100021))
    local = pd.DataFrame({col: [0] * 20 for col in fet.LOCAL_COLUMNS})
    local["DFCI_MRN"] = ids
    local["TREATMENT_ANCHOR_DATE"] = "2020-01-01"
    for indicator, date in fet.ENDPOINT_SOURCES.values():
        local[indicator] = [0] * 19 + [1]
        local[date] = "2020-12-31"
    local.to_csv(env["DATA_PATH"], index=False)
    text = 'synthetic\ttext\nsecond line with "quotes" and café'
    pl.DataFrame({"DFCI_MRN": ids, "EVENT_DATE": ["2021-01-01"] * 20,
                  "RPT_ID": list(range(20)), "RPT_TEXT": [text] * 20}).write_parquet(env["PATHOLOGY_PATH"])
    _run_cells(env, 4, 5, 7, 9, 11)
    manifest = pd.read_csv(env["OUTPUT_DIR"] / "patient_manifest.csv")
    records = json.loads((env["OUTPUT_DIR"] / "pathology_reports.json").read_text(encoding="utf-8"))
    reports = pd.DataFrame(records)
    assert not (env["OUTPUT_DIR"] / "pathology_reports.tsv").exists()
    assert all(record["note_date"] == "2021-01-01T00:00:00.000" for record in records)
    assert all(record["platinum_event"] is False and record["days_adt_to_platinum"] is None
               for record in records if record["DFCI_MRN"] != 100020)
    assert set(manifest["DFCI_MRN"]) == {100001, 100002, 100019, 100020}
    assert len(reports) == manifest["n_pathology_notes"].sum() == 4
    assert reports["note_text"].eq(text).all()
    assert reports["note_days_from_adt_start"].eq(366).all()
    assert manifest.loc[~manifest["platinum_event"], "days_adt_to_platinum"].isna().all()
    output = capsys.readouterr().out
    assert "Manifest ready: 4 patients" in output
    assert text not in output
    assert not any(str(patient_id) in output for patient_id in ids)


IDS = list(range(100001, 100021))          # scores 0..19: bottom 100001-2, top 100019-20, middle the rest
TOP, BOTTOM = {100019, 100020}, {100001, 100002}


def _write_term_inputs(env, scores):
    """ADT start 2018-01-01 for all, so every dated report is post-landmark; reports separate the deciles from the middle."""
    ids, top, bottom = IDS, TOP, BOTTOM
    scores.to_csv(env["SCORES_PATH"], index=False)
    local = pd.DataFrame({col: [0] * 20 for col in fet.LOCAL_COLUMNS})
    local["DFCI_MRN"] = ids
    local["TREATMENT_ANCHOR_DATE"] = "2018-01-01"
    for indicator, date in fet.ENDPOINT_SOURCES.values():
        local[indicator] = 0
        local[date] = "2020-12-31"
    local.to_csv(env["DATA_PATH"], index=False)

    notes = []
    for pid in ids:
        notes.append((pid, "2019-06-01", "adenocarcinoma of the prostate"))   # post-landmark at every landmark
        if pid in top:
            notes.append((pid, "2019-06-01", "neuroendocrine features present"))  # separate report
        if pid in top:
            notes.append((pid, "2021-01-01", "chromogranin positive"))       # post-landmark at every landmark
        if pid in bottom:
            notes.append((pid, "2020-05-01", "ductal pattern"))              # post-landmark at every landmark
            notes.append((pid, None, "undated cribriform"))                  # undated: dropped
    pl.DataFrame({"DFCI_MRN": [n[0] for n in notes], "EVENT_DATE": [n[1] for n in notes],
                  "RPT_ID": list(range(len(notes))), "RPT_TEXT": [n[2] for n in notes]}
                 ).write_parquet(env["PATHOLOGY_PATH"])
    env.update(TERM_MIN_PATIENTS=2, TERM_NGRAMS=(1, 2), TERM_FDR=0.05, TERM_TOP_N=5,
               ADJUST_REPORT_BANDS=(1, 2, 4), SPECIMEN_HEADER_CHARS=300)
    return notes


def test_term_enrichment_contrasts_deciles_with_middle_on_post_landmark_reports(extraction, capsys):
    env, scores = extraction
    ids = IDS
    _write_term_inputs(env, scores)
    _run_analysis(env, 'risk-scores', 'reports', 'term-enrichment', 'term-display')

    all_res = pd.read_csv(env["OUTPUT_DIR"] / "pathology_term_enrichment.csv")
    assert set(zip(all_res["group"], all_res["control"])) == {
        ("top", "middle"), ("bottom", "middle"), ("top", "bottom"), ("bottom", "top")}
    assert (all_res["n_group"] == 2).all()
    vs_middle = all_res["control"].eq("middle")
    assert (all_res.loc[vs_middle, "n_control"] == 16).all() and (all_res.loc[~vs_middle, "n_control"] == 2).all()
    res = all_res[vs_middle]
    for lm in (0, 90, 180):
        t = res[(res["landmark"] == lm) & (res["group"] == "top")].set_index("term")
        assert t.loc["neuroendocrine", "pct_group"] == 1 and t.loc["neuroendocrine", "pct_control"] == 0
        assert t.loc["neuroendocrine", "p_value"] == pytest.approx(1 / 153)   # 1 / C(18, 2)
        assert t["p_value"].min() == t.loc["neuroendocrine", "p_value"]
        assert t.loc["prostate", "pct_control"] == 1
        # Bigrams stay within one report: "prostate neuroendocrine" would span two.
        assert "prostate neuroendocrine" not in t.index and "neuroendocrine features" in t.index
    # Post-landmark reports are used at every landmark; undated reports never are.
    for term, group in (("chromogranin", "top"), ("ductal", "bottom")):
        hit = res[res["term"].eq(term) & res["group"].eq(group)]
        assert set(hit["landmark"]) == {0, 90, 180} and hit["pct_group"].eq(1).all()
    assert "cribriform" not in set(all_res["term"])

    # Top vs bottom, one-sided in each direction: 2/2 vs 0/2 gives p = 1 / C(4, 2).
    def contrast(lm, group, control):
        return all_res[(all_res["landmark"] == lm) & (all_res["group"] == group)
                       & (all_res["control"] == control)].set_index("term")
    for lm in (0, 90, 180):
        assert contrast(lm, "top", "bottom").loc["neuroendocrine", "p_value"] == pytest.approx(1 / 6)
        assert contrast(lm, "bottom", "top").loc["neuroendocrine", "p_value"] == pytest.approx(1)
        assert contrast(lm, "top", "bottom").loc["prostate", "pct_control"] == 1
    for lm in (0, 90, 180):
        assert contrast(lm, "bottom", "top").loc["ductal", "p_value"] == pytest.approx(1 / 6)
    sizes = env["group_sizes"]
    assert sizes.loc[(0, "middle"), "patients_scored"] == 16
    for lm in (0, 90, 180):
        assert sizes.loc[(lm, "bottom"), "reports"] == 4 and sizes.loc[(lm, "top"), "reports"] == 6   # undated cribriform dropped

    output = capsys.readouterr().out
    assert "Landmark 0 · top decile vs middle" in output
    assert "Landmark 180 · bottom decile vs top decile" in output
    assert not any(str(patient_id) in output for patient_id in ids)
    assert "neuroendocrine features present" not in output


def test_manifest_flags_are_false_for_middle_and_absent_landmarks(extraction):
    env, scores = extraction
    patient_id = IDS[0]
    scores.loc[(scores[fs.ID_COL] == patient_id) & (scores["landmark_days"] == 90), "risk_score"] = 10.0
    scores = scores[~((scores[fs.ID_COL] == patient_id) & (scores["landmark_days"] == 180))]
    _write_term_inputs(env, scores)
    _run_cells(env, 4, 5, 7, 9, 11)

    manifest = pd.read_csv(env["OUTPUT_DIR"] / "patient_manifest.csv").set_index("DFCI_MRN")
    patient = manifest.loc[patient_id]
    assert patient["bottom_decile_lm0"] and patient["risk_group"] == "bottom"
    assert pd.isna(patient["decile_group_lm90"]) and pd.notna(patient["risk_score_lm90"])
    assert patient["decile_group_lm180"] == "not scored" and pd.isna(patient["risk_score_lm180"])
    flags = [f"{group}_decile_lm{lm}" for group in ("top", "bottom") for lm in (90, 180)]
    assert patient[flags].notna().all()
    assert patient[flags].eq(False).all()


def test_term_enrichment_handles_empty_patient_vocabulary(extraction, capsys):
    env, scores = extraction
    _write_term_inputs(env, scores)
    env["TERM_MIN_PATIENTS"] = 5
    # Five reports pass min_df, but one distinct patient cannot pass the patient threshold.
    pl.DataFrame({"DFCI_MRN": [IDS[0]] * 5, "EVENT_DATE": ["2019-06-01"] * 5,
                  "RPT_TEXT": ["adenocarcinoma prostate"] * 5}).write_parquet(env["PATHOLOGY_PATH"])
    _run_analysis(env, 'risk-scores', 'reports', 'term-enrichment', 'term-display')

    result = pd.read_csv(env["OUTPUT_DIR"] / "pathology_term_enrichment.csv")
    assert result.empty and list(result.columns) == env["ENRICHMENT_COLS"]
    assert "Landmark 180 · top decile vs middle: no terms" in capsys.readouterr().out


TOPIC_MODULES = ("zstandard", "bertopic", "umap", "hdbscan")
TERM_CELLS = ("risk-scores", "reports", "term-enrichment", "term-display")
TOPIC_CELLS = ("topic-fit", "topic-enrichment")


def test_bertopic_section_skips_cleanly_without_packages(extraction, capsys, monkeypatch):
    env, scores = extraction
    _write_term_inputs(env, scores)
    for name in TOPIC_MODULES:
        monkeypatch.setitem(sys.modules, name, None)      # import raises ImportError
    env.update(EMBED_METADATA_PATH=Path("/nonexistent/meta.parquet"), EMBED_ARRAY_PATH=Path("/nonexistent/a.npy.zst"),
               TOPIC_MIN_SIZE=2, TOPIC_SEED=0)
    _run_analysis(env, *TERM_CELLS, *TOPIC_CELLS)
    out = capsys.readouterr().out
    assert "BERTopic section skipped (packages not installed: zstandard, bertopic, umap-learn, hdbscan)" in out
    assert (env["OUTPUT_DIR"] / "pathology_term_enrichment.csv").exists()
    assert not (env["OUTPUT_DIR"] / "pathology_topics.csv").exists()
    assert not (env["OUTPUT_DIR"] / "pathology_topic_enrichment.csv").exists()


class _ShortReads:
    """Uncompressed stand-in for zstandard's stream reader that returns at most 7 bytes per read."""

    def __init__(self, fh):
        self.fh = fh

    def read(self, n=-1):
        return self.fh.read(7 if n is None or n < 0 else min(n, 7))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeBERTopic:
    """Topic = argmax of the embedding row; an all-zero row is an outlier (-1)."""
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, documents, embeddings=None):
        assert self.kwargs["embedding_model"] is None and embeddings is not None
        # Real BERTopic fits its class TF-IDF on text with [^A-Za-z0-9 ] deleted, then scores representative
        # documents on the text as passed: a vocabulary term only in the latter has an infinite IDF.
        vec = self.kwargs["vectorizer_model"]
        cleaned = [re.sub(r"[^A-Za-z0-9 ]+", "", d.replace("\n", " ").replace("\t", " ")) for d in documents]
        fitted = np.asarray(vec.fit_transform(cleaned).sum(axis=0)).ravel()
        raw = np.asarray(vec.transform(list(documents)).sum(axis=0)).ravel()
        if np.any((raw > 0) & (fitted == 0)):
            raise ValueError("Input contains infinity or a value too large for dtype('float64').")
        _FakeBERTopic.calls.append((list(documents), embeddings, self.kwargs))
        self.topics = [int(np.argmax(e)) if np.any(e) else -1 for e in embeddings]
        self.docs = list(documents)
        return self.topics, None

    def get_topic_info(self):
        ts = sorted(set(self.topics))
        return pd.DataFrame({"Topic": ts, "Count": [self.topics.count(t) for t in ts], "Name": [f"{t}_x" for t in ts],
                             "Representative_Docs": [[d for d, k in zip(self.docs, self.topics) if k == t] for t in ts]})

    def get_topic(self, t):
        return [({0: "neuroendocrine", 1: "ductal", 2: "adenocarcinoma"}.get(t, "other"), 1.0)]


def _fake_topic_modules(monkeypatch):
    mods = {name: types.ModuleType(name) for name in TOPIC_MODULES}
    mods["zstandard"].ZstdDecompressor = lambda: types.SimpleNamespace(stream_reader=_ShortReads)
    mods["bertopic"].BERTopic = _FakeBERTopic
    mods["umap"].UMAP = lambda **kw: ("umap", kw)
    mods["hdbscan"].HDBSCAN = lambda **kw: ("hdbscan", kw)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _FakeBERTopic.calls.clear()
    return mods


def test_bertopic_enriches_topics_from_precomputed_embeddings_without_leaking_text(extraction, capsys, monkeypatch,
                                                                                   tmp_path):
    env, scores = extraction
    notes = _write_term_inputs(env, scores)
    # Hyphenated text: BERTopic's own cleaning would merge it ("neuroendocrinefeatures") and fail the fit.
    path = pd.read_parquet(env["PATHOLOGY_PATH"])
    path["RPT_TEXT"] = path["RPT_TEXT"].str.replace("neuroendocrine features", "neuroendocrine-features")
    pl.from_pandas(path).write_parquet(env["PATHOLOGY_PATH"])
    mods = _fake_topic_modules(monkeypatch)

    # One embedding per pathology report (RPT_ID = note position), in reverse order, with an imaging note that
    # shares every RPT_ID interleaved; the imaging rows point at the "neuroendocrine" axis and must be ignored.
    def axis(rpt_id, text, pid):
        if "neuroendocrine" in text:
            return [1, 0, 0]
        if "ductal" in text:
            return [0, 1, 0]
        return [0, 0, 0] if pid == 100005 else [0, 0, 1]      # 100005's only report is an outlier
    rows, meta = [], []
    for rpt_id in reversed(range(len(notes))):
        pid, _, text = notes[rpt_id]
        meta.append({"DFCI_MRN": pid, "RPT_ID": float(rpt_id), "NOTE_TYPE": "Pathology", "EMBEDDING_INDEX": len(rows)})
        rows.append(axis(rpt_id, text, pid))
        meta.append({"DFCI_MRN": pid, "RPT_ID": float(rpt_id), "NOTE_TYPE": "Imaging", "EMBEDDING_INDEX": len(rows)})
        rows.append([1, 0, 0])
    array_path = tmp_path / "emb.npy"
    np.save(array_path, np.asarray(rows, dtype=np.float16))
    pl.DataFrame(meta).write_parquet(tmp_path / "meta.parquet")
    env.update(EMBED_METADATA_PATH=tmp_path / "meta.parquet", EMBED_ARRAY_PATH=array_path,
               TOPIC_MIN_SIZE=2, TOPIC_SEED=0)
    _run_analysis(env, *TERM_CELLS, *TOPIC_CELLS)

    (docs, embeddings, kwargs), = _FakeBERTopic.calls
    assert len(docs) == 26 and embeddings.dtype == np.float32      # every dated (post-landmark) report
    assert sum("chromogranin" in d for d in docs) == 2 and sum("cribriform" in d for d in docs) == 0
    assert sum(e[0] == 1 for e in embeddings) == 2                  # no imaging row leaked in
    vocab = kwargs["vectorizer_model"].vocabulary
    assert "neuroendocrine" in vocab and "prostate" in vocab and kwargs["umap_model"][1]["random_state"] == 0

    topics = pd.read_csv(env["OUTPUT_DIR"] / "pathology_topics.csv")
    assert list(topics.columns) == ["topic", "n_reports", "n_patients", "top_words"]
    assert topics.set_index("topic").loc[-1, "n_reports"] == 1
    res = pd.read_csv(env["OUTPUT_DIR"] / "pathology_topic_enrichment.csv")
    assert not any("Representative" in c for c in topics.columns.tolist() + res.columns.tolist())
    vs_middle = res["control"].eq("middle")
    assert (res.loc[vs_middle, "n_control"] == 16).all()   # outlier patient stays in the denominator
    assert (res["n_group"] == 2).all() and (res.loc[~vs_middle, "n_control"] == 2).all()
    for lm in (0, 90, 180):
        tvb = res[(res["landmark"] == lm) & (res["group"] == "top") & (res["control"] == "bottom")].set_index("topic")
        assert tvb.loc[0, "p_value"] == pytest.approx(1 / 6)
        top = res[(res["landmark"] == lm) & (res["group"] == "top") & vs_middle].set_index("topic")
        assert top.loc[0, "top_words"] == "neuroendocrine" and top.loc[0, "pct_group"] == 1
        assert top.loc[0, "p_value"] == pytest.approx(1 / 153)
        assert top.loc[2, "pct_control"] == pytest.approx(15 / 16)
    assert set(res.loc[res["topic"].eq(1), "landmark"]) == {0, 90, 180}   # post-landmark ductal reports count
    assert -1 not in set(res["topic"])

    out = capsys.readouterr().out
    assert "26 of 26 pathology reports have a precomputed embedding" in out
    assert "Landmark 0 · top decile vs middle (topics)" in out
    assert "Landmark 0 · top decile vs bottom decile (topics)" in out
    assert not any(str(patient_id) in out for patient_id in IDS)
    # Whole reports never print; two-word reports such as "ductal pattern" are also legitimate aggregate bigrams.
    assert [n[2] for n in notes if len(n[2].split()) > 2 and n[2] in out] == []

    # The streaming reader keeps exactly the requested rows across chunk boundaries and short reads.
    want = np.array([0, 3, 4, 9, len(rows) - 1])
    got = env["read_embedding_rows"](array_path, want, mods["zstandard"], chunk_rows=2)
    np.testing.assert_array_equal(got, np.load(array_path)[want].astype(np.float32))


# --- Sensitivity and complementary analyses ---
EXTRA_CELLS = ("lexicon", "score-association")


def _no_topic_modules(env, monkeypatch):
    for name in TOPIC_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    env.update(EMBED_METADATA_PATH=Path("/nonexistent/meta.parquet"), EMBED_ARRAY_PATH=Path("/nonexistent/a.npy.zst"),
               TOPIC_MIN_SIZE=2, TOPIC_SEED=0)


def _write_notes(env, notes):
    pl.DataFrame({"DFCI_MRN": [n[0] for n in notes], "EVENT_DATE": [n[1] for n in notes],
                  "RPT_ID": list(range(len(notes))), "RPT_TEXT": [n[2] for n in notes]}
                 ).write_parquet(env["PATHOLOGY_PATH"])


def _write_extra_inputs(env, scores):
    """The term inputs plus a Gleason 9 report for both top patients and a "non-small cell" mention in a middle one.

    Two middle patients get three more benign reports, so they share the top patients' 4+ report stratum.
    """
    notes = _write_term_inputs(env, scores)
    notes += [(pid, "2019-06-01", "Gleason score 4+5=9 (grade group 5)") for pid in sorted(TOP)]
    notes += [(pid, "2019-06-01", "benign prostatic tissue") for pid in (100006, 100007) for _ in range(3)]
    notes += [(100005, "2019-06-01", "non-small cell carcinoma, lung")]
    _write_notes(env, notes)
    return notes


def test_lexicon_enrichment_uses_post_landmark_reports(extraction, capsys, monkeypatch):
    env, scores = extraction
    _write_extra_inputs(env, scores)
    _no_topic_modules(env, monkeypatch)
    _run_analysis(env, *TERM_CELLS, *TOPIC_CELLS, 'lexicon')

    hits = env["lexicon_hits"]("Negative for synaptophysin, chromogranin positive.")
    assert hits == {"NE marker (any)": True, "synaptophysin": True, "chromogranin": True}
    assert "small cell" not in env["lexicon_hits"]("non-small cell carcinoma")
    assert env["lexicon_hits"]("Gleason 3+4=7, grade group 4")["gleason_max"] == 8

    lex = pd.read_csv(env["OUTPUT_DIR"] / "pathology_lexicon_enrichment.csv")
    assert list(lex.columns) == env["LEXICON_COLS"]
    assert not lex["feature"].str.contains("negated").any()
    for lm in (0, 90, 180):
        x = lex[(lex["landmark"] == lm) & (lex["group"] == "top") & (lex["control"] == "middle")].set_index("feature")
        assert x.loc["NE term (any)", "group_with_feature"] == 2 and x.loc["NE term (any)", "control_with_feature"] == 0
        assert x.loc["chromogranin", "group_with_feature"] == 2       # report dated after every landmark
        assert x.loc["Gleason max 8-10", "group_with_feature"] == 2
        assert x.loc["Gleason max 8-10", "p_value"] == pytest.approx(1 / 153)
        assert "Gleason max 7" not in x.index and "sarcomatoid" not in x.index      # unseen features are not tested
        assert "small cell" not in x.index                                      # "non-small cell" is not a mention
        b = lex[(lex["landmark"] == lm) & (lex["group"] == "bottom") & (lex["control"] == "middle")].set_index("feature")
        assert "cribriform" not in b.index and b.loc["ductal", "group_with_feature"] == 2   # undated: dropped
    assert "Landmark 0 · curated lexicon" in capsys.readouterr().out


def test_mantel_haenszel_adjusts_for_strata(extraction):
    env, _ = extraction
    env.update(TERM_NGRAMS=(1, 2), TERM_MIN_PATIENTS=2)
    for i in ("reports", "term-enrichment"):
        src = _analysis_source(i)
        cut = src.index("# Per-landmark group") if i == "reports" else src.index("term_matrices = {}")
        exec(compile(src[:cut], f"cell_{i}_defs", "exec"), env)
    from scipy import sparse

    def run(present, label, strata):
        res = env["fisher_enrichment"](sparse.csr_matrix(np.array(present).reshape(-1, 1)), np.array(label),
                                       np.array(["f"], dtype=object), 0, None if strata is None else np.array(strata))
        return res[(res["group"] == "top") & (res["control"] == "middle")].iloc[0]

    # One stratum: MH odds ratio = the Haldane-corrected odds ratio (zero cell), p from the continuity-corrected MH.
    one = run([1, 1, 0, 0, 0, 0], ["top", "top", "middle", "middle", "middle", "middle"], [0] * 6)
    assert one["odds_ratio_mh"] == pytest.approx(one["odds_ratio"])
    # Confounding: within each stratum the feature rate is the same in both groups (8/16 vs 2/4, then 0/4 vs 0/16),
    # but top sits mostly in the stratum where the feature is common, so the pooled Fisher test is significant.
    label = ["top"] * 16 + ["middle"] * 4 + ["top"] * 4 + ["middle"] * 16
    strata = [0] * 20 + [1] * 20
    present = [1] * 8 + [0] * 8 + [1] * 2 + [0] * 2 + [0] * 20
    res = run(present, label, strata)
    assert res["p_value"] < 0.05
    assert res["p_value_mh"] > 0.5 and res["odds_ratio_mh"] < res["odds_ratio"] / 3
    # Strata that never contain both groups carry no information.
    confounded = run([1, 1, 0, 0], ["top", "top", "middle", "middle"], [0, 0, 1, 1])
    assert np.isnan(confounded["odds_ratio_mh"]) and np.isnan(confounded["p_value_mh"])
    assert run([1, 1, 0, 0], ["top", "top", "middle", "middle"], None)[["odds_ratio_mh", "p_value_mh"]].isna().all()


def test_score_association(extraction, capsys, monkeypatch):
    env, scores = extraction
    notes = _write_extra_inputs(env, scores)
    _no_topic_modules(env, monkeypatch)
    _run_analysis(env, *TERM_CELLS, *TOPIC_CELLS, *EXTRA_CELLS)

    assoc = pd.read_csv(env["OUTPUT_DIR"] / "pathology_score_association.csv")
    assert list(assoc.columns) == env["ASSOC_COLS"]
    for lm in (0, 90, 180):
        a = assoc[(assoc["landmark"] == lm) & (assoc["feature_set"] == "term")].set_index("feature")
        assert a.loc["neuroendocrine", "auc"] == 1 and a.loc["neuroendocrine", "p_value"] < 0.05
        assert a.loc["neuroendocrine", "n_patients"] == 20
        x = assoc[(assoc["landmark"] == lm) & (assoc["feature_set"] == "lexicon")].set_index("feature")
        assert x.loc["Gleason max 8-10", "auc"] == 1 and np.isfinite(x.loc["Gleason max 8-10", "auc_adj"])
        assert a.loc["ductal", "auc"] == 0 and a.loc["ductal", "mean_percentile_with"] < 0.1
    assert ((assoc["n_with_feature"] > 0) & (assoc["n_with_feature"] < assoc["n_patients"])).all()

    assert not (env["OUTPUT_DIR"] / "pathology_text_classifier.csv").exists()   # classifier removed

    out = capsys.readouterr().out
    assert not any(str(patient_id) in out for patient_id in IDS)
    assert [n[2] for n in notes if len(n[2].split()) > 2 and n[2] in out] == []


# Continuous topic models use raw scores and keep patients (not notes) as independent units.
def _continuous_definitions():
    import importlib
    env = dict(np=np, pd=pd, optional_import=importlib.import_module, CONTINUOUS_MIN_PATIENTS=30,
               CONTINUOUS_MIN_TOPIC_PATIENTS=5, TOPIC_PERMUTATIONS=99, TOPIC_SEED=42)
    src = _analysis_source("continuous-topics")
    exec(compile(src[:src.index("continuous_frames, curve_frames")], "continuous_defs", "exec"), env)
    return env


def _continuous_inputs():
    rng = np.random.default_rng(71)
    risk = pd.Series(np.linspace(-2, 2, 160), index=pd.Index(range(160), name="DFCI_MRN"))
    # Repeated patient outcomes intentionally induce strong within-patient correlation.
    topic = np.where(rng.random(len(risk)) < 1 / (1 + np.exp(-risk.to_numpy())), 0, 1)
    docs = pd.DataFrame({"DFCI_MRN": risk.index, "topic": topic, "specimen_site": "prostate"})
    docs = docs.loc[docs.index.repeat(3)].reset_index(drop=True)
    return docs, risk


def test_continuous_topics_cluster_by_patient_and_retain_raw_score_scale():
    pytest.importorskip("statsmodels.api")
    env = _continuous_definitions()
    docs, risk = _continuous_inputs()
    run = env["continuous_topic_association"]
    result, curves = run(docs, risk, 0, {0: "synthetic positive", 1: "synthetic negative"})
    assert result["status"].eq("ok").all()
    gee = result[result["model"].eq("gee")].set_index("topic")
    assert gee.loc[0, "odds_ratio_per_sd"] > 1 and gee.loc[0, "p_value"] < 0.01
    assert gee.loc[1, "odds_ratio_per_sd"] < 1
    assert gee["risk_sd"].eq(risk.std(ddof=0)).all()
    assert curves["probability"].between(0, 1).all()
    assert (curves["ci95_low"] <= curves["probability"]).all()
    assert (curves["ci95_high"] >= curves["probability"]).all()
    # Uniform note duplication cannot create new independent patients or tighten clustered intervals.
    repeated, _ = run(pd.concat([docs] * 5, ignore_index=True), risk, 0, {})
    for col in ("beta_per_sd", "ci95_low", "ci95_high", "spearman_rho", "p_value"):
        np.testing.assert_allclose(result[col], repeated[col], rtol=1e-7, atol=1e-10, equal_nan=True)
    reversed_result, _ = run(docs, -risk, 0, {})
    np.testing.assert_allclose(result["beta_per_sd"], -reversed_result["beta_per_sd"], atol=1e-10, equal_nan=True)
    # Scaling raw risk retains the per-SD OR; no decile or percentile inputs enter the model.
    scaled, scaled_curves = run(docs, 10 + 3 * risk, 0, {})
    np.testing.assert_allclose(result["odds_ratio_per_sd"], scaled["odds_ratio_per_sd"], equal_nan=True)
    np.testing.assert_allclose(scaled_curves["risk_score"], 10 + 3 * curves["risk_score"])


def test_continuous_topics_exclude_invalid_scores_keep_outliers_and_handle_constant_risk():
    pytest.importorskip("statsmodels.api")
    env = _continuous_definitions()
    docs, risk = _continuous_inputs()
    docs.loc[docs["DFCI_MRN"].eq(0), "topic"] = -1
    risk.loc[1] = np.inf
    risk.loc[2] = np.nan
    result, _ = env["continuous_topic_association"](docs, risk, 90, {})
    assert result["n_patients"].eq(158).all()  # outlier-only patient retained, two invalid scores excluded
    assert result["n_reports"].eq(474).all()
    assert -1 not in result["topic"].values
    constant, curves = env["continuous_topic_association"](docs, risk * 0 + 1, 90, {})
    assert constant["status"].str.contains("constant risk").all()
    assert constant["p_value"].isna().all() and curves.empty
    empty, curves = env["continuous_topic_association"](docs.iloc[:0], risk, 90, {})
    assert empty.empty and list(empty.columns) == env["CONTINUOUS_COLS"] and curves.empty


def test_continuous_patient_fraction_runs_without_statsmodels():
    env = _continuous_definitions()
    env["optional_import"] = lambda name: None
    docs, risk = _continuous_inputs()
    result, curves = env["continuous_topic_association"](docs, risk, 0, {})
    assert result.loc[result["model"].str.startswith("gee"), "status"].str.contains("statsmodels").all()
    fraction = result[result["model"].eq("patient_fraction_spearman")].set_index("topic")
    assert fraction["status"].eq("ok").all() and fraction.loc[0, "spearman_rho"] > 0
    assert fraction["p_value"].between(0, 1).all() and curves.empty


def test_analysis_runs_standalone_without_export_or_local_endpoint_data(extraction, monkeypatch):
    env, scores = extraction
    _write_term_inputs(env, scores)
    _no_topic_modules(env, monkeypatch)
    # The exploratory notebooks need scores, notes and ADT anchors, not the extraction's endpoint table.
    pd.read_csv(env["DATA_PATH"])[["DFCI_MRN", "TREATMENT_ANCHOR_DATE"]].to_csv(env["DATA_PATH"], index=False)
    env.update(CONTINUOUS_MIN_PATIENTS=30, CONTINUOUS_MIN_TOPIC_PATIENTS=5, TOPIC_PERMUTATIONS=99)
    for notebook in (ANALYSIS_NOTEBOOK, TOPIC_NOTEBOOK):
        _run_analysis(env, *_code_cell_ids(notebook), notebook=notebook)
    assert (env["OUTPUT_DIR"] / "pathology_term_enrichment.csv").exists()
    assert not (env["OUTPUT_DIR"] / "patient_manifest.csv").exists()
    assert not (env["OUTPUT_DIR"] / "pathology_reports.json").exists()
    assert not (env["OUTPUT_DIR"] / "pathology_reports.tsv").exists()


def test_continuous_topics_write_aggregate_results_with_global_fdr(extraction, monkeypatch):
    pytest.importorskip("statsmodels.api")
    import importlib
    from scipy.stats import false_discovery_control
    env, _ = extraction
    docs, risk = _continuous_inputs()
    docs["rpt_key"] = np.arange(len(docs)).astype(str)
    docs["RPT_ID"] = docs["rpt_key"]
    env.update(topic_docs=docs, topic_table=pd.DataFrame({"topic": [0, 1], "top_words": ["positive", "negative"]}),
               by_lm={lm: pd.DataFrame({fs.ID_COL: risk.index, "risk_score": risk.values}) for lm in env["LANDMARKS"]},
               reports_by_lm={lm: docs for lm in env["LANDMARKS"]},
               optional_import=importlib.import_module, pathology_keys=lambda frame: frame["rpt_key"],
               CONTINUOUS_MIN_PATIENTS=30, CONTINUOUS_MIN_TOPIC_PATIENTS=5, TOPIC_PERMUTATIONS=99,
               TOPIC_SEED=42, TERM_FDR=0.05, TERM_TOP_N=5, display=lambda *args: None)
    # Reuse the existing tested BH implementation without running term enrichment.
    source = _analysis_source("term-enrichment")
    exec(source[:source.index("term_matrices = {}")], env)
    _run_analysis(env, "continuous-topics")
    result = pd.read_csv(env["OUTPUT_DIR"] / "pathology_topic_continuous_association.csv")
    assert len(result) == 18 and set(result["landmark"]) == {0, 90, 180}
    for _, sub in result.groupby("model"):
        np.testing.assert_allclose(sub["q_value"], false_discovery_control(sub["p_value"]))
    for filename in ("pathology_topic_continuous_association.csv", "pathology_topic_continuous_coverage.csv",
                     "pathology_topic_risk_curves.csv"):
        table = pd.read_csv(env["OUTPUT_DIR"] / filename)
        assert not {"DFCI_MRN", "RPT_ID", "RPT_TEXT", "Representative_Docs"}.intersection(table.columns)


def test_analysis_does_not_require_distinct_decile_cutoffs(extraction):
    env, scores = extraction
    scores.loc[:, "risk_score"] = 1.0
    scores.to_csv(env["SCORES_PATH"], index=False)
    _run_analysis(env, "risk-scores")
    assert all(g["risk_score"].notna().all() for g in env["by_lm"].values())
    assert all(g["decile_group"].isna().all() for g in env["by_lm"].values())


CROSS_NOTEBOOK = ANALYSIS_DIR / "14_clinical_notes_topics_vs_risk.ipynb"
CROSS_TOPIC_NOTEBOOK = ANALYSIS_DIR / "16_clinical_notes_bertopic_vs_risk.ipynb"   # BERTopic, split out of 14


def _cross_source(cell_id):
    return _cell_source((CROSS_NOTEBOOK, CROSS_TOPIC_NOTEBOOK), cell_id)[1]


def _run_cross(env, *cell_ids, notebook=None):
    for cell_id in cell_ids:
        path, src = _cell_source((notebook,) if notebook else (CROSS_NOTEBOOK, CROSS_TOPIC_NOTEBOOK), cell_id)
        exec(compile(src, f"{path}:{cell_id}", "exec"), env)


def _run_cross_notebooks(env):
    """Both cross-note notebooks end to end (configuration comes from the test), terms first."""
    for notebook in (CROSS_NOTEBOOK, CROSS_TOPIC_NOTEBOOK):
        _run_cross(env, *_code_cell_ids(notebook), notebook=notebook)


def _write_cross_inputs(env, scores, tmp_path):
    scores.to_csv(env["SCORES_PATH"], index=False)
    pd.DataFrame({"DFCI_MRN": IDS, "TREATMENT_ANCHOR_DATE": "2020-01-01"}).to_csv(env["DATA_PATH"], index=False)
    kinds = ("Clinician", "Imaging", "Pathology")
    env.update(Path=Path, NOTE_TYPES=kinds, NOTE_PATHS={},
               TERM_MIN_PATIENTS=2, TERM_NGRAMS=(1, 2), TERM_FDR=0.05, TERM_TOP_N=3,
               ADJUST_REPORT_BANDS=(1, 2, 4), TOPIC_MIN_SIZE=2, TOPIC_SEED=0,
               CONTINUOUS_MIN_PATIENTS=10, CONTINUOUS_MIN_TOPIC_PATIENTS=2, TOPIC_PERMUTATIONS=19)
    rows, meta = [], []
    for k, kind in enumerate(kinds):
        records = []
        for i, pid in enumerate(IDS):
            # Same report ID across patients AND across note types: only the full identity is unique.
            topic = int((i + k) % 4 == 0 or i < 10)
            records.append(dict(DFCI_MRN=pid, RPT_ID=7, EVENT_DATE="2021-01-01",   # post-landmark at every landmark
                                RPT_TEXT=("ductal pattern" if topic else "neuroendocrine features present")))
            # One missing embedding in each of two types; never treat them as topic-negative notes.
            if (kind, i) in (("Imaging", 0), ("Clinician", 19)):
                continue
            meta.append(dict(DFCI_MRN=float(pid), NOTE_TYPE=kind, RPT_ID=7.0, EMBEDDING_INDEX=len(rows)))
            rows.append([0, 1, 0] if topic else [1, 0, 0])
        path = tmp_path / f"{kind}.parquet"
        pl.DataFrame(records).write_parquet(path)
        env["NOTE_PATHS"][kind] = path
    # One irrelevant type must not join, even when patient and report IDs match.
    meta.append(dict(DFCI_MRN=float(IDS[0]), NOTE_TYPE="Other", RPT_ID=7.0, EMBEDDING_INDEX=len(rows)))
    rows.append([0, 0, 1])
    np.save(tmp_path / "cross_emb.npy", np.asarray(rows, dtype=np.float32))
    pl.DataFrame(meta).write_parquet(tmp_path / "cross_meta.parquet")
    env.update(EMBED_ARRAY_PATH=tmp_path / "cross_emb.npy", EMBED_METADATA_PATH=tmp_path / "cross_meta.parquet")


def test_cross_note_full_analysis_joins_by_type_patient_and_report(extraction, monkeypatch, tmp_path, capsys):
    env, scores = extraction
    _write_cross_inputs(env, scores, tmp_path)
    _fake_topic_modules(monkeypatch)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    _run_cross_notebooks(env)
    assert len(_FakeBERTopic.calls) == 1  # one shared topic vocabulary, no per-type refits
    assert len(env["all_notes"]) == 60 and env["all_notes"]["rpt_key"].nunique() == 60
    assert len(env["embed_docs"]) == 58 and set(env["embed_docs"]["NOTE_TYPE"]) == set(env["NOTE_TYPES"])
    for row, embedding in zip(env["embed_docs"].itertuples(), env["report_embeddings"]):
        assert np.argmax(embedding) == int("ductal" in row.RPT_TEXT)
    composition = pd.read_csv(env["OUTPUT_DIR"] / "clinical_notes_topic_by_note_type.csv")
    assert composition.groupby("note_type")["n_reports"].sum().to_dict() == {"Clinician": 19, "Imaging": 19, "Pathology": 20}
    np.testing.assert_allclose(composition.groupby("note_type")["fraction_of_type_notes"].sum(), 1)
    result = pd.read_csv(env["OUTPUT_DIR"] / "clinical_notes_topic_continuous_association.csv")
    assert set(result["note_type"]) == {"All", *env["NOTE_TYPES"]}
    assert result.loc[result["note_type"].eq("All"), "n_patients"].eq(20).all()
    assert result.loc[result["note_type"].eq("Imaging"), "n_patients"].eq(19).all()
    assert result.loc[result["note_type"].eq("Clinician"), "n_reports"].eq(19).all()
    for _, sub in result.groupby("model"):
        np.testing.assert_allclose(sub["q_value"], env["bh_qvalues"](sub["p_value"]), equal_nan=True)
    coverage = pd.read_csv(env["OUTPUT_DIR"] / "clinical_notes_topic_continuous_coverage.csv")
    assert len(coverage) == 12
    assert coverage["patients_with_report"].eq(20).all()
    enrichment = pd.read_csv(env["OUTPUT_DIR"] / "clinical_notes_topic_enrichment.csv")
    assert set(enrichment["note_type"]) == {"All", *env["NOTE_TYPES"]}
    for _, sub in enrichment.groupby(["group", "control"]):
        np.testing.assert_allclose(sub["q_value"], env["bh_qvalues"](sub["p_value"]), equal_nan=True)
    for name in ("term_enrichment", "lexicon_enrichment", "score_association"):
        assert (env["OUTPUT_DIR"] / f"clinical_notes_{name}.csv").exists()
    assert not (env["OUTPUT_DIR"] / "clinical_notes_text_classifier.csv").exists()   # classifier removed
    for path in env["OUTPUT_DIR"].glob("*.csv"):
        columns = pd.read_csv(path).columns
        assert not {"DFCI_MRN", "RPT_TEXT", "RPT_ID", "rpt_key", "Representative_Docs"}.intersection(columns)
    assert not any(str(pid) in capsys.readouterr().out for pid in IDS)


def test_cross_note_runs_without_topic_packages(extraction, monkeypatch, tmp_path):
    env, scores = extraction
    _write_cross_inputs(env, scores, tmp_path)
    for name in TOPIC_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    _run_cross_notebooks(env)
    assert len(env["all_notes"]) == 60
    assert (env["OUTPUT_DIR"] / "clinical_notes_score_association.csv").exists()
    assert not (env["OUTPUT_DIR"] / "clinical_notes_topics.csv").exists()


def test_cross_note_missing_selected_source_and_ambiguous_embeddings_stop(extraction, monkeypatch, tmp_path):
    env, scores = extraction
    _write_cross_inputs(env, scores, tmp_path)
    _fake_topic_modules(monkeypatch)
    path = env["NOTE_PATHS"]["Imaging"]
    env["NOTE_PATHS"]["Imaging"] = tmp_path / "missing.parquet"
    _run_cross(env, "risk-scores")
    with pytest.raises(FileNotFoundError, match="Selected Imaging notes not found"):
        _run_cross(env, "reports")
    env["NOTE_PATHS"]["Imaging"] = path
    _run_cross(env, "reports", "term-enrichment")
    metadata = pl.read_parquet(env["EMBED_METADATA_PATH"])
    conflict = metadata.head(1).with_columns(pl.lit(999).cast(pl.Int64).alias("EMBEDDING_INDEX"))
    pl.concat([metadata, conflict]).write_parquet(env["EMBED_METADATA_PATH"])
    with pytest.raises(ValueError, match="Ambiguous embedding metadata"):
        _run_cross(env, "topic-fit")


def test_cross_note_adjusted_gee_includes_note_type(extraction, tmp_path, monkeypatch):
    sm = pytest.importorskip("statsmodels.api")
    env, scores = extraction
    _write_cross_inputs(env, scores, tmp_path)
    _run_cross(env, "risk-scores", "reports")
    import importlib
    env["optional_import"] = importlib.import_module
    source = _cross_source("continuous-topics")
    exec(source[:source.index("continuous_frames, curve_frames")], env)
    docs, risk = _continuous_inputs()
    # A different note mix by patient, with each type represented; no specimen fields required.
    docs["NOTE_TYPE"] = ["Clinician", "Imaging", "Pathology"] * len(risk)
    docs = docs.drop(columns="specimen_site")
    seen = []
    gee = sm.GEE
    def record_design(endog, exog, **kwargs):
        seen.append((list(exog.columns), kwargs["groups"].copy()))
        return gee(endog, exog, **kwargs)
    monkeypatch.setattr(sm, "GEE", record_design)
    result, _ = env["continuous_topic_association"](docs, risk, 0, {})
    assert result["status"].eq("ok").all()
    assert any("note_type_Imaging" in columns and "note_type_Pathology" in columns for columns, _ in seen)
    assert all(len(np.unique(groups)) == 160 for _, groups in seen)
    seen.clear()
    env["continuous_topic_association"](docs[docs["NOTE_TYPE"].eq("Imaging")], risk, 0, {})
    assert all(not any(name.startswith("note_type_") for name in columns) for columns, _ in seen)


def _write_pathology_topic_inputs(env, scores, tmp_path):
    _write_cross_inputs(env, scores, tmp_path)
    env.update(PATHOLOGY_PATH=env["NOTE_PATHS"]["Pathology"], SPECIMEN_HEADER_CHARS=300)
    notes = pl.read_parquet(env["PATHOLOGY_PATH"])
    # All patients share RPT_ID=7, but one patient's report stops being eligible at +180.
    notes = notes.with_columns(pl.when(pl.col("DFCI_MRN") == IDS[0]).then(pl.lit("2020-04-30"))
                              .otherwise(pl.col("EVENT_DATE")).alias("EVENT_DATE"))
    notes.write_parquet(env["PATHOLOGY_PATH"])


def test_pathology_topics_keep_patient_identity_across_landmarks(extraction, monkeypatch, tmp_path):
    env, scores = extraction
    _write_pathology_topic_inputs(env, scores, tmp_path)
    _fake_topic_modules(monkeypatch)
    _run_analysis(env, *_code_cell_ids(TOPIC_NOTEBOOK), notebook=TOPIC_NOTEBOOK)
    assert len(env["embed_docs"]) == 20
    assert env["embed_docs"]["rpt_key"].nunique() == 20
    for row, embedding in zip(env["embed_docs"].itertuples(), env["report_embeddings"]):
        assert np.argmax(embedding) == int("ductal" in row.RPT_TEXT)
    coverage = pd.read_csv(env["OUTPUT_DIR"] / "pathology_topic_continuous_coverage.csv")
    assert coverage["patients_with_embedded_report"].tolist() == [20, 20, 19]
    enrichment = pd.read_csv(env["OUTPUT_DIR"] / "pathology_topic_enrichment.csv")
    bottom = enrichment[enrichment["group"].eq("bottom") & enrichment["control"].eq("middle")]
    assert bottom.groupby("landmark")["n_group"].first().to_dict() == {0: 2, 90: 2, 180: 1}


@pytest.mark.parametrize("conflict", ["identity", "embedding_row", "fractional_index"])
def test_pathology_topics_reject_ambiguous_embeddings(extraction, monkeypatch, tmp_path, conflict):
    env, scores = extraction
    _write_pathology_topic_inputs(env, scores, tmp_path)
    _fake_topic_modules(monkeypatch)
    meta = pd.read_parquet(env["EMBED_METADATA_PATH"])
    rows = meta.index[meta["NOTE_TYPE"].eq("Pathology")]
    if conflict == "identity":
        extra = meta.loc[[rows[0]]].assign(EMBEDDING_INDEX=999)
        meta = pd.concat([meta, extra], ignore_index=True)
    elif conflict == "embedding_row":
        meta.loc[rows[1], "EMBEDDING_INDEX"] = meta.loc[rows[0], "EMBEDDING_INDEX"]
    else:
        meta["EMBEDDING_INDEX"] = meta["EMBEDDING_INDEX"].astype(float)
        meta.loc[rows[0], "EMBEDDING_INDEX"] = 0.5
    meta.to_parquet(env["EMBED_METADATA_PATH"], index=False)
    _run_analysis(env, "risk-scores", "reports", "term-definitions", notebook=TOPIC_NOTEBOOK)
    with pytest.raises(ValueError, match="Ambiguous embedding metadata|finite nonnegative integers"):
        _run_analysis(env, "topic-fit", notebook=TOPIC_NOTEBOOK)


@pytest.mark.parametrize("notebook", [CROSS_NOTEBOOK, CROSS_TOPIC_NOTEBOOK])
@pytest.mark.parametrize("undated_first", [True, False])
def test_cross_note_duplicate_dates_preserve_landmark_membership(extraction, tmp_path, notebook, undated_first):
    env, scores = extraction
    _write_cross_inputs(env, scores, tmp_path)
    path = env["NOTE_PATHS"]["Clinician"]
    notes = pl.read_parquet(path)
    dated = notes.head(1).with_columns(pl.lit("2020-04-30").alias("EVENT_DATE"))
    undated = dated.with_columns(pl.lit(None).cast(pl.String).alias("EVENT_DATE"))
    copies = [undated, dated] if undated_first else [dated, undated]
    pl.concat([*copies, notes.slice(1)]).write_parquet(path)
    _run_cross(env, "risk-scores", "reports", notebook=notebook)
    for lm, expected in ((0, 1), (90, 1), (180, 0)):
        reports = env["reports_by_lm"][lm]
        assert ((reports["DFCI_MRN"] == IDS[0]) & reports["NOTE_TYPE"].eq("Clinician")).sum() == expected


@pytest.mark.parametrize("notebook", [CROSS_NOTEBOOK, CROSS_TOPIC_NOTEBOOK])
def test_cross_note_duplicate_conflicting_dates_stop(extraction, tmp_path, notebook):
    env, scores = extraction
    _write_cross_inputs(env, scores, tmp_path)
    path = env["NOTE_PATHS"]["Clinician"]
    notes = pl.read_parquet(path)
    conflict = notes.head(1).with_columns(pl.lit("2019-01-01").alias("EVENT_DATE"))
    pl.concat([conflict, notes]).write_parquet(path)
    _run_cross(env, "risk-scores", notebook=notebook)
    with pytest.raises(ValueError, match="Conflicting note dates"):
        _run_cross(env, "reports", notebook=notebook)


# --- Pre-landmark copies of 13-16: notes dated at or before ADT + landmark (pre-ADT included, undated dropped) ---
PRE_NOTEBOOK = PRE_DIR / "13_pathology_topics_vs_risk.ipynb"
PRE_TOPIC_NOTEBOOK = PRE_DIR / "15_pathology_bertopic_vs_risk.ipynb"
PRE_CROSS_NOTEBOOK = PRE_DIR / "14_clinical_notes_topics_vs_risk.ipynb"
PRE_CROSS_TOPIC_NOTEBOOK = PRE_DIR / "16_clinical_notes_bertopic_vs_risk.ipynb"


def _set_anchor(env, date):
    pd.DataFrame({"DFCI_MRN": IDS, "TREATMENT_ANCHOR_DATE": date}).to_csv(env["DATA_PATH"], index=False)


def _run_pre(env, *cell_ids):
    for cell_id in cell_ids:
        path, src = _cell_source((PRE_NOTEBOOK, PRE_TOPIC_NOTEBOOK), cell_id)
        exec(compile(src, f"{path}:{cell_id}", "exec"), env)


def test_pre_landmark_window_boundaries(extraction):
    env, scores = extraction
    notes = _write_term_inputs(env, scores)
    _set_anchor(env, "2019-03-01")
    # 2018-06-01 is before ADT start (day -273): used at every landmark. 2019-06-01 is day 92: only at +180.
    # 2020-05-01 and 2021-01-01 are later than every landmark; the undated note is never used.
    notes += [(pid, "2018-06-01", "prior biopsy") for pid in IDS]
    _write_notes(env, notes)
    _run_pre(env, "risk-scores", "reports")
    reports = env["reports_by_lm"]
    assert [len(reports[lm]) for lm in (0, 90, 180)] == [20, 20, 42]
    assert set(reports[0]["RPT_TEXT"]) == {"prior biopsy"}
    assert not any("ductal" in t or "chromogranin" in t or "cribriform" in t for t in reports[180]["RPT_TEXT"])
    assert [r["in_window"] for r in env["window_rows"]] == [20, 20, 42]
    assert [r["undated"] for r in env["window_rows"]] == [2, 2, 2]


def test_pre_landmark_term_enrichment_runs_on_pre_adt_notes(extraction, capsys):
    env, scores = extraction
    _write_term_inputs(env, scores)
    _set_anchor(env, "2022-01-01")   # every dated note precedes ADT start, so all are pre-landmark at every landmark
    _run_pre(env, *TERM_CELLS)
    all_res = pd.read_csv(env["OUTPUT_DIR"] / "pathology_term_enrichment.csv")
    assert {"neuroendocrine", "chromogranin", "ductal"} <= set(all_res["term"])
    assert "cribriform" not in set(all_res["term"])   # undated: dropped
    sizes = env["group_sizes"]
    for lm in (0, 90, 180):
        assert sizes.loc[(lm, "bottom"), "reports"] == 4 and sizes.loc[(lm, "top"), "reports"] == 6
    assert "Pre-landmark notes (note date <= ADT + landmark; undated notes dropped):" in capsys.readouterr().out


def test_pre_landmark_notebooks_are_the_post_notebooks_with_the_window_flipped():
    for name in ("13_pathology_topics_vs_risk", "14_clinical_notes_topics_vs_risk",
                 "15_pathology_bertopic_vs_risk", "16_clinical_notes_bertopic_vs_risk"):
        post = {c["id"]: "".join(c["source"]) for c in json.loads((ANALYSIS_DIR / f"{name}.ipynb").read_text())["cells"]}
        pre = {c["id"]: "".join(c["source"]) for c in json.loads((PRE_DIR / f"{name}.ipynb").read_text())["cells"]}
        assert post.keys() == pre.keys()
        assert "days.gt(lm) & days.le(WINDOW_MAX_DAYS)" in post["reports"] and "days.le(lm)" in pre["reports"]
        assert 'OUTPUT_DIR / "post_landmark"' in post["config"] and 'OUTPUT_DIR / "pre_landmark"' in pre["config"]
        assert "WINDOW_MAX_DAYS" not in "".join(pre.values())
        for cell_id in pre:   # every analysis cell outside the window logic is identical
            if cell_id not in ("overview", "config", "reports", "term-description", "continuous-description",
                               "topic-description", "sensitivity-description", "reports-description", "topic-fit"):
                assert pre[cell_id] == post[cell_id], (name, cell_id)
        assert "post-landmark" not in "".join(pre.values()).lower().replace("dated after adt", "")


def test_pre_landmark_cross_note_notebooks_run_end_to_end(extraction, monkeypatch, tmp_path):
    env, scores = extraction
    _write_cross_inputs(env, scores, tmp_path)
    _set_anchor(env, "2022-01-01")   # the cross-note fixture's notes (2021-01-01) all precede ADT start
    _fake_topic_modules(monkeypatch)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    for notebook in (PRE_CROSS_NOTEBOOK, PRE_CROSS_TOPIC_NOTEBOOK):
        _run_cross(env, *_code_cell_ids(notebook), notebook=notebook)
    assert len(env["all_notes"]) == 60 and len(env["embed_docs"]) == 58
    for lm in (0, 90, 180):
        assert len(env["reports_by_lm"][lm]) == 60
    for name in ("term_enrichment", "lexicon_enrichment", "score_association", "topic_continuous_association"):
        assert (env["OUTPUT_DIR"] / f"clinical_notes_{name}.csv").exists()
