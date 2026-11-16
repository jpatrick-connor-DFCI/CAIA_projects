"""Exercise the patient-extraction notebook with synthetic data only."""
import json
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
               ENDPOINTS=("avpc", "platinum", "nepc"))
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
    text = "synthetic\ttext\nsecond line"
    pl.DataFrame({"DFCI_MRN": ids, "EVENT_DATE": ["2021-01-01"] * 20,
                  "RPT_ID": list(range(20)), "RPT_TEXT": [text] * 20}).write_parquet(env["PATHOLOGY_PATH"])
    _run_cells(env, 4, 5, 7, 9, 11)
    manifest = pd.read_csv(env["OUTPUT_DIR"] / "patient_manifest.csv")
    reports = pd.read_csv(env["OUTPUT_DIR"] / "pathology_reports.tsv", sep="\t")
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
    """ADT start 2020-01-01 for all; pathology reports that separate the deciles from the middle."""
    ids, top, bottom = IDS, TOP, BOTTOM
    scores.to_csv(env["SCORES_PATH"], index=False)
    local = pd.DataFrame({col: [0] * 20 for col in fet.LOCAL_COLUMNS})
    local["DFCI_MRN"] = ids
    local["TREATMENT_ANCHOR_DATE"] = "2020-01-01"
    for indicator, date in fet.ENDPOINT_SOURCES.values():
        local[indicator] = 0
        local[date] = "2020-12-31"
    local.to_csv(env["DATA_PATH"], index=False)

    notes = []
    for pid in ids:
        notes.append((pid, "2019-06-01", "adenocarcinoma of the prostate"))   # pre-ADT: every landmark
        if pid in top:
            notes.append((pid, "2019-06-01", "neuroendocrine features present"))  # separate report
        if pid in top:
            notes.append((pid, "2021-01-01", "chromogranin positive"))       # after every landmark
        if pid in bottom:
            notes.append((pid, "2020-05-01", "ductal pattern"))              # in window at 180 only
            notes.append((pid, None, "undated cribriform"))                  # undated: always excluded
    pl.DataFrame({"DFCI_MRN": [n[0] for n in notes], "EVENT_DATE": [n[1] for n in notes],
                  "RPT_ID": list(range(len(notes))), "RPT_TEXT": [n[2] for n in notes]}
                 ).write_parquet(env["PATHOLOGY_PATH"])
    env.update(TERM_MIN_PATIENTS=2, TERM_NGRAMS=(1, 2), TERM_FDR=0.05, TERM_TOP_N=5)
    return notes


def test_term_enrichment_contrasts_deciles_with_middle_on_pre_landmark_reports(extraction, capsys):
    env, scores = extraction
    ids = IDS
    _write_term_inputs(env, scores)
    _run_cells(env, 4, 5, 7, 9, 11, 13, 14, 15)

    res = pd.read_csv(env["OUTPUT_DIR"] / "pathology_term_enrichment.csv")
    assert set(res["group"]) == {"top", "bottom"}
    assert (res["n_control"] == 16).all() and (res["n_group"] == 2).all()
    for lm in (0, 90, 180):
        t = res[(res["landmark"] == lm) & (res["group"] == "top")].set_index("term")
        assert t.loc["neuroendocrine", "pct_group"] == 1 and t.loc["neuroendocrine", "pct_control"] == 0
        assert t.loc["neuroendocrine", "p_value"] == pytest.approx(1 / 153)   # 1 / C(18, 2)
        assert t["p_value"].min() == t.loc["neuroendocrine", "p_value"]
        assert t.loc["prostate", "pct_control"] == 1
        # Bigrams stay within one report: "prostate neuroendocrine" would span two.
        assert "prostate neuroendocrine" not in t.index and "neuroendocrine features" in t.index
    assert not res["term"].str.contains("chromogranin|cribriform").any()
    ductal = res[res["term"].eq("ductal")]
    assert set(ductal["landmark"]) == {180}
    assert ductal.loc[ductal["group"].eq("bottom"), "pct_group"].eq(1).all()
    sizes = env["group_sizes"]
    assert sizes.loc[(0, "middle"), "patients_scored"] == 16
    assert sizes.loc[(180, "bottom"), "reports"] == 4 and sizes.loc[(0, "bottom"), "reports"] == 2
    assert sizes.loc[(0, "top"), "reports"] == 4

    output = capsys.readouterr().out
    assert "Landmark 0 · top decile vs middle" in output
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
    _run_cells(env, 4, 5, 7, 9, 11, 13, 14, 15)

    result = pd.read_csv(env["OUTPUT_DIR"] / "pathology_term_enrichment.csv")
    assert result.empty and list(result.columns) == env["ENRICHMENT_COLS"]
    assert "Landmark 180 · top decile vs middle: no terms" in capsys.readouterr().out


TOPIC_MODULES = ("zstandard", "bertopic", "umap", "hdbscan")
TERM_CELLS = (4, 5, 7, 9, 11, 13, 14, 15)
TOPIC_CELLS = (17, 18)


def test_bertopic_section_skips_cleanly_without_packages(extraction, capsys, monkeypatch):
    env, scores = extraction
    _write_term_inputs(env, scores)
    for name in TOPIC_MODULES:
        monkeypatch.setitem(sys.modules, name, None)      # import raises ImportError
    env.update(EMBED_METADATA_PATH=Path("/nonexistent/meta.parquet"), EMBED_ARRAY_PATH=Path("/nonexistent/a.npy.zst"),
               TOPIC_MIN_SIZE=2, TOPIC_SEED=0)
    _run_cells(env, *TERM_CELLS, *TOPIC_CELLS)
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
        meta.append({"RPT_ID": float(rpt_id), "NOTE_TYPE": "Pathology", "EMBEDDING_INDEX": len(rows)})
        rows.append(axis(rpt_id, text, pid))
        meta.append({"RPT_ID": float(rpt_id), "NOTE_TYPE": "Imaging", "EMBEDDING_INDEX": len(rows)})
        rows.append([1, 0, 0])
    array_path = tmp_path / "emb.npy"
    np.save(array_path, np.asarray(rows, dtype=np.float16))
    pl.DataFrame(meta).write_parquet(tmp_path / "meta.parquet")
    env.update(EMBED_METADATA_PATH=tmp_path / "meta.parquet", EMBED_ARRAY_PATH=array_path,
               TOPIC_MIN_SIZE=2, TOPIC_SEED=0)
    _run_cells(env, *TERM_CELLS, *TOPIC_CELLS)

    (docs, embeddings, kwargs), = _FakeBERTopic.calls
    assert len(docs) == 24 and embeddings.dtype == np.float32      # 20 pre-ADT + 2 neuroendocrine + 2 ductal
    assert not any("chromogranin" in d or "cribriform" in d for d in docs)
    assert sum(e[0] == 1 for e in embeddings) == 2                  # no imaging row leaked in
    vocab = kwargs["vectorizer_model"].vocabulary
    assert "neuroendocrine" in vocab and "prostate" in vocab and kwargs["umap_model"][1]["random_state"] == 0

    topics = pd.read_csv(env["OUTPUT_DIR"] / "pathology_topics.csv")
    assert list(topics.columns) == ["topic", "n_reports", "n_patients", "top_words"]
    assert topics.set_index("topic").loc[-1, "n_reports"] == 1
    res = pd.read_csv(env["OUTPUT_DIR"] / "pathology_topic_enrichment.csv")
    assert not any("Representative" in c for c in topics.columns.tolist() + res.columns.tolist())
    assert (res["n_control"] == 16).all() and (res["n_group"] == 2).all()   # outlier patient stays in the denominator
    for lm in (0, 90, 180):
        top = res[(res["landmark"] == lm) & (res["group"] == "top")].set_index("topic")
        assert top.loc[0, "top_words"] == "neuroendocrine" and top.loc[0, "pct_group"] == 1
        assert top.loc[0, "p_value"] == pytest.approx(1 / 153)
        assert top.loc[2, "pct_control"] == pytest.approx(15 / 16)
    assert set(res.loc[res["topic"].eq(1), "landmark"]) == {180}
    assert -1 not in set(res["topic"])

    out = capsys.readouterr().out
    assert "24 of 24 in-window pathology reports have a precomputed embedding" in out
    assert "Landmark 0 · top decile vs middle (topics)" in out
    assert not any(str(patient_id) in out for patient_id in IDS)
    # Whole reports never print; two-word reports such as "ductal pattern" are also legitimate aggregate bigrams.
    assert [n[2] for n in notes if len(n[2].split()) > 2 and n[2] in out] == []

    # The streaming reader keeps exactly the requested rows across chunk boundaries and short reads.
    want = np.array([0, 3, 4, 9, len(rows) - 1])
    got = env["read_embedding_rows"](array_path, want, mods["zstandard"], chunk_rows=2)
    np.testing.assert_array_equal(got, np.load(array_path)[want].astype(np.float32))
