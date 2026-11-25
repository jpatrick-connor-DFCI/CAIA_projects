import sys
from pathlib import Path

import pytest

pl = pytest.importorskip("polars")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.prostate_ihc import extract_prostate_ihc as ex  # noqa: E402
from COMPASS.prostate_ihc import oncopanel_ihc as op  # noqa: E402
from COMPASS.prostate_ihc import paths as ihc_paths  # noqa: E402


@pytest.mark.parametrize("raw, expected", [
    ("BS-13-23027-C12", "BS13-23027"),
    ("BS-23-W46905-A1", "BS23-46905"),
    ("BS-23-46905", "BS23-46905"),
    ("BWHPowerpath_BS1323027", "BS13-23027"),
    (None, None),
    ("no accession here", None),
])
def test_key_normalises_accessions(raw, expected):
    assert op.key(raw) == expected


# The original parser (kept as is): a result word right after the marker wins, else the last one before it.
# "staining" / "expression" directly after a marker count as positive, so the last four cases are known
# misreads of the original logic, pinned here so any change to them is deliberate.
@pytest.mark.parametrize("sentence, marker, expected", [
    ("Tumor cells are positive for PSMA.", "PSMA", "positive"),
    ("PTEN: lost in tumor cells.", "PTEN", "negative"),
    ("PTEN is lost.", "PTEN", "negative"),
    ("Negative for AMACR.", "AMACR", "negative"),
    ("AMACR mentioned.", "AMACR", None),
    ("PSMA staining is negative.", "PSMA", "positive"),
    ("PSMA expression is absent.", "PSMA", "positive"),
    ("Loss of PTEN expression.", "PTEN", "positive"),
    ("Synaptophysin positive, PSMA staining is pending.", "PSMA", "positive"),
])
def test_polarity_original_behaviour(sentence, marker, expected):
    i = sentence.index(marker)
    assert ex.polarity(sentence, i, i + len(marker)) == expected


def _sample(mrn, acc, organ, histology, dx_text, date, outside_id=None):
    return {"MRN_INT": mrn, "DFCI_MRN": mrn, "ACCESSION": acc, "PART": "A", "SPEC_DATE": date,
            "OUTSIDE_SPECIMEN_DATE": None, "OUTSIDE_SPECIMEN_ID": outside_id, "site": organ, "organ": organ,
            "procedure": "biopsy", "category": "malignant", "histology": histology, "metastatic": organ != "prostate",
            "grade_gleason": None, "grade_grade_group": None, "RAW_TEXT": dx_text, "LATEST_VERSION": True}


def _ihc(mrn, acc, marker, result, text, target="lesional", doc="D"):
    return {"MRN_INT": mrn, "DFCI_MRN": mrn, "ACCESSION": acc, "PART": "A", "SPEC_DATE": f"202{mrn % 10}-02-01",
            "OUTSIDE_SPECIMEN_DATE": None, "RPT_DATE": "2021-01-01", "DOC_ID": f"{doc}{mrn}", "method": "ihc",
            "target": target, "marker": marker, "result": result, "qualifier": None, "pct": None, "score": None,
            "RAW_TEXT": text, "LATEST_VERSION": True}


def _df(rows):
    """Rows -> DataFrame; all-null columns become Utf8 (pct Float64), as in the real extracts."""
    df = pl.DataFrame(rows)
    return df.with_columns([pl.col(c).cast(pl.Float64 if c == "pct" else pl.Utf8)
                            for c, t in df.schema.items() if t == pl.Null])


@pytest.fixture
def inputs(tmp_path):
    path_dir, pq_dir = tmp_path / "pathology_extracts", tmp_path / "parquet"
    path_dir.mkdir(); pq_dir.mkdir()
    _df([{"MRN_INT": 1, "SITE_CD": "C61.9", "DIAGNOSIS_DT": "2019-05-01 00:00:00",
                   "HISTOLOGY_DSC": "Adenocarcinoma"},
                  {"MRN_INT": 2, "SITE_CD": "C61", "DIAGNOSIS_DT": "2018-02-01",   # met-only pathology
                   "HISTOLOGY_DSC": "Adenocarcinoma"},
                  {"MRN_INT": 9, "SITE_CD": "C34.1", "DIAGNOSIS_DT": "2019-05-01", "HISTOLOGY_DSC": "Lung"}]
                 ).write_parquet(pq_dir / "IDM_CARG_DIAGNOSIS.parquet")
    _df([
        _sample(1, "BS-20-A100", "prostate", "adenocarcinoma", "Prostate, biopsy: adenocarcinoma.", "2021-03-01"),
        _sample(2, "BS-21-200", "liver", "adenocarcinoma",
                "Liver: metastatic carcinoma, consistent with prostatic origin.", "2022-01-05"),
        _sample(3, "BS-22-300", "prostate", "adenocarcinoma", "Prostate: adenocarcinoma.", "2022-06-01"),
        _sample(9, "BS-22-900", "lung", "adenocarcinoma", "Lung adenocarcinoma.", "2022-06-01"),
    ]).write_parquet(path_dir / "samples.parquet")
    _df([
        _ihc(1, "BS-20-A100", "SYNAPTOPHYSIN", "positive", "Synaptophysin positive; NKX3.1 and PSA negative. PTEN is lost."),
        _ihc(1, "BS-20-A100", "NKX31", "negative", "Synaptophysin positive; NKX3.1 and PSA negative. PTEN is lost."),
        _ihc(1, "BS-20-A100", "PSA", "negative", "Synaptophysin positive; NKX3.1 and PSA negative. PTEN is lost."),
        _ihc(2, "BS-21-200", "SYNAPTOPHYSIN", "positive", "Synaptophysin positive, NKX3.1 positive."),
        _ihc(2, "BS-21-200", "NKX31", "positive", "Synaptophysin positive, NKX3.1 positive."),
        _ihc(3, "BS-22-300", "P63", "negative", "p63 negative."),
        _ihc(3, "BS-22-300", "P63", "positive", "p63 positive in benign glands.", target="background", doc="E"),
        _ihc(9, "BS-22-900", "TTF1", "positive", "TTF-1 positive."),   # not in the cohort
    ]).write_parquet(path_dir / "ihc.parquet")
    _df([
        {"MRN_INT": 1, "UNIQUE_SAMPLE_ID": "U1", "SAMPLE_ID": "S1", "BLOCK_ACCESSION_NBR": "BS-20-100-B3",
         "CANCER_TYPE": "Prostate Cancer", "PRIMARY_CANCER_DIAGNOSIS": "Prostate", "ORIGINAL_PATH_DIAGNOSIS": "Small cell carcinoma",
         "SAMPLE_COLLECTION_DT": "2021-03-01 00:00:00"},
        {"MRN_INT": 2, "UNIQUE_SAMPLE_ID": "U2", "SAMPLE_ID": "S2", "BLOCK_ACCESSION_NBR": "BWHPowerpath_BS2100200",
         "CANCER_TYPE": "Prostate Cancer", "PRIMARY_CANCER_DIAGNOSIS": "Prostate", "ORIGINAL_PATH_DIAGNOSIS": "Adenocarcinoma",
         "SAMPLE_COLLECTION_DT": "2022-01-05"},
        {"MRN_INT": 4, "UNIQUE_SAMPLE_ID": "U4", "SAMPLE_ID": "S4", "BLOCK_ACCESSION_NBR": "BS-19-999",
         "CANCER_TYPE": "Prostate Cancer", "PRIMARY_CANCER_DIAGNOSIS": "Prostate", "ORIGINAL_PATH_DIAGNOSIS": "Adenocarcinoma",
         "SAMPLE_COLLECTION_DT": "2019-01-01"},
        {"MRN_INT": 9, "UNIQUE_SAMPLE_ID": "U9", "SAMPLE_ID": "S9", "BLOCK_ACCESSION_NBR": "BS-22-900",
         "CANCER_TYPE": "Lung Cancer", "PRIMARY_CANCER_DIAGNOSIS": "Lung", "ORIGINAL_PATH_DIAGNOSIS": "Adeno",
         "SAMPLE_COLLECTION_DT": "2022-06-01"},
    ]).write_parquet(pq_dir / "CDM_SPEC_CON_ids.parquet")
    return path_dir, pq_dir, tmp_path / "intermediate", tmp_path / "results"


def test_step1_cohort_calls_and_evidence(inputs):
    path_dir, pq_dir, inter, _ = inputs
    out = ex.run(path_dir, pq_dir, inter)
    assert all((inter / f).exists() for f in ihc_paths.INTERMEDIATE_FILES)
    cohort = out["cohort"].sort("MRN_INT")
    assert cohort["MRN_INT"].to_list() == [1, 2, 3]          # registry C61 or malignant prostate pathology only
    assert cohort["in_registry"].to_list() == [True, True, False]
    assert cohort["in_pathology"].to_list() == [True, False, True]
    assert cohort["has_panel_ihc"].to_list() == [True, True, True]

    long = out["long"]
    pten = long.filter(pl.col("marker_norm") == "PTEN")
    assert pten["result"].to_list() == ["lost"] and pten["source"].to_list() == ["supplement"]
    assert 9 not in long["MRN_INT"].to_list()

    spec = {r["MRN_INT"]: r for r in out["spec"].iter_rows(named=True)}
    assert spec[1]["ihc_phenotype"] == "NE+, AR-axis negative (NEPC-like)"
    assert spec[1]["PTEN"] == "lost"
    assert spec[1]["prostate_evidence"] == "2 prostate site"
    assert spec[2]["ihc_phenotype"] == "NE+, AR-axis positive (amphicrine / focal NE)"
    assert spec[2]["prostate_evidence"] == "1 prostate lineage IHC positive"
    # background p63 does not override the lesional call
    assert spec[3]["p63"] == "negative" and spec[3]["ihc_phenotype"] == "no NE or AR-axis stain"
    assert all(spec[m]["prostate_relevant"] for m in (1, 2, 3))
    assert set(ex.summary_tables(out)) == {"cohort", "markers", "evidence", "phenotype_relevant", "relevant"}


def test_step2_matches_oncopanel_and_flags(inputs):
    path_dir, pq_dir, inter, res = inputs
    ex.run(path_dir, pq_dir, inter)
    spec, long = op.run(pq_dir, path_dir, inter, res)
    assert all((res / f).exists() for f in ihc_paths.RESULT_FILES)
    rows = {r["UNIQUE_SAMPLE_ID"]: r for r in spec.iter_rows(named=True)}
    assert set(rows) == {"U1", "U2", "U4"}                       # prostate OncoPanel only
    assert rows["U1"]["path_matched"] and rows["U1"]["path_accession"] == "BS-20-A100"  # letter-dropped match
    assert rows["U2"]["path_matched"]                            # BWHPowerpath block
    assert not rows["U4"]["path_matched"] and rows["U4"]["ihc_phenotype"] == "no pathology report matched"
    assert rows["U1"]["NEPC_ihc"] and not rows["U1"]["amphicrine_ihc"] and rows["U1"]["NE_label_oncopanel"]
    assert rows["U1"]["ARaxis_any_neg"] and not rows["U1"]["ARaxis_any_pos"]
    assert rows["U2"]["amphicrine_ihc"] and not rows["U2"]["NEPC_ihc"]
    assert set(long["UNIQUE_SAMPLE_ID"]) == {"U1", "U2"}
    assert long["specimen_date"].drop_nulls().to_list()

    spec2, long2 = op.load(res)
    assert spec2.height == spec.height and long2.height == long.height
    counts = op.summary_tables(spec)["counts"]
    assert counts.filter(pl.col("step") == "  NEPC-like")["specimens"].item() == 1


def test_discordant_marker_counts_as_neither(inputs):
    """Original rule: a marker both positive and negative on the specimen is 'discordant', which is neither."""
    path_dir, pq_dir, inter, res = inputs
    ihc = pl.read_parquet(path_dir / "ihc.parquet")
    extra = _df([_ihc(1, "BS-20-A100", "SYNAPTOPHYSIN", "negative", "Synaptophysin negative.", doc="F")]).cast(ihc.schema)
    pl.concat([ihc, extra]).write_parquet(path_dir / "ihc.parquet")
    step1 = ex.run(path_dir, pq_dir, inter)
    spec, _ = op.run(pq_dir, path_dir, inter, res)
    for row in (step1["spec"].filter(pl.col("MRN_INT") == 1).row(0, named=True),
                spec.filter(pl.col("UNIQUE_SAMPLE_ID") == "U1").row(0, named=True)):
        assert row["Synaptophysin"] == "discordant" and not row["NE_any_pos"]
        assert not row["ARaxis_any_pos"] and "NEPC-like" not in row["ihc_phenotype"]


def test_load_missing_outputs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ex.load(tmp_path)
    with pytest.raises(FileNotFoundError):
        op.load(tmp_path)


def test_default_paths_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPASS_IHC_EXTRACTION_ROOT", str(tmp_path))
    d = ihc_paths.default_paths()
    assert d.intermediate_dir == tmp_path / "intermediate" and d.results_dir == tmp_path / "results"


def test_step2_csvs_feed_ihc_lr(inputs):
    """Step-2 CSVs written to disk -> ihc_lr flags, OncoPanel patients and a frame."""
    pd = pytest.importorskip("pandas")
    from COMPASS.multivariable_models import ihc_lr as il
    path_dir, pq_dir, inter, res = inputs
    ex.run(path_dir, pq_dir, inter)
    op.run(pq_dir, path_dir, inter, res)
    spec = il.load_oncopanel_ihc(res / ihc_paths.ONCOPANEL_SPECIMEN_FILE, res / ihc_paths.ONCOPANEL_LONG_FILE)
    flags = spec.set_index("UNIQUE_SAMPLE_ID")
    assert flags.loc["U1", ["ihc_any", "ne_pos", "nepc_like", "ar_low"]].tolist() == [True] * 4
    assert flags.loc["U2", ["ihc_any", "ne_pos", "nepc_like", "ar_low"]].tolist() == [True, True, False, False]
    assert not flags.loc["U4", ["ihc_any", "ne_pos", "nepc_like", "ar_low"]].any()
    assert flags.loc["U1", "specimen_date"] == pd.Timestamp("2021-02-01")      # lesional stain row date
    assert il.load_oncopanel_patients(res / ihc_paths.ONCOPANEL_SPECIMEN_FILE) == {1, 2, 4}

    scores = pd.DataFrame({"person_id": [1, 2, 4], "landmark_days": 0, "risk_score": [1., 2., 3.]})
    anchors = pd.Series(pd.Timestamp("2020-01-01"), index=[1, 2, 4])
    frame = il.build_ihc_frame(scores, spec, [0], anchors=anchors).set_index("person_id")
    assert frame["ar_low_future"].to_dict() == {1: 1, 2: 0, 4: 0}
    assert frame["nepc_like_future"].to_dict() == {1: 1, 2: 0, 4: 0}
    assert frame["unmatched_specimen_future"].to_dict() == {1: 0, 2: 0, 4: 0}   # U4 collected 2019, before ADT


@pytest.mark.parametrize("target,expected", [("lesional", True), ("background", False)])
def test_ki67_only_workup_counts_as_ihc(inputs, target, expected):
    from COMPASS.multivariable_models import ihc_lr as il
    path_dir, pq_dir, inter, res = inputs
    ihc = pl.read_parquet(path_dir / "ihc.parquet")
    row = _ihc(1, "BS-20-A100", "KI67", "positive", "Ki-67 stain.", target=target)
    row["pct"] = 80.0
    pl.concat([ihc.filter(pl.col("MRN_INT") != 1), _df([row]).cast(ihc.schema)]).write_parquet(path_dir / "ihc.parquet")
    ex.run(path_dir, pq_dir, inter)
    spec, long = op.run(pq_dir, path_dir, inter, res)
    flags = il.load_oncopanel_ihc(spec.to_pandas(), long.to_pandas()).set_index("UNIQUE_SAMPLE_ID")
    assert bool(flags.loc["U1", "ihc_any"]) == expected
    assert not flags.loc["U1", "ne_pos"] and not flags.loc["U1", "ar_low"]
