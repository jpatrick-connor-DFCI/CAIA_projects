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


@pytest.mark.parametrize("sentence, marker, expected", [
    ("Tumor cells are positive for PSMA.", "PSMA", "positive"),
    ("PTEN: lost in tumor cells.", "PTEN", "negative"),
    ("Negative for AMACR.", "AMACR", "negative"),
    ("AMACR mentioned.", "AMACR", None),
    ("PSMA staining is negative.", "PSMA", "negative"),
    ("PTEN staining is lost.", "PTEN", "negative"),
    ("PSMA expression is absent.", "PSMA", "negative"),
    ("PSMA stain is non-reactive.", "PSMA", "negative"),
    ("PSMA staining is positive.", "PSMA", "positive"),
    ("PTEN expression is retained.", "PTEN", "positive"),
    ("PSMA staining was performed.", "PSMA", None),
    # Look-back phrasings: stain / expression before the marker are positive.
    ("Tumor cells show strong expression of PSMA.", "PSMA", "positive"),
    ("Diffuse expression of AMACR is seen.", "AMACR", "positive"),
    ("AMACR stains the tumor cells.", "AMACR", "positive"),
    ("Tumor cells are stained by AMACR.", "AMACR", "positive"),
    ("Tumor cells stain for PSMA.", "PSMA", "positive"),
    ("Loss of expression of PTEN.", "PTEN", "negative"),
    ("Lack of staining for PTEN.", "PTEN", "negative"),
    ("Tumor cells are not stained by AMACR.", "AMACR", "negative"),
    ("AMACR stains are negative.", "AMACR", "negative"),
    ("AMACR stains were performed.", "AMACR", None),
    ("Immunostains include PSMA.", "PSMA", None),
    ("No significant expression of PSMA is seen.", "PSMA", "negative"),
    ("Absence of expression of PSMA.", "PSMA", "negative"),
    ("Loss of cytoplasmic expression of PTEN.", "PTEN", "negative"),
    ("Without detectable nuclear expression of PTEN.", "PTEN", "negative"),
    ("PTEN loss, but strong expression of PSMA.", "PSMA", "positive"),
    ("No expression of PSMA, but AMACR stains the tumor cells.", "AMACR", "positive"),
    ("PSMA stains both benign and tumor cells.", "PSMA", "positive"),
    ("PSMA stains will be performed.", "PSMA", None),
    ("AMACR stains have been ordered.", "AMACR", None),
    ("Synaptophysin positive, PSMA stains will be performed.", "PSMA", None),
    ("Synaptophysin negative, AMACR stains have been ordered.", "AMACR", None),
    ("Synaptophysin positive, PSMA staining is pending.", "PSMA", None),
    ("Synaptophysin positive, recommend PSMA staining.", "PSMA", None),
    ("Synaptophysin positive, planned staining for PSMA.", "PSMA", None),
    # Coordinated marker lists before an unresulted phrase.
    ("Stains for PSMA and AMACR were ordered.", "PSMA", None),
    ("Stains for PSMA and AMACR were ordered.", "AMACR", None),
    ("Stained for PSMA and AMACR, results to follow.", "PSMA", None),
    ("Immunostains for PSMA, AMACR and PTEN are pending.", "AMACR", None),
    ("PSMA and AMACR are positive.", "AMACR", "positive"),
    # A result word that belongs to the marker survives a trailing "performed".
    ("Positive staining for AMACR was performed on block A2.", "AMACR", "positive"),
    ("Tumor cells are positive for AMACR performed on block A3.", "AMACR", "positive"),
    ("Negative staining for PSMA was performed.", "PSMA", "negative"),
    ("No significant expression of PSMA was done.", "PSMA", "negative"),
])
def test_polarity(sentence, marker, expected):
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
    assert cohort["MRN_INT"].to_list() == [1, 2, 3, 4]
    assert cohort["in_registry"].to_list() == [True, True, False, False]
    assert cohort["in_oncopanel"].to_list() == [True, True, False, True]
    assert cohort["has_panel_ihc"].to_list() == [True, True, True, False]

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
    assert rows["U2"]["amphicrine_ihc"] and not rows["U2"]["NEPC_ihc"]
    assert set(long["UNIQUE_SAMPLE_ID"]) == {"U1", "U2"}
    assert long["specimen_date"].drop_nulls().to_list()

    spec2, long2 = op.load(res)
    assert spec2.height == spec.height and long2.height == long.height
    counts = op.summary_tables(spec)["counts"]
    assert counts.filter(pl.col("step") == "  NEPC-like")["specimens"].item() == 1


def test_load_missing_outputs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ex.load(tmp_path)
    with pytest.raises(FileNotFoundError):
        op.load(tmp_path)


def test_default_paths_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPASS_IHC_EXTRACTION_ROOT", str(tmp_path))
    d = ihc_paths.default_paths()
    assert d.intermediate_dir == tmp_path / "intermediate" and d.results_dir == tmp_path / "results"


def test_step1_intermediates_feed_annotations(inputs):
    pd = pytest.importorskip("pandas")
    from COMPASS.prostate_ihc import annotations as ann
    path_dir, pq_dir, inter, res = inputs
    ex.run(path_dir, pq_dir, inter)
    out = ann.build(inter)["specimens"]
    step1 = ex.load(inter)
    assert len(out) == step1["spec"].height
    mine = out.set_index(["MRN_INT", "ACCESSION", "PART"])
    for r in step1["spec"].iter_rows(named=True):
        row = mine.loc[(r["MRN_INT"], r["ACCESSION"] or "", r["PART"] or "")]
        assert row["ne_pos"] == r["NE_any_pos"] and row["ARaxis_any_pos"] == r["ARaxis_any_pos"]
        assert row["nepc_like"] == ("NEPC-like" in r["ihc_phenotype"])
        assert row["prostate_relevant"] == r["prostate_relevant"]
    assert pd.api.types.is_datetime64_any_dtype(out["specimen_date"])


def pd_isna(x):
    import pandas as pd
    return pd.isna(x)


def _flags(spec, long):
    """Per-UNIQUE_SAMPLE_ID flags from the step-2 frames (matched specimens only)."""
    les = long.filter(pl.col("target") == "lesional")
    dates = les.group_by("UNIQUE_SAMPLE_ID").agg(pl.col("specimen_date").min())
    out = (spec.join(dates, on="UNIQUE_SAMPLE_ID", how="left")
           .with_columns((pl.col("path_matched") & pl.col("UNIQUE_SAMPLE_ID").is_in(les["UNIQUE_SAMPLE_ID"].implode()))
                         .alias("ihc_any"),
                         (pl.col("path_matched") & pl.col("NE_any_pos")).alias("ne_pos"),
                         (pl.col("path_matched") & pl.col("NEPC_ihc")).alias("nepc_like")))
    return out.select("UNIQUE_SAMPLE_ID", "ihc_any", "ne_pos", "nepc_like", "specimen_date").to_pandas().set_index("UNIQUE_SAMPLE_ID")


def _run_pipeline(inputs):
    path_dir, pq_dir, inter, res = inputs
    step1 = ex.run(path_dir, pq_dir, inter)
    spec, long = op.run(pq_dir, path_dir, inter, res)
    return step1, spec, long, _flags(spec, long)


@pytest.mark.parametrize("marker,result,text,nepc", [
    ("SYNAPTOPHYSIN", "negative", "Synaptophysin negative.", True),
    ("NKX31", "positive", "NKX3.1 positive.", False),
])
def test_discordant_stains_preserve_positive_evidence(inputs, marker, result, text, nepc):
    path_dir = inputs[0]
    ihc = pl.read_parquet(path_dir / "ihc.parquet")
    extra = _df([_ihc(1, "BS-20-A100", marker, result, text, doc="F")]).cast(ihc.schema)
    pl.concat([ihc, extra]).write_parquet(path_dir / "ihc.parquet")
    step1, spec, _, lr = _run_pipeline(inputs)
    marker_norm = ex.PANEL[marker][0]
    first = step1["spec"].filter(pl.col("MRN_INT") == 1).row(0, named=True)
    final = spec.filter(pl.col("UNIQUE_SAMPLE_ID") == "U1").row(0, named=True)
    assert first[marker_norm] == final[marker_norm] == "discordant"
    assert first["NE_any_pos"] and final["NE_any_pos"]
    assert first["ARaxis_any_pos"] == final["ARaxis_any_pos"] == (not nepc)
    assert ("NEPC-like" in first["ihc_phenotype"]) == nepc
    assert final["NEPC_ihc"] == nepc
    assert lr.loc["U1", ["ihc_any", "ne_pos", "nepc_like"]].tolist() == [True, True, nepc]
    assert spec.filter(pl.col(marker_norm).is_in(["positive", "discordant"])).height == 2
    # LR annotations: discordant NE -> positive, discordant AR-axis -> neither. So a discordant NKX3.1 does not
    # block AR-low the way the extraction's AR-axis-positive flag does; the other AR-axis negatives decide.
    from COMPASS.multivariable_models import ihc_lr as il
    annotations = il.load_ihc_specimens(intermediate_dir=inputs[2]).set_index('DFCI_MRN')
    assert annotations.loc[1, ['ne_pos', 'ar_low', 'nepc_like']].astype(bool).tolist() == [True, True, True]


@pytest.mark.parametrize("text", [
    "Synaptophysin positive; PSMA staining is negative; PTEN staining is lost.",
    "Synaptophysin positive; No significant expression of PSMA is seen; Loss of cytoplasmic expression of PTEN.",
    "Synaptophysin positive; Absence of expression of PSMA; Without detectable nuclear expression of PTEN.",
])
def test_supplement_negative_staining_does_not_flip_phenotype(inputs, text):
    path_dir = inputs[0]
    ihc = pl.read_parquet(path_dir / "ihc.parquet")
    ihc.with_columns(pl.when(pl.col("MRN_INT") == 1).then(pl.lit(text))
                     .otherwise(pl.col("RAW_TEXT")).alias("RAW_TEXT")).write_parquet(path_dir / "ihc.parquet")
    step1, spec, _, lr = _run_pipeline(inputs)
    for table in (step1["spec"], spec):
        row = table.filter(pl.col("MRN_INT") == 1).row(0, named=True)
        assert row["PSMA"] == "negative" and row["PTEN"] == "lost"
        assert "NEPC-like" in row["ihc_phenotype"]
    assert lr.loc["U1", "nepc_like"]


@pytest.mark.parametrize("phrase", [
    "PSMA stains will be performed", "PSMA stains have been ordered",
    "PSMA staining is pending", "recommend PSMA staining", "AMACR stains have been ordered",
])
def test_planned_supplement_stains_do_not_borrow_another_markers_result(inputs, phrase):
    path_dir = inputs[0]
    ihc = pl.read_parquet(path_dir / "ihc.parquet")
    ihc.with_columns(pl.when(pl.col("MRN_INT") == 1).then(pl.lit(f"Synaptophysin positive, {phrase}."))
                     .otherwise(pl.col("RAW_TEXT")).alias("RAW_TEXT")).write_parquet(path_dir / "ihc.parquet")
    step1, spec, long, lr = _run_pipeline(inputs)
    for table in (step1["long"], long):
        assert table.filter((pl.col("MRN_INT") == 1) & pl.col("marker_norm").is_in(["PSMA", "AMACR"])).is_empty()
    assert spec.filter(pl.col("UNIQUE_SAMPLE_ID") == "U1")["NEPC_ihc"].item()
    assert lr.loc["U1", "nepc_like"]


@pytest.mark.parametrize("target,pct,expected", [
    ("lesional", 80.0, True), ("lesional", None, True), ("background", 80.0, False),
])
def test_ki67_only_workup_counts_as_ihc(inputs, target, pct, expected):
    path_dir = inputs[0]
    ihc = pl.read_parquet(path_dir / "ihc.parquet")
    row = _ihc(1, "BS-20-A100", "KI67", "positive", "Ki-67 stain.", target=target)
    row["pct"] = pct
    pl.concat([ihc.filter(pl.col("MRN_INT") != 1), _df([row]).cast(ihc.schema)]).write_parquet(path_dir / "ihc.parquet")
    _, _, _, lr = _run_pipeline(inputs)
    assert bool(lr.loc["U1", "ihc_any"]) == expected
    assert not lr.loc["U1", "ne_pos"]
    assert pd_isna(lr.loc["U1", "specimen_date"]) == (not expected)


def test_metastatic_oncopanel_patient_without_registry_retains_stains(inputs):
    registry_path = inputs[1] / "IDM_CARG_DIAGNOSIS.parquet"
    reg = pl.read_parquet(registry_path)
    reg.filter(pl.col("MRN_INT") != 2).write_parquet(registry_path)
    step1, _, _, lr = _run_pipeline(inputs)
    patient = step1["cohort"].filter(pl.col("MRN_INT") == 2).row(0, named=True)
    assert not patient["in_registry"] and not patient["in_pathology"]
    assert patient["in_oncopanel"] and patient["has_panel_ihc"]
    assert lr.loc["U2", ["ihc_any", "ne_pos", "nepc_like"]].tolist() == [True, True, False]


def test_shared_accession_keeps_stains_context_and_dates_with_patient(inputs):
    path_dir = inputs[0]
    samples = pl.read_parquet(path_dir / "samples.parquet")
    # Patient 3 shares the accession and part, but has a different specimen date
    # and histology. Neither their context nor their positive AR stain may leak.
    samples.with_columns(
        pl.when(pl.col("MRN_INT") == 3).then(pl.lit("BS-20-A100")).otherwise(pl.col("ACCESSION")).alias("ACCESSION"),
        pl.when(pl.col("MRN_INT") == 3).then(pl.lit("small cell carcinoma")).otherwise(pl.col("histology")).alias("histology"),
    ).write_parquet(path_dir / "samples.parquet")
    ihc = pl.read_parquet(path_dir / "ihc.parquet")
    extra = _df([_ihc(3, "BS-20-A100", "NKX31", "positive", "NKX3.1 positive.", doc="F")]).cast(ihc.schema)
    ihc = ihc.with_columns(pl.when(pl.col("MRN_INT") == 3).then(pl.lit("BS-20-A100"))
                           .otherwise(pl.col("ACCESSION")).alias("ACCESSION"))
    pl.concat([ihc, extra]).write_parquet(path_dir / "ihc.parquet")
    step1, spec, long, lr = _run_pipeline(inputs)
    first = {r["MRN_INT"]: r for r in step1["spec"].iter_rows(named=True)}
    assert first[1]["specimen_date"] == "2021-03-01" and first[1]["histology"] == "adenocarcinoma"
    assert first[3]["specimen_date"] == "2022-06-01" and first[3]["histology"] == "small cell carcinoma"
    assert long.filter(pl.col("UNIQUE_SAMPLE_ID") == "U1")["MRN_INT"].unique().to_list() == [1]
    row = spec.filter(pl.col("UNIQUE_SAMPLE_ID") == "U1").row(0, named=True)
    assert row["path_histologies"] == "adenocarcinoma" and not row["NE_histology"]
    assert row["NKX3.1"] == "negative" and row["NEPC_ihc"]
    assert lr.loc["U1", "nepc_like"]


@pytest.mark.parametrize('text,target', [
    ('Synaptophysin positive in tumor cells; PSMA positive in benign glands.', 'background'),
    ('Synaptophysin positive in tumor cells, whereas PSMA positive in benign glands.', 'background'),
    ('Benign glands express PSMA; tumor cells express synaptophysin.', 'background'),
    ('PSMA stains both benign and tumor cells.', 'unknown'),
    ('Tumor cells express PSMA.', 'lesional'),
])
def test_supplement_target_reaches_lr(inputs, text, target):
    from COMPASS.multivariable_models import ihc_lr as il
    path_dir, pq_dir, inter, res = inputs
    path = path_dir / 'ihc.parquet'
    ihc = pl.read_parquet(path)
    ihc.with_columns(pl.when(pl.col('MRN_INT') == 1).then(pl.lit(text))
                     .otherwise(pl.col('RAW_TEXT')).alias('RAW_TEXT')).write_parquet(path)
    step1 = ex.run(path_dir, pq_dir, inter)
    psma = step1['long'].filter((pl.col('MRN_INT') == 1) & (pl.col('marker_norm') == 'PSMA'))
    assert psma['target'].to_list() == [target]
    annotations = il.load_ihc_specimens(intermediate_dir=inter).set_index('DFCI_MRN')
    assert bool(annotations.loc[1, 'nepc_like']) == (target != 'lesional')
    spec, _ = op.run(pq_dir, path_dir, inter, res)
    assert spec.filter(pl.col('MRN_INT') == 1)['NEPC_ihc'].item() == (target != 'lesional')


def test_supplement_conflicting_source_targets_are_order_independent():
    rows = [_ihc(1, 'A', 'SYNAPTOPHYSIN', 'positive', 'Synaptophysin positive; PSMA positive.', target=t)
            for t in ('background', 'lesional')]
    for ordered in (rows, rows[::-1]):
        assert ex.supplement(_df(ordered))['target'].to_list() == ['unknown']


@pytest.mark.parametrize('histology,text,expected', [
    ('adenocarcinoma', 'No neuroendocrine differentiation.', False),
    ('adenocarcinoma', 'No evidence of small cell carcinoma.', False),
    ('adenocarcinoma', 'Neuroendocrine differentiation is not identified.', False),
    ('non-small cell carcinoma', '', False),
    ('non small cell carcinoma', '', False),
    ('adenocarcinoma', 'Neuroendocrine markers were ordered.', False),
    ('small cell carcinoma', '', True),
    ('adenocarcinoma', 'Neuroendocrine differentiation is present.', True),
    ('adenocarcinoma', 'No small cell carcinoma, but neuroendocrine differentiation is present.', True),
])
def test_ne_histology_assertion(histology, text, expected):
    assert ex.has_ne_histology(histology, text) == expected


def test_negated_ne_histology_does_not_create_lr_event(inputs):
    from COMPASS.multivariable_models import ihc_lr as il
    import pandas as pd
    path_dir, pq_dir, inter, res = inputs
    path = path_dir / 'samples.parquet'
    samples = pl.read_parquet(path)
    samples.with_columns(
        pl.when(pl.col('MRN_INT') == 1).then(pl.lit('liver')).otherwise(pl.col('organ')).alias('organ'),
        pl.when(pl.col('MRN_INT') == 1).then(pl.lit('Poorly differentiated carcinoma. No neuroendocrine differentiation.'))
        .otherwise(pl.col('RAW_TEXT')).alias('RAW_TEXT'),
    ).write_parquet(path)
    path = path_dir / 'ihc.parquet'
    ihc = pl.read_parquet(path)
    ihc.filter(~((pl.col('MRN_INT') == 1) & (pl.col('marker') == 'SYNAPTOPHYSIN'))).write_parquet(path)
    ex.run(path_dir, pq_dir, inter)
    annotations = il.load_ihc_specimens(intermediate_dir=inter)
    row = annotations.set_index('DFCI_MRN').loc[1]
    assert not row['NE_histology'] and not row['prostate_relevant']
    scores = pd.DataFrame({'person_id': [1], 'landmark_days': [0], 'risk_score': [1.]})
    frame = il.build_ihc_frame(scores, annotations, [0], anchors=pd.Series({1: pd.Timestamp('2020-01-01')}))
    assert not frame[list(il.SPECIMEN_FLAGS)].to_numpy().any()
    spec, _ = op.run(pq_dir, path_dir, inter, res)
    assert not spec.filter(pl.col('MRN_INT') == 1)['NE_histology'].item()
