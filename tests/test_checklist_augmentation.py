"""Patient leakage, frozen scoring, sparse support and explicit availability contracts."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from COMPASS.federated_models import checklist_augmentation as ca
from COMPASS.federated_models import checklist_augmentation_inputs as ci
from survival_common import federated_endpoint_transfer as fet
from COMPASS.multivariable_models.predictive_checklists.features import FeatureBinarizer
from COMPASS.multivariable_models.predictive_checklists.optimizer import ChecklistModel
from COMPASS.multivariable_models.predictive_checklists.evaluation import ProbabilityCalibrator


def population(n=160, seed=4, endpoints=("platinum",)):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"person_id": np.arange(n), "age": rng.normal(70, 7, n),
                      "gleason": rng.integers(6, 11, n), "L": rng.integers(0, 4, n)})
    d["B"] = d.L.ge(2).astype(int)
    d["complete_labs"], d["evaluable"], d["clock"] = True, True, "adt"
    blocks = []
    for ep in endpoints:
        event_time = rng.exponential(3200 / np.exp(.25 * d.L + .2 * (d.gleason - 8))) + 10
        censor = rng.uniform(1900, 3600, n)
        blocks.append(d.assign(endpoint=ep, duration_days=np.minimum(event_time, censor), event=(event_time < censor).astype(int)))
    return pd.concat(blocks, ignore_index=True)


def settings(**kwargs):
    return replace(ca.Settings(), bootstrap=0, penalties=(.1,), min_cases=2, min_controls=3, **kwargs)


def test_patient_folds_group_bootstrap_copies_and_share_endpoints():
    frame = ca.resample_patients(population(endpoints=("platinum", "avpc")), np.random.default_rng(1))
    assert frame.duplicated(["person_id", "endpoint"]).any()
    folds, method = ca.patient_folds(frame, 5, 7)
    assigned = frame.assign(fold=frame.person_id.map(folds))
    assert assigned.groupby("person_id").fold.nunique().eq(1).all()
    assert set(folds) == set(range(5)) and "platinum/AVPC" in method
    for k in range(5):
        assert not set(assigned.loc[assigned.fold.eq(k), "person_id"]) & set(assigned.loc[assigned.fold.ne(k), "person_id"])


def test_cox_predictions_exclude_held_out_outcomes(monkeypatch):
    d, cfg = population(), settings()
    folds, _ = ca.patient_folds(d, cfg.outer_folds, cfg.seed)
    original = ca.patient_folds
    monkeypatch.setattr(ca, "patient_folds", lambda frame, n, seed:
                        (folds, "fixed test assignments") if n == cfg.outer_folds else original(frame, n, seed))
    p, fits, _ = ca.cross_validate(d, cfg)
    altered = d.copy()
    held = altered.person_id.map(folds).eq(0)
    altered.loc[held, "event"] = 1 - altered.loc[held, "event"]
    altered.loc[held, "duration_days"] = 100
    q, _, _ = ca.cross_validate(altered, cfg)
    models = list(ca.design_models())
    a = p.loc[p.fold.eq(0) & p.model.isin(models), "score"].to_numpy()
    b = q.loc[q.fold.eq(0) & q.model.isin(models), "score"].to_numpy()
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b)
    assert ((a >= 0) & (a <= 1)).all()
    assert set(fits.loc[fits.phase.eq("inner"), "penalty"]) == {.1}


def test_nepc_is_frozen_or_platinum_transfer_never_endpoint_trained():
    frame = population(endpoints=("platinum", "nepc"))
    predictions, fits, _ = ca.cross_validate(frame, settings())
    nepc = predictions.loc[predictions.endpoint.eq("nepc")]
    assert "augmented" not in set(nepc.model)
    assert "augmented_platinum_transfer" in set(nepc.model)
    assert fits.loc[fits.endpoint.eq("nepc") & fits.model.str.endswith("_transfer"), "training_endpoint"].eq("platinum").all()
    for m in ca.design_models():
        p = predictions.loc[predictions.endpoint.eq("platinum") & predictions.model.eq(m)].set_index("person_id").score
        q = nepc.loc[nepc.model.eq(m + "_platinum_transfer")].set_index("person_id").score
        np.testing.assert_allclose(p, q, equal_nan=True)


def test_training_age_median_and_scaling_do_not_use_validation():
    d = population()
    d.loc[0, "age"] = np.nan
    fit = ca.fit_cox(d.iloc[:130], ["age", "gleason", "L"], .1, settings())
    assert fit[1]["age"] == d.iloc[:130].age.median()
    assert fit[2]["age"] == pytest.approx(d.iloc[:130].age.fillna(fit[1]["age"]).mean())
    bad = d.iloc[:130].copy()
    bad.loc[0, "gleason"] = np.nan
    with pytest.raises(ValueError, match="cannot be imputed"):
        ca.fit_cox(bad, ["age", "gleason", "L"], .1, settings())
    with pytest.raises(ValueError, match="validation.*cannot be imputed"):
        ca.predict_cox(fit, bad, settings())


def test_failed_fold_does_not_change_the_paired_population(monkeypatch):
    d, cfg = population(), settings()
    original = ca.tune_cox
    def fail_one(train, cols, settings, seed):
        if cols == ["age", "gleason", "L"] and seed == cfg.seed:
            raise ValueError("forced unsupported training fold")
        return original(train, cols, settings, seed)
    monkeypatch.setattr(ca, "tune_cox", fail_one)
    predictions, fits, _ = ca.cross_validate(d, cfg)
    metrics, _, _ = ca.evaluate_predictions(predictions, cfg)
    assert metrics.n_patients.eq(len(d)).all()
    assert np.isnan(metrics.set_index("model").loc["augmented", "auc"])
    assert np.isfinite(metrics.set_index("model").loc["clinical", "auc"])
    assert fits.status.str.contains("forced unsupported").any()


def test_refit_bootstrap_actually_repeats_nested_pipeline(monkeypatch):
    cfg = replace(settings(), bootstrap=2)
    calls = []
    original = ca.cross_validate
    def observe(frame, *args, **kwargs):
        calls.append(frame.copy())
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(ca, "cross_validate", observe)
    result = ca.analyze_cohort(population(), cfg)
    assert len(calls) == 3
    assert calls[1].person_id.duplicated().any()
    assert len(result["bootstrap_diagnostics"]) == 2
    assert result["metrics"].auc_lo.isna().all()  # two draws cannot form a reported CI
    assert result["metrics"].interval_method.str.contains("repeating nested").all()


def test_unsupported_horizon_keeps_counts_without_auc():
    d = population()
    d["duration_days"] = 400
    assert np.isnan(ca.auc(d, d.L.to_numpy(), settings()))
    with pytest.raises(ValueError, match="horizon"):
        ca.targets(d, settings())


def test_auc_and_ipcw_screening_match_manuscript_binary_estimator():
    d, cfg = population(), settings()
    expected = fet.fixed_horizon_mean_auc(d.duration_days, d.event, d.B, [1825])
    assert ca.auc(d, d.B.to_numpy(), cfg) == pytest.approx(expected)
    m = ca.screening(d, d.B.to_numpy(), cfg)
    assert .5 * (m["sensitivity"] + m["specificity"]) == pytest.approx(expected)
    # Early censoring is unknown, not a five-year negative.
    d = pd.DataFrame(dict(person_id=[1, 2, 3, 4], duration_days=[100, 200, 2500, 3000], event=[1, 0, 0, 0]))
    m = ca.screening(d, np.array([1, 1, 0, 0]), cfg)
    assert m["ppv"] == 1 and m["npv"] == 1 and m["flag_rate"] == .5


def test_workload_neighbors_and_simple_nested_flag_sets():
    values = np.array([0, 1, 1, 2, 3])
    thresholds = ca.workload_thresholds(values, .5)
    assert (values >= thresholds["lower_workload"]).mean() <= .5
    assert (values >= thresholds["upper_workload"]).mean() >= .5
    d = population()
    pred = ca.frozen_predictions(d, False)
    assert np.all(pred["gleason_ge_m"] >= pred["B"])
    assert np.all(pred["gleason_ge_m_plus1"] <= pred["B"])
    assert set(ca.workload_thresholds(np.ones(5), 0).values()) == {np.inf}


@pytest.mark.parametrize("landmark", [0, 180])
def test_additional_landmarks_use_their_own_frozen_m_and_n(landmark, tmp_path):
    cfg = settings(landmark=landmark)
    d = population().assign(frozen_M=3, frozen_N=5)
    d.loc[d.person_id.mod(5).eq(0), "L"] = 5
    d["B"] = d.L.ge(3).astype(int)
    frozen = ca.frozen_predictions(d, False)
    np.testing.assert_array_equal(frozen["B"], d.L.ge(3))
    np.testing.assert_array_equal(frozen["gleason_ge_m"], (d.L + d.gleason.ge(8)).ge(3))
    np.testing.assert_array_equal(frozen["gleason_ge_m_plus1"], (d.L + d.gleason.ge(8)).ge(4))
    predictions, _, _ = ca.cross_validate(d, cfg)
    assert predictions.frozen_M.eq(3).all() and predictions.frozen_N.eq(5).all()
    assert predictions.loc[predictions.model.eq("augmented"), "score"].notna().all()
    assert ca.landmark_directory(tmp_path, landmark) == tmp_path / f"lm{landmark}_h1825"


def test_landmark_selection_keeps_five_years_and_primary_output():
    assert ca.parse_landmarks() == (90, 0, 180)
    assert ca.parse_landmarks("0,180") == (0, 180)
    for bad in ("", "0,0", "365", "0,", "90.5"):
        with pytest.raises(ValueError): ca.parse_landmarks(bad)
    with pytest.raises(ValueError): ca.Settings(landmark=0, horizon=365)
    assert str(ca.landmark_directory("/tmp/example", 90)) == "/tmp/example"


def test_documented_gleason_availability_and_grade_group():
    raw = pd.DataFrame(dict(DFCI_MRN=[1, 1, 2, 3], gleason_date=["2010-01-01"] * 4,
                            source_note_date=["2020-05-01", "2019-12-01", None, "2019-01-01"],
                            gleason_total=[9, 7, 8, 9], gleason_primary=[4, 3, 4, 3], gleason_secondary=[5, 4, 4, 4]))
    strict = ci.strict_gleason(raw)
    anchors = pd.Series(pd.to_datetime(["2020-01-01"] * 3), index=pd.Index([1, 2, 3], name="DFCI_MRN"))
    selected = ci.gleason_at_clock(strict, anchors, settings()).set_index("person_id")
    assert selected.loc[1, "gleason"] == 7  # future note with historical score excluded
    assert 2 not in selected.index  # no documentation date, not historical specimen date
    assert selected.loc[1, "grade_group"] == 2
    assert np.isnan(selected.loc[3, "grade_group"])  # patterns do not sum to total


def test_gene_negatives_require_matching_report_coverage_and_no_fill_zero():
    specimen = pd.DataFrame({"SEQUENCING_DATE": pd.to_datetime(["2020-01-01"] * 3),
                             "TP53_ANY": [0, 0, 1], "RB1_ANY": [0, np.nan, 0], "PTEN_ANY": [0, 0, 0]},
                            index=pd.Index([1, 2, 3], name="DFCI_MRN"))
    assert not ci.assessable_specimen(specimen, None).genomic_available.any()
    cov = pd.DataFrame(dict(person_id=[1, 2, 3], REPORT_DT=["2020-01-01", "2020-01-01", "2021-01-01"],
                            TP53_assessable=1, RB1_assessable=1, PTEN_assessable=1))
    assessed = ci.assessable_specimen(specimen, cov).set_index("person_id")
    assert assessed.genomic_available.to_dict() == {1: True, 2: False, 3: False}
    assert np.isnan(assessed.loc[2, "RB1"])
    for g in ca.GENES:
        cov[g + "_pathogenic"] = 0
    with pytest.raises(ValueError, match="annotation_rule"):
        ci.assessable_specimen(specimen, cov)


def test_four_groups_include_empty_cells_and_competing_death():
    d = pd.DataFrame(dict(person_id=[1, 2, 3, 4], endpoint="platinum", gleason=7, B=1,
                          duration_days=[100, 200, 2500, 3000], event=[1, 0, 0, 0],
                          death_days=[np.nan, 150, np.nan, np.nan]))
    risk, differences = ca.four_group_risks(d, settings(), death_available=True)
    assert len(risk) == 8 and len(differences) == 2
    assert risk.loc[risk.n_patients.eq(0), "risk"].isna().all()
    assert ca.cumulative_incidence(d, 1825) == pytest.approx(.25)
    assert ca.cumulative_incidence(d.assign(death_days=[100, 150, np.nan, np.nan]), 1825) == pytest.approx(.25)


def test_run_archive_and_manifest_exclude_patient_outputs(tmp_path):
    cfg = settings()
    ca.start_run(tmp_path, cfg)
    ca.write_analysis({"metrics": pd.DataFrame({"auc": [.7]}), "predictions": population(),
                       "folds": pd.DataFrame({"person_id": [1], "fold": [0]})}, tmp_path)
    ca.complete_run(tmp_path, cfg)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert [f["name"] for f in manifest["files"]] == ["metrics.csv"]
    ca.start_run(tmp_path, cfg)
    assert not (tmp_path / "aggregate").exists()
    assert list((tmp_path / "previous_runs").glob("*/patient_level/predictions.csv"))
    assert json.loads((tmp_path / "run_manifest.json").read_text())["status"] == "running"


def test_analysis_finishes_without_optional_genomic_or_grade_group_inputs(tmp_path):
    d = population(endpoints=("platinum", "avpc", "nepc"))
    d["grade_group"], d["death_days"] = np.nan, np.nan
    for g in ca.GENES:
        d[g], d[g + "_pathogenic"] = np.nan, np.nan
    frames = dict(gleason=d, genomic=d.iloc[:0], genomic_adt=d.iloc[:0], full_checklist=d,
                  historical_paired=d, death_source_available=False)
    result = ca.run_analyses(frames, settings(), progress=lambda _: None)
    assert set(result["metrics"].cohort) == {"gleason"}
    assert result["analysis_status"].set_index("cohort").loc["genomic", "status"].startswith("unavailable")
    assert set(result["four_groups"].endpoint) == {"platinum", "avpc", "nepc"}
    ca.write_analysis(result, tmp_path)
    ca.complete_run(tmp_path, settings())
    for path in (tmp_path / "aggregate").glob("*.csv"):
        assert "person_id" not in pd.read_csv(path).columns


def frozen_bundle(tmp_path, landmark=90):
    names = ["Alanine_aminotransferase__min", "Alkaline_phosphatase__max", "Testosterone__max"]
    binarizer = FeatureBinarizer()
    binarizer.feature_columns = names
    binarizer.medians = dict(zip(names, [10., 120., 600.]))
    binarizer.coverage = dict.fromkeys(names, 1.)
    binarizer.items = [dict(feature=f, group=f.rsplit("__", 1)[0], op=op, threshold=t)
                       for f, op, t in zip(names, ["<=", ">=", "<="], [14.8, 101., 552.])]
    binarizer._fitted = True
    model = dict(landmark_days=landmark, horizon_days=1825, config="both",
                 checklist=dict(schema_version=1, binarizer=binarizer.to_dict(),
                                model=ChecklistModel([0, 1, 2], 2, 3, {}).to_dict(),
                                calibrator=ProbabilityCalibrator("constant", {"probability": .2}).to_dict()))
    path = tmp_path / "no_DFCI" / "checklist.json"
    path.parent.mkdir()
    path.write_text(json.dumps(dict(format="caia-federated-model-bundle", format_version=1,
                                    model_family="predictive_checklist", training_sites=["FH", "JHU", "MSK"], models=[model])))
    return path


def test_feature_set_export_without_training_arm_and_frozen_imputation(tmp_path):
    path = frozen_bundle(tmp_path)
    bundle, items = ci.frozen_model(path, settings())
    assert list(items.threshold_value) == [15, 100, 550]
    from survival_common.federated_inference import predict_checklist
    wide = pd.DataFrame({"Alanine_aminotransferase__min": [np.nan, 30],
                         "Alkaline_phosphatase__max": [np.nan, 20], "Testosterone__max": [np.nan, 100]})
    pred = predict_checklist(wide, bundle["models"][0])
    assert list(pred.score) == [2, 1]  # absent ALT/ALP use frozen medians, never normal/zero
    assert list(pred.flag) == [1, 0]
    raw = json.loads(path.read_text())
    raw["training_sites"][0] = "DFCI"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="non-DFCI"):
        ci.frozen_model(path, settings())


@pytest.mark.parametrize("landmark", [0, 180])
def test_additional_landmark_model_does_not_inherit_day90_thresholds(tmp_path, landmark):
    path = frozen_bundle(tmp_path)
    raw = json.loads(path.read_text())
    import copy
    extra = copy.deepcopy(raw["models"][0])
    extra["landmark_days"] = landmark
    extra["checklist"]["model"].update(threshold=1, selected_indices=[0, 2])
    extra["checklist"]["binarizer"]["items"][0]["threshold"] = 20
    raw["models"].append(extra)
    path.write_text(json.dumps(raw))
    bundle, items = ci.frozen_model(path, settings(landmark=landmark))
    assert bundle["models"][0]["landmark_days"] == landmark
    assert len(items) == 2 and items.M.eq(1).all()
    assert items.threshold_value.iloc[0] == 20
    _, primary = ci.frozen_model(path, settings())
    assert len(primary) == 3 and primary.M.eq(2).all()


@pytest.mark.parametrize("landmark", [90, 0, 180])
def test_input_pipeline_recomputes_clocks_excludes_future_information_and_audits_reproduction(tmp_path, landmark):
    anchor = pd.Timestamp("2020-01-01")
    n = 30
    local = []
    for pid in range(n):
        for name, value in [("ALT", 10 + pid), ("ALP", 90 + pid), ("T", 530 + pid)]:
            for day in (-300, 89, 90, 150):
                row = dict(DFCI_MRN=pid, AGE_AT_TREATMENTSTART=70, DIAGNOSIS_DATE=anchor-pd.Timedelta(days=365),
                           TREATMENT_ANCHOR_DATE=anchor, LAST_CONTACT_DATE=anchor+pd.Timedelta(days=3100),
                           LAB_NAME=name, LAB_VALUE=value if day < 90 else 10000,
                           LAB_DATE=anchor+pd.Timedelta(days=day))
                for ep in ("PLATINUM", "AVPC", "NEPC"):
                    row[ep] = int(pid % 3 == 0)
                    row[ep + "_DATE"] = anchor+pd.Timedelta(days=500+pid*10) if row[ep] else pd.NaT
                local.append(row)
    local = pd.DataFrame(local)
    raw_g = pd.DataFrame(dict(DFCI_MRN=np.arange(n), gleason_date=anchor-pd.Timedelta(days=365),
                              gleason_total=8, source_note_date=anchor-pd.Timedelta(days=20)))
    raw_g.loc[5, "source_note_date"] = pd.NaT
    offsets = np.resize([-200, -90, 0, 40, 120], n)
    somatic = pd.DataFrame(dict(DFCI_MRN=np.arange(n), REPORT_DT=anchor+pd.to_timedelta(offsets,unit="D"),
                                TP53_SNV=np.arange(n)%2, RB1_DEL=0, PTEN_DEL=0))
    manifest = pd.DataFrame(dict(FEATURE=["TP53_SNV", "RB1_DEL", "PTEN_DEL"]))
    coverage = pd.DataFrame(dict(person_id=np.arange(n),REPORT_DT=somatic.REPORT_DT,
                                 TP53_assessable=1,RB1_assessable=1,PTEN_assessable=1))
    coverage.loc[2, "RB1_assessable"] = 0
    mapping = pd.DataFrame(dict(mapping_status="mapped",collapsed_measurement=["ALT", "ALP", "T"],
                                omop_measurement_name=["Alanine aminotransferase", "Alkaline phosphatase", "Testosterone"]))
    paths = {}
    for name, table in dict(local=local, gleason=raw_g, somatic=somatic, somatic_manifest=manifest,
                             coverage=coverage, lab_mapping=mapping).items():
        paths[name] = tmp_path / f"{name}.csv"
        table.to_csv(paths[name], index=False)
    paths.update(bundle=frozen_bundle(tmp_path, landmark), transfer=tmp_path / "transfer", death=None)
    frames, audit, definitions = ci.build_inputs(paths, settings(landmark=landmark))
    assert 5 not in set(frames["gleason"].person_id)
    assert 2 not in set(frames["genomic"].person_id)
    assert not set(np.flatnonzero(offsets <= -landmark)) & set(frames["genomic"].person_id)
    assert not set(np.flatnonzero(offsets >= landmark)) & set(frames["genomic_adt"].person_id)
    adt = frames["full_checklist"].query("endpoint == 'platinum'").set_index("person_id")
    # Measurements on or after each landmark are excluded on the ADT clock.
    assert adt.loc[4, "item_3"] == int(landmark <= 90)
    genomic = frames["genomic"].query("endpoint == 'platinum'").set_index("person_id")
    assert genomic.loc[4, "item_3"] == 0  # day 150 is available by report+90 = ADT+210
    assert genomic.loc[4, "duration_days"] == adt.loc[4, "duration_days"] - 120
    assert audit["cohort_flow"].stage.str.contains("not after ADT").any()
    assert "no lower lookback" in definitions["lab_window"]
    with pytest.raises(ValueError, match="unresolved"):
        ci.require_reproduction(audit)
    # Build exact historical reference files and verify all machine-precision checks.
    root = paths["transfer"] / "checklist_grid_guidelines"
    root.mkdir(parents=True)
    for frame, filename in [(frames["full_checklist"], "transfer_checklist_patients.csv"),
                             (frames["historical_paired"], "transfer_checklist_comparison_patients.csv")]:
        frame.rename(columns={"L":"risk_score", "B":"flag"}).assign(landmark_days=landmark,model_horizon_days=1825).to_csv(root/filename,index=False)
    clin_g, som, features = ci.fcc.load_clinical_sources(paths["gleason"],paths["somatic"],paths["somatic_manifest"])
    specimen = ci.fcc.select_specimen(local,som,features)
    seq_local = ci.fcc.sequencing_anchored_local(local,specimen)
    clinical = ci.fcc.build_clinical_features(seq_local,[landmark],gleason=clin_g,specimen=specimen,somatic_features=features)
    comparison=[]
    for endpoint,d in frames["historical_paired"].groupby("endpoint"):
        rules={"checklist":d.B,"gleason":d.GLEASON_SCORE.ge(8),"tsg":d[list(ci.fcc.TRIO_FEATURES)].eq(1).any(axis=1)}
        for rule,score in rules.items():
            comparison.append(dict(endpoint=endpoint,rule=rule,landmark_days=landmark,horizon_days=1825,
                n_patients=len(d),n_events=int(d.event.sum()),n_positive=int(score.sum()),
                auc=fet.fixed_horizon_mean_auc(d.duration_days,d.event,score,[1825])))
    pd.DataFrame(comparison).to_csv(root/"transfer_checklist_comparison.csv",index=False)
    all_seq,_,_=ci.score_clock(seq_local,ci.frozen_model(paths["bundle"],settings(landmark=landmark))[0]["models"][0],
                               fet.load_lab_mapping(paths["lab_mapping"]),settings(landmark=landmark),"sequencing")
    checked,_=ci.reproduction_audit(frames["full_checklist"],all_seq,clinical,paths["transfer"],settings(landmark=landmark))
    assert checked.loc[~checked.check.str.startswith("manuscript"),"status"].eq("ok").all()
    if landmark != 90:
        assert not checked.check.str.startswith("manuscript").any()
        ci.require_reproduction({"reproduction": checked})
    if landmark == 0:
        assert 4 in set(frames["genomic"].person_id)  # anchoring report available at sequencing day 0
        assert 1 in set(frames["genomic_adt"].person_id)  # pre-ADT report only on the ADT day-0 clock
    # Changing a historical item count must fail, even if the resulting flag stays the same.
    ref=pd.read_csv(root/"transfer_checklist_patients.csv")
    ref.loc[0,"risk_score"]+=1
    ref.to_csv(root/"transfer_checklist_patients.csv",index=False)
    checked,_=ci.reproduction_audit(frames["full_checklist"],all_seq,clinical,paths["transfer"],settings(landmark=landmark))
    assert checked.loc[checked.clock.eq("adt") & checked.check.eq("L"),"status"].eq("mismatch").any()
