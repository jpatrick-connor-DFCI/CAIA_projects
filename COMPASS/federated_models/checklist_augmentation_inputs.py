"""Figure 6 reproduction and temporally valid augmentation input adapters.

Historical Figure 6 is reproduced with its original clinical loader. Revised
cohorts require documented Gleason availability and explicit gene assessability.
Missing assessability is never inferred from an unaltered entry in a matrix.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from COMPASS.federated_models import federated_clinical_comparison as fcc
from COMPASS.federated_models import checklist_augmentation as ca
from survival_common import federated_endpoint_transfer as fet, federated_scoring as fs
from survival_common.federated_inference import load_bundle, predict_checklist, _checklist_components


def read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path, low_memory=False)


def require_input_files(paths):
    required = [Path(paths[k]) for k in ("local", "bundle", "gleason", "somatic", "somatic_manifest", "lab_mapping")]
    required += [Path(paths["transfer"]) / "checklist_grid_guidelines" / name for name in (
        "transfer_checklist_patients.csv", "transfer_checklist_comparison_patients.csv", "transfer_checklist_comparison.csv")]
    required += [Path(paths[k]) for k in ("coverage", "death") if paths.get(k)]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Augmentation inputs are unavailable; no analysis started:\n" + "\n".join(missing))


def dates(values):
    return pd.to_datetime(values, errors="coerce", format="mixed").dt.tz_localize(None)


def normalize_ids(d):
    d = d.copy()
    col = "person_id" if "person_id" in d else "DFCI_MRN"
    ids = pd.to_numeric(d[col], errors="raise")
    if ids.isna().any() or ids.mod(1).ne(0).any():
        raise ValueError("Patient IDs must be nonmissing integers")
    d["person_id"] = ids.astype("int64")
    return d


def source_record(path, content=False):
    path = Path(path)
    record = dict(path=str(path.resolve()), exists=path.is_file())
    if path.is_file():
        stat = path.stat()
        record.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if content:
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def frozen_model(bundle_path, cfg):
    bundle = load_bundle(bundle_path)
    if bundle.get("model_family") != "predictive_checklist":
        raise ValueError("Expected the no-DFCI predictive checklist bundle")
    models = [m for m in bundle["models"] if m.get("config") == "both"
              and int(m["landmark_days"]) == cfg.landmark and int(m.get("horizon_days") or 0) == cfg.horizon]
    if len(models) != 1:
        raise ValueError(f"Expected exactly one both/day-{cfg.landmark}/five-year checklist")
    model = models[0]
    # The feature-set exporter records training_sites, but no training_arm field.
    # Retain its no_DFCI delivery provenance and reject any declared DFCI trainer.
    sites = bundle.get("training_sites", [])
    no_dfci_path = any(p.lower() == "no_dfci" or p.lower().endswith("_no_dfci") for p in Path(bundle_path).parts)
    declared = str(model.get("training_arm", "")).lower()
    if (not sites or len(sites) != 3 or any("dfci" in str(s).lower() or "dana" in str(s).lower() for s in sites)
            or (declared != "no_dfci" and not no_dfci_path) or declared not in ("", "no_dfci")):
        raise ValueError("Primary checklist requires three non-DFCI training sites and no_DFCI delivery provenance")
    if bundle.get("endpoint", "platinum") != "platinum":
        raise ValueError("Checklist must be platinum trained")
    identity = model.get("checklist", {}).get("identity", {})
    for key, expected_value in dict(endpoint="platinum", arm="adt", cohort="all", exclusion="none").items():
        if key in identity and identity[key] != expected_value:
            raise ValueError(f"Reference checklist identity requires {key}={expected_value}")
    selected = dict(bundle, models=[model])
    guide = fcc.guideline_bundle(selected)
    model = guide["models"][0]
    items = fcc.guideline_items(guide, selected)
    if items.empty or len(set(items.M)) != 1 or not 1 <= items.M.iloc[0] <= len(items):
        raise ValueError("Invalid frozen M-of-N checklist")
    if cfg.landmark == 90 and (len(items) != 3 or set(items.M) != {2}):
        raise ValueError("Reference is the frozen 2-of-3 checklist; resolve the bundle discrepancy")
    expected = [("alanine", "__min", "<=", 15), ("alkaline", "__max", ">=", 100),
                ("testosterone", "__max", "<=", 550)]
    for token, stat, op, threshold in expected if cfg.landmark == 90 else []:
        match = items.feature.str.lower().str.contains(token, regex=False) & items.feature.str.endswith(stat)
        if match.sum() != 1 or not (items.loc[match, "op"].iloc[0] == op
                                  and np.isclose(items.loc[match, "threshold_value"].iloc[0], threshold)):
            raise ValueError(f"Frozen reference mismatch: expected {token} {stat} {op} {threshold}")
    items["units"] = ["ng/dL" if "testosterone" in x.lower() else "U/L"
                      if "alanine" in x.lower() or "alkaline" in x.lower() else None for x in items.feature]
    return guide, items


def score_clock(local, model, mapping, cfg, clock):
    rows, flows, missingness = [], [], []
    for endpoint in ("platinum", "avpc", "nepc"):
        long, prep = fet.to_federated_long(local, endpoint, mapping)
        wide, stats = fs.build_landmark_frame(long, cfg.landmark)
        flows.extend([dict(clock=clock, endpoint=endpoint, stage=key, n=value)
                      for key, value in {**prep, **stats}.items() if key.startswith("n_")])
        if wide.empty:
            continue
        pred = predict_checklist(wide, model)
        binarizer, checklist, _ = _checklist_components(model)
        matrix = binarizer.transform(wide)
        selected = checklist.selected_indices
        out = pd.DataFrame({"person_id": wide[fs.ID_COL].to_numpy(), "endpoint": endpoint,
                            "duration_days": wide[fs.DURATION_COL].to_numpy(), "event": wide[fs.EVENT_COL].to_numpy(),
                            "age": wide[fs.AGE_COL].to_numpy(), "L": pred.score.to_numpy(), "B": pred.flag.to_numpy(),
                            "clock": clock, "frozen_M": checklist.threshold, "frozen_N": len(selected)})
        complete = np.ones(len(out), bool)
        for position, index in enumerate(selected, start=1):
            item = binarizer.items[index]
            raw = pd.to_numeric(wide[item["feature"]], errors="coerce").replace([np.inf, -np.inf], np.nan)
            absent = raw.isna().to_numpy()
            out[f"item_{position}"], out[f"item_{position}_missing"] = matrix[:, index], absent
            complete &= ~absent
            missingness.append(dict(clock=clock, endpoint=endpoint, item=position, feature=item["feature"],
                                    n_patients=len(out), n_missing=int(absent.sum()),
                                    frozen_imputation=binarizer.medians[item["feature"]]))
        out["complete_labs"] = complete
        out["evaluable"] = np.isfinite(out.L) & np.isfinite(out.B)
        rows.append(normalize_ids(out))
    columns = ["person_id", "endpoint", "duration_days", "event", "age", "L", "B", "clock", "complete_labs", "evaluable", "frozen_M", "frozen_N"]
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns),
            pd.DataFrame(flows), pd.DataFrame(missingness))


def strict_gleason(raw):
    """No fallback from missing documentation date to a historical specimen date."""
    d = normalize_ids(raw)
    d["DFCI_MRN"] = d.person_id
    d["gleason_date"] = dates(d.gleason_date)
    note = dates(d.source_note_date) if "source_note_date" in d else pd.Series(pd.NaT, index=d.index)
    d[fcc.sg.GLEASON_AVAILABLE_DATE] = pd.concat([d.gleason_date, note], axis=1).max(axis=1).where(note.notna())
    d["gleason"] = pd.to_numeric(d.gleason_total, errors="coerce")
    d.loc[~d.gleason.between(2, 10), "gleason"] = np.nan
    d["grade_group"] = np.nan
    if {"gleason_primary", "gleason_secondary"} <= set(d):
        p, s = (pd.to_numeric(d[c], errors="coerce") for c in ("gleason_primary", "gleason_secondary"))
        consistent = p.between(1, 5) & s.between(1, 5) & (p + s).eq(d.gleason)
        gg = np.select([d.gleason.le(6), p.eq(3) & s.eq(4), p.eq(4) & s.eq(3), d.gleason.eq(8), d.gleason.ge(9)],
                       [1., 2., 3., 4., 5.], default=np.nan)
        d["grade_group"] = np.where(consistent, gg, np.nan)
    return d


def gleason_at_clock(gleason, anchors, cfg):
    cutoffs = anchors.rename("anchor").reset_index()
    cutoffs["_landmark_date"] = cutoffs.anchor + pd.to_timedelta(cfg.landmark - 1, unit="D")
    out = fcc.sg.closest_available_by_landmark(gleason, cutoffs, date_col="gleason_date",
            available_date_col=fcc.sg.GLEASON_AVAILABLE_DATE, value_cols=["gleason", "grade_group"])
    return normalize_ids(out.reset_index())[["person_id", "gleason", "grade_group"]]


def assessable_specimen(specimen, coverage):
    out = normalize_ids(specimen.reset_index())
    out["REPORT_DT"] = dates(out[fcc.sg.SEQUENCING_DATE])
    if coverage is None:
        for gene in ca.GENES:
            out[gene], out[gene + "_pathogenic"] = np.nan, np.nan
        out["genomic_available"] = False
        return out
    cov = normalize_ids(coverage)
    required = ["REPORT_DT", *[g + "_assessable" for g in ca.GENES]]
    if not set(required) <= set(cov):
        raise ValueError(f"Coverage table requires person_id/DFCI_MRN and {required}")
    cov["REPORT_DT"] = dates(cov.REPORT_DT)
    if cov.REPORT_DT.isna().any() or cov.duplicated(["person_id", "REPORT_DT"]).any():
        raise ValueError("Coverage requires one dated record per patient/report (resolve specimen ties upstream)")
    for g in ca.GENES:
        if not cov[g + "_assessable"].dropna().isin([0, 1, True, False]).all():
            raise ValueError(f"{g}_assessable must be 0/1; missing means unavailable")
    keep = ["person_id", *required, *[g + "_pathogenic" for g in ca.GENES if g + "_pathogenic" in cov]]
    pathogenic = all(g + "_pathogenic" in cov for g in ca.GENES)
    if pathogenic:
        if "annotation_rule" not in cov or cov.annotation_rule.fillna("").str.strip().eq("").any():
            raise ValueError("Pathogenic sensitivity needs a frozen annotation_rule for every report")
        for g in ca.GENES:
            if not cov[g + "_pathogenic"].dropna().isin([0, 1]).all():
                raise ValueError("Pathogenic calls must be 0/1/unknown")
    out = out.merge(cov[keep], on=["person_id", "REPORT_DT"], how="left", validate="one_to_one")
    for g in ca.GENES:
        calls = pd.to_numeric(out[g + "_ANY"], errors="coerce")
        out[g] = calls.where(out[g + "_assessable"].eq(1) & calls.isin([0, 1]))
        if g + "_pathogenic" not in out:
            out[g + "_pathogenic"] = np.nan
        out[g + "_pathogenic"] = out[g + "_pathogenic"].where(out[g + "_assessable"].eq(1))
    out["genomic_available"] = out[list(ca.GENES)].notna().all(axis=1)
    return out


def reproduction_audit(adt, seq, clinical_seq, transfer_root, cfg):
    root = Path(transfer_root) / "checklist_grid_guidelines"
    paired = seq.merge(normalize_ids(clinical_seq)[["person_id", fcc.GLEASON_FEATURE, fcc.HAS_SEQUENCING, *fcc.TRIO_FEATURES]],
                       on="person_id", how="left", validate="many_to_one")
    paired = paired.loc[paired[fcc.GLEASON_FEATURE].notna() & paired[fcc.HAS_SEQUENCING].notna()]
    audits = []
    for clock, actual, filename in [("adt", adt, "transfer_checklist_patients.csv"),
                                    ("sequencing", paired, "transfer_checklist_comparison_patients.csv")]:
        path = root / filename
        if not path.exists():
            audits.append(dict(clock=clock, endpoint="all", check="patient reproduction", status="missing reference", file=str(path)))
            continue
        ref = normalize_ids(read_table(path))
        ref = ref.loc[ref.landmark_days.eq(cfg.landmark) & ref.model_horizon_days.eq(cfg.horizon)]
        for ep in ("platinum", "avpc", "nepc"):
            a, r = actual.loc[actual.endpoint.eq(ep)], ref.loc[ref.endpoint.eq(ep)]
            if r.person_id.duplicated().any() or a.person_id.duplicated().any():
                raise ValueError("Reproduction has duplicate endpoint/patient rows")
            joined = a.merge(r, on="person_id", how="outer", suffixes=("", "_ref"), indicator=True)
            aligned = joined._merge.eq("both").all() and len(joined) > 0
            for new, old in [("L", "risk_score"), ("B", "flag"), ("duration_days", "duration_days_ref"), ("event", "event_ref")]:
                ok = aligned and np.allclose(joined[new], joined[old], rtol=1e-9, atol=1e-9, equal_nan=True)
                audits.append(dict(clock=clock, endpoint=ep, check=new, status="ok" if ok else "mismatch",
                                   n_actual=len(a), n_reference=len(r), file=str(path)))
    path = root / "transfer_checklist_comparison.csv"
    if path.exists():
        ref = read_table(path)
        ref = ref.loc[ref.landmark_days.eq(cfg.landmark) & ref.horizon_days.eq(cfg.horizon)]
        for endpoint, d in paired.groupby("endpoint"):
            rules = {"checklist": d.B, "gleason": d[fcc.GLEASON_FEATURE].ge(8),
                     "tsg": d[list(fcc.TRIO_FEATURES)].eq(1).any(axis=1)}
            for rule, score in rules.items():
                r = ref.loc[ref.endpoint.eq(endpoint) & ref.rule.eq(rule)]
                value = fet.fixed_horizon_mean_auc(d.duration_days, d.event, np.asarray(score, float), [cfg.horizon])
                ok = len(r) == 1 and np.isclose(value, r.auc.iloc[0], atol=1e-8, equal_nan=True)
                for col, val in [("n_patients", len(d)), ("n_events", int(d.event.sum())), ("n_positive", int(score.sum()))]:
                    ok = ok and int(r[col].iloc[0]) == val
                audits.append(dict(clock="sequencing", endpoint=endpoint, check=rule + " AUC/counts",
                                   status="ok" if ok else "mismatch", actual_auc=value,
                                   reference_auc=r.auc.iloc[0] if len(r) == 1 else np.nan, file=str(path)))
    else:
        audits.append(dict(clock="sequencing", endpoint="all", check="AUC/counts", status="missing reference", file=str(path)))
    # Published targets refer to day 90 only; other landmarks reproduce their own notebook-04 grid cells.
    targets = [("platinum", "checklist", .73), ("platinum", "gleason", .71),
               ("avpc", "checklist", .67), ("avpc", "gleason", .65)] if cfg.landmark == 90 else []
    for ep, rule, target in targets:
        d = paired.loc[paired.endpoint.eq(ep)]
        score = d.B if rule == "checklist" else d[fcc.GLEASON_FEATURE].ge(8)
        value = fet.fixed_horizon_mean_auc(d.duration_days, d.event, np.asarray(score, float), [cfg.horizon])
        audits.append(dict(clock="sequencing", endpoint=ep, check="manuscript " + rule,
                           status="ok" if np.isfinite(value) and abs(value - target) <= .00501 else "mismatch",
                           actual_auc=value, reference_auc=target))
    return pd.DataFrame(audits), paired


def build_inputs(paths, cfg):
    local = fet.read_local_frame(paths["local"], ("platinum", "avpc", "nepc"))
    guide, items = frozen_model(paths["bundle"], cfg)
    model = guide["models"][0]
    mapping = fet.load_lab_mapping(paths["lab_mapping"])
    historical_g, somatic, features = fcc.load_clinical_sources(paths["gleason"], paths["somatic"], paths["somatic_manifest"])
    specimen = fcc.select_specimen(local, somatic, features)
    anchors = fcc._patient_anchors(local).anchor.rename_axis("DFCI_MRN")
    local_seq = fcc.sequencing_anchored_local(local, specimen)
    adt, flow_adt, miss_adt = score_clock(local, model, mapping, cfg, "adt")
    seq, flow_seq, miss_seq = score_clock(local_seq, model, mapping, cfg, "sequencing")
    historical_clin = fcc.build_clinical_features(local_seq, [cfg.landmark], gleason=historical_g,
                                                specimen=specimen, somatic_features=features)
    reproduction, historical_paired = reproduction_audit(adt, seq, historical_clin, paths["transfer"], cfg)
    raw_g = read_table(paths["gleason"])
    strict_g = strict_gleason(raw_g)
    coverage = read_table(paths["coverage"]) if paths.get("coverage") else None
    genes_seq = assessable_specimen(specimen, coverage)
    # ADT sensitivity selects the closest-to-ADT report available strictly before this landmark.
    eligible = somatic.merge(anchors.rename("_adt"), left_on="DFCI_MRN", right_index=True, validate="many_to_one")
    eligible = eligible.loc[eligible[fcc.sg.SEQUENCING_DATE].lt(eligible._adt + pd.Timedelta(days=cfg.landmark))]
    early_specimen = fcc.select_specimen(local, eligible, features)
    genes_adt = assessable_specimen(early_specimen, coverage)
    death = None
    if paths.get("death"):
        raw_death = normalize_ids(read_table(paths["death"]))
        raw_death["death_date"] = dates(raw_death.DEATH_DATE)
        if raw_death.groupby("person_id").death_date.nunique().gt(1).any():
            raise ValueError("Conflicting death dates")
        death = raw_death.groupby("person_id").death_date.first()
    frames, flow, support = {}, [], []
    for cohort, scores, clock_anchors, genes in [
        ("gleason", adt, anchors, None),
        ("genomic", seq, specimen[fcc.sg.SEQUENCING_DATE].rename_axis("DFCI_MRN"), genes_seq),
        ("genomic_adt", adt, anchors, genes_adt),
    ]:
        d = scores.merge(gleason_at_clock(strict_g, clock_anchors, cfg), on="person_id", how="left", validate="many_to_one")
        d["prediction_date"] = d.person_id.map(clock_anchors) + pd.Timedelta(days=cfg.landmark)
        d["adt_date"] = d.person_id.map(anchors)
        d["death_days"] = ((d.person_id.map(death) - d.prediction_date).dt.days if death is not None else np.nan)
        if genes is not None:
            cols = ["person_id", "REPORT_DT", "genomic_available", *ca.GENES, *[g + "_pathogenic" for g in ca.GENES]]
            d = d.merge(genes[cols], on="person_id", how="left", validate="many_to_one")
        reasons = pd.Series("included", index=d.index)
        # Ordered exclusions make counts add up, with endpoint-specific risk sets already built by fs.
        gates = [(d.prediction_date.isna(), "missing prediction date"),
                 (d.prediction_date.le(d.adt_date) if cohort == "genomic" else pd.Series(False, index=d.index), "genomic landmark not after ADT"),
                 (d.death_days.le(0), "death by landmark"), (~d.evaluable, "checklist not evaluable"),
                 (d.gleason.isna(), "no documented Gleason available")]
        if genes is not None:
            # On the sequencing clock the anchoring report is known at time zero.
            # The ADT-clock sensitivity retains strict pre-landmark availability.
            late_report = d.REPORT_DT.gt(d.prediction_date) if cohort == "genomic" else d.REPORT_DT.ge(d.prediction_date)
            gates += [(~d.genomic_available.eq(True), "unassessable or unavailable TSG"),
                      (late_report | d.REPORT_DT.isna(), "report unavailable at landmark")]
        for bad, label in gates:
            reasons.loc[reasons.eq("included") & bad] = label
        for ep in ("platinum", "avpc", "nepc"):
            for reason, n in reasons.loc[d.endpoint.eq(ep)].value_counts().items():
                flow.append(dict(cohort=cohort, endpoint=ep, stage=reason, n=int(n)))
        d = d.loc[reasons.eq("included")].copy()
        frames[cohort] = d
        for ep, group in d.groupby("endpoint"):
            row = dict(cohort=cohort, clock=group.clock.iloc[0], endpoint=ep, **ca.counts(group, cfg))
            try:
                _, _, diagnostics = ca.targets(group, cfg)
                row.update(diagnostics)
            except ValueError as error:
                row.update(status=str(error))
            support.append(row)
    frames["full_checklist"] = adt
    frames["historical_paired"] = historical_paired
    frames["death_source_available"] = death is not None
    raw_somatic = read_table(paths["somatic"])
    annotation_audit = []
    for gene in ca.GENES:
        for kind in ("SNV", "DEL", "AMP", "SV"):
            col = f"{gene}_{kind}"
            annotation_audit.append(dict(gene=gene, variant_class=kind, column_available=col in raw_somatic,
                                         positive_records=int(pd.to_numeric(raw_somatic[col], errors="coerce").eq(1).sum()) if col in raw_somatic else np.nan,
                                         definition="Historical ANY includes SNV/DEL/AMP/SV; no functional-loss claim"))
    status = pd.DataFrame([
        dict(analysis="genomic assessability", status="provided" if coverage is not None else "unavailable: no coverage table"),
        dict(analysis="pathogenic sensitivity", status="provided" if coverage is not None and all(g + "_pathogenic" in coverage for g in ca.GENES) else "unavailable: no frozen annotation rules/calls"),
        dict(analysis="VUS and functional-loss audit", status="Wide ANY indicators cannot distinguish VUS/pathogenicity or establish functional loss; external annotations required"),
        dict(analysis="Grade Group", status="available where patterns agree" if strict_g.grade_group.notna().any() else "unavailable: primary/secondary patterns missing or inconsistent"),
        dict(analysis="competing death", status="provided" if death is not None else "unavailable: no verified death-date source"),
        dict(analysis="Gleason documentation", status=f"{strict_g[fcc.sg.GLEASON_AVAILABLE_DATE].isna().sum()} records have no documented availability"),
    ])
    definitions = dict(settings=asdict(cfg), training_sites=guide.get("training_sites"),
        frozen_model=model, frozen_items=items.to_dict("records"),
        lab_mapping=mapping, lab_window=f"all measurements strictly before anchor+{cfg.landmark}; no lower lookback bound",
        lab_units="Existing transfer values used without additional conversions; canonical units as in the existing checklist renderer",
        lab_missingness="Frozen binarizer training medians; not normal/zero; complete-lab sensitivity",
        genomic_clock=f"Original closest-to-ADT report +{cfg.landmark}; must be after ADT; inputs recomputed on this clock",
        analysis_role="primary" if cfg.landmark == 90 else "additional landmark requested after the original plan",
        gleason="Closest observation among scores documented strictly before landmark; no specimen-date fallback",
        gene_primary="Manuscript ANY (SNV/DEL/AMP/SV), requiring explicit gene assessability",
        gene_sensitivity="Externally specified pathogenic calls with annotation_rule; ≥2 genes is not proof of functional loss",
        censoring="Original endpoint-specific transfer risk sets and 3650-day cap; evaluation-cohort reverse KM IPCW",
        probability="Cox baseline hazards fitted within training; five-year net event risk, not competing-risk CIF",
        sources={k: source_record(v, content=k in ("bundle", "somatic_manifest", "lab_mapping", "coverage"))
                 for k, v in paths.items() if v and k != "transfer"},
        code={p.name: source_record(p, content=True) for p in [Path(__file__), Path(ca.__file__), Path(fet.__file__), Path(fs.__file__),
              Path(fcc.__file__), Path(fcc.sg.__file__), Path(__file__).with_name("Figure_6_augmentation_analysis_plan.md")]},
        pathogenic_annotation_rules=coverage.annotation_rule.dropna().unique().tolist() if coverage is not None and "annotation_rule" in coverage else [])
    offset = specimen[fcc.sg.SEQUENCING_DATE] + pd.Timedelta(days=cfg.landmark) - anchors.reindex(specimen.index)
    temporal_audit = pd.DataFrame([
        dict(stage="selected sequencing reports", n=len(specimen)),
        dict(stage="genomic prediction landmark before ADT (excluded)", n=int(offset.dt.days.lt(0).sum())),
        dict(stage="genomic prediction landmark on ADT (excluded)", n=int(offset.dt.days.eq(0).sum())),
        dict(stage=f"report available before ADT+{cfg.landmark} (sensitivity)", n=len(early_specimen)),
    ])
    tables = dict(reproduction=reproduction, frozen_items=items, cohort_flow=pd.DataFrame(flow), temporal_audit=temporal_audit,
                  transfer_flow=pd.concat([flow_adt, flow_seq], ignore_index=True),
                  missingness=pd.concat([miss_adt, miss_seq], ignore_index=True), followup_support=pd.DataFrame(support),
                  annotation_audit=pd.DataFrame(annotation_audit), availability=status)
    tables = {name: table.assign(landmark_days=cfg.landmark, horizon_days=cfg.horizon)
              for name, table in tables.items()}
    return frames, tables, definitions


def require_reproduction(tables):
    failed = tables["reproduction"].loc[lambda d: d.status.ne("ok")]
    if len(failed):
        raise ValueError(f"Figure 6 reproduction has {len(failed)} unresolved checks. Review aggregate/reproduction.csv before fitting; do not change definitions to match performance.")
