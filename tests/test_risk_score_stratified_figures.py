"""The stratified risk-score figures must not invent data they do not have.

The failure modes that matter here are silent ones: treating an unsequenced
patient as wild-type, averaging two landmarks' scores into one curve, or
comparing a feature's C-index against a risk score measured on a different
cohort. Each of those would produce a plausible-looking figure that overstates
or understates the model, so they are asserted against directly.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.survival_analysis import risk_score_stratified_figures as rsf  # noqa: E402


ID = "DFCI_MRN"


def _risks(n=8, endpoint="platinum", landmark=180):
    return pd.DataFrame({
        ID: [str(i) for i in range(n)],
        "endpoint": endpoint,
        "dataset": "test",
        "landmark_days": landmark,
        "duration_days": np.linspace(100, 800, n),
        "event": [1, 0] * (n // 2),
        "risk_score": np.linspace(-1.0, 1.0, n),
    })


def _write(tmp_path, frame, name="patient_risks.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


class TestLoadPatientRisks:
    def test_keeps_only_the_requested_endpoint_and_landmark(self, tmp_path):
        mixed = pd.concat([
            _risks(4, endpoint="platinum", landmark=180),
            _risks(4, endpoint="nepc", landmark=180),
            _risks(4, endpoint="platinum", landmark=0),
        ], ignore_index=True)
        out = rsf.load_patient_risks(
            _write(tmp_path, mixed), endpoint="platinum", landmark_day=180, id_col=ID
        )
        assert len(out) == 4

    def test_drops_train_side_rows(self, tmp_path):
        frame = _risks(4)
        train = frame.copy()
        train["dataset"] = "train"
        both = pd.concat([frame, train], ignore_index=True)
        out = rsf.load_patient_risks(
            _write(tmp_path, both), endpoint="platinum", landmark_day=180, id_col=ID
        )
        assert len(out) == 4

    def test_duplicate_patients_raise_rather_than_double_count(self, tmp_path):
        frame = pd.concat([_risks(4), _risks(4)], ignore_index=True)
        frame["dataset"] = "test"
        with pytest.raises(ValueError, match="duplicate"):
            rsf.load_patient_risks(
                _write(tmp_path, frame), endpoint="platinum",
                landmark_day=180, id_col=ID,
            )

    def test_missing_schema_columns_raise(self, tmp_path):
        frame = _risks(4).drop(columns=["risk_score"])
        with pytest.raises(ValueError, match="missing required columns"):
            rsf.load_patient_risks(
                _write(tmp_path, frame), endpoint="platinum",
                landmark_day=180, id_col=ID,
            )

    def test_absent_endpoint_raises_rather_than_returning_empty(self, tmp_path):
        with pytest.raises(ValueError, match="for endpoint='avpc'"):
            rsf.load_patient_risks(
                _write(tmp_path, _risks(4)), endpoint="avpc",
                landmark_day=180, id_col=ID,
            )

    def test_requesting_oof_from_a_test_only_file_names_what_is_there(self, tmp_path):
        """A file without --out-of-fold-risks rows must say so, not read empty.

        The two schemes come from different models, so quietly falling back to
        the test block would draw a figure labelled full-cohort from a subset.
        """
        with pytest.raises(ValueError, match=r"no dataset=='cv_oof' rows"):
            rsf.load_patient_risks(
                _write(tmp_path, _risks(4)), endpoint="platinum",
                landmark_day=180, id_col=ID, dataset="cv_oof",
            )

    def test_oof_rows_are_selected_without_the_test_rows(self, tmp_path):
        """Selecting one scheme must exclude the other, never pool them."""
        test_rows = _risks(4)
        oof_rows = _risks(6)
        oof_rows["dataset"] = "cv_oof"
        both = pd.concat([test_rows, oof_rows], ignore_index=True)

        out = rsf.load_patient_risks(
            _write(tmp_path, both), endpoint="platinum",
            landmark_day=180, id_col=ID, dataset="cv_oof",
        )
        assert len(out) == 6


class TestGeneIndicators:
    def test_absent_gene_is_missing_not_wild_type(self):
        frame = pd.DataFrame({"TP53_SNV": [1, 0]})
        indicator, cols = rsf.gene_alteration_indicator(frame, "RB1")
        assert cols == []
        assert indicator.isna().all()

    def test_any_variant_class_counts_as_altered(self):
        frame = pd.DataFrame({"PTEN_SNV": [0, 0], "PTEN_DEL": [1, 0]})
        indicator, cols = rsf.gene_alteration_indicator(frame, "PTEN")
        assert set(cols) == {"PTEN_SNV", "PTEN_DEL"}
        assert indicator.tolist() == [1.0, 0.0]

    def test_all_missing_patient_stays_missing(self):
        frame = pd.DataFrame({"TP53_SNV": [np.nan, 0], "TP53_DEL": [np.nan, 1]})
        indicator, _ = rsf.gene_alteration_indicator(frame, "TP53")
        assert np.isnan(indicator.iloc[0])
        assert indicator.iloc[1] == 1.0

    def test_burden_requires_all_three_genes_observed(self):
        frame = pd.DataFrame({
            "TP53_SNV": [1, 1],
            "RB1_SNV": [0, 1],
            # PTEN present but unobserved for the first patient.
            "PTEN_DEL": [np.nan, 0],
        })
        burden = rsf._trio_burden(frame)
        assert pd.isna(burden.iloc[0])
        assert burden.iloc[1] == "2+ altered"

    def test_tp53_rb1_pair_is_mutually_exclusive(self):
        frame = pd.DataFrame({
            "TP53_SNV": [0, 1, 0, 1],
            "RB1_SNV": [0, 0, 1, 1],
        })
        assert rsf._tp53_rb1_pair(frame).tolist() == [
            "Neither", "TP53 only", "RB1 only", "TP53+RB1",
        ]

    def test_trio_combinations_label_altered_genes(self):
        frame = pd.DataFrame({
            "TP53_SNV": [0, 1, 1],
            "PTEN_DEL": [0, 0, 1],
            "RB1_SNV": [0, 0, 1],
        })
        combos = rsf._trio_combinations(frame)
        assert combos.tolist() == ["None altered", "TP53", "TP53+PTEN+RB1"]


class TestGroupings:
    def test_risk_split_puts_ties_in_the_low_group(self):
        frame = pd.DataFrame({"risk_score": [1.0, 1.0, 1.0, 2.0]})
        groups = rsf.risk_group(frame, cutpoint=1.0)
        assert (groups == rsf.RISK_LOW).sum() == 3
        assert (groups == rsf.RISK_HIGH).sum() == 1

    def test_gleason_bins(self):
        frame = pd.DataFrame({rsf.GLEASON_FEATURE: [5, 6, 7, 8, 10, np.nan]})
        groups = rsf._gleason_group(frame)
        assert groups.tolist()[:5] == [
            "Gleason <=6", "Gleason <=6", "Gleason 7",
            "Gleason 8-10", "Gleason 8-10",
        ]
        assert pd.isna(groups.iloc[5])

    def test_stage_accepts_roman_and_numeric(self):
        frame = pd.DataFrame({rsf.STAGE_COLUMN: ["I", "2", "III", "4.0", "X"]})
        groups = rsf._stage_group(frame)
        assert groups.tolist()[:4] == [
            "Stage I-II", "Stage I-II", "Stage III", "Stage IV",
        ]
        assert pd.isna(groups.iloc[4])

    def test_stratifiers_with_one_group_are_dropped(self):
        frame = _risks(6)
        # Every patient wild-type -> single-valued, nothing to compare.
        frame["TP53_SNV"] = 0
        keys = {s.key for s in rsf.build_stratifiers(frame, cutpoint=0.0)}
        assert "tp53" not in keys
        assert "risk_score" in keys


class TestDiscriminationTable:
    def test_pairs_each_feature_against_the_same_cohort(self):
        frame = _risks(8)
        # Gleason observed for half the cohort only.
        frame[rsf.GLEASON_FEATURE] = [6, 7, 8, 9, np.nan, np.nan, np.nan, np.nan]
        stratifiers = rsf.build_stratifiers(frame, cutpoint=0.0)
        table = rsf.discrimination_table(frame, stratifiers)
        row = table.loc[table["feature"] == "gleason"].iloc[0]
        assert row["n_observed"] == 4
        assert row["n_held_out"] == 8
        # The paired column is computed on the feature's 4 patients, so it is
        # free to differ from the full-cohort number -- that is the point.
        assert "risk_score_same_cohort_c_index" in table.columns
        assert "risk_score_full_cohort_c_index" in table.columns

    def test_unordered_groups_report_no_feature_c_index(self):
        frame = _risks(8)
        frame["TP53_SNV"] = [0, 1, 0, 1, 0, 1, 0, 1]
        frame["PTEN_DEL"] = [0, 0, 1, 1, 0, 0, 1, 1]
        frame["RB1_SNV"] = [0, 0, 0, 0, 1, 1, 1, 1]
        stratifiers = rsf.build_stratifiers(frame, cutpoint=0.0)
        table = rsf.discrimination_table(frame, stratifiers)
        row = table.loc[table["feature"] == "trio_combinations"].iloc[0]
        assert np.isnan(row["feature_c_index"])
        assert "unordered" in row["note"]

    def test_risk_score_is_not_scored_against_itself(self):
        frame = _risks(8)
        stratifiers = rsf.build_stratifiers(frame, cutpoint=0.0)
        table = rsf.discrimination_table(frame, stratifiers)
        assert "risk_score" not in set(table.get("feature", []))


class TestGeneSourceManifest:
    def test_records_which_classes_backed_each_gene(self):
        frame = pd.DataFrame({"TP53_SNV": [1, 0], "PTEN_DEL": [0, 1]})
        manifest = rsf.gene_source_manifest(frame).set_index("gene")
        assert manifest.loc["TP53", "gene_sources"] == "TP53_SNV"
        assert manifest.loc["PTEN", "gene_sources"] == "PTEN_DEL"
        assert manifest.loc["RB1", "gene_sources"] == ""
        assert manifest.loc["RB1", "n_observed"] == 0


class TestEndToEnd:
    def test_writes_table_and_manifest_without_plotting(self, tmp_path):
        risks = _risks(12)
        clinical = pd.DataFrame({
            ID: [str(i) for i in range(12)],
            rsf.GLEASON_FEATURE: [6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9],
            rsf.STAGE_COLUMN: ["I", "II", "III", "IV"] * 3,
            "TP53_SNV": [0, 1] * 6,
            "RB1_SNV": [0, 0, 1, 1] * 3,
            "PTEN_DEL": [1, 0] * 6,
        })
        risk_path = _write(tmp_path, risks)
        clin_path = tmp_path / "clinical.csv"
        clinical.to_csv(clin_path, index=False)
        out_dir = tmp_path / "out"
        rsf.main([
            "--patient-risks", str(risk_path),
            "--clinical-features", str(clin_path),
            "--output-dir", str(out_dir),
            "--endpoint", "platinum",
            "--landmark-days", "180",
            "--id-col", ID,
            "--no-plot",
        ])
        table = pd.read_csv(out_dir / "risk_stratified_discrimination_platinum_landmark180.csv")
        assert {"gleason", "stage", "tp53"} <= set(table["feature"])
        manifest = pd.read_csv(out_dir / "risk_stratified_gene_sources_platinum_landmark180.csv")
        assert set(manifest["gene"]) == set(rsf.TRIO_GENES)

    def test_runs_with_risk_scores_alone(self, tmp_path):
        out_dir = tmp_path / "out"
        rsf.main([
            "--patient-risks", str(_write(tmp_path, _risks(8))),
            "--output-dir", str(out_dir),
            "--endpoint", "platinum",
            "--id-col", ID,
            "--no-plot",
        ])
        assert (out_dir / "risk_stratified_gene_sources_platinum.csv").exists()
