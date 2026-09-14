from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import polars as pl
import pytest

from COMPASS.survival_analysis.prepare_metastatic_figure_labels import build_labels, collapse_stage

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "COMPASS/survival_analysis"


def fixture_inputs():
    intent = pl.DataFrame({
        "DFCI_MRN": [1, 2, 3, 4, 5, 6],
        "ADT_FIRST_DATE": ["2020-01-01"] * 6,
        "ADT_INTENT": ["METASTATIC", "LOCALIZED_ADJUVANT", "METASTATIC",
                       "LOCALIZED_ADJUVANT", "INDETERMINATE", "LOCALIZED_ADJUVANT"],
    })
    notes = pl.DataFrame({
        "DFCI_MRN": [1, 2, 3, 3, 4, 4, 5, 5, 6],
        "EVENT_DATE": [datetime(2019, 12, 1)] * 3 + [datetime(2020, 2, 1)] +
                      [datetime(2019, 12, 1)] * 4 + [datetime(2017, 1, 1)],
        "DERIVED_STAGE_MERGED": [4, 1, 3, 4, 2, 4, 2, 5, 4],
    })
    llm = pl.DataFrame({"DFCI_MRN": [1, 2, 2, 3, 4, 5],
                        "has_metastatic_disease": [True, False, True, False, None, False]})
    return intent, notes, llm


def test_local_is_stages_one_to_three_only():
    values = ["I", "II", "III", "IV", "1", "2", "3", "4", None, "0", "V", "unknown"]
    assert pl.DataFrame({"stage": values}).select(collapse_stage("stage")).to_series().to_list() == [
        "Local", "Local", "Local", "Metastatic", "Local", "Local", "Local", "Metastatic",
        None, None, None, None]


def test_stage_timing_ties_unknowns_and_pairwise_labels():
    result = build_labels(*fixture_inputs()).sort("DFCI_MRN")
    assert result["REGEX_LABEL"].to_list() == ["Metastatic", "Local", "Local", "Metastatic", "Local", None]
    assert result["REGEX_MAX_AFTER"].to_list() == [None, None, "Metastatic", None, None, None]
    assert result["REGEX_MAX_BEFORE"].to_list()[-1] == "Metastatic"
    assert result["ADT_LABEL"].to_list()[4] is None
    assert result["LLM_LABEL"].to_list() == ["Metastatic", "Metastatic", "Local", None, "Local", None]
    with pytest.raises(ValueError, match="one row per patient"):
        intent, notes, llm = fixture_inputs()
        build_labels(pl.concat([intent, intent.head(1)]), notes, llm)


def test_empty_stage_and_llm_sources_preserve_unclassified_patients():
    intent, notes, llm = fixture_inputs()
    result = build_labels(intent, notes.head(0), llm.head(0))
    assert result.height == intent.height
    assert result["REGEX_LABEL"].null_count() == intent.height
    assert result["LLM_LABEL"].null_count() == intent.height


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R required")
def test_bridge_r_comparisons_and_standalone_diagnostics(tmp_path):
    intent, notes, llm = fixture_inputs()
    # Enough labelled patients for both lab trajectories, plus a patient with
    # no intent label to exercise explicit missing-label coverage.
    intent = pl.concat([intent.with_columns((pl.col("DFCI_MRN") + 10*i)) for i in range(6)])
    notes = pl.concat([notes.with_columns((pl.col("DFCI_MRN") + 10*i)) for i in range(6)])
    llm = pl.concat([llm.with_columns((pl.col("DFCI_MRN") + 10*i)) for i in range(6)])
    intent.write_csv(tmp_path / "intent.csv")
    notes.write_parquet(tmp_path / "stage.parquet")
    llm.write_parquet(tmp_path / "llm.parquet")
    pl.DataFrame({"DFCI_MRN": intent["DFCI_MRN"],
                  "DIAGNOSIS_ICD10_CD": ["C79.51"] * intent.height,
                  "START_DT": ["2019-12-01"] * intent.height}).write_csv(tmp_path / "icd.csv")
    script = f'''
      suppressPackageStartupMessages({{library(dplyr); library(tidyr); library(ggplot2)}})
      source({json.dumps(str(SCRIPTS / "metastatic_figure_supplements.R"))})
      for (expr in parse({json.dumps(str(SCRIPTS / "COMPASS_generate_figures_pipeline.R"))})) {{
        if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
            identical(expr[[2]], as.name("figure_output_tier"))) eval(expr)
      }}
      stopifnot(figure_output_tier("adt", "platinum", "figure3_test") == "main",
        figure_output_tier("adt", "nepc", "figure3_test") == "supplements",
        figure_output_tier("adt_metastatic_llm", "platinum", "figure3_test") == "supplements",
        figure_output_tier("adt_noprecastrate", "platinum", "figure3_test") == "supplements",
        figure_output_tier("adt", "platinum", "adt_labels_coverage") == "supplements",
        figure_output_tier("adt", "platinum", "longitudinal_has_nepc_psa") == "supplements",
        figure_output_tier("adt", "platinum", "figure2v3_enrichment") == "main")
      root <- {json.dumps(str(tmp_path))}
      config <- list(python = {json.dumps(sys.executable)},
        script = {json.dumps(str(SCRIPTS / "prepare_metastatic_figure_labels.py"))},
        intent = file.path(root, "intent.csv"), stage = file.path(root, "stage.parquet"),
        llm = file.path(root, "llm.parquet"), icd = file.path(root, "icd.csv"))
      patients <- readr::read_csv(config$intent, show_col_types = FALSE) %>%
        transmute(DFCI_MRN = as.character(DFCI_MRN), t_death = 400 + row_number(),
          t_platinum = 100 + row_number(), t_nepc = 200 + row_number(),
          t_last_contact = 1000, DEATH = row_number() %% 2,
          PLATINUM = row_number() %% 2, NEPC = row_number() %% 2)
      labs <- expand_grid(DFCI_MRN = patients$DFCI_MRN, LAB_GROUP = c("PSA", "Testosterone"),
                          t_rel = c(-90, 90, 270)) %>% mutate(LAB_VALUE = rep(1:9, length.out = n()))
      saved <- character()
      labels <- render_metastatic_supplements(config, patients, labs,
        save_panel = function(plot, stem, width, height) {{
          saved <<- c(saved, stem)
          ggsave(file.path(root, paste0(stem, ".png")), plot, width = width, height = height,
                 dpi = 100, device = ragg::agg_png)
        }}, notify = function(...) NULL)
      stopifnot(setequal(saved, metastatic_supplement_stems()), !anyDuplicated(saved))
      result <- metastatic_pair_counts(labels, "ADT_LABEL", "REGEX_LABEL")
      stopifnot(result$paired == 24, result$missing == 12, result$agreement == .5,
                sum(result$counts$n) == 24)
      empty <- metastatic_pair_counts(labels[FALSE, ], "ADT_LABEL", "REGEX_LABEL")
      stopifnot(empty$paired == 0, is.na(empty$agreement), all(is.na(empty$counts$fraction)))
    '''
    path = tmp_path / "check.R"
    path.write_text(script)
    result = subprocess.run(["Rscript", str(path)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(tmp_path.glob("*.png"))) == 22
