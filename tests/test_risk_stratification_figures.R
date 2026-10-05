# Clinical-combination C-index, risk-stratification KM, and TP53/PTEN/RB1
# co-mutation figures (Figure 4e-h). Run from the repo root:
#   Rscript tests/test_risk_stratification_figures.R
source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_workflow.R")

root <- tempfile("riskstrat-figures-"); dir.create(root)
on.exit(unlink(root, recursive = TRUE), add = TRUE)

# Synthetic risk_score_stratified_figures.py outputs for one scheme/landmark.
write_strata <- function(directory, endpoint, landmark, n = 80, comparison = FALSE) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  set.seed(landmark + n)
  genes <- expand.grid(TP53 = 0:1, PTEN = 0:1, RB1 = 0:1)[rep(1:8, length.out = n), ]
  trio <- apply(genes, 1, function(g) {
    altered <- c("TP53", "PTEN", "RB1")[g == 1]
    if (length(altered)) paste(altered, collapse = "+") else "None altered"
  })
  risk <- rnorm(n)
  patients <- tibble(
    DFCI_MRN = as.character(seq_len(n)), duration_days = rexp(n, 1 / 400) + 1,
    event = rbinom(n, 1, .5), risk_score = ifelse(risk > median(risk),
      "High risk (above median)", "Low risk (at or below median)"),
    gleason = rep(c("Gleason <=6", "Gleason 7", "Gleason 8-10", NA), length.out = n),
    tp53 = ifelse(genes$TP53 == 1, "Altered", "Wild-type"),
    trio_combinations = trio)
  meta <- tibble(
    stratifier = c("risk_score", "gleason", "tp53", "trio_combinations"),
    title = c("Model risk score (held-out)", "Gleason score", "TP53 alteration",
              "TP53/PTEN/RB1 combinations"),
    order = c("High risk (above median)|Low risk (at or below median)",
              "Gleason <=6|Gleason 7|Gleason 8-10", "Altered|Wild-type",
              "None altered|TP53|PTEN|RB1|TP53+PTEN|TP53+RB1|PTEN+RB1|TP53+PTEN+RB1"),
    ordinal = c(TRUE, TRUE, TRUE, FALSE), n = n)
  if (comparison) {
    patients[["comparison__gleason-labs"]] <- rev(patients$risk_score)
    meta <- bind_rows(meta, tibble(stratifier = "comparison__gleason-labs",
      title = "gleason-labs risk score (held-out)",
      order = "High risk (above median)|Low risk (at or below median)", ordinal = TRUE, n = n))
  }
  suffix <- sprintf("_%s_landmark%s", endpoint, landmark)
  write_csv(patients, file.path(directory, paste0("risk_stratified_patients", suffix, ".csv")), na = "")
  write_csv(meta, file.path(directory, paste0("risk_stratified_strata", suffix, ".csv")))
}

# ---- readers ---------------------------------------------------------------
full_dir <- file.path(root, "risk_stratification", "cv_oof", "full_cohort", "landmark_90")
write_strata(full_dir, "platinum", 90)
stopifnot(is.null(read_risk_strata(full_dir, "platinum", 0)))
strata <- read_risk_strata(full_dir, "platinum", 90)
stopifnot(nrow(strata$patients) == 80, is.numeric(strata$patients$time),
          identical(strata$meta$stratifier, c("risk_score", "gleason", "tp53", "trio_combinations")),
          length(risk_stratum_order(strata$meta, "trio_combinations")) == 8)
# Unclassifiable patients drop out of that stratifier only.
d_gleason <- risk_strata_frame(strata$patients, "gleason")
stopifnot(nrow(d_gleason) == 60, nrow(risk_strata_frame(strata$patients, "risk_score")) == 80)

# ---- KM helper: eight groups, no CI ribbons, fixed colors --------------------
d_trio <- risk_strata_frame(strata$patients, "trio_combinations")
p <- plot_stratified_platinum(d_trio, "Trio", "landmark", risk_stratum_order(strata$meta, "trio_combinations"),
                              colors = RISK_STRATUM_COLORS, ci = FALSE, legend_ncol = 2)
stopifnot(inherits(p, "ggplot"), !any(vapply(p$layers, function(l) inherits(l$geom, "GeomRibbon"), logical(1))))
built <- ggplot_build(p)
stopifnot(length(unique(built$data[[1]]$colour)) == 8)
invisible(ggplotGrob(p))
# Defaults are unchanged for existing callers: ribbons on, default palette.
p_default <- plot_stratified_platinum(d_gleason, "Gleason", "ADT initiation")
stopifnot(any(vapply(p_default$layers, function(l) inherits(l$geom, "GeomRibbon"), logical(1))),
          p_default$labels$y == "Platinum-free probability")

# ---- figure 4e-h block, evaluated with stubbed save/table writers ----------
lines <- readLines("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
start <- grep("^  # Clinical combinations \\(compass_pipeline", lines)
end <- grep("^  # Retired: supplemental all-model comparison\\.", lines)
stopifnot(length(start) == 1, length(end) == 1, start < end)
block <- parse(text = lines[start:(end - 1)])

base <- file.path(root, "local_runs_adt")
dir.create(base)
invisible(file.rename(file.path(root, "risk_stratification"), file.path(base, "risk_stratification")))
write_strata(file.path(base, "risk_stratification", "test", "full_cohort", "landmark_90"), "platinum", 90, n = 40)
write_strata(file.path(base, "risk_stratification", "test", "matched", "gleason", "cox", "landmark_90"),
             "platinum", 90, n = 40, comparison = TRUE)
for (arm in c("labs", "gleason", "gleason-labs")) for (model in c("cox", "xgboost")) {
  directory <- file.path(base, "clinical_combinations", "gleason", arm, model, "landmark_90", "both")
  dir.create(directory, recursive = TRUE)
  write_csv(tibble(endpoint = "platinum", test_c_index = .6 + nchar(arm) / 100,
                   test_mean_auc_t = .7, n_train_val = 300, n_test = 100),
            file.path(directory, if (model == "cox") "cox_agg_multivariable_metrics.csv" else
              "landmark_xgboost_metrics.csv"))
}
saved <- new.env(); tables <- list()
env <- new.env(parent = globalenv())
with(env, {
  BASE <- base; LANDMARKS <- c(0, 90); ENDPOINT <- "platinum"; ENDPOINT_DISPLAY <- "PLATINUM"
  ANCHOR_LABEL <- "ADT initiation"; COHORT_DISPLAY <- "ADT."; OUT_DIR <- root; show <- FALSE
  lab_stem_slug <- function(x) tolower(gsub("[^A-Za-z0-9]+", "_", x))
  save_fig <- function(plot, out_dir, stem, width, height) {
    stopifnot(inherits(plot, "ggplot"))
    invisible(ggplotGrob(plot))
    assign(stem, plot, envir = saved)
  }
  write_table1 <- function(table, out_base) tables[[basename(out_base)]] <<- table
})
for (expr in block) eval(expr, env)
stems <- ls(saved)

# Raw C-index only: one figure per combination cohort that has metrics, no deltas.
stopifnot("figure4e_combinations_gleason_cindex_platinum" %in% stems,
          !"figure4e_combinations_gleason_somatic_cindex_platinum" %in% stems)
combo <- tables[["figure4e_combinations_gleason_cindex_platinum"]]
stopifnot(nrow(combo) == 2 * 2 * 3, !any(grepl("delta|diff", names(combo))),
          sum(is.finite(combo$cindex)) == 6)
combo_plot <- saved[["figure4e_combinations_gleason_cindex_platinum"]]
stopifnot(all(na.omit(combo_plot$data$cindex) == combo$cindex[is.finite(combo$cindex)]))

# Each scheme/source gets its own KMs; test and cv_oof never share a stem.
stopifnot(all(c("figure4f_riskstrat_cv_oof_full_risk_score_landmark90",
                "figure4f_riskstrat_test_full_trio_combinations_landmark90",
                "figure4f_riskstrat_test_gleason_cox_comparison_gleason_labs_landmark90") %in% stems),
          saved[["figure4f_riskstrat_test_gleason_cox_comparison_gleason_labs_landmark90"]]$labels$title ==
            "Gleason + labs risk score",
          !any(grepl("landmark0$", stems)))
# Within-level panels for full cohorts only, split by the labs risk groups.
within <- grep("^figure4g_withinstrat_", stems, value = TRUE)
stopifnot("figure4g_withinstrat_cv_oof_gleason_gleason_8_10_landmark90" %in% within,
          !any(grepl("risk_score|comparison", within)))
# Co-mutation KMs prefer the full out-of-fold cohort.
stopifnot("figure4h_comutation_trio_combinations_landmark90" %in% stems)
comutation <- saved[["figure4h_comutation_trio_combinations_landmark90"]]
stopifnot(grepl("Full landmark cohort", comutation$labels$caption),
          sum(ggplot_build(comutation)$data[[1]]$x == 0) == 8)

# ---- routing and compilation ------------------------------------------------
spec <- figure_compilation_spec("figure4f_riskstrat_test_gleason_somatic_xgboost_tp53_landmark180")
stopifnot(spec$key == "riskstrat_test_gleason_somatic_xgboost_lm180", spec$page_size == 9)
stopifnot(figure_compilation_spec("figure4f_riskstrat_cv_oof_full_risk_score_landmark0")$key ==
            "riskstrat_cv_oof_full_lm0",
          figure_compilation_spec("figure4f_riskstrat_test_gleason_cox_rb1_landmark90")$key ==
            "riskstrat_test_gleason_cox_lm90")
stopifnot(figure_compilation_spec("figure4g_withinstrat_test_tp53_rb1_tp53_only_landmark90")$key ==
            "withinstrat_test_tp53_rb1_lm90",
          figure_compilation_spec("figure4g_withinstrat_cv_oof_tp53_altered_landmark0")$key ==
            "withinstrat_cv_oof_tp53_lm0",
          figure_compilation_spec("figure4g_withinstrat_test_trio_combinations_tp53_rb1_landmark0")$key ==
            "withinstrat_test_trio_combinations_lm0")
stopifnot(is.null(figure_compilation_spec("figure4e_combinations_gleason_cindex_platinum")),
          is.null(figure_compilation_spec("figure4h_comutation_trio_combinations_landmark90")))
leaf <- "platinum__all__incl"
stopifnot(figure_public_path(file.path("/x/ADT/by_figure/figure4/e_combinations_gleason_cindex_platinum", leaf)) ==
            file.path("/x/ADT/prediction", paste0("combinations_gleason_cindex__", leaf)),
          figure_public_path(file.path("/x/ADT/by_figure/figure4/h_comutation_trio_combinations_landmark90", leaf)) ==
            file.path("/x/ADT/prediction", paste0("comutation_trio_combinations_lm90__", leaf)),
          figure_public_path(file.path("/x/ADT/by_figure/figure4/a_discrimination_auc_platinum", leaf)) ==
            file.path("/x/ADT/prediction", paste0("discrimination_auc__", leaf)))

# Compiled pages keep the stratifier on each panel and size to the rows used.
items <- lapply(c("risk_score", "tp53"), function(key) list(
  stem = sprintf("figure4f_riskstrat_cv_oof_full_%s_landmark90", key),
  plot = saved[[sprintf("figure4f_riskstrat_cv_oof_full_%s_landmark90", key)]]))
page_spec <- figure_compilation_spec(items[[1]]$stem)
page_spec$height <- page_spec[["page_row_height"]] * ceiling(length(items) / page_spec$cols)
compiled <- figure_combine(items, page_spec, tempdir())
stopifnot(isTRUE(attr(compiled, "compass_compiled")), page_spec$height == 6.5)
cat("risk stratification figure tests passed\n")
