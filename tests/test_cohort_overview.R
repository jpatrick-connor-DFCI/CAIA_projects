source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_supplements.R")
cohorts <- c("adt", "adt_noprecastrate", "adt_metastatic_adt",
  "adt_metastatic_adt_noprecastrate", "adt_metastatic_llm", "adt_metastatic_llm_noprecastrate")
counts <- tibble(cohort = rep(cohorts, each = 2), endpoint = rep(c("platinum", "nepc"), 6),
  n_patients = c(2171, 2137, 2018, 1983, 1343, 1323, 1211, 1191, 1232, 1207, 1100, 1074),
  n_events = c(157, 84, 141, 80, 151, 69, 135, 65, 147, 71, 132, 67), status = "ok")
labels <- tibble(adt_label = rep(c("Metastatic", "Local"), each = 3),
  llm_label = rep(c("Metastatic", "Local", "Unlabelled"), 2), n = c(1117, 304, 23, 193, 647, 3))
plots <- list(incidence = plot_cohort_event_overview(counts, cohorts), labels = plot_stage1_label_overview(labels))
stopifnot(all(vapply(plots, inherits, logical(1), "gtable")))
thin <- counts; thin$n_events[1:2] <- c(0, 1); thin$status[3] <- "missing"
withCallingHandlers({
  plot_cohort_event_overview(thin, cohorts)
  plot_stage1_label_overview(bind_rows(labels, tibble(adt_label = "Unlabelled", llm_label = "Local", n = 1)))
}, warning = function(w) stop(w))
stopifnot(is.null(plot_cohort_event_overview(mutate(counts, status = "missing"), cohorts)),
          is.null(plot_stage1_label_overview(labels[0, ])))
output <- Sys.getenv("COMPASS_TEST_OVERVIEW_OUTPUT", "")
if (nzchar(output)) {
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  for (name in names(plots)) ggsave(file.path(output, paste0(name, ".png")), plots[[name]], width = 16, height = 7.5, dpi = 120, bg = "white")
}
cat("Cohort overview plots: complete, thin/zero-event, missing cells and unlabelled patients passed.\n")
