source("COMPASS/figures/COMPASS_generate_figures_pipeline.R")
source("COMPASS/figures/figure_supplements.R")
cohorts <- c("adt", "adt_noprecastrate", "adt_metastatic_adt",
  "adt_metastatic_adt_noprecastrate", "adt_metastatic_llm", "adt_metastatic_llm_noprecastrate")
counts <- tibble(cohort = rep(cohorts, each = 2), endpoint = rep(c("platinum", "nepc"), 6),
  n_patients = c(2171, 2137, 2018, 1983, 1343, 1323, 1211, 1191, 1232, 1207, 1100, 1074),
  n_events = c(157, 84, 141, 80, 151, 69, 135, 65, 147, 71, 132, 67), status = "ok")
plots <- list(incidence = plot_cohort_event_overview(counts, cohorts))
stopifnot(all(vapply(plots, inherits, logical(1), "gtable")))
thin <- counts; thin$n_events[1:2] <- c(0, 1); thin$status[3] <- "missing"
withCallingHandlers(plot_cohort_event_overview(thin, cohorts), warning = function(w) stop(w))
stopifnot(is.null(plot_cohort_event_overview(mutate(counts, status = "missing"), cohorts)))
output <- Sys.getenv("COMPASS_TEST_OVERVIEW_OUTPUT", "")
if (nzchar(output)) {
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  for (name in names(plots)) ggsave(file.path(output, paste0(name, ".pdf")), plots[[name]], width = 16, height = 7.5, bg = "white", device = grDevices::cairo_pdf)
}
cat("Cohort overview plots: complete, thin/zero-event and missing cells passed.\n")
