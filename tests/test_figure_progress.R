# Run from the repository root: Rscript tests/test_figure_progress.R
# No plotting packages or cluster inputs needed.
source_exprs <- parse("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
for (expr in source_exprs) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
      identical(expr[[2]], as.name("new_figure_progress"))) eval(expr)
}
local({
  lines <- character()
  tracker <- new_figure_progress(c("ADT / NEPC", "ADT / PLATINUM"),
                                report = function(line) lines <<- c(lines, line))
  on.exit(tracker$close())
  stopifnot(tracker$snapshot()$finished == 0L,
            grepl("0/2 sets finished", lines[[1]], fixed = TRUE))
  tracker$update(1, "start")
  tracker$update(1, "panel_start", "figure1a_consort")
  stopifnot(tracker$snapshot()$panels == 0L)
  tracker$update(1, "panel_done", "figure1a_consort")
  tracker$update(1, "panel_done", "figure1a_consort")
  stopifnot(tracker$snapshot()$panels == 1L) # duplicate events cannot overcount
  tracker$update(1, "complete")
  tracker$update(2, "start")
  tracker$update(2, "panel_start", "figure3_univariate")
  tracker$update(2, "failed", "graphics device failed")
  s <- tracker$snapshot()
  stopifnot(s$finished == 2L, s$successful == 1L, s$failed == 1L,
            s$panels == 1L, s$elapsed >= 0,
            grepl("graphics device failed", tail(lines, 1), fixed = TRUE))
})

# Exercise the actual notebook scheduler, including error handling and both
# sequential and forked rendering. Substitute only the expensive renderer.
notebook <- readLines("COMPASS/survival_analysis/05_figures.Rmd", warn = FALSE)
start <- match("```{r render-figures}", notebook)
end <- which(seq_along(notebook) > start & notebook == "```")[[1]]
render_code <- parse(text = notebook[seq.int(start + 1L, end - 1L)])
for (workers in if (.Platform$OS.type == "windows") 1L else c(1L, 2L)) {
  env <- new.env(parent = globalenv())
  env$new_figure_progress <- function(labels) {
    new_figure_progress(labels, report = function(line) invisible(NULL))
  }
  env$COHORTS <- c("adt", "adt_test")
  env$ENDPOINTS <- c("nepc", "platinum")
  env$RENDER_WORKERS <- workers
  env$cohort_display <- identity
  env$NEPC_PROJ_PATH <- env$FIG_ROOT <- tempdir()
  env$PLOT_NON_ANDROGEN_DISTRIBUTIONS <- env$PLOT_NON_ANDROGEN_LAB_FIGURES <- FALSE
  env$PLOT_GAM_TRAJECTORIES <- env$PLOT_ADT_INTENT_SUPPLEMENT <- FALSE
  env$RENDER_DPI <- 200
  env$RENDER_PDF <- FALSE
  env$RENDER_OUTPUT_MODE <- "panels"
  env$generate_figures <- function(cohort, ..., endpoint, progress) {
    progress("stage", "synthetic inputs")
    for (i in 1:3) {
      progress("panel_start", paste0("panel", i))
      if (cohort == "adt_test" && endpoint == "platinum" && i == 2L)
        stop("synthetic device failure")
      progress("panel_done", paste0("panel", i))
    }
  }
  failure <- tryCatch({
    suppressMessages(eval(render_code, env))
    NULL
  }, error = identity)
  stopifnot(inherits(failure, "error"),
            grepl("1 of 4 figure sets failed", conditionMessage(failure), fixed = TRUE),
            env$progress_summary$finished == 4L,
            env$progress_summary$successful == 3L,
            env$progress_summary$failed == 1L,
            env$progress_summary$panels == 10L,
            length(env$render_failures) == 1L)
}
cat("Figure progress checks passed: sequential, parallel, failures, and panel counts.\n")
