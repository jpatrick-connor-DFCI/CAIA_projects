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
  tracker$update(2, "panel_skipped", "existing_panel")
  tracker$update(2, "panel_skipped", "existing_panel")
  stopifnot(tracker$snapshot()$skipped == 1L)
  tracker$update(2, "panel_start", "figure3_univariate")
  tracker$update(2, "failed", "graphics device failed")
  s <- tracker$snapshot()
  stopifnot(s$finished == 2L, s$successful == 1L, s$failed == 1L,
            s$panels == 1L, s$elapsed >= 0,
            grepl("graphics device failed", tail(lines, 1), fixed = TRUE))
})

# Exercise the actual notebook scheduler, including error handling and both
# Exercise the scheduler shared by preparation and rendering, sequentially and
# with forks. Failures are values so remaining jobs always complete.
source("COMPASS/survival_analysis/figure_data_cache.R")
for (workers in if (.Platform$OS.type == "windows") 1L else c(1L, 2L)) {
  results <- figure_parallel(as.list(1:4), function(i) {
    if (i == 2) stop("synthetic device failure")
    list(panels = 3L)
  }, workers)
  stopifnot(length(results) == 4L,
            identical(results[[2]]$error, "synthetic device failure"),
            sum(vapply(results, function(x) is.null(x$error), logical(1))) == 3L,
            sum(vapply(results, function(x) if (is.null(x$panels)) 0L else x$panels, integer(1))) == 9L)
}
cat("Figure progress checks passed: sequential, parallel, failures, and panel counts.\n")
