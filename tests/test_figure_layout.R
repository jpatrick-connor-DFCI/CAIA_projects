source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_workflow.R")
local({
  root <- tempfile("figure-layout-"); dir.create(root)
  on.exit(unlink(root, recursive = TRUE))
  old <- file.path(root, "ADT", "by_figure", "main", "figure3", "panel", "platinum.csv")
  other <- file.path(root, "ADT", "by_figure", "supplements", "figure3", "panel", "nepc.csv")
  dest <- file.path(root, "ADT", "by_figure", "figure3", "panel", "platinum.csv")
  for (p in c(old, other, dest)) dir.create(dirname(p), recursive = TRUE, showWarnings = FALSE)
  writeLines("original", old); writeLines("other", other); writeLines("conflict", dest)
  failure <- tryCatch(figure_flatten_export_layout(root), error = identity)
  stopifnot(inherits(failure, "error"), file.exists(old), file.exists(other), readLines(dest) == "conflict")
  writeLines("original", dest)
  stopifnot(figure_flatten_export_layout(root) == 2L, readLines(dest) == "original",
    readLines(file.path(dirname(dest), "nepc.csv")) == "other",
    !any(basename(list.dirs(root)) %in% c("main", "supplements")),
    figure_flatten_export_layout(root) == 0L)
})
cat("Flat figure layout: migration, duplicate consolidation, collision safety, and idempotence passed.\n")
