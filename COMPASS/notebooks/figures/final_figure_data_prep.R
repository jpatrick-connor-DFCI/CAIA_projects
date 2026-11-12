#!/usr/bin/env Rscript
# Figure data prep: builds the cached tables and scene objects the R figures consume (no rendering).
#
#   Rscript COMPASS/notebooks/figures/final_figure_data_prep.R
#   COMPASS_FIGURE_GROUPS=checklist_augmentation Rscript COMPASS/notebooks/figures/final_figure_data_prep.R
# The augmentation group consumes notebook 07's aggregate receipt independently of notebook 05.
#
# Toggles, paths and environment overrides live in COMPASS/figures/figure_run.R;
# the plotting and workflow code it sources stays in COMPASS/figures/.
FIGURE_STAGE <- "prepare"
script_dir <- local({
  frames <- Filter(function(path) is.character(path) && length(path) == 1L && nzchar(path),
    lapply(sys.frames(), function(frame) frame$ofile))
  path <- if (length(frames)) tail(frames, 1L)[[1L]] else
    sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L])
  if (is.na(path)) file.path(Sys.getenv("CAIA_PROJECT_ROOT", "/data/gusev/USERS/jpconnor/code/CAIA"),
    "COMPASS", "figures") else normalizePath(file.path(dirname(normalizePath(path, mustWork = TRUE)),
    "..", "..", "figures"), mustWork = TRUE)
})
source(file.path(script_dir, "figure_run.R"))
