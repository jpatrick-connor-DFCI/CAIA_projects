#!/usr/bin/env Rscript
# Figure render: draws the PDF and SVG figures from the prepared cache and writes the figure data tables.
#
#   Rscript COMPASS/notebooks/run_figure_render.R
#   COMPASS_FIGURE_GROUPS=checklist_augmentation Rscript COMPASS/notebooks/run_figure_render.R
# Notebook 07's alternative Figure 6 goes to main/alternative_06 in COMPASS_MANUSCRIPT_FIG_ROOT.
#
# Toggles, paths and environment overrides live in COMPASS/figures/figure_run.R;
# the plotting and workflow code it sources stays in COMPASS/figures/.
FIGURE_STAGE <- "render"
script_dir <- local({
  frames <- Filter(function(path) is.character(path) && length(path) == 1L && nzchar(path),
    lapply(sys.frames(), function(frame) frame$ofile))
  path <- if (length(frames)) tail(frames, 1L)[[1L]] else
    sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L])
  if (is.na(path)) file.path(Sys.getenv("CAIA_PROJECT_ROOT", "/data/gusev/USERS/jpconnor/code/CAIA"),
    "COMPASS", "figures") else normalizePath(file.path(dirname(normalizePath(path, mustWork = TRUE)),
    "..", "figures"), mustWork = TRUE)
})
source(file.path(script_dir, "figure_run.R"))
