#!/usr/bin/env Rscript
# Figure 1: cohort overview and its cohort data.
project_root <- Sys.getenv("CAIA_PROJECT_ROOT")
if (!nzchar(project_root)) {
  script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (!length(script_arg)) stop("Set CAIA_PROJECT_ROOT when sourcing a figure entry point")
  script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), mustWork = TRUE)
  project_root <- normalizePath(file.path(dirname(script_path), "..", "..", ".."), mustWork = TRUE)
  Sys.setenv(CAIA_PROJECT_ROOT = project_root)
}
source(file.path(project_root, "COMPASS", "survival_analysis", "07_figures", "_common.R"))
figure_run <- run_figure("01_cohort_overview")
