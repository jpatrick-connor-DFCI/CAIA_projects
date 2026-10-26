#!/usr/bin/env Rscript
# Federated Figure 5: transfer panels and supplemental data.
project_root <- Sys.getenv("CAIA_PROJECT_ROOT")
if (!nzchar(project_root)) {
  source_files <- lapply(sys.frames(), function(frame) frame$ofile)
  source_files <- Filter(function(path) is.character(path) && length(path) == 1L && nzchar(path), source_files)
  if (length(source_files)) {
    script_path <- tail(source_files, 1L)[[1L]]
  } else {
    script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else NULL
  }
  project_root <- if (is.null(script_path)) "/data/gusev/USERS/jpconnor/code/CAIA" else
    normalizePath(file.path(dirname(normalizePath(script_path, mustWork = TRUE)), "..", "..", ".."), mustWork = TRUE)
  Sys.setenv(CAIA_PROJECT_ROOT = project_root)
}
source(file.path(project_root, "COMPASS", "survival_analysis", "07_figures", "_common.R"))
figure_run <- run_figure("federated_05_transfer_no_dfci")
