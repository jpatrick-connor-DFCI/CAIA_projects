#!/usr/bin/env Rscript
# Standalone checklist-supplement figures from collect_supplement.py's CSVs.
# The cached COMPASS/figures/all.R workflow renders the same figures; use this to
# preview a pair of runs without the figure cache.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  stop("Usage: Rscript render_supplement.R <collect_supplement-output-dir> [figure-directory]")
}
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])))
source(file.path(script_dir, "..", "figure_pipeline", "checklist_supplement.R"))
root <- normalizePath(args[[1L]], mustWork = TRUE)
out <- if (length(args) >= 2L) args[[2L]] else file.path(root, "figures")
save_pdf <- function(plot, path, width, height, dpi, overwrite) {
  figure_require_font()
  plot <- figure_style_plot(plot)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(path, plot, width = width, height = height, bg = "white", device = figure_pdf_device)
  message("Wrote ", path)
}
render_checklist_supplement(root, out, 200, TRUE, save_pdf)
