# Run from the repository root: Rscript tests/test_figure_resume.R
# Exercise the real writer with tiny PNG/PDF files, without cluster inputs.
suppressPackageStartupMessages(library(ggplot2))
expressions <- parse("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
find_assignment <- function(expr, name) {
  if (missing(expr) || !is.call(expr)) return(NULL)
  if (identical(expr[[1]], as.name("<-")) && identical(expr[[2]], as.name(name)))
    return(expr)
  for (child in as.list(expr)[-1]) {
    found <- find_assignment(child, name)
    if (!is.null(found)) return(found)
  }
  NULL
}
local({
  root <- tempfile("compass-resume-test-")
  dir.create(root)
  on.exit(unlink(root, recursive = TRUE))
  env <- new.env(parent = globalenv())
  for (expr in expressions) {
    for (name in c("figure_file_complete", "reuse_previous_figure_layout", "prepare_figure_text", "save_fig")) {
      assignment <- find_assignment(expr, name)
      if (!is.null(assignment)) eval(assignment, env)
    }
  }
  env$COHORT_LEAF <- "nepc__all__incl"
  env$output_dir_for_stem <- function(stem) file.path(root, stem)
  env$save_dpi <- 50
  env$save_pdf <- FALSE
  env$overwrite <- FALSE
  env$HAS_RAGG <- requireNamespace("ragg", quietly = TRUE)
  env$events <- character()
  env$progress <- function(event, detail) env$events <- c(env$events, event)
  env$notify_progress <- env$progress
  env$saves <- character()
  real_save <- function(filename, ...) {
    env$saves <- c(env$saves, tools::file_ext(filename))
    ggplot2::ggsave(filename, ...)
  }
  env$ggsave <- real_save
  plot <- ggplot(data.frame(x = 1:3, y = 1:3), aes(x, y)) + geom_point()
  save <- function(...) suppressMessages(env$save_fig(..., out_dir = root,
                                                    stem = "test_panel", width = 2, height = 2))
  png <- file.path(root, "test_panel", "nepc__all__incl.png")
  pdf <- sub("png$", "pdf", png)
  bytes <- function(path) readBin(path, "raw", n = file.info(path)$size)

  save(plot)
  stopifnot(identical(env$saves, "png"), env$figure_file_complete(png),
            identical(tail(env$events, 1), "panel_done"))
  png_bytes <- bytes(png)
  save(stop("Completed figures must not force the plot expression"))
  stopifnot(identical(env$saves, "png"),
            identical(tail(env$events, 1), "panel_skipped"),
            identical(bytes(png), png_bytes))

  env$save_pdf <- TRUE
  save(plot)
  stopifnot(identical(env$saves, c("png", "pdf")), env$figure_file_complete(pdf),
            identical(bytes(png), png_bytes))
  pdf_bytes <- bytes(pdf)
  save(stop("Both requested formats already exist"))
  stopifnot(length(env$saves) == 2L, tail(env$events, 1) == "panel_skipped")

  # A different endpoint cannot reuse this endpoint's files.
  save(plot, prefix = "platinum__all__incl")
  stopifnot(length(env$saves) == 4L)

  # Main/supplement reorganization reuses only the same leaf and artifact.
  old <- file.path(root, "by_figure", "main", "figure3", "test", "platinum__all__incl.png")
  dir.create(dirname(old), recursive = TRUE)
  file.copy(png, old)
  new <- file.path(root, "by_figure", "figure3", "test", "platinum__all__incl.png")
  stopifnot(env$reuse_previous_figure_layout(new), identical(bytes(new), bytes(old)),
    !env$reuse_previous_figure_layout(sub("platinum__", "nepc__", new)),
    !env$reuse_previous_figure_layout(new))

  # Empty/truncated images regenerate; an intact PDF is retained.
  writeBin(raw(), png)
  stopifnot(!env$figure_file_complete(png))
  save(plot)
  stopifnot(tail(env$saves, 1) == "png", identical(bytes(pdf), pdf_bytes))
  writeBin(png_bytes[seq_len(length(png_bytes) %/% 2)], png)
  stopifnot(!env$figure_file_complete(png))
  save(plot)
  stopifnot(env$figure_file_complete(png))
  writeBin(pdf_bytes[seq_len(length(pdf_bytes) %/% 2)], pdf)
  stopifnot(!env$figure_file_complete(pdf))
  save(plot)
  stopifnot(tail(env$saves, 1) == "pdf", env$figure_file_complete(pdf))

  env$overwrite <- TRUE
  before <- length(env$saves)
  save(plot)
  stopifnot(length(env$saves) == before + 2L,
            identical(tail(env$saves, 2), c("png", "pdf")))

  # A failed overwrite must preserve the prior complete output and must not
  # publish the temporary partial file or emit a panel_done event.
  png_bytes <- bytes(png)
  env$ggsave <- function(filename, ...) {
    writeBin(charToRaw("partial image"), filename)
    stop("synthetic graphics failure")
  }
  failure <- tryCatch(save(plot), error = identity)
  stopifnot(inherits(failure, "error"), identical(bytes(png), png_bytes),
            tail(env$events, 1) == "panel_start",
            !any(startsWith(list.files(dirname(png), all.files = TRUE), ".compass-render-")))
})
# Exercise the actual routing: PSA and testosterone share the new supplement
# group, so their lab tokens must remain in the artifact name.
local({
  env <- new.env(parent = globalenv())
  for (expr in expressions) {
    for (name in c("figure_group", "artifact_name_for_stem",
                   "output_dir_for_stem", "lab_stem_slug", "match_lab_in_stem")) {
      assignment <- find_assignment(expr, name)
      if (!is.null(assignment)) eval(assignment, env)
    }
  }
  sys.source("COMPASS/survival_analysis/figure_supplements.R", env)
  env$FIG_ROOT <- tempdir()
  env$COHORT <- "adt"
  env$ENDPOINT <- "platinum"
  env$LAB_SLUG_TO_NAME <- c(psa = "PSA", testosterone = "Testosterone")
  # Default output is the three confusion matrices only.
  stems <- env$metastatic_supplement_stems()
  paths <- vapply(stems, env$output_dir_for_stem, character(1))
  stopifnot(!anyDuplicated(paths), all(grepl("/metastatic_labels/", paths)),
    setequal(stems, paste0("adt_labels_", c("llm_vs_regex_max_any",
      "adt_vs_regex_max_any", "adt_vs_llm"))))
  # The retained panels still route correctly when re-enabled; PSA and
  # testosterone share the group, so their lab tokens must survive in the name.
  old <- Sys.getenv("COMPASS_METASTATIC_EXTRA_PANELS", unset = NA_character_)
  Sys.setenv(COMPASS_METASTATIC_EXTRA_PANELS = "1")
  on.exit(if (is.na(old)) Sys.unsetenv("COMPASS_METASTATIC_EXTRA_PANELS") else
    Sys.setenv(COMPASS_METASTATIC_EXTRA_PANELS = old), add = TRUE)
  extra <- env$metastatic_supplement_stems()
  extra_paths <- vapply(extra, env$output_dir_for_stem, character(1))
  stopifnot(!anyDuplicated(extra_paths), length(extra) == 18,
    endsWith(extra_paths[["adt_labels_adt_trajectory_psa"]], "/adt_trajectory_psa"),
    endsWith(extra_paths[["adt_labels_adt_trajectory_testosterone"]], "/adt_trajectory_testosterone"))
})
cat("Figure resume checks passed: skip, formats, endpoint isolation, truncation, overwrite, failed saves.\n")
