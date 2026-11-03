# Run from the repository root: Rscript tests/test_figure_preprocessing.R
# Exercise the actual data helpers without requiring plotting-only packages
# or the cluster data. No files containing real patient data are needed.
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})
options(readr.num_threads = 1L)
pipeline <- new.env(parent = globalenv())
for (expr in parse("COMPASS/figures/COMPASS_generate_figures_pipeline.R")) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-"))) eval(expr, pipeline)
}

local({
  csv <- tempfile(fileext = ".csv")
  on.exit(unlink(csv))
  # Repeated metadata, a non-androgen lab, a PSA alias, missing values,
  # no-lab patients, negative measurements, and both sides of day zero.
  input <- tibble(
    DFCI_MRN = c("001", "001", "001", "002", "002", "003", "004", "004"),
    LAB_NAME = c("PSA", "Prostate specific Ag serum", "Sodium", "Testosterone",
                 "PSA", NA, "PSA", "PSA"),
    LAB_VALUE = c(1, 9, 140, 2, -1, NA, 0, NA),
    LAB_DATE = c("2020-01-01", "2020-01-11", "2020-01-21", "2020-02-01",
                 "2020-02-11", NA, NA, NA),
    t_lab = c(-1, 0, 10, -180, 180, NA, 0, NA),
    PLATINUM = c(1, 1, 1, 0, 0, 0, NA, NA),
    DIAGNOSIS_DATE = "2019-01-01",
    TREATMENT_ANCHOR_DATE = "2020-01-01",
    UNUSED_FEATURE = seq_len(8)
  )
  write_csv(input, csv)
  pipeline$clear_read_cache()
  split <- suppressWarnings(pipeline$cached_profile_patient_and_labs(csv))
  patients <- split$patient_df
  stopifnot(
    identical(patients$DFCI_MRN, c("001", "002", "003", "004")),
    identical(patients$t_dx_to_anchor, rep(365, 4)),
    identical(patients$lab_rows, c(3L, 2L, NA_integer_, 2L)),
    identical(patients$record_span_days, c(20, 10, NA_real_, -Inf)),
    inherits(split$labs_df$LAB_DATE, "Date"),
    !"UNUSED_FEATURE" %in% names(pipeline$cached_figure_longitudinal(csv))
  )

  # Filtering cached spans to a cohort must equal filtering measurements first.
  for (ids in list(c("001", "002"), "002", c("001", "004"), character())) {
    legacy <- split$labs_df %>% filter(DFCI_MRN %in% ids) %>%
      group_by(DFCI_MRN) %>%
      summarise(record_span_days = suppressWarnings(as.numeric(
        max(LAB_DATE, na.rm = TRUE) - min(LAB_DATE, na.rm = TRUE)
      )), .groups = "drop")
    cached <- patients %>% filter(DFCI_MRN %in% ids, !is.na(lab_rows)) %>%
      select(DFCI_MRN, record_span_days) %>% arrange(DFCI_MRN)
    stopifnot(isTRUE(all.equal(cached, legacy)))
  }

  # Forked reads must inherit the processed cache, with no new CSV parses.
  pipeline$clear_raw_read_cache()
  stopifnot(
    identical(ls(pipeline$.read_cache, all.names = TRUE), ".stats"),
    identical(pipeline$cached_profile_patient_and_labs(csv), split),
    pipeline$read_cache_stats()$misses == 1L
  )
  if (.Platform$OS.type != "windows") {
    results <- parallel::mclapply(1:2, function(i) {
      value <- pipeline$cached_profile_patient_and_labs(csv)
      stopifnot(identical(value, split), pipeline$read_cache_stats()$misses == 1L)
      TRUE
    }, mc.cores = 2L, mc.preschedule = FALSE)
    stopifnot(identical(results, list(TRUE, TRUE)))
  }
  for (dates in list(as.Date(c("2020-01-01", NA)),
                     c("2020-01-01", NA, "2020-01-02", "2020-01-01"),
                     c(NA_character_, NA_character_), character())) {
    stopifnot(identical(pipeline$figure_dates(dates), as.Date(dates)))
  }
  pipeline$clear_read_cache()
  stopifnot(length(ls(pipeline$.processed_read_cache)) == 0L)
})
cat("Figure preprocessing regression checks passed.\n")
