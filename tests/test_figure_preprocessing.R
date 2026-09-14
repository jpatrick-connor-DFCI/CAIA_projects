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
for (expr in parse("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-"))) eval(expr, pipeline)
}

local({
  csv <- tempfile(fileext = ".csv")
  on.exit(unlink(csv))
  # Repeated metadata, an unplotted canonical lab, a PSA alias, missing values,
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

  androgen <- pipeline$cached_canonical_longitudinal(csv, labs = pipeline$ANDROGEN)
  all_labs <- pipeline$cached_canonical_longitudinal(csv)
  stopifnot(
    nrow(androgen) == 5L, nrow(all_labs) == 6L,
    identical(androgen$LAB_GROUP, c("PSA", "PSA", "Testosterone", "PSA", "PSA")),
    !"DIAGNOSIS_DATE" %in% names(androgen),
    identical(androgen, all_labs %>% filter(LAB_GROUP %in% pipeline$ANDROGEN)),
    identical(androgen, pipeline$cached_canonical_longitudinal(csv, rev(pipeline$ANDROGEN))),
    pipeline$read_cache_stats()$misses == 1L
  )

  # Execute the actual nested trajectory helper against the compact cache.
  trajectory <- new.env(parent = pipeline)
  collect <- function(expr) {
    if (missing(expr) || !is.call(expr)) return(invisible(NULL))
    if (identical(expr[[1]], as.name("<-")) && is.symbol(expr[[2]]) &&
        as.character(expr[[2]]) %in% c("anchored_bin_edges", "patient_bin_trajectory")) {
      eval(expr, trajectory)
    } else {
      for (child in as.list(expr)[-1]) collect(child)
    }
  }
  collect(body(pipeline$generate_figures))
  trajectory$PRE_DAYS <- 365.25
  trajectory$POST_DAYS <- 5 * 365.25
  trajectory$BIN_WIDTH_DAYS <- 180
  bins <- trajectory$patient_bin_trajectory(androgen, "PSA")
  log_bins <- trajectory$patient_bin_trajectory(androgen, "PSA", log_scale = TRUE)
  stopifnot(
    nrow(bins) == 4L, nrow(log_bins) == 3L,
    sum(bins$DFCI_MRN == "001") == 2L,
    any(bins$t_mid[bins$DFCI_MRN == "001"] < 0),
    any(bins$t_mid[bins$DFCI_MRN == "001"] > 0),
    isTRUE(all.equal(sort(log_bins$LAB_VALUE), sort(log1p(c(1, 9, 0)))))
  )
  lookup <- tibble(DFCI_MRN = c("001", "004"), stratum = c("NEPC+", "NEPC-"))
  labeled <- trajectory$patient_bin_trajectory(androgen, "PSA", stratum_values = lookup)
  stopifnot(nrow(labeled) == 3L, setequal(labeled$stratum, lookup$stratum))

  # Forked reads must inherit the processed cache, with no new CSV parses.
  pipeline$clear_raw_read_cache()
  stopifnot(
    identical(ls(pipeline$.read_cache, all.names = TRUE), ".stats"),
    identical(pipeline$cached_profile_patient_and_labs(csv), split),
    identical(pipeline$cached_canonical_longitudinal(csv, pipeline$ANDROGEN), androgen),
    pipeline$read_cache_stats()$misses == 1L
  )
  if (.Platform$OS.type != "windows") {
    results <- parallel::mclapply(1:2, function(i) {
      value <- pipeline$cached_canonical_longitudinal(csv, pipeline$ANDROGEN)
      stopifnot(identical(value, androgen), pipeline$read_cache_stats()$misses == 1L)
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
