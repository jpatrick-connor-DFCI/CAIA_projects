# Rscript tests/test_figure_cache.R: differential preparation and render-cache tests.
source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_data_cache.R")
options(readr.num_threads = 1L)
local({
  root <- tempfile("figure-cache-test-")
  dir.create(root)
  on.exit(unlink(root, recursive = TRUE))
  input <- tibble(
    DFCI_MRN = c("001", "001", "001", "001", "002", "003", "004", "005"),
    LAB_NAME = c("PSA", "Prostate specific Ag serum", "PSA", "Sodium", "Testosterone", NA, "PSA", "PSA"),
    LAB_VALUE = c(1, 9, 3, 140, 2, NA, -1, 0),
    LAB_DATE = c("2020-01-01", "2020-01-11", "2020-01-11", "2020-01-21", "2020-02-01", NA, NA, NA),
    t_lab = c(-1, 0, 0, 10, -180, NA, 180, -1826.25),
    PLATINUM = c(1, 1, 1, 1, 0, 0, NA, 0),
    DIAGNOSIS_DATE = "2019-01-01", TREATMENT_ANCHOR_DATE = "2020-01-01")
  source_path <- file.path(root, "longitudinal_prediction_data_adt.csv")
  write_csv(input, source_path)
  dir.create(file.path(root, "survival_analysis", "prediction_inputs_adt"), recursive = TRUE)
  config <- list(data_root = root, cache_root = file.path(root, "cache"),
    cohorts = "adt", endpoints = "platinum", labs = c("PSA", "Testosterone"),
    classifier_path = file.path(root, "classifier"), federated_path = file.path(root, "fed.csv"))
  config$metastatic <- TRUE; config$metastatic_extra <- TRUE
  config$metastatic_sources <- list(intent = file.path(root, "intent.csv"), stage = file.path(root, "stage.parquet"),
    llm = file.path(root, "llm.parquet"), icd = file.path(root, "icd.csv"))
  write_csv(tibble(DFCI_MRN = 1:4, ADT_FIRST_DATE = "2019-11-01",
    ADT_INTENT = c("METASTATIC", "LOCALIZED_ADJUVANT", "INDETERMINATE", "METASTATIC")), config$metastatic_sources$intent)
  nanoparquet::write_parquet(data.frame(DFCI_MRN = 1:5, EVENT_DATE = c("2019-12-01", "2020-02-01", "2019-01-01", "2019-12-01", "2019-12-01"),
    DERIVED_STAGE_MERGED = c("IV", "III", "II", "I", "IV")), config$metastatic_sources$stage)
  nanoparquet::write_parquet(data.frame(DFCI_MRN = c(1, 2, 2, 5), has_metastatic_disease = c(TRUE, FALSE, TRUE, NA)), config$metastatic_sources$llm)
  write_csv(tibble(DFCI_MRN = c(1, 1, 2, 4), START_DT = c("2019-12-01", "2020-01-01", "2020-02-01", "2019-12-01"),
    DIAGNOSIS_ICD10_CD = c("C79.51", "C77.0", "C78.7", "C79.9")), config$metastatic_sources$icd)
  config_path <- file.path(root, "config.json")
  jsonlite::write_json(config, config_path, auto_unbox = TRUE)
  status <- system2(Sys.getenv("COMPASS_FIGURE_PYTHON", "python3"),
    c(shQuote("COMPASS/survival_analysis/prepare_figure_data.py"), "--config", shQuote(config_path)))
  stopifnot(status == 0L)
  manifest <- jsonlite::read_json(file.path(root, "cache", "manifest.json"))

  # Extract actual nested implementation, not a duplicate of the algorithm.
  nested <- new.env(parent = globalenv())
  collect <- function(expr) {
    if (missing(expr) || !is.call(expr)) return(invisible(NULL))
    if (identical(expr[[1]], as.name("<-")) && is.symbol(expr[[2]]) &&
        as.character(expr[[2]]) %in% c("anchored_bin_edges", "patient_bin_trajectory", "bin_group_ci")) eval(expr, nested)
    else for (child in as.list(expr)[-1]) collect(child)
  }
  collect(body(generate_figures))
  nested$PRE_DAYS <- 365.25; nested$POST_DAYS <- 1826.25
  nested$BIN_WIDTH_DAYS <- 180; nested$MIN_BIN_PATIENTS <- 1
  nested$LONGITUDINAL_CSV <- source_path
  options(compass.figure_data_manifest = NULL)
  legacy_patient <- suppressWarnings(cached_profile_patient_and_labs(source_path)$patient_df)
  legacy_labs <- cached_canonical_longitudinal(source_path, ANDROGEN)
  lookups <- list(NULL, tibble(DFCI_MRN = c("001", "004"), stratum = c("NEPC+", "NEPC-")))
  lookups[[3]] <- bind_rows(lookups[[2]], lookups[[2]],
    tibble(DFCI_MRN = "outside-cohort", stratum = c("NEPC+", "NEPC-")))
  expected <- lapply(c(FALSE, TRUE), function(log) lapply(lookups, function(lookup)
    nested$patient_bin_trajectory(legacy_labs, "PSA", stratum_values = lookup, log_scale = log)))
  old <- options(compass.figure_data_manifest = manifest)
  on.exit(options(old), add = TRUE)
  prepared <- cached_profile_patient_and_labs(source_path)$patient_df
  stopifnot(identical(prepared$DFCI_MRN, legacy_patient$DFCI_MRN),
            isTRUE(all.equal(prepared$record_span_days, legacy_patient$record_span_days)),
            identical(prepared$lab_rows, legacy_patient$lab_rows),
            is.null(cached_profile_patient_and_labs(source_path)$labs_df))
  actual_labs <- cached_canonical_longitudinal(source_path, ANDROGEN)
  stopifnot(isTRUE(all.equal(as.data.frame(actual_labs[names(legacy_labs)]), as.data.frame(legacy_labs), check.attributes = FALSE)))
  normalize <- function(d) d %>% mutate(stratum = as.character(stratum)) %>%
    arrange(DFCI_MRN, t_mid, stratum) %>% select(DFCI_MRN, t_bin, stratum, LAB_VALUE, t_mid)
  for (i in 1:2) for (j in seq_along(lookups)) {
    actual <- nested$patient_bin_trajectory(actual_labs, "PSA", stratum_values = lookups[[j]], log_scale = c(FALSE, TRUE)[i])
    stopifnot(isTRUE(all.equal(normalize(actual), normalize(expected[[i]][[j]]), check.attributes = FALSE)))
  }
  for (i in 1:2) stopifnot(identical(normalize(expected[[i]][[2]]), normalize(expected[[i]][[3]])))
  conflict <- tryCatch(nested$patient_bin_trajectory(legacy_labs, "PSA",
    stratum_values = tibble(DFCI_MRN = c("001", "001"), stratum = c("NEPC+", "NEPC-"))), error = identity)
  stopifnot(inherits(conflict, "error"), grepl("Conflicting trajectory labels for 1 patient", conditionMessage(conflict)))
  source("COMPASS/survival_analysis/prepare_metastatic_figure_labels.R")
  legacy_labels <- prepare_metastatic_figure_labels(config$metastatic_sources, legacy_patient) %>% arrange(DFCI_MRN)
  prepared_labels <- figure_read_parquet(manifest$metastatic_labels) %>% arrange(DFCI_MRN)
  label_comparison <- all.equal(as.data.frame(prepared_labels[names(legacy_labels)]),
                               as.data.frame(legacy_labels), check.attributes = FALSE)
  if (!isTRUE(label_comparison)) stop(paste(label_comparison, collapse = "\n"))

  # Gtable capture evaluates and discards plot environments; render-only paths
  # must work with all patient data/caches removed from the R session.
  directory <- file.path(root, "scenes")
  destination <- file.path(root, "figures", "panel")
  builds <- 0L
  build <- function() {
    builds <<- builds + 1L
    plot <- ggplot(data.frame(x = 1:3, y = 3:1), aes(x, y)) + geom_point()
    getOption("compass.figure_capture")(plot, destination, 4, 3, "test")
  }
  m <- prepare_figure_scenes(directory, "source-v1", build)
  stopifnot(builds == 1L, length(m$scenes) == 1L, file.size(m$scenes[[1]]$path) < 1000000)
  prepare_figure_scenes(directory, "source-v1", build)
  stopifnot(builds == 1L)
  clear_read_cache()
  options(compass.figure_data_manifest = NULL)
  first <- render_figure_scene(m$scenes[[1]], m$signature, 72, TRUE)
  stopifnot(first$rendered == 2L, figure_file_complete(paste0(destination, ".png")),
            figure_file_complete(paste0(destination, ".pdf")))
  unchanged <- figure_file_identity(paste0(destination, c(".png", ".pdf")))
  second <- render_figure_scene(m$scenes[[1]], m$signature, 72, TRUE)
  stopifnot(second$rendered == 0L, identical(unchanged, figure_file_identity(unchanged$path)))
  higher <- render_figure_scene(m$scenes[[1]], m$signature, 90, TRUE)
  stopifnot(higher$rendered == 1L,
            identical(unchanged[2, ], figure_file_identity(unchanged$path)[2, ]))
  con <- file(paste0(destination, ".png"), "wb"); writeBin(charToRaw("bad"), con); close(con)
  stopifnot(render_figure_scene(m$scenes[[1]], m$signature, 90, TRUE)$rendered == 1L)
  stopifnot(is.null(figure_scene_manifest(directory, "source-v2")))
})
cat("Prepared figure differential and cache tests passed\n")
