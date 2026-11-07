# Rscript tests/test_figure_cache.R: differential preparation and render-cache tests.
source("COMPASS/figures/COMPASS_generate_figures_pipeline.R")
source("COMPASS/figures/figure_workflow.R")
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
    cohorts = "adt", endpoints = "platinum",
    classifier_path = file.path(root, "classifier"), federated_path = file.path(root, "fed.csv"))
  config$metastatic <- TRUE
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
    c(shQuote("COMPASS/figures/prepare_figure_data.py"), "--config", shQuote(config_path)))
  stopifnot(status == 0L)
  manifest <- jsonlite::read_json(file.path(root, "cache", "manifest.json"))

  # The prepared patients table must match the raw-CSV split exactly.
  options(compass.figure_data_manifest = NULL)
  legacy_patient <- suppressWarnings(cached_profile_patient_and_labs(source_path)$patient_df)
  old <- options(compass.figure_data_manifest = manifest)
  on.exit(options(old), add = TRUE)
  prepared <- cached_profile_patient_and_labs(source_path)$patient_df
  stopifnot(identical(prepared$DFCI_MRN, legacy_patient$DFCI_MRN),
            isTRUE(all.equal(prepared$record_span_days, legacy_patient$record_span_days)),
            identical(prepared$lab_rows, legacy_patient$lab_rows),
            is.null(cached_profile_patient_and_labs(source_path)$labs_df))
  source("COMPASS/figures/figure_supplements.R")
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
  toggle_build <- function() {
    plot <- ggplot(data.frame(x = 1, y = 1), aes(x, y)) + geom_point()
    for (section in c("03_univariate/internal", "04_multivariate/internal", "03_univariate/federated"))
      getOption("compass.figure_capture")(plot, file.path(root, "ADT", section, "panel"), 4, 3, "toggle")
  }
  hidden <- prepare_figure_scenes(file.path(root, "toggle-off"), "off", toggle_build, include_internal = FALSE)
  shown <- prepare_figure_scenes(file.path(root, "toggle-on"), "on", toggle_build, include_internal = TRUE)
  stopifnot(length(hidden$scenes) == 1L, length(shown$scenes) == 3L,
            !figure_internal_path(hidden$scenes[[1]]$destination),
            sum(vapply(shown$scenes, function(s) figure_internal_path(s$destination), logical(1))) == 2L)
  prepare_figure_scenes(directory, "source-v1", build)
  stopifnot(builds == 1L)
  clear_read_cache()
  options(compass.figure_data_manifest = NULL)
  first <- render_figure_scene(m$scenes[[1]], m$signature, 72, TRUE)
  stopifnot(first$rendered == 1L, figure_file_complete(paste0(destination, ".pdf")),
            figure_render_complete(paste0(destination, ".svg")))
  tall_scene <- m$scenes[[1]]
  tall_scene$path <- file.path(root, "tall-scene.rds")
  stopifnot(file.copy(m$scenes[[1]]$path, tall_scene$path))
  tall_scene$destination <- file.path(root, "figures", "tall-panel")
  tall_scene$height <- 51
  stopifnot(render_figure_scene(tall_scene, m$signature, 72, TRUE, overwrite = TRUE)$rendered == 1L,
            figure_file_complete(paste0(tall_scene$destination, ".pdf")))
  receipt <- figure_receipt_path(m$scenes[[1]], "pdf")
  stopifnot(file.exists(receipt), startsWith(receipt, directory),
            length(list.files(file.path(root, "figures"), pattern = "\\.rds$", recursive = TRUE)) == 0L)
  # Migrate old sidecars into data, preserving exact contents and all images.
  legacy <- paste0(destination, ".pdf.cache.rds")
  file.copy(receipt, legacy)
  original_hash <- unname(tools::md5sum(legacy))
  images_before <- figure_file_identity(paste0(destination, ".pdf"))
  stopifnot(figure_archive_legacy_receipts(file.path(root, "figures"), file.path(root, "cache")) == 1L,
            !file.exists(legacy), identical(images_before, figure_file_identity(images_before$path)))
  archived <- list.files(file.path(root, "cache", "legacy_render_receipts"), full.names = TRUE)
  stopifnot(length(archived) == 1L, identical(unname(tools::md5sum(archived)), original_hash),
            figure_archive_legacy_receipts(file.path(root, "figures"), file.path(root, "cache")) == 0L)
  unchanged <- figure_file_identity(paste0(destination, ".pdf"))
  second <- render_figure_scene(m$scenes[[1]], m$signature, 72, TRUE)
  stopifnot(second$rendered == 0L, identical(unchanged, figure_file_identity(unchanged$path)))
  # A vector PDF does not depend on the requested DPI.
  stopifnot(render_figure_scene(m$scenes[[1]], m$signature, 90, TRUE)$rendered == 0L,
            identical(unchanged, figure_file_identity(unchanged$path)))
  unlink(paste0(destination, ".svg"))
  stopifnot(!figure_scene_complete(m$scenes[[1]], m$signature, 72),
            render_figure_scene(m$scenes[[1]], m$signature, 72)$rendered == 1L,
            figure_render_complete(paste0(destination, ".svg")),
            identical(unchanged, figure_file_identity(unchanged$path)))
  con <- file(paste0(destination, ".pdf"), "wb"); writeBin(charToRaw("bad"), con); close(con)
  stopifnot(render_figure_scene(m$scenes[[1]], m$signature, 90, TRUE)$rendered == 1L)
  stopifnot(is.null(figure_scene_manifest(directory, "source-v2")))
})
cat("Prepared figure differential and cache tests passed\n")
