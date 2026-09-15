# End-to-end synthetic workflow: two endpoints, independent federation, resume,
# format-only rendering, and missing optional inputs. No clinical data needed.
source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_data_cache.R")
local({
  root <- tempfile("figure-workflow-", tmpdir = Sys.getenv("COMPASS_TEST_OUTPUT_ROOT", tempdir())); dir.create(root)
  if (Sys.getenv("COMPASS_TEST_KEEP_OUTPUT") == "1") message("Synthetic outputs: ", root)
  else on.exit(unlink(root, recursive = TRUE))
  ids <- as.character(1:40)
  patient <- tibble(DFCI_MRN = ids, AGE_AT_TREATMENTSTART = 50 + seq_along(ids)/2,
    FIRST_RECORD_DATE = "2018-01-01", DIAGNOSIS_DATE = "2019-01-01",
    TREATMENT_ANCHOR_DATE = "2020-01-01", LAST_CONTACT_DATE = "2030-01-01",
    PLATINUM_DATE = "2022-01-01", PLATINUM_MEDICATION = "carboplatin",
    DEATH = rep(c(0, 1), 20), PLATINUM = rep(c(0, 1), 20),
    NEPC = rep(c(0, 0, 1, 1), 10), t_nepc = 3000, t_platinum = 800 + seq_along(ids),
    t_death = 1500 + seq_along(ids), t_last_contact = 3650, t_diagnosis = -365,
    t_first_treatment = 0)
  long <- crossing(DFCI_MRN = ids, LAB_NAME = c("PSA", "Testosterone"), t_lab = c(-180, 0, 180, 360)) %>%
    left_join(patient, by = "DFCI_MRN") %>%
    mutate(LAB_VALUE = as.numeric(DFCI_MRN) + t_lab/180 + 1,
           LAB_DATE = as.character(as.Date("2020-01-01") + t_lab))
  write_csv(long, file.path(root, "longitudinal_prediction_data_adt.csv"))
  dir.create(file.path(root, "mrn_lists")); dir.create(file.path(root, "LLM_NEPC_labels"))
  flags <- tibble(DFCI_MRN = ids)
  for (col in c("HAS_NON_PROSTATE_PRIMARY", "HAS_POST_ADT_EXCLUSION_CANCER", "PARPI_EXPOSED", "PLATINUM_BEFORE_DIAGNOSIS")) flags[[col]] <- 0
  for (col in c("DATED_PROSTATE_DIAGNOSIS", "MALE", "ARPI_DOCETAXEL_EXPOSED", "ADT_EXPOSED", "HAS_5_OR_MORE_PSA_TESTS", "ELIGIBLE")) flags[[col]] <- 1
  write_csv(flags, file.path(root, "mrn_lists", "icd_prostate_mrn_flags.csv"))
  write_csv(patient %>% filter(PLATINUM == 1) %>% select(DFCI_MRN), file.path(root, "mrn_lists", "platinum_MRN_list.csv"))
  write_csv(tibble(DFCI_MRN = ids, NEPC = patient$NEPC,
    simplified_manual_platinum_reason = if_else(patient$NEPC == 1, "nepc", "conventional")),
    file.path(root, "LLM_NEPC_labels", "baca_lab_annotations.csv"))
  federated <- list()
  for (endpoint in c("platinum", "nepc")) {
    suffix <- if (endpoint == "platinum") "" else "_nepc"
    inputs <- file.path(root, "survival_analysis", paste0("prediction_inputs_adt", suffix))
    dir.create(inputs, recursive = TRUE)
    write_csv(tibble(DFCI_MRN = ids, eligible_landmark_0 = TRUE), file.path(inputs, "landmark_mrn_availability.csv"))
    jsonlite::write_json(list(eligible_by_landmark = list(`0` = 40L, `90` = 40L, `180` = 40L)),
      file.path(inputs, "landmark_attrition.json"), auto_unbox = TRUE)
    for (lm in c(0L, 90L, 180L)) {
      results <- file.path(root, "survival_analysis", paste0("local_runs_adt", suffix), "cox", paste0("landmark_", lm), "both")
      dir.create(results, recursive = TRUE)
      d <- tibble(landmark_days = lm, endpoint = endpoint,
        feature = c("PSA__mean", "Testosterone__mean"), lab_name = c("PSA", "Testosterone"),
        feature_stat = "mean", coef_feature = c(.3, -.3), hazard_ratio_per_sd = exp(coef_feature),
        ci_lower = exp(coef_feature - .1), ci_upper = exp(coef_feature + .1), p_value = c(.001, .1),
        q_value = c(.01, .2), n_patients_used = 40, n_events_used = 20)
      write_csv(d, file.path(results, "cox_agg_univariate_nobs_adjusted.csv"))
      if (endpoint == "platinum") federated[[length(federated) + 1L]] <- d
      write_csv(patient %>% mutate(PSA__mean = as.numeric(DFCI_MRN), Testosterone__mean = 50 - as.numeric(DFCI_MRN)),
        file.path(inputs, paste0("aggregated_landmark", lm, ".csv")))
    }
  }
  fed_path <- file.path(root, "federated.csv"); write_csv(bind_rows(federated), fed_path)
  pipeline_path <- normalizePath("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
  cfg <- list(data_root = root, cache_root = file.path(root, "cache"), fig_root = file.path(root, "figures"),
    cohorts = "adt", endpoints = c("platinum", "nepc"), scope = "all", labs = ANDROGEN,
    classifier_path = file.path(root, "classifier"), gam = FALSE, adt_intent = FALSE,
    metastatic = FALSE, metastatic_extra = FALSE, forest_cohorts = "adt", forest_landmark = 180L,
    federated = TRUE, federated_path = fed_path)
  dir.create(cfg$classifier_path)
  classifier <- tibble(DFCI_MRN = ids, primary_label = if_else(patient$NEPC == 1, "nepc", "conventional"),
    has_nepc = patient$NEPC, has_avpc = 0)
  # Repeated consistent NEPC labels must not abort the cached trajectory path.
  write_tsv(bind_rows(classifier, classifier[1:3, ]),
    file.path(cfg$classifier_path, "LLM_NEPC_classifier_labels.tsv"))
  write_csv(tibble(DFCI_MRN = ids, ADT_INTENT = rep(c("METASTATIC", "LOCALIZED_ADJUVANT"), 20)),
    file.path(root, "mrn_lists", "adt_intent_labels_model_cohort.csv"))
  write_csv(tibble(DFCI_MRN = ids[-1], LLM_METASTATIC = patient$NEPC[-1] == 1),
    file.path(root, "mrn_lists", "llm_met_labels_model_cohort.csv"))
  fc <- list(script = file.path(dirname(pipeline_path), "federated_no_msk_figures.R"), results = fed_path)
  forest <- list(script = file.path(dirname(pipeline_path), "cohort_forest_figures.R"), cohorts = "adt", landmark = 180L)
  # Fixture setup represents the separate notebook preparation stage. Python
  # is then unavailable for every R workflow stage, not just render-only.
  missing <- tryCatch(figure_notebook_manifest(cfg), error = identity)
  stopifnot(inherits(missing, "error"), grepl("04_prep_figure_data.ipynb", conditionMessage(missing), fixed = TRUE))
  config_path <- file.path(root, "fixture-config.json")
  jsonlite::write_json(cfg, config_path, auto_unbox = TRUE)
  stopifnot(system2("python3", c(shQuote(file.path(dirname(pipeline_path), "prepare_figure_data.py")),
    "--config", shQuote(config_path))) == 0L)
  for (name in c("system", "system2")) assign(name, function(...) stop("Process launch forbidden during R figures"), .GlobalEnv)
  on.exit(rm(list = c("system", "system2"), envir = .GlobalEnv), add = TRUE)
  mismatch <- tryCatch(figure_notebook_manifest(modifyList(cfg, list(cohorts = "arpi"))), error = identity)
  stopifnot(inherits(mismatch, "error"), grepl("04_prep_figure_data.ipynb", conditionMessage(mismatch), fixed = TRUE))
  t1 <- system.time(first <- run_cached_figure_workflow(cfg, pipeline_path, "all", dpi = 60,
    prepare_workers = 1L, render_workers = 2L, forest_config = forest, federated_config = fc))[["elapsed"]]
  stopifnot(length(first$prepared) == 4L, length(first$rendered) > 20L,
            all(vapply(first$rendered, function(x) x$rendered == 1L, logical(1))))
  t2 <- system.time(second <- run_cached_figure_workflow(cfg, pipeline_path, "all", dpi = 60,
    forest_config = forest, federated_config = fc))[["elapsed"]]
  stopifnot(all(vapply(second$rendered, function(x) x$rendered == 0L, logical(1))))
  prepared_only <- run_cached_figure_workflow(cfg, pipeline_path, "prepare", dpi = 60,
    forest_config = forest, federated_config = fc)
  stopifnot(length(prepared_only$prepared) == 4L, length(prepared_only$rendered) == 0L)
  # Render-only must not even start Python or read a raw input.
  python <- Sys.getenv("COMPASS_FIGURE_PYTHON", unset = NA)
  Sys.setenv(COMPASS_FIGURE_PYTHON = "/no/python/allowed")
  on.exit(if (is.na(python)) Sys.unsetenv("COMPASS_FIGURE_PYTHON") else Sys.setenv(COMPASS_FIGURE_PYTHON = python), add = TRUE)
  file.rename(file.path(root, "longitudinal_prediction_data_adt.csv"), file.path(root, "raw_hidden.csv"))
  third <- run_cached_figure_workflow(cfg, pipeline_path, "render", dpi = 60,
    forest_config = forest, federated_config = fc)
  stopifnot(all(vapply(third$rendered, function(x) x$rendered == 0L, logical(1))))
  stale <- tryCatch(figure_notebook_manifest(cfg), error = identity)
  stopifnot(inherits(stale, "error"), grepl("04_prep_figure_data.ipynb", conditionMessage(stale), fixed = TRUE))
  # Execute the actual Rmd chunks with a federation-only scope and raw patient
  # input hidden. This verifies configuration wiring as well as the helper API.
  variables <- c(COMPASS_DATA_ROOT = root, COMPASS_FIG_ROOT = cfg$fig_root,
    COMPASS_FIGURE_DATA_ROOT = cfg$cache_root, COMPASS_FIGURE_SCOPE = "federated",
    COMPASS_FEDERATED_NO_MSK_RESULTS = fed_path)
  previous <- Sys.getenv(names(variables), unset = NA)
  on.exit(for (name in names(previous)) {
    if (is.na(previous[[name]])) Sys.unsetenv(name)
    else do.call(Sys.setenv, setNames(list(previous[[name]]), name))
  }, add = TRUE)
  do.call(Sys.setenv, as.list(variables))
  # Federated volcanoes do not require a matching local result tree.
  stopifnot(file.rename(file.path(root, "survival_analysis", "local_runs_adt"),
                        file.path(root, "local_runs_hidden")))
  rmd <- readLines(file.path(dirname(pipeline_path), "05_figures.Rmd"))
  inside <- FALSE; code <- character()
  for (line in rmd) {
    if (grepl("^```\\{r", line)) { inside <- TRUE; next }
    if (line == "```") { inside <- FALSE; next }
    if (inside) code <- c(code, line)
  }
  env <- new.env(parent = globalenv())
  eval(parse(text = code), env)
  stopifnot(identical(names(env$figure_run$prepared), "federated"))
  stopifnot(identical(vapply(env$figure_run$prepared$federated$scenes, `[[`, character(1), "stem"),
                      paste0("volcano_landmark", c(0, 90, 180))))
  cat(sprintf("Synthetic workflow: first %.2fs; unchanged %.2fs; %d panels\n", t1, t2, length(first$rendered)))
})
