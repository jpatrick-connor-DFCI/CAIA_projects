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
  site_dir <- file.path(root, "nvflare_within_site_cox_univariate"); dir.create(site_dir)
  write_csv(tibble(site_name = rep(c("site_a", "site_b"), each = 3), analysis_label = "adt",
                   landmark_days = rep(c(0, 90, 180), 2), n_patients = 40, n_events = 20),
            file.path(site_dir, "cox_within_site_all_sites_cohort.csv"))
  within_path <- file.path(site_dir, "cox_within_site_all_sites_results.csv")
  write_csv(bind_rows(lapply(c("site_a", "site_b"), function(site)
    mutate(bind_rows(federated), site_name = site, analysis_label = "adt"))), within_path)
  xgb_dir <- file.path(root,"federated_xgboost"); dir.create(xgb_dir)
  xgb_metrics_path <- file.path(xgb_dir,"xgboost_federated_metrics_adt.csv")
  xgb_importance_path <- file.path(xgb_dir,"xgboost_federated_importance_adt.csv")
  write_csv(expand_grid(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(analysis_label="adt",endpoint="platinum",model="xgboost_cox",cohort="all",
      test_c_index=.7,test_mean_auc_t=.75,n_test=20,n_events_test=5),xgb_metrics_path)
  write_csv(expand_grid(landmark_days=c(0,90,180),feature=c("PSA__mean","Testosterone__last","age")) %>%
    mutate(analysis_label="adt",endpoint="platinum",gain=seq_len(n())),xgb_importance_path)
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
            all(vapply(first$rendered, function(x) x$rendered ==
              if(grepl("^0[1-9]_",x$stem)) 2L else 1L, logical(1))))
  manuscript_scenes <- Filter(function(s) isTRUE(s$manuscript),
    unlist(lapply(first$prepared,`[[`,"scenes"),recursive=FALSE))
  stopifnot(length(manuscript_scenes)>=7L,
    all(vapply(manuscript_scenes,function(s) file.exists(paste0(s$destination,".pdf")) &&
      file.exists(paste0(s$destination,".md")) && s$width==7.2,logical(1))))
  platinum_stems <- vapply(first$prepared$adt__platinum$scenes, `[[`, character(1), "stem")
  nepc_stems <- vapply(first$prepared$adt__nepc$scenes, `[[`, character(1), "stem")
  stopifnot(all(c("figure1_cohort", "figure1a_consort", "figure1b_km", "figure1c_span",
    "figure1c_dx_to_tx", "figure1c_time_to_platinum", "figure2v3_llm",
    "figure2v3_confusion_matrix", "figure2v3_metric_bar", "figure2v3_subtype_landscape",
    "figure2v3_enrichment", "classifier_validation", "subtype_platinum") %in% platinum_stems),
    !any(startsWith(nepc_stems,"figure2v3")))
  overview <- Filter(function(s) startsWith(s$stem,"dfci_labs_overview_"),first$prepared$adt__platinum$scenes)
  stopifnot(length(overview)==1L,grepl("/compiled/associations/",overview[[1]]$destination),
    !any(startsWith(nepc_stems,"dfci_labs_overview_")))
  panels <- Filter(function(g) inherits(g,"gtable"),readRDS(overview[[1]]$path)$grobs)
  stopifnot(length(panels)==7)
  stopifnot(length(list.files(cfg$fig_root, pattern = "\\.rds$", recursive = TRUE)) == 0L,
            length(list.files(cfg$cache_root, pattern = "\\.receipt\\.rds$", recursive = TRUE)) > 20L)
  stopifnot(!any(basename(list.dirs(cfg$fig_root, recursive = TRUE)) %in% c("main", "supplements")))
  nested_cache <- tryCatch(run_cached_figure_workflow(modifyList(cfg, list(cache_root = cfg$fig_root)),
    pipeline_path), error = identity)
  stopifnot(inherits(nested_cache, "error"), grepl("non-nested", conditionMessage(nested_cache)))
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
  # Federated forests do not require a matching local result tree.
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
                      c("psa_forest", "testosterone_forest", "site_incidence_lm000",
                        "xgboost_performance", "xgboost_importance",
                        "08_federated_xgboost")))
  # Both forest scenes keep widescreen slide dimensions through 05's cache;
  # the day-zero incidence layout is unchanged.
  for(scene in env$figure_run$prepared$federated$scenes[1:2]) {
    stopifnot(scene$width == 16, scene$height == 9)
  }
  incidence_scene <- env$figure_run$prepared$federated$scenes[[3]]
  stopifnot(incidence_scene$width == 14, incidence_scene$height == 5.5)
  site_export <- read_csv(file.path(cfg$fig_root, "ADT", "federated_no_msk",
                                    "site_incidence_lm000__platinum.csv"), show_col_types = FALSE)
  stopifnot(file.exists(file.path(cfg$fig_root,"ADT","index.html")),
            file.exists(file.path(cfg$fig_root,"ADT","manifest.csv")),
            !dir.exists(file.path(cfg$fig_root,"ADT","by_figure")))
  stopifnot(nrow(site_export) == 2L, all(site_export$landmark_days == 0))
  federated_cfg <- cfg; federated_cfg$scope <- "federated"
  # Both XGBoost files require a refreshed 04 manifest after source changes.
  for(path in c(xgb_metrics_path,xgb_importance_path)) {
    original_time <- file.info(path)$mtime
    Sys.setFileTime(path,original_time+5)
    changed <- tryCatch(figure_notebook_manifest(federated_cfg),error=identity)
    stopifnot(inherits(changed,"error"),grepl(basename(path),conditionMessage(changed),fixed=TRUE))
    Sys.setFileTime(path,original_time)
  }
  Sys.setFileTime(within_path, file.info(within_path)$mtime + 5)
  changed_within <- tryCatch(figure_notebook_manifest(federated_cfg), error = identity)
  stopifnot(inherits(changed_within, "error"),
            grepl("cox_within_site_all_sites_results.csv", conditionMessage(changed_within), fixed = TRUE))
  # Site-count changes must invalidate federation-only runs as well as all runs.
  site_path <- file.path(site_dir, "cox_within_site_all_sites_cohort.csv")
  Sys.setFileTime(site_path, file.info(site_path)$mtime + 5)
  changed_sites <- tryCatch(figure_notebook_manifest(federated_cfg), error = identity)
  stopifnot(inherits(changed_sites, "error"),
            grepl("04_prep_figure_data.ipynb", conditionMessage(changed_sites), fixed = TRUE))
  cat(sprintf("Synthetic workflow: first %.2fs; unchanged %.2fs; %d panels\n", t1, t2, length(first$rendered)))
})
