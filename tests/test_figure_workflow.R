# End-to-end synthetic workflow: two endpoints, independent federation, resume,
# format-only rendering, and missing optional inputs. No clinical data needed.
source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_workflow.R")
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
  write_csv(patient %>% filter(PLATINUM == 1) %>% select(DFCI_MRN, PLATINUM_DATE),
    file.path(root, "mrn_lists", "platinum_any_MRN_list.csv"))
  write_csv(tibble(DFCI_MRN = ids,
    updated_manual_low_confidence = if_else(patient$NEPC == 1, "nepc", "conventional"),
    updated_manual_high_confidence = updated_manual_low_confidence),
    file.path(root, "LLM_NEPC_labels", "full_manual_annotations.csv"))
  federated <- list()
  for (endpoint in c("platinum", "nepc")) {
    suffix <- if (endpoint == "platinum") "" else "_nepc"
    inputs <- file.path(root, "survival_analysis", paste0("prediction_inputs_adt", suffix))
    dir.create(inputs, recursive = TRUE)
    write_csv(tibble(DFCI_MRN = ids, eligible_landmark_0 = TRUE), file.path(inputs, "landmark_mrn_availability.csv"))
    jsonlite::write_json(list(eligible_by_landmark = list(`0` = 40L, `90` = 40L, `180` = 40L)),
      file.path(inputs, "landmark_attrition.json"), auto_unbox = TRUE)
    dir.create(file.path(inputs, "somatic_gleason"))
    jsonlite::write_json(list(cohort_sizes = list(gleason = 30L, sequencing_snv = 12L, sequencing_any = 15L),
      available_case_sensitivities = list(gleason = list(n_patients_by_landmark = list(`0` = 30L, `90` = 31L)))),
      file.path(inputs, "somatic_gleason", "build_manifest.json"), auto_unbox = TRUE)
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
  site_dir <- file.path(root, "nvflare_within_site_univariate_cox"); dir.create(site_dir)
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
  cox_dir <- file.path(root,"federated_cox_multivariate"); dir.create(cox_dir)
  cox_metrics_path <- file.path(cox_dir,"cox_federated_elasticnet_metrics_adt.csv")
  cox_coefficients_path <- file.path(cox_dir,"cox_federated_elasticnet_coefficients_adt.csv")
  write_csv(expand_grid(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(analysis_label="adt",endpoint="platinum",model="elastic_net_cox",cohort="all",
      test_c_index=.68,test_mean_auc_t=.72,n_test=20,n_events_test=5),cox_metrics_path)
  write_csv(expand_grid(landmark_days=c(0,90,180),feature=c("PSA__mean","Testosterone__last__missing","age")) %>%
    mutate(analysis_label="adt",endpoint="platinum",coefficient=rep(c(.2,-.1,.3),3)),cox_coefficients_path)
  # 04: bundles refit without Dana-Farber, scored on local endpoints.
  no_dfci_dir <- file.path(root,"rhino_results","federated_transfer_adt_no_DFCI"); dir.create(no_dfci_dir,recursive=TRUE)
  transfer_metrics_path <- file.path(no_dfci_dir,"transfer_metrics.csv")
  write_csv(expand_grid(endpoint=c("nepc","avpc","platinum"),
      bundle=c("xgboost_federated_model_adt","cox_federated_elasticnet_model_adt"),
      landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(model_family=if_else(startsWith(bundle,"xgboost"),"xgboost_cox","elastic_net_cox"),
      n_patients=40,n_events=8,c_index=if_else(config=="both",.66,.55),c_index_lo=c_index-.05,c_index_hi=c_index+.05,
      delta_c_vs_baseline=if_else(config=="both",.11,NA_real_),delta_c_lo=delta_c_vs_baseline-.04,
      delta_c_hi=delta_c_vs_baseline+.04,mean_auc=if_else(config=="both",.68,.56),mean_auc_lo=mean_auc-.05,
      mean_auc_hi=mean_auc+.05,mean_auc_baseline=.56,delta_auc_vs_baseline=if_else(config=="both",.12,NA_real_),
      delta_auc_lo=delta_auc_vs_baseline-.04,delta_auc_hi=delta_auc_vs_baseline+.04),transfer_metrics_path)
  write_csv(expand_grid(endpoint="nepc",bundle="xgboost_federated_model_adt",landmark_days=0,config="both",
    horizon_days=c(365,730)) %>% mutate(auc=.7,n_cases=4),file.path(no_dfci_dir,"transfer_auc_t.csv"))
  no_dfci_clinical_path <- file.path(no_dfci_dir,"transfer_clinical_comparison.csv")
  write_csv(expand_grid(endpoint=c("avpc","nepc","platinum"),landmark_days=c(0,90,180),
      subpopulation=c("gleason","trio","trio_gleason"),
      bundle=c("xgboost_federated_model_adt","cox_federated_elasticnet_model_adt"),
      model=c("transferred_labs","transferred_age","clinical_cox","combined_cox")) %>%
    mutate(model_family=if_else(startsWith(bundle,"xgboost"),"xgboost_cox","elastic_net_cox"),
      n_patients=20,n_events=6,n_scored=20,n_scored_events=6,c_index=.64,c_index_lo=.58,c_index_hi=.7,
      delta_c_vs_transferred=if_else(model=="transferred_labs",NA_real_,.02),
      delta_lo=delta_c_vs_transferred-.05,delta_hi=delta_c_vs_transferred+.05,
      mean_auc=.66,mean_auc_lo=.6,mean_auc_hi=.72,delta_auc_vs_transferred=if_else(model=="transferred_labs",NA_real_,.03),
      delta_auc_lo=delta_auc_vs_transferred-.05,delta_auc_hi=delta_auc_vs_transferred+.05),no_dfci_clinical_path)
  pipeline_path <- normalizePath("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
  cfg <- list(data_root = root, cache_root = file.path(root, "cache"), fig_root = file.path(root, "figures"),
    cohorts = "adt", endpoints = c("platinum", "nepc"), scope = "all", labs = ANDROGEN,
    classifier_path = file.path(root, "classifier"), gam = FALSE, adt_intent = FALSE,
    metastatic = FALSE, metastatic_extra = FALSE, forest_cohorts = "adt", forest_landmark = 180L,
    federated = TRUE, federated_path = fed_path)
  dir.create(cfg$classifier_path)
  classifier <- tibble(DFCI_MRN = ids, primary_label = if_else(patient$NEPC == 1, "nepc", "conventional"),
    nepc_status = if_else(patient$NEPC == 1, "nepc", "non_nepc"), is_avpc = FALSE)
  # Repeated consistent NEPC labels must not abort the cached trajectory path.
  write_labels <- if (requireNamespace("arrow", quietly = TRUE)) arrow::write_parquet else nanoparquet::write_parquet
  write_labels(bind_rows(classifier, classifier[1:3, ]),
    file.path(cfg$classifier_path, "primary_subtype_labels.parquet"))
  write_csv(tibble(DFCI_MRN = ids, ADT_INTENT = rep(c("METASTATIC", "LOCALIZED_ADJUVANT"), 20)),
    file.path(root, "mrn_lists", "adt_intent_labels_model_cohort.csv"))
  write_csv(tibble(DFCI_MRN = ids[-1], LLM_METASTATIC = patient$NEPC[-1] == 1),
    file.path(root, "mrn_lists", "llm_met_labels_model_cohort.csv"))
  fc <- list(results = fed_path)
  forest <- list(script = file.path(dirname(pipeline_path), "figure_supplements.R"), cohorts = "adt", landmark = 180L)
  # Fixture setup represents the separate notebook preparation stage. Python
  # is then unavailable for every R workflow stage, not just render-only.
  missing <- tryCatch(figure_notebook_manifest(cfg), error = identity)
  stopifnot(inherits(missing, "error"), grepl("06_prep_figure_data.ipynb", conditionMessage(missing), fixed = TRUE))
  config_path <- file.path(root, "fixture-config.json")
  jsonlite::write_json(cfg, config_path, auto_unbox = TRUE)
  stopifnot(system2("python3", c(shQuote(file.path(dirname(pipeline_path), "prepare_figure_data.py")),
    "--config", shQuote(config_path))) == 0L)
  for (name in c("system", "system2")) assign(name, function(...) stop("Process launch forbidden during R figures"), .GlobalEnv)
  on.exit(rm(list = c("system", "system2"), envir = .GlobalEnv), add = TRUE)
  mismatch <- tryCatch(figure_notebook_manifest(modifyList(cfg, list(cohorts = "arpi"))), error = identity)
  stopifnot(inherits(mismatch, "error"), grepl("06_prep_figure_data.ipynb", conditionMessage(mismatch), fixed = TRUE))
  t1 <- system.time(first <- run_cached_figure_workflow(cfg, pipeline_path, "all", dpi = 60,
    prepare_workers = 1L, render_workers = 2L, forest_config = forest, federated_config = fc))[["elapsed"]]
  stopifnot(length(first$prepared) == 4L, length(first$rendered) > 20L,
            all(vapply(first$rendered, function(x) x$rendered ==
              if(grepl("^0[1-9]_",x$stem)) 2L else 1L, logical(1))))
  manuscript_scenes <- Filter(function(s) isTRUE(s$manuscript),
    unlist(lapply(first$prepared,`[[`,"scenes"),recursive=FALSE))
  stopifnot(length(manuscript_scenes)>=7L,
    all(vapply(manuscript_scenes,function(s) grepl("/ADT/0[1-6]_[a-z_]+(/internal|/federated)?/manuscript/",s$destination) &&
      file.exists(paste0(s$destination,".pdf")) &&
      file.exists(figure_sidecar_path(paste0(s$destination,".md"))) && s$width==7.2,logical(1))))
  overview_stems <- vapply(first$prepared$cohort_overview$scenes,`[[`,character(1),"stem")
  stopifnot("06_dfci_cohort_sensitivity" %in% overview_stems)
  platinum_stems <- vapply(first$prepared$adt__platinum$scenes, `[[`, character(1), "stem")
  nepc_stems <- vapply(first$prepared$adt__nepc$scenes, `[[`, character(1), "stem")
  # Compiled pages whose panels all appear in a manuscript figure are dropped
  # as duplicates; their individual panels are still exported.
  stopifnot(all(c("figure1a_consort", "figure1b_km", "figure1c_span",
    "figure1c_dx_to_tx", "figure1c_time_to_platinum",
    "figure1d_dfci_data", "figure1e_site_counts",
    unlist(lapply(names(FIGURE2_COHORTS), figure2v3_stems))) %in% platinum_stems),
    !any(c("figure1_cohort", paste0(c("figure2v3_llm_", "classifier_validation_", "subtype_platinum_"),
      rep(names(FIGURE2_COHORTS), each = 3))) %in% platinum_stems),
    all(paste0("02_llm_nepc_avpc__", names(FIGURE2_COHORTS)) %in% platinum_stems),
    "01_cohort_overview" %in% platinum_stems,
    !any(startsWith(platinum_stems,"dfci_labs_overview_")),
    !any(startsWith(nepc_stems,"figure2v3")), !any(startsWith(nepc_stems,"dfci_labs_overview_")))
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
  stopifnot(inherits(stale, "error"), grepl("06_prep_figure_data.ipynb", conditionMessage(stale), fixed = TRUE))
  # Run the actual 07_figures.R with a federation-only scope and raw patient
  # input hidden. This verifies configuration wiring as well as the helper API.
  variables <- c(CAIA_PROJECT_ROOT = normalizePath("."),
    COMPASS_DATA_ROOT = root, COMPASS_FIG_ROOT = cfg$fig_root,
    COMPASS_FIGURE_DATA_ROOT = cfg$cache_root, COMPASS_FIGURE_SCOPE = "federated",
    COMPASS_FEDERATED_RESULTS = fed_path)
  previous <- Sys.getenv(names(variables), unset = NA)
  on.exit(for (name in names(previous)) {
    if (is.na(previous[[name]])) Sys.unsetenv(name)
    else do.call(Sys.setenv, setNames(list(previous[[name]]), name))
  }, add = TRUE)
  do.call(Sys.setenv, as.list(variables))
  # Federated forests do not require a matching local result tree.
  stopifnot(file.rename(file.path(root, "survival_analysis", "local_runs_adt"),
                        file.path(root, "local_runs_hidden")))
  env <- new.env(parent = globalenv())
  source(file.path(dirname(pipeline_path), "07_figures.R"), local = env)
  stopifnot(identical(names(env$figure_run$prepared), "federated"))
  stopifnot(identical(vapply(env$figure_run$prepared$federated$scenes, `[[`, character(1), "stem"),
                      c("psa_forest", "testosterone_forest", "site_incidence_lm000",
                        "elasticnet_performance", "elasticnet_coefficients",
                        "xgboost_performance", "xgboost_importance",
                        paste0("federated_volcano_landmark", c(0, 90, 180)),
                        "transfer_discrimination_no_dfci", "transfer_clinical_comparison_no_dfci",
                        "08_federated_multivariable",
                        "federated_05_transfer_no_dfci",
                        "federated_05_transfer_no_dfci__no_d")))
  coefficient_export <- read_csv(file.path(cfg$fig_root, "ADT", "04_multivariate", "federated", "data",
    "elasticnet_coefficients__platinum.csv"), show_col_types = FALSE)
  stopifnot(nrow(coefficient_export) == 9L, sum(coefficient_export$displayed) == 6L,
    "last missing" %in% coefficient_export$feature_stat)
  transfer_export <- read_csv(file.path(cfg$fig_root, "ADT", "05_transfer", "data",
    "transfer_discrimination_no_dfci__platinum.csv"), show_col_types = FALSE)
  stopifnot(nrow(transfer_export) == 36L, file.exists(file.path(cfg$fig_root, "ADT", "05_transfer", "data", c(
    "transfer_auc_t_no_dfci__platinum.csv", "transfer_clinical_comparison_no_dfci__platinum.csv"))))
  # Nothing is rendered for the removed with-Dana-Farber arm.
  stopifnot(!any(file.exists(file.path(cfg$fig_root, "ADT", "05_transfer", "data", c(
    "transfer_discrimination__platinum.csv", "transfer_dfci_ablation__platinum.csv")))))
  # Both forest scenes keep widescreen slide dimensions through 07's cache;
  # the day-zero incidence layout is unchanged.
  for(scene in env$figure_run$prepared$federated$scenes[1:2]) {
    stopifnot(scene$width == 16, scene$height == 9)
  }
  incidence_scene <- env$figure_run$prepared$federated$scenes[[3]]
  stopifnot(incidence_scene$width == 14, incidence_scene$height == 5.5)
  site_export <- read_csv(file.path(cfg$fig_root, "ADT", "01_cohort", "data",
                                    "site_incidence_lm000__platinum.csv"), show_col_types = FALSE)
  stopifnot(file.exists(file.path(cfg$fig_root,"ADT","data","index.html")),
            file.exists(file.path(cfg$fig_root,"ADT","data","manifest.csv")),
            !dir.exists(file.path(cfg$fig_root,"ADT","by_figure")),
            !any(dir.exists(file.path(cfg$fig_root,"ADT",FIGURE_RETIRED_FOLDERS))))
  stopifnot(nrow(site_export) == 2L, all(site_export$landmark_days == 0))
  federated_cfg <- cfg; federated_cfg$scope <- "federated"
  # Elastic-net, XGBoost and 04 transfer files require a refreshed 06 manifest after source changes.
  for(path in c(xgb_metrics_path,xgb_importance_path,cox_metrics_path,cox_coefficients_path,
                transfer_metrics_path,no_dfci_clinical_path)) {
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
            grepl("06_prep_figure_data.ipynb", conditionMessage(changed_sites), fixed = TRUE))
  # Optional DeepHit and checklist exports render after a later delivery. The
  # first workflow run above already exercised the missing-input fallback.
  for(suffix in c("","_no_DFCI")) {
    directory <- file.path(root,paste0("fast_federated_deephit",suffix))
    dir.create(directory)
    write_csv(expand_grid(landmark_days=c(0,90,180),training_arm=c("landmark","dynamic")) %>%
      mutate(analysis_label="adt",endpoint="platinum",config="both",
        test_mean_auc_t=if_else(training_arm=="dynamic",.74,.70),
        test_c_index=.68,n_test=20,n_events_test=5),
      file.path(directory,"deephit_federated_metrics_adt.csv"))
  }
  dh <- load_federated_dynamic_deephit(fed_path)
  dh_no_dfci <- load_federated_dynamic_deephit(fed_path,"no_dfci")
  stopifnot(nrow(dh)==3L,nrow(dh_no_dfci)==3L,
    all(dh$name=="Dynamic DeepHit"),
    all(dh_no_dfci$name=="Dynamic DeepHit, no DFCI"))
  render_federated_deephit_performance(file.path(cfg$fig_root,"ADT"),fed_path,60,TRUE)
  stopifnot(file.exists(federated_output_path(file.path(cfg$fig_root,"ADT"),
    "deephit_performance",".png")))
  perf <- bind_rows(
    read_csv(file.path(cfg$fig_root,"ADT","04_multivariate","federated","data",
      "elasticnet_performance__platinum.csv"),show_col_types=FALSE),
    read_csv(file.path(cfg$fig_root,"ADT","04_multivariate","federated","data",
      "xgboost_performance__platinum.csv"),show_col_types=FALSE),dh)
  importance_export <- read_csv(file.path(cfg$fig_root,"ADT","04_multivariate","federated","data",
    "xgboost_importance__platinum.csv"),show_col_types=FALSE)
  main_with_dynamic <- manuscript_federated_multivariable(perf,coefficient_export,importance_export)
  stopifnot("Dynamic DeepHit" %in% as.character(main_with_dynamic$data$name))
  no_dfci_perf <- bind_rows(filter(perf,family!="dynamic_deephit"),dh_no_dfci)
  main_no_dfci <- manuscript_federated_multivariable_no_dfci(perf,no_dfci_perf,
    coefficient_export,importance_export)
  stopifnot("Dynamic DeepHit, no DFCI" %in% as.character(main_no_dfci$data$name))
  all_models_dir <- file.path(no_dfci_dir,"all_models"); dir.create(all_models_dir)
  write_csv(expand_grid(endpoint=c("platinum","nepc"),landmark_days=c(0,90,180),
      model_horizon_days=c(365,1825),config=c("both","baseline")) %>%
    mutate(bundle="predictive_checklist_federated_model_adt",
      model_family="predictive_checklist",evaluation_horizon_days=model_horizon_days,
      mean_auc=if_else(config=="both",.72,.61),mean_auc_lo=mean_auc-.04,
      mean_auc_hi=mean_auc+.04,brier=.13),
    file.path(all_models_dir,"transfer_metrics.csv"))
  all_model_metrics <- read_csv(file.path(all_models_dir,"transfer_metrics.csv"),show_col_types=FALSE)
  write_csv(bind_rows(all_model_metrics,
    tibble(bundle=c("cox_bundle","deephit_bundle"),
      model_family=c("elastic_net_cox","dynamic_deephit"))),
    file.path(all_models_dir,"transfer_metrics.csv"))
  auc_grid <- expand_grid(endpoint=c("platinum","nepc"),landmark_days=c(0,90,180),
      horizon_days=c(365,1825),
      bundle=c("cox_bundle","deephit_bundle","predictive_checklist_federated_model_adt")) %>%
    mutate(config="both",auc=.68,n_cases=5,
      model_horizon_days=if_else(bundle=="predictive_checklist_federated_model_adt",horizon_days,NA_real_),
      evaluation_horizon_days=if_else(bundle=="cox_bundle",NA_real_,horizon_days),
      training_arm=if_else(bundle=="deephit_bundle","landmark",NA_character_))
  write_csv(bind_rows(auc_grid,
    filter(auc_grid,bundle=="deephit_bundle") %>% mutate(training_arm="dynamic",auc=.72)),
    file.path(all_models_dir,"transfer_auc_t.csv"))
  render_federated_checklist_supplement(file.path(cfg$fig_root,"ADT"),all_models_dir,60,TRUE)
  stopifnot(file.exists(federated_output_path(file.path(cfg$fig_root,"ADT"),
    "transfer_predictive_checklist_supplement",".png")))
  all_model_plot <- render_federated_all_model_transfer_supplement(
    file.path(cfg$fig_root,"ADT"),all_models_dir,60,TRUE)
  stopifnot(all(c("DeepHit landmark","DeepHit dynamic","Predictive checklist") %in%
    unique(all_model_plot$series)),
    file.exists(federated_output_path(file.path(cfg$fig_root,"ADT"),
      "transfer_all_models_supplement",".png")))
  # Checklist supplement: cards for a fitted landmark and an unfit one, no workflow cache needed.
  source("COMPASS/survival_analysis/checklist_supplement.R")
  checklist_dir <- file.path(root,"checklist_tables"); dir.create(checklist_dir)
  cells <- expand_grid(feature_set=c("labs","labs_gleason_somatic"),endpoint=c("platinum","nepc"),
      landmark_days=c(0,90)) %>%
    mutate(horizon_days=1825,population="full",n_test=100,n_events_total=30,
      status=if_else(landmark_days==0,"ok","unsupported"),
      reason=if_else(landmark_days==0,NA_character_,"Too few events"),
      threshold=if_else(landmark_days==0,2,NA_real_),n_items=if_else(landmark_days==0,3L,NA_integer_))
  write_csv(cells,file.path(checklist_dir,"checklist_cells.csv"))
  write_csv(filter(cells,status=="ok") %>% select(feature_set,endpoint,landmark_days,horizon_days,population) %>%
    slice(rep(row_number(),each=3)) %>% group_by(feature_set,endpoint,landmark_days) %>%
    mutate(item_order=row_number(),name=paste("PSA >= ",item_order,"ng/mL, last value before the landmark"),
      feature="PSA",op=">=",threshold=item_order,group="PSA") %>% ungroup(),
    file.path(checklist_dir,"checklist_items_long.csv"))
  write_csv(filter(cells,status=="ok") %>% select(feature_set,endpoint,landmark_days,horizon_days,population) %>%
    slice(rep(row_number(),each=4)) %>% group_by(feature_set,endpoint,landmark_days) %>%
    mutate(model=c("checklist","age_cox","elastic_net","xgboost"),dataset="test",status="ok",
      auc=.7,auc_lower=.6,auc_upper=.8,brier=.12,sensitivity=.6,specificity=.7,ppv=.3,npv=.9) %>% ungroup(),
    file.path(checklist_dir,"checklist_metrics.csv"))
  captured <- character()
  render_checklist_supplement(checklist_dir,file.path(root,"checklist_figs"),60,TRUE,
    function(plot,path,width,height,dpi,overwrite) {
      stopifnot(inherits(plot,"ggplot")); captured <<- c(captured,basename(path))
    }, figure_sidecar_path)
  stopifnot(setequal(captured,c("figure4s_checklist_labs__platinum.png","figure4s_checklist_labs__nepc.png",
      "figure4s_checklist_labs_gleason_somatic__platinum.png","figure4s_checklist_labs_gleason_somatic__nepc.png")),
    file.exists(file.path(root,"checklist_figs","data","figure4s_checklist_items_long.csv")))
  stopifnot(is.null(suppressWarnings(render_checklist_supplement(file.path(root,"absent"),root,60,TRUE,NULL))))
  cat(sprintf("Synthetic workflow: first %.2fs; unchanged %.2fs; %d panels\n", t1, t2, length(first$rendered)))
})
