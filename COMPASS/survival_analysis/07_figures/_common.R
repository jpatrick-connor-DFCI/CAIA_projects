# Shared configuration for the manuscript figure entry points.
# ---- Configuration ----------------------------------------------------------
options(readr.num_threads = 1L, readr.read_lazy = FALSE)
# Absolute code root (same as GAM/04_gam.ipynb) so the script runs from any
# working directory; set CAIA_PROJECT_ROOT to use a different checkout.
PROJECT_ROOT <- Sys.getenv("CAIA_PROJECT_ROOT", "/data/gusev/USERS/jpconnor/code/CAIA")
pipeline_path <- normalizePath(file.path(PROJECT_ROOT, "COMPASS", "survival_analysis",
                                         "COMPASS_generate_figures_pipeline.R"), mustWork = TRUE)
source(pipeline_path)
source(file.path(dirname(pipeline_path), "figure_workflow.R"))

env_flag <- function(name, default = FALSE) {
  value <- tolower(Sys.getenv(name, if (default) "true" else "false"))
  if (!value %in% c("true", "false", "1", "0", "yes", "no")) stop(name, " must be true or false")
  value %in% c("true", "1", "yes")
}
env_values <- function(name, default) trimws(strsplit(Sys.getenv(name, default), ",", fixed = TRUE)[[1]])
env_workers <- function(name, default) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, as.character(default))))
  if (is.na(value) || value < 1L) stop(name, " must be a positive integer")
  value
}
NEPC_PROJ_PATH <- Sys.getenv("COMPASS_DATA_ROOT", "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS")
FIG_ROOT <- Sys.getenv("COMPASS_FIG_ROOT", "/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS")
FIGURE_DATA_ROOT <- Sys.getenv("COMPASS_FIGURE_DATA_ROOT", file.path(NEPC_PROJ_PATH, "figure_data"))
FIGURE_STAGE <- match.arg(Sys.getenv("COMPASS_FIGURE_STAGE", "all"), c("all", "prepare", "render"))
FIGURE_SCOPE <- match.arg(Sys.getenv("COMPASS_FIGURE_SCOPE", "all"), c("all", "federated"))
# Platinum only by default; set COMPASS_FIGURE_ENDPOINTS=nepc,platinum to restore
# the NEPC landmark figures alongside it (or =nepc for that endpoint alone).
ENDPOINTS <- env_values("COMPASS_FIGURE_ENDPOINTS", "platinum")
ADT_COHORT_OPTIONS <- c("adt", "adt_noprecastrate", "adt_metastatic_adt",
  "adt_metastatic_adt_noprecastrate", "adt_metastatic_llm", "adt_metastatic_llm_noprecastrate")
COHORTS <- env_values("COMPASS_FIGURE_COHORTS", "adt")
stopifnot(length(COHORTS) > 0L, length(ENDPOINTS) > 0L,
          all(COHORTS %in% SUPPORTED_COHORTS), all(ENDPOINTS %in% SUPPORTED_ENDPOINTS),
          !anyDuplicated(COHORTS), !anyDuplicated(ENDPOINTS))
COHORT_FOREST_CONFIG <- list(script = file.path(dirname(pipeline_path), "figure_supplements.R"),
                             cohorts = ADT_COHORT_OPTIONS, landmark = 180L)
PLOT_NON_ANDROGEN_LAB_FIGURES <- env_flag("COMPASS_NON_ANDROGEN_LABS")
# Extra per-lab DFCI trajectory/KM figures beside PSA/testosterone (supplement);
# set to "" for the androgen axis only. Must match the 06 notebook's LABS.
SUPPLEMENT_LABS <- setdiff(env_values("COMPASS_SUPPLEMENT_LABS",
  paste(DEFAULT_SUPPLEMENT_LABS, collapse = ",")), "")
if (!all(SUPPLEMENT_LABS %in% names(CATEGORY_MAP)))
  stop("COMPASS_SUPPLEMENT_LABS has unknown labs: ",
       paste(setdiff(SUPPLEMENT_LABS, names(CATEGORY_MAP)), collapse = ", "))
PLOT_GAM_TRAJECTORIES <- env_flag("COMPASS_GAM_TRAJECTORIES")
PLOT_ADT_INTENT_SUPPLEMENT <- env_flag("COMPASS_ADT_INTENT_SUPPLEMENT")
PLOT_METASTATIC_LABEL_SUPPLEMENT <- env_flag("COMPASS_METASTATIC_LABEL_SUPPLEMENT", TRUE)
PLOT_FEDERATED_SUPPLEMENT <- env_flag("COMPASS_FEDERATED_SUPPLEMENT", TRUE)
PLOT_CHECKLIST_SUPPLEMENT <- env_flag("COMPASS_CHECKLIST_SUPPLEMENT", TRUE)
CHECKLIST_RUNS <- list(
  labs = Sys.getenv("COMPASS_CHECKLIST_LABS_RUN", file.path(
    NEPC_PROJ_PATH, "survival_analysis", "predictive_checklists", "full_adt")),
  trio = Sys.getenv("COMPASS_CHECKLIST_TRIO_RUN", file.path(
    NEPC_PROJ_PATH, "survival_analysis", "predictive_checklists", "gleason_trio_labs")))
METASTATIC_SUPPLEMENT <- list(
  prepare_script = file.path(dirname(pipeline_path), "figure_supplements.R"),
  r_script = file.path(dirname(pipeline_path), "figure_supplements.R"),
  intent = file.path(NEPC_PROJ_PATH, "mrn_lists", "adt_intent_labels_model_cohort.csv"),
  stage = Sys.getenv("COMPASS_REGEX_STAGE_PATH", file.path(
    Sys.getenv("PROFILE_DATA_PATH", "/data/gusev/USERS/jpconnor/data/PROFILE_DATA"),
    "CANCER_ANNOTATIONS", "CANCER_STAGE_NOTE_LEVEL.parquet")),
  llm = Sys.getenv("LLM_MET_LABELS_PATH", "/data/gusev/USERS/jpconnor/data/LLM_annotations/LLM_met_diagnosis/met_dx_labels.parquet"),
  icd = file.path(NEPC_PROJ_PATH, "prostate_icd_data.csv"))
RHINO_ROOT <- Sys.getenv("COMPASS_RHINO_ROOT", file.path(NEPC_PROJ_PATH, "rhino_data_v2"))
FEDERATED_SUPPLEMENT <- list(
  results = Sys.getenv("COMPASS_FEDERATED_RESULTS", file.path(RHINO_ROOT, "cox_federated_univariate_adt.csv")),
  transfer_no_dfci = Sys.getenv("COMPASS_FEDERATED_TRANSFER_NO_DFCI",
    file.path(RHINO_ROOT, "federated_transfer_adt_no_DFCI")),
  figure_data = Sys.getenv("COMPASS_FEDERATED_FIGURE_DATA", file.path(RHINO_ROOT, "figure_data")))
RENDER_PROFILE <- match.arg(tolower(Sys.getenv("COMPASS_RENDER_PROFILE", "fast")), c("fast", "publication"))
RENDER_DPI <- if (RENDER_PROFILE == "publication") 600 else 200
RENDER_PDF <- env_flag("COMPASS_RENDER_PDF")
RENDER_OVERWRITE <- env_flag("COMPASS_RENDER_OVERWRITE")
PREPARE_OVERWRITE <- env_flag("COMPASS_PREPARE_OVERWRITE")
RENDER_WORKERS <- env_workers("COMPASS_RENDER_WORKERS", 2L)
PREPARE_WORKERS <- env_workers("COMPASS_PREPARE_WORKERS", 1L)
source(file.path(dirname(pipeline_path), "figure_supplements.R"))
source(file.path(dirname(pipeline_path), "checklist_supplement.R"))

figure_config <- list(
  data_root = NEPC_PROJ_PATH, cache_root = FIGURE_DATA_ROOT, fig_root = FIG_ROOT,
  cohorts = COHORTS, endpoints = ENDPOINTS, scope = FIGURE_SCOPE,
  labs = if (PLOT_NON_ANDROGEN_LAB_FIGURES) names(CATEGORY_MAP) else c(ANDROGEN, SUPPLEMENT_LABS),
  classifier_path = DEFAULT_LLM_ANNOTATIONS_PATH,
  forest_cohorts = COHORT_FOREST_CONFIG$cohorts, forest_landmark = COHORT_FOREST_CONFIG$landmark,
  gam = PLOT_GAM_TRAJECTORIES, adt_intent = PLOT_ADT_INTENT_SUPPLEMENT,
  metastatic = PLOT_METASTATIC_LABEL_SUPPLEMENT, metastatic_extra = metastatic_extra_panels_enabled(),
  metastatic_sources = METASTATIC_SUPPLEMENT[c("intent", "stage", "llm", "icd")],
  federated = PLOT_FEDERATED_SUPPLEMENT,
  federated_path = FEDERATED_SUPPLEMENT$results,
  federated_transfer_no_dfci_path = FEDERATED_SUPPLEMENT$transfer_no_dfci,
  federated_figure_data_path = FEDERATED_SUPPLEMENT$figure_data,
  checklist_supplement = PLOT_CHECKLIST_SUPPLEMENT,
  checklist_labs_run = CHECKLIST_RUNS$labs, checklist_trio_run = CHECKLIST_RUNS$trio,
  checklist_horizon = 1825L)


run_figure <- function(figure_key = NULL) {
  config <- figure_config
  if (!is.null(figure_key)) {
    config$cohorts <- "adt"
    config$endpoints <- "platinum"
    if (startsWith(figure_key, "federated_") ||
        figure_key %in% c("07_federated_incidence_associations", "08_federated_multivariable")) {
      config$scope <- "federated"
    } else if (figure_key != "01_cohort_overview") {
      config$federated <- FALSE
    }
    if (figure_key != "04_multivariable_labs")
      config$checklist_supplement <- FALSE
  }
  figure_run <- run_cached_figure_workflow(
    config, pipeline_path, FIGURE_STAGE,
    prepare_workers = PREPARE_WORKERS, render_workers = RENDER_WORKERS,
    dpi = RENDER_DPI, pdf = RENDER_PDF,
    prepare_overwrite = PREPARE_OVERWRITE, render_overwrite = RENDER_OVERWRITE,
    metastatic_config = METASTATIC_SUPPLEMENT,
    forest_config = COHORT_FOREST_CONFIG,
    federated_config = FEDERATED_SUPPLEMENT,
    figure_key = figure_key)
  invisible(figure_run)
}
