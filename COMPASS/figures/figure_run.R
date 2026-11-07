# Shared run logic for the COMPASS manuscript figures (ADT cohort, platinum endpoint).
# Sourced by the two entry scripts in COMPASS/notebooks/:
#   final_figure_data_prep.R  stage "prepare": builds the cached figure data R needs
#   run_figure_render.R       stage "render":  draws PDFs and writes figure data tables
# Both expect `script_dir` (this folder) and `FIGURE_STAGE` to be set before sourcing.
# Run 05_prep_figure_data.ipynb first. Flip the toggles below to run a subset of
# figure groups. Only manuscript figures are rendered.
#
# Paths and render settings can be overridden with the environment variables
# read in the configuration section. COMPASS_FIGURE_GROUPS (comma-separated
# group names) overrides the toggles, e.g. for cluster jobs:
#
#   COMPASS_FIGURE_GROUPS=transfer_survival,transfer_checklists Rscript COMPASS/notebooks/run_figure_render.R

# ---- Figure groups ----------------------------------------------------------
# Prepare and render only the manuscript figures.
MANUSCRIPT_ONLY <- TRUE
# Groups select manuscript figures under <COMPASS_FIG_ROOT>/ADT_figures/main or supplement;
# tables, legends, captions, manifest and index go to <COMPASS_FIG_ROOT>/ADT_figure_data/.
FIGURES <- c(
  cohort                  = TRUE,  # 01_cohort: Fig 1 (consort, KM, spans, DFCI data, site counts)
  llm                     = TRUE,  # 02_llm: Fig 2 LLM subtype validation, day-0 cohort
  cohort_sensitivity      = TRUE,  # 03_univariate/cohort_sensitivity: Fig 3 supplement, metastatic labels
  federated_univariable   = TRUE,  # 03_univariate/federated: federated Fig 3, forests, KMs, trajectories
  federated_multivariable = TRUE,  # 04_multivariate/federated: federated Fig 4, performance, checklist cards
  transfer_survival       = TRUE,  # 05_transfer_survival: federated Fig 5 (no-DFCI transfer)
  transfer_checklists     = TRUE)  # 06_transfer_checklists: federated Fig 6 (transfer checklists)

# ---- Configuration ----------------------------------------------------------
options(readr.num_threads = 1L, readr.read_lazy = FALSE)

if (!exists("script_dir")) script_dir <- file.path(Sys.getenv("CAIA_PROJECT_ROOT",
  "/data/gusev/USERS/jpconnor/code/CAIA"), "COMPASS", "figures")
pipeline_path <- normalizePath(file.path(script_dir, "COMPASS_generate_figures_pipeline.R"), mustWork = TRUE)
source(pipeline_path)
source(file.path(script_dir, "figure_workflow.R"))
source(file.path(script_dir, "figure_supplements.R"))
source(file.path(script_dir, "checklist_supplement.R"))

env_flag <- function(name, default = FALSE) {
  value <- tolower(Sys.getenv(name, if (default) "true" else "false"))
  if (!value %in% c("true", "false", "1", "0", "yes", "no")) stop(name, " must be true or false")
  value %in% c("true", "1", "yes")
}
env_workers <- function(name, default) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, as.character(default))))
  if (is.na(value) || value < 1L) stop(name, " must be a positive integer")
  value
}

NEPC_PROJ_PATH <- Sys.getenv("COMPASS_DATA_ROOT", "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS")
FIG_ROOT <- Sys.getenv("COMPASS_FIG_ROOT", "/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS")
FIGURE_DATA_ROOT <- Sys.getenv("COMPASS_FIGURE_DATA_ROOT", file.path(NEPC_PROJ_PATH, "figure_data"))
FIGURE_STAGE <- match.arg(get0("FIGURE_STAGE", ifnotfound = "all"), c("all", "prepare", "render"))

# Cohorts compared in the Figure 3 cohort-sensitivity forest. Must match the 05 notebook.
ADT_COHORT_OPTIONS <- c("adt", "adt_noprecastrate", "adt_metastatic_adt",
  "adt_metastatic_adt_noprecastrate", "adt_metastatic_llm", "adt_metastatic_llm_noprecastrate")
COHORT_FOREST_CONFIG <- list(script = file.path(script_dir, "figure_supplements.R"),
                             cohorts = ADT_COHORT_OPTIONS, landmark = 180L)
METASTATIC_SUPPLEMENT <- list(
  prepare_script = file.path(script_dir, "figure_supplements.R"),
  r_script = file.path(script_dir, "figure_supplements.R"),
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
RENDER_OVERWRITE <- env_flag("COMPASS_RENDER_OVERWRITE")
PREPARE_OVERWRITE <- env_flag("COMPASS_PREPARE_OVERWRITE")
RENDER_WORKERS <- env_workers("COMPASS_RENDER_WORKERS", 2L)
PREPARE_WORKERS <- env_workers("COMPASS_PREPARE_WORKERS", 4L)
MANUSCRIPT_ONLY <- TRUE

# ---- Run --------------------------------------------------------------------
stopifnot(setequal(names(FIGURES), names(FIGURE_GROUPS)))
env_groups <- trimws(strsplit(Sys.getenv("COMPASS_FIGURE_GROUPS", ""), ",", fixed = TRUE)[[1]])
env_groups <- env_groups[nzchar(env_groups)]
if (length(env_groups)) {
  unknown <- setdiff(env_groups, names(FIGURES))
  if (length(unknown)) stop("Unknown COMPASS_FIGURE_GROUPS: ", paste(unknown, collapse = ", "))
  FIGURES[] <- names(FIGURES) %in% env_groups
}
groups <- names(FIGURES)[FIGURES]
if (!length(groups)) stop("Enable at least one figure group in FIGURES.")
# Groups drawn only from federated exports skip the DFCI patient-level checks.
federated_only <- all(vapply(FIGURE_GROUPS[groups], function(g) identical(g$jobs, "federated"), logical(1)))

figure_config <- list(
  data_root = NEPC_PROJ_PATH, cache_root = FIGURE_DATA_ROOT, fig_root = FIG_ROOT,
  cohorts = "adt", endpoints = "platinum", scope = if (federated_only) "federated" else "all",
  classifier_path = DEFAULT_LLM_ANNOTATIONS_PATH,
  forest_cohorts = COHORT_FOREST_CONFIG$cohorts, forest_landmark = COHORT_FOREST_CONFIG$landmark,
  metastatic = TRUE,
  metastatic_sources = METASTATIC_SUPPLEMENT[c("intent", "stage", "llm", "icd")],
  federated = TRUE,
  federated_path = FEDERATED_SUPPLEMENT$results,
  federated_transfer_no_dfci_path = FEDERATED_SUPPLEMENT$transfer_no_dfci,
  federated_figure_data_path = FEDERATED_SUPPLEMENT$figure_data)

message("Figure groups: ", paste(groups, collapse = ", "))
message("Manuscript-only figures: ", MANUSCRIPT_ONLY)
figure_run <- run_cached_figure_workflow(
  figure_config, pipeline_path, FIGURE_STAGE,
  prepare_workers = PREPARE_WORKERS, render_workers = RENDER_WORKERS,
  dpi = RENDER_DPI, pdf = FALSE,
  prepare_overwrite = PREPARE_OVERWRITE, render_overwrite = RENDER_OVERWRITE,
  metastatic_config = METASTATIC_SUPPLEMENT,
  forest_config = COHORT_FOREST_CONFIG,
  federated_config = FEDERATED_SUPPLEMENT,
  groups = groups, manuscript_only = MANUSCRIPT_ONLY)
