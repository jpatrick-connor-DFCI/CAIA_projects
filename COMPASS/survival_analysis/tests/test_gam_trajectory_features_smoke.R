# test_gam_trajectory_features_smoke.R
#
# Self-checking synthetic smoke test for gam_trajectory_features.R. Plants
# three labs with known trajectory shapes -- one downward slope, one upward
# slope, one flat noise -- across 200 patients with deliberately sparse
# per-patient observation counts (1-8), writes them in the exact
# pre_treatment_lab_long_landmark{D}.csv schema, runs the real script as a
# subprocess, and asserts:
#   (a) one output row per patient (the hierarchical shrinkage keeps even
#       1-observation patients defined -- this is the whole point of bs="fs"),
#   (b) the recovered __gam_slope sign and rough magnitude match what was
#       planted for each lab.
#
# Run with: Rscript COMPASS/survival_analysis/tests/test_gam_trajectory_features_smoke.R
# Exits non-zero on any assertion failure.

suppressPackageStartupMessages({
  library(data.table)
})

get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- cmd_args[grepl("^--file=", cmd_args)]
  if (length(file_arg) == 0) return(getwd())
  dirname(normalizePath(sub("^--file=", "", file_arg)))
}

tests_dir <- get_script_dir()
script_path <- normalizePath(file.path(tests_dir, "..", "gam_trajectory_features.R"))
stopifnot(file.exists(script_path))

set.seed(20250101)

N_PATIENTS <- 200
LANDMARK_DAY <- 180
T_LAB_MIN <- -200
T_LAB_MAX <- LANDMARK_DAY - 1  # strict '<' filter upstream guarantees this

TRUE_SLOPES <- c(LabDown = -0.08, LabUp = 0.05, LabNoise = 0.0)
TRUE_INTERCEPTS <- c(LabDown = 50, LabUp = 20, LabNoise = 30)
PATIENT_INTERCEPT_SD <- 2
RESIDUAL_SD <- 1.5

patients <- sprintf("SIM%04d", seq_len(N_PATIENTS))

rows <- list()
for (lab in names(TRUE_SLOPES)) {
  b_i <- rnorm(N_PATIENTS, mean = 0, sd = PATIENT_INTERCEPT_SD)
  names(b_i) <- patients
  for (p in patients) {
    n_obs <- sample(1:8, 1)
    t_lab <- sort(runif(n_obs, T_LAB_MIN, T_LAB_MAX))
    mu <- TRUE_INTERCEPTS[[lab]] + TRUE_SLOPES[[lab]] * t_lab + b_i[[p]]
    value <- mu + rnorm(n_obs, sd = RESIDUAL_SD)
    rows[[length(rows) + 1]] <- data.table(
      DFCI_MRN = p, LAB_NAME = lab, LAB_VALUE = value, t_lab = t_lab
    )
  }
}
lab_long <- rbindlist(rows)

canonical_labs <- data.table(
  landmark_days = LANDMARK_DAY,
  lab_name = names(TRUE_SLOPES)
)

inputs_dir <- tempfile("gam_smoke_inputs_")
dir.create(inputs_dir)
fwrite(lab_long, file.path(inputs_dir, sprintf("pre_treatment_lab_long_landmark%d.csv", LANDMARK_DAY)))
fwrite(canonical_labs, file.path(inputs_dir, "canonical_labs_train_val.csv"))

cat(sprintf("Synthetic inputs written to %s (%d rows across %d patients x %d labs)\n",
            inputs_dir, nrow(lab_long), N_PATIENTS, length(TRUE_SLOPES)))

result <- system2(
  "Rscript",
  args = c(shQuote(script_path), "--inputs-dir", shQuote(inputs_dir), "--landmark-days", as.character(LANDMARK_DAY)),
  stdout = TRUE, stderr = TRUE
)
cat(paste(result, collapse = "\n"), "\n")
status <- attr(result, "status")
if (!is.null(status) && status != 0) {
  cat(sprintf("FAIL: gam_trajectory_features.R exited with status %s\n", status))
  quit(status = 1)
}

out_path <- file.path(inputs_dir, sprintf("gam_trajectory_features_landmark%d.csv", LANDMARK_DAY))
diag_path <- file.path(inputs_dir, sprintf("gam_fit_diagnostics_landmark%d.csv", LANDMARK_DAY))
curve_path <- file.path(inputs_dir, sprintf("gam_trajectory_curves_landmark%d.csv", LANDMARK_DAY))
stopifnot(file.exists(out_path), file.exists(diag_path), file.exists(curve_path))

features <- fread(out_path)
diagnostics <- fread(diag_path)
curves <- fread(curve_path)

failures <- character(0)
check <- function(cond, msg) {
  if (!isTRUE(cond)) failures[[length(failures) + 1]] <<- msg
}

check(nrow(features) == N_PATIENTS,
      sprintf("expected %d output rows, got %d", N_PATIENTS, nrow(features)))

check(nrow(diagnostics) == length(TRUE_SLOPES),
      sprintf("expected %d diagnostic rows, got %d", length(TRUE_SLOPES), nrow(diagnostics)))
check(all(diagnostics$basis_used %in% c("fs", "re_fallback")),
      sprintf("unexpected basis_used values: %s", paste(unique(diagnostics$basis_used), collapse = ", ")))

expected_curve_columns <- c("DFCI_MRN", "LAB_NAME", "t_lab", "GAM_FITTED")
check(identical(names(curves), expected_curve_columns),
      sprintf("unexpected curve columns: %s", paste(names(curves), collapse = ", ")))
check(nrow(curves) == N_PATIENTS * length(TRUE_SLOPES) * 25L,
      sprintf("expected %d curve rows, got %d",
              N_PATIENTS * length(TRUE_SLOPES) * 25L, nrow(curves)))
check(setequal(unique(curves$LAB_NAME), names(TRUE_SLOPES)),
      sprintf("unexpected curve labs: %s", paste(unique(curves$LAB_NAME), collapse = ", ")))
check(!anyNA(curves$GAM_FITTED), "curve output contains missing GAM_FITTED values")
curve_grid_counts <- curves[, .N, by = .(DFCI_MRN, LAB_NAME)]$N
check(all(curve_grid_counts == 25L), "not every patient/lab curve has the expected 25 grid points")

for (lab in names(TRUE_SLOPES)) {
  slope_col <- paste0(lab, "__gam_slope")
  check(slope_col %in% names(features), sprintf("missing column %s", slope_col))
  if (slope_col %in% names(features)) {
    vals <- features[[slope_col]]
    coverage <- mean(!is.na(vals))
    check(coverage > 0.95, sprintf("%s: coverage only %.2f (expected >0.95, this is the shrinkage guarantee)", slope_col, coverage))
    mean_slope <- mean(vals, na.rm = TRUE)
    cat(sprintf("  %s: mean recovered slope = %.4f (planted = %.4f), coverage = %.2f\n",
                slope_col, mean_slope, TRUE_SLOPES[[lab]], coverage))
    if (TRUE_SLOPES[[lab]] < 0) {
      check(mean_slope < -0.01, sprintf("%s: expected clearly negative mean slope, got %.4f", slope_col, mean_slope))
    } else if (TRUE_SLOPES[[lab]] > 0) {
      check(mean_slope > 0.01, sprintf("%s: expected clearly positive mean slope, got %.4f", slope_col, mean_slope))
    } else {
      check(abs(mean_slope) < 0.02, sprintf("%s: expected near-zero mean slope, got %.4f", slope_col, mean_slope))
    }
  }
}

cat("\nFit timing per lab:\n")
print(diagnostics[, .(lab_name, n_patients, n_obs, basis_used, fit_seconds, converged)])

if (length(failures) > 0) {
  cat("\nFAIL:\n")
  cat(paste(" -", failures, collapse = "\n"), "\n")
  quit(status = 1)
}

cat("\nPASS: all smoke test assertions succeeded.\n")
unlink(inputs_dir, recursive = TRUE)
