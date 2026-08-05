# test_gam_cox_nonlinearity_smoke.R
#
# Self-checking synthetic smoke test for gam_cox_nonlinearity.R. Simulates a
# Cox survival dataset where one feature ("LabLinear") has a genuinely linear
# log-hazard effect and another ("LabCurve") has a planted quadratic
# (U-shaped) log-hazard effect, writes the exact aggregated_landmark{D}.csv +
# cox_agg_feature_selection.csv schema the real script expects, runs it as a
# subprocess, and asserts:
#   (a) LabCurve is flagged nonlinear: edf clearly > 1, small q_lrt
#   (b) LabLinear is not: edf close to 1
#   (c) LabLinear's coef_linear (R, mgcv::cox.ph) numerically matches
#       coef_feature from Python's run_univariate_nobs_adjusted_associations
#       (survival_common/cox_models.py) run on the SAME csv -- this is the
#       cross-language preprocessing-parity check called for in
#       GAM_SURVIVAL_PLAN.md's verification item 4. Continuous simulated
#       survival times mean ~zero ties, so mgcv's Peto correction and
#       lifelines' Efron approximation should agree closely here.
#
# Run with: Rscript COMPASS/survival_analysis/tests/test_gam_cox_nonlinearity_smoke.R
# The cross-check needs a Python with pandas/sklearn/lifelines importable and
# this repo's survival_common on its path (PYTHONPATH is set automatically
# below). Defaults to `python3` on PATH; override via the GAM_SMOKE_PYTHON
# env var if that's not the right interpreter, e.g.:
#   GAM_SMOKE_PYTHON=/path/to/conda/envs/foo/bin/python Rscript ...
# If Python fails to import required packages, the R-only assertions above
# it still ran and are still meaningful -- only the cross-language coefficient
# parity check is skipped.
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
script_path <- normalizePath(file.path(tests_dir, "..", "gam_cox_nonlinearity.R"))
repo_root <- normalizePath(file.path(tests_dir, "..", "..", ".."))
stopifnot(file.exists(script_path))

set.seed(20250202)

N <- 500
LANDMARK_DAY <- 0
CENSOR_TIME <- 500
H0 <- 0.004
BETA_LINEAR <- 0.7
BETA_CURVE <- 0.6
BETA_AGE <- 0.15

patients <- sprintf("SIM%04d", seq_len(N))
lab_linear_raw <- rnorm(N, 0, 1)
lab_curve_raw <- rnorm(N, 0, 1)
age <- rnorm(N, 65, 8)
age_z <- (age - mean(age)) / sd(age)

lp <- BETA_LINEAR * lab_linear_raw + BETA_CURVE * (lab_curve_raw^2 - 1) + BETA_AGE * age_z
u <- runif(N)
t_event <- -log(u) / (H0 * exp(lp))
duration <- pmin(t_event, CENSOR_TIME)
event <- as.integer(t_event <= CENSOR_TIME)

cat(sprintf("Simulated %d patients, %d events (%.1f%%)\n", N, sum(event), 100 * mean(event)))

aggregated <- data.table(
  DFCI_MRN = patients,
  t_platinum = duration,
  PLATINUM = event,
  AGE_AT_TREATMENTSTART = age,
  LabLinear__gam_slope = lab_linear_raw,
  LabLinear__n_observations = sample(1:10, N, replace = TRUE),
  LabCurve__gam_slope = lab_curve_raw,
  LabCurve__n_observations = sample(1:10, N, replace = TRUE)
)

feature_selection <- data.table(
  landmark_days = LANDMARK_DAY,
  feature = c("LabLinear__gam_slope", "LabCurve__gam_slope"),
  lab_name = c("LabLinear", "LabCurve"),
  feature_stat = "gam_slope",
  coverage = 1.0,
  unique_non_missing = N
)

inputs_dir <- tempfile("gam_cox_smoke_inputs_")
dir.create(inputs_dir)
fwrite(aggregated, file.path(inputs_dir, sprintf("aggregated_landmark%d.csv", LANDMARK_DAY)))
fwrite(feature_selection, file.path(inputs_dir, "cox_agg_feature_selection.csv"))

output_dir <- tempfile("gam_cox_smoke_output_")
dir.create(output_dir)

result <- system2(
  "Rscript",
  args = c(
    shQuote(script_path),
    "--inputs-dir", shQuote(inputs_dir),
    "--output-dir", shQuote(output_dir),
    "--landmark-days", as.character(LANDMARK_DAY)
  ),
  stdout = TRUE, stderr = TRUE
)
cat(paste(result, collapse = "\n"), "\n")
status <- attr(result, "status")
if (!is.null(status) && status != 0) {
  cat(sprintf("FAIL: gam_cox_nonlinearity.R exited with status %s\n", status))
  quit(status = 1)
}

out_path <- file.path(output_dir, sprintf("gam_cox_nonlinearity_landmark%d.csv", LANDMARK_DAY))
stopifnot(file.exists(out_path))
r_results <- fread(out_path)
print(r_results[, .(feature, n_used, n_events, edf, p_smooth, p_lrt, q_lrt, delta_aic, coef_linear, p_linear)])

failures <- character(0)
check <- function(cond, msg) {
  if (!isTRUE(cond)) failures[[length(failures) + 1]] <<- msg
}

row_linear <- r_results[feature == "LabLinear__gam_slope"]
row_curve <- r_results[feature == "LabCurve__gam_slope"]
check(nrow(row_linear) == 1, "missing LabLinear__gam_slope row in R output")
check(nrow(row_curve) == 1, "missing LabCurve__gam_slope row in R output")

if (nrow(row_linear) == 1) {
  check(row_linear$edf < 2.5, sprintf("LabLinear: expected edf < 2.5 (near-linear), got %.2f", row_linear$edf))
  check(row_linear$coef_linear > 0.3, sprintf("LabLinear: expected coef_linear > 0.3 (planted %.2f), got %.3f", BETA_LINEAR, row_linear$coef_linear))
}
if (nrow(row_curve) == 1) {
  check(row_curve$edf > 1.5, sprintf("LabCurve: expected edf > 1.5 (planted quadratic), got %.2f", row_curve$edf))
  check(!is.na(row_curve$q_lrt) && row_curve$q_lrt < 0.05,
        sprintf("LabCurve: expected q_lrt < 0.05, got %s", row_curve$q_lrt))
}

# --- Cross-check against Python's run_univariate_nobs_adjusted_associations ---
py_script <- file.path(inputs_dir, "run_python_univariate_check.py")
writeLines(c(
  "import sys",
  "import pandas as pd",
  "from survival_common.cox_models import run_univariate_nobs_adjusted_associations",
  "",
  sprintf("inputs_dir = %s", shQuote(inputs_dir, type = "sh")),
  sprintf("landmark_day = %d", LANDMARK_DAY),
  "df = pd.read_csv(f'{inputs_dir}/aggregated_landmark{landmark_day}.csv')",
  "endpoint_map = {'platinum': {'duration_col': 't_platinum', 'event_col': 'PLATINUM'}}",
  "result = run_univariate_nobs_adjusted_associations(",
  "    df,",
  "    feature_cols=['LabLinear__gam_slope', 'LabCurve__gam_slope'],",
  "    endpoint='platinum',",
  "    min_events_per_feature=10,",
  "    fallback_penalizer=0.05,",
  "    endpoint_map=endpoint_map,",
  "    age_col='AGE_AT_TREATMENTSTART',",
  ")",
  "result[['feature', 'coef_feature', 'p_value', 'fit_penalizer', 'note']].to_csv(",
  "    f'{inputs_dir}/python_univariate_check.csv', index=False",
  ")"
), py_script)

python_bin <- Sys.getenv("GAM_SMOKE_PYTHON", unset = "python3")

# system2's `env=` argument does not shell-quote values, so it breaks
# silently (or errors outright) when repo_root contains spaces -- as this
# repo's path does. Sys.setenv() sets the child process environment directly
# with no shell parsing involved, so it is safe regardless of spaces.
old_pythonpath <- Sys.getenv("PYTHONPATH", unset = NA)
Sys.setenv(PYTHONPATH = repo_root)
py_result <- system2(python_bin, args = shQuote(py_script), stdout = TRUE, stderr = TRUE)
if (is.na(old_pythonpath)) Sys.unsetenv("PYTHONPATH") else Sys.setenv(PYTHONPATH = old_pythonpath)
cat(paste(py_result, collapse = "\n"), "\n")
py_status <- attr(py_result, "status")
if (!is.null(py_status) && py_status != 0) {
  cat(sprintf(
    "\nSKIPPED cross-language coefficient check: '%s' exited with status %s (likely a broken/mismatched Python env, not a bug in gam_cox_nonlinearity.R).\n",
    python_bin, py_status
  ))
  cat(sprintf("Set GAM_SMOKE_PYTHON to a working interpreter to enable this check, e.g.:\n  GAM_SMOKE_PYTHON=/path/to/env/bin/python Rscript %s\n", script_path))
  py_result <- NULL
}

py_csv_path <- file.path(inputs_dir, "python_univariate_check.csv")
py_results <- if (!is.null(py_result) && file.exists(py_csv_path)) fread(py_csv_path) else NULL

if (is.null(py_results)) {
  cat("\n(Cross-language coefficient check skipped -- see message above.)\n")
} else {
  print(py_results)
  py_linear <- py_results[feature == "LabLinear__gam_slope"]
  check(nrow(py_linear) == 1, "missing LabLinear__gam_slope row in Python output")
  if (nrow(py_linear) == 1 && nrow(row_linear) == 1) {
    r_coef <- row_linear$coef_linear
    py_coef <- py_linear$coef_feature
    rel_diff <- abs(r_coef - py_coef) / abs(py_coef)
    cat(sprintf("\nLabLinear coef cross-check: R (mgcv cox.ph) = %.4f, Python (lifelines) = %.4f, rel_diff = %.3f\n",
                r_coef, py_coef, rel_diff))
    check(rel_diff < 0.15, sprintf("R vs Python coef_linear mismatch too large: %.4f vs %.4f (rel_diff %.3f)", r_coef, py_coef, rel_diff))
  }
}

# --- Worker-identity check: --n-workers 2 must produce output identical to
# --n-workers 1, including q_lrt (BH adjustment is order-dependent, so this
# specifically validates that mclapply's order-preserving guarantee holds). ---
output_dirs <- list(w1 = tempfile("gam_cox_smoke_out_w1_"), w2 = tempfile("gam_cox_smoke_out_w2_"))
for (d in output_dirs) dir.create(d)

run_with_workers <- function(out_dir, n_workers) {
  res <- system2(
    "Rscript",
    args = c(
      shQuote(script_path),
      "--inputs-dir", shQuote(inputs_dir),
      "--output-dir", shQuote(out_dir),
      "--landmark-days", as.character(LANDMARK_DAY),
      "--n-workers", as.character(n_workers)
    ),
    stdout = TRUE, stderr = TRUE
  )
  st <- attr(res, "status")
  if (!is.null(st) && st != 0) {
    cat(paste(res, collapse = "\n"), "\n")
    cat(sprintf("FAIL: gam_cox_nonlinearity.R (--n-workers %d) exited with status %s\n", n_workers, st))
    quit(status = 1)
  }
  res
}

run_with_workers(output_dirs$w1, 1)
run_with_workers(output_dirs$w2, 2)

w1_path <- file.path(output_dirs$w1, sprintf("gam_cox_nonlinearity_landmark%d.csv", LANDMARK_DAY))
w2_path <- file.path(output_dirs$w2, sprintf("gam_cox_nonlinearity_landmark%d.csv", LANDMARK_DAY))
w1_out <- fread(w1_path)
w2_out <- fread(w2_path)
eq <- isTRUE(all.equal(w1_out, w2_out))
check(eq, "--n-workers 1 vs 2 output mismatch in gam_cox_nonlinearity_landmark0.csv")
cat(sprintf("  worker-identity check (gam_cox_nonlinearity_landmark%d.csv): %s\n",
            LANDMARK_DAY, if (eq) "identical" else "MISMATCH"))
unlink(output_dirs$w1, recursive = TRUE)
unlink(output_dirs$w2, recursive = TRUE)

if (length(failures) > 0) {
  cat("\nFAIL:\n")
  cat(paste(" -", failures, collapse = "\n"), "\n")
  quit(status = 1)
}

cat("\nPASS: all smoke test assertions succeeded.\n")
unlink(inputs_dir, recursive = TRUE)
unlink(output_dir, recursive = TRUE)
