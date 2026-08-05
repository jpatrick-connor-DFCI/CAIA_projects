# gam_trajectory_features.R
#
# Fits a hierarchical GAM (population smooth + shrunk per-patient deviation,
# mgcv::bam(..., bs = "fs")) to each canonical lab's pre-landmark trajectory
# and extracts per-patient level/slope/curvature/AUC/deviation features at
# the landmark boundary. These replace the two-point `<LAB>__delta` feature
# (last - first) with a denoised trend that uses every observation and, via
# shrinkage toward the population curve, stays defined for patients with very
# few labs.
#
# Inputs (all written by COMPASS/data_preprocessing/build_prediction_inputs.py):
#   <inputs-dir>/pre_treatment_lab_long_landmark{D}.csv   DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab
#   <inputs-dir>/canonical_labs_train_val.csv             landmark_days, lab_name
#   <inputs-dir>/split_assignments_landmark{D}.csv        DFCI_MRN, split   (only for --fit-split train_val)
#
# Outputs (written back into <inputs-dir>, one pair per requested landmark):
#   gam_trajectory_features_landmark{D}.csv   one row per DFCI_MRN; columns
#                                              <LAB>__gam_{level,slope,curvature,auc,dev}
#   gam_fit_diagnostics_landmark{D}.csv       one row per lab; basis used, EDF,
#                                              fit seconds, convergence flag
#
# Leakage stance: the smooth is unsupervised -- it never sees t_platinum or
# PLATINUM. The default (--fit-split all) fits on every cohort patient in the
# pre-landmark window. This is NOT the same category as the "canonical labs
# are training-block artifacts" invariant elsewhere in this pipeline, which
# governs outcome-adjacent *selection* decisions; nothing here is selected
# using the outcome. Pass --fit-split train_val to refit the population
# smooths on train+valid only, as a leakage sensitivity check -- note the
# result of that comparison in the commit message, it is not run by default.
#
# Feature extraction never evaluates the fitted curve past the landmark
# boundary: all derivative/AUC/deviation grid points use backward
# differences ending at t_star = landmark_offset_days - epsilon, so no
# post-landmark information can leak into a feature even indirectly through
# curve shape.

suppressWarnings({
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required to fit trajectory smooths. Install it before running this script.")
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("data.table is required for fast CSV I/O. Install it before running this script.")
  }
})
suppressPackageStartupMessages({
  library(mgcv)
  library(data.table)
})

# ---------------------------------------------------------------------------
# Defaults (mirrors the DEFAULT_* constant convention used throughout the
# Python side of this pipeline, e.g. COMPASS/survival_analysis/cox_aggregated.py)
# ---------------------------------------------------------------------------
DEFAULT_INPUTS_DIR <- "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/prediction_inputs"
DEFAULT_LANDMARK_DAYS <- "0,90,180"
DEFAULT_K_POP <- 10
DEFAULT_K_PAT <- 5
DEFAULT_TRAILING_WINDOW_DAYS <- 180
DEFAULT_DERIV_STEP_DAYS <- 14
DEFAULT_EPSILON_DAYS <- 0.5
DEFAULT_NTHREADS <- 1
DEFAULT_FIT_TIMEOUT_SECONDS <- 120
DEFAULT_FIT_SPLIT <- "all"
DEFAULT_MIN_OBS_PER_LAB <- 20L
DEFAULT_ID_COL <- "DFCI_MRN"

GAM_STATS <- c("gam_level", "gam_slope", "gam_curvature", "gam_auc", "gam_dev")

# ---------------------------------------------------------------------------
# Minimal CLI parsing (base R only -- no optparse dependency). Every flag is
# `--flag value`; boolean flags (none currently) would be `--flag` alone.
# ---------------------------------------------------------------------------
parse_cli_args <- function(args, defaults) {
  out <- defaults
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument (expected --flag): %s", key))
    }
    name <- gsub("-", "_", sub("^--", "", key))
    if (!name %in% names(defaults)) {
      stop(sprintf("Unknown argument: %s", key))
    }
    if (i == length(args)) {
      stop(sprintf("Flag %s requires a value.", key))
    }
    value <- args[[i + 1L]]
    default_val <- defaults[[name]]
    out[[name]] <- if (is.numeric(default_val)) as.numeric(value) else value
    i <- i + 2L
  }
  out
}

args_list <- parse_cli_args(
  commandArgs(trailingOnly = TRUE),
  list(
    inputs_dir = DEFAULT_INPUTS_DIR,
    landmark_days = DEFAULT_LANDMARK_DAYS,
    k_pop = DEFAULT_K_POP,
    k_pat = DEFAULT_K_PAT,
    trailing_window_days = DEFAULT_TRAILING_WINDOW_DAYS,
    deriv_step_days = DEFAULT_DERIV_STEP_DAYS,
    nthreads = DEFAULT_NTHREADS,
    fit_timeout_seconds = DEFAULT_FIT_TIMEOUT_SECONDS,
    fit_split = DEFAULT_FIT_SPLIT,
    min_obs_per_lab = DEFAULT_MIN_OBS_PER_LAB,
    id_col = DEFAULT_ID_COL
  )
)

landmark_days <- as.integer(strsplit(as.character(args_list$landmark_days), ",")[[1]])
inputs_dir <- args_list$inputs_dir
id_col <- args_list$id_col
k_pop <- as.integer(args_list$k_pop)
k_pat <- as.integer(args_list$k_pat)
trailing_window_days <- as.numeric(args_list$trailing_window_days)
deriv_step_days <- as.numeric(args_list$deriv_step_days)
nthreads <- as.integer(args_list$nthreads)
fit_timeout_seconds <- as.numeric(args_list$fit_timeout_seconds)
fit_split <- as.character(args_list$fit_split)
min_obs_per_lab <- as.integer(args_list$min_obs_per_lab)

if (!fit_split %in% c("all", "train_val")) {
  stop(sprintf("--fit-split must be 'all' or 'train_val', got %s", fit_split))
}

# ---------------------------------------------------------------------------
# Per-lab hierarchical fit, with a timeout-triggered fallback to an additive
# random-intercept + random-slope model when bs = "fs" is too slow at scale
# (mgcv's well-known bottleneck for factor-smooth interactions with many
# levels). setTimeLimit() is base R's standard idiom for this; it polls at
# bytecode boundaries, so it is not a hard real-time guarantee for a single
# long C call inside bam(), but it is adequate for steering a batch script.
# ---------------------------------------------------------------------------
fit_lab_trajectory <- function(dt_lab, k_pop, k_pat, nthreads, timeout_seconds) {
  n_patients <- data.table::uniqueN(dt_lab$MRN_f)
  n_obs <- nrow(dt_lab)
  n_unique_t <- data.table::uniqueN(dt_lab$t_lab)

  k_pop_use <- max(3L, min(k_pop, n_unique_t - 1L))
  k_pat_use <- max(3L, min(k_pat, n_unique_t - 1L))

  fit_fs <- function() {
    bam(
      LAB_VALUE ~ s(t_lab, k = k_pop_use, bs = "tp") +
        s(t_lab, MRN_f, bs = "fs", k = k_pat_use, m = 1),
      data = dt_lab, method = "fREML", discrete = TRUE, nthreads = nthreads
    )
  }
  fit_re <- function() {
    bam(
      LAB_VALUE ~ s(t_lab, k = k_pop_use, bs = "tp") +
        s(MRN_f, bs = "re") + s(MRN_f, t_lab, bs = "re"),
      data = dt_lab, method = "fREML", discrete = TRUE, nthreads = nthreads
    )
  }

  # mgcv warns "repeated 1-d smooths of same variable" for s(t_lab) + s(t_lab,
  # MRN_f, bs="fs") -- expected and harmless for this population+patient
  # hierarchical design (the standard Pedersen et al. HGAM "GS" model form),
  # not a sign of misspecification. Suppressed to keep batch logs readable.
  t0 <- Sys.time()
  setTimeLimit(elapsed = timeout_seconds, transient = TRUE)
  fit <- suppressWarnings(tryCatch(fit_fs(), error = function(e) NULL))
  setTimeLimit(elapsed = Inf, transient = FALSE)
  basis_used <- "fs"

  if (is.null(fit)) {
    t0 <- Sys.time()
    fit <- tryCatch(fit_re(), error = function(e) NULL)
    basis_used <- if (is.null(fit)) "failed" else "re_fallback"
  }
  elapsed_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  converged <- if (is.null(fit)) NA else tryCatch(isTRUE(fit$converged), error = function(e) NA)

  edf_pop <- NA_real_
  edf_patient <- NA_real_
  if (!is.null(fit)) {
    s_table <- tryCatch(summary(fit)$s.table, error = function(e) NULL)
    if (!is.null(s_table)) {
      pop_row <- rownames(s_table)[grepl("^s\\(t_lab\\)$", rownames(s_table))]
      pat_rows <- rownames(s_table)[grepl("MRN_f", rownames(s_table))]
      if (length(pop_row) > 0) edf_pop <- s_table[pop_row[1], "edf"]
      if (length(pat_rows) > 0) edf_patient <- sum(s_table[pat_rows, "edf"])
    }
  }

  list(
    fit = fit, basis_used = basis_used, fit_seconds = elapsed_seconds,
    n_patients = n_patients, n_obs = n_obs, converged = converged,
    edf_pop = edf_pop, edf_patient = edf_patient
  )
}

# ---------------------------------------------------------------------------
# Feature extraction. All evaluation points are <= t_star (backward
# differences), so the fitted curve is never queried past the landmark.
# ---------------------------------------------------------------------------
extract_lab_features <- function(fit_result, patients, landmark_day, deriv_step_days,
                                  trailing_window_days, epsilon_days = DEFAULT_EPSILON_DAYS) {
  fit <- fit_result$fit
  basis_used <- fit_result$basis_used
  t_star <- landmark_day - epsilon_days
  h <- deriv_step_days

  deriv_points <- c(t_star, t_star - h, t_star - 2 * h)
  window_points <- seq(t_star - trailing_window_days, t_star, length.out = 25)
  eval_points <- sort(unique(c(deriv_points, window_points)))

  newdata <- data.table::CJ(MRN_f = patients, t_lab = eval_points, sorted = FALSE)
  fitted_vals <- as.numeric(predict(fit, newdata = newdata, type = "response"))
  term_mat <- predict(fit, newdata = newdata, type = "terms")
  term_cols <- colnames(term_mat)
  dev_cols <- if (basis_used == "fs") {
    term_cols[grepl("MRN_f", term_cols) & grepl("t_lab", term_cols)]
  } else {
    term_cols[grepl("MRN_f", term_cols)]
  }
  dev_vals <- if (length(dev_cols) > 1) rowSums(term_mat[, dev_cols, drop = FALSE]) else as.numeric(term_mat[, dev_cols])

  newdata[, fitted := fitted_vals]
  newdata[, dev := dev_vals]

  at_0 <- newdata[t_lab == t_star, .(MRN_f, f0 = fitted)]
  at_1 <- newdata[t_lab == t_star - h, .(MRN_f, f1 = fitted)]
  at_2 <- newdata[t_lab == t_star - 2 * h, .(MRN_f, f2 = fitted)]

  deriv_dt <- merge(merge(at_0, at_1, by = "MRN_f"), at_2, by = "MRN_f")
  deriv_dt[, `:=`(
    gam_level = f0,
    gam_slope = (f0 - f1) / h,
    gam_curvature = (f0 - 2 * f1 + f2) / h^2
  )]
  level_dt <- deriv_dt[, .(MRN_f, gam_level)]
  slope_dt <- deriv_dt[, .(MRN_f, gam_slope)]
  curv_dt <- deriv_dt[, .(MRN_f, gam_curvature)]

  window_dt <- newdata[t_lab %in% window_points]
  auc_dt <- window_dt[, .(gam_auc = mean(fitted)), by = MRN_f]
  dev_dt <- window_dt[, .(gam_dev = sqrt(mean(dev^2))), by = MRN_f]

  out <- Reduce(function(a, b) merge(a, b, by = "MRN_f", all = TRUE),
                list(level_dt, slope_dt, curv_dt, auc_dt, dev_dt))
  out
}

# ---------------------------------------------------------------------------
# Main per-landmark loop
# ---------------------------------------------------------------------------
canonical_path <- file.path(inputs_dir, "canonical_labs_train_val.csv")
if (!file.exists(canonical_path)) {
  stop(sprintf("Missing %s. Run build_prediction_inputs.py first.", canonical_path))
}
canonical_labs_all <- fread(canonical_path, colClasses = list(character = "lab_name"))

for (landmark_day in landmark_days) {
  cat(sprintf("\n##### GAM TRAJECTORY FEATURES: LANDMARK +%dD #####\n", landmark_day))

  pre_path <- file.path(inputs_dir, sprintf("pre_treatment_lab_long_landmark%d.csv", landmark_day))
  if (!file.exists(pre_path)) {
    stop(sprintf("Missing %s. Run build_prediction_inputs.py first.", pre_path))
  }
  lab_long <- fread(pre_path)
  lab_long[[id_col]] <- as.character(lab_long[[id_col]])

  if (fit_split == "train_val") {
    split_path <- file.path(inputs_dir, sprintf("split_assignments_landmark%d.csv", landmark_day))
    if (!file.exists(split_path)) {
      stop(sprintf("Missing %s (required for --fit-split train_val).", split_path))
    }
    splits <- fread(split_path)
    splits[[id_col]] <- as.character(splits[[id_col]])
    train_val_mrns <- splits[split %in% c("train", "valid")][[id_col]]
    lab_long <- lab_long[get(id_col) %in% train_val_mrns]
    cat(sprintf("  [fit-split=train_val] restricted to %d patients\n", length(unique(train_val_mrns))))
  }

  landmark_labs <- canonical_labs_all[landmark_days == landmark_day]$lab_name
  cat(sprintf("Canonical labs at this landmark: %d\n", length(landmark_labs)))

  feature_frames <- list()
  diagnostic_rows <- list()

  for (lab in landmark_labs) {
    dt_lab <- lab_long[LAB_NAME == lab & is.finite(LAB_VALUE) & is.finite(t_lab)]
    dt_lab <- dt_lab[, .(DFCI_MRN = get(id_col), LAB_VALUE, t_lab)]
    if (nrow(dt_lab) < min_obs_per_lab) {
      cat(sprintf("  [%s] skipped: only %d observations (< %d)\n", lab, nrow(dt_lab), min_obs_per_lab))
      diagnostic_rows[[length(diagnostic_rows) + 1]] <- data.table(
        landmark_days = landmark_day, lab_name = lab, n_patients = data.table::uniqueN(dt_lab$DFCI_MRN),
        n_obs = nrow(dt_lab), basis_used = "skipped_too_few_obs", edf_pop = NA_real_,
        edf_patient = NA_real_, fit_seconds = NA_real_, converged = NA
      )
      next
    }
    dt_lab[, MRN_f := factor(DFCI_MRN)]

    fit_result <- fit_lab_trajectory(dt_lab, k_pop, k_pat, nthreads, fit_timeout_seconds)
    cat(sprintf(
      "  [%s] n_patients=%d n_obs=%d basis=%s fit_seconds=%.1f converged=%s\n",
      lab, fit_result$n_patients, fit_result$n_obs, fit_result$basis_used,
      fit_result$fit_seconds, as.character(fit_result$converged)
    ))
    diagnostic_rows[[length(diagnostic_rows) + 1]] <- data.table(
      landmark_days = landmark_day, lab_name = lab, n_patients = fit_result$n_patients,
      n_obs = fit_result$n_obs, basis_used = fit_result$basis_used, edf_pop = fit_result$edf_pop,
      edf_patient = fit_result$edf_patient, fit_seconds = fit_result$fit_seconds,
      converged = fit_result$converged
    )

    if (is.null(fit_result$fit)) next

    patients <- levels(dt_lab$MRN_f)
    lab_features <- extract_lab_features(fit_result, patients, landmark_day, deriv_step_days, trailing_window_days)
    data.table::setnames(lab_features, GAM_STATS, paste0(lab, "__", GAM_STATS))
    data.table::setnames(lab_features, "MRN_f", id_col)
    lab_features[[id_col]] <- as.character(lab_features[[id_col]])
    feature_frames[[lab]] <- lab_features
  }

  if (length(feature_frames) > 0) {
    combined <- Reduce(function(a, b) merge(a, b, by = id_col, all = TRUE), feature_frames)
  } else {
    combined <- data.table()
    combined[[id_col]] <- character(0)
  }
  out_path <- file.path(inputs_dir, sprintf("gam_trajectory_features_landmark%d.csv", landmark_day))
  fwrite(combined, out_path)
  cat(sprintf("Wrote %s (%d patients x %d feature columns)\n", out_path, nrow(combined), ncol(combined) - 1))

  diag_path <- file.path(inputs_dir, sprintf("gam_fit_diagnostics_landmark%d.csv", landmark_day))
  fwrite(rbindlist(diagnostic_rows), diag_path)
  cat(sprintf("Wrote %s\n", diag_path))
}
