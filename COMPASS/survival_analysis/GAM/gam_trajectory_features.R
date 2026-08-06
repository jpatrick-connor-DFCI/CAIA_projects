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
# Cohort inputs (all written by COMPASS/data_preprocessing/build_prediction_inputs.py):
#   <inputs-dir>/pre_treatment_lab_long_landmark{D}.csv   DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab
#   <inputs-dir>/canonical_labs_train_val.csv             landmark_days, lab_name
#   <inputs-dir>/split_assignments_landmark{D}.csv        DFCI_MRN, split   (only for --fit-split train_val)
#
# Outputs (written into <output-dir>, separate from the main pipeline):
#   gam_trajectory_features_landmark{D}.csv   one row per DFCI_MRN; columns
#                                              <LAB>__gam_{level,slope,curvature,auc,dev}
#   gam_fit_diagnostics_landmark{D}.csv       one row per lab; basis used, EDF,
#                                              fit seconds, convergence flag
#   gam_trajectory_curves_landmark{D}.csv     fitted per-patient curves on the
#                                              trailing-window grid, used only
#                                              for GAM-specific figures
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
  library(parallel)
})

# ---------------------------------------------------------------------------
# Defaults (mirrors the DEFAULT_* constant convention used throughout the
# Python side of this pipeline, e.g. COMPASS/survival_analysis/cox_aggregated.py)
# ---------------------------------------------------------------------------
DEFAULT_INPUTS_DIR <- "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/prediction_inputs"
DEFAULT_OUTPUT_DIR <- "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/GAM"
DEFAULT_LANDMARK_DAYS <- "0,90,180"
DEFAULT_K_POP <- 10
DEFAULT_K_PAT <- 5
DEFAULT_TRAILING_WINDOW_DAYS <- 180
DEFAULT_DERIV_STEP_DAYS <- 14
DEFAULT_EPSILON_DAYS <- 0.5
DEFAULT_NTHREADS <- 1
DEFAULT_FIT_TIMEOUT_SECONDS <- 120
DEFAULT_MAX_FS_PATIENTS <- 500L
DEFAULT_PATIENT_RIDGE_LAMBDA <- 5
DEFAULT_FIT_SPLIT <- "all"
DEFAULT_MIN_OBS_PER_LAB <- 20L
DEFAULT_ID_COL <- "DFCI_MRN"
DEFAULT_N_WORKERS <- 1L
DEFAULT_WRITE_CURVES <- "true"
DEFAULT_CURVE_GRID_POINTS <- 25L
DEFAULT_CURVE_LABS <- ""
DEFAULT_AUC_GRID_POINTS <- 25L
DEFAULT_RESUME <- "false"

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
    output_dir = DEFAULT_OUTPUT_DIR,
    landmark_days = DEFAULT_LANDMARK_DAYS,
    k_pop = DEFAULT_K_POP,
    k_pat = DEFAULT_K_PAT,
    trailing_window_days = DEFAULT_TRAILING_WINDOW_DAYS,
    deriv_step_days = DEFAULT_DERIV_STEP_DAYS,
    nthreads = DEFAULT_NTHREADS,
    fit_timeout_seconds = DEFAULT_FIT_TIMEOUT_SECONDS,
    max_fs_patients = DEFAULT_MAX_FS_PATIENTS,
    patient_ridge_lambda = DEFAULT_PATIENT_RIDGE_LAMBDA,
    fit_split = DEFAULT_FIT_SPLIT,
    min_obs_per_lab = DEFAULT_MIN_OBS_PER_LAB,
    id_col = DEFAULT_ID_COL,
    n_workers = DEFAULT_N_WORKERS,
    write_curves = DEFAULT_WRITE_CURVES,
    curve_grid_points = DEFAULT_CURVE_GRID_POINTS,
    curve_labs = DEFAULT_CURVE_LABS,
    auc_grid_points = DEFAULT_AUC_GRID_POINTS,
    resume = DEFAULT_RESUME
  )
)

landmark_days <- as.integer(strsplit(as.character(args_list$landmark_days), ",")[[1]])
inputs_dir <- args_list$inputs_dir
output_dir <- args_list$output_dir
id_col <- args_list$id_col
k_pop <- as.integer(args_list$k_pop)
k_pat <- as.integer(args_list$k_pat)
trailing_window_days <- as.numeric(args_list$trailing_window_days)
deriv_step_days <- as.numeric(args_list$deriv_step_days)
nthreads <- as.integer(args_list$nthreads)
fit_timeout_seconds <- as.numeric(args_list$fit_timeout_seconds)
max_fs_patients <- as.integer(args_list$max_fs_patients)
patient_ridge_lambda <- as.numeric(args_list$patient_ridge_lambda)
fit_split <- as.character(args_list$fit_split)
min_obs_per_lab <- as.integer(args_list$min_obs_per_lab)
n_workers <- as.integer(args_list$n_workers)
write_curves <- as.character(args_list$write_curves)
curve_grid_points <- as.integer(args_list$curve_grid_points)
curve_labs <- as.character(args_list$curve_labs)
auc_grid_points <- as.integer(args_list$auc_grid_points)
resume <- as.character(args_list$resume)

if (!fit_split %in% c("all", "train_val")) {
  stop(sprintf("--fit-split must be 'all' or 'train_val', got %s", fit_split))
}
if (max_fs_patients < 0L) stop("--max-fs-patients must be at least 0 (0 = always two_stage_ridge)")
if (!is.finite(patient_ridge_lambda) || patient_ridge_lambda <= 0) {
  stop("--patient-ridge-lambda must be positive")
}
if (n_workers < 1L) stop("--n-workers must be at least 1")
if (!write_curves %in% c("true", "false")) {
  stop(sprintf("--write-curves must be 'true' or 'false', got %s", write_curves))
}
write_curves <- identical(write_curves, "true")
if (curve_grid_points < 2L) stop("--curve-grid-points must be at least 2")
if (auc_grid_points < 2L) stop("--auc-grid-points must be at least 2")
if (!resume %in% c("true", "false")) {
  stop(sprintf("--resume must be 'true' or 'false', got %s", resume))
}
resume <- identical(resume, "true")
curve_lab_filter <- if (nzchar(curve_labs)) {
  trimws(strsplit(curve_labs, ",")[[1]])
} else {
  NULL
}
detected_cores <- parallel::detectCores()
if (!is.na(detected_cores) && n_workers * nthreads > detected_cores) {
  warning(sprintf(
    "--n-workers %d x --nthreads %d = %d exceeds %d detected cores (oversubscription risk)",
    n_workers, nthreads, n_workers * nthreads, detected_cores
  ))
}

# ---------------------------------------------------------------------------
# Per-lab hierarchical fit, with a timeout-triggered fallback to an additive
# random-intercept + random-slope model when bs = "fs" is too slow at scale
# (mgcv's well-known bottleneck for factor-smooth interactions with many
# levels). setTimeLimit() is base R's standard idiom for this; it polls at
# bytecode boundaries, so it is not a hard real-time guarantee for a single
# long C call inside bam(), but it is adequate for steering a batch script.
# ---------------------------------------------------------------------------
fit_lab_trajectory <- function(dt_lab, k_pop, k_pat, nthreads, timeout_seconds,
                               max_fs_patients, patient_ridge_lambda) {
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

  # The fs/re designs create O(number of patients) coefficients inside one
  # dense mgcv fit and can be killed by the OS before R can raise an error.
  # Above max_fs_patients, use the equivalent scalable decomposition: one
  # population GAM plus independently ridge-shrunk patient intercept/slope
  # deviations. Memory then grows linearly with observations, not quadratically
  # with the number of patient coefficients.
  fit_two_stage <- function() {
    population_fit <- bam(
      LAB_VALUE ~ s(t_lab, k = k_pop_use, bs = "tp"),
      data = dt_lab, method = "fREML", discrete = TRUE, nthreads = nthreads
    )
    t_center <- mean(dt_lab$t_lab)
    t_scale <- stats::sd(dt_lab$t_lab)
    if (!is.finite(t_scale) || t_scale <= 0) t_scale <- 1
    residual_dt <- data.table::copy(dt_lab)
    residual_dt[, `:=`(
      t_scaled = (t_lab - t_center) / t_scale,
      pop_residual = LAB_VALUE - as.numeric(predict(population_fit, newdata = dt_lab))
    )]
    patient_effects <- residual_dt[, {
      x <- cbind(1, t_scaled)
      penalty <- diag(patient_ridge_lambda, 2L)
      beta <- solve(crossprod(x) + penalty, crossprod(x, pop_residual))
      .(patient_intercept = beta[[1]], patient_slope = beta[[2]])
    }, by = MRN_f]
    list(
      fit = population_fit,
      patient_effects = patient_effects,
      t_center = t_center,
      t_scale = t_scale
    )
  }

  if (n_patients > max_fs_patients) {
    t0 <- Sys.time()
    scalable <- tryCatch(fit_two_stage(), error = function(e) NULL)
    elapsed_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    fit <- if (is.null(scalable)) NULL else scalable$fit
    return(list(
      fit = fit,
      basis_used = if (is.null(fit)) "failed" else "two_stage_ridge",
      fit_seconds = elapsed_seconds,
      n_patients = n_patients,
      n_obs = n_obs,
      converged = if (is.null(fit)) NA else isTRUE(fit$converged),
      edf_pop = if (is.null(fit)) NA_real_ else sum(fit$edf),
      edf_patient = NA_real_,
      patient_effects = if (is.null(scalable)) NULL else scalable$patient_effects,
      t_center = if (is.null(scalable)) NA_real_ else scalable$t_center,
      t_scale = if (is.null(scalable)) NA_real_ else scalable$t_scale
    ))
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
                                  trailing_window_days, epsilon_days = DEFAULT_EPSILON_DAYS,
                                  auc_grid_points = 25L, curve_grid_points = 25L,
                                  write_curves = TRUE) {
  fit <- fit_result$fit
  basis_used <- fit_result$basis_used
  t_star <- landmark_day - epsilon_days
  h <- deriv_step_days

  deriv_points <- c(t_star, t_star - h, t_star - 2 * h)
  auc_points <- seq(t_star - trailing_window_days, t_star, length.out = auc_grid_points)
  curve_points <- if (write_curves) {
    seq(t_star - trailing_window_days, t_star, length.out = curve_grid_points)
  } else {
    numeric(0)
  }
  eval_points <- sort(unique(c(deriv_points, auc_points, curve_points)))
  stopifnot(max(eval_points) <= t_star)

  newdata <- data.table::CJ(MRN_f = patients, t_lab = eval_points, sorted = FALSE)
  if (basis_used == "two_stage_ridge") {
    newdata <- merge(newdata, fit_result$patient_effects, by = "MRN_f", all.x = TRUE)
    newdata[is.na(patient_intercept), patient_intercept := 0]
    newdata[is.na(patient_slope), patient_slope := 0]
    pop_fitted <- as.numeric(predict(fit, newdata = newdata, type = "response"))
    dev_vals <- newdata$patient_intercept + newdata$patient_slope *
      ((newdata$t_lab - fit_result$t_center) / fit_result$t_scale)
    fitted_vals <- pop_fitted + dev_vals
  } else {
    fitted_vals <- as.numeric(predict(fit, newdata = newdata, type = "response"))
    term_mat <- predict(fit, newdata = newdata, type = "terms")
    term_cols <- colnames(term_mat)
    dev_cols <- if (basis_used == "fs") {
      term_cols[grepl("MRN_f", term_cols) & grepl("t_lab", term_cols)]
    } else {
      term_cols[grepl("MRN_f", term_cols)]
    }
    dev_vals <- if (length(dev_cols) > 1) {
      rowSums(term_mat[, dev_cols, drop = FALSE])
    } else {
      as.numeric(term_mat[, dev_cols])
    }
  }

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

  auc_window_dt <- newdata[t_lab %in% auc_points]
  auc_dt <- auc_window_dt[, .(gam_auc = mean(fitted)), by = MRN_f]
  dev_dt <- auc_window_dt[, .(gam_dev = sqrt(mean(dev^2))), by = MRN_f]

  features <- Reduce(function(a, b) merge(a, b, by = "MRN_f", all = TRUE),
                     list(level_dt, slope_dt, curv_dt, auc_dt, dev_dt))
  curves <- if (write_curves) {
    newdata[t_lab %in% curve_points, .(MRN_f, t_lab, GAM_FITTED = fitted)]
  } else {
    NULL
  }
  list(features = features, curves = curves)
}

# ---------------------------------------------------------------------------
# Per-lab worker body. Returns a single list instead of mutating shared
# accumulators, so it can be dispatched via mclapply() with mclapply's own
# input-order-preserving semantics doing the reassembly for free. `log_lines`
# is returned rather than cat()'d directly -- under parallel dispatch,
# printing immediately would interleave unreadably across workers, so the
# parent cat()s each lab's buffered lines after collection, keeping the batch
# log deterministic regardless of --n-workers.
# ---------------------------------------------------------------------------
process_one_lab <- function(lab, dt_lab, landmark_day, k_pop, k_pat, nthreads,
                            fit_timeout_seconds, max_fs_patients, patient_ridge_lambda,
                            min_obs_per_lab, deriv_step_days, trailing_window_days,
                            auc_grid_points, curve_grid_points, write_this_labs_curves,
                            id_col) {
  log_lines <- character(0)
  log <- function(...) log_lines[[length(log_lines) + 1]] <<- sprintf(...)

  if (nthreads == 1L) Sys.setenv(OMP_NUM_THREADS = "1")

  if (nrow(dt_lab) < min_obs_per_lab) {
    log("  [%s] skipped: only %d observations (< %d)", lab, nrow(dt_lab), min_obs_per_lab)
    diagnostics <- data.table(
      landmark_days = landmark_day, lab_name = lab, n_patients = data.table::uniqueN(dt_lab$DFCI_MRN),
      n_obs = nrow(dt_lab), basis_used = "skipped_too_few_obs", edf_pop = NA_real_,
      edf_patient = NA_real_, fit_seconds = NA_real_, converged = NA
    )
    return(list(features = NULL, curves = NULL, diagnostics = diagnostics, log_lines = log_lines))
  }
  dt_lab[, MRN_f := factor(DFCI_MRN)]

  log(
    "  [%s] starting fit: n_patients=%d n_obs=%d path=%s",
    lab, uniqueN(dt_lab$MRN_f), nrow(dt_lab),
    if (uniqueN(dt_lab$MRN_f) > max_fs_patients) "two_stage_ridge" else "factor_smooth"
  )
  fit_result <- fit_lab_trajectory(
    dt_lab, k_pop, k_pat, nthreads, fit_timeout_seconds,
    max_fs_patients, patient_ridge_lambda
  )
  log(
    "  [%s] n_patients=%d n_obs=%d basis=%s fit_seconds=%.1f converged=%s",
    lab, fit_result$n_patients, fit_result$n_obs, fit_result$basis_used,
    fit_result$fit_seconds, as.character(fit_result$converged)
  )
  diagnostics <- data.table(
    landmark_days = landmark_day, lab_name = lab, n_patients = fit_result$n_patients,
    n_obs = fit_result$n_obs, basis_used = fit_result$basis_used, edf_pop = fit_result$edf_pop,
    edf_patient = fit_result$edf_patient, fit_seconds = fit_result$fit_seconds,
    converged = fit_result$converged
  )

  if (is.null(fit_result$fit)) {
    return(list(features = NULL, curves = NULL, diagnostics = diagnostics, log_lines = log_lines))
  }

  patients <- levels(dt_lab$MRN_f)
  extracted <- extract_lab_features(
    fit_result, patients, landmark_day, deriv_step_days, trailing_window_days,
    auc_grid_points = auc_grid_points, curve_grid_points = curve_grid_points,
    write_curves = write_this_labs_curves
  )
  lab_features <- extracted$features
  data.table::setnames(lab_features, GAM_STATS, paste0(lab, "__", GAM_STATS))
  data.table::setnames(lab_features, "MRN_f", id_col)
  data.table::set(lab_features, j = id_col, value = as.character(lab_features[[id_col]]))

  lab_curves <- if (is.null(extracted$curves)) {
    NULL
  } else {
    lc <- extracted$curves
    data.table::setnames(lc, "MRN_f", id_col)
    data.table::set(lc, j = id_col, value = as.character(lc[[id_col]]))
    data.table::set(lc, j = "LAB_NAME", value = lab)
    data.table::setcolorder(lc, c(id_col, "LAB_NAME", "t_lab", "GAM_FITTED"))
    lc
  }

  list(features = lab_features, curves = lab_curves, diagnostics = diagnostics, log_lines = log_lines)
}

# ---------------------------------------------------------------------------
# Main per-landmark loop
# ---------------------------------------------------------------------------
canonical_path <- file.path(inputs_dir, "canonical_labs_train_val.csv")
if (!file.exists(canonical_path)) {
  stop(sprintf("Missing %s. Run build_prediction_inputs.py first.", canonical_path))
}
canonical_labs_all <- fread(canonical_path, colClasses = list(character = "lab_name"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

for (landmark_day in landmark_days) {
  cat(sprintf("\n##### GAM TRAJECTORY FEATURES: LANDMARK +%dD #####\n", landmark_day))
  flush.console()

  out_path <- file.path(output_dir, sprintf("gam_trajectory_features_landmark%d.csv", landmark_day))
  diag_path <- file.path(output_dir, sprintf("gam_fit_diagnostics_landmark%d.csv", landmark_day))
  curve_path <- file.path(output_dir, sprintf("gam_trajectory_curves_landmark%d.csv", landmark_day))
  if (resume && file.exists(out_path)) {
    cat(sprintf("  [resume] %s already exists -- skipping landmark %d\n", out_path, landmark_day))
    next
  }

  pre_path <- file.path(inputs_dir, sprintf("pre_treatment_lab_long_landmark%d.csv", landmark_day))
  if (!file.exists(pre_path)) {
    stop(sprintf("Missing %s. Run build_prediction_inputs.py first.", pre_path))
  }
  # Read only the columns the fit actually uses; the long table may carry
  # extra columns that are irrelevant here (data.table::fread(select=) skips
  # parsing them entirely rather than reading then dropping).
  lab_long <- fread(pre_path, select = c(id_col, "LAB_NAME", "LAB_VALUE", "t_lab"))
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

  # Apply the finite-value filter once, up front, then split by LAB_NAME so
  # each worker only ever touches its own disjoint chunk -- this both avoids
  # an unindexed full-table scan per lab (previously `lab_long[LAB_NAME ==
  # lab & ...]` inside the loop) and keeps forked workers from materializing
  # pages of the full table. Labs with zero surviving rows are absent from
  # split()'s output, so lookup-with-default below is required to keep the
  # skipped_too_few_obs diagnostic row for those labs.
  lab_long <- lab_long[is.finite(LAB_VALUE) & is.finite(t_lab)]
  setnames(lab_long, id_col, "DFCI_MRN")
  setkey(lab_long, LAB_NAME)
  lab_chunks <- split(lab_long[, .(DFCI_MRN, LAB_VALUE, t_lab)], lab_long$LAB_NAME)
  rm(lab_long); invisible(gc(verbose = FALSE))
  empty_chunk <- data.table(DFCI_MRN = character(0), LAB_VALUE = numeric(0), t_lab = numeric(0))

  write_curves_by_lab <- if (write_curves) {
    if (is.null(curve_lab_filter)) {
      setNames(rep(TRUE, length(landmark_labs)), landmark_labs)
    } else {
      setNames(landmark_labs %in% curve_lab_filter, landmark_labs)
    }
  } else {
    setNames(rep(FALSE, length(landmark_labs)), landmark_labs)
  }

  per_lab <- parallel::mclapply(
    landmark_labs,
    function(lab) {
      dt_lab <- if (is.null(lab_chunks[[lab]])) data.table::copy(empty_chunk) else lab_chunks[[lab]]
      process_one_lab(
        lab, dt_lab, landmark_day, k_pop, k_pat, nthreads, fit_timeout_seconds,
        max_fs_patients, patient_ridge_lambda, min_obs_per_lab, deriv_step_days,
        trailing_window_days, auc_grid_points, curve_grid_points,
        write_curves_by_lab[[lab]], id_col
      )
    },
    mc.cores = n_workers, mc.preschedule = FALSE
  )
  names(per_lab) <- landmark_labs

  failed <- vapply(per_lab, function(x) inherits(x, "try-error"), logical(1))
  if (any(failed)) {
    stop(sprintf(
      "Lab fit(s) failed under mclapply: %s\n%s",
      paste(landmark_labs[failed], collapse = ", "),
      paste(vapply(per_lab[failed], as.character, character(1)), collapse = "\n")
    ))
  }

  for (lab in landmark_labs) {
    log_lines <- per_lab[[lab]]$log_lines
    if (length(log_lines) > 0) cat(paste(log_lines, collapse = "\n"), "\n")
  }
  flush.console()

  diagnostic_rows <- lapply(per_lab, `[[`, "diagnostics")
  feature_frames <- Filter(Negate(is.null), lapply(per_lab, `[[`, "features"))
  curve_frames <- Filter(Negate(is.null), lapply(per_lab, `[[`, "curves"))

  if (length(feature_frames) > 0) {
    all_ids <- unique(unlist(lapply(feature_frames, `[[`, id_col)))
    combined <- setNames(data.table(all_ids), id_col)
    setkeyv(combined, id_col)
    for (fr in feature_frames) {
      setkeyv(fr, id_col)
      cols <- setdiff(names(fr), id_col)
      combined[fr, (cols) := mget(paste0("i.", cols))]
    }
  } else {
    combined <- data.table()
    combined[[id_col]] <- character(0)
  }
  fwrite(combined, out_path)
  cat(sprintf("Wrote %s (%d patients x %d feature columns)\n", out_path, nrow(combined), ncol(combined) - 1))

  fwrite(rbindlist(diagnostic_rows), diag_path)
  cat(sprintf("Wrote %s\n", diag_path))

  combined_curves <- if (length(curve_frames) > 0) {
    rbindlist(curve_frames, use.names = TRUE)
  } else {
    empty_curves <- data.table(
      LAB_NAME = character(), t_lab = numeric(), GAM_FITTED = numeric()
    )
    empty_curves[[id_col]] <- character()
    setcolorder(empty_curves, c(id_col, "LAB_NAME", "t_lab", "GAM_FITTED"))
    empty_curves
  }
  fwrite(combined_curves, curve_path)
  cat(sprintf("Wrote %s (%d fitted curve points)\n", curve_path, nrow(combined_curves)))
}
