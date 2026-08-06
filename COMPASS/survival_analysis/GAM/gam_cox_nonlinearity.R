# gam_cox_nonlinearity.R
#
# For each GAM feature that survives an independent train_val coverage/
# variability gate at a landmark, fits a penalized-spline Cox model and a linear
# Cox model on the SAME rows and asks whether the smooth is doing real work:
# EDF meaningfully above 1 plus a small LRT p-value flags a feature whose
# `hazard_ratio_per_sd` (from run_univariate_nobs_adjusted_associations) is
# summarizing a relationship that is not actually linear -- a threshold or
# U-shape the existing per-feature Cox model is blind to.
#
# This script is independent of the main Python model run, but deliberately
# mirrors survival_common/cox_models.py's
# run_univariate_nobs_adjusted_associations() row-for-row:
#   - fits on ALL splits (train+valid+test), not just train_val; the independent
#     coverage/variability gate uses train+valid only
#   - COMPASS has zero baseline covariates (compass_profile.py does not
#     override static_covariates); the adjustment set here is exactly
#     n_obs_z + age (+ x_missing when there is partial missingness), matching
#     the Python model. Do not add gender/cancer-type terms -- that is IPIO's
#     configuration, not this one.
#   - mean-imputes the feature, z-scores it and n_observations, and
#     standardizes age using the same population statistics (ddof=0) as
#     sklearn's SimpleImputer/the manual z-scoring in cox_models.py
#
# Inputs:
#   aggregated_landmark{D}.csv               base per-patient feature table
#                                             under <inputs-dir>
#   gam_trajectory_features_landmark{D}.csv  GAM-only features under
#                                             <gam-features-dir>
#
# Output: gam_cox_nonlinearity_landmark{D}.csv with columns
#   landmark_days, endpoint, feature, lab_name, feature_stat, n_used,
#   n_events, edf, p_smooth, p_lrt, q_lrt, delta_aic, coef_linear, p_linear,
#   q_linear, note
#
# Correctness check: coef_linear here should match coef_feature in
# cox_agg_univariate_nobs_adjusted.csv for the same feature/landmark, to
# within the tie-handling difference noted below. Mismatch means the
# preprocessing above was not mirrored correctly.
#
# Caveats (state plainly, do not oversell precision):
#   - mgcv's cox.ph() uses Peto's correction for tied event times; lifelines'
#     CoxPHFitter (used on the Python side) defaults to the Efron
#     approximation. With few ties the two coefficients should agree closely;
#     with many ties expect small, not exact, agreement.
#   - anova(mod_l, mod_s, test="Chisq") on penalized GAM fits is an
#     *approximate* LRT (the EDF-based reference distribution is
#     conservative per mgcv's own documentation). Read delta_aic alongside
#     p_lrt/q_lrt rather than trusting either alone.

suppressWarnings({
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for the penalized-spline Cox fits. Install it before running this script.")
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

DEFAULT_INPUTS_DIR <- "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/prediction_inputs"
DEFAULT_GAM_FEATURES_DIR <- "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/GAM"
DEFAULT_OUTPUT_DIR <- DEFAULT_GAM_FEATURES_DIR
DEFAULT_LANDMARK_DAYS <- "0,90,180"
DEFAULT_ENDPOINT <- "platinum"
DEFAULT_DURATION_COL <- "t_platinum"
DEFAULT_EVENT_COL <- "PLATINUM"
DEFAULT_AGE_COL <- "AGE_AT_TREATMENTSTART"
DEFAULT_ID_COL <- "DFCI_MRN"
DEFAULT_MIN_EVENTS_PER_FEATURE <- 10L
DEFAULT_K_SMOOTH <- 10
DEFAULT_MIN_PATIENT_COVERAGE <- 0.20
DEFAULT_N_WORKERS <- 1L

parse_cli_args <- function(args, defaults) {
  out <- defaults
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop(sprintf("Unexpected argument (expected --flag): %s", key))
    name <- gsub("-", "_", sub("^--", "", key))
    if (!name %in% names(defaults)) stop(sprintf("Unknown argument: %s", key))
    if (i == length(args)) stop(sprintf("Flag %s requires a value.", key))
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
    gam_features_dir = DEFAULT_GAM_FEATURES_DIR,
    output_dir = DEFAULT_OUTPUT_DIR,
    landmark_days = DEFAULT_LANDMARK_DAYS,
    endpoint = DEFAULT_ENDPOINT,
    duration_col = DEFAULT_DURATION_COL,
    event_col = DEFAULT_EVENT_COL,
    age_col = DEFAULT_AGE_COL,
    id_col = DEFAULT_ID_COL,
    min_events_per_feature = DEFAULT_MIN_EVENTS_PER_FEATURE,
    k_smooth = DEFAULT_K_SMOOTH,
    min_patient_coverage = DEFAULT_MIN_PATIENT_COVERAGE,
    n_workers = DEFAULT_N_WORKERS
  )
)

landmark_days <- as.integer(strsplit(as.character(args_list$landmark_days), ",")[[1]])
inputs_dir <- args_list$inputs_dir
gam_features_dir <- args_list$gam_features_dir
output_dir <- args_list$output_dir
endpoint <- args_list$endpoint
duration_col <- args_list$duration_col
event_col <- args_list$event_col
age_col <- args_list$age_col
id_col <- args_list$id_col
min_events_per_feature <- as.integer(args_list$min_events_per_feature)
k_smooth <- as.integer(args_list$k_smooth)
min_patient_coverage <- as.numeric(args_list$min_patient_coverage)
n_workers <- as.integer(args_list$n_workers)

if (n_workers < 1L) stop("--n-workers must be at least 1")
if (!is.finite(min_patient_coverage) || min_patient_coverage < 0 || min_patient_coverage > 1) {
  stop("--min-patient-coverage must be between 0 and 1")
}
detected_cores <- parallel::detectCores()
if (!is.na(detected_cores) && n_workers > detected_cores) {
  warning(sprintf(
    "--n-workers %d exceeds %d detected cores (oversubscription risk)",
    n_workers, detected_cores
  ))
}

parse_feature_name <- function(feature) {
  # Mirrors survival_common/cox_engine.py's rsplit("__", 1): split on the
  # LAST "__" so lab names containing "__" (none currently do) stay intact.
  parts <- regmatches(feature, regexpr("__[^_]*(_[^_]+)*$", feature))
  if (length(parts) == 0 || nchar(parts) == 0) return(c(feature, NA_character_))
  stat <- sub("^__", "", parts)
  lab <- substr(feature, 1, nchar(feature) - nchar(parts))
  c(lab, stat)
}

fit_one_feature <- function(d, feature, n_obs_feature, duration_col, event_col, age_col,
                             min_events_per_feature, k_smooth) {
  note <- ""
  base_cols <- c(feature, n_obs_feature, duration_col, event_col, age_col)
  fd <- d[, ..base_cols]
  required_cols <- c(duration_col, event_col, age_col)
  fd <- fd[stats::complete.cases(fd[, ..required_cols])]

  n_used <- nrow(fd)
  observed_non_missing <- sum(!is.na(fd[[feature]]))
  observed_n_obs <- sum(!is.na(fd[[n_obs_feature]]))
  n_events <- if (n_used > 0) sum(fd[[event_col]]) else 0L

  empty <- function(note) {
    list(n_used = n_used, n_events = n_events, edf = NA_real_, p_smooth = NA_real_,
         p_lrt = NA_real_, delta_aic = NA_real_, coef_linear = NA_real_,
         p_linear = NA_real_, note = note)
  }

  if (n_used == 0) return(empty("no_rows_with_outcomes"))
  if (observed_non_missing == 0) return(empty("no_non_missing_rows"))
  if (observed_n_obs == 0) return(empty("no_non_missing_n_obs_rows"))
  if (n_events < min_events_per_feature) {
    return(empty(sprintf("too_few_events_lt_%d", min_events_per_feature)))
  }

  missing_indicator <- as.numeric(is.na(fd[[feature]]))
  include_missing_indicator <- length(unique(missing_indicator)) > 1

  feature_mean <- mean(fd[[feature]], na.rm = TRUE)
  feature_values <- ifelse(is.na(fd[[feature]]), feature_mean, fd[[feature]])
  feature_sd <- sqrt(mean((feature_values - mean(feature_values))^2))  # population sd, ddof=0
  if (!is.finite(feature_sd) || feature_sd <= 0) return(empty("feature_has_no_variation"))

  n_obs_mean <- mean(fd[[n_obs_feature]], na.rm = TRUE)
  n_obs_values <- ifelse(is.na(fd[[n_obs_feature]]), n_obs_mean, fd[[n_obs_feature]])
  n_obs_sd <- sqrt(mean((n_obs_values - mean(n_obs_values))^2))
  if (!is.finite(n_obs_sd) || n_obs_sd <= 0) return(empty("n_obs_has_no_variation"))

  feature_z <- (feature_values - mean(feature_values)) / feature_sd
  n_obs_z <- (n_obs_values - mean(n_obs_values)) / n_obs_sd

  age_values <- as.numeric(fd[[age_col]])
  age_sd <- sqrt(mean((age_values - mean(age_values))^2))
  age_z <- if (is.finite(age_sd) && age_sd > 0) (age_values - mean(age_values)) / age_sd else age_values - mean(age_values)

  dur <- as.numeric(fd[[duration_col]])
  ev <- as.numeric(fd[[event_col]])

  model_df <- data.table(
    feature_z = feature_z, n_obs_z = n_obs_z, age = age_z,
    duration = dur, event = ev
  )
  if (include_missing_indicator) model_df[, feature_missing := missing_indicator]

  n_unique_feature_z <- data.table::uniqueN(feature_z)
  k_use <- max(3L, min(k_smooth, n_unique_feature_z - 1L))

  rhs_extra <- if (include_missing_indicator) " + feature_missing" else ""
  f_smooth <- as.formula(sprintf("duration ~ s(feature_z, k = %d) + n_obs_z + age%s", k_use, rhs_extra))
  f_linear <- as.formula(sprintf("duration ~ feature_z + n_obs_z + age%s", rhs_extra))

  mod_s <- tryCatch(
    gam(f_smooth, family = cox.ph(), weights = event, data = model_df, method = "REML"),
    error = function(e) NULL
  )
  mod_l <- tryCatch(
    gam(f_linear, family = cox.ph(), weights = event, data = model_df, method = "REML"),
    error = function(e) NULL
  )

  if (is.null(mod_s) || is.null(mod_l)) return(empty("gam_fit_failed"))

  s_table <- tryCatch(summary(mod_s)$s.table, error = function(e) NULL)
  edf <- if (!is.null(s_table) && nrow(s_table) > 0) s_table[1, "edf"] else NA_real_
  p_smooth <- if (!is.null(s_table) && nrow(s_table) > 0) s_table[1, ncol(s_table)] else NA_real_

  lrt <- tryCatch(anova(mod_l, mod_s, test = "Chisq"), error = function(e) NULL)
  p_lrt <- if (!is.null(lrt)) lrt[["Pr(>Chi)"]][2] else NA_real_
  delta_aic <- tryCatch(AIC(mod_l) - AIC(mod_s), error = function(e) NA_real_)

  p_table <- summary(mod_l)$p.table
  coef_linear <- if ("feature_z" %in% rownames(p_table)) p_table["feature_z", "Estimate"] else NA_real_
  p_linear <- if ("feature_z" %in% rownames(p_table)) p_table["feature_z", ncol(p_table)] else NA_real_

  list(n_used = n_used, n_events = n_events, edf = edf, p_smooth = p_smooth,
       p_lrt = p_lrt, delta_aic = delta_aic, coef_linear = coef_linear,
       p_linear = p_linear, note = "ok")
}

# ---------------------------------------------------------------------------
# Per-feature worker body. Returns a single data.table row instead of
# appending to a shared `rows` list, so it can be dispatched via mclapply().
# mclapply preserves input order, so `q_lrt`'s BH adjustment (order-dependent)
# stays bit-identical regardless of --n-workers. `log_lines` is buffered and
# printed by the parent after collection so progress lines from concurrent
# workers don't interleave.
# ---------------------------------------------------------------------------
process_one_feature <- function(feature, aggregated, landmark_day, endpoint, duration_col,
                                 event_col, age_col, min_events_per_feature, k_smooth) {
  log_lines <- character(0)
  log <- function(...) log_lines[[length(log_lines) + 1]] <<- sprintf(...)

  parsed <- parse_feature_name(feature)
  lab_name <- parsed[1]
  feature_stat <- parsed[2]

  if (identical(feature_stat, "n_observations")) {
    row <- data.table(
      landmark_days = landmark_day, endpoint = endpoint, feature = feature,
      lab_name = lab_name, feature_stat = feature_stat, n_used = NA_integer_,
      n_events = NA_integer_, edf = NA_real_, p_smooth = NA_real_, p_lrt = NA_real_,
      delta_aic = NA_real_, coef_linear = NA_real_, p_linear = NA_real_,
      note = "target_is_n_observations"
    )
    return(list(row = row, log_lines = log_lines))
  }

  n_obs_feature <- paste0(lab_name, "__n_observations")
  if (!n_obs_feature %in% names(aggregated)) {
    row <- data.table(
      landmark_days = landmark_day, endpoint = endpoint, feature = feature,
      lab_name = lab_name, feature_stat = feature_stat, n_used = NA_integer_,
      n_events = NA_integer_, edf = NA_real_, p_smooth = NA_real_, p_lrt = NA_real_,
      delta_aic = NA_real_, coef_linear = NA_real_, p_linear = NA_real_,
      note = "missing_matching_n_obs_feature"
    )
    return(list(row = row, log_lines = log_lines))
  }

  result <- fit_one_feature(
    aggregated, feature, n_obs_feature, duration_col, event_col, age_col,
    min_events_per_feature, k_smooth
  )
  log(
    "  [%s] n_used=%s n_events=%s edf=%s p_lrt=%s note=%s",
    feature, result$n_used, result$n_events,
    if (is.na(result$edf)) "NA" else sprintf("%.2f", result$edf),
    if (is.na(result$p_lrt)) "NA" else sprintf("%.3g", result$p_lrt),
    result$note
  )
  row <- data.table(
    landmark_days = landmark_day, endpoint = endpoint, feature = feature,
    lab_name = lab_name, feature_stat = feature_stat, n_used = result$n_used,
    n_events = result$n_events, edf = result$edf, p_smooth = result$p_smooth,
    p_lrt = result$p_lrt, delta_aic = result$delta_aic,
    coef_linear = result$coef_linear, p_linear = result$p_linear, note = result$note
  )
  list(row = row, log_lines = log_lines)
}

for (landmark_day in landmark_days) {
  cat(sprintf("\n##### GAM COX NONLINEARITY: LANDMARK +%dD #####\n", landmark_day))

  agg_path <- file.path(inputs_dir, sprintf("aggregated_landmark%d.csv", landmark_day))
  if (!file.exists(agg_path)) stop(sprintf("Missing %s. Run build_prediction_inputs.py first.", agg_path))
  aggregated <- fread(agg_path)
  aggregated[[id_col]] <- as.character(aggregated[[id_col]])

  gam_path <- file.path(gam_features_dir, sprintf("gam_trajectory_features_landmark%d.csv", landmark_day))
  if (!file.exists(gam_path)) stop(sprintf("Missing %s. Run gam_trajectory_features.R first.", gam_path))
  gam_features <- fread(gam_path)
  gam_features[[id_col]] <- as.character(gam_features[[id_col]])
  gam_cols <- setdiff(names(gam_features), id_col)
  if (length(gam_cols) == 0) stop(sprintf("%s contains no GAM feature columns.", gam_path))
  if (any(!grepl("__gam_", gam_cols, fixed = TRUE))) {
    stop(sprintf("Non-GAM columns found in %s: %s", gam_path,
                 paste(gam_cols[!grepl("__gam_", gam_cols, fixed = TRUE)], collapse = ", ")))
  }
  overlap <- intersect(gam_cols, names(aggregated))
  if (length(overlap) > 0) {
    stop(sprintf("GAM feature columns unexpectedly overlap the main aggregate table: %s",
                 paste(overlap, collapse = ", ")))
  }
  aggregated <- merge(aggregated, gam_features, by = id_col, all.x = TRUE)
  cat(sprintf("Joined %d GAM-only feature columns from %s\n", length(gam_cols), gam_path))

  if (!"split" %in% names(aggregated)) stop(sprintf("%s is missing the split column.", agg_path))
  train_val <- aggregated[split %in% c("train", "valid")]
  selection_rows <- lapply(gam_cols, function(feature) {
    values <- train_val[[feature]]
    parsed <- parse_feature_name(feature)
    data.table(
      landmark_days = landmark_day,
      feature = feature,
      lab_name = parsed[[1]],
      feature_stat = parsed[[2]],
      coverage = mean(!is.na(values)),
      unique_non_missing = uniqueN(values[!is.na(values)])
    )
  })
  selection <- rbindlist(selection_rows)
  selection[, selected := coverage >= min_patient_coverage & unique_non_missing > 1]
  features_to_test <- selection[selected == TRUE]$feature
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  selection_path <- file.path(output_dir, sprintf("gam_feature_selection_landmark%d.csv", landmark_day))
  fwrite(selection, selection_path)
  cat(sprintf("Selected %d/%d GAM features at coverage >= %.2f; wrote %s\n",
              length(features_to_test), length(gam_cols), min_patient_coverage, selection_path))
  if (length(features_to_test) == 0) {
    stop(sprintf("No GAM features passed selection for landmark %d.", landmark_day))
  }

  per_feature <- parallel::mclapply(
    features_to_test,
    process_one_feature,
    aggregated = aggregated, landmark_day = landmark_day, endpoint = endpoint,
    duration_col = duration_col, event_col = event_col, age_col = age_col,
    min_events_per_feature = min_events_per_feature, k_smooth = k_smooth,
    mc.cores = n_workers, mc.preschedule = FALSE
  )

  failed <- vapply(per_feature, function(x) inherits(x, "try-error"), logical(1))
  if (any(failed)) {
    stop(sprintf(
      "Feature fit(s) failed under mclapply: %s\n%s",
      paste(features_to_test[failed], collapse = ", "),
      paste(vapply(per_feature[failed], as.character, character(1)), collapse = "\n")
    ))
  }

  for (res in per_feature) {
    if (length(res$log_lines) > 0) cat(paste(res$log_lines, collapse = "\n"), "\n")
  }
  flush.console()

  rows <- lapply(per_feature, `[[`, "row")
  out <- rbindlist(rows)
  out[, q_lrt := stats::p.adjust(p_lrt, method = "BH")]
  out[, q_linear := stats::p.adjust(p_linear, method = "BH")]
  setcolorder(out, c(
    "landmark_days", "endpoint", "feature", "lab_name", "feature_stat", "n_used",
    "n_events", "edf", "p_smooth", "p_lrt", "q_lrt", "delta_aic", "coef_linear",
    "p_linear", "q_linear", "note"
  ))

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(output_dir, sprintf("gam_cox_nonlinearity_landmark%d.csv", landmark_day))
  fwrite(out, out_path)
  cat(sprintf("Wrote %s (%d features)\n", out_path, nrow(out)))

  flagged <- out[!is.na(q_lrt) & q_lrt < 0.05 & edf > 1.5]
  if (nrow(flagged) > 0) {
    cat(sprintf("\n%d feature(s) with edf > 1.5 and q_lrt < 0.05 (possible nonlinear hazard):\n", nrow(flagged)))
    print(flagged[order(q_lrt), .(feature, edf, p_lrt, q_lrt, delta_aic)])
  } else {
    cat("\nNo features flagged as nonlinear at q_lrt < 0.05 with edf > 1.5.\n")
  }
}
