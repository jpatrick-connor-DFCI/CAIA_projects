source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/federated_no_msk_figures.R")
local({
  path <- tempfile(fileext = ".csv"); on.exit(unlink(path))
  d <- tibble(landmark_days = 0, endpoint = "platinum", feature = paste0("feature", 1:7),
    lab_name = c("Prostate_specific_Ag__Mass_volume__in_Serum_or_Plasma", "Testosterone", "PSA",
                 "PSA", "Testosterone", "Albumin", "PSA"),
    feature_stat = c("mean", "mean", "min", "max", "delta", "mean", "n_observations"),
    hazard_ratio_per_sd = c(1.2, .7, 1, 1.1, 1, 1, 1), ci_lower = .5, ci_upper = 2,
    p_value = c(.01, .001, .05, .01, .001, .001, .001),
    q_value = c(.1, .02, .05, NA, .01, .01, .01))
  write_csv(d, path)
  prepared <- load_federated_no_msk_forest(path)
  stopifnot(nrow(prepared) == 4L, all(prepared$lab_name %in% c("PSA", "Testosterone")),
    !any(prepared$feature_stat %in% c("delta", "n_observations")),
    identical(prepared$q_value, d$q_value[1:4]),
    identical(prepared$significance, c("Nominal p < 0.05 only", "FDR q < 0.05", "Not significant", "Significance unavailable")))
  for (lm in c(0, 90, 180)) {
    p <- plot_federated_no_msk_forest(prepared, lm)
    stopifnot(nrow(p$data) == 8L)
    withCallingHandlers(invisible(ggplotGrob(p)), warning = function(w) stop(w))
  }
  # Invalid estimates/intervals stay visible as labelled unavailable rows.
  broken <- prepared; broken$hazard_ratio_per_sd[1] <- Inf; broken$ci_upper[2] <- NA_real_
  write_csv(broken, path); broken <- load_federated_no_msk_forest(path)
  stopifnot(!broken$valid_estimate[1], !broken$valid_ci[2])
  withCallingHandlers(invisible(ggplotGrob(plot_federated_no_msk_forest(broken, 0))), warning = function(w) stop(w))
  write_csv(bind_rows(d, d[1, ]), path)
  duplicate <- tryCatch(load_federated_no_msk_forest(path), error = identity)
  stopifnot(inherits(duplicate, "error"))
})
cat("Federated forest: analytes/stats, supplied p/q thresholds, missing estimates and duplicate checks passed.\n")
