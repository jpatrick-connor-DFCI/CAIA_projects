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
local({
  root <- tempfile("federated-sites-"); dir.create(root)
  on.exit(unlink(root, recursive = TRUE))
  fed <- file.path(root, "federated.csv")
  dir.create(file.path(root, "nvflare_within_site_cox_univariate"))
  path <- file.path(root, "nvflare_within_site_cox_univariate", "cox_within_site_all_sites_cohort.csv")
  stopifnot(is.null(suppressWarnings(load_federated_no_msk_sites(fed))))
  d <- tibble(site_name = c("dana_farber_caia_1_1", "jhu_caia_1_1", "fred_hutch_caia_1_1"),
              analysis_label = "adt", landmark_days = 0, n_patients = c(100, 10, 0), n_events = c(4, 0, 0))
  write_csv(d, path)
  counts <- load_federated_no_msk_sites(fed)
  stopifnot(counts$event_incidence_pct[counts$site == "Dana-Farber"] == 4,
            counts$event_incidence_pct[counts$site == "Johns Hopkins"] == 0,
            is.na(counts$event_incidence_pct[counts$site == "Fred Hutch"]))
  p <- plot_federated_no_msk_sites(counts)
  stopifnot(nrow(p$data) == 18, all(is.na(p$data$n_patients[p$data$landmark_days == 90])))
  withCallingHandlers(invisible(ggplotGrob(p)), warning = function(w) stop(w))
  for (bad in list(bind_rows(d, d[1, ]), mutate(d, n_events = 101),
                   mutate(d, n_patients = -1), mutate(d, n_events = NA_real_),
                   mutate(d, site_name = "msk_caia_1_1"))) {
    write_csv(bad, path)
    stopifnot(inherits(tryCatch(load_federated_no_msk_sites(fed), error = identity), "error"))
  }
})
cat("Federated forests and site counts: significance, incidence, missing data and invalid input checks passed.\n")
