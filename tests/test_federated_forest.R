source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_workflow.R")
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
  prepared <- load_federated_forest(path)
  stopifnot(nrow(prepared) == 4L, all(prepared$lab_name %in% c("PSA", "Testosterone")),
    !any(prepared$feature_stat %in% c("delta", "n_observations")),
    identical(prepared$q_value, d$q_value[1:4]),
    identical(prepared$significance, c("Nominal p < 0.05 only", "FDR q < 0.05", "Not significant", "Significance unavailable")))
  for (lm in c(0, 90, 180)) {
    p <- plot_federated_forest(prepared, lm)
    stopifnot(nrow(p$data) == 8L)
    withCallingHandlers(invisible(ggplotGrob(p)), warning = function(w) stop(w))
  }
  # Invalid estimates/intervals stay visible as labelled unavailable rows.
  broken <- prepared; broken$hazard_ratio_per_sd[1] <- Inf; broken$ci_upper[2] <- NA_real_
  write_csv(broken, path); broken <- load_federated_forest(path)
  stopifnot(!broken$valid_estimate[1], !broken$valid_ci[2])
  withCallingHandlers(invisible(ggplotGrob(plot_federated_forest(broken, 0))), warning = function(w) stop(w))
  write_csv(bind_rows(d, d[1, ]), path)
  duplicate <- tryCatch(load_federated_forest(path), error = identity)
  stopifnot(inherits(duplicate, "error"))
  within <- bind_rows(mutate(d, site_name = "jhu_caia_1_1", analysis_label = "adt"),
                      mutate(d, site_name = "dana_farber_caia_1_1", analysis_label = "adt", hazard_ratio_per_sd = 1.5))
  write_csv(within, path)
  prepared <- load_federated_forest(path, within_site = TRUE)
  stopifnot(nrow(prepared) == 8L, identical(prepared$q_value, rep(d$q_value[1:4], 2)))
  for (site in unique(prepared$site_name)) {
    p <- plot_federated_forest(prepared, 0, site_name = site)
    stopifnot(all(na.omit(p$data$site_name) == site), nrow(p$data) == 8L,
              grepl(federated_site_label(site), p$labels$title, fixed = TRUE))
    withCallingHandlers(invisible(ggplotGrob(p)), warning = function(w) stop(w))
  }
  stopifnot(inherits(tryCatch(plot_federated_forest(prepared, 0), error = identity), "error"))
  write_csv(bind_rows(within, within[1, ]), path)
  stopifnot(inherits(tryCatch(load_federated_forest(path, within_site = TRUE), error = identity), "error"))
  # MSK is a participating site: it must load and carry its display label.
  write_csv(mutate(within, site_name = "msk_caia_prod_1"), path)
  msk <- load_federated_forest(path, within_site = TRUE)
  stopifnot(nrow(msk) == 8L, all(msk$site == "MSK"))
  withCallingHandlers(invisible(ggplotGrob(plot_federated_forest(msk, 0, site_name = "msk_caia_prod_1"))),
                      warning = function(w) stop(w))
})
local({
  root <- tempfile("federated-sites-"); dir.create(root)
  on.exit(unlink(root, recursive = TRUE))
  fed <- file.path(root, "federated.csv")
  dir.create(file.path(root, "nvflare_within_site_univariate_cox"))
  path <- file.path(root, "nvflare_within_site_univariate_cox", "cox_within_site_all_sites_cohort.csv")
  stopifnot(is.null(suppressWarnings(load_federated_sites(fed))))
  d <- tibble(site_name = c("dana_farber_caia_1_1", "jhu_caia_1_1", "fred_hutch_caia_1_1"),
              analysis_label = "adt", landmark_days = 0, n_patients = c(100, 10, 0), n_events = c(4, 0, 0))
  write_csv(d, path)
  counts <- load_federated_sites(fed)
  stopifnot(counts$event_incidence_pct[counts$site == "Dana-Farber"] == 4,
            counts$event_incidence_pct[counts$site == "Johns Hopkins"] == 0,
            is.na(counts$event_incidence_pct[counts$site == "Fred Hutch"]))
  # Later landmark counts must never enter the day-0 figure or exported table.
  all_counts <- bind_rows(counts, mutate(counts, landmark_days = 90, n_events = n_patients))
  incidence <- prepare_federated_site_incidence(all_counts)
  stopifnot(nrow(incidence) == 3L, all(incidence$landmark_days == 0),
            sum(incidence$n_events) == 4,
            all(incidence$ci_lower_pct[incidence$available] >= 0),
            all(incidence$ci_upper_pct[incidence$available] <= 100))
  expected <- wilson_ci(4, 100)
  row <- filter(incidence, site == "Dana-Farber")
  stopifnot(isTRUE(all.equal(c(row$ci_lower_pct, row$ci_upper_pct), unname(expected[2:3])*100)))
  withCallingHandlers(stopifnot(inherits(plot_federated_sites(all_counts), "gtable")),
                      warning = function(w) stop(w))
  no_day0 <- prepare_federated_site_incidence(filter(all_counts, landmark_days == 90))
  stopifnot(all(!no_day0$available), all(is.na(no_day0$n_events)))
  withCallingHandlers(invisible(plot_federated_sites(filter(all_counts, landmark_days == 90))),
                      warning = function(w) stop(w))
  for (bad in list(bind_rows(d, d[1, ]), mutate(d, n_events = 101),
                   mutate(d, n_patients = -1), mutate(d, n_events = NA_real_))) {
    write_csv(bad, path)
    stopifnot(inherits(tryCatch(load_federated_sites(fed), error = identity), "error"))
  }
  # A four-site bundle including MSK loads and labels cleanly.
  write_csv(bind_rows(d, tibble(site_name = "msk_caia_prod_1", analysis_label = "adt",
    landmark_days = 0, n_patients = 9939, n_events = 308)), path)
  four <- load_federated_sites(fed)
  stopifnot(nrow(four) == 4L, "MSK" %in% four$site,
            isTRUE(all.equal(four$event_incidence_pct[four$site == "MSK"], 100*308/9939)))
})
cat("Federated forests and site counts: significance, incidence, missing data and invalid input checks passed.\n")
