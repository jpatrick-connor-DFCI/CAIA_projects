local({
  old <- options(compass.esmo.autorun = FALSE)
  on.exit(options(old))
  source("COMPASS/figures/ESMO_AI_Figures.R", local = .GlobalEnv)
})
esmo_load_helpers()

expect_error <- function(code, pattern) {
  error <- tryCatch({ force(code); NULL }, error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error)))
}

# Labels outside the day-0 risk set cannot enter any Results-1 denominator.
patients <- tibble(DFCI_MRN = 1:6, PLATINUM = c(0, 1, 0, 1, 1, 0))
labels <- tibble(DFCI_MRN = 1:7,
  primary_label = c("conventional", "conventional", "avpc", "avpc", "nepc", "nepc", "nepc"))
manual <- tibble(DFCI_MRN = 1:7, manual_label = labels$primary_label)
d <- esmo_llm_data(patients, labels, manual)
stopifnot(d$audit$n_classified == 6, d$audit$n_reviewed == 6, d$audit$agreement == 1,
  sum(d$landscape$n) == 6, sum(d$enrichment$n) == 6,
  all(d$enrichment$lo <= d$enrichment$probability),
  all(d$enrichment$hi >= d$enrichment$probability))
expect_error(esmo_llm_data(bind_rows(patients, patients[1, ]), labels, manual), "Duplicate")

# A missing model/landmark cell must not yield a seemingly complete figure.
performance <- tidyr::expand_grid(name = c("Cox baseline (age)", "Elastic-Net Cox", "XGBoost Survival"),
  landmark = c(0, 90, 180)) %>% mutate(auc = .75, n_test = 100, n_events_test = 20)
expect_error(esmo_result3(performance[-1, ], tibble()), "Incomplete")
expect_error(esmo_result3(bind_rows(performance, performance[1, ]), tibble()), "Duplicate")
importance <- tibble(landmark_days = 180, feature = c("ALT__max", "age", "PSA__missing"),
  feature_stat = c("max", "", "missing"), lab_name = c("ALT", "age", "PSA"), gain = c(10, 100, 200))
r3 <- esmo_result3(performance, importance)
stopifnot(identical(r3$tables$top_features$feature, "ALT__max"))

# Preserve the delivered KM fraction (the proposal contains 20% curves),
# rather than relabeling it as the manuscript pipeline's current 10% default.
volcano <- tidyr::expand_grid(landmark_days = c(0, 180), lab_name = c("PSA", "Testosterone")) %>%
  mutate(feature_stat = "mean", coef_feature = c(.5, -.4, .6, -.5), p_value = .001,
    q_value = .002, ci_lower = .5, ci_upper = 2)
km_summary <- tibble(scope = "pooled", site_name = NA_character_, lab = c("PSA", "Testosterone"),
  landmark_days = 180, cutoff_scope = "pooled", status = "ok", extreme_fraction = .2,
  n_bottom = 20, n_top = 20, events_bottom = 3, events_top = 8,
  q_low = 1, q_high = 5, logrank_p = .001)
km <- tidyr::expand_grid(lab = c("PSA", "Testosterone"), stratum = c("Bottom 20%", "Top 20%"),
  time = c(0, 365, 730)) %>% mutate(scope = "pooled", site_name = NA_character_,
    landmark_days = 180, cutoff_scope = "pooled", survival = 1 - time / 10000,
    lower = survival - .02, upper = pmin(1, survival + .02))
r2 <- esmo_result2(prepare_mean_univariable(volcano), list(km = km, km_summary = km_summary))
stopifnot(length(r2$plots) == 4, all(r2$tables$km_summary$extreme_fraction == .2))

# Paired sequencing/Gleason subset, not the ADT clock or patients missing a
# clinical stratifier. Compute score cutpoints only after this restriction.
transfer <- tidyr::expand_grid(endpoint = c("platinum", "avpc", "nepc"), person_id = 1:20) %>%
  mutate(clock = "sequencing", model_family = "xgboost_cox", bundle = "frozen_xgb",
    landmark_days = 180, sequenced = TRUE, risk_score = as.numeric(person_id),
    duration_days = if_else(person_id %% 2 == 0, 2000, 1000), event = as.integer(person_id %% 2 != 0),
    GLEASON_SCORE = if_else(person_id <= 10, 7, 8),
    TP53_ANY = as.integer(person_id > 10), RB1_ANY = 0, PTEN_ANY = 0)
unpaired <- transfer[transfer$person_id == 20, ] %>% mutate(person_id = 99, risk_score = 1e6, GLEASON_SCORE = NA_real_)
wrong_clock <- transfer %>% mutate(clock = "adt", person_id = person_id + 100)
gradient <- esmo_transfer_gradient(bind_rows(transfer, unpaired, wrong_clock))
stopifnot(all(gradient$clock == "sequencing"), all(gradient$horizon_days == 1825),
  all(gradient$n[gradient$stratifier == "score_extreme"] == 4),
  all(gradient$n[gradient$stratifier %in% c("gleason_group", "trio_group")] == 10))
expect_error(esmo_transfer_gradient(bind_rows(transfer, transfer[1, ])), "Duplicate")

metrics <- tidyr::expand_grid(endpoint = c("platinum", "avpc", "nepc"), landmark_days = c(0, 90, 180)) %>%
  mutate(model_family = "xgboost_cox", config = "both", bundle = "frozen_xgb", n_patients = 20,
    n_events = 10, mean_auc = .75, mean_auc_lo = .65, mean_auc_hi = .85)
r4 <- esmo_result4(metrics, gradient)
risks <- r4$tables$event_probabilities
stopifnot(length(r4$plots) == 4,
  all(abs(risks$probability - (1 - risks$event_free)) < 1e-12),
  all(abs(risks$probability_lo - (1 - risks$hi)) < 1e-12),
  all(abs(risks$probability_hi - (1 - risks$lo)) < 1e-12))
expect_error(esmo_result4(metrics %>% mutate(mean_auc_lo = NA_real_), gradient), "95%")
expect_error(esmo_result4(metrics, gradient %>% mutate(clock = "adt")), "sequencing clock")

# All four compositions and their individual panels must render, with only
# aggregate tables written beside them.
root <- tempfile("esmo-figures-")
dir.create(root)
tryCatch({
  results <- list(esmo_result1(d), r2, r3, r4)
  for (i in seq_along(results)) esmo_export(results[[i]], as.character(i), root)
  pdfs <- list.files(root, pattern = "[.]pdf$", full.names = TRUE, recursive = TRUE)
  stopifnot(length(pdfs) == 17, all(vapply(pdfs, figure_file_complete, logical(1))),
    !length(list.files(root, pattern = "[.](svg|png)$", recursive = TRUE)))
  for (path in list.files(root, pattern = "[.]csv$", full.names = TRUE, recursive = TRUE)) {
    table <- readr::read_csv(path, show_col_types = FALSE)
    stopifnot(!any(c("person_id", "DFCI_MRN", "risk_score") %in% names(table)))
  }
}, finally = unlink(root, recursive = TRUE))
cat("ESMO AI figure checks passed.\n")
