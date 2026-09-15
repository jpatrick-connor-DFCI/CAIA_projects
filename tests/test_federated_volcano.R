source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/federated_no_msk_figures.R")
local({
  path <- tempfile(fileext = ".csv"); on.exit(unlink(path))
  d <- tibble(landmark_days = 0, endpoint = "platinum", feature = paste0("feature", 1:7),
    lab_name = c("Prostate_specific_Ag__Mass_volume__in_Serum_or_Plasma", "Testosterone", "Hemoglobin",
                 "Albumin", "Body_height", "Sodium", "Potassium"), feature_stat = "mean",
    coef_feature = c(.5, -.4, .3, 5, .2, .2, .2), ci_lower = .8,
    ci_upper = c(2, 2, 2, 2, 2, 1000, 2), p_value = c(0, .01, .5, .001, .2, .2, -1), q_value = .04)
  write_csv(d, path)
  prepared <- load_federated_no_msk_volcano(path)
  stopifnot(identical(prepared$eligible_volcano, c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE)),
            prepared$lab_name[1] == "PSA", assign_category(prepared$lab_name[1]) == "Androgen axis")
  p <- plot_federated_no_msk_volcano(prepared, 0)
  local_plot <- plot_volcano_panel(filter(prepared, eligible_volcano), "local")
  stopifnot(identical(p$labels$x, local_plot$labels$x), identical(p$labels$y, local_plot$labels$y),
            isTRUE(all.equal(ggplot_build(p)$data, ggplot_build(local_plot)$data)))
  for (lm in c(0, 90, 180)) withCallingHandlers(invisible(ggplotGrob(plot_federated_no_msk_volcano(prepared, lm))),
    warning = function(w) stop(w))
  # Duplicate fits must never silently add extra points.
  write_csv(bind_rows(d, d[1, ]), path)
  duplicate <- tryCatch(load_federated_no_msk_volcano(path), error = identity)
  stopifnot(inherits(duplicate, "error"))
})
cat("Federated volcano: shared renderer, aliases, filters, p=0, empty landmarks, and duplicate checks passed.\n")
