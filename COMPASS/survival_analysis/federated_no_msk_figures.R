# Supplemental, aggregate-only comparison of the no-MSK federated screen with
# the canonical local ADT platinum univariate screen.  This intentionally
# reads coefficient summaries only; no patient-level federated inputs are
# required or copied into the figure tree.

federated_no_msk_local_result_path <- function(data_root, landmark) {
  file.path(data_root, "survival_analysis", "local_runs_adt", "cox",
            paste0("landmark_", landmark), "both",
            "cox_agg_univariate_nobs_adjusted.csv")
}

federated_no_msk_normalize <- function(frame, source) {
  required <- c("landmark_days", "endpoint", "feature", "hazard_ratio_per_sd",
                "p_value", "q_value")
  absent <- setdiff(required, names(frame))
  if (length(absent))
    stop(source, " results are missing required columns: ", paste(absent, collapse = ", "))

  frame %>%
    transmute(
      landmark_days = suppressWarnings(as.integer(.data$landmark_days)),
      endpoint = tolower(as.character(.data$endpoint)),
      feature = as.character(.data$feature),
      hr = suppressWarnings(as.numeric(.data$hazard_ratio_per_sd)),
      ci_lower = if ("ci_lower" %in% names(frame)) suppressWarnings(as.numeric(.data$ci_lower)) else NA_real_,
      ci_upper = if ("ci_upper" %in% names(frame)) suppressWarnings(as.numeric(.data$ci_upper)) else NA_real_,
      p_value = suppressWarnings(as.numeric(.data$p_value)),
      q_value = suppressWarnings(as.numeric(.data$q_value)),
      n_patients_used = if ("n_patients_used" %in% names(frame)) suppressWarnings(as.numeric(.data$n_patients_used)) else NA_real_,
      n_events_used = if ("n_events_used" %in% names(frame)) suppressWarnings(as.numeric(.data$n_events_used)) else NA_real_
    ) %>%
    filter(.data$endpoint == "platinum", !is.na(.data$landmark_days), !is.na(.data$feature)) %>%
    mutate(source = source)
}

load_federated_no_msk_comparison <- function(data_root, federated_path,
                                              landmarks = c(0L, 90L, 180L)) {
  if (!file.exists(federated_path))
    stop("Federated no-MSK results not found: ", federated_path)

  federated <- federated_no_msk_normalize(
    readr::read_csv(federated_path, show_col_types = FALSE), "Federated (no MSK)"
  ) %>%
    filter(.data$landmark_days %in% landmarks)

  local_paths <- vapply(landmarks, federated_no_msk_local_result_path,
                        character(1), data_root = data_root)
  missing_local <- local_paths[!file.exists(local_paths)]
  if (length(missing_local))
    stop("Local ADT platinum results are missing:\n  ",
         paste(unname(missing_local), collapse = "\n  "))
  local <- dplyr::bind_rows(lapply(local_paths, function(path) {
    federated_no_msk_normalize(
      readr::read_csv(path, show_col_types = FALSE), "Local COMPASS"
    )
  })) %>% filter(.data$landmark_days %in% landmarks)

  # Each source should have a single fit per feature/landmark.  Joining a
  # duplicated key would make an apparent agreement figure from a Cartesian
  # product, so fail explicitly instead.
  for (d in list(local = local, federated = federated)) {
    if (anyDuplicated(d[c("landmark_days", "feature")]))
      stop("Duplicate feature/landmark rows in ", d$source[[1]], " results")
  }

  joined <- inner_join(
    local %>% select(-all_of("source")), federated %>% select(-all_of("source")),
    by = c("landmark_days", "feature"), suffix = c("_local", "_federated")
  ) %>%
    mutate(
      valid_hr_local = is.finite(.data$hr_local) & .data$hr_local > 0,
      valid_hr_federated = is.finite(.data$hr_federated) & .data$hr_federated > 0,
      valid_hr = .data$valid_hr_local & .data$valid_hr_federated,
      valid_p_local = is.finite(.data$p_value_local) & .data$p_value_local > 0 & .data$p_value_local <= 1,
      valid_p_federated = is.finite(.data$p_value_federated) & .data$p_value_federated > 0 & .data$p_value_federated <= 1,
      usable_effect = .data$valid_hr & .data$valid_p_local & .data$valid_p_federated,
      log_hr_local = ifelse(.data$valid_hr_local, log(.data$hr_local), NA_real_),
      log_hr_federated = ifelse(.data$valid_hr_federated, log(.data$hr_federated), NA_real_),
      local_fdr = is.finite(.data$q_value_local) & .data$q_value_local < 0.05,
      federated_fdr = is.finite(.data$q_value_federated) & .data$q_value_federated < 0.05,
      discovery_class = case_when(
        .data$local_fdr & .data$federated_fdr ~ "FDR < 0.05 in both",
        .data$local_fdr ~ "Local only",
        .data$federated_fdr ~ "Federated only",
        TRUE ~ "Neither"
      ),
      direction_agrees = sign(.data$log_hr_local) == sign(.data$log_hr_federated)
    )
  joined$discovery_class <- factor(
    joined$discovery_class,
    levels = c("Neither", "Local only", "Federated only", "FDR < 0.05 in both")
  )
  joined
}

plot_federated_no_msk_hr_agreement <- function(d) {
  plot_data <- filter(d, .data$usable_effect)
  if (!nrow(plot_data)) return(NULL)
  lim <- max(abs(c(plot_data$log_hr_local, plot_data$log_hr_federated)), na.rm = TRUE)
  lim <- max(lim * 1.06, 0.25)
  ggplot(plot_data, aes(.data$log_hr_local, .data$log_hr_federated,
                        color = .data$discovery_class)) +
    geom_abline(slope = 1, intercept = 0, color = "grey55", linetype = "dashed") +
    geom_hline(yintercept = 0, color = "grey85", linewidth = .3) +
    geom_vline(xintercept = 0, color = "grey85", linewidth = .3) +
    geom_point(alpha = .72, size = 1.7) +
    coord_equal(xlim = c(-lim, lim), ylim = c(-lim, lim)) +
    facet_wrap(~landmark_days, labeller = labeller(landmark_days = function(x) paste0("+", x, " days"))) +
    scale_color_manual(values = c("Neither" = "grey70", "Local only" = "#0072B2",
                                  "Federated only" = "#D55E00", "FDR < 0.05 in both" = "#009E73"),
                       name = NULL) +
    labs(
      title = "Local and federated univariate Cox effects agree across landmarks",
      subtitle = "ADT platinum endpoint; each point is a shared laboratory feature/statistic",
      x = "Local COMPASS log hazard ratio per SD",
      y = "Federated no-MSK log hazard ratio per SD",
      caption = "Dashed line denotes identical effect sizes. Fits with non-finite/non-positive HR or a p-value that underflowed to zero are excluded."
    ) + theme_fig() + theme(legend.position = "bottom")
}

plot_federated_no_msk_discoveries <- function(d) {
  if (!any(d$usable_effect, na.rm = TRUE)) {
    return(ggplot() +
      annotate("text", x = 0, y = 0, label = "No usable shared tests", color = COLOR_NEUTRAL_INK) +
      theme_void() +
      labs(title = "FDR discovery overlap between local and federated screens",
           subtitle = sprintf("%d shared rows before numerical filtering; see the comparison table for fit validity.", nrow(d))))
  }
  summary <- d %>% filter(.data$usable_effect) %>%
    count(.data$landmark_days, .data$discovery_class, name = "n") %>%
    complete(landmark_days, discovery_class, fill = list(n = 0L)) %>%
    group_by(.data$landmark_days) %>%
    mutate(total = sum(.data$n), proportion = ifelse(.data$total > 0, .data$n / .data$total, NA_real_)) %>%
    ungroup()
  ggplot(summary, aes(factor(.data$landmark_days), .data$n, fill = .data$discovery_class)) +
    geom_col(width = .72) +
    geom_text(aes(label = ifelse(.data$n > 0, .data$n, "")),
              position = position_stack(vjust = .5), size = 3, color = "white") +
    scale_fill_manual(values = c("Neither" = "grey70", "Local only" = "#0072B2",
                                 "Federated only" = "#D55E00", "FDR < 0.05 in both" = "#009E73"),
                      name = NULL) +
    labs(
      title = "FDR discovery overlap between local and federated screens",
      subtitle = "Benjamini–Hochberg q < 0.05 within each landmark/run",
      x = "Landmark after ADT initiation (days)", y = "Number of shared feature/statistic tests",
      caption = "Numerically unusable fits (non-finite/non-positive HR or p-value underflow) are excluded before counting discoveries."
    ) + theme_fig() + theme(legend.position = "bottom")
}

plot_federated_no_msk_top_effects <- function(d, n_per_landmark = 10L) {
  plot_data <- d %>%
    filter(.data$usable_effect, .data$local_fdr | .data$federated_fdr) %>%
    mutate(best_q = pmin(.data$q_value_local, .data$q_value_federated, na.rm = TRUE)) %>%
    group_by(.data$landmark_days) %>%
    slice_min(.data$best_q, n = n_per_landmark, with_ties = FALSE) %>%
    ungroup()
  if (!nrow(plot_data)) return(NULL)
  plot_data <- plot_data %>%
    mutate(label = stringr::str_trunc(sub("__[^_]+$", "", .data$feature), 42))
  # A row-specific factor retains the same feature label in different landmark
  # facets without accidentally merging them.
  plot_data$key <- paste(plot_data$landmark_days, plot_data$label, sep = "__")
  plot_data$key <- factor(plot_data$key, levels = rev(unique(plot_data$key)))
  long <- plot_data %>%
    select(all_of(c("landmark_days", "key", "label", "log_hr_local",
                    "log_hr_federated"))) %>%
    pivot_longer(all_of(c("log_hr_local", "log_hr_federated")),
                 names_to = "run", values_to = "log_hr") %>%
    mutate(run = recode(.data$run, log_hr_local = "Local COMPASS",
                        log_hr_federated = "Federated no-MSK"))
  ggplot(long, aes(.data$log_hr, .data$key, color = .data$run)) +
    geom_vline(xintercept = 0, color = "grey55", linetype = "dashed") +
    geom_point(size = 2) +
    facet_wrap(~landmark_days, scales = "free_y",
               labeller = labeller(landmark_days = function(x) paste0("+", x, " days"))) +
    scale_y_discrete(labels = setNames(as.character(plot_data$label), as.character(plot_data$key))) +
    scale_color_manual(values = c("Local COMPASS" = "#0072B2", "Federated no-MSK" = "#D55E00"), name = NULL) +
    labs(
      title = "Top discoveries: local and federated effect direction",
      subtitle = sprintf("Up to %d features per landmark, selected by the smaller local/federated q-value", n_per_landmark),
      x = "Log hazard ratio per SD", y = NULL,
      caption = "Shown when q < 0.05 in at least one screen; labels omit the terminal feature statistic for readability."
    ) + theme_fig() + theme(legend.position = "bottom", axis.text.y = element_text(size = 7))
}

save_federated_no_msk_panel <- function(plot, path, width, height, dpi, overwrite) {
  capture <- getOption("compass.figure_capture")
  if (is.function(capture)) {
    capture(plot, sub("\\.png$", "", path), width, height, basename(dirname(path)))
    return(invisible(TRUE))
  }
  if (is.null(plot)) return(FALSE)
  complete <- exists("figure_file_complete", mode = "function") && figure_file_complete(path)
  if (!overwrite && complete) { message("Skipped ", path); return(TRUE) }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temp <- tempfile(".federated-", tmpdir = dirname(path), fileext = ".png")
  on.exit(unlink(temp), add = TRUE)
  ggplot2::ggsave(temp, plot, width = width, height = height, dpi = dpi, bg = "white",
                  device = if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png else "png")
  if (!file.rename(temp, path)) stop("Could not publish federated supplement panel: ", path)
  message("Wrote ", path)
  TRUE
}

render_federated_no_msk_supplement <- function(data_root, fig_root, federated_path,
                                                dpi = 200, overwrite = FALSE) {
  d <- load_federated_no_msk_comparison(data_root, federated_path)
  root <- file.path(fig_root, "ADT", "by_figure", "supplements", "federated_no_msk")
  # The comparison table is deliberately aggregate-only and makes every plotted
  # input auditable without reproducing patient-level source data.
  table_path <- file.path(root, "comparison_table", "platinum__all__incl.csv")
  dir.create(dirname(table_path), recursive = TRUE, showWarnings = FALSE)
  if (overwrite || !file.exists(table_path)) readr::write_csv(d, table_path)
  capture <- getOption("compass.figure_table_capture")
  if (is.function(capture)) capture(table_path)
  save_federated_no_msk_panel(plot_federated_no_msk_hr_agreement(d),
    file.path(root, "hr_agreement", "platinum__all__incl.png"), 11, 4.8, dpi, overwrite)
  save_federated_no_msk_panel(plot_federated_no_msk_discoveries(d),
    file.path(root, "discovery_overlap", "platinum__all__incl.png"), 8.5, 5.5, dpi, overwrite)
  save_federated_no_msk_panel(plot_federated_no_msk_top_effects(d),
    file.path(root, "top_discovery_effects", "platinum__all__incl.png"), 12, 8, dpi, overwrite)
  invisible(d)
}
