# Federated-only PSA/testosterone forests and site cohort counts. Significance uses the supplied
# p/q values, never an FDR recalculation on this selected subset.
federated_lab_label <- function(raw) {
  short <- gsub("_", " ", sub("__.*$", "", raw))
  aliases <- c("Alanine aminotransferase" = "ALT", "Aspartate aminotransferase" = "AST",
    "Bilirubin direct" = "Direct bilirubin", "Bilirubin total" = "Total bilirubin",
    "Carbon dioxide" = "CO2", "Erythrocytes" = "RBC", "Leukocytes" = "WBC",
    "Prostate specific Ag" = "PSA", "Protein" = "Total protein",
    "Prothrombin time" = "PT", "Thyrotropin" = "TSH", "Urea nitrogen" = "BUN",
    "Basophils" = "Basophils absolute", "Eosinophils" = "Eosinophils absolute",
    "Lymphocytes" = "Lymphocytes absolute", "Monocytes" = "Monocytes absolute",
    "Neutrophils" = "Neutrophils absolute")
  mapped <- unname(aliases[short])
  short[!is.na(mapped)] <- mapped[!is.na(mapped)]
  short[grepl("^Erythrocyte__DistWidth", raw)] <- "RDW"
  short
}

load_federated_no_msk_forest <- function(path) {
  d <- readr::read_csv(path, show_col_types = FALSE)
  needed <- c("landmark_days", "endpoint", "feature", "ci_lower", "ci_upper", "p_value", "q_value")
  missing <- setdiff(needed, names(d))
  if (length(missing)) stop("Federated forest input is missing: ", paste(missing, collapse = ", "))
  if (!"hazard_ratio_per_sd" %in% names(d)) {
    if (!"coef_feature" %in% names(d)) stop("Federated forest needs hazard_ratio_per_sd or coef_feature")
    d$hazard_ratio_per_sd <- suppressWarnings(exp(as.numeric(d$coef_feature)))
  }
  for (column in c("hazard_ratio_per_sd", "ci_lower", "ci_upper", "p_value", "q_value", "landmark_days"))
    d[[column]] <- suppressWarnings(as.numeric(d[[column]]))
  if (!"lab_name" %in% names(d)) d$lab_name <- sub("__[^_]+$", "", d$feature)
  if (!"feature_stat" %in% names(d)) d$feature_stat <- sub("^.*__", "", d$feature)
  d$raw_lab_name <- d$lab_name
  d$lab_name <- federated_lab_label(d$lab_name)
  d <- d %>% mutate(feature_stat = tolower(trimws(feature_stat))) %>%
    filter(tolower(endpoint) == "platinum", landmark_days %in% c(0, 90, 180),
           lab_name %in% c("PSA", "Testosterone"), feature_stat %in% c("mean", "min", "max", "last"))
  if (anyDuplicated(d[c("landmark_days", "lab_name", "feature_stat")]))
    stop("Duplicate analyte/statistic/landmark rows in federated forest input")
  d %>% mutate(
    nominal_significant = if_else(is.finite(p_value) & p_value >= 0 & p_value <= 1, p_value < .05, NA),
    fdr_significant = if_else(is.finite(q_value) & q_value >= 0 & q_value <= 1, q_value < .05, NA),
    significance = case_when(
      fdr_significant ~ "FDR q < 0.05",
      nominal_significant & !is.na(fdr_significant) ~ "Nominal p < 0.05 only",
      !is.na(nominal_significant) & !is.na(fdr_significant) ~ "Not significant",
      TRUE ~ "Significance unavailable"),
    valid_estimate = is.finite(hazard_ratio_per_sd) & hazard_ratio_per_sd > 0,
    valid_ci = is.finite(ci_lower) & ci_lower > 0 & is.finite(ci_upper) &
      ci_upper >= ci_lower & ci_lower <= hazard_ratio_per_sd & ci_upper >= hazard_ratio_per_sd)
}

plot_federated_no_msk_forest <- function(d, landmark) {
  stats <- c("mean", "min", "max", "last")
  stat_labels <- c(mean = "Mean", min = "Minimum", max = "Maximum", last = "Last")
  sub <- filter(d, landmark_days == landmark) %>%
    complete(lab_name = c("PSA", "Testosterone"), feature_stat = stats) %>%
    mutate(lab_name = factor(lab_name, levels = c("PSA", "Testosterone")),
           row_key = paste(lab_name, feature_stat, sep = "__"),
           significance = coalesce(significance, "Significance unavailable"),
           valid_estimate = coalesce(valid_estimate, FALSE), valid_ci = coalesce(valid_ci, FALSE))
  sub$row_key <- factor(sub$row_key, levels = rev(as.vector(t(outer(
    c("PSA", "Testosterone"), stats, paste, sep = "__")))))
  fmt <- function(x) ifelse(is.finite(x) & x >= 0 & x <= 1, format.pval(x, digits = 2, eps = 1e-300), "NA")
  row_labels <- setNames(paste0(stat_labels[sub$feature_stat], "\np=", fmt(sub$p_value), "; q=", fmt(sub$q_value),
    ifelse(!sub$valid_estimate, "\nEstimate unavailable", ifelse(!sub$valid_ci, "\nCI unavailable", ""))), sub$row_key)
  limits <- range(c(1, sub$hazard_ratio_per_sd[sub$valid_estimate],
    sub$ci_lower[sub$valid_estimate & sub$valid_ci], sub$ci_upper[sub$valid_estimate & sub$valid_ci]), finite = TRUE)
  limits <- exp(log(limits) + c(-1, 1)*max(diff(log(limits))*.12, .1))
  ggplot(sub, aes(hazard_ratio_per_sd, row_key, color = lab_name)) +
    geom_vline(xintercept = 1, color = "grey55", linetype = "dashed", linewidth = .5) +
    geom_blank(aes(x = 1)) +
    geom_errorbar(data = filter(sub, valid_estimate, valid_ci), aes(xmin = ci_lower, xmax = ci_upper),
                  orientation = "y", width = .17, linewidth = .8) +
    geom_point(data = filter(sub, valid_estimate), aes(shape = significance), size = 3.5, stroke = 1,
               show.legend = TRUE) +
    scale_shape_manual(values = c("Not significant" = 1, "Nominal p < 0.05 only" = 16,
      "FDR q < 0.05" = 18, "Significance unavailable" = 4),
      limits = c("Not significant", "Nominal p < 0.05 only", "FDR q < 0.05", "Significance unavailable"),
      breaks = c("Not significant", "Nominal p < 0.05 only", "FDR q < 0.05",
                 if (any(sub$significance == "Significance unavailable")) "Significance unavailable"),
      drop = FALSE, name = NULL) +
    scale_color_manual(values = c(PSA = "#0072B2", Testosterone = "#D55E00"), guide = "none") +
    scale_x_log10(limits = limits, labels = scales::label_number(accuracy = .01)) +
    scale_y_discrete(labels = row_labels, expand = expansion(add = .7)) +
    facet_wrap(~lab_name, nrow = 1, scales = "free_y", drop = FALSE) +
    labs(x = "Hazard ratio per SD (95% CI; log scale)", y = NULL,
      title = sprintf("Federated no-MSK | ADT platinum | +%d days", landmark),
      subtitle = "PSA and testosterone: mean, minimum, maximum, and last value",
      caption = paste("Nominal significance: p < 0.05; FDR significance: supplied q < 0.05.",
        "Delta and observation-count features excluded. q-values are not recomputed for this subset.",
        "Open circle: not significant; filled circle: nominal only; diamond: FDR significant.", sep = "\n")) +
    theme_classic(base_size = 11) +
    theme(strip.background = element_blank(), strip.text = element_text(face = "bold", size = 12),
      axis.text.y = element_text(size = 10), legend.position = "bottom",
      plot.caption = element_text(hjust = 0, size = 9), plot.title.position = "plot",
      plot.caption.position = "plot", panel.spacing = grid::unit(1.5, "lines"),
      plot.margin = margin(12, 16, 12, 12))
}

load_federated_no_msk_sites <- function(federated_path) {
  path <- file.path(dirname(federated_path), "nvflare_within_site_cox_univariate",
                    "cox_within_site_all_sites_cohort.csv")
  if (!file.exists(path)) {
    warning("Federated site counts unavailable; missing: ", path)
    return(NULL)
  }
  d <- readr::read_csv(path, show_col_types = FALSE)
  needed <- c("site_name", "analysis_label", "landmark_days", "n_patients", "n_events")
  if (length(setdiff(needed, names(d)))) stop("Federated site counts missing required columns: ",
    paste(setdiff(needed, names(d)), collapse = ", "))
  # This bundle's ADT cohort counts describe the platinum endpoint. If an
  # endpoint column is supplied by a newer writer, select it explicitly.
  if ("endpoint" %in% names(d)) d <- filter(d, tolower(endpoint) == "platinum")
  d <- filter(d, tolower(analysis_label) == "adt", landmark_days %in% c(0, 90, 180))
  if (!nrow(d)) stop("No ADT platinum site counts at landmarks 0/90/180")
  if (any(is.na(d$site_name) | !nzchar(trimws(d$site_name)))) stop("Missing federated site name")
  if (any(grepl("msk|sloan", d$site_name, ignore.case = TRUE)))
    stop("MSK site found in no-MSK cohort counts")
  if (anyDuplicated(d[c("site_name", "landmark_days")])) stop("Duplicate federated site/landmark counts")
  for (column in c("n_patients", "n_events")) {
    d[[column]] <- suppressWarnings(as.numeric(d[[column]]))
    if (any(!is.finite(d[[column]]) | d[[column]] < 0 | d[[column]] != floor(d[[column]])))
      stop("Invalid federated site counts: ", column)
  }
  if (any(d$n_events > d$n_patients)) stop("Federated events exceed patients")
  labels <- c(dana_farber_caia_1_1 = "Dana-Farber", fred_hutch_caia_1_1 = "Fred Hutch",
              jhu_caia_1_1 = "Johns Hopkins")
  d %>% mutate(endpoint = "platinum", site = coalesce(unname(labels[site_name]), site_name),
    event_incidence_pct = if_else(n_patients > 0, 100 * n_events / n_patients, NA_real_)) %>%
    arrange(landmark_days, site)
}

plot_federated_no_msk_sites <- function(d) {
  # Missing site/landmark cells are unavailable, never zero-sized cohorts.
  d <- d %>% select(site, landmark_days, n_patients, n_events, event_incidence_pct) %>%
    complete(site, landmark_days = c(0, 90, 180)) %>%
    mutate(site = factor(site, levels = rev(sort(unique(site)))),
      landmark = factor(landmark_days, levels = c(0, 90, 180), labels = c("0 days", "+90 days", "+180 days")))
  number <- function(x) format(x, big.mark = ",", scientific = FALSE, trim = TRUE)
  counts <- d %>% mutate(panel = "Cohort size", value = n_patients,
    label = if_else(is.na(n_patients), "Unavailable",
      paste0(number(n_events), "/", number(n_patients))))
  incidence <- d %>% mutate(panel = "Observed event incidence", value = event_incidence_pct,
    label = if_else(is.finite(event_incidence_pct), sprintf("%.2f%%", event_incidence_pct), "Unavailable"))
  plot_data <- bind_rows(counts, incidence) %>% mutate(
    panel = factor(panel, levels = c("Cohort size", "Observed event incidence")))
  ggplot(plot_data, aes(value, site)) +
    geom_col(fill = "#0072B2", width = .65, na.rm = TRUE) +
    geom_text(aes(x = coalesce(value, 0), label = label), hjust = -.08, size = 3.4) +
    facet_grid(landmark ~ panel, scales = "free_x") +
    scale_x_continuous(expand = expansion(mult = c(0, .32))) +
    labs(x = "Patients (left) / observed event incidence, % (right)", y = NULL,
      title = "Federated no-MSK | ADT platinum | Cohort counts and event incidence by site",
      subtitle = "Count labels: events / analyzed patients; separate cohort at each landmark",
      caption = paste("Observed incidence = events / analyzed patients over available follow-up after each landmark.",
        "Not a fixed-horizon cumulative incidence estimate. Patients can recur across landmarks; do not sum rows.",
        "Source: cox_within_site_all_sites_cohort.csv; denominators are not feature-specific complete cases.", sep = "\n")) +
    theme_classic(base_size = 11) +
    theme(strip.background = element_blank(), strip.text = element_text(face = "bold"),
      plot.title.position = "plot", plot.caption.position = "plot",
      plot.caption = element_text(hjust = 0, size = 9), panel.spacing = grid::unit(1.3, "lines"),
      plot.margin = margin(12, 16, 12, 12))
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
  d <- load_federated_no_msk_forest(federated_path)
  root <- file.path(fig_root, "ADT", "by_figure", "federated_no_msk")
  table_path <- file.path(root, "forest_input", "platinum__all__incl.csv")
  dir.create(dirname(table_path), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(d, table_path)
  capture <- getOption("compass.figure_table_capture")
  if (is.function(capture)) capture(table_path)
  for (landmark in c(0L, 90L, 180L)) {
    p <- plot_federated_no_msk_forest(d, landmark)
    save_federated_no_msk_panel(p,
      file.path(root, paste0("psa_testosterone_forest_landmark", landmark), "platinum__all__incl.png"),
      12, 7, dpi, overwrite)
  }
  sites <- load_federated_no_msk_sites(federated_path)
  if (!is.null(sites)) {
    site_path <- file.path(root, "site_cohort_counts", "platinum__all__incl.csv")
    dir.create(dirname(site_path), recursive = TRUE, showWarnings = FALSE)
    readr::write_csv(sites, site_path)
    if (is.function(capture)) capture(site_path)
    save_federated_no_msk_panel(plot_federated_no_msk_sites(sites),
      file.path(root, "site_cohort_counts", "platinum__all__incl.png"), 12, 8, dpi, overwrite)
  }
  invisible(d)
}
