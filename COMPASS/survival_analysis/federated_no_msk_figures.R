# Federated-only PSA/testosterone forests. Significance uses the supplied
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
  invisible(d)
}
