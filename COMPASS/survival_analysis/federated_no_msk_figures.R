# Federated-only volcano panels, using the exact local category-colored renderer.
# No local result joins, agreement panels, or discovery-overlap panels.
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

load_federated_no_msk_volcano <- function(path) {
  d <- readr::read_csv(path, show_col_types = FALSE)
  needed <- c("landmark_days", "endpoint", "feature", "ci_lower", "ci_upper", "p_value", "q_value")
  missing <- setdiff(needed, names(d))
  if (length(missing)) stop("Federated volcano input is missing: ", paste(missing, collapse = ", "))
  if (!"coef_feature" %in% names(d)) {
    if (!"hazard_ratio_per_sd" %in% names(d)) stop("Federated volcano needs coef_feature or hazard_ratio_per_sd")
    d$coef_feature <- suppressWarnings(log(as.numeric(d$hazard_ratio_per_sd)))
  }
  for (column in c("coef_feature", "ci_lower", "ci_upper", "p_value", "q_value", "landmark_days"))
    d[[column]] <- suppressWarnings(as.numeric(d[[column]]))
  d <- d %>% filter(tolower(endpoint) == "platinum", landmark_days %in% c(0, 90, 180))
  if (anyDuplicated(d[c("landmark_days", "feature")]))
    stop("Duplicate feature/landmark rows in federated volcano input")
  if (!"lab_name" %in% names(d)) d$lab_name <- sub("__[^_]+$", "", d$feature)
  if (!"feature_stat" %in% names(d)) d$feature_stat <- sub("^.*__", "", d$feature)
  d$raw_lab_name <- d$lab_name
  d$lab_name <- federated_lab_label(d$lab_name)
  # Same thresholds as the local univariate volcano; additionally reject
  # malformed/nonfinite input explicitly. A p-value of zero is kept and capped
  # by the shared renderer, not treated as an automatic failed fit.
  d %>% mutate(eligible_volcano =
    !is.na(lab_name) & !lab_name %in% DROP &
    is.finite(coef_feature) & abs(coef_feature) <= 4 &
    is.finite(ci_lower) & ci_lower > 0 & is.finite(ci_upper) & ci_upper >= ci_lower &
    ci_upper / ci_lower < 100 &
    is.finite(p_value) & p_value >= 0 & p_value <= 1 &
    is.finite(q_value) & q_value >= 0 & q_value <= 1)
}

plot_federated_no_msk_volcano <- function(d, landmark) {
  at_landmark <- filter(d, landmark_days == landmark)
  sub <- filter(at_landmark, eligible_volcano)
  title <- sprintf("Federated no-MSK | ADT platinum | +%d days", landmark)
  if (!nrow(sub)) return(ggplot() +
    annotate("text", x = 0, y = 0, label = "No usable fits at this landmark", color = COLOR_NEUTRAL_INK) +
    theme_void() + labs(title = title, caption = sprintf("%d input rows; none pass local volcano filters.", nrow(at_landmark))))
  p <- plot_volcano_panel(sub, title)
  p + labs(caption = paste(p$labels$caption,
    sprintf("%d input rows excluded; local filters: |log HR| ≤ 4 and CI ratio < 100. Triangles indicate −log10(p) > 30.",
            nrow(at_landmark)-nrow(sub)), sep = "\n"))
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
  d <- load_federated_no_msk_volcano(federated_path)
  root <- file.path(fig_root, "ADT", "by_figure", "supplements", "federated_no_msk")
  table_path <- file.path(root, "volcano_input", "platinum__all__incl.csv")
  dir.create(dirname(table_path), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(d, table_path)
  capture <- getOption("compass.figure_table_capture")
  if (is.function(capture)) capture(table_path)
  for (landmark in c(0L, 90L, 180L)) {
    p <- plot_federated_no_msk_volcano(d, landmark)
    save_federated_no_msk_panel(p,
      file.path(root, paste0("volcano_landmark", landmark), "platinum__all__incl.png"),
      9, 7, dpi, overwrite)
  }
  invisible(d)
}
