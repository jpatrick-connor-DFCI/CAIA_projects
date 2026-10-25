# Predictive-checklist supplement: one figure per endpoint x feature set, one
# card per landmark, each showing the fitted M-of-N checklist and its held-out
# metrics. Pure consumer of the CSVs written by
# predictive_checklists/collect_supplement.py (run by 06_prep_figure_data); it
# never fits, reads patient data or calls Python. Sourced by 07_figures.R and
# by the standalone predictive_checklists/render_supplement.R.
suppressPackageStartupMessages({library(dplyr); library(ggplot2)})

CHECKLIST_FEATURE_SET_LABELS <- c(labs = "Labs only",
  labs_gleason_somatic = "Labs + Gleason + somatic")
CHECKLIST_ENDPOINT_LABELS <- c(platinum = "Platinum", nepc = "NEPC")
CHECKLIST_MODEL_LABELS <- c(checklist = "Checklist", age_cox = "Age Cox",
  elastic_net = "Elastic-Net Cox", xgboost = "XGBoost")
CHECKLIST_WRAP_WIDTH <- 50L
CHECKLIST_LINE_STEP <- .024
CHECKLIST_TABLES <- c("checklist_cells.csv", "checklist_items_long.csv", "checklist_metrics.csv")

checklist_number <- function(x, digits = 2)
  ifelse(is.finite(x), formatC(x, format = "f", digits = digits), "NA")
checklist_interval <- function(row, metric) {
  value <- row[[metric]]
  lower <- row[[paste0(metric, "_lower")]]
  upper <- row[[paste0(metric, "_upper")]]
  if (is.null(lower) || is.null(upper) || !is.finite(lower) || !is.finite(upper))
    return(checklist_number(value))
  sprintf("%s (%s-%s)", checklist_number(value), checklist_number(lower), checklist_number(upper))
}

load_checklist_tables <- function(data_dir) {
  read <- function(name) readr::read_csv(file.path(data_dir, name), show_col_types = FALSE,
    col_types = readr::cols(.default = readr::col_guess()))
  metrics <- read("checklist_metrics.csv")
  for (column in c("dataset", "status")) if (!column %in% names(metrics)) metrics[[column]] <- NA_character_
  metrics <- dplyr::filter(metrics, dataset == "test", status == "ok")
  list(cells = read("checklist_cells.csv"), items = read("checklist_items_long.csv"),
       metrics = metrics)
}

# One card's text as a data frame of lines (text, fontface), top to bottom.
checklist_card_lines <- function(cell, items, metrics, horizon_years) {
  lines <- list()
  add <- function(text, face = "plain") for (piece in strsplit(text, "\n", fixed = TRUE)[[1L]])
    lines[[length(lines) + 1L]] <<- data.frame(text = piece, face = face, stringsAsFactors = FALSE)
  gap <- function() add(" ")
  add(sprintf("Landmark ADT +%d days", cell$landmark_days), "bold")
  gap()
  if (cell$status != "ok") {
    add("No checklist fit", "italic")
    add(paste(strwrap(sprintf("Reason: %s", cell$reason), width = CHECKLIST_WRAP_WIDTH), collapse = "\n"))
  } else {
    add(sprintf("Flag when at least %d of %d items are checked.", cell$threshold, cell$n_items), "italic")
    gap()
    for (i in order(items$item_order))
      add(paste(strwrap(sprintf("[ ] %s", items$name[i]), width = CHECKLIST_WRAP_WIDTH, exdent = 4),
                collapse = "\n"))
  }
  gap()
  add(sprintf("Held-out test, n = %s; %d-year horizon",
    if (is.finite(cell$n_test)) format(cell$n_test, big.mark = ",") else "NA", horizon_years), "bold")
  add(sprintf("%-16s %-18s %s", "Model", "AUC (95% CI)", "Brier"), "bold")
  for (model in names(CHECKLIST_MODEL_LABELS)) {
    row <- metrics[metrics$model == model, , drop = FALSE]
    if (nrow(row) == 0L) next
    add(sprintf("%-16s %-18s %s", CHECKLIST_MODEL_LABELS[[model]], checklist_interval(row[1L, ], "auc"),
                checklist_number(row$brier[1L], 3)))
  }
  checklist <- metrics[metrics$model == "checklist", , drop = FALSE]
  if (cell$status == "ok" && nrow(checklist) > 0L) {
    gap()
    add(sprintf("Checklist at flag: sens %s, spec %s\n                   PPV %s, NPV %s",
      checklist_number(checklist$sensitivity[1L]), checklist_number(checklist$specificity[1L]),
      checklist_number(checklist$ppv[1L]), checklist_number(checklist$npv[1L])))
  }
  out <- do.call(rbind, lines)
  out$y <- 1 - .02 - (seq_len(nrow(out)) - 1L) * CHECKLIST_LINE_STEP
  out
}

plot_checklist_cards <- function(cards, title) {
  ggplot(cards, aes(x = .03, y = y, label = text, fontface = face)) +
    geom_text(hjust = 0, vjust = 1, family = "mono", size = 3) +
    facet_wrap(~ landmark_label, nrow = 1) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    labs(title = title,
         caption = "Unit-weight M-of-N checklists with IPCW-calibrated horizon risk; baselines are fit on the same development patients.") +
    theme_void() +
    theme(panel.border = element_rect(colour = "grey40", fill = NA),
          strip.text = element_blank(), panel.spacing = grid::unit(.8, "lines"),
          plot.title = element_text(face = "bold", size = 12, margin = margin(b = 6)),
          plot.caption = element_text(size = 8, colour = "grey30", hjust = 0, margin = margin(t = 6)),
          plot.margin = margin(8, 8, 8, 8), plot.background = element_rect(fill = "white", colour = NA))
}

# `save_panel(plot, path, width, height, dpi, overwrite)` writes (or, inside the
# cached workflow, captures) one PNG; `sidecar(path)` maps a table path to the
# data folder. Returns the cards invisibly.
render_checklist_supplement <- function(data_dir, out_dir, dpi, overwrite, save_panel,
                                        sidecar = function(path) path, capture_table = NULL) {
  if (!all(file.exists(file.path(data_dir, CHECKLIST_TABLES)))) {
    warning("Checklist supplement skipped; missing tables in ", data_dir)
    return(invisible(NULL))
  }
  tables <- load_checklist_tables(data_dir)
  cells <- tables$cells
  if (!nrow(cells)) return(invisible(NULL))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  for (name in CHECKLIST_TABLES) {
    path <- sidecar(file.path(out_dir, sub("^checklist_", "figure4s_checklist_", name)))
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    file.copy(file.path(data_dir, name), path, overwrite = TRUE)
    if (is.function(capture_table)) capture_table(path)
  }
  all_cards <- list()
  for (fs in unique(cells$feature_set)) for (ep in unique(cells$endpoint[cells$feature_set == fs])) {
    sub <- cells %>% filter(feature_set == fs, endpoint == ep) %>% arrange(landmark_days)
    horizon_years <- round(sub$horizon_days[1L] / 365)
    cell_rows <- function(d, cell) d %>% filter(feature_set == fs, endpoint == ep,
                                                landmark_days == cell$landmark_days)
    cards <- do.call(rbind, lapply(seq_len(nrow(sub)), function(k) {
      cell <- sub[k, , drop = FALSE]
      lines <- checklist_card_lines(cell, cell_rows(tables$items, cell), cell_rows(tables$metrics, cell), horizon_years)
      lines$landmark_label <- factor(sprintf("+%d days", cell$landmark_days),
        levels = sprintf("+%d days", sub$landmark_days))
      lines
    }))
    title <- sprintf("%s predictive checklists, %d-year recurrence: %s",
      CHECKLIST_ENDPOINT_LABELS[[ep]], horizon_years, CHECKLIST_FEATURE_SET_LABELS[[fs]])
    stem <- sprintf("figure4s_checklist_%s", fs)
    save_panel(plot_checklist_cards(cards, title), file.path(out_dir, paste0(stem, "__", ep, ".png")),
               5.2 * nrow(sub), 7.5, dpi, overwrite)
    all_cards[[paste(fs, ep)]] <- cards
  }
  invisible(all_cards)
}
