if (!exists("figure_style_plot", mode="function"))
  source(file.path(dirname(tail(Filter(Negate(is.null), lapply(sys.frames(), function(frame) frame$ofile)), 1)[[1]]),
    "figure_typography.R"), local=TRUE)

# Predictive-checklist cards: one card per landmark, each showing the fitted
# M-of-N checklist and its held-out metrics. Used by the federated no-DFCI
# transfer checklist figures (render_federated_checklist_cards in
# figure_workflow.R); never fits, reads patient data or calls Python.
# Sourced by COMPASS/figures/run_figures.R.
suppressPackageStartupMessages({library(dplyr); library(ggplot2)})

CHECKLIST_MODEL_LABELS <- c(checklist = "Checklist", age_cox = "Age Cox",
  elastic_net = "Elastic-Net Cox", xgboost = "XGBoost")
CHECKLIST_WRAP_WIDTH <- 50L
CHECKLIST_LINE_STEP <- .024
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

# One card's text as a data frame of lines (text, fontface), top to bottom.
# `cell$n_events_test`, when present, adds the held-out event count.
checklist_card_lines <- function(cell, items, metrics, horizon_years, model_labels = CHECKLIST_MODEL_LABELS) {
  lines <- list()
  add <- function(text, face = "plain") for (piece in strsplit(text, "\n", fixed = TRUE)[[1L]])
    lines[[length(lines) + 1L]] <<- data.frame(text = piece, face = face,
      metric_model=NA_character_, metric_auc=NA_character_, metric_brier=NA_character_,
      stringsAsFactors = FALSE)
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
  events <- if (is.null(cell$n_events_test) || !is.finite(cell$n_events_test)) "" else
    sprintf(" (%s events)", format(cell$n_events_test, big.mark = ","))
  add(sprintf("Held-out test, n = %s%s; %d-year horizon",
    if (is.finite(cell$n_test)) format(cell$n_test, big.mark = ",") else "NA", events, horizon_years), "bold")
  width <- max(16L, nchar(model_labels))
  add_metrics <- function(model, auc, brier, face="plain") {
    add(sprintf("%-*s %-18s %s",width,model,auc,brier),face)
    lines[[length(lines)]][c("metric_model","metric_auc","metric_brier")] <<- list(model,auc,brier)
  }
  add_metrics("Model", "AUC (95% CI)", "Brier", "bold")
  for (model in names(model_labels)) {
    row <- metrics[metrics$model == model, , drop = FALSE]
    if (nrow(row) == 0L) next
    add_metrics(model_labels[[model]],checklist_interval(row[1L, ], "auc"),
                checklist_number(row$brier[1L], 3))
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

plot_checklist_cards <- function(cards, title,
    caption = "Unit-weight M-of-N checklists with IPCW-calibrated horizon risk; baselines are fit on the same development patients.") {
  # Arial is proportional: position table cells explicitly instead of relying
  # on spaces to align numeric columns. Keep the original text in sidecars.
  cards$x <- .03
  # Cards without a metrics table (the guideline checklist) keep their text rows as they are.
  if("metric_model" %in% names(cards) && any(!is.na(cards$metric_model))) {
    metric_rows <- !is.na(cards$metric_model)
    table <- cards[metric_rows,,drop=FALSE]
    columns <- c(metric_model=.03,metric_auc=.43,metric_brier=.82)
    cells <- lapply(names(columns),function(column) {
      cell <- table; cell$x <- columns[[column]]; cell$text <- cell[[column]]; cell
    })
    cards <- dplyr::bind_rows(c(list(cards[!metric_rows,,drop=FALSE]),cells))
  }
  ggplot(cards, aes(x = x, y = y, label = text, fontface = face)) +
    geom_text(hjust = 0, vjust = 1, family = FIGURE_FONT, size = FIGURE_FONT_SIZES[["body"]] / ggplot2::.pt) +
    facet_wrap(~ landmark_label, nrow = 1) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    labs(title = title, caption = caption) +
    theme_void(base_family=FIGURE_FONT, base_size=FIGURE_FONT_SIZES[["body"]]) +
    theme(panel.border = element_rect(colour = "grey40", fill = NA, linewidth = .5),
          strip.text = element_blank(), panel.spacing = grid::unit(.8, "lines"),
          plot.title = element_text(face = "bold", size = 12, margin = margin(b = 6)),
          plot.caption = element_text(size = 8, colour = "grey30", hjust = 0, margin = margin(t = 6)),
          plot.margin = margin(8, 8, 8, 8), plot.background = element_rect(fill = "white", colour = NA))
}
