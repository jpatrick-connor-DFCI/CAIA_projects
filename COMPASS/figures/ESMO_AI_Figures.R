#!/usr/bin/env Rscript
# Figures for Results 1-4 of COMPASS_ESMO_AI_proposed (9 November 2026).
# Run from any directory: Rscript /path/to/COMPASS/figures/ESMO_AI_Figures.R
# Or use source("COMPASS/figures/ESMO_AI_Figures.R") / RStudio's Source button.
# To load functions without running, set options(compass.esmo.autorun = FALSE).
# See --help and README.md for paths. Uses original analysis exports, never
# poster screenshots, PDF-derived estimates, or hard-coded manuscript results.
# Patient-level inputs are used in memory; only aggregate tables are exported.

esmo_script_dir <- local({
  paths <- Filter(function(x) is.character(x) && length(x) == 1L && nzchar(x),
    lapply(sys.frames(), function(x) x$ofile))
  path <- if (length(paths)) tail(paths, 1L)[[1L]] else
    sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L])
  if (is.na(path)) stop("Cannot locate ESMO_AI_Figures.R; source it by filename.")
  # R can encode spaces in --file as ~+~, including in this repository's path.
  if (!file.exists(path)) path <- gsub("~+~", " ", path, fixed = TRUE)
  dirname(normalizePath(path, mustWork = TRUE))
})

esmo_load_helpers <- function() {
  target <- environment(esmo_load_helpers)
  source(file.path(esmo_script_dir, "COMPASS_generate_figures_pipeline.R"), local = target)
  source(file.path(esmo_script_dir, "figure_workflow.R"), local = target)
}

esmo_require <- function(d, what) {
  if (is.null(d) || !nrow(d)) stop("Missing ", what, ". See ESMO_AI_Figures.R --help.", call. = FALSE)
  d
}

esmo_unique <- function(d, keys, what) {
  if (anyDuplicated(d[keys])) stop("Duplicate ", what, ": ", paste(keys, collapse = "/"), call. = FALSE)
  invisible(d)
}

esmo_grid <- function(d, keys, expected, what) {
  esmo_unique(d, keys, what)
  if (nrow(d) != nrow(expected) || nrow(dplyr::anti_join(expected, d, by = keys)))
    stop("Incomplete ", what, "; expected every ", paste(keys, collapse = " × "), call. = FALSE)
  invisible(d)
}

# The same day-0 prediction risk set and labels as manuscript Figure 2.
esmo_llm_data <- function(patients, labels, manual) {
  classes <- c("conventional", "avpc", "nepc")
  for (item in list(patients, labels, manual)) {
    if (anyNA(item$DFCI_MRN)) stop("Missing patient ID in LLM inputs")
    esmo_unique(item, "DFCI_MRN", "LLM input patient")
  }
  if (anyNA(patients$PLATINUM) || any(!patients$PLATINUM %in% c(0, 1)))
    stop("Day-0 PLATINUM must be binary and nonmissing")
  labeled <- dplyr::inner_join(labels, patients, by = "DFCI_MRN")
  classified <- esmo_require(dplyr::filter(labeled, primary_label %in% classes), "classified day-0 patients")
  reviewed <- dplyr::inner_join(manual, classified, by = "DFCI_MRN")
  metrics <- multiclass_metrics(reviewed$manual_label, reviewed$primary_label, classes)
  if (!metrics$N) stop("No evaluable manual-review patients in the day-0 cohort")
  landscape <- classified %>% mutate(primary_label = as.character(primary_label)) %>%
    count(primary_label, PLATINUM, name = "n") %>%
    tidyr::complete(primary_label = classes, PLATINUM = c(0, 1), fill = list(n = 0L)) %>%
    group_by(PLATINUM) %>% mutate(fraction = n / sum(n)) %>% ungroup()
  aggressive <- classified$primary_label %in% c("avpc", "nepc")
  positive <- classified$PLATINUM == 1
  ct <- matrix(c(sum(aggressive & positive), sum(aggressive & !positive),
    sum(!aggressive & positive), sum(!aggressive & !positive)), nrow = 2, byrow = TRUE)
  if (any(rowSums(ct) == 0) || any(colSums(ct) == 0))
    stop("Platinum enrichment requires both phenotype groups and both platinum outcomes")
  test <- stats::fisher.test(ct, alternative = "greater")
  intervals <- t(vapply(1:2, function(i) wilson_ci(ct[i, 1], sum(ct[i, ])), numeric(3)))
  enrichment <- tibble(group = c("Aggressive\n(AVPC + NEPC)", "Conventional"),
    n = rowSums(ct), events = ct[, 1], probability = intervals[, 1],
    lo = intervals[, 2], hi = intervals[, 3], odds_ratio = unname(test$estimate),
    fisher_p = test$p.value)
  list(confusion = metrics$cm, landscape = landscape, enrichment = enrichment,
    audit = tibble(n_day0 = nrow(patients), n_labeled = nrow(labeled),
      n_classified = nrow(classified), n_reviewed = metrics$N, agreement = metrics$Accuracy))
}

esmo_result1 <- function(d) {
  classes <- c("conventional", "avpc", "nepc")
  # Figure labels abbreviate "Conventional" (defined in the poster caption);
  # exported tables keep the full name.
  class_labels <- c("Conv.", "AVPC", "NEPC")
  cm <- d$confusion %>% group_by(truth) %>%
    mutate(row_fraction = if (sum(n) > 0) n / sum(n) else 0) %>% ungroup() %>%
    mutate(truth = factor(truth, levels = rev(classes), labels = rev(class_labels)),
      pred = factor(pred, levels = classes, labels = class_labels))
  p1 <- ggplot(cm, aes(pred, truth, fill = row_fraction)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = scales::comma(n), color = row_fraction > .5), size = 5, fontface = "bold") +
    scale_fill_gradient(low = "#eaf1fb", high = COLOR_PLATINUM_POS, limits = c(0, 1), guide = "none") +
    scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#0b0b0b"), guide = "none") +
    scale_x_discrete(position = "bottom") +
    labs(x = "LLM label (prediction)", y = "Manual annotation (truth)") + theme_fig() +
    theme(axis.text.x = element_text(angle = 0, hjust = .5, margin = margin(t = 2, b = 5)))
  landscape <- d$landscape %>% mutate(primary_label = factor(primary_label, levels = classes, labels = class_labels),
    platinum = factor(PLATINUM, levels = c(1, 0), labels = c("Platinum+", "Platinum−")))
  counts <- vapply(c(1, 0), function(x) sum(landscape$n[landscape$PLATINUM == x]), numeric(1))
  p2 <- ggplot(landscape, aes(primary_label, fraction, fill = platinum)) +
    geom_col(position = position_dodge(.8), width = .72) +
    scale_fill_manual(values = c(COLOR_PLATINUM_POS, COLOR_PLATINUM_NEG), name = NULL,
      labels = sprintf(c("Platinum+ (n=%s)", "Platinum− (n=%s)"), scales::comma(counts))) +
    scale_x_discrete(expand = expansion(add = .45)) +
    # Bars start on the x axis, matching the enrichment panel; the headroom
    # keeps the top-right legend clear of the tallest bar.
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .25), labels = scales::label_percent(),
      expand = expansion(mult = c(0, .04))) +
    labs(x = NULL, y = "Fraction within platinum group") + theme_fig() +
    guides(fill = guide_legend(ncol = 1)) +
    theme(legend.margin = margin(0, 0, 0, 0),
      legend.background = element_rect(fill = "white", color = NA))
  enrichment <- d$enrichment %>% mutate(group = sub("^Conventional$", "Conv.", group),
    group = factor(group, levels = group))
  p3 <- ggplot(enrichment, aes(group, probability, fill = group)) +
    geom_col(width = .55) + geom_errorbar(aes(ymin = lo, ymax = hi), width = .15) +
    scale_fill_manual(values = c(COLOR_PLATINUM_POS, "#9a9890"), guide = "none") +
    scale_x_discrete(expand = expansion(add = .5)) +
    scale_y_continuous(labels = scales::label_percent(), expand = expansion(mult = c(0, .24))) +
    annotate("text", x = 1.5, y = Inf, vjust = 1.3, size = 4.2, family = FIGURE_FONT,
      label = sprintf("OR = %.1f (p=%.1e)", enrichment$odds_ratio[1], enrichment$fisher_p[1])) +
    labs(x = NULL, y = "Platinum-positive fraction") + theme_fig()
  list(plots = list(confusion_matrix = p1, subtype_fractions = p2, platinum_enrichment = p3),
    # Keep room for the full confusion-matrix labels; use narrower bar panels,
    # especially c, and minimal outer margins to bring the panels together.
    # Export aligns the plotting areas; b's legend occupies its upper right.
    # Drawn at its poster size (column A, 10.35 in wide). The height keeps
    # the confusion-matrix panel square; Results 2 takes the rest of column A.
    tables = d, layout = matrix(1:3, nrow = 1), col_widths = c(4, 4.05, 3.2), width = 10.35, height = 3.90,
    plot_theme = theme(plot.margin = margin(4, 2, 4, 2),
      axis.title.x = element_text(margin = margin(t = 2)),
      axis.title.y = element_text(margin = margin(r = 2)),
      axis.text.y = element_text(margin = margin(r = 2))),
    # Space under the panel letters drops the panels so the long y titles
    # (which run above the panel tops) clear the letters.
    composite_theme = theme(plot.tag = element_text(margin = margin(b = 12))),
    compact_x_titles = TRUE,
    # b's legend sits in the top-right corner, centred near the 95% gridline
    # (y range is 0-104%), where only the short AVPC/NEPC bars are below it.
    legend_inside = list(subtype_fractions = list(c(1, .95 / 1.04), c(1, .5))),
    caption = paste("Results 1. Day-0 ADT prediction cohort intersected with LLM subtype labels.",
      "Manual truth is on the y axis; tile color is row-normalized and labels are counts.",
      "Subtype fractions are conditional on platinum status. Enrichment compares AVPC/NEPC with conventional labels;",
      "intervals are 95% Wilson intervals and the Fisher test is one-sided (greater)."))
}

esmo_volcano <- function(d, title) {
  p <- plot_volcano_panel(d, title, hr2_reference = FALSE)
  # The source plot targets a much wider manuscript panel. Scale its marks and
  # labels for these five-inch panels, retaining every selected label and point.
  p$layers <- lapply(p$layers, function(layer) {
    copy <- ggplot2::ggproto(NULL, layer)
    if (inherits(layer$geom, "GeomTextRepel")) {
      labels <- layer$data
      labels$repel_label <- gsub("Alkaline phosphatase", "ALP", labels$repel_label, fixed = TRUE)
      labels$repel_label <- gsub(" absolute", " abs.", labels$repel_label, fixed = TRUE)
      # A borderless white backing keeps leader lines from crossing the larger
      # text. Keep labels above the q threshold and all points in the repulsion.
      # An iteration budget, rather than wall time, makes placement repeatable.
      repel <- ggrepel::geom_label_repel(data = labels, mapping = layer$mapping,
        size = 4.2, family = FIGURE_FONT, fill = "white", label.size = 0,
        label.padding = grid::unit(.05, "lines"), label.r = grid::unit(0, "lines"),
        point.size = 3.2, box.padding = .5, point.padding = .6,
        force = 5, force_pull = .1, nudge_x = .07, nudge_y = 1,
        ylim = c(-log10(.05) + 1.5, NA), max.overlaps = Inf,
        max.time = Inf, max.iter = 50000, seed = 0,
        segment.color = "#95a5a6", segment.size = .3, min.segment.length = 0,
        show.legend = FALSE)
      # Recent ggrepel uses linewidth; the cluster's older release uses label.size.
      if ("linewidth" %in% names(repel$geom$default_aes)) repel$aes_params$linewidth <- 0
      return(repel)
    }
    if (inherits(layer$geom, "GeomPoint")) {
      copy$aes_params$stroke <- .4
      if (!is.null(layer$aes_params$size))
        copy$aes_params$size <- if (identical(layer$aes_params$shape, 24)) 3.4 else 1.8
    }
    copy
  })
  suppressMessages(p + scale_size_manual(values = c(`TRUE` = 3.2, `FALSE` = 2.5), guide = "none")) +
    guides(fill = guide_legend(nrow = 2, byrow = TRUE,
      override.aes = list(shape = 21, size = 3.2, alpha = 1)))
}

# Add thousands separators to standalone integers in figure text (e.g. the
# shared KM legend's "n=1234; events=567" labels) without touching decimals.
esmo_comma_text <- function(x) {
  m <- gregexpr("(?<![0-9.,])[0-9]{4,}(?![0-9.,])", x, perl = TRUE)
  regmatches(x, m) <- lapply(regmatches(x, m), function(v) scales::comma(as.numeric(v)))
  x
}

esmo_result2 <- function(volcano, fig) {
  plots <- lapply(c(0, 180), function(lm) {
    d <- esmo_require(dplyr::filter(volcano, landmark_days == lm), paste("Cox associations at day", lm))
    esmo_volcano(d, paste("ADT day", lm))
  })
  # Apply the same scales to both selected landmarks. Both retain labeled axes.
  xlim <- range(unlist(lapply(plots, function(p) p$coordinates$limits$x)))
  ylim <- range(unlist(lapply(plots, function(p) p$coordinates$limits$y)))
  plots <- lapply(plots, function(p) p + coord_cartesian(xlim = xlim, ylim = ylim))
  for (lab in c("PSA", "Testosterone")) {
    summary <- fig$km_summary %>% filter(scope == "pooled", .data$lab == .env$lab,
      landmark_days == 180, cutoff_scope == "pooled")
    if (nrow(summary) != 1L || !identical(summary$status, "ok") ||
        !is.finite(summary$extreme_fraction) || summary$extreme_fraction <= 0 || summary$extreme_fraction >= .5)
      stop("Results 2 requires pooled extreme-group ", lab, " KM exports at day 180")
    p <- plot_federated_lab_km(fig, lab, 180)
    if (is.null(p)) stop("Missing day-180 ", lab, " survival curves")
    # The shared KM helper prints raw patient/event counts in its legend.
    km_colour <- p$scales$get_scales("colour")
    km_colour$labels <- setNames(esmo_comma_text(km_colour$labels), names(km_colour$labels))
    plots[[length(plots) + 1L]] <- p +
      annotate("text", x = -Inf, y = -Inf, label = p$labels$subtitle,
        hjust = -.05, vjust = -.6, size = 4, family = FIGURE_FONT) +
      labs(title = paste(lab, "at ADT day 180"), subtitle = NULL,
        y = "Platinum-free probability", x = "Days from day-180 landmark")
  }
  names(plots) <- c("volcano_day0", "volcano_day180", "psa_km_day180", "testosterone_km_day180")
  # Cell 5 holds the single volcano legend shared by panels a and b.
  # Drawn at its poster size: the rest of column A below Results 1
  # (10.35 x 8.99 in). Row heights are in inches; the shared legend row
  # grows with the poster font and the volcano and KM rows share the rest.
  height <- 8.99
  legend_row <- .25 * ESMO_FONT_SCALE
  row_heights <- c(.475 * (height - legend_row), legend_row, .525 * (height - legend_row))
  list(plots = plots, layout = rbind(1:2, c(5, 5), 3:4), row_heights = row_heights,
    shared_legend = list(panels = 1:2, cell = 5, nrow = 1), width = 10.35, height = height,
    plot_theme = theme(plot.margin = margin(1, 5, 1, 5),
      plot.title = element_text(margin = margin(b = 2)),
      legend.box.spacing = grid::unit(2, "pt"), legend.margin = margin(1, 1, 1, 1)),
    composite_theme = theme(plot.tag.position = c(0, 1),
      plot.tag = element_text(hjust = 0, vjust = 1),
      plot.title = element_text(margin = margin(t = ESMO_TAG_BASELINE, b = 2))),
    tables = list(associations = filter(volcano, landmark_days %in% c(0, 180)),
      km_curves = filter(fig$km, scope == "pooled", landmark_days == 180,
        lab %in% c("PSA", "Testosterone"), cutoff_scope == "pooled"),
      km_summary = filter(fig$km_summary, scope == "pooled", landmark_days == 180,
        lab %in% c("PSA", "Testosterone"), cutoff_scope == "pooled")),
    caption = paste("Results 2. Pooled federated univariable mean-laboratory Cox associations at ADT days 0 and 180.",
      "The y axis is −log10(q), with BH adjustment across the mean-feature testing family before display exclusions;",
      "triangles mark values above the display cap. HRs are oriented toward the higher-hazard direction in each label.",
      "Day-180 KM curves compare pre-landmark mean PSA or testosterone using the exported pooled percentile cutoffs,",
      "retaining tied values. The percentage is read from each source summary and shown in the legend,",
      "rather than assumed to be 10% or 20%. Shading is 95% CI; legends give patients/events; log-rank p values appear at the lower left of each KM panel."))
}

esmo_auc_transform <- function() {
  # Reserve a short strip for zero and the break; keep AUC >= .25 linear.
  scales::trans_new("esmo_auc_break",
    transform = function(x) ifelse(x < .25, x * .36, x - .16),
    inverse = function(x) ifelse(x < .09, x / .36, x + .16))
}

esmo_auc_axis <- function(p, upper = 1) {
  # Bars remain continuous to zero across the compressed interval; only the
  # y-axis receives a break mark in esmo_plot_grob().
  suppressMessages(p +
    scale_y_continuous(transform = esmo_auc_transform(), breaks = seq(0, 1, .25),
      labels = scales::label_number(accuracy = .01), expand = expansion(mult = c(0, 0))) +
    coord_cartesian(ylim = c(0, upper)))
}

esmo_result3 <- function(performance, importance) {
  models <- c("Cox baseline (age)", "Elastic-Net Cox", "XGBoost Survival")
  d <- performance %>% filter(as.character(name) %in% models) %>%
    mutate(name = as.character(name))
  esmo_grid(d, c("name", "landmark"), tidyr::expand_grid(name = models, landmark = c(0, 90, 180)),
    "federated held-out performance")
  if (any(!is.finite(d$auc))) stop("Missing held-out mean AUC(t)")
  # These models must describe the same held-out population at each landmark.
  if (nrow(distinct(d, landmark, n_test, n_events_test)) != 3L)
    stop("Model families have inconsistent held-out patient/event counts")
  selected <- esmo_require(federated_manuscript_features(importance, "xgb", 180), "day-180 XGBoost gains")
  p1 <- esmo_auc_axis(plot_model_discrimination(d, "auc", "Held-out mean AUC(t)",
    FEDERATED_MODEL_COLORS[models], show_legend = TRUE), upper = max(1, max(d$auc) * 1.08)) +
    labs(title = "Four-center held-out performance") + guides(fill = guide_legend(nrow = 1)) +
    # Centre the legend under the whole panel (it is wider than the plot area).
    theme(legend.location = "plot")
  p2 <- plot_model_importance(selected, "xgb", "Day 180: top laboratory features", top_n = 15) +
    scale_x_continuous(expand = expansion(mult = c(0, .04)))
  # Bold androgen-axis feature names (plotmath, so no vectorized theme faces).
  androgen <- as.character(p2$data$label[p2$data$category == "Androgen axis"])
  p2 <- suppressMessages(p2 + scale_y_discrete(labels = function(x) as.expression(lapply(x, function(s) {
      label <- manuscript_federated_feature_label(s)
      if (s %in% androgen) bquote(bold(.(label))) else label
    }))) +
    # List only categories present in the top features so the legend fits.
    scale_fill_manual(values = CATEGORY_COLORS, name = NULL,
      breaks = intersect(LEGEND_ORDER, as.character(p2$data$category))) +
    guides(fill = guide_legend(nrow = 1)))
  list(plots = list(heldout_performance = p1, xgboost_features_day180 = p2),
    tables = list(performance = d, top_features = selected), layout = matrix(1:2, nrow = 1),
    broken_auc = "heldout_performance",
    # Panel letters sit on the title line (as in Results 2); titles are
    # indented past the letter.
    composite_theme = theme(plot.tag.position = c(0, 1),
      plot.tag = element_text(hjust = 0, vjust = 1),
      plot.title = element_text(margin = margin(t = ESMO_TAG_BASELINE, l = 16 * ESMO_FONT_SCALE, b = 5.5))),
    # Drawn at its poster size (top of column B).
    col_widths = c(1.12, 1), width = 10.09, height = 4.48,
    # Titles start at the left of each panel, so the long feature-panel title fits.
    plot_theme = theme(plot.margin = margin(4, 6, 4, 6), plot.title.position = "plot",
      legend.box.spacing = grid::unit(2, "pt"), legend.margin = margin(1, 1, 1, 1)),
    caption = paste("Results 3. Four-center held-out test mean time-dependent AUC(t) at ADT days 0, 90 and 180.",
      "Age-only Cox, elastic-net Cox and XGBoost are shown; no uncertainty intervals are supplied by these exports.",
      "The AUC axis shows zero with a marked break before 0.25; values from 0.25 upward use a linear scale.",
      "The feature panel ranks up to 15 eligible positive-gain laboratory features at day 180, excluding age,",
      "body height, identifiers, measurement counts and missingness indicators from display.",
      "Gain measures importance, not effect direction. Counts/missingness can still contribute to the fitted model.",
      paste(sprintf("Day %d test n/events: %s/%s.", sort(unique(d$landmark)),
        scales::comma(d$n_test[match(sort(unique(d$landmark)), d$landmark)]),
        scales::comma(d$n_events_test[match(sort(unique(d$landmark)), d$landmark)])), collapse = " ")))
}

esmo_transfer_gradient <- function(patients) {
  # All three stratifiers use the same eligible patients for an endpoint.
  # Freeze the transferred model; calculate score quintiles within this subset.
  d <- patients %>% filter(model_family == "xgboost_cox", landmark_days == 180,
    clock == "sequencing", sequenced, is.finite(risk_score),
    is.finite(GLEASON_SCORE), GLEASON_SCORE > 0,
    TP53_ANY %in% c(0, 1), RB1_ANY %in% c(0, 1), PTEN_ANY %in% c(0, 1))
  esmo_require(d, "paired sequencing/Gleason transfer patients at day 180")
  esmo_unique(d, c("endpoint", "person_id"), "paired transfer patient (one full-model bundle required)")
  missing <- setdiff(names(FEDERATED_TRANSFER_ENDPOINTS), unique(d$endpoint))
  if (length(missing)) stop("No paired transfer patients for ", paste(missing, collapse = ", "))
  tables <- federated_transfer_km_tables(d)
  gradient <- tables$gradient %>% filter(stratifier %in% c("score_extreme", "gleason_group", "trio_group"))
  for (ep in names(FEDERATED_TRANSFER_ENDPOINTS)) {
    counts <- gradient %>% filter(endpoint == ep) %>% group_by(stratifier) %>% summarise(n = sum(n), .groups = "drop")
    # Extreme score groups intentionally omit the middle 60%; the clinical
    # stratifiers must each include the full paired subset.
    n <- sum(d$endpoint == ep)
    if (sum(counts$stratifier %in% c("gleason_group", "trio_group") & counts$n == n) != 2L ||
        sum(gradient$endpoint == ep & gradient$stratifier == "score_extreme") != 2L)
      stop("Incomplete paired stratification groups for ", ep, "; inspect score ties or clinical data")
  }
  gradient
}

esmo_result4 <- function(metrics, gradient) {
  d <- metrics %>% filter(model_family == "xgboost_cox", config == "both", landmark_days %in% c(0, 90, 180))
  esmo_grid(d, c("endpoint", "landmark_days"),
    tidyr::expand_grid(endpoint = names(FEDERATED_TRANSFER_ENDPOINTS), landmark_days = c(0, 90, 180)),
    "DFCI XGBoost transfer performance")
  if (any(!is.finite(d$mean_auc) | !is.finite(d$mean_auc_lo) | !is.finite(d$mean_auc_hi)) ||
      any(d$mean_auc_lo > d$mean_auc | d$mean_auc_hi < d$mean_auc))
    stop("Results 4 requires all nine transfer mean AUC(t) estimates and their 95% bootstrap CIs; rerun notebook 04")
  if (any(gradient$clock != "sequencing") || any(gradient$horizon_days != 1825))
    stop("Results 4 probabilities require a five-year horizon on the sequencing clock")
  p1 <- esmo_auc_axis(plot_federated_transfer_bars(
    prepare_federated_transfer(d, federated_transfer_strip_counts(d, 0)), ymax = 1)) +
    labs(title = "Three-site XGBoost model applied to DFCI without refitting", subtitle = NULL) +
    theme(plot.title.position = "panel", plot.margin = margin(4, 2, 1, 2),
      axis.title.y = element_text(margin = margin(r = 2)),
      axis.text.y = element_text(margin = margin(r = 2)))
  probabilities <- lapply(names(FEDERATED_TRANSFER_ENDPOINTS), function(ep) {
    p <- plot_federated_transfer_horizon_risk(list(gradient = gradient), "xgboost_cox", 180,
      endpoint = ep, score_stratifier = "score_extreme")
    levels(p$data$stratifier_label) <- c("XGBoost", "Gleason", "TSG")
    p
  })
  # Match the proposal's common 0-0.6 scale, expanding only if a real CI needs it.
  upper <- max(c(.6, unlist(lapply(probabilities, function(p) p$data$probability_hi))), na.rm = TRUE)
  ymax <- min(1, ceiling(upper * 5) / 5)
  probabilities <- lapply(seq_along(probabilities), function(i) probabilities[[i]] +
    coord_cartesian(ylim = c(0, ymax)) +
    # Reclaim outer/facet padding for wider scatter areas and more separation
    # between paired tick labels, retaining the full-size poster typography.
    scale_x_discrete(expand = expansion(add = .45)) +
    labs(title = NULL, subtitle = NULL,
      y = paste("Five-year", c("platinum", "AVPC", "NEPC")[[i]], "probability")) +
    theme(axis.text.x = element_text(angle = 0, hjust = .5),
      axis.title.y = element_text(margin = margin(r = 2)),
      axis.text.y = element_text(margin = margin(r = 2)),
      panel.spacing.x = grid::unit(4, "pt"),
      plot.title.position = "panel", plot.margin = margin(1, 6, 12, 2)))
  list(plots = c(list(transfer_auc = p1), setNames(probabilities, paste0(names(FEDERATED_TRANSFER_ENDPOINTS), "_five_year"))),
    tables = list(transfer_performance = d, stratification = gradient,
      event_probabilities = bind_rows(lapply(probabilities, function(p) p$data))),
    # The upper panel spans the full combined width of the three lower panels.
    # Drawn at its poster size (bottom of column B); row heights are in inches.
    layout = rbind(c(1, 1, 1), 2:4),
    row_heights = c(3.92, 4.23), width = 10.09, height = 8.15,
    broken_auc = "transfer_auc",
    # Panel letters sit on the title (a) or strip (b-d) line, as in Results 2.
    composite_theme = theme(plot.tag.position = c(0, 1),
      plot.tag = element_text(hjust = 0, vjust = 1),
      plot.title = element_text(margin = margin(t = ESMO_TAG_BASELINE, b = 5.5))),
    caption = paste("Results 4. XGBoost trained for platinum without DFCI, transferred without refitting.",
      "Top: mean AUC(t) and patient-bootstrap 95% CIs at ADT days 0/90/180; counts are endpoint-specific.",
      "The AUC axis shows zero with a marked break before 0.25; values from 0.25 upward use a linear scale.",
      "Bottom: five-year event probability (1 − KM survival) and 95% CI from 180 days after the sequencing report.",
      "For each endpoint, score, Gleason and TSG use the same sequenced patients with Gleason and known TP53/RB1/PTEN status.",
      "Q1/Q5 are the bottom/top score quintiles in that paired subset, retaining ties and omitting the middle 60%.",
      "Gleason groups are ≤7/≥8; TSG groups are WT/any alteration in TP53, RB1 or PTEN.",
      "Unavailable horizon risks are labeled, not extrapolated. Death is censored; these are not competing-risk incidences.",
      "This descriptive comparison does not test incremental clinical utility or superiority."))
}

esmo_inside_legend <- function(position, justification = position) {
  if (utils::packageVersion("ggplot2") >= "3.5.0")
    theme(legend.position = "inside", legend.position.inside = position, legend.justification = justification)
  else theme(legend.position = position, legend.justification = justification)
}

# Each composite is drawn at the size it is placed on the 16:9 poster
# (160 x 90 cm board, 34.5 x 19.4 in slide), so one set of point sizes prints
# identically in Results 1-4. All poster text, including text layers, is
# multiplied by this factor; 1 gives 12 pt tick labels. 1.05 is the largest
# that keeps every label clear (Results 1 is the limiting figure); a larger
# value needs the Results 1 height re-fitted to keep its matrix square.
ESMO_FONT_SCALE <- 1.05
# Results 2-4 draw each panel letter on its title line; this top margin (pt)
# drops the smaller title onto the letter's baseline.
ESMO_TAG_BASELINE <- 1.8 * ESMO_FONT_SCALE

esmo_style_plot <- function(p) {
  # Poster typography is local to this entry point, leaving the shared
  # manuscript theme alone. Set axis children too: some source plots give
  # their tick labels explicit sizes that otherwise override axis.text.
  pt <- function(x) x * ESMO_FONT_SCALE
  p <- p + labs(caption = NULL) +
    theme(text = element_text(family = FIGURE_FONT, size = pt(12)),
      plot.title = element_text(size = pt(14), face = "bold"),
      plot.subtitle = element_text(size = pt(11.5)), plot.tag = element_text(size = pt(16), face = "bold"),
      axis.title = element_text(size = pt(13)), axis.title.x = element_text(size = pt(13)),
      axis.title.y = element_text(size = pt(13)), axis.text = element_text(size = pt(12)),
      axis.text.x = element_text(size = pt(12)), axis.text.y = element_text(size = pt(12)),
      strip.text = element_text(size = pt(12)), legend.text = element_text(size = pt(12)),
      legend.title = element_text(size = pt(12)), legend.key.height = grid::unit(pt(14), "pt"),
      legend.key.width = grid::unit(pt(14), "pt"), legend.position = "bottom")
  p$layers <- lapply(p$layers, function(layer) {
    if (!inherits(layer$geom, c("GeomText", "GeomLabel", "GeomTextRepel", "GeomLabelRepel"))) return(layer)
    copy <- ggplot2::ggproto(NULL, layer)
    # Enlarge numeric bar labels and missing-estimate annotations as well as
    # theme text, while retaining the larger confusion counts and OR label.
    copy$aes_params$size <- pt(max(c(3.8, layer$aes_params$size)))
    copy$aes_params$family <- FIGURE_FONT
    copy
  })
  p
}

esmo_plot_grob <- function(p, broken_auc = FALSE) {
  g <- ggplot2::ggplotGrob(p)
  if (!broken_auc) return(g)
  built <- ggplot2::ggplot_build(p)
  limits <- built$layout$panel_params[[1]]$y.range
  y <- (esmo_auc_transform()$transform(.125) - limits[1]) / diff(limits)
  panels <- g$layout[grepl("^panel($|-)", g$layout$name), ]
  panel <- panels[which.min(panels$l), ]
  # Overlay a white-backed double slash on the y-axis, inside the omitted strip.
  mark <- grid::grobTree(
    grid::rectGrob(x = 0, y = y, width = grid::unit(8, "pt"), height = grid::unit(10, "pt"),
      gp = grid::gpar(fill = "white", col = NA)),
    grid::segmentsGrob(x0 = grid::unit(-3, "pt"), x1 = grid::unit(3, "pt"),
      y0 = grid::unit(y, "npc") + grid::unit(c(-4, 0), "pt"),
      y1 = grid::unit(y, "npc") + grid::unit(c(0, 4), "pt"),
      gp = grid::gpar(col = "black", lwd = 1)))
  gtable::gtable_add_grob(g, mark, t = panel$t, b = panel$b, l = panel$l, r = panel$r,
    clip = "off", z = Inf, name = "auc-axis-break")
}

esmo_export <- function(result, section, output) {
  # Measure text on an explicit Unicode-capable device. ggplotGrob() otherwise
  # opens Rplots.pdf in a batch run and can substitute glyphs in the composite.
  measurement <- tempfile(".esmo-measure-", fileext = ".png")
  figure_measure_device(measurement, result$width, result$height)
  measurement_device <- grDevices::dev.cur()
  on.exit({
    if (measurement_device %in% grDevices::dev.list()) grDevices::dev.off(measurement_device)
    unlink(measurement)
  }, add = TRUE)
  folder <- file.path(output, paste0("results_", section))
  dir.create(file.path(folder, "data"), recursive = TRUE, showWarnings = FALSE)
  writeLines(result$caption, file.path(folder, "caption.txt"), useBytes = TRUE)
  for (name in names(result$tables)) readr::write_csv(result$tables[[name]], file.path(folder, "data", paste0(name, ".csv")))
  # Figures are exported as PDF only.
  save <- function(p, stem, width, height) {
    destination <- file.path(folder, paste0(stem, ".pdf"))
    temporary <- tempfile(".esmo-", tmpdir = folder, fileext = ".pdf")
    tryCatch({
      ggplot2::ggsave(temporary, plot = p, device = figure_pdf_device, width = width, height = height,
        bg = "white", limitsize = FALSE)
      if (is.na(file.info(temporary)$size) || file.info(temporary)$size == 0 || !file.rename(temporary, destination))
        stop("Cannot publish ", destination)
    }, finally = unlink(temporary))
  }
  # Captions live beside the graphics; keep the individual panels easy to place.
  plots <- lapply(result$plots, esmo_style_plot)
  if (!is.null(result$plot_theme)) plots <- lapply(plots, function(p) p + result$plot_theme)
  for (name in names(result$legend_inside))
    # Legends drawn over a panel are set a step smaller than the outside ones.
    plots[[name]] <- plots[[name]] + do.call(esmo_inside_legend, as.list(result$legend_inside[[name]])) +
      theme(legend.text = element_text(size = 10.5 * ESMO_FONT_SCALE),
        legend.key.height = grid::unit(11 * ESMO_FONT_SCALE, "pt"),
        legend.key.width = grid::unit(11 * ESMO_FONT_SCALE, "pt"))
  row_weights <- if (is.null(result$row_heights)) rep(1, nrow(result$layout)) else result$row_heights
  col_weights <- if (is.null(result$col_widths)) rep(1, ncol(result$layout)) else result$col_widths
  for (i in seq_along(plots)) {
    cells <- which(result$layout == i, arr.ind = TRUE)
    # Avoid a floating-point value just below 12 becoming a 863-point PDF.
    width <- round(result$width * sum(col_weights[unique(cells[, 2])]) / sum(col_weights), 10)
    height <- result$height * sum(row_weights[unique(cells[, 1])]) / sum(row_weights)
    p <- if (names(plots)[i] %in% result$broken_auc) esmo_plot_grob(plots[[i]], TRUE) else plots[[i]]
    save(p, names(plots)[i], width, height)
  }
  # Individual panels keep their legends; the composite can share one.
  shared <- result$shared_legend
  grobs <- lapply(seq_along(plots), function(i) {
    p <- plots[[i]] + labs(tag = letters[i])
    if (!is.null(result$composite_theme)) p <- p + result$composite_theme
    if (i %in% shared$panels) p <- p + theme(legend.position = "none")
    esmo_plot_grob(p, names(plots)[i] %in% result$broken_auc)
  })
  # Panels confined to one layout row share gtable heights, so their x axes
  # (and panel bottoms) line up even when only some carry a bottom legend.
  for (r in seq_len(nrow(result$layout))) {
    in_row <- Filter(function(i) i <= length(plots) &&
      all(which(result$layout == i, arr.ind = TRUE)[, 1] == r), stats::na.omit(unique(result$layout[r, ])))
    if (length(in_row) > 1L && length(unique(vapply(grobs[in_row], function(g) length(g$heights), integer(1)))) == 1L) {
      heights <- do.call(grid::unit.pmax, lapply(grobs[in_row], function(g) g$heights))
      for (i in in_row) {
        original <- grobs[[i]]$heights
        grobs[[i]]$heights <- heights
        # Keep panel bottoms aligned, but don't put extra tick-label space
        # between a one-line x axis and its title (Figure 1's matrix).
        axis_row <- unique(grobs[[i]]$layout$t[grobs[[i]]$layout$name == "axis-b"])
        if (isTRUE(result$compact_x_titles) && !is.null(plots[[i]]$labels$x) && length(axis_row) == 1L) {
          gap <- heights[axis_row] - original[axis_row]
          grobs[[i]]$heights[axis_row] <- original[axis_row]
          bottom <- length(heights)
          grobs[[i]]$heights[bottom] <- heights[bottom] + gap
        }
      }
    }
  }
  if (!is.null(shared)) {
    legend_plot <- plots[[shared$panels[1]]]
    if (!is.null(shared$nrow)) legend_plot <- legend_plot +
      guides(fill = guide_legend(nrow = shared$nrow, byrow = TRUE,
        override.aes = list(shape = 21, size = 3.2, alpha = 1)))
    g <- ggplot2::ggplotGrob(legend_plot)
    boxes <- Filter(function(x) inherits(x, "gtable"), g$grobs[grepl("^guide-box", g$layout$name)])
    if (!length(boxes)) stop("Results ", section, " shared legend could not be extracted")
    grobs[[shared$cell]] <- boxes[[1]]
  }
  combined <- gridExtra::arrangeGrob(grobs = grobs, layout_matrix = result$layout,
    heights = row_weights, widths = col_weights)
  save(combined, paste0("ESMO_AI_results_", section), result$width, result$height)
  message("Saved Results ", section, " to ", folder)
  invisible(folder)
}

esmo_from_aggregates <- function(root, sections = as.character(1:4)) {
  tables <- list(`1` = c("confusion", "landscape", "enrichment", "audit"),
    `2` = c("associations", "km_curves", "km_summary"),
    `3` = c("performance", "top_features"),
    `4` = c("transfer_performance", "stratification"))
  results <- lapply(sections, function(section) {
    message("Preparing Results ", section, " from saved aggregate tables")
    paths <- file.path(root, paste0("results_", section), "data", paste0(tables[[section]], ".csv"))
    if (any(!file.exists(paths))) stop("Missing aggregate input: ", paste(paths[!file.exists(paths)], collapse = ", "))
    d <- setNames(lapply(paths, function(path) {
      table <- readr::read_csv(path, show_col_types = FALSE)
      readr::stop_for_problems(table)
      esmo_require(table, path)
    }), tables[[section]])
    switch(section,
      `1` = esmo_result1(d),
      `2` = esmo_result2(d$associations, list(km = d$km_curves, km_summary = d$km_summary)),
      `3` = esmo_result3(d$performance, d$top_features),
      `4` = esmo_result4(d$transfer_performance, d$stratification))
  })
  setNames(results, sections)
}

esmo_main <- function(args = commandArgs(TRUE)) {
  if (length(args)) {
    if (!identical(args, "--help")) stop("Only --help is accepted; configure paths with environment variables.")
    cat(paste(c("Generate Results 1-4 from COMPASS_ESMO_AI_proposed.",
      "Usage: Rscript COMPASS/figures/ESMO_AI_Figures.R",
      "COMPASS_ESMO_SECTIONS           1,2,3,4 (comma-separated subset)",
      "COMPASS_DATA_ROOT               COMPASS analysis data (existing cluster default)",
      "COMPASS_LLM_ANNOTATIONS_PATH     directory containing primary_subtype_labels.parquet",
      "COMPASS_RHINO_ROOT              <COMPASS_DATA_ROOT>/rhino_data_v2",
      "COMPASS_FEDERATED_RESULTS       <RHINO_ROOT>/cox_federated_univariate_adt.csv",
      "COMPASS_FEDERATED_FIGURE_DATA    <RHINO_ROOT>/figure_data (pooled KM exports)",
      "COMPASS_FEDERATED_TRANSFER_NO_DFCI <RHINO_ROOT>/federated_transfer_adt_no_DFCI",
      "COMPASS_ESMO_OUTPUT_ROOT        <COMPASS_FIG_ROOT>/ESMO_AI",
      "COMPASS_ESMO_AGGREGATE_ROOT     optional prior ESMO_AI output folder; rerender its saved CSVs",
      "Run preprocessing/univariate/multivariate/transfer notebooks 01-04 first.",
      "Results 1 reads day-0 prediction inputs + manual/LLM labels; 2 uses federated Cox/KM exports;",
      "3 uses federated held-out metrics/gains; 4 uses notebook-04 transfer metrics/patients.",
      "Writes four PDF composites, individual PDF panels, aggregate CSVs, captions and sessionInfo.txt."), collapse = "\n"), "\n")
    return(invisible(NULL))
  }
  sections <- trimws(strsplit(Sys.getenv("COMPASS_ESMO_SECTIONS", "1,2,3,4"), ",", fixed = TRUE)[[1]])
  if (!length(sections) || any(!sections %in% as.character(1:4)) || anyDuplicated(sections)) stop("Invalid COMPASS_ESMO_SECTIONS")
  data <- Sys.getenv("COMPASS_DATA_ROOT", "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS")
  rhino <- Sys.getenv("COMPASS_RHINO_ROOT", file.path(data, "rhino_data_v2"))
  federated <- Sys.getenv("COMPASS_FEDERATED_RESULTS", file.path(rhino, "cox_federated_univariate_adt.csv"))
  transfer <- Sys.getenv("COMPASS_FEDERATED_TRANSFER_NO_DFCI", file.path(rhino, "federated_transfer_adt_no_DFCI"))
  output <- Sys.getenv("COMPASS_ESMO_OUTPUT_ROOT", file.path(Sys.getenv("COMPASS_FIG_ROOT",
    "/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS"), "ESMO_AI"))
  message("ESMO AI figures: Results ", paste(sections, collapse = ", "))
  message("Output directory: ", normalizePath(output, mustWork = FALSE))
  esmo_load_helpers()
  # Build every requested section before publishing; a missing analysis cannot
  # silently turn a requested panel into a partial poster.
  aggregate <- Sys.getenv("COMPASS_ESMO_AGGREGATE_ROOT", "")
  if (nzchar(aggregate)) {
    results <- esmo_from_aggregates(aggregate, sections)
  } else {
    results <- list()
    if ("1" %in% sections) {
      message("Preparing Results 1: LLM validation and platinum enrichment")
      inputs <- file.path(data, "survival_analysis", "prediction_inputs_adt")
      patients <- readr::read_csv(file.path(inputs, "aggregated_landmark0.csv"),
        col_select = all_of(c("DFCI_MRN", "PLATINUM")), show_col_types = FALSE)
      patients$DFCI_MRN <- suppressWarnings(as.numeric(patients$DFCI_MRN))
      labels <- esmo_require(load_llm_strata(Sys.getenv("COMPASS_LLM_ANNOTATIONS_PATH", DEFAULT_LLM_ANNOTATIONS_PATH)), "LLM labels")
      manual <- load_manual_annotations(file.path(data, "LLM_NEPC_labels", MANUAL_ANNOTATIONS_FILE))
      results[["1"]] <- esmo_result1(esmo_llm_data(patients, labels, manual))
    }
    if ("2" %in% sections) {
      message("Preparing Results 2: lab associations and survival curves")
      fig <- load_federated_figure_data(Sys.getenv("COMPASS_FEDERATED_FIGURE_DATA", file.path(rhino, "figure_data")))
      if (is.null(fig)) stop("Missing federated aggregate KM exports for Results 2")
      results[["2"]] <- esmo_result2(federated_volcano_data(federated), fig)
    }
    if ("3" %in% sections) {
      message("Preparing Results 3: federated performance and feature importance")
      paths <- lapply(c("elastic_net", "xgboost"), function(family) federated_multivariable_files(federated, family, "pooled"))
      performance <- bind_rows(lapply(seq_along(paths), function(i) {
        family <- c("elastic_net", "xgboost")[i]
        metrics <- esmo_require(load_federated_multivariable(paths[[i]]$metrics, family, "metrics"), paste(family, "metrics"))
        prepare_federated_performance(metrics, family)
      }))
      importance <- esmo_require(load_federated_multivariable(paths[[2]]$features, "xgboost", "features"), "XGBoost importance")
      results[["3"]] <- esmo_result3(performance, importance)
    }
    if ("4" %in% sections) {
      message("Preparing Results 4: DFCI transfer and five-year stratification")
      metrics <- esmo_require(load_federated_transfer(transfer, "metrics"), "transfer metrics")
      patients <- esmo_require(load_federated_transfer_km(transfer), "transfer patients")
      results[["4"]] <- esmo_result4(metrics, esmo_transfer_gradient(patients))
    }
  }
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  message("Rendering ", length(results), " figure sets (PDF)")
  for (section in sections) esmo_export(results[[section]], section, output)
  writeLines(capture.output(utils::sessionInfo()), file.path(output, "sessionInfo.txt"))
  message("Finished ESMO AI figures in ", normalizePath(output, mustWork = TRUE))
  invisible(results)
}

# source() and RStudio run the file inside call frames too. Use an explicit
# opt-out for tests/library use; a call-depth guard silently skips normal runs.
if (!identical(getOption("compass.esmo.autorun"), FALSE)) {
  esmo_main(if (sys.nframe() == 0L) commandArgs(TRUE) else character())
}
