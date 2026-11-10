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
  class_labels <- c("Conventional", "AVPC", "NEPC")
  cm <- d$confusion %>% group_by(truth) %>%
    mutate(row_fraction = if (sum(n) > 0) n / sum(n) else 0) %>% ungroup() %>%
    mutate(truth = factor(truth, levels = rev(classes), labels = rev(class_labels)),
      pred = factor(pred, levels = classes, labels = class_labels))
  p1 <- ggplot(cm, aes(pred, truth, fill = row_fraction)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = n, color = row_fraction > .5), size = 4, fontface = "bold") +
    scale_fill_gradient(low = "#eaf1fb", high = COLOR_PLATINUM_POS, limits = c(0, 1), guide = "none") +
    scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#0b0b0b"), guide = "none") +
    scale_x_discrete(position = "top") +
    labs(x = "LLM label (prediction)", y = "Manual annotation (truth)") + theme_fig() +
    theme(axis.text.x.top = element_text(angle = 0, hjust = .5))
  landscape <- d$landscape %>% mutate(primary_label = factor(primary_label, levels = classes, labels = class_labels),
    platinum = factor(PLATINUM, levels = c(1, 0), labels = c("Platinum+", "Platinum−")))
  counts <- vapply(c(1, 0), function(x) sum(landscape$n[landscape$PLATINUM == x]), numeric(1))
  p2 <- ggplot(landscape, aes(primary_label, fraction, fill = platinum)) +
    geom_col(position = position_dodge(.8), width = .72) +
    scale_fill_manual(values = c(COLOR_PLATINUM_POS, COLOR_PLATINUM_NEG), name = NULL,
      labels = sprintf(c("Platinum+ (n=%s)", "Platinum− (n=%s)"), scales::comma(counts))) +
    # Bars start on the x axis, matching the enrichment panel.
    scale_y_continuous(limits = c(0, 1), labels = scales::label_percent(),
      expand = expansion(mult = c(0, .04))) +
    labs(x = NULL, y = "Fraction within\nplatinum group") + theme_fig() +
    guides(fill = guide_legend(ncol = 1)) +
    theme(legend.text = element_text(size = 9), legend.key.size = grid::unit(9, "pt"),
      legend.margin = margin(2, 2, 2, 2), legend.background = element_rect(fill = "white", color = NA))
  enrichment <- d$enrichment %>% mutate(group = factor(group, levels = group))
  p3 <- ggplot(enrichment, aes(group, probability, fill = group)) +
    geom_col(width = .55) + geom_errorbar(aes(ymin = lo, ymax = hi), width = .18) +
    scale_fill_manual(values = c(COLOR_PLATINUM_POS, "#9a9890"), guide = "none") +
    scale_y_continuous(labels = scales::label_percent(), expand = expansion(mult = c(0, .24))) +
    annotate("text", x = 1.5, y = Inf, vjust = 1.3, size = 3.7, family = FIGURE_FONT,
      label = sprintf("OR = %.1f (p=%.1e)", enrichment$odds_ratio[1], enrichment$fisher_p[1])) +
    labs(x = NULL, y = "Platinum-positive fraction") + theme_fig()
  list(plots = list(confusion_matrix = p1, subtype_fractions = p2, platinum_enrichment = p3),
    # Wider panels keep the top prediction labels horizontal and readable.
    # The subtype legend sits in b's empty upper right, so no panel carries a
    # bottom legend; export aligns the three plotting areas.
    tables = d, layout = matrix(1:3, nrow = 1), col_widths = c(4.4, 4.8, 4), width = 13.2, height = 3.9,
    legend_inside = list(subtype_fractions = c(.98, .98)),
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
      copy$data$repel_label <- gsub("Alkaline phosphatase", "ALP", copy$data$repel_label, fixed = TRUE)
      copy$data$repel_label <- gsub(" absolute", " abs.", copy$data$repel_label, fixed = TRUE)
      copy$aes_params$size <- 3.8
      copy$aes_params$point.size <- 3.2
      copy$geom_params$box.padding <- grid::unit(.5, "lines")
      copy$geom_params$point.padding <- grid::unit(.6, "lines")
      copy$geom_params$force <- 5
      copy$geom_params$force_pull <- .1
      copy$position <- ggrepel::position_nudge_repel(x = .07, y = 1)
      # An iteration budget, rather than wall time, makes placement repeatable.
      copy$geom_params$max.time <- Inf
      copy$geom_params$max.iter <- 50000
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
      override.aes = list(shape = 21, size = 2.5, alpha = 1))) +
    theme(axis.title = element_text(size = 11), axis.text = element_text(size = 10),
      legend.text = element_text(size = 9))
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
    plots[[length(plots) + 1L]] <- p + labs(title = paste(lab, "at ADT day 180"),
      y = "Platinum-free probability", x = "Days from day-180 landmark")
  }
  names(plots) <- c("volcano_day0", "volcano_day180", "psa_km_day180", "testosterone_km_day180")
  # Cell 5 holds the single volcano legend shared by panels a and b.
  list(plots = plots, layout = rbind(1:2, c(5, 5), 3:4), row_heights = c(1, .05, 1),
    shared_legend = list(panels = 1:2, cell = 5, nrow = 1), width = 10, height = 8.9,
    plot_theme = theme(plot.margin = margin(4, 5, 4, 5), plot.subtitle = element_text(size = 10),
      legend.box.spacing = grid::unit(2, "pt"), legend.margin = margin(1, 1, 1, 1),
      legend.key.height = grid::unit(11, "pt")),
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
      "rather than assumed to be 10% or 20%. Shading is 95% CI; legends give patients/events; subtitles give log-rank p values."))
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
  p1 <- plot_model_discrimination(d, "auc", "Held-out mean AUC(t)",
    FEDERATED_MODEL_COLORS[models], show_legend = TRUE) +
    scale_y_continuous(breaks = seq(0, 1, .25), expand = expansion(mult = c(0, .02))) +
    labs(title = "Four-center held-out performance") + guides(fill = guide_legend(nrow = 1))
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
    col_widths = c(1, 1.1), width = 12, height = 5.4,
    caption = paste("Results 3. Four-center held-out test mean time-dependent AUC(t) at ADT days 0, 90 and 180.",
      "Age-only Cox, elastic-net Cox and XGBoost are shown; no uncertainty intervals are supplied by these exports.",
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
  # Zoom (not limit) to 0.25-1 so bars are clipped at the axis, not dropped.
  p1 <- suppressMessages(plot_federated_transfer_bars(prepare_federated_transfer(d, federated_transfer_strip_counts(d, 0)), ymax = 1) +
    scale_y_continuous(breaks = seq(.25, 1, .25), expand = expansion(mult = c(0, 0))) +
    coord_cartesian(ylim = c(.25, 1))) +
    labs(title = "Three-site XGBoost → DFCI without refitting", subtitle = "Landmarks after ADT initiation; 95% bootstrap CI") +
    theme(plot.title.position = "panel", plot.margin = margin(12, 16, 1, 12))
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
    labs(title = FEDERATED_TRANSFER_ENDPOINTS_SHORT[[i]], y = "Five-year event probability",
      subtitle = "Sequencing report +180 days") +
    theme(axis.text.x = element_text(angle = 0, hjust = .5), strip.text = element_text(size = 8),
      plot.title.position = "panel", plot.margin = margin(1, 16, 12, 12)))
  list(plots = c(list(transfer_auc = p1), setNames(probabilities, paste0(names(FEDERATED_TRANSFER_ENDPOINTS), "_five_year"))),
    tables = list(transfer_performance = d, stratification = gradient,
      event_probabilities = bind_rows(lapply(probabilities, function(p) p$data))),
    layout = rbind(c(1, 1, 1), 2:4), row_heights = c(1.5, 1.2), width = 12, height = 8.2,
    caption = paste("Results 4. XGBoost trained for platinum without DFCI, transferred without refitting.",
      "Top: mean AUC(t) and patient-bootstrap 95% CIs at ADT days 0/90/180; counts are endpoint-specific; the y axis starts at 0.25.",
      "Bottom: five-year event probability (1 − KM survival) and 95% CI from 180 days after the sequencing report.",
      "For each endpoint, score, Gleason and TSG use the same sequenced patients with Gleason and known TP53/RB1/PTEN status.",
      "Q1/Q5 are the bottom/top score quintiles in that paired subset, retaining ties and omitting the middle 60%.",
      "Gleason groups are ≤7/≥8; TSG groups are WT/any alteration in TP53, RB1 or PTEN.",
      "Unavailable horizon risks are labeled, not extrapolated. Death is censored; these are not competing-risk incidences.",
      "This descriptive comparison does not test incremental clinical utility or superiority."))
}

esmo_inside_legend <- function(position) {
  if (utils::packageVersion("ggplot2") >= "3.5.0")
    theme(legend.position = "inside", legend.position.inside = position, legend.justification = position)
  else theme(legend.position = position, legend.justification = position)
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
  plots <- lapply(result$plots, function(p) p + labs(caption = NULL) +
    theme(plot.title = element_text(size = 11, face = "bold"),
      plot.subtitle = element_text(size = 9), legend.position = "bottom"))
  if (!is.null(result$plot_theme)) plots <- lapply(plots, function(p) p + result$plot_theme)
  for (name in names(result$legend_inside))
    plots[[name]] <- plots[[name]] + esmo_inside_legend(result$legend_inside[[name]])
  row_weights <- if (is.null(result$row_heights)) rep(1, nrow(result$layout)) else result$row_heights
  col_weights <- if (is.null(result$col_widths)) rep(1, ncol(result$layout)) else result$col_widths
  for (i in seq_along(plots)) {
    cells <- which(result$layout == i, arr.ind = TRUE)
    width <- result$width * sum(col_weights[unique(cells[, 2])]) / sum(col_weights)
    height <- result$height * sum(row_weights[unique(cells[, 1])]) / sum(row_weights)
    save(plots[[i]], names(plots)[i], width, height)
  }
  # Individual panels keep their legends; the composite can share one.
  shared <- result$shared_legend
  grobs <- lapply(seq_along(plots), function(i) {
    p <- plots[[i]] + labs(tag = letters[i])
    if (i %in% shared$panels) p <- p + theme(legend.position = "none")
    ggplot2::ggplotGrob(p)
  })
  # Panels confined to one layout row share gtable heights, so their x axes
  # (and panel bottoms) line up even when only some carry a bottom legend.
  for (r in seq_len(nrow(result$layout))) {
    in_row <- Filter(function(i) i <= length(plots) &&
      all(which(result$layout == i, arr.ind = TRUE)[, 1] == r), unique(result$layout[r, ]))
    if (length(in_row) > 1L && length(unique(vapply(grobs[in_row], function(g) length(g$heights), integer(1)))) == 1L) {
      heights <- do.call(grid::unit.pmax, lapply(grobs[in_row], function(g) g$heights))
      for (i in in_row) grobs[[i]]$heights <- heights
    }
  }
  if (!is.null(shared)) {
    legend_plot <- plots[[shared$panels[1]]]
    if (!is.null(shared$nrow)) legend_plot <- legend_plot +
      guides(fill = guide_legend(nrow = shared$nrow, byrow = TRUE,
        override.aes = list(shape = 21, size = 2.5, alpha = 1)))
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
