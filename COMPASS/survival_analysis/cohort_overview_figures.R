# R-only rendering of small summaries prepared by notebook 04. Populations
# match notebook 07: Stage-3 landmark inputs and the Stage-1 intent label table.
overview_theme <- function() {
  theme_classic(base_size = 11) + theme(
    panel.grid.major.x = element_line(color = "grey92", linewidth = .3),
    axis.ticks.y = element_blank(), axis.line.y = element_blank(),
    plot.title = element_text(face = "bold", size = 12),
    legend.position = "bottom", legend.title = element_blank(),
    plot.margin = margin(12, 20, 12, 12))
}

overview_combine <- function(a, b, title, subtitle) {
  a <- ggplotGrob(a); b <- ggplotGrob(b)
  a$heights <- b$heights <- grid::unit.pmax(a$heights, b$heights)
  gridExtra::arrangeGrob(a, b, ncol = 2, top = grid::textGrob(
    paste(title, subtitle, sep = "\n"), gp = grid::gpar(fontsize = 14),
    x = .5, y = .65), padding = grid::unit(2.3, "lines"))
}

plot_cohort_event_overview <- function(d, cohorts, landmark = 180L) {
  d <- d %>% mutate(cohort = as.character(cohort), endpoint = as.character(endpoint))
  good <- d %>% filter(status == "ok", n_patients > 0)
  if (!nrow(good)) return(NULL)
  # Incidence is the observed event fraction among landmark-eligible patients,
  # not cumulative incidence by day 180 and not a censoring-adjusted estimator.
  z <- 1.96
  good <- good %>% mutate(p = n_events / n_patients, denominator = 1 + z^2 / n_patients,
    centre = (p + z^2 / (2*n_patients)) / denominator,
    half = z * sqrt(p*(1-p)/n_patients + z^2/(4*n_patients^2)) / denominator,
    rate = 100*p, lo = 100*pmax(0, centre-half), hi = 100*pmin(1, centre+half),
    y = length(cohorts) + 1 - match(cohort, cohorts) + if_else(endpoint == "platinum", .19, -.19),
    thin = n_events < 25, rate_label = sprintf("%.1f%s", rate, if_else(thin, "*", "")))
  colors <- c(platinum = "#2a78d6", nepc = "#eb6834")
  labels <- sub("^adt$", "all", cohorts)
  labels <- sub("^adt_", "", labels)
  labels <- sub("^noprecastrate$", "all +noprecastrate", labels)
  labels <- sub("_noprecastrate$", " +noprecastrate", labels)
  yscale <- function(show = TRUE) scale_y_continuous(breaks = rev(seq_along(cohorts)),
    labels = if (show) labels else NULL, limits = c(.4, length(cohorts)+.6))
  a <- ggplot(good, aes(y = y, fill = endpoint)) +
    geom_rect(aes(xmin = 0, xmax = rate, ymin = y-.16, ymax = y+.16), color = "white")
  # Clip diagonal segments to each thin-cell bar; avoids a pattern dependency.
  hatch <- bind_rows(lapply(which(good$thin & good$rate > 0), function(i) {
    width <- max(good$hi) / 25
    starts <- seq(-width, good$rate[i], by = width*.6)
    x <- pmax(0, starts); xend <- pmin(good$rate[i], starts + width)
    tibble(x = x, xend = xend, y = good$y[i]-.16 + (x-starts)/width*.32,
           yend = good$y[i]-.16 + (xend-starts)/width*.32) %>% filter(xend > x)
  }))
  if (nrow(hatch)) a <- a + geom_segment(data = hatch, aes(x=x, xend=xend, y=y, yend=yend),
    inherit.aes = FALSE, color = "white", linewidth = .35)
  a <- a + geom_errorbar(aes(xmin = lo, xmax = hi), orientation = "y", width = .10,
                         color = "grey30", linewidth = .5) +
    geom_text(aes(x = hi, label = rate_label), hjust = -.2, size = 3, color = "grey30") +
    scale_fill_manual(values = colors, breaks = names(colors), drop = FALSE) + yscale() +
    scale_x_continuous(limits = c(0, max(good$hi)*1.17), expand = expansion(mult = c(0,.01))) +
    labs(x = "Event rate (%)", y = NULL,
         title = sprintf("(a) incidence at landmark +%dd (95%% Wilson CI)", landmark)) + overview_theme()
  b <- ggplot(good, aes(y = y, fill = endpoint)) +
    geom_rect(aes(xmin = 0, xmax = n_patients, ymin = y-.16, ymax = y+.16), alpha = .28) +
    geom_rect(aes(xmin = 0, xmax = n_events, ymin = y-.16, ymax = y+.16)) +
    geom_text(aes(x = n_patients, label = paste0(scales::comma(n_events), "/", scales::comma(n_patients))),
              hjust = -.1, size = 3, color = "grey30") +
    scale_fill_manual(values = colors, breaks = names(colors), drop = FALSE) + yscale(FALSE) +
    scale_x_continuous(limits = c(0, max(good$n_patients)*1.27), expand = expansion(mult = c(0,.01))) +
    labs(x = "Patients (events overlaid)", y = NULL, title = "(b) cohort size and event count",
         caption = "Pale = patients; solid = events; label = events/patients") + overview_theme()
  unavailable <- nrow(d) - nrow(good)
  overview_combine(a, b, sprintf("Event incidence across the analysis cohorts | landmark +%dd", landmark),
    sprintf("Hatched/* = <25 events (%d of %d available cells); %d unavailable cells left blank.\nObserved events during follow-up, not risk by day %d; overlapping cohorts are descriptive.",
            sum(good$thin), nrow(good), unavailable, landmark))
}

plot_stage1_label_overview <- function(d) {
  if (!nrow(d)) return(NULL)
  adt_order <- c("Metastatic", "Local", if (any(d$adt_label == "Unlabelled")) "Unlabelled")
  llm_order <- c("Metastatic", "Local", "Unlabelled")
  d <- d %>% complete(adt_label = adt_order, llm_label = llm_order, fill = list(n = 0L)) %>%
    mutate(adt_label = factor(adt_label, levels = adt_order),
           llm_label = factor(llm_label, levels = llm_order)) %>%
    arrange(adt_label, llm_label) %>% group_by(adt_label) %>%
    mutate(right = cumsum(n), left = right - n) %>% ungroup() %>%
    mutate(y = length(adt_order) + 1 - as.integer(adt_label), midpoint = (left+right)/2,
           small = n <= .07 * max(right), label_y = y + .40 + .08*as.integer(llm_label))
  joint <- d %>% filter(adt_label != "Unlabelled", llm_label != "Unlabelled")
  joint <- joint %>% mutate(cell = case_when(
    adt_label == "Metastatic" & llm_label == "Metastatic" ~ "both metastatic",
    adt_label == "Metastatic" ~ "ADT only", llm_label == "Metastatic" ~ "LLM only",
    TRUE ~ "both non-metastatic"),
    cell = factor(cell, levels = c("both metastatic", "ADT only", "LLM only", "both non-metastatic")),
    agreement = if_else(as.character(adt_label) == as.character(llm_label), "labels agree", "labels disagree"))
  palette <- c(Metastatic = "#1f5fa8", Local = "#7aa6d4", Unlabelled = "#c6c5c0")
  a <- ggplot(d, aes(fill = llm_label)) +
    geom_rect(aes(xmin = left, xmax = right, ymin = y-.30, ymax = y+.30), color = "white", linewidth = .8) +
    geom_text(data = filter(d, n > 0, !small), aes(x = midpoint, y = y, label = scales::comma(n)),
              color = "white", size = 3.5) +
    geom_segment(data = filter(d, n > 0, small), aes(x = midpoint, xend = midpoint, y = y+.3, yend = label_y-.05),
                 color = "grey55", linewidth = .3) +
    geom_text(data = filter(d, n > 0, small), aes(x = midpoint, y = label_y, label = scales::comma(n)), size = 3) +
    scale_fill_manual(values = palette, breaks = llm_order,
      labels = c("LLM: metastatic", "LLM: non-metastatic", "(no LLM label)"), drop = FALSE) +
    scale_y_continuous(breaks = rev(seq_along(adt_order)),
      labels = c("ADT: metastatic", "ADT: localized", if (length(adt_order) == 3) "(no ADT label)"),
      limits = c(.4, length(adt_order)+.8)) +
    scale_x_continuous(labels = scales::comma, expand = expansion(mult = c(0,.06))) +
    labs(x = "Patients", y = NULL, title = "(a) label overlap") + overview_theme()
  b <- ggplot(joint, aes(cell, n, fill = agreement)) + geom_col(width = .68) +
    geom_text(aes(label = scales::comma(n)), vjust = -.4, size = 3.5) +
    scale_fill_manual(values = c("labels agree" = "#52514e", "labels disagree" = "#eb6834")) +
    scale_x_discrete(labels = c("both\nmetastatic", "ADT\nonly", "LLM\nonly", "both\nnon-metastatic")) +
    scale_y_continuous(expand = expansion(mult = c(0,.16)), labels = scales::comma) +
    labs(x = NULL, y = "Patients", title = "(b) agreement cell size") + overview_theme()
  n <- sum(d$n); paired <- sum(joint$n)
  overview_combine(a, b, "ADT-intent vs LLM metastatic labels",
    sprintf("%s Stage 1 patients; %s (%s) jointly labelled",
      scales::comma(n), scales::comma(paired), if (n) scales::percent(paired/n, accuracy = 1) else "NA"))
}

render_cohort_overview <- function(manifest, config) {
  prepared <- manifest$cohort_overview
  if (is.null(prepared)) stop("Run 04_prep_figure_data.ipynb to prepare the new cohort overview tables.")
  incidence <- figure_read_parquet(file.path(prepared$directory, "incidence.parquet"))
  overlap <- figure_read_parquet(file.path(prepared$directory, "label_overlap.parquet"))
  landmark <- config$forest_landmark
  plots <- list(event_incidence = plot_cohort_event_overview(incidence, config$forest_cohorts, landmark),
                metastatic_label_overlap = plot_stage1_label_overview(overlap))
  base <- file.path(config$fig_root, "ADT", "by_figure", "supplements", "cohort_comparison")
  for (name in names(plots)) {
    if (is.null(plots[[name]])) { message("Cohort overview skipped: ", name, " (no available inputs)"); next }
    leaf <- if (name == "event_incidence") paste0(name, "_landmark", landmark) else name
    destination <- file.path(base, leaf, "platinum__all__incl")
    getOption("compass.figure_capture")(plots[[name]], destination, 16, 7.5, leaf)
    dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
    table <- paste0(destination, ".csv")
    readr::write_csv(if (name == "event_incidence") incidence else overlap, table)
    getOption("compass.figure_table_capture")(table)
  }
}
