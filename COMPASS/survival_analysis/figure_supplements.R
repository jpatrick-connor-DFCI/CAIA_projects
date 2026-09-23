# COMPASS supplementary figures sourced on demand by the pipeline and the
# cached workflow: cohort forest, cohort overview, metastatic-label prep and
# diagnostics. Standalone CLI (notebook 07):
#   Rscript figure_supplements.R input.csv output_root landmark dpi overwrite

# ============================================================================
# ---- PSA/testosterone forest across ADT cohorts (+ notebook 07 CLI) --------
# ============================================================================
# Shared by 05_figures.Rmd and the forest section of 07_cohort_comparison.ipynb.
# One faceted figure per endpoint: PSA and testosterone rows, no delta column.

# Sized to drop into a 16:9 PowerPoint slide (13.33 x 7.5 in) as a single
# full-bleed figure, leaving ~0.4 in of margin on each edge. Both save sites --
# the Rmd pipeline and the notebook adapter below -- use this one constant so a
# deck never receives two differently-shaped copies of the same forest.
COHORT_FOREST_SLIDE_SIZE <- c(width = 12.5, height = 6.8)
cohort_forest_labels <- c(
  adt = "All ADT", adt_noprecastrate = "All ADT; no pre-ADT castrate",
  adt_metastatic_adt = "Metastatic: ADT intent",
  adt_metastatic_adt_noprecastrate = "Metastatic: ADT intent; no pre-ADT castrate",
  adt_metastatic_llm = "Metastatic: LLM",
  adt_metastatic_llm_noprecastrate = "Metastatic: LLM; no pre-ADT castrate"
)

prepare_cohort_forest <- function(frame, endpoint, landmark = 180L) {
  if (!nrow(frame)) return(frame)
  needed <- c("endpoint", "cohort", "feature", "hazard_ratio_per_sd", "ci_lower", "ci_upper")
  if (length(setdiff(needed, names(frame)))) stop("Cohort forest is missing: ", paste(setdiff(needed, names(frame)), collapse = ", "))
  if ("landmark_days" %in% names(frame)) frame <- frame[frame$landmark_days == landmark, , drop = FALSE]
  if (!"lab_name" %in% names(frame)) frame$lab_name <- sub("__.*$", "", frame$feature)
  if (!"feature_stat" %in% names(frame)) frame$feature_stat <- sub("^.*__", "", frame$feature)
  raw_lab <- tolower(frame$lab_name)
  frame$analyte <- ifelse(raw_lab == "psa" | grepl("prostate specific ag", raw_lab, fixed = TRUE), "PSA",
                          ifelse(grepl("testosterone", raw_lab, fixed = TRUE), "Testosterone", NA_character_))
  frame$feature_stat <- tolower(as.character(frame$feature_stat))
  frame <- frame[tolower(frame$endpoint) == endpoint & !is.na(frame$analyte) &
                   frame$feature_stat %in% c("mean", "min", "max", "last"), , drop = FALSE]
  if (!nrow(frame)) return(frame)
  for (column in c("hazard_ratio_per_sd", "ci_lower", "ci_upper")) frame[[column]] <- as.numeric(frame[[column]])
  valid <- is.finite(frame$hazard_ratio_per_sd) & frame$hazard_ratio_per_sd > 0
  if (any(!valid)) warning(sum(!valid), " non-finite/non-positive cohort forest estimates omitted")
  frame <- frame[valid, , drop = FALSE]
  # Keep estimates with unavailable CIs; mark their missing bounds in the caption.
  frame$ci_lower[!is.finite(frame$ci_lower) | frame$ci_lower <= 0] <- NA_real_
  frame$ci_upper[!is.finite(frame$ci_upper) | frame$ci_upper <= 0] <- NA_real_
  frame$significant <- if ("q_value" %in% names(frame)) !is.na(frame$q_value) & frame$q_value < .05 else FALSE
  # Accept the human-readable notebook keys and the canonical run names.
  keys <- c("all", "all +noprecastrate", "metastatic_adt", "metastatic_adt +noprecastrate",
            "metastatic_llm", "metastatic_llm +noprecastrate")
  lookup <- setNames(names(cohort_forest_labels), keys)
  frame$cohort <- as.character(frame$cohort)
  mapped <- unname(lookup[frame$cohort])
  frame$cohort[!is.na(mapped)] <- mapped[!is.na(mapped)]
  extra <- setdiff(unique(frame$cohort), names(cohort_forest_labels))
  order <- c(names(cohort_forest_labels), sort(extra))
  frame$cohort <- factor(frame$cohort, levels = rev(order))
  frame$analyte <- factor(frame$analyte, levels = c("PSA", "Testosterone"))
  frame$feature_stat <- factor(frame$feature_stat, levels = c("mean", "min", "max", "last"),
                               labels = c("Mean", "Minimum", "Maximum", "Last"))
  if (anyDuplicated(frame[c("cohort", "analyte", "feature_stat")]))
    stop("Duplicate cohort/analyte/statistic estimates in cohort forest")
  frame
}

load_cohort_forest <- function(data_root, cohorts, endpoint, landmark) {
  rows <- lapply(cohorts, function(cohort) {
    suffix <- if (endpoint == "platinum") "" else "_nepc"
    path <- file.path(data_root, "survival_analysis", paste0("local_runs_", cohort, suffix),
                      "cox", paste0("landmark_", landmark), "both", "cox_agg_univariate_nobs_adjusted.csv")
    if (!file.exists(path)) { message("Cohort forest: absent result ", cohort, " / ", endpoint); return(NULL) }
    d <- readr::read_csv(path, show_col_types = FALSE)
    d$cohort <- cohort
    d$landmark_days <- landmark
    d
  })
  d <- dplyr::bind_rows(rows)
  prepare_cohort_forest(d, endpoint, landmark)
}

plot_cohort_forest <- function(d, endpoint, landmark = 180L) {
  if (!nrow(d)) return(NULL)
  limits <- range(c(d$hazard_ratio_per_sd, d$ci_lower, d$ci_upper, 1), finite = TRUE)
  limits <- exp(log(limits) + c(-1, 1) * max(diff(log(limits)) * .06, .05))
  labels <- function(x) {
    out <- unname(cohort_forest_labels[x]); out[is.na(out)] <- x[is.na(out)]
    # Four stat columns leave the shared y-axis narrower than the old 2-column
    # layout did, so wrap the cohort names harder to keep them off the panels.
    stringr::str_wrap(out, 22)
  }
  ggplot(d, aes(hazard_ratio_per_sd, cohort, color = analyte)) +
    geom_vline(xintercept = 1, linewidth = .4, linetype = "dashed", color = "grey55") +
    geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper), orientation = "y", width = .2, linewidth = .7, na.rm = TRUE) +
    geom_point(aes(shape = significant), size = 2.8, stroke = .9) +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1), labels = c(`TRUE` = "q < 0.05", `FALSE` = "q ≥ 0.05 or unavailable"), name = NULL) +
    scale_color_manual(values = c(PSA = "#0072B2", Testosterone = "#D55E00"), guide = "none") +
    scale_x_log10(labels = scales::label_number(), limits = limits) +
    scale_y_discrete(labels = labels, drop = TRUE, expand = expansion(add = .7)) +
    # Analyte rows x statistic columns: a 2x4 grid is wide and short, so the
    # figure fills a 16:9 slide instead of the old 4x2 near-square arrangement.
    # No switch = "y" here: that was for the previous orientation, where it moved
    # the statistic strip left. In this one it would drop the analyte strip onto
    # the shared y-axis, on top of the wrapped cohort names.
    facet_grid(analyte ~ feature_stat) +
    labs(x = "Hazard ratio per SD (95% CI; log scale)", y = NULL,
         title = paste("PSA and testosterone associations with", toupper(endpoint)),
         subtitle = sprintf("Treatment landmark: +%d days", landmark),
         caption = paste("Delta and observation-count statistics excluded. Cohorts overlap; estimates shown descriptively.",
                         "Points remain visible when a confidence interval is unavailable.", sep = "\n")) +
    theme_classic(base_size = 11) +
    theme(strip.background = element_blank(), strip.text = element_text(face = "bold", size = 11),
          strip.text.y = element_text(angle = -90),
          axis.text.y = element_text(size = 9),
          axis.text.x = element_text(size = 9), panel.spacing = grid::unit(14, "pt"),
          plot.title.position = "plot", plot.caption.position = "plot", plot.caption = element_text(hjust = 0, size = 9),
          legend.position = "bottom", plot.margin = margin(12, 16, 12, 12))
}


# ============================================================================
# ---- cohort overview -------------------------------------------------------
# ============================================================================
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
  combined <- gridExtra::arrangeGrob(a, b, ncol = 2, bottom = grid::textGrob(
    subtitle, gp = grid::gpar(fontsize = 10)), padding = grid::unit(1, "lines"))
  attr(combined,"compass_compiled") <- TRUE
  combined
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

render_cohort_overview <- function(manifest, config, forest_config=NULL) {
  prepared <- manifest$cohort_overview
  if (is.null(prepared)) stop("Run 04_prep_figure_data.ipynb to prepare the new cohort overview tables.")
  incidence <- figure_read_parquet(file.path(prepared$directory, "incidence.parquet"))
  overlap <- figure_read_parquet(file.path(prepared$directory, "label_overlap.parquet"))
  landmark <- config$forest_landmark
  plots <- list(event_incidence = plot_cohort_event_overview(incidence, config$forest_cohorts, landmark),
                metastatic_label_overlap = plot_stage1_label_overview(overlap))
  base <- file.path(config$fig_root, "ADT", "by_figure", "cohort_comparison")
  for (name in names(plots)) {
    if (is.null(plots[[name]])) { message("Cohort overview skipped: ", name, " (no available inputs)"); next }
    leaf <- if (name == "event_incidence") paste0(name, "_landmark", landmark) else name
    destination <- file.path(base, leaf, "platinum__all__incl")
    getOption("compass.figure_capture")(plots[[name]], destination, 16, 7.5, leaf)
    if(exists("figure_public_path",mode="function")) destination <- figure_public_path(destination)
    dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
    table <- paste0(destination, ".csv")
    readr::write_csv(if (name == "event_incidence") incidence else overlap, table)
    getOption("compass.figure_table_capture")(table)
  }
  # Figure 6 combines this job's incidence table with the cohort forest. The
  # canonical ADT job also renders the standalone forest, but manuscript
  # assembly is job-local, so register the same source plot here without
  # publishing a duplicate standalone scene.
  if(!is.null(forest_config)) {
    stopifnot(as.integer(forest_config$landmark)==as.integer(landmark))
    forest <- load_cohort_forest(config$data_root,forest_config$cohorts,"platinum",landmark)
    if(nrow(forest)) {
      forest_plot <- plot_cohort_forest(forest,"platinum",landmark)
      manuscript_capture <- getOption("compass.manuscript_capture")
      if(!is.function(manuscript_capture)) stop("Manuscript capture is unavailable for cohort Figure 6")
      manuscript_capture(forest_plot,
        sprintf("cohort_forest_platinum_landmark%d",landmark))
    } else message("Cohort overview: no platinum forest available for manuscript Figure 6")
  }
}

# ============================================================================
# ---- metastatic-label preparation (R fallback for notebook 04's Polars prep) 
# ============================================================================
# R-only preparation for 05_figures.Rmd. Uses the existing intent labels,
# metastatic-diagnosis LLM task and dated regex stages; no model refits.
metastatic_parquet_backend <- function() {
  if (requireNamespace("arrow", quietly = TRUE)) return("arrow")
  if (requireNamespace("nanoparquet", quietly = TRUE)) return("nanoparquet")
  stop('Metastatic figures need an R Parquet reader. Run install.packages("nanoparquet") ',
       'in this R environment, then re-knit. The R package arrow is also supported.', call. = FALSE)
}

read_metastatic_parquet <- function(path, columns) {
  if (metastatic_parquet_backend() == "arrow")
    return(as.data.frame(arrow::read_parquet(path, col_select = columns)))
  nanoparquet::read_parquet(path, col_select = columns)
}

metastatic_normalize_id <- function(frame) {
  id <- suppressWarnings(as.numeric(as.character(frame$DFCI_MRN)))
  keep <- is.finite(id)
  frame <- frame[keep, , drop = FALSE]
  frame$DFCI_MRN <- format(trunc(id[keep]), scientific = FALSE, trim = TRUE)
  frame
}

metastatic_datetime <- function(values) {
  if (inherits(values, "POSIXt")) return(as.POSIXct(values, tz = "UTC"))
  if (inherits(values, "Date")) return(as.POSIXct(values, tz = "UTC"))
  values <- trimws(as.character(values))
  parsed <- suppressWarnings(readr::parse_datetime(values))
  for (format in c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y")) {
    missing <- is.na(parsed) & !is.na(values) & nzchar(values)
    if (!any(missing)) break
    parsed[missing] <- suppressWarnings(readr::parse_datetime(values[missing], format = format))
  }
  parsed
}

metastatic_stage_number <- function(values) {
  key <- toupper(trimws(as.character(values)))
  lookup <- c(`1`=1L, `2`=2L, `3`=3L, `4`=4L, I=1L, II=2L, III=3L, IV=4L,
              `1.0`=1L, `2.0`=2L, `3.0`=3L, `4.0`=4L)
  unname(lookup[key])
}

metastatic_collapse_stage <- function(values) {
  stage <- metastatic_stage_number(values)
  ifelse(is.na(stage), NA_character_, ifelse(stage == 4L, "Metastatic", "Local"))
}

build_metastatic_labels <- function(intent, notes, llm, analysis_anchors = NULL) {
  intent <- metastatic_normalize_id(intent) %>%
    transmute(DFCI_MRN, ADT_FIRST_DATE = metastatic_datetime(ADT_FIRST_DATE),
      ADT_LABEL = unname(c(LOCALIZED_ADJUVANT = "Local", METASTATIC = "Metastatic")[as.character(ADT_INTENT)]))
  if (anyDuplicated(intent$DFCI_MRN)) stop("ADT intent input must contain exactly one row per patient")
  if (is.null(analysis_anchors)) {
    # Standalone use without an analysis cohort retains the intent-file clock.
    intent <- intent %>% mutate(ANALYSIS_ANCHOR_DATE = ADT_FIRST_DATE)
  } else {
    if (!all(c("DFCI_MRN", "TREATMENT_ANCHOR_DATE") %in% names(analysis_anchors)))
      stop("Metastatic supplement requires patient DFCI_MRN and TREATMENT_ANCHOR_DATE")
    anchors <- metastatic_normalize_id(analysis_anchors) %>%
      transmute(DFCI_MRN, ANALYSIS_ANCHOR_DATE = metastatic_datetime(TREATMENT_ANCHOR_DATE))
    if (anyDuplicated(anchors$DFCI_MRN)) stop("Analysis anchors must contain exactly one row per patient")
    # Match by patient ID, never row order. Preserve patients without an intent
    # label so their independent LLM and regex evidence is still represented.
    intent <- anchors %>% left_join(intent, by = "DFCI_MRN")
  }
  intent <- intent %>% mutate(ANCHOR_DELTA_DAYS = as.numeric(
    as.Date(ANALYSIS_ANCHOR_DATE, tz = "UTC") - as.Date(ADT_FIRST_DATE, tz = "UTC")))
  llm <- metastatic_normalize_id(llm) %>%
    mutate(.verdict = tolower(as.character(has_metastatic_disease))) %>%
    group_by(DFCI_MRN) %>% summarise(LLM_LABEL = case_when(
      any(.verdict %in% c("true", "1", "1.0")) ~ "Metastatic",
      any(.verdict %in% c("false", "0", "0.0")) ~ "Local",
      TRUE ~ NA_character_), .groups = "drop")
  notes <- metastatic_normalize_id(notes) %>%
    transmute(DFCI_MRN, .date = metastatic_datetime(EVENT_DATE),
              .stage = metastatic_stage_number(DERIVED_STAGE_MERGED)) %>%
    filter(!is.na(.date), !is.na(.stage)) %>%
    inner_join(intent %>% select(DFCI_MRN, ANALYSIS_ANCHOR_DATE), by = "DFCI_MRN") %>%
    mutate(.days = trunc(as.numeric(difftime(.date, ANALYSIS_ANCHOR_DATE, units = "days"))))
  before <- notes %>% filter(.days <= 0)
  nearest <- before %>% filter(.days >= -365) %>% arrange(desc(.days), desc(.stage)) %>%
    distinct(DFCI_MRN, .keep_all = TRUE) %>% transmute(DFCI_MRN, REGEX_STAGE = .stage)
  max_before <- before %>% group_by(DFCI_MRN) %>%
    summarise(REGEX_MAX_BEFORE_STAGE = if (length(.stage)) max(.stage) else NA_integer_, .groups = "drop")
  max_after <- notes %>% filter(.days > 0) %>% group_by(DFCI_MRN) %>%
    summarise(REGEX_MAX_AFTER_STAGE = if (length(.stage)) max(.stage) else NA_integer_, .groups = "drop")
  # Maximum across the entire record: pre- and post-ADT staging pooled, with no
  # anchor window. Post-ADT progression therefore counts toward the label, so
  # this is not a point-in-time status at ADT the way REGEX_LABEL is.
  max_any <- notes %>% group_by(DFCI_MRN) %>%
    summarise(REGEX_MAX_ANY_STAGE = if (length(.stage)) max(.stage) else NA_integer_, .groups = "drop")
  intent %>% left_join(llm, by = "DFCI_MRN") %>% left_join(nearest, by = "DFCI_MRN") %>%
    left_join(max_before, by = "DFCI_MRN") %>% left_join(max_after, by = "DFCI_MRN") %>%
    left_join(max_any, by = "DFCI_MRN") %>%
    mutate(REGEX_LABEL = metastatic_collapse_stage(REGEX_STAGE),
           REGEX_MAX_BEFORE = metastatic_collapse_stage(REGEX_MAX_BEFORE_STAGE),
           REGEX_MAX_AFTER = metastatic_collapse_stage(REGEX_MAX_AFTER_STAGE),
           REGEX_MAX_ANY = metastatic_collapse_stage(REGEX_MAX_ANY_STAGE)) %>%
    select(DFCI_MRN, ADT_FIRST_DATE, ANALYSIS_ANCHOR_DATE, ANCHOR_DELTA_DAYS,
           ADT_LABEL, LLM_LABEL, REGEX_LABEL, REGEX_MAX_BEFORE, REGEX_MAX_AFTER,
           REGEX_MAX_ANY)
}

metastatic_icd_site <- function(values) {
  code <- gsub("[^A-Z0-9]", "", toupper(trimws(as.character(values))))
  valid <- !is.na(code) & nchar(code) >= 4L & grepl("^C7[789]", code) & code != "C799"
  site <- rep(NA_character_, length(code))
  site[valid] <- "other"
  # Mirrors validate_adt_intent.py: spinal/cranial-nerve codes remain "other";
  # unspecified C79.9 and C7B neuroendocrine codes do not count as organ sites.
  prefixes <- c(C795="bone", C7931="brain", C7932="brain", C797="adrenal",
                C780="lung", C782="lung", C787="liver", C786="peritoneal", C77="node")
  for (prefix in names(prefixes)) site[valid & startsWith(code, prefix)] <- prefixes[[prefix]]
  site
}

metastatic_burden_at_adt <- function(icds, labels) {
  groups <- c("brain", "bone", "liver", "lung", "node", "adrenal", "peritoneal", "other")
  anchors <- labels %>% select(DFCI_MRN, ANALYSIS_ANCHOR_DATE) %>% filter(!is.na(ANALYSIS_ANCHOR_DATE))
  coded <- metastatic_normalize_id(icds) %>% inner_join(anchors, by = "DFCI_MRN") %>%
    mutate(.date = metastatic_datetime(START_DT), .site = metastatic_icd_site(DIAGNOSIS_ICD10_CD)) %>%
    filter(!is.na(.date), !is.na(.site), .date <= ANALYSIS_ANCHOR_DATE) %>% distinct(DFCI_MRN, .site)
  base <- labels %>% distinct(DFCI_MRN)
  for (group in groups)
    base[[paste0("MET_SITE_", group)]] <- as.integer(base$DFCI_MRN %in% coded$DFCI_MRN[coded$.site == group])
  # With no analysis date, "before ADT" is unknown rather than zero burden.
  unavailable <- !base$DFCI_MRN %in% anchors$DFCI_MRN
  for (column in paste0("MET_SITE_", groups)) base[[column]][unavailable] <- NA_integer_
  base$N_MET_SITES <- rowSums(base[paste0("MET_SITE_", groups)])
  base
}

prepare_metastatic_figure_labels <- function(config, analysis_anchors = NULL) {
  intent <- readr::read_csv(config$intent, show_col_types = FALSE,
    col_select = all_of(c("DFCI_MRN", "ADT_FIRST_DATE", "ADT_INTENT")),
    col_types = readr::cols(.default = readr::col_character()))
  notes <- read_metastatic_parquet(config$stage, c("DFCI_MRN", "EVENT_DATE", "DERIVED_STAGE_MERGED"))
  llm <- read_metastatic_parquet(config$llm, c("DFCI_MRN", "has_metastatic_disease"))
  labels <- build_metastatic_labels(intent, notes, llm, analysis_anchors)
  if (!is.null(config$icd) && file.exists(config$icd)) {
    icds <- readr::read_csv(config$icd, show_col_types = FALSE,
      col_select = all_of(c("DFCI_MRN", "DIAGNOSIS_ICD10_CD", "START_DT")),
      col_types = readr::cols(.default = readr::col_character()))
    labels <- labels %>% left_join(metastatic_burden_at_adt(icds, labels), by = "DFCI_MRN")
  }
  labels
}

# ============================================================================
# ---- metastatic-label diagnostics ------------------------------------------
# ============================================================================
# Standalone ADT/LLM/regex diagnostics used by 05_figures.Rmd. All writes use
# the main pipeline's save_fig, so progress, atomic saves and overwrite apply.
# The three requested confusion matrices: each of the LLM and ADT metastatic
# labels against the regex maximum stage across the record, plus ADT vs LLM.
metastatic_label_pairs <- function() {
  list(llm_vs_regex_max_any = c("LLM_LABEL", "REGEX_MAX_ANY"),
       adt_vs_regex_max_any = c("ADT_LABEL", "REGEX_MAX_ANY"),
       adt_vs_llm = c("ADT_LABEL", "LLM_LABEL"))
}

# The coverage, burden, site, KM and trajectory panels are retained but no
# longer emitted. Set COMPASS_METASTATIC_EXTRA_PANELS=1 to restore them.
metastatic_extra_panels_enabled <- function() {
  !identical(tolower(trimws(Sys.getenv("COMPASS_METASTATIC_EXTRA_PANELS", ""))), "") &&
    !identical(tolower(trimws(Sys.getenv("COMPASS_METASTATIC_EXTRA_PANELS", ""))), "0")
}

metastatic_pair_counts <- function(labels, first, second) {
  levels <- c("Local", "Metastatic")
  paired <- labels %>% filter(.data[[first]] %in% levels, .data[[second]] %in% levels)
  counts <- paired %>% transmute(first = .data[[first]], second = .data[[second]]) %>%
    count(first, second, name = "n") %>%
    complete(first = levels, second = levels, fill = list(n = 0L)) %>%
    group_by(first) %>% mutate(row_n = sum(n), fraction = if_else(row_n > 0, n / row_n, NA_real_)) %>%
    ungroup()
  list(counts = counts, paired = nrow(paired), missing = nrow(labels) - nrow(paired),
       agreement = if (nrow(paired)) mean(paired[[first]] == paired[[second]]) else NA_real_)
}

metastatic_supplement_stems <- function() {
  stems <- names(metastatic_label_pairs())
  if (metastatic_extra_panels_enabled())
    stems <- c("coverage", stems,
      outer(c("adt", "llm"), c("burden", "sites", "km_death", "km_platinum", "km_nepc",
                                "trajectory_psa", "trajectory_testosterone"), paste, sep = "_"))
  paste0("adt_labels_", stems)
}

render_metastatic_supplements <- function(config, patient_df, labs, save_panel,
                                         notify, show = FALSE) {
  notify("stage", "Supplement: ADT, LLM metastatic and collapsed regex stage")
  prepared_path <- getOption("compass.figure_data_manifest")$metastatic_labels
  required <- if (!is.null(prepared_path)) prepared_path else c(intent = config$intent, stage = config$stage, llm = config$llm)
  missing <- required[!file.exists(required)]
  if (length(missing)) {
    warning("Metastatic-label supplement skipped; missing input(s): ",
            paste(missing, collapse = ", "),
            ". ADT labels are generated by compass_pipeline.build_adt_intent_mrn_lists().")
    return(invisible(NULL))
  }
  if (!is.null(prepared_path)) {
    labels <- figure_read_parquet(prepared_path)
    ids <- format(trunc(suppressWarnings(as.numeric(patient_df$DFCI_MRN))), scientific = FALSE, trim = TRUE)
    labels <- labels %>% filter(DFCI_MRN %in% ids)
  } else {
    prep <- new.env(parent = environment())
    sys.source(config$prepare_script, envir = prep)
    labels <- prep$prepare_metastatic_figure_labels(config, analysis_anchors = patient_df)
  }
  comparable <- !is.na(labels$ANCHOR_DELTA_DAYS)
  different <- comparable & labels$ANCHOR_DELTA_DAYS != 0
  message(sprintf(paste0("Metastatic supplement: stage/ICD windows use longitudinal TREATMENT_ANCHOR_DATE; ",
                         "%d of %d comparable intent dates differ. Original dates retained; ",
                         "%d patients lack an analysis anchor."),
                  sum(different), sum(comparable), sum(is.na(labels$ANALYSIS_ANCHOR_DATE))))
  names_pretty <- c(ADT_LABEL = "ADT intent", LLM_LABEL = "LLM metastatic status",
                    REGEX_LABEL = "Regex stage nearest before ADT (365 days)",
                    REGEX_MAX_BEFORE = "Regex maximum stage before ADT",
                    REGEX_MAX_AFTER = "Regex maximum stage after ADT",
                    REGEX_MAX_ANY = "Regex maximum stage across record")
  colors <- c(Local = "#0b6ba8", Metastatic = "#c1272d", Unclassified = "#999999")
  caption <- paste("Canonical ADT analysis cohort; pre-ADT castrate patients included.",
                   "Regex I–III = local; IV = metastatic. ADT/LLM labels use full observed history.",
                   "Stage/ICD windows use the longitudinal analysis ADT date.",
                   "Agreement is descriptive; definitions and observation windows differ.", sep = "\n")
  emit <- function(plot, name, width = 8, height = 5.5) {
    save_panel(plot, paste0("adt_labels_", name), width, height)
    if (show) print(plot)
  }
  if (metastatic_extra_panels_enabled()) {
    coverage <- labels %>% select(all_of(names(names_pretty))) %>%
      pivot_longer(everything(), names_to = "source", values_to = "label") %>%
      mutate(label = coalesce(label, "Unclassified")) %>% count(source, label)
    emit(ggplot(coverage, aes(x = source, y = n, fill = label)) + geom_col() +
           geom_text(aes(label = n), position = position_stack(vjust = .5), size = 3) +
           scale_x_discrete(labels = names_pretty) + scale_fill_manual(values = colors) +
           coord_flip() + labs(x = NULL, y = "Patients", fill = NULL,
                              title = "Label coverage and local/metastatic composition", caption = caption) +
           theme_bw(), "coverage", 9, 6)
  }

  for (name in names(metastatic_label_pairs())) {
    pair <- metastatic_label_pairs()[[name]]
    result <- metastatic_pair_counts(labels, pair[1], pair[2])
    d <- result$counts %>% mutate(text = if_else(row_n > 0,
      sprintf("n = %s\n%.1f%%", n, 100 * fraction), sprintf("n = %s\nNA", n)))
    subtitle <- sprintf("Both labelled: %s of %s; missing either: %s; agreement: %s",
      result$paired, nrow(labels), result$missing,
      if (result$paired) sprintf("%.1f%%", 100 * result$agreement) else "NA")
    emit(ggplot(d, aes(second, first, fill = fraction)) + geom_tile(color = "white") +
           geom_text(aes(label = text), size = 4) +
           scale_fill_gradient(low = "white", high = "#56B4E9", limits = c(0, 1), na.value = "grey90") +
           scale_x_discrete(limits = c("Local", "Metastatic")) +
           scale_y_discrete(limits = c("Metastatic", "Local")) +
           coord_fixed(ratio = 1) +
           labs(x = names_pretty[[pair[2]]], y = names_pretty[[pair[1]]],
                title = "Local versus metastatic label agreement", subtitle = subtitle,
                fill = "Row fraction", caption = caption) + theme_bw(), name, 7, 6)
  }

  # Merge the old ADT filtering diagnostics, also displaying the LLM definition.
  # Reuse the parent's patient and androgen caches; never parse the large CSV again.
  # Retained but not emitted by default; see metastatic_extra_panels_enabled().
  for (source in if (metastatic_extra_panels_enabled()) c("ADT_LABEL", "LLM_LABEL") else character()) {
    slug <- if (source == "ADT_LABEL") "adt" else "llm"
    grouped <- labels %>% mutate(label = .data[[source]]) %>% filter(!is.na(label))
    if ("N_MET_SITES" %in% names(grouped)) {
      burden <- grouped %>% filter(!is.na(N_MET_SITES)) %>%
        mutate(sites = if_else(N_MET_SITES >= 2, "2+", as.character(N_MET_SITES))) %>% count(label, sites)
      emit(ggplot(burden, aes(sites, n, fill = label)) + geom_col(position = "dodge") +
             scale_fill_manual(values = colors) + labs(x = "Distinct ICD metastatic sites at ADT", y = "Patients",
             fill = names_pretty[[source]], caption = caption) + theme_bw(), paste0(slug, "_burden"))
      sites <- grouped %>% select(label, starts_with("MET_SITE_")) %>%
        pivot_longer(-label, names_to = "site", values_to = "present") %>%
        filter(!is.na(present)) %>% group_by(label, site) %>% summarise(fraction = mean(present), .groups = "drop")
      if (nrow(sites)) emit(ggplot(sites, aes(site, fraction, fill = label)) + geom_col(position = "dodge") +
        coord_flip() + scale_fill_manual(values = colors) + labs(x = NULL, y = "Proportion with coded site",
        fill = names_pretty[[source]], caption = caption) + theme_bw(), paste0(slug, "_sites"))
    }
    patients <- patient_df %>% mutate(DFCI_MRN = as.character(DFCI_MRN)) %>%
      inner_join(grouped %>% select(DFCI_MRN, label), by = "DFCI_MRN")
    for (endpoint in c("death", "platinum", "nepc")) {
      duration_col <- paste0("t_", endpoint)
      event_col <- toupper(endpoint)
      if (!all(c(duration_col, event_col) %in% names(patients))) next
      d <- patients %>% transmute(label, duration = as.numeric(.data[[duration_col]]),
                                  event = as.numeric(.data[[event_col]]))
      # Death uses last contact for censored patients, as the parent KM does.
      if (endpoint == "death" && "t_last_contact" %in% names(patients))
        d$duration <- ifelse(d$event == 0, as.numeric(patients$t_last_contact), d$duration)
      d <- d %>% filter(is.finite(duration), duration > 0, event %in% c(0, 1))
      if (!nrow(d)) next
      fit <- survival::survfit(survival::Surv(duration, event) ~ label, data = d)
      sf <- summary(fit, censored = TRUE)
      strata <- if (is.null(sf$strata)) rep(unique(d$label), length(sf$time)) else
        sub("^label=", "", as.character(sf$strata))
      curve <- tibble(time = sf$time / 365.25, survival = sf$surv, label = strata)
      curve <- bind_rows(tibble(time = 0, survival = 1, label = unique(d$label)), curve)
      counts <- d %>% group_by(label) %>% summarise(n = n(), events = sum(event), .groups = "drop")
      emit(ggplot(curve, aes(time, survival, color = label)) + geom_step() +
        scale_color_manual(values = colors) + coord_cartesian(ylim = c(0, 1)) +
        labs(x = "Years from ADT", y = if (endpoint == "death") "Overall survival" else "Event-free probability",
             title = paste(toupper(endpoint), "by", names_pretty[[source]]),
             subtitle = paste(sprintf("%s: n=%s, events=%s", counts$label, counts$n, counts$events), collapse = "; "),
             color = NULL, caption = caption) + theme_bw(), paste0(slug, "_km_", endpoint))
    }
    for (lab in c("PSA", "Testosterone")) {
      d <- labs %>% filter(LAB_GROUP == lab, is.finite(LAB_VALUE), LAB_VALUE >= 0,
                          t_rel >= -365, t_rel <= 5 * 365) %>%
        mutate(DFCI_MRN = as.character(DFCI_MRN), bin = floor(t_rel / 180) * 180) %>%
        inner_join(grouped %>% select(DFCI_MRN, label), by = "DFCI_MRN") %>%
        group_by(DFCI_MRN, label, bin) %>% summarise(value = mean(LAB_VALUE), .groups = "drop") %>%
        group_by(label, bin) %>% summarise(n = n(), median = median(value),
          q1 = quantile(value, .25), q3 = quantile(value, .75), .groups = "drop") %>% filter(n >= 5)
      if (!nrow(d)) next
      if (lab == "PSA") d <- d %>% mutate(across(c(median, q1, q3), ~ pmax(.x, .01)))
      p <- ggplot(d, aes((bin + 90) / 365.25, median, color = label, fill = label)) +
        geom_ribbon(aes(ymin = q1, ymax = q3), alpha = .15, color = NA) + geom_line() +
        geom_vline(xintercept = 0, linetype = "dashed") +
        scale_color_manual(values = colors) + scale_fill_manual(values = colors) +
        labs(x = "Years from ADT", y = lab, color = NULL, fill = NULL,
             title = paste(lab, "by", names_pretty[[source]]),
             subtitle = "180-day bins; patient means then median/IQR; at least 5 patients per bin",
             caption = caption) + theme_bw()
      if (lab == "PSA") p <- p + scale_y_log10()
      if (lab == "Testosterone") p <- p + geom_hline(yintercept = c(20, 50), linetype = "dotted")
      emit(p, paste0(slug, "_trajectory_", tolower(lab)))
    }
  }
  invisible(labels)
}

# ============================================================================
# ---- notebook 07 forest CLI ------------------------------------------------
# ============================================================================
# Notebook adapter: Rscript figure_supplements.R input.csv output_root landmark dpi overwrite
# The notebook and Rmd deliberately share the same renderer and destinations.
if (sys.nframe() == 0L) {
  suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(tidyr)})
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 5L) stop("Expected input.csv output_root landmark dpi overwrite")
  frame <- readr::read_csv(args[1], show_col_types = FALSE)
  landmark <- as.integer(args[3]); dpi <- as.numeric(args[4]); overwrite <- tolower(args[5]) == "true"
  for (endpoint in c("platinum", "nepc")) {
    d <- prepare_cohort_forest(frame, endpoint, landmark)
    p <- plot_cohort_forest(d, endpoint, landmark)
    if (is.null(p)) next
    directory <- file.path(args[2], "ADT", "cohort")
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    destination <- file.path(directory, sprintf("cohort_forest_lm%d__%s__all__incl.png", landmark, endpoint))
    # PIL-style header and end marker check without loading a graphics device.
    complete <- function(path) {
      if (!file.exists(path) || file.size(path) < 20) return(FALSE)
      con <- file(path, "rb"); on.exit(close(con))
      head <- readBin(con, "raw", 8); seek(con, -12, "end")
      tail <- readBin(con, "raw", 12)
      identical(head, as.raw(c(137, 80, 78, 71, 13, 10, 26, 10))) &&
        identical(tail[5:8], charToRaw("IEND"))
    }
    if (!overwrite && complete(destination)) { message("Skipped ", destination); next }
    temporary <- tempfile(tmpdir = directory, fileext = ".png")
    tryCatch({
      ggsave(temporary, p, width = COHORT_FOREST_SLIDE_SIZE[["width"]],
             height = COHORT_FOREST_SLIDE_SIZE[["height"]],
             dpi = dpi, bg = "white", device = if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png else "png")
      if (!complete(temporary) || !file.rename(temporary, destination)) stop("Could not publish ", destination)
    }, finally = unlink(temporary))
    message("Wrote ", destination)
  }
}
