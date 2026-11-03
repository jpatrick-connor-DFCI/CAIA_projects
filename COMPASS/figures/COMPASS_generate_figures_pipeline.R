# Per-cohort figure-generation pipeline called from COMPASS/figures/run_figures.R once per
# cohort in the same R session.
#
# Required packages (install once on the cluster):
#   install.packages(c("tidyverse", "survival", "survminer", "broom",
#                       "ggrepel", "jsonlite", "scales", "mgcv", "systemfonts"))
#   # optional: install.packages(c("ggpattern", "ragg"))
#   #   ggpattern -> striped baseline bars in Fig 4a
#   #   ragg      -> text measurement device (falls back to default if absent)
suppressPackageStartupMessages({
  library(tidyverse)   # dplyr, ggplot2, readr, tidyr, purrr, stringr, forcats
  library(survival)
  library(survminer)
  library(broom)
  library(ggrepel)
  library(jsonlite)
  library(scales)
})
source(file.path(dirname(tail(Filter(Negate(is.null), lapply(sys.frames(), function(frame) frame$ofile)), 1)[[1]]), "figure_typography.R"), local=TRUE)

# Shared ggplot theme: publication typography, thin spines, white background.
theme_fig <- function(base_size = FIGURE_FONT_SIZES[["body"]]) {
  theme_classic(base_size = base_size, base_family = FIGURE_FONT) +
    theme(
      plot.title        = element_text(face = "bold", size = base_size + 1),
      plot.subtitle     = element_text(size = base_size - 1, color = "#52514e"),
      axis.title        = element_text(size = base_size),
      axis.text         = element_text(size = base_size - 1.5, color = "#1a1a1a"),
      legend.key        = element_blank(),
      legend.background = element_blank(),
      legend.text       = element_text(size = base_size - 2),
      axis.line         = element_line(linewidth = 0.5),
      axis.ticks        = element_line(linewidth = 0.5),
      plot.background    = element_rect(fill = "white", color = NA),
      panel.background   = element_rect(fill = "white", color = NA),
      legend.position    = "bottom",
      legend.box         = "horizontal",
      legend.key.width   = grid::unit(14, "pt"),
      plot.margin        = margin(12, 16, 12, 12)
    ) + figure_font_theme()
}
theme_set(theme_fig())

# Keep equal measurements together; never divide tied values by patient order.
figure_tertiles <- function(values) {
  values <- suppressWarnings(as.numeric(values))
  valid <- is.finite(values)
  result <- rep(NA_character_, length(values))
  if (sum(valid) < 3L) return(result)
  cuts <- as.numeric(quantile(values[valid], c(1/3, 2/3), names = FALSE))
  if (cuts[1] >= cuts[2]) return(result)
  result[valid] <- as.character(cut(values[valid], c(-Inf, cuts, Inf),
    labels = c("Low tertile", "Middle tertile", "High tertile"), right = TRUE))
  if (length(unique(result[valid])) != 3L) result[] <- NA_character_
  attr(result, "cutpoints") <- cuts
  result
}

# Bottom vs top `fraction` (default 20%); the middle is dropped rather than
# plotted. Same tie contract as figure_tertiles: equal measurements never land
# in opposite arms. This matters most for post-ADT PSA, where a pile of tied
# values sits at the assay floor and can straddle the bottom cut -- splitting
# those by row order would manufacture separation between clinically identical
# patients.
figure_extreme_quintiles <- function(values, fraction = 0.2) {
  stopifnot(fraction > 0, fraction < 0.5)
  values <- suppressWarnings(as.numeric(values))
  valid <- is.finite(values)
  result <- rep(NA_character_, length(values))
  if (sum(valid) < 2L) return(result)
  cuts <- as.numeric(quantile(values[valid], c(fraction, 1 - fraction), names = FALSE))
  if (cuts[1] >= cuts[2]) return(result)
  # Closed at the bottom cut, open at the top: ties at either edge stay whole.
  low <- valid & values <= cuts[1]
  high <- valid & values > cuts[2]
  labels <- figure_extreme_labels(fraction)
  result[low] <- labels[1]
  result[high] <- labels[2]
  if (!any(low) || !any(high)) return(rep(NA_character_, length(values)))
  attr(result, "cutpoints") <- cuts
  result
}

figure_extreme_labels <- function(fraction) {
  if (isTRUE(all.equal(fraction, 1 / 3))) return(c("Bottom tertile", "Top tertile"))
  pct <- round(100 * fraction)
  c(sprintf("Bottom %d%%", pct), sprintf("Top %d%%", pct))
}

# PSA/testosterone extreme-group KM panels compare the bottom vs top 10%.
LAB_EXTREME_FRACTION <- 0.1

figure_gleason_groups <- function(values) {
  x <- suppressWarnings(as.numeric(values))
  valid <- is.finite(x) & x == floor(x) & x >= 2 & x <= 10
  out <- rep(NA_character_, length(x))
  out[valid] <- ifelse(x[valid] <= 7, "Gleason ≤7", "Gleason ≥8")
  out
}

significant_mutation_features <- function(results) {
  if (!all(c("feature", "q_value") %in% names(results))) return(character())
  selected <- !is.na(results$q_value) & results$q_value < .05 &
    grepl("_(SNV|SV|AMP|DEL|ANY)$", as.character(results$feature))
  unique(as.character(results$feature[selected]))
}

# Input durations are already rebased by the landmark/index builders. In
# particular, do not combine index-relative t_platinum with ADT-relative t_death.
figure_platinum_strata <- function(frame, groups) {
  stopifnot(length(groups) == nrow(frame))
  if (anyDuplicated(frame$DFCI_MRN)) stop("KM input must have one row per patient")
  time <- suppressWarnings(as.numeric(frame$t_platinum))
  event <- suppressWarnings(as.numeric(frame$PLATINUM))
  keep <- is.finite(time) & time > 0 & event %in% c(0, 1) & !is.na(groups)
  tibble(DFCI_MRN = as.character(frame$DFCI_MRN[keep]), time = time[keep],
         event = event[keep], stratum = as.character(groups[keep]))
}

plot_stratified_platinum <- function(d, title, origin, group_order = unique(d$stratum),
                                     note = NULL, colors = NULL, ci = TRUE,
                                     ylab = "Platinum-free probability", legend_ncol = 1,
                                     zoom_y = FALSE) {
  if (!nrow(d) || length(unique(d$stratum)) < 2L) return(NULL)
  group_order <- group_order[group_order %in% d$stratum]
  d$stratum <- factor(d$stratum, levels = group_order)
  curves <- bind_rows(lapply(group_order, function(group) {
    z <- d[d$stratum == group, ]
    fit <- survival::survfit(survival::Surv(time, event) ~ 1, data = z)
    tibble(stratum = group, time = c(0, fit$time), survival = c(1, fit$surv),
           lower = c(1, fit$lower), upper = c(1, fit$upper))
  }))
  counts <- d %>% group_by(stratum, .drop = TRUE) %>%
    summarise(n = n(), events = sum(event), .groups = "drop")
  logrank <- tryCatch({
    test <- survival::survdiff(survival::Surv(time, event) ~ stratum, data = d)
    pchisq(test$chisq, length(test$n) - 1, lower.tail = FALSE)
  }, error = function(e) NA_real_)
  plot_km_curves(curves, counts, logrank, title, origin, group_order, note, colors, ci, ylab,
                 legend_ncol, zoom_y)
}

# Platinum is rare, so a 0-1 axis flattens every curve against the top. Zoomed
# panels start just below the lowest plotted KM estimate (in 0.05 steps); CI
# ribbons below that are clipped, not dropped.
km_zoom_ylim <- function(survival, pad = 0.02, step = 0.05) {
  lo <- suppressWarnings(min(survival, na.rm = TRUE))
  if (!is.finite(lo)) return(c(0, 1.02))
  c(max(0, floor((lo - pad) / step) * step), 1.005)
}

# Draws already-estimated KM curves (stratum, time, survival, lower, upper), e.g.
# the federated aggregate export, in the same style as locally fitted ones.
plot_km_curves <- function(curves, counts, logrank, title, origin, group_order = unique(curves$stratum),
                           note = NULL, colors = NULL, ci = TRUE,
                           ylab = "Platinum-free probability", legend_ncol = 1,
                           zoom_y = FALSE) {
  group_order <- group_order[group_order %in% curves$stratum]
  curves$stratum <- factor(curves$stratum, levels = group_order)
  legend_labels <- setNames(sprintf("%s (n=%s; events=%s)", counts$stratum,
                                    counts$n, counts$events), as.character(counts$stratum))
  default_colors <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                      "#56B4E9", "#F0E442", "#000000")
  colors <- if (is.null(colors)) setNames(default_colors[seq_along(group_order)], group_order) else {
    # Named overrides first; unnamed groups fall back to the default sequence.
    fill <- setdiff(default_colors, colors[names(colors) %in% group_order])
    missing <- setdiff(group_order, names(colors))
    c(colors[names(colors) %in% group_order], setNames(fill[seq_along(missing)], missing))[group_order]
  }
  ggplot(curves, aes(time, survival, color = stratum, fill = stratum)) +
    (if (ci) geom_ribbon(aes(ymin = lower, ymax = upper), alpha = .08, color = NA,
                         show.legend = FALSE, na.rm = TRUE)) +
    geom_step(linewidth = .85) +
    scale_color_manual(values = colors, breaks = group_order, labels = legend_labels, name = NULL) +
    scale_fill_manual(values = colors, guide = "none") +
    coord_cartesian(ylim = if (zoom_y) km_zoom_ylim(curves$survival) else c(0, 1.02)) +
    labs(x = paste("Days from", origin), y = ylab, title = title,
         subtitle = if (is.finite(logrank)) sprintf("Log-rank p = %.3g", logrank) else "Log-rank p unavailable",
         caption = note) + theme_fig() +
    guides(color = guide_legend(ncol = legend_ncol)) +
    theme(legend.position = "bottom", legend.text = element_text(size = 10))
}

# Binned mean +/- 95% CI trajectories (t_mid, mean, ci_lo, ci_hi, stratum).
# Shared by the local patient-level panels and the federated aggregate export.
PLATINUM_TRAJECTORY_COLORS <- c(`0` = "#1f3a93", `1` = "#8e1c2b")
plot_trajectory_ci <- function(binned, lab_group, title, stratum_legend = NULL,
                               stratum_colors = NULL, log_scale = FALSE,
                               pre_days = 365.25, post_days = 5 * 365.25, bin_width = 180,
                               min_bin_patients = 10, anchor_label = "ADT initiation") {
  if (is.null(stratum_legend)) stratum_legend <- c(`0` = "Non-platinum", `1` = "Platinum")
  if (is.null(stratum_colors)) stratum_colors <- PLATINUM_TRAJECTORY_COLORS
  binned <- binned %>% mutate(stratum = factor(stratum)) %>% arrange(t_mid)
  ggplot(binned, aes(t_mid, mean, color = stratum, fill = stratum)) +
    geom_vline(xintercept = 0, color = "#2c3e50", linetype = "dotted", linewidth = 1, alpha = 0.6) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.2, color = NA) +
    geom_line(linewidth = 0.8) + geom_point(size = 1.6) +
    scale_color_manual(values = stratum_colors, labels = stratum_legend, name = NULL) +
    scale_fill_manual(values = stratum_colors, guide = "none") +
    scale_x_continuous(
      breaks = seq(-pre_days, post_days, by = 365.25),
      labels = function(d) sprintf("%g", round(d / 365.25))
    ) +
    labs(x = sprintf("Years from %s (binned, %dd windows)", anchor_label, as.integer(bin_width)),
         y = sprintf("%s%s (mean +/- 95%% CI)",
                     if (isTRUE(log_scale)) "log1p(" else "",
                     if (isTRUE(log_scale)) paste0(lab_group, ")") else lab_group),
         title = title,
         caption = sprintf("Only bins with at least %d patients in a stratum are shown.",
                           as.integer(min_bin_patients))) +
    theme_fig() +
    theme(plot.title = element_text(face = "bold", size = 11),
          plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK, hjust = 0))
}

# Risk-score stratification outputs (risk_score_stratified_figures.py, run by
# compass_pipeline.run_risk_stratification). Group labels are read from the
# per-patient file rather than rederived, so the KMs show exactly the groups
# that script analysed. Colors mirror its STRATUM_COLORS.
RISK_STRATUM_COLORS <- c(
  "High risk (above median)" = "#b2182b", "Low risk (at or below median)" = "#2166ac",
  "Gleason <=6" = "#92c5de", "Gleason 7" = "#f4a582", "Gleason 8-10" = "#b2182b",
  "Stage I-II" = "#92c5de", "Stage III" = "#f4a582", "Stage IV" = "#b2182b",
  "Altered" = "#b2182b", "Wild-type" = "#2166ac",
  "0 altered" = "#2166ac", "1 altered" = "#f4a582", "2+ altered" = "#b2182b",
  "Neither" = "#2166ac", "TP53 only" = "#f4a582", "RB1 only" = "#92c5de", "TP53+RB1" = "#b2182b",
  # Eight TP53/PTEN/RB1 combinations; TP53+RB1 keeps its pairwise red.
  "None altered" = "#7f7f7f", "TP53" = "#E69F00", "PTEN" = "#56B4E9", "RB1" = "#009E73",
  "TP53+PTEN" = "#CC79A7", "PTEN+RB1" = "#0072B2", "TP53+PTEN+RB1" = "#000000"
)
RISK_STRATIFICATION_SCHEMES <- c(test = "held-out test block", cv_oof = "out-of-fold (nested CV), full cohort")
RISK_STRATIFICATION_MATCHED <- c("gleason_somatic")
# Matched cohorts that use somatic data run on a sequencing-date clock
# (build_somatic_gleason_inputs.py), as do their TP53/PTEN/RB1 stratifiers.
RISK_STRATIFICATION_SEQUENCING <- c("gleason_somatic")
SEQUENCING_ANCHOR_LABEL <- "sequencing date"

# NULL when the run predates the per-patient export or the stage was not run.
read_risk_strata <- function(directory, endpoint, landmark) {
  suffix <- sprintf("_%s_landmark%s", endpoint, landmark)
  patients_path <- file.path(directory, paste0("risk_stratified_patients", suffix, ".csv"))
  meta_path <- file.path(directory, paste0("risk_stratified_strata", suffix, ".csv"))
  if (!file.exists(patients_path) || !file.exists(meta_path)) return(NULL)
  meta <- read_csv(meta_path, show_col_types = FALSE,
                   col_types = cols(.default = col_character()))
  patients <- read_csv(patients_path, show_col_types = FALSE,
                       col_types = cols(.default = col_character()))
  patients$time <- suppressWarnings(as.numeric(patients$duration_days))
  patients$event <- suppressWarnings(as.numeric(patients$event))
  meta <- meta[meta$stratifier %in% names(patients), , drop = FALSE]
  list(patients = patients, meta = meta)
}

risk_strata_frame <- function(patients, column, keep = rep(TRUE, nrow(patients))) {
  groups <- patients[[column]]
  ok <- keep & is.finite(patients$time) & patients$time >= 0 &
    patients$event %in% c(0, 1) & !is.na(groups) & nzchar(groups)
  tibble(time = patients$time[ok], event = patients$event[ok], stratum = groups[ok])
}

risk_stratum_order <- function(meta, key) {
  strsplit(meta$order[match(key, meta$stratifier)], "|", fixed = TRUE)[[1]]
}

prepare_figure_text <- function(plot, width) {
  if (!inherits(plot, "ggplot")) return(figure_style_grob(plot))
  # Wrap outside-panel text against the output width, without changing labels
  # embedded in the data. plot alignment gives captions the full device width.
  widths <- c(title = floor(width * 8), subtitle = floor(width * 10), caption = floor(width * 12))
  for (field in names(widths)) {
    value <- plot$labels[[field]]
    if (is.character(value) && length(value) == 1L)
      plot$labels[[field]] <- paste(vapply(strsplit(value, "\n", fixed = TRUE)[[1]],
        function(line) paste(strwrap(line, width = widths[[field]]), collapse = "\n"), character(1)), collapse = "\n")
  }
  figure_style_plot(plot + theme(plot.title.position = "plot", plot.caption.position = "plot",
               legend.text = element_text(margin = margin(l = 5, r = 10)),
               legend.spacing.x = grid::unit(10,"pt"),
               axis.title.y = element_text(margin = margin(r = 10)),
               plot.margin = margin(12, 16, 12, 12)))
}


# Colorblind-safe categorical palette (fixed assignment), shared by the
# Figure 2 (LLM subtype / platinum enrichment) panels:
#   blue -> platinum+, orange -> platinum-. Kept consistent everywhere.
COLOR_PLATINUM_POS <- "#2a78d6"   # blue
COLOR_PLATINUM_NEG <- "#eb6834"   # orange
COLOR_NEUTRAL_INK  <- "#52514e"   # secondary ink, for annotations/text only

# Root for the per-patient primary-subtype labels (primary_subtype_labels.parquet,
# written by LLM_clinical_annotations notebooks/primary_subtype.ipynb from the
# per-note NEPC and AVPC pipelines). Deliberately a different root than
# NEPC_PROJ_PATH: this file lives under the user's data/LLM_annotations/ tree,
# not the CAIA/COMPASS project tree, and is the sole LLM label source for
# Figure 2 v3 and the Section 4-5 stratifications. It is the same file Stage 1
# reads for the nepc/avpc endpoints. Overridable per-call by generate_figures().
DEFAULT_LLM_ANNOTATIONS_PATH <- "/data/gusev/USERS/jpconnor/data/LLM_annotations/LLM_primary_subtype"
LLM_SUBTYPE_LABELS_FILE <- "primary_subtype_labels.parquet"

# Chart-review truth for Figure 2 Panel A, under NEPC_PROJ_PATH/LLM_NEPC_labels.
# updated_manual_low_confidence keeps low-confidence calls, so it is the label
# used here (updated_manual_high_confidence is the stricter alternative).
MANUAL_ANNOTATIONS_FILE <- "full_manual_annotations.csv"
MANUAL_LABEL_COL <- "updated_manual_low_confidence"
MANUAL_LABEL_LEVELS <- c("conventional", "avpc", "nepc")

# Figure 2 uses primary subtypes for landscape/enrichment panels. Longitudinal
# lab figures use only the binary NEPC stratum in FIGURE_LLM_STRATA below.
LLM_STRATA <- list(
  primary_label = list(
    col = "primary_label",
    levels = c("conventional", "avpc", "nepc")
  ),
  has_nepc = list(
    col = "has_nepc",
    levels = c(0, 1),
    labels = c("has_nepc=0", "has_nepc=1")
  ),
  has_avpc = list(
    col = "has_avpc",
    levels = c(0, 1),
    labels = c("has_avpc=0", "has_avpc=1")
  )
)
FIGURE_LLM_STRATA <- LLM_STRATA["has_nepc"]

# load_llm_strata: read primary_subtype_labels.parquet and derive the
# has_nepc/has_avpc strata from the NEPC and AVPC runners. Returns NULL with a
# message if the file is absent, so every downstream stratified-plot loop can
# skip cleanly rather than erroring.
# ---------------------------------------------------------------------------
# Loop-invariant read cache.
#
# The figure workflow can call generate_figures() more than once per R session.
# Several inputs do not vary across those calls (the LLM label set, the
# platinum MRN list, the ICD flag table), so they are read once.
#
# cached_read_csv() memoizes on the resolved path plus the readr arguments that
# affect the RESULT (col_select, col_types, n_max), so a header-only probe and
# a full read of the same file stay distinct entries. Values are returned as-is
# rather than copied: every consumer below treats them as read-only, and dplyr
# verbs copy on modify anyway.
#
# Correctness note: the cache lives for the R session, so a run that
# regenerates an input mid-run would serve a stale frame. That does not happen
# here -- 01/02/03 finish before COMPASS/figures/run_figures.R runs -- but clear_read_cache()
# exists for interactive use after rebuilding inputs.
.read_cache <- new.env(parent = emptyenv())

clear_read_cache <- function() {
  rm(list = ls(.read_cache, all.names = TRUE), envir = .read_cache)
  rm(list = ls(.processed_read_cache, all.names = TRUE),
     envir = .processed_read_cache)
  invisible(NULL)
}

read_cache_stats <- function() as.list(.read_cache[[".stats"]])

# Once processed plotting tables are warm, the full CSV is redundant. Release
# raw frames before forking; preserve processed products and read counters.
clear_raw_read_cache <- function() {
  keys <- setdiff(ls(.read_cache, all.names = TRUE), ".stats")
  rm(list = keys, envir = .read_cache)
  invisible(NULL)
}

cached_read_csv <- function(path, ...) {
  args <- list(...)
  # normalizePath so two spellings of one file share an entry; falls back to
  # the literal path when the file is absent (the caller handles that).
  resolved <- tryCatch(normalizePath(path, mustWork = TRUE),
                       error = function(e) path)
  key <- paste0(resolved, "|",
                paste(names(args), vapply(args, function(a)
                  paste(format(a), collapse = ","), character(1)),
                  sep = "=", collapse = "|"))
  if (!is.null(.read_cache[[key]])) {
    st <- .read_cache[[".stats"]]; st$hits <- st$hits + 1L
    .read_cache[[".stats"]] <- st
    return(.read_cache[[key]])
  }
  # Inject explicitly quoted tidyselect expressions at the readr call site.
  # This lets their syntax participate in the key without evaluating any_of()
  # outside a selection context or reading the CSV header on every cache hit.
  value <- do.call(read_csv, c(list(file = path), args))
  .read_cache[[key]] <- value
  st <- .read_cache[[".stats"]]
  if (is.null(st)) st <- list(hits = 0L, misses = 0L)
  st$misses <- st$misses + 1L
  .read_cache[[".stats"]] <- st
  value
}

# Parquet twin of cached_read_csv, for the LLM primary-subtype label table.
cached_read_parquet <- function(path) {
  resolved <- tryCatch(normalizePath(path, mustWork = TRUE),
                       error = function(e) path)
  key <- paste0("parquet:", resolved)
  if (!is.null(.read_cache[[key]])) {
    st <- .read_cache[[".stats"]]; st$hits <- st$hits + 1L
    .read_cache[[".stats"]] <- st
    return(.read_cache[[key]])
  }
  value <- if (requireNamespace("arrow", quietly = TRUE)) {
    as_tibble(arrow::read_parquet(path))
  } else if (requireNamespace("nanoparquet", quietly = TRUE)) {
    as_tibble(nanoparquet::read_parquet(path))
  } else {
    stop('Reading LLM labels requires R package "arrow" or "nanoparquet".')
  }
  .read_cache[[key]] <- value
  st <- .read_cache[[".stats"]]
  if (is.null(st)) st <- list(hits = 0L, misses = 0L)
  st$misses <- st$misses + 1L
  .read_cache[[".stats"]] <- st
  value
}

# Column spec for the longitudinal labs CSV. read_csv() with only
# show_col_types = FALSE still INFERS types -- that flag silences the report,
# it does not skip the work -- and guess_max = 100000 makes it scan 100k rows
# to do so. Naming the handful of columns the figures actually use skips the
# inference pass entirely; anything unnamed still gets guessed, so this stays
# correct if the upstream schema grows a column.
#
# BEHAVIOR CHANGE, deliberate: DFCI_MRN is pinned to character. Left to guess,
# readr types an all-numeric MRN column as double, and a 12345 -> 12345.0
# round-trip then breaks the joins and %in% set operations downstream (several
# call sites already defend with as.character(DFCI_MRN), which is the symptom).
# Pinning it here makes the ID a string at the source, matching how every other
# MRN list in this pipeline is read.
LONGITUDINAL_COL_TYPES <- cols(
  DFCI_MRN  = col_character(),
  LAB_NAME  = col_character(),
  LAB_VALUE = col_double(),
  t_lab     = col_double()
)

# Do not parse/store upstream feature columns that no figure consumes. This
# table has one row per measurement, so even patient metadata is repeated
# millions of times. Both processed loaders must use the same read-cache key.
FIGURE_PATIENT_COLS <- c(
  "DFCI_MRN", "AGE_AT_TREATMENTSTART", "FIRST_RECORD_DATE", "DIAGNOSIS_DATE",
  "TREATMENT_ANCHOR_DATE", "LAST_CONTACT_DATE", "DEATH", "PLATINUM_MEDICATION",
  "PLATINUM_DATE", "PLATINUM", "t_diagnosis", "t_first_treatment", "t_platinum",
  "t_last_contact", "t_death", "t_dx_to_anchor", "NEPC", "t_nepc"
)
FIGURE_LAB_COLS <- c("DFCI_MRN", "LAB_NAME", "LAB_VALUE", "LAB_UNIT", "LAB_DATE", "t_lab")
cached_figure_longitudinal <- function(path, id_col = "DFCI_MRN") {
  selected <- rlang::expr(any_of(!!unique(c(id_col, FIGURE_PATIENT_COLS, FIGURE_LAB_COLS))))
  cached_read_csv(
    path, show_col_types = FALSE, col_types = LONGITUDINAL_COL_TYPES,
    col_select = selected,
    # Materialize in the parent; no deferred vroom parsing in forked workers.
    lazy = FALSE, num_threads = 1L
  )
}

# Parse each distinct date string once. as.Date.character otherwise repeats
# strptime for every measurement, including all copies of patient-level dates.
figure_dates <- function(x) {
  if (inherits(x, "Date")) return(x)
  if (!is.character(x)) return(suppressWarnings(as.Date(x)))
  values <- unique(x)
  suppressWarnings(as.Date(values))[match(x, values)]
}

# Expensive transformations of the arm-level longitudinal file are invariant
# across cohort/endpoint cells.  Keep processed products beside the raw-read
# cache so a two-endpoint render does not repeatedly convert dates, split the
# patient/lab tables, or classify every lab row.
.processed_read_cache <- new.env(parent = emptyenv())

cached_profile_patient_and_labs <- function(path, id_col = "DFCI_MRN") {
  if (!is.null(getOption("compass.figure_data_manifest"))) {
    patients <- figure_prepared_table(path, "patients")
    return(list(patient_df = patients, labs_df = NULL))
  }
  resolved <- tryCatch(normalizePath(path, mustWork = TRUE),
                       error = function(e) path)
  key <- paste0("profile-split:", resolved, ":", id_col)
  if (exists(key, envir = .processed_read_cache, inherits = FALSE))
    return(.processed_read_cache[[key]])

  df <- cached_figure_longitudinal(path, id_col)
  patient_level <- unique(c(id_col, FIGURE_PATIENT_COLS))
  patient_df <- df %>%
    select(any_of(patient_level)) %>%
    distinct(.data[[id_col]], .keep_all = TRUE)
  date_cols <- c("DIAGNOSIS_DATE", "TREATMENT_ANCHOR_DATE", "PLATINUM_DATE",
                 "LAST_CONTACT_DATE", "FIRST_RECORD_DATE")
  for (col in intersect(date_cols, names(patient_df)))
    patient_df[[col]] <- figure_dates(patient_df[[col]])
  if (all(c("DIAGNOSIS_DATE", "TREATMENT_ANCHOR_DATE") %in% names(patient_df)))
    patient_df$t_dx_to_anchor <- as.numeric(
      patient_df$TREATMENT_ANCHOR_DATE - patient_df$DIAGNOSIS_DATE
    )

  labs_df <- df %>% select(any_of(c(id_col, FIGURE_LAB_COLS))) %>%
    filter(!is.na(LAB_NAME))
  labs_df$LAB_DATE <- figure_dates(labs_df$LAB_DATE)
  # Figure 1 needs only counts/spans. Aggregate once for the arm, then filter
  # patients per cell instead of copying and regrouping every lab row 12 times.
  lab_summary <- labs_df %>% group_by(.data[[id_col]]) %>%
    summarise(
      lab_rows = n(),
      record_span_days = as.numeric(max(LAB_DATE, na.rm = TRUE) -
                                     min(LAB_DATE, na.rm = TRUE)),
      .groups = "drop"
    )
  patient_df <- patient_df %>% left_join(lab_summary, by = id_col)
  value <- list(patient_df = patient_df, labs_df = labs_df)
  .processed_read_cache[[key]] <- value
  value
}

# load_manual_annotations: one row per chart-reviewed patient with
# manual_label in MANUAL_LABEL_LEVELS. Blank labels become NA so multiclass_metrics
# excludes them instead of scoring them as conventional.
load_manual_annotations <- function(path) {
  manual <- cached_read_csv(path, show_col_types = FALSE)
  missing <- setdiff(c("DFCI_MRN", MANUAL_LABEL_COL), names(manual))
  if (length(missing) > 0) {
    stop(sprintf("%s is missing required columns: %s", path, paste(missing, collapse = ", ")))
  }
  manual <- manual %>%
    transmute(
      DFCI_MRN = suppressWarnings(as.numeric(sub("\\.0$", "", str_trim(as.character(DFCI_MRN))))),
      manual_label = str_to_lower(str_trim(as.character(.data[[MANUAL_LABEL_COL]]))),
      manual_label = if_else(
        is.na(manual_label) |
          manual_label %in% c("", "nan", "na", "null", "none"),
        NA_character_,
        manual_label
      )
    )
  unknown <- setdiff(unique(na.omit(manual$manual_label)), MANUAL_LABEL_LEVELS)
  if (length(unknown) > 0) {
    stop(sprintf("%s has unexpected %s values: %s (expected %s)", path, MANUAL_LABEL_COL,
                 paste(unknown, collapse = ", "), paste(MANUAL_LABEL_LEVELS, collapse = ", ")))
  }
  if (anyNA(manual$DFCI_MRN) || anyDuplicated(manual$DFCI_MRN)) {
    stop(sprintf("%s must have one row per non-missing DFCI_MRN", path))
  }
  manual
}

load_llm_strata <- function(llm_annotations_path) {
  path <- file.path(llm_annotations_path, LLM_SUBTYPE_LABELS_FILE)
  if (!file.exists(path)) {
    message(sprintf("load_llm_strata: %s not found -- skipping LLM-strata plots", path))
    return(NULL)
  }
  strata <- cached_read_parquet(path)
  required <- c("DFCI_MRN", "primary_label", "nepc_status", "is_avpc")
  missing <- setdiff(required, names(strata))
  if (length(missing) > 0) {
    stop(sprintf("%s is missing required columns: %s", path, paste(missing, collapse = ", ")))
  }
  # has_nepc/has_avpc are NA when that pipeline never resolved the patient
  # (review_status failed/not_run), so binary metrics exclude them.
  strata <- strata %>%
    transmute(
      DFCI_MRN = suppressWarnings(as.numeric(sub("\\.0$", "", str_trim(as.character(DFCI_MRN))))),
      primary_label = str_to_lower(str_trim(as.character(primary_label))),
      primary_label = if_else(
        is.na(primary_label) |
          primary_label %in% c("", "nan", "na", "null", "none"),
        NA_character_,
        primary_label
      ),
      has_nepc = if_else(is.na(nepc_status), NA_real_,
                         as.numeric(as.character(nepc_status) == "nepc")),
      has_avpc = as.numeric(as.logical(is_avpc))
    )
  n_missing_category <- sum(is.na(strata$primary_label))
  if (n_missing_category > 0) {
    message(sprintf(
      "load_llm_strata: removed %s row(s) with a missing/NaN primary-label category",
      format(n_missing_category, big.mark = ",")
    ))
    strata <- strata %>% filter(!is.na(primary_label))
  }
  invalid_primary <- setdiff(unique(na.omit(strata$primary_label)),
                             LLM_STRATA$primary_label$levels)
  if (length(invalid_primary) > 0) {
    warning(sprintf(
      "load_llm_strata: unrecognized primary_label value(s) converted to NA: %s",
      paste(invalid_primary, collapse = ", ")
    ))
  }
  for (col in c("has_nepc", "has_avpc")) {
    invalid_binary <- setdiff(unique(na.omit(strata[[col]])), c(0, 1))
    if (length(invalid_binary) > 0) {
      warning(sprintf(
        "load_llm_strata: non-binary %s value(s) will be excluded from binary metrics: %s",
        col, paste(invalid_binary, collapse = ", ")
      ))
    }
  }
  strata %>%
    mutate(primary_label = factor(primary_label, levels = LLM_STRATA$primary_label$levels))
}

figure_trajectory_lookup <- function(values, patient_ids = NULL) {
  required <- c("DFCI_MRN", "stratum")
  missing <- setdiff(required, names(values))
  if (length(missing)) stop("Trajectory labels missing: ", paste(missing, collapse = ", "))
  lookup <- values %>% transmute(DFCI_MRN = as.character(DFCI_MRN),
                                 stratum = as.character(stratum)) %>%
    filter(!is.na(DFCI_MRN), !is.na(stratum))
  if (!is.null(patient_ids)) lookup <- lookup %>% filter(DFCI_MRN %in% as.character(patient_ids))
  # Classifier exports may repeat a patient across records/subtypes. Repeated
  # identical labels are harmless, but do not choose an arbitrary conflicting label.
  lookup <- distinct(lookup, DFCI_MRN, stratum)
  conflicts <- lookup %>% count(DFCI_MRN) %>% filter(n > 1L)
  if (nrow(conflicts)) stop(sprintf(
    "Conflicting trajectory labels for %d patient(s); resolve the classifier labels before rerunning. No label was selected arbitrarily.",
    nrow(conflicts)))
  lookup
}

# ---- helpers ported from the Python notebook ----

# binary_metrics: confusion-matrix-derived classification metrics for a 0/1
# (or logical) truth/pred pair. Returns a named list (mirrors the pd.Series).
binary_metrics <- function(y_true, y_pred) {
  if (length(y_true) != length(y_pred)) {
    stop("binary_metrics: y_true and y_pred must have the same length")
  }
  as_binary_integer <- function(x) {
    if (is.logical(x)) return(as.integer(x))
    if (is.factor(x)) x <- as.character(x)
    suppressWarnings(as.integer(x))
  }
  y_true <- as_binary_integer(y_true)
  y_pred <- as_binary_integer(y_pred)
  valid <- !is.na(y_true) & !is.na(y_pred) &
    y_true %in% c(0L, 1L) & y_pred %in% c(0L, 1L)
  excluded <- sum(!valid)
  if (excluded > 0) {
    warning(sprintf(
      "binary_metrics: excluded %s row(s) with missing or non-binary truth/prediction",
      format(excluded, big.mark = ",")
    ))
  }
  y_true <- y_true[valid]
  y_pred <- y_pred[valid]
  tp <- sum(y_true == 1 & y_pred == 1)
  tn <- sum(y_true == 0 & y_pred == 0)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  safe <- function(num, den) if (den > 0) num / den else 0
  list(
    Accuracy  = safe(tp + tn, tp + tn + fp + fn),
    Precision = safe(tp, tp + fp),
    Recall    = safe(tp, tp + fn),
    TPR       = safe(tp, tp + fn),
    FPR       = safe(fp, fp + tn),
    TNR       = safe(tn, tn + fp),
    FNR       = safe(fn, fn + tp),
    TP = tp, FP = fp, TN = tn, FN = fn,
    N = length(y_true), Excluded = excluded
  )
}

# multiclass_metrics: full confusion matrix plus one-vs-rest binary_metrics per
# class for a categorical truth/pred pair. Rows with missing or out-of-level
# labels are excluded (and counted) rather than scored as any class.
multiclass_metrics <- function(y_true, y_pred, levels) {
  if (length(y_true) != length(y_pred)) {
    stop("multiclass_metrics: y_true and y_pred must have the same length")
  }
  y_true <- as.character(y_true)
  y_pred <- as.character(y_pred)
  valid <- !is.na(y_true) & !is.na(y_pred) & y_true %in% levels & y_pred %in% levels
  excluded <- sum(!valid)
  if (excluded > 0) {
    warning(sprintf(
      "multiclass_metrics: excluded %s row(s) with missing or unrecognized truth/prediction",
      format(excluded, big.mark = ",")
    ))
  }
  y_true <- factor(y_true[valid], levels = levels)
  y_pred <- factor(y_pred[valid], levels = levels)
  cm <- as.data.frame(table(truth = y_true, pred = y_pred), responseName = "n",
                      stringsAsFactors = FALSE) %>%
    as_tibble() %>%
    mutate(truth = factor(truth, levels = levels), pred = factor(pred, levels = levels),
           n = as.integer(n))
  per_class <- bind_rows(lapply(levels, function(cls) {
    m <- binary_metrics(y_true == cls, y_pred == cls)
    tibble(class = cls, Accuracy = m$Accuracy, Precision = m$Precision,
           Recall = m$Recall, Specificity = m$TNR,
           TP = m$TP, FP = m$FP, TN = m$TN, FN = m$FN)
  })) %>% mutate(class = factor(class, levels = levels))
  list(cm = cm, per_class = per_class,
       Accuracy = if (length(y_true) > 0) mean(y_true == y_pred) else 0,
       N = length(y_true), Excluded = excluded)
}

# wilson_ci: Wilson score interval for a binomial proportion. Returns c(phat, lo, hi).
wilson_ci <- function(successes, n, z = 1.96) {
  if (n == 0) return(c(phat = NA_real_, lo = NA_real_, hi = NA_real_))
  phat   <- successes / n
  denom  <- 1 + z^2 / n
  center <- (phat + z^2 / (2 * n)) / denom
  half   <- (z * sqrt((phat * (1 - phat) + z^2 / (4 * n)) / n)) / denom
  c(phat = phat, lo = max(0, center - half), hi = min(1, center + half))
}


CBC <- c("WBC","RBC","Hemoglobin","Hematocrit","MCV","MCH","MCHC","RDW","Platelets",
         "Neutrophils absolute","Lymphocytes absolute","Monocytes absolute",
         "Eosinophils absolute","Basophils absolute")
CMP <- c("Sodium","Potassium","Chloride","CO2","BUN","Creatinine","Glucose","Calcium")
LFT <- c("ALT","AST","Alkaline phosphatase","Total bilirubin","Direct bilirubin",
         "Albumin","Globulin","Total protein","PT")
VITALS <- c("Body weight","Body temperature","Heart rate","Respiratory rate",
            "Systolic blood pressure","Diastolic blood pressure")
ANDROGEN <- c("PSA","Testosterone")
OTHER <- c("TSH","LDH")
DROP <- c("Body height")

CATEGORY_MAP <- c(
  setNames(rep("CBC", length(CBC)), CBC),
  setNames(rep("CMP", length(CMP)), CMP),
  setNames(rep("LFT", length(LFT)), LFT),
  setNames(rep("Vitals", length(VITALS)), VITALS),
  setNames(rep("Androgen axis", length(ANDROGEN)), ANDROGEN),
  setNames(rep("Other", length(OTHER)), OTHER)
)

DRAW_ORDER   <- c("Other","Vitals","CMP","LFT","CBC","Androgen axis")
LEGEND_ORDER <- c("Androgen axis","CBC","LFT","CMP","Vitals","Other")

CATEGORY_COLORS <- c(
  "Androgen axis" = "#8e1c2b",
  "CBC"           = "#16a085",
  "LFT"           = "#e67e22",
  "CMP"           = "#7d3c98",
  "Vitals"        = "#5d6d7e",
  "Other"         = "#95a5a6"
)
NS_COLOR <- "#9ba4ae"

assign_category <- function(lab_name) {
  out <- unname(CATEGORY_MAP[lab_name])
  ifelse(is.na(out), "Other", out)
}

format_label <- function(lab_name, feature_stat) {
  ifelse(is.na(feature_stat) | feature_stat == "",
         lab_name, sprintf("%s (%s)", lab_name, feature_stat))
}

# Split "LAB_NAME__stat" into (lab_name, stat); ("", ) when no "__".
parse_feature <- function(name) {
  if (grepl("__", name, fixed = TRUE)) {
    parts <- strsplit(name, "__", fixed = TRUE)[[1]]
    c(lab_name = parts[1], feature_stat = paste(parts[-1], collapse = "__"))
  } else {
    c(lab_name = name, feature_stat = "")
  }
}

# Shared local/federated Figure 4 presentation. Callers supply already fitted
# held-out metrics and feature gains; these functions never fit or pool models.
plot_model_discrimination <- function(data, metric, ylabel, colors,
                                     landmarks=c(0,90,180), show_legend=FALSE) {
  preferred <- c("Cox baseline (age)", "Elastic-Net Cox", "XGBoost Survival")
  existing <- if (is.factor(data$name)) levels(data$name) else unique(as.character(data$name))
  existing <- unique(c(existing, as.character(data$name)))
  series_order <- c(intersect(preferred, existing), setdiff(existing, preferred))
  d <- data %>%
    transmute(name = factor(as.character(name), levels = series_order),
      baseline = name %in% c("Cox baseline (age)","XGBoost baseline (age)"),
      landmark = factor(sprintf("%s%d days", ifelse(landmark > 0, "+", ""), landmark),
        levels = sprintf("%s%d days", ifelse(landmarks > 0,"+",""), landmarks)),
      value = .data[[metric]])
  finite_max <- suppressWarnings(max(d$value, na.rm = TRUE))
  ymax <- min(1.0, (if (is.finite(finite_max)) finite_max else 0.7) * 1.12)
  base <- ggplot(d, aes(landmark, value, group = name))
  if (requireNamespace("ggpattern", quietly=TRUE)) {
    p <- base + ggpattern::geom_col_pattern(
      aes(fill = name, pattern = baseline),
      position = position_dodge(width = 0.85), width = 0.8,
      color = "white", pattern_fill = "white", pattern_density = 0.08,
      pattern_spacing = 0.02, pattern_angle = 45) +
      ggpattern::scale_pattern_manual(values = c(`TRUE` = "stripe", `FALSE` = "none"), guide = "none")
  } else {
    p <- base + geom_col(aes(fill = name, linetype = baseline),
      position = position_dodge(width = 0.85), width = 0.8, color = "white") +
      scale_linetype_manual(values = c(`TRUE` = "dashed", `FALSE` = "solid"), guide = "none")
  }
  p + geom_text(aes(label = ifelse(is.finite(value), sprintf("%.3f", value), "")),
      position = position_dodge(width = 0.85), vjust = -0.4, size = 2.5, show.legend = FALSE) +
    geom_hline(yintercept = 0.5, color = "grey", linetype = "dotted", linewidth = 0.9) +
    scale_fill_manual(values = colors, name = NULL, drop = FALSE) +
    coord_cartesian(ylim = c(0, max(1.04, ymax))) +
    labs(x = NULL, y = ylabel) + theme_fig() +
    guides(fill = guide_legend(ncol = 2, byrow = TRUE)) +
    theme(legend.position = if (show_legend) "top" else "none",
      legend.justification = "right", legend.box.just = "right",
      legend.direction = "horizontal", panel.grid.major.x = element_blank())
}

plot_model_importance <- function(df, kind, title, top_n=15, exclude_drop=TRUE) {
  if (exclude_drop) df <- df %>% filter(!lab_name %in% DROP)
  df <- df %>%
    mutate(category = vapply(lab_name, assign_category, character(1)))
  if (nrow(df) == 0) {
    return(ggplot() + annotate("text", x = 0, y = 0, label = "(no features to display)",
      color = "#7f8c8d") + theme_void() + labs(title = title) +
      theme(plot.title = element_text(face = "bold", size = 11)))
  }
  if (kind == "cox") {
    df <- df %>% arrange(desc(abs(coef))) %>% head(top_n)
    df$value <- df$coef; xlabel <- "log HR coefficient"
  } else {
    df <- df %>% arrange(desc(gain)) %>% head(top_n)
    df$value <- df$gain; xlabel <- "XGBoost gain"
  }
  df <- df %>% mutate(label = mapply(format_label, lab_name, feature_stat),
    category = factor(category, levels = LEGEND_ORDER))
  if (anyDuplicated(df$label)) {
    duplicates <- duplicated(df$label) | duplicated(df$label, fromLast=TRUE)
    df$label[duplicates] <- paste0(df$label[duplicates], " [", df$feature[duplicates], "]")
  }
  df$label <- factor(df$label, levels = rev(df$label))
  p <- ggplot(df, aes(value, label, fill = category)) +
    geom_col(color = "white", linewidth = 0.5, show.legend = TRUE) +
    scale_fill_manual(values = CATEGORY_COLORS, breaks = LEGEND_ORDER, name = NULL, drop = FALSE) +
    labs(x = xlabel, y = NULL, title = title) + theme_fig() +
    theme(plot.title = element_text(face = "bold", size = 11), axis.text.y = element_text(size = 10))
  if (kind == "cox") p <- p + geom_vline(xintercept = 0, color = "black", linewidth = 0.5)
  p
}

# Read held-out discrimination/calibration metrics for one endpoint. Every model
# family now writes the canonical schema (survival_common/metrics_schema.py), so
# there are no per-family column candidates to reconcile -- `endpoint`,
# `test_c_index`, `test_mean_auc_t`, and `test_integrated_brier` are read
# directly. Missing optional outputs still return NA so figure generation can
# continue without a torch install.
#
# Metrics CSVs written before the schema cutover lack these columns and will
# read as NA; refit rather than re-adding fallbacks.
#
# `endpoint` must match the lowercase key written by cox_aggregated.py's
# ENDPOINTS map ("platinum" or "nepc"). It is required rather than defaulted:
# a wrong-but-silent default here yields an all-NA panel that looks like a
# missing torch install instead of an endpoint mismatch.
read_endpoint_performance <- function(path, endpoint) {
  missing_metrics <- c(auc = NA_real_, cindex = NA_real_, brier = NA_real_,
                       n_train_val = NA_real_, n_test = NA_real_)
  if (!file.exists(path)) return(missing_metrics)

  df <- read_csv(path, show_col_types = FALSE)
  if (!("endpoint" %in% names(df))) return(missing_metrics)
  # `.env` is required: the argument shares its name with the data column, and
  # under dplyr's data masking a bare `endpoint` would resolve to the column,
  # making this an elementwise self-comparison that keeps every row.
  wanted_endpoint <- tolower(endpoint)
  row <- df %>%
    filter(tolower(as.character(.data[["endpoint"]])) == .env$wanted_endpoint)
  if (nrow(row) == 0) return(missing_metrics)

  metric <- function(column) {
    if (!(column %in% names(row))) return(NA_real_)
    suppressWarnings(as.numeric(row[[column]][1]))
  }
  c(
    auc = metric("test_mean_auc_t"),
    cindex = metric("test_c_index"),
    brier = metric("test_integrated_brier"),
    n_train_val = metric("n_train_val"),
    n_test = metric("n_test")
  )
}


# `cohort` here is the composed RUN LABEL that names the data trees, matching
# compass_pipeline.make_runs(): <arm><cohort suffix><exclusion suffix>. It is a
# superset of the treatment arm, which is why the arm is recovered from it
# below rather than assumed.
COHORT_ARMS <- c("arpi", "adt")
# Patient-subset suffixes that make_runs() appends to the arm. "" is the arm's
# own full survival cohort.
# "_localized" and "_nonmetastatic_llm" are retired: neither the
# localized-adjuvant nor the LLM non-metastatic cohort is modelled, so no such
# tree is generated. Re-add them here AND to the notebook COHORTS if those runs
# are ever reinstated.
COHORT_SUBSET_SUFFIXES <- c(
  "",
  "_metastatic_adt",   # medication-derived ADT intent
  "_metastatic_llm"    # met_diagnosis LLM metastatic status
)
# Orthogonal exclusion suffixes, likewise appended by make_runs().
COHORT_EXCLUSION_SUFFIXES <- c("", "_noprecastrate")

SUPPORTED_COHORTS <- as.vector(outer(
  as.vector(outer(COHORT_ARMS, COHORT_SUBSET_SUFFIXES, paste0)),
  COHORT_EXCLUSION_SUFFIXES, paste0
))
# Notebook default: render the unrestricted ADT arm only. Pass any other
# SUPPORTED_COHORTS value to generate_figures() to render that tree.
COHORTS <- c("adt")

# Keyed by suffix, parallel to COHORT_SUBSET_SUFFIXES / _EXCLUSION_SUFFIXES.
# Positional rather than named because R cannot use "" as a name, and the
# un-suffixed entry is exactly the one that carries no title text.
COHORT_SUBSET_LABELS <- c(
  "",                              # ""
  " / metastatic (ADT intent)",    # _metastatic_adt
  " / metastatic (LLM)"            # _metastatic_llm
)
COHORT_EXCLUSION_LABELS <- c(
  "",                          # ""
  " / no pre-ADT castrate"     # _noprecastrate
)
stopifnot(
  length(COHORT_SUBSET_LABELS) == length(COHORT_SUBSET_SUFFIXES),
  length(COHORT_EXCLUSION_LABELS) == length(COHORT_EXCLUSION_SUFFIXES)
)

# The treatment arm a run label belongs to; the arm alone drives the anchor
# label, the longitudinal CSV, and the landmarks, none of which a patient
# subset or an exclusion changes.
cohort_arm <- function(cohort) {
  hits <- COHORT_ARMS[startsWith(cohort, COHORT_ARMS)]
  if (!length(hits))
    stop(sprintf("Cannot determine treatment arm for cohort=%s", cohort))
  # Longest match, so a future arm that prefixes another is unambiguous.
  hits[which.max(nchar(hits))]
}

# Display title for a run label, composed the same way its directory name is.
cohort_display <- function(cohort) {
  arm <- cohort_arm(cohort)
  rest <- substring(cohort, nchar(arm) + 1L)
  exclusion_label <- ""
  for (i in seq_along(COHORT_EXCLUSION_SUFFIXES)) {
    suffix <- COHORT_EXCLUSION_SUFFIXES[i]
    if (nzchar(suffix) && endsWith(rest, suffix)) {
      exclusion_label <- COHORT_EXCLUSION_LABELS[i]
      rest <- substring(rest, 1L, nchar(rest) - nchar(suffix))
      break
    }
  }
  i <- match(rest, COHORT_SUBSET_SUFFIXES)
  if (is.na(i))
    stop(sprintf("Unrecognized cohort subset suffix '%s' in cohort=%s", rest, cohort))
  paste0(toupper(arm), COHORT_SUBSET_LABELS[i], exclusion_label)
}

# Directory-safe slugs for the two cohort axes a run label composes, so the
# figure tree can nest by axis instead of by the flattened run label. The
# composite label (e.g. "adt_metastatic_llm_noprecastrate") is fine for naming
# DATA trees, where it matches the Python pipeline one-to-one, but it is the
# wrong key for FIGURES: comparing one panel across cohorts should be a single
# directory listing, not twelve paths differing mid-string.
#
#   cohort_subset_slug("adt")                          -> "all"
#   cohort_subset_slug("adt_metastatic_llm_noprecastrate") -> "metastatic_llm"
#   cohort_exclusion_slug("adt")                       -> "incl"
#   cohort_exclusion_slug("adt_..._noprecastrate")     -> "noprecastrate"
#
# The no-exclusion case is "incl" rather than "" so every leaf filename has the
# same shape and the two variants of a subset sort adjacently.
COHORT_SUBSET_SLUGS <- c("all", "metastatic_adt", "metastatic_llm")
stopifnot(length(COHORT_SUBSET_SLUGS) == length(COHORT_SUBSET_SUFFIXES))

# Splits a run label into (subset suffix, exclusion suffix) using the same
# longest-arm-match rule as cohort_arm(), so a future arm that prefixes another
# stays unambiguous. Errors on an unrecognized subset rather than silently
# routing figures to a wrong or invented directory.
cohort_axes <- function(cohort) {
  arm <- cohort_arm(cohort)
  rest <- substring(cohort, nchar(arm) + 1L)
  exclusion <- ""
  for (suffix in COHORT_EXCLUSION_SUFFIXES) {
    if (nzchar(suffix) && endsWith(rest, suffix)) {
      exclusion <- suffix
      rest <- substring(rest, 1L, nchar(rest) - nchar(suffix))
      break
    }
  }
  i <- match(rest, COHORT_SUBSET_SUFFIXES)
  if (is.na(i))
    stop(sprintf("Unrecognized cohort subset suffix '%s' in cohort=%s", rest, cohort))
  list(arm = arm, subset = COHORT_SUBSET_SLUGS[i], exclusion = exclusion)
}

cohort_subset_slug <- function(cohort) cohort_axes(cohort)$subset

cohort_exclusion_slug <- function(cohort) {
  ex <- cohort_axes(cohort)$exclusion
  if (nzchar(ex)) sub("^_", "", ex) else "incl"
}

# The leaf filename identity for a run: "<subset>__<exclusion>". Double
# underscore separates the two AXES; single underscores live inside a slug.
cohort_leaf_slug <- function(cohort) {
  paste0(cohort_subset_slug(cohort), "__", cohort_exclusion_slug(cohort))
}

# Every cohort arm uses the same 0-, 90-, and 180-day landmarks, and a patient
# subset does not change them.
COHORT_LANDMARKS <- list(
  arpi = c(0, 90, 180),
  adt = c(0, 90, 180)
)

# Survival endpoints with an independent results tree, as built by
# compass_pipeline.make_endpoint_runs(). The suffix must match the
# `output_suffix` that make_runs() appends to BOTH prediction_inputs_* and
# local_runs_*: "" for platinum, "_nepc" for NEPC.
# "avpc" is retired alongside the localized cohort; no _avpc tree is generated.
SUPPORTED_ENDPOINTS <- c("platinum", "nepc")
ENDPOINT_SUFFIXES <- c(
  platinum = "",
  nepc = "_nepc"
)
# Figure 1/2 and the descriptive longitudinal lab panels are classifier- and
# platinum-cohort figures. Their subject is the cohort/stratification itself,
# not the modelled endpoint, so they are generated once under the platinum
# endpoint rather than duplicated per endpoint.
ENDPOINT_INDEPENDENT_FIGURES_ENDPOINT <- "platinum"

# Check existing graphics without decoding whole images. Empty/truncated files
# lack the format header or closing marker and must be regenerated.
figure_file_complete <- function(path) {
  info <- file.info(path)
  if (is.na(info$size) || isTRUE(info$isdir) || info$size < 12 ||
      !endsWith(tolower(path), ".pdf")) return(FALSE)
  tryCatch({
    con <- file(path, "rb")
    on.exit(close(con))
    header <- readBin(con, "raw", 5L)
    seek(con, max(0, info$size - 1024), origin = "start")
    footer <- readBin(con, "raw", 1024L)
    nonspace <- which(!footer %in% as.raw(c(9, 10, 13, 32)))
    last <- if (length(nonspace)) tail(nonspace, 1L) else 0L
    identical(header, charToRaw("%PDF-")) && last >= 5L &&
      identical(footer[seq.int(last - 4L, last)], charToRaw("%%EOF"))
  }, error = function(e) FALSE)
}

# Each worker owns one small status file. Atomic replacement lets concurrent
# workers report a shared total without racing on a shared counter. These are
# session-temporary progress records, never figure-output completion markers.
new_figure_progress <- function(labels,
                                report = function(line) cat(line, "\n", file = stderr())) {
  stopifnot(is.character(labels), length(labels) > 0L, !anyNA(labels),
            is.function(report))
  progress_dir <- tempfile("compass-figure-progress-")
  if (!dir.create(progress_dir)) stop("Could not create figure progress directory")
  paths <- file.path(progress_dir, sprintf("cell-%03d.rds", seq_along(labels)))
  started <- Sys.time()
  for (path in paths)
    saveRDS(list(status = "queued", detail = "queued", panels = character(),
                 skipped = character()), path,
            compress = FALSE)

  snapshot <- function() {
    states <- lapply(paths, readRDS)
    statuses <- vapply(states, function(x) x$status, character(1))
    list(
      total = length(labels),
      finished = sum(statuses %in% c("complete", "failed")),
      successful = sum(statuses == "complete"),
      failed = sum(statuses == "failed"),
      panels = sum(vapply(states, function(x) length(x$panels), integer(1))),
      skipped = sum(vapply(states, function(x) length(x$skipped), integer(1))),
      elapsed = as.numeric(difftime(Sys.time(), started, units = "secs")),
      states = states
    )
  }
  display <- function(detail) {
    s <- snapshot()
    filled <- floor(20 * s$finished / s$total)
    bar <- paste0(strrep("=", filled), strrep("-", 20 - filled))
    seconds <- floor(s$elapsed)
    clock <- sprintf("%02d:%02d:%02d", seconds %/% 3600,
                     (seconds %% 3600) %/% 60, seconds %% 60)
    report(sprintf(
      "[%s] %d/%d sets finished (%3.0f%%) | %d saved, %d skipped | %d failed | %s elapsed | %s",
      bar, s$finished, s$total, 100 * s$finished / s$total,
      s$panels, s$skipped, s$failed, clock, gsub("[\r\n]+", " ", detail)
    ))
    invisible(s)
  }
  update <- function(i, event, detail = "") {
    stopifnot(length(i) == 1L, !is.na(i), i %in% seq_along(paths))
    event <- match.arg(event, c("start", "stage", "panel_start", "panel_done", "panel_skipped",
                               "complete", "failed"))
    state <- readRDS(paths[[i]])
    state$status <- if (event %in% c("complete", "failed")) event else "running"
    state$detail <- detail
    if (event == "panel_done") {
      state$panels <- unique(c(state$panels, detail))
      state$skipped <- setdiff(state$skipped, detail)
    }
    if (event == "panel_skipped" && !detail %in% state$panels)
      state$skipped <- unique(c(state$skipped, detail))
    pending <- paste0(paths[[i]], ".", Sys.getpid(), ".tmp")
    saveRDS(state, pending, compress = FALSE)
    if (!file.rename(pending, paths[[i]]))
      stop("Could not publish figure progress for ", labels[[i]])
    verb <- switch(event, start = "starting", stage = "preparing",
                   panel_start = "rendering", panel_done = "saved", panel_skipped = "skipped",
                   complete = "complete", failed = "FAILED")
    display(sprintf("%s: %s %s", labels[[i]], verb, detail))
  }
  close <- function() {
    # Only this tracker's newly created temporary directory is removed.
    unlink(progress_dir, recursive = TRUE)
    invisible(NULL)
  }
  display("starting figure generation")
  list(update = update, snapshot = snapshot, close = close)
}

reuse_previous_figure_layout <- function(destination) {
  if (figure_file_complete(destination)) return(invisible(FALSE))
  candidates <- c(sub("/by_figure/", "/by_figure/main/", destination),
                  sub("/by_figure/", "/by_figure/supplements/", destination))
  candidates <- candidates[candidates != destination & vapply(candidates, figure_file_complete, logical(1))]
  if (!length(candidates)) return(invisible(FALSE))
  if (length(unique(unname(tools::md5sum(candidates)))) > 1L)
    stop("Conflicting old figure copies for ", destination)
  previous <- candidates[[1]]
  dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(".compass-copy-", tmpdir = dirname(destination),
                        fileext = paste0(".", tools::file_ext(destination)))
  on.exit(unlink(temporary))
  if (!file.copy(previous, temporary, overwrite = TRUE) || !figure_file_complete(temporary) ||
      !file.rename(temporary, destination)) stop("Could not reuse previous figure: ", previous)
  message("reused completed figure in new layout: ", destination)
  invisible(TRUE)
}

# Render the full COMPASS figure set for one cohort arm and one survival
# endpoint. Mirrors the body of the former figure notebook's per-cohort cells
# (Figures 1-7 + Table 1), so the R Markdown document can call it once per
# (cohort, endpoint) pair in the same R session.
# ----------------------------- labeling knobs ---------------------------
TOP_K_PER_PANEL <- 4
ALWAYS_LABEL    <- c("Hemoglobin", "Albumin", "Alkaline phosphatase")
PANEL_XLIM      <- c(-1.5, 1.5)
Y_MAX_CAP       <- 30   # -log10(p) ceiling; values above are drawn at the cap as triangles
ALBUMIN_DAY180_NUDGE <- c(x=-0.8,y=1.7)

q_threshold_neglog10p <- function(sub) {
  sig <- sub$p_value[sub$q_value < 0.05]
  if (length(sig) == 0) return(NA_real_)
  -log10(max(max(sig), 1e-300))
}

# Which rows to label, following the Python _auto_label selection rules.
labels_for_panel <- function(sub, top_k, always_label) {
  sig <- sub %>% filter(sig)
  if (nrow(sig) == 0) return(sig[0, ])
  # Label the mean and strongest other statistic per androgen lab. All
  # observations remain plotted, without ten near-identical labels piling up.
  androgen_rows <- sig %>% filter(category == "Androgen axis") %>%
    arrange(desc(feature_stat == "mean"), p_value) %>%
    group_by(lab_name) %>% slice_head(n = 2L) %>% ungroup()
  non_andro <- sig %>% filter(category != "Androgen axis") %>%
    arrange(p_value) %>% distinct(lab_name, .keep_all = TRUE)
  always_sig <- non_andro %>% filter(lab_name %in% always_label)
  extra <- non_andro %>% filter(!lab_name %in% always_label) %>% head(top_k)
  non_andro_label <- bind_rows(always_sig, extra) %>% distinct(lab_name, .keep_all = TRUE)
  bind_rows(androgen_rows, non_andro_label) %>%
    distinct(lab_name, feature_stat, .keep_all = TRUE)
}

# Define the mean-feature testing family before any effect-size, CI, height,
# labeling or plotting filters. Keep the supplied adjustment for auditability.
prepare_mean_univariable <- function(d) {
  d <- d %>% filter(tolower(feature_stat) == "mean")
  if (!"q_value_original" %in% names(d))
    d$q_value_original <- if ("q_value" %in% names(d)) d$q_value else NA_real_
  groups <- intersect(c("analysis_label", "endpoint", "landmark_days"), names(d))
  key <- c(groups, if ("feature" %in% names(d)) "feature" else "lab_name")
  if (anyDuplicated(d[key])) stop("Duplicate mean-feature tests within a Figure 3 testing family")
  if (!"q_value_mean" %in% names(d)) {
    d <- d %>% group_by(across(all_of(groups))) %>% mutate(
      q_value_mean = {
        valid <- is.finite(p_value) & p_value >= 0 & p_value <= 1
        q <- rep(NA_real_, n())
        q[valid] <- stats::p.adjust(p_value[valid], method = "BH")
        q
      },
      n_mean_tests = sum(is.finite(p_value) & p_value >= 0 & p_value <= 1)) %>% ungroup()
  }
  d <- d %>% mutate(q_value = q_value_mean,
    risk_direction = if_else(coef_feature < 0, "Lower", "Higher"),
    risk_coef = abs(coef_feature), risk_hr = exp(risk_coef),
    risk_label = paste(risk_direction, lab_name))
  if (all(c("ci_lower", "ci_upper") %in% names(d))) d <- d %>% mutate(
    risk_ci_lower = if_else(coef_feature < 0, 1 / ci_upper, ci_lower),
    risk_ci_upper = if_else(coef_feature < 0, 1 / ci_lower, ci_upper))
  d
}

plot_volcano_panel <- function(sub, title) {
  sub <- prepare_mean_univariable(sub) %>%
    filter(!lab_name %in% DROP, is.finite(risk_hr), risk_hr > 0, is.finite(q_value)) %>%
    mutate(
      .point_id  = row_number(),
      category  = vapply(lab_name, assign_category, character(1)),
      neglog10q = -log10(pmax(q_value, 1e-300)),
      sig       = q_value < 0.05,
      capped    = neglog10q > Y_MAX_CAP,
      y         = pmin(neglog10q, Y_MAX_CAP),
      label     = risk_label
    )
  ns  <- sub %>% filter(!sig)
  sigd <- sub %>% filter(sig) %>%
    mutate(category = factor(category, levels = DRAW_ORDER),
           is_hero  = category == "Androgen axis")
  y_max <- if (nrow(sub)) max(sub$y) else 5
  q_y <- -log10(.05)
  lab_df <- labels_for_panel(sub, TOP_K_PER_PANEL, ALWAYS_LABEL)
  # Keep every point in the repel calculation, including unlabeled points, so
  # text cannot settle on top of a nearby observation.
  sub <- sub %>% mutate(repel_label = ifelse(.point_id %in% lab_df$.point_id, label, ""))
  label_nudge_x <- rep(.06 / log(10), nrow(sub))
  label_nudge_y <- rep(.35, nrow(sub))

  n_tested <- nrow(sub); n_sig <- sum(sub$sig)
  breakdown <- sub %>% filter(sig) %>% count(category)
  short <- c("Androgen axis"="Androgen","CBC"="CBC","LFT"="LFT","CMP"="CMP",
             "Vitals"="Vitals")
  bd_str <- paste(vapply(setdiff(LEGEND_ORDER, "Other"), function(c) {
    n <- breakdown$n[match(c, breakdown$category)]; if (is.na(n)) n <- 0
    sprintf("%s %d", short[[c]], n)
  }, character(1)), collapse = "  ")
  footer <- sprintf("%d / %d displayed means q<0.05; BH across %d mean tests   \u00b7   %s",
    n_sig, n_tested, if(nrow(sub)) sub$n_mean_tests[1] else 0L, bd_str)

  p <- ggplot() +
    geom_vline(xintercept = 1, color = "grey", linewidth = 0.7) +
    geom_vline(xintercept = 2, color = "grey", linetype = "dashed",
               linewidth = 0.6, alpha = 0.7) +
    { if (!is.na(q_y)) geom_hline(yintercept = q_y, color = "black",
                                  linetype = "dotted", linewidth = 0.9) } +
    geom_point(data = ns, aes(risk_hr, y), size = 1.6 * 1.5, color = NS_COLOR, alpha = 0.7) +
    geom_point(data = sigd %>% filter(!capped),
               aes(risk_hr, y, color = category, size = is_hero),
               shape = 21, fill = NA, stroke = 0.9, show.legend = FALSE) +
    geom_point(data = sigd %>% filter(!capped),
               aes(risk_hr, y, fill = category, size = is_hero),
               shape = 21, color = "white", stroke = 0.6, alpha = 0.92, show.legend = TRUE) +
    geom_point(data = sigd %>% filter(capped),
               aes(risk_hr, y, fill = category), shape = 24,
               size = 3.4 * 1.5, color = "white", stroke = 0.6, alpha = 0.92, show.legend = FALSE) +
    ggrepel::geom_text_repel(
      data = sub,
      aes(risk_hr, y, label = repel_label, color = category),
      size = 4.2, fontface = "plain", segment.color = "#95a5a6", segment.size = 0.3,
      max.overlaps = Inf, min.segment.length = 0, box.padding = 0.55,
      point.padding = 1, point.size = 4, force = 3, force_pull = 0.2,
      max.time = 4, max.iter = 100000, seed = 0,
      nudge_x = label_nudge_x, nudge_y = label_nudge_y,
      show.legend = FALSE) +
    scale_color_manual(values = CATEGORY_COLORS, breaks = LEGEND_ORDER, name = NULL) +
    scale_fill_manual(values = CATEGORY_COLORS, breaks = LEGEND_ORDER, limits = LEGEND_ORDER,
                      drop = FALSE, name = NULL) +
    guides(color = "none", fill = guide_legend(nrow = 2, override.aes = list(shape = 21, size = 3, alpha = 1))) +
    scale_size_manual(values = c(`TRUE` = 3.2 * 1.5, `FALSE` = 2.1 * 1.5), guide = "none") +
    scale_x_log10(breaks = c(1, 1.25, 1.5, 2, 3, 5, 10, 20, 50, 100),
                  labels = scales::label_number()) +
    coord_cartesian(xlim = c(1, exp(max(c(.7, sub$risk_coef), na.rm = TRUE) * 1.15)),
                    ylim = c(-0.2, max(y_max * 1.20, 5))) +
    labs(x = "Hazard ratio per SD (log scale)", y = "−log10(q)", title = title,
         caption = str_wrap(footer, 88)) +
    theme_fig() +
    theme(plot.title = element_text(face = "bold", size = 16),
          plot.caption = element_text(size = 11, color = "#5d6d7e", family = FIGURE_FONT,
                                      hjust = 0, lineheight = 1.05),
          axis.title = element_text(size = 14),
          axis.text  = element_text(size = 12),
          legend.text = element_text(size = 12),
          legend.position = "bottom", legend.justification = "center")
  p
}

generate_figures <- function(cohort, nepc_proj_path, fig_root,
                             endpoint = "platinum",
                             cohorts = SUPPORTED_COHORTS, show = FALSE,
                             llm_annotations_path = DEFAULT_LLM_ANNOTATIONS_PATH,
                             save_pdf = FALSE,
                             output_mode = "panels",
                             progress = NULL,
                             overwrite = FALSE,
                             metastatic_supplement = NULL,
                             cohort_forest_config = NULL,
                             federated_path = NULL) {
  if (!is.null(progress) && !is.function(progress))
    stop("progress must be NULL or a function(event, detail)")
  notify_progress <- function(event, detail) {
    if (!is.null(progress)) progress(event, detail)
    invisible(NULL)
  }
  # Rscript opens `Rplots.pdf` when any plot is drawn without an explicit device.
  # All intended outputs below use ggsave(), so route any incidental drawing to a
  # temporary Arial-capable PDF device during non-interactive runs.
  if (!show) {
    previous_device_option <- getOption("device")
    options(device = function(...) figure_pdf_device(tempfile(fileext=".pdf"), ...))
    on.exit(options(device = previous_device_option), add = TRUE)
  }

  NEPC_PROJ_PATH <- nepc_proj_path
  COHORT <- cohort
  if (!COHORT %in% cohorts)
    stop(sprintf("Unknown cohort=%s; expected one of %s",
                 COHORT, paste(cohorts, collapse = ", ")))
  if (!is.character(endpoint) || length(endpoint) != 1 || is.na(endpoint))
    stop("endpoint must be one non-missing character value")
  ENDPOINT <- tolower(endpoint)
  if (!ENDPOINT %in% SUPPORTED_ENDPOINTS)
    stop(sprintf("Unknown endpoint=%s; expected one of %s",
                 ENDPOINT, paste(SUPPORTED_ENDPOINTS, collapse = ", ")))
  ENDPOINT_SUFFIX <- unname(ENDPOINT_SUFFIXES[[ENDPOINT]])
  output_mode <- match.arg(output_mode)
  if (!is.logical(save_pdf) || length(save_pdf) != 1L || is.na(save_pdf))
    stop("save_pdf must be one non-missing logical value")
  if (!is.logical(overwrite) || length(overwrite) != 1L || is.na(overwrite))
    stop("overwrite must be one non-missing logical value")
  # Endpoint-independent figures describe the platinum-labelled cohort itself.
  # Emit them only on the platinum pass so a NEPC run does not create a second,
  # misleadingly endpoint-labelled copy.
  EMIT_ENDPOINT_INDEPENDENT <- identical(ENDPOINT, ENDPOINT_INDEPENDENT_FIGURES_ENDPOINT)
  if (!is.null(federated_path) &&
      (!is.character(federated_path) || length(federated_path) != 1L || is.na(federated_path)))
    stop("federated_path must be NULL or one non-missing path")
  COHORT_DISPLAY <- cohort_display(COHORT)
  ENDPOINT_DISPLAY <- toupper(ENDPOINT)
  message(sprintf("Generating figures for cohort: %s, endpoint: %s",
                  COHORT_DISPLAY, ENDPOINT_DISPLAY))

  # The arm, not the full run label: every restricted cohort of the ADT arm is
  # still anchored on ADT initiation and still reads the ADT longitudinal CSV.
  COHORT_ARM <- cohort_arm(COHORT)
  IS_ADT <- identical(COHORT_ARM, "adt")
  IS_CANONICAL_ADT <- identical(COHORT, "adt")
  ANCHOR_LABEL <- if (IS_ADT) "ADT initiation" else "ARPI/chemo initiation"

  # Both trees are endpoint-suffixed, matching make_runs(): the NEPC cohort is
  # gated on t_nepc > 0, so its prediction inputs are a different patient set
  # and must be read from prediction_inputs_<cohort>_nepc, not the platinum build.
  BASE <- file.path(NEPC_PROJ_PATH, "survival_analysis",
                    paste0("local_runs_", COHORT, ENDPOINT_SUFFIX))
  LONGITUDINAL_CSV <- file.path(
    NEPC_PROJ_PATH,
    if (IS_ADT) "longitudinal_prediction_data_adt.csv" else "longitudinal_prediction_data.csv"
  )
  INPUTS_DIR <- file.path(NEPC_PROJ_PATH, "survival_analysis",
                          paste0("prediction_inputs_", COHORT, ENDPOINT_SUFFIX))
  ICD_PROSTATE_MRN_FLAGS_CSV <- file.path(
    NEPC_PROJ_PATH, "mrn_lists", "icd_prostate_mrn_flags.csv"
  )

  # All endpoints, cohort variants, and diagnostics share figure-family folders.
  # Each subpanel keeps its own
  # artifact directory and endpoint/subset/exclusion leaf to avoid collisions.
  COHORT_ARM_DIR <- toupper(cohort_arm(COHORT))
  FIG_ROOT <- file.path(fig_root, COHORT_ARM_DIR)
  # Leaf identity for this run: which of the 12 cohort x endpoint cells a file
  # represents. Endpoint leads so a directory listing groups by endpoint first.
  COHORT_LEAF <- paste0(ENDPOINT, "__", cohort_leaf_slug(COHORT))
  # Canonical-lab names sorted longest-first so e.g. "Direct bilirubin" is
  # matched before "Total bilirubin" would ever partially collide, and so a
  # lab-specific stem is never mis-routed to a shorter substring match.
  CATEGORY_MAP_LABS <- names(CATEGORY_MAP)[order(-nchar(names(CATEGORY_MAP)))]
  lab_stem_slug <- function(lab_name) {
    tolower(gsub("[^A-Za-z0-9]+", "_", lab_name))
  }
  LAB_SLUG_TO_NAME <- setNames(CATEGORY_MAP_LABS, lab_stem_slug(CATEGORY_MAP_LABS))
  # Identify which canonical lab (if any) a plot stem is about, by matching
  # against every registered lab's slug as a token within the stem.
  match_lab_in_stem <- function(plot_stem) {
    for (slug in names(LAB_SLUG_TO_NAME)) {
      if (grepl(paste0("(^|_)", slug, "(_|$)"), plot_stem)) return(LAB_SLUG_TO_NAME[[slug]])
    }
    NA_character_
  }
  # Legacy logical addresses, retained as stable inputs to the 07 compiler.
  # figure_workflow.R intercepts these before any image export, compiles
  # related panels, and routes them to numbered section folders. Direct legacy
  # callers retain the addressing below.
  # Logical layout:
  #   FIG_ROOT/by_figure/<group>/<trimmed-name>/<endpoint>__<subset>__<exclusion>.pdf
  # Per-lab panels keep their category/lab nesting for the same reason as
  # before -- ~40 labs x 4 strata would otherwise dump 160+ entries into one
  # directory -- and still bottom out at <= 12 files per leaf:
  #   FIG_ROOT/by_figure/labs/<category>/<lab>/longitudinal/<endpoint>__<subset>__<exclusion>.pdf
  figure_group <- function(plot_stem) {
    # Checked before the "figure1" prefix so the supplement gets its own
    # directory instead of being swallowed by the main Figure 1 group.
    # ADT-intent supplement. Checked before the "figure1"/"figure2" prefixes
    # for the same reason figure1s is: an "adt_intent_" stem is a supplement,
    # not a member of any numbered figure group.
    if (startsWith(plot_stem, "adt_intent_")) return("supplement_adt_intent")
    if (startsWith(plot_stem, "adt_labels_")) return("metastatic_labels")
    if (startsWith(plot_stem, "cohort_forest_")) return("cohort_comparison")
    if (startsWith(plot_stem, "figure1s")) return("figure1s_analysis_sets")
    if (startsWith(plot_stem, "figure1") || startsWith(plot_stem, "table1")) return("figure1")
    if (startsWith(plot_stem, "figure2v3")) return("figure2v3_llm")
    if (startsWith(plot_stem, "figure3b")) return("figure3b")
    if (startsWith(plot_stem, "figure3")) return("figure3")
    if (startsWith(plot_stem, "figure4")) return("figure4")
    if (startsWith(plot_stem, "km_llm_")) return("km_llm")
    if (startsWith(plot_stem, "km_tertile_")) {
      lab <- match_lab_in_stem(plot_stem)
      if (!is.na(lab)) return(file.path("labs", assign_category(lab), lab, "km_tertile"))
    }
    if (startsWith(plot_stem, "km_quartile_")) {
      lab <- match_lab_in_stem(plot_stem)
      if (!is.na(lab)) return(file.path("labs", assign_category(lab), lab, "km_quartile"))
    }
    if (startsWith(plot_stem, "km_quintile_")) {
      lab <- match_lab_in_stem(plot_stem)
      if (!is.na(lab)) return(file.path("labs", assign_category(lab), lab, "km_quintile"))
    }
    if (startsWith(plot_stem, "km_")) return("KM_curves")
    if (startsWith(plot_stem, "androgen_dist_") || startsWith(plot_stem, "dist_")) {
      lab <- match_lab_in_stem(plot_stem)
      if (!is.na(lab)) return(file.path("labs", assign_category(lab), lab, "distribution"))
      return("androgen_distributions")
    }
    if (startsWith(plot_stem, "pre_adt_coverage_"))
      return("androgen_pre_adt_coverage")
    if (startsWith(plot_stem, "gam_trajectory_") ||
        startsWith(plot_stem, "gam_longitudinal_")) {
      lab <- match_lab_in_stem(plot_stem)
      if (!is.na(lab)) return(file.path("labs", assign_category(lab), lab, "gam_trajectory"))
      return("gam_trajectories")
    }
    if (startsWith(plot_stem, "androgen_longitudinal_") || startsWith(plot_stem, "longitudinal_")) {
      lab <- match_lab_in_stem(plot_stem)
      if (!is.na(lab)) return(file.path("labs", assign_category(lab), lab, "longitudinal"))
      return("androgen_trajectories")
    }
    stop(sprintf("Unmapped figure output stem: %s", plot_stem))
  }
  # Remove information already carried by parent folders from the artifact
  # directory name. Internal stems remain unchanged because figure routing and
  # output-mode selection depend on them. Examples:
  #   figure3/figure3_univariate_nepc_landmark0 -> figure3/univariate_nepc_landmark0
  #   labs/.../PSA/longitudinal/longitudinal_platinum_psa_log -> .../platinum_log
  artifact_name_for_stem <- function(plot_stem, group = figure_group(plot_stem)) {
    group_leaf <- basename(group)
    prefix_pattern <- switch(
      group_leaf,
      figure1 = "^figure1_?",
      figure1s_analysis_sets = "^figure1s_analysis_sets_?",
      figure2v3_llm = "^figure2v3_?",
      figure3 = "^figure3_?",
      figure3b = "^figure3b_?",
      figure4 = "^figure4_?",
      km_llm = "^km_llm_?",
      KM_curves = "^km_?",
      km_quartile = "^km_quartile_?",
      km_quintile = "^km_quintile_?",
      km_tertile = "^km_tertile_?",
      distribution = "^(androgen_)?dist(ribution)?_?",
      longitudinal = "^(androgen_)?longitudinal_?",
      gam_trajectory = "^gam_(trajectory|longitudinal)_?",
      androgen_pre_adt_coverage = "^pre_adt_coverage_?",
      metastatic_labels = "^adt_labels_?",
      cohort_comparison = "^cohort_forest_?",
      NULL
    )
    artifact_name <- plot_stem
    if (!is.null(prefix_pattern))
      artifact_name <- sub(prefix_pattern, "", artifact_name, perl = TRUE)

    # Per-lab paths already name the lab two levels above the artifact. Remove
    # that exact slug token as well, retaining endpoint, scale, and landmark.
    lab <- match_lab_in_stem(plot_stem)
    if (!is.na(lab) && group != "metastatic_labels") {
      lab_slug <- lab_stem_slug(lab)
      artifact_name <- gsub(
        paste0("(^|_)", lab_slug, "(_|$)"), "_", artifact_name, perl = TRUE
      )
    }
    artifact_name <- gsub("_+", "_", artifact_name)
    artifact_name <- sub("^_", "", sub("_$", "", artifact_name))
    if (!nzchar(artifact_name)) "figure" else artifact_name
  }

  # Every stem routes to .../by_figure/<group>/<trimmed-artifact-name>/, whose leaf holds one
  # file per cohort x endpoint cell. Uniform for numbered and supplemental
  # groups alike: the per-artifact level is what keeps sibling panels in a group
  # from sharing a leaf, and COHORT_LEAF (endpoint + subset + exclusion) is
  # what keeps the twelve cells within a stem from colliding.
  output_dir_for_stem <- function(plot_stem) {
    group <- figure_group(plot_stem)
    file.path(FIG_ROOT, "by_figure", group, artifact_name_for_stem(plot_stem, group))
  }

  # save_fig checks only the exact requested
  # destinations; unrelated artifacts are never migrated or cleaned.

  # Compatibility shim: call sites still pass an `out_dir`, but actual routing
  # is derived from each exact `stem` inside save_fig/write_table1. Kept so the
  # 30+ existing call sites need no edit.
  fig_dir <- function(plot_stem) {
    file.path(FIG_ROOT, plot_stem)
  }

  LANDMARKS <- COHORT_LANDMARKS[[COHORT_ARM]]
  TOP_N <- 15

  LLM_LABEL_PATH <- file.path(NEPC_PROJ_PATH, "LLM_NEPC_labels")
  # Cohort- and endpoint-invariant: read once per session, not once per pass.
  manual_annotations <- load_manual_annotations(file.path(LLM_LABEL_PATH, MANUAL_ANNOTATIONS_FILE))
  platinum_mrns <- cached_read_csv(file.path(NEPC_PROJ_PATH, "mrn_lists/platinum_MRN_list.csv"),
                                   show_col_types = FALSE)
  platinum_set <- unique(platinum_mrns$DFCI_MRN)

  # Classifier-derived labels (primary_label/has_nepc/has_avpc). The sole label
  # source: it drives Figure 2 v3 and the all-lab longitudinal/KM
  # stratification in Sections 4-5. NULL if absent.
  llm_classifier_labels <- load_llm_strata(llm_annotations_path)
  if (!is.null(llm_classifier_labels)) {
    llm_classifier_labels <- llm_classifier_labels %>%
      mutate(is_platinum = DFCI_MRN %in% platinum_set)
    cat(sprintf("llm_classifier_labels: %s rows (%s platinum+, %s platinum-)\n",
                format(nrow(llm_classifier_labels), big.mark = ","),
                format(sum(llm_classifier_labels$is_platinum), big.mark = ","),
                format(sum(!llm_classifier_labels$is_platinum), big.mark = ",")))
  }

  # Reuse the same artifact/cohort/endpoint from the preceding figure-major
  # layout, then render the PDF if it is missing. Overwrite always regenerates.
  save_fig <- function(plot, out_dir, stem, width, height, prefix = COHORT_LEAF) {
    capture <- getOption("compass.figure_capture")
    if (is.function(capture)) {
      capture(plot, file.path(output_dir_for_stem(stem), prefix), width, height, stem)
      notify_progress("panel_done", paste("prepared", stem))
      return(invisible(NULL))
    }
    save_started <- proc.time()[["elapsed"]]
    # `out_dir` is retained for call-site compatibility. The directory already
    # encodes group/artifact/endpoint, so the filename carries only the cohort
    # identity -- that is what makes one leaf directory a six-way comparison.
    output_dir <- output_dir_for_stem(stem)
    output_stem <- prefix

    pdf_out <- file.path(output_dir, paste0(output_stem, ".pdf"))
    if (!overwrite) reuse_previous_figure_layout(pdf_out)
    need_pdf <- overwrite || !figure_file_complete(pdf_out)
    if (!need_pdf) {
      message("skipped completed figure: ", stem)
      notify_progress("panel_skipped", stem)
      # Do not force a lazy plot expression when its file already exists.
      return(invisible(NULL))
    }
    if (is.null(progress)) message("rendering ", stem, " ...")
    notify_progress("panel_start", stem)
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    plot <- prepare_figure_text(plot, width)
    figure_require_font()
    # A failed/interrupted graphics device cannot leave a completed-looking
    # destination. Keep any previous figure until the replacement is ready.
    temporary <- tempfile(".compass-render-", tmpdir = output_dir, fileext = ".pdf")
    on.exit(unlink(temporary), add = TRUE)
    ggsave(temporary, plot = plot, width = width, height = height, units = "in",
           bg = "white", device = figure_pdf_device)
    if (!figure_file_complete(temporary))
      stop("Graphics device produced an incomplete file for ", pdf_out)
    if (!file.rename(temporary, pdf_out))
      stop("Could not publish figure ", pdf_out)
    message("wrote ", pdf_out)

    message(sprintf("rendered %s in %.1f seconds", stem,
                    proc.time()[["elapsed"]] - save_started))
    notify_progress("panel_done", stem)
    invisible(plot)
  }

  COHORT_LABEL <- "PROFILE"

  load_profile_patient_and_labs <- function(path, id_col = "DFCI_MRN") {
    cached_profile_patient_and_labs(path, id_col)
  }

  restrict_to_base_landmark_cohort <- function(patient_df, inputs_dir,
                                               id_col, landmark) {
    availability_path <- file.path(inputs_dir, "landmark_mrn_availability.csv")
    if (!file.exists(availability_path))
      stop(availability_path, " not found -- re-run build_prediction_inputs.py")
    eligible_col <- sprintf("eligible_landmark_%s", landmark)
    availability <- read_csv(availability_path, col_select = all_of(c(id_col, eligible_col)),
                             show_col_types = FALSE)
    eligible <- tolower(as.character(availability[[eligible_col]])) %in% c("true", "1")
    cohort_ids <- suppressWarnings(as.numeric(availability[[id_col]][eligible]))
    cohort_ids <- unique(cohort_ids[is.finite(cohort_ids)])
    patient_ids <- suppressWarnings(as.numeric(patient_df[[id_col]]))
    patient_df[patient_ids %in% cohort_ids, , drop = FALSE]
  }

  load_attrition <- function(inputs_dir) {
    landmark_path <- file.path(inputs_dir, "landmark_attrition.json")
    if (!file.exists(landmark_path))
      stop(landmark_path, " not found -- re-run build_prediction_inputs.py")
    fromJSON(landmark_path, simplifyVector = FALSE)
  }

  load_icd_prostate_mrn_flags <- function(path) {
    if (!file.exists(path))
      stop(path, " not found -- re-run compile_COMPASS_cohort_data.py")
    required <- c(
      ID_COL,
      "HAS_NON_PROSTATE_PRIMARY",
      "DATED_PROSTATE_DIAGNOSIS",
      "MALE",
      "HAS_POST_ADT_EXCLUSION_CANCER",
      "PARPI_EXPOSED",
      "ARPI_DOCETAXEL_EXPOSED",
      "ADT_EXPOSED",
      "PLATINUM_BEFORE_DIAGNOSIS",
      "HAS_5_OR_MORE_PSA_TESTS",
      "ELIGIBLE"
    )
    flags <- cached_read_csv(path, show_col_types = FALSE)
    missing <- setdiff(required, names(flags))
    if (length(missing) > 0)
      stop(sprintf("%s is missing required columns: %s",
                   path, paste(missing, collapse = ", ")))
    if (any(is.na(flags[[ID_COL]])))
      stop(path, " contains missing DFCI_MRN values")
    if (anyDuplicated(flags[[ID_COL]]) > 0)
      stop(path, " must contain exactly one row per DFCI_MRN")
    flags <- flags %>%
      select(all_of(required)) %>%
      mutate(across(-all_of(ID_COL), ~ suppressWarnings(as.numeric(.x))))
    flag_cols <- setdiff(required, ID_COL)
    invalid <- flag_cols[
      vapply(flags[flag_cols], function(x) any(is.na(x) | !x %in% c(0, 1)), logical(1))
    ]
    if (length(invalid) > 0)
      stop(sprintf("%s has non-binary or missing values in: %s",
                   path, paste(invalid, collapse = ", ")))
    flags
  }

  render_consort_panel <- function(mrn_flags) {
    keep <- rep(TRUE, nrow(mrn_flags))
    steps <- list(
      c("ICD-defined prostate cancer", sum(keep))
    )
    keep <- keep & mrn_flags$DATED_PROSTATE_DIAGNOSIS == 1
    steps[[length(steps) + 1]] <- c("Dated prostate cancer diagnosis", sum(keep))
    keep <- keep & mrn_flags$MALE == 1
    steps[[length(steps) + 1]] <- c("Male sex", sum(keep))
    keep <- keep & mrn_flags$HAS_5_OR_MORE_PSA_TESTS == 1
    steps[[length(steps) + 1]] <- c("\u22655 PSA tests", sum(keep))
    keep <- keep & mrn_flags$ADT_EXPOSED == 1
    steps[[length(steps) + 1]] <- c("ADT on/after prostate diagnosis", sum(keep))
    keep <- keep & mrn_flags$PARPI_EXPOSED == 0
    steps[[length(steps) + 1]] <- c("No PARPi exposure", sum(keep))
    keep <- keep & mrn_flags$PLATINUM_BEFORE_DIAGNOSIS == 0
    steps[[length(steps) + 1]] <- c("No platinum before prostate diagnosis", sum(keep))
    keep <- keep & mrn_flags$HAS_POST_ADT_EXCLUSION_CANCER == 0
    steps[[length(steps) + 1]] <- c(
      "No bladder, lung, head and neck,\nor testicular cancer after first ADT",
      sum(keep)
    )
    if (any(keep != (mrn_flags$ELIGIBLE == 1)))
      stop("CONSORT criteria do not reproduce the ELIGIBLE flag")
    if (!IS_ADT) {
      keep <- keep & mrn_flags$ARPI_DOCETAXEL_EXPOSED == 1
      steps[[length(steps) + 1]] <- c("ARPI/docetaxel exposure", sum(keep))
    }
    n_steps <- length(steps)
    df <- tibble(i = seq_len(n_steps) - 1,
                 label = vapply(steps, `[`, character(1), 1),
                 n = as.numeric(vapply(steps, `[`, character(1), 2))) %>%
      mutate(ytop = n_steps - i, ycen = ytop - 0.5,
             text = sprintf("%s\nn = %s", label, format(n, big.mark = ",")),
             xmin = 0.07, xmax = 0.93, ymin = ycen - 0.31, ymax = ycen + 0.31)
    arrows <- df %>% filter(i < n_steps - 1) %>%
      mutate(y = ymin, yend = df$ymax[match(i + 1, df$i)], x = 0.5, xend = 0.5)
    ggplot(df) +
      geom_rect(aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
                fill = "#eef1f5", color = "#5d6d7e", linewidth = 0.4) +
      geom_text(aes(x = 0.5, y = ycen, label = text), size = 3.1, lineheight = 0.9) +
      geom_segment(data = arrows, aes(x = x, xend = xend, y = y, yend = yend),
                   arrow = arrow(length = unit(0.12, "cm"), type = "closed"),
                   color = "#5d6d7e", linewidth = 0.45) +
      labs(title = "ICD prostate cohort workflow") +
      xlim(0, 1) + ylim(0, n_steps + 0.25) +
      theme_void() + theme(plot.title = element_text(face = "bold", size = 12.5, hjust = 0.5))
  }

  # Patient counts for the DFCI data-availability panel. The Gleason and
  # sequencing counts are the available-case subsets of the ADT day-0 cohort
  # written by build_somatic_gleason_inputs.py.
  dfci_data_availability <- function(mrn_flags, attrition, inputs_dir) {
    manifest_path <- file.path(inputs_dir, "somatic_gleason", "build_manifest.json")
    if (!file.exists(manifest_path)) {
      message("Skipping figure1d_dfci_data: missing ", manifest_path)
      return(NULL)
    }
    manifest <- fromJSON(manifest_path, simplifyVector = FALSE)
    count <- function(x, what) {
      x <- suppressWarnings(as.numeric(x))
      if (length(x) != 1L || !is.finite(x) || x < 0)
        stop("Data-availability count missing or invalid: ", what)
      x
    }
    tibble(
      label = c("ICD-10 prostate cancer", "ADT-exposed", "Gleason before ADT day 0",
                "Tumor sequencing", "ADT day-0 prediction cohort"),
      n = c(nrow(mrn_flags), sum(mrn_flags$ADT_EXPOSED == 1),
            count(manifest$available_case_sensitivities$gleason$n_patients_by_landmark[["0"]],
                  "Gleason day 0"),
            count(manifest$cohort_sizes$sequencing_any, "sequencing_any"),
            count(attrition[["eligible_by_landmark"]][["0"]], "eligible day 0")))
  }

  render_data_availability_panel <- function(counts) {
    counts <- counts %>% mutate(y = rev(seq_len(n())))
    ggplot(counts, aes(y = y)) +
      geom_rect(aes(xmin = 0, xmax = n, ymin = y - 0.32, ymax = y + 0.32), fill = "#5d6d7e") +
      geom_text(aes(x = n, label = format(n, big.mark = ",")),
                hjust = -0.12, size = 3.1, color = "#3a3a38") +
      scale_y_continuous(breaks = counts$y, labels = counts$label,
                         limits = c(0.5, nrow(counts) + 0.5), expand = expansion(mult = 0)) +
      scale_x_continuous(limits = c(0, max(counts$n) * 1.18), labels = scales::comma,
                         expand = expansion(mult = c(0, 0.01))) +
      labs(title = "DFCI data availability", x = "Patients", y = NULL,
           subtitle = "Gleason and sequencing counts are within the ADT day-0 cohort") +
      theme_classic(base_size = 11) +
      theme(panel.grid.major.x = element_line(color = "grey92", linewidth = 0.3),
            axis.ticks.y = element_blank(), axis.line.y = element_blank(),
            plot.title = element_text(face = "bold", size = 12.5))
  }

  platinum_km_inputs <- function(df) {
    t_platinum <- suppressWarnings(as.numeric(df$t_platinum))
    platinum <- coalesce(suppressWarnings(as.numeric(df$PLATINUM)), 0) == 1
    t_death <- if ("t_death" %in% names(df)) suppressWarnings(as.numeric(df$t_death)) else t_platinum
    death <- if ("DEATH" %in% names(df))
      coalesce(suppressWarnings(as.numeric(df$DEATH)), 0) == 1 else rep(FALSE, nrow(df))
    death_time <- ifelse(death, t_death, Inf)
    duration <- pmin(t_platinum, death_time, na.rm = TRUE)
    duration[is.infinite(duration)] <- NA_real_
    event <- platinum & !is.na(t_platinum) & (t_platinum <= death_time)
    valid <- !is.na(duration) & is.finite(duration) & duration >= 0
    tibble(row_id = which(valid), time = duration[valid], event = as.integer(event[valid]))
  }

  render_km_panel <- function(patient_df) {
    d <- platinum_km_inputs(patient_df)
    fit <- survfit(Surv(time, event) ~ 1, data = d)
    lab <- sprintf("%s — %s (n=%s)", COHORT_LABEL, COHORT_DISPLAY,
                   format(nrow(d), big.mark = ","))
    gg <- ggsurvplot(fit, data = d, conf.int = TRUE, censor = FALSE,
                     palette = "#1f3a93", legend = "none",
                     xlab = sprintf("Days from %s", ANCHOR_LABEL),
                     ylab = "Platinum-free probability", title = "Platinum-free survival",
                     ggtheme = theme_fig())
    gg$plot + coord_cartesian(ylim = c(0, 1.02)) +
      labs(subtitle = lab)
  }

  overlay_hist <- function(series, xlab, title, bins = 50, xlim_max = NULL,
                           use_count = FALSE, axis_text_size = NULL) {
    v <- suppressWarnings(as.numeric(series)); v <- v[is.finite(v)]
    n_total <- length(v)
    if (!is.null(xlim_max)) v <- v[v >= 0 & v <= xlim_max]
    if (!length(v)) return(ggplot() + annotate("text", x = 0, y = 0, label = "(no data)") +
                             theme_void() + labs(title = title))
    lab <- if (length(v) == n_total) {
      sprintf("%s (n=%s)", COHORT_LABEL, format(n_total, big.mark = ","))
    } else {
      sprintf("%s (n=%s; %s shown)", COHORT_LABEL,
              format(n_total, big.mark = ","), format(length(v), big.mark = ","))
    }
    ylab <- if (use_count) "Count" else "Density"
    p <- ggplot(tibble(v = v), aes(v))
    if (use_count) {
      p <- p + geom_histogram(bins = bins, fill = "#1f3a93", color = "white",
                              linewidth = 0.15, alpha = 0.75)
    } else {
      p <- p + geom_histogram(aes(y = after_stat(density)), bins = bins,
                              fill = "#1f3a93", color = "white", linewidth = 0.15, alpha = 0.75)
    }
    p <- p +
      labs(x = xlab, y = ylab, title = title) +
      labs(subtitle = lab)
    if (!is.null(xlim_max)) p <- p + coord_cartesian(xlim = c(0, xlim_max))
    if (!is.null(axis_text_size))
      p <- p + theme(axis.title = element_text(size = axis_text_size + 1.5),
                     axis.text  = element_text(size = axis_text_size))
    p
  }

  # --- Table 1 helpers (baseline characteristics) ---
  mean_sd <- function(x) {
    s <- suppressWarnings(as.numeric(x)); s <- s[is.finite(s)]
    if (length(s) == 0) return("n/a")
    sprintf("%.1f \u00b1 %.1f", mean(s), sd(s))
  }
  median_iqr <- function(x) {
    s <- suppressWarnings(as.numeric(x)); s <- s[is.finite(s)]
    if (length(s) == 0) return("n/a")
    q <- quantile(s, c(0.25, 0.5, 0.75), names = FALSE)
    sprintf("%.1f (%.1f\u2013%.1f)", q[2], q[1], q[3])
  }
  count_pct <- function(mask, total) {
    n <- sum(mask, na.rm = TRUE)
    if (total == 0) return("n/a")
    sprintf("%s (%.1f%%)", format(n, big.mark = ","), 100 * n / total)
  }

  build_table1 <- function(patient_df) {
    n <- nrow(patient_df)
    rows <- list(c("N", format(n, big.mark = ",")))
    add <- function(k, v) rows[[length(rows) + 1]] <<- c(k, v)

    age <- patient_df[["AGE_AT_TREATMENTSTART"]]
    add("Age at first treatment, mean \u00b1 SD", mean_sd(age))
    add("Age at first treatment, median (IQR)",    median_iqr(age))
    plat <- suppressWarnings(as.numeric(patient_df[["PLATINUM"]])); plat[is.na(plat)] <- 0
    add("Platinum exposure, n (%)", count_pct(plat > 0, n))
    add(sprintf("Median follow-up from %s, days (IQR)", ANCHOR_LABEL),
        median_iqr(patient_df[["t_last_contact"]]))

    tibble(Characteristic = vapply(rows, `[`, character(1), 1),
           Value          = vapply(rows, `[`, character(1), 2))
  }

  to_markdown_table <- function(df) {
    header <- paste0("| ", paste(names(df), collapse = " | "), " |")
    sep    <- paste0("| ", paste(rep("---", ncol(df)), collapse = " | "), " |")
    body   <- apply(df, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
    paste(c(header, sep, body), collapse = "\n")
  }

  write_table1 <- function(table1, out_base) {
    stem <- basename(out_base)
    output_dir <- output_dir_for_stem(stem)
    out_base <- file.path(output_dir, COHORT_LEAF)
    if(exists("figure_public_path",mode="function")) out_base <- figure_public_path(out_base)
    if(exists("figure_sidecar_path",mode="function")) out_base <- figure_sidecar_path(out_base)
    dir.create(dirname(out_base),recursive=TRUE,showWarnings=FALSE)
    csv <- paste0(out_base, ".csv"); md_p <- paste0(out_base, ".md")
    write_csv(table1, csv)
    writeLines(to_markdown_table(table1), md_p)
    capture <- getOption("compass.figure_table_capture")
    if (is.function(capture)) capture(c(csv, md_p))
    c(csv, md_p)
  }

  OUT_DIR <- fig_dir("figure1_cohort")
  ID_COL <- "DFCI_MRN"

  notify_progress("stage", "Figure 1: cohort inputs")
  message(sprintf("Loading cohort-specific attrition counts from %s ...", INPUTS_DIR))
  attrition <- load_attrition(INPUTS_DIR)
  message(sprintf("Loading ICD prostate MRN workflow flags from %s ...",
                  ICD_PROSTATE_MRN_FLAGS_CSV))
  icd_prostate_mrn_flags <- load_icd_prostate_mrn_flags(ICD_PROSTATE_MRN_FLAGS_CSV)
  message(sprintf("Loading %s ...", LONGITUDINAL_CSV))
  split <- load_profile_patient_and_labs(LONGITUDINAL_CSV, id_col = ID_COL)
  base_landmark <- min(as.integer(names(attrition[["eligible_by_landmark"]])))
  if (IS_ADT && base_landmark != 0L)
    stop("Figure 2/2v2 require the ADT time-0 cohort, but the earliest available landmark is ",
         base_landmark)
  patient_df <- restrict_to_base_landmark_cohort(split$patient_df, INPUTS_DIR,
                                                ID_COL, base_landmark)
  expected_n <- as.integer(attrition[["eligible_by_landmark"]][[as.character(base_landmark)]])
  stopifnot(nrow(patient_df) == expected_n)
  message(sprintf("  selected base-landmark cohort: patients=%s  labs=%s",
                  format(nrow(patient_df), big.mark = ","),
                  format(sum(patient_df$lab_rows, na.rm = TRUE), big.mark = ",")))

  pA <- render_consort_panel(icd_prostate_mrn_flags)
  save_fig(pA, OUT_DIR, "figure1a_consort", 7.5, 8.0)
  if (show) print(pA)
  pB <- render_km_panel(patient_df)
  save_fig(pB, OUT_DIR, "figure1b_km", 6.5, 4.8)
  if (show) print(pB)
  if (IS_CANONICAL_ADT && EMIT_ENDPOINT_INDEPENDENT) {
    availability <- dfci_data_availability(icd_prostate_mrn_flags, attrition, INPUTS_DIR)
    if (!is.null(availability)) {
      pD <- render_data_availability_panel(availability)
      save_fig(pD, OUT_DIR, "figure1d_dfci_data", 6.0, 3.4)
      if (show) print(pD)
    }
    # Federated per-site counts come from the within-site export; the plot
    # helpers live in figure_workflow.R, which COMPASS/figures/run_figures.R sources alongside.
    sites <- if (is.null(federated_path)) NULL else load_federated_sites(federated_path)
    if (!is.null(sites)) {
      pE <- figure1_site_counts_plot(sites)
      save_fig(pE, OUT_DIR, "figure1e_site_counts", 6.0, 3.0)
      if (show) print(pE)
    } else message("Skipping figure1e_site_counts: no federated within-site cohort counts")
  }

  span_series <- patient_df$record_span_days
  km_inputs <- platinum_km_inputs(patient_df)
  event_rows <- km_inputs$row_id[km_inputs$event == 1]
  dx_to_anchor <- suppressWarnings(as.numeric(patient_df$t_dx_to_anchor))
  dx_to_anchor_finite <- dx_to_anchor[is.finite(dx_to_anchor) & dx_to_anchor >= 0]
  dx_to_anchor_cap <- if (length(dx_to_anchor_finite)) {
    as.numeric(quantile(dx_to_anchor_finite, 0.99, names = FALSE))
  } else NULL
  timing_panels <- list(
    overlay_hist(span_series, "Record span (days)", "Per-patient lab record span", 50),
    overlay_hist(dx_to_anchor, sprintf("Days: diagnosis → %s", ANCHOR_LABEL),
                 sprintf("Diagnosis → %s (through 99th percentile)", ANCHOR_LABEL), 50,
                 xlim_max = dx_to_anchor_cap),
    overlay_hist(patient_df$t_platinum[event_rows],
                 sprintf("Days from %s to platinum", ANCHOR_LABEL),
                 "Time to platinum", 40,
                 xlim_max = 6 * 365.25, use_count = TRUE, axis_text_size = 14) +
      theme(plot.title = element_text(size = 14))
  )
  timing_stems <- c("figure1c_span", "figure1c_dx_to_tx", "figure1c_time_to_platinum")
  for (i in seq_along(timing_panels)) {
    save_fig(timing_panels[[i]], OUT_DIR, timing_stems[[i]], 5.5, 4.2)
    if (show) print(timing_panels[[i]])
  }

  table1 <- build_table1(patient_df)
  for (p in write_table1(table1, file.path(OUT_DIR, "table1_baseline_characteristics")))
    message(sprintf("wrote %s", p))
  if (show) print(table1)

  if (!IS_ADT)
    message(paste0("Figure 2 v3: emitted once from the ADT pass over the ADT-exposed ",
                   "universe -- skipping for the ARPI figure pass."))

  # Figure 2 validates the LLM classifier against manual annotations and
  # measures platinum enrichment. Its subject is the label set and the platinum
  # MRN list, neither of which depends on the modelled endpoint, so it is
  # emitted once from the canonical ADT/platinum pass rather than duplicated
  # across endpoint and restricted-cohort cells.
  if (IS_ADT && (!EMIT_ENDPOINT_INDEPENDENT || !IS_CANONICAL_ADT))
    message(sprintf(paste0("Figure 2 v3: classifier/platinum-enrichment figures are ",
                           "global and are emitted only from cohort=adt, endpoint=%s -- skipping ",
                           "for cohort=%s, endpoint=%s."),
                    ENDPOINT_INDEPENDENT_FIGURES_ENDPOINT, COHORT, ENDPOINT))

  if (IS_CANONICAL_ADT && EMIT_ENDPOINT_INDEPENDENT) {
  notify_progress("stage", "Figure 2: validation, subtype landscape, and platinum enrichment")
  ## ---- Figure 2 v3 -- classifier labels on the ADT day-0 prediction cohort ----
  # v3 uses primary_subtype_labels.parquet (stems figure2v3_lm0_*). The ICD-10
  # and ADT-exposed MRN sets below are still reported against that cohort.
  if (is.null(llm_classifier_labels)) {
    message("Figure 2 v3: llm_classifier_labels unavailable -- skipping.")
  } else {
    OUT_DIR_V3 <- fig_dir("figure2v3_llm")

    # Panels B/C restored from 3c4efc4 (before their removal in fec33f4).
    # Fixed class order groups the two aggressive classes together for readability.
    CLASS_ORDER <- c("conventional", "avpc", "nepc")
    CLASS_LABELS <- c(conventional = "Conventional", avpc = "AVPC", nepc = "NEPC")

    count_labels <- function(df) {
      df %>% count(primary_label, name = "count") %>%
        mutate(frac = count / sum(count))
    }

    # Annotated 3x3 confusion matrix from multiclass_metrics: manual truth on the
    # y axis (Conventional at top), LLM prediction on the x axis.
    render_confusion_panel <- function(metrics,
                                       title = "Panel A1 — confusion matrix") {
      cm <- metrics$cm %>%
        mutate(truth = factor(CLASS_LABELS[as.character(truth)], levels = rev(CLASS_LABELS[CLASS_ORDER])),
               pred  = factor(CLASS_LABELS[as.character(pred)],  levels = CLASS_LABELS[CLASS_ORDER]))
      thresh <- max(cm$n) / 2
      ggplot(cm, aes(pred, truth, fill = n)) +
        geom_tile(color = "white", linewidth = 1) +
        geom_text(aes(label = format(n, big.mark = ","),
                      color = n > thresh), size = 5, fontface = "bold") +
        scale_fill_gradient(low = "#eaf1fb", high = COLOR_PLATINUM_POS, guide = "none") +
        scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#0b0b0b"), guide = "none") +
        labs(x = "LLM label (prediction)", y = "Manual annotation (truth)", title = title) +
        theme_fig() +
        theme(panel.grid = element_blank(),
              plot.title = element_text(face = "bold", size = 11))
    }

    # One-vs-rest Accuracy, Precision, Recall, Specificity, faceted by class.
    render_metric_bar_panel <- function(metrics) {
      metric_levels <- c("Accuracy","Precision","Recall","Specificity")
      d <- metrics$per_class %>%
        select(class, all_of(metric_levels)) %>%
        tidyr::pivot_longer(-class, names_to = "metric", values_to = "value") %>%
        mutate(metric = factor(metric, levels = metric_levels),
               class  = factor(CLASS_LABELS[as.character(class)], levels = CLASS_LABELS[CLASS_ORDER]))
      ggplot(d, aes(metric, value)) +
        geom_col(fill = COLOR_PLATINUM_POS, width = 0.6) +
        geom_text(aes(label = sprintf("%.2f", value)), vjust = -0.4, size = 3.2, color = "#0b0b0b") +
        facet_wrap(~ class, nrow = 1) +
        coord_cartesian(ylim = c(0, 1.05)) +
        labs(x = NULL, y = "Metric value (one-vs-rest)", title = "Panel A2 — classifier metrics") +
        theme_fig() +
        theme(plot.title = element_text(face = "bold", size = 11),
              axis.text.x = element_text(angle = 45, hjust = 1),
              strip.text = element_text(face = "bold"))
    }

    render_landscape_panel <- function(label_distributions, n_pos, n_neg,
                                       title = "Panel B — subtype landscape by platinum status (descriptive)") {
      if (n_pos + n_neg == 0)
        return(ggplot() + annotate("text", x = 0, y = 0, label = "(no classified patients)") +
                 theme_void() + labs(title = str_wrap(title, 65)))
      d <- label_distributions %>%
        mutate(primary_label   = factor(primary_label, levels = CLASS_ORDER,
                                        labels = CLASS_LABELS[CLASS_ORDER]),
               platinum_status = factor(platinum_status, levels = c("positive","negative"))) %>%
        # Anything outside CLASS_ORDER became NA in the factor() above; keeping it
        # would draw an "NA" column. Callers are expected to have filtered already.
        filter(!is.na(primary_label))
      ggplot(d, aes(primary_label, frac, fill = platinum_status)) +
        geom_col(position = position_dodge(width = 0.8), width = 0.72) +
        scale_fill_manual(
          values = c(positive = COLOR_PLATINUM_POS, negative = COLOR_PLATINUM_NEG),
          labels = c(sprintf("Platinum+ (n=%s)", format(n_pos, big.mark = ",")),
                     sprintf("Platinum- (n=%s)", format(n_neg, big.mark = ","))),
          name = NULL) +
        coord_cartesian(ylim = c(0, 1.0)) +
        # Dodged pairs leave each class only ~1/3 of the panel, so wrap the class
        # names rather than letting neighbouring labels collide at this text size.
        scale_x_discrete(labels = function(x) str_wrap(x, 12)) +
        labs(x = NULL, y = "Fraction within platinum group", title = str_wrap(title, 65)) +
        theme_fig() +
        # Keep cohort counts on separate rows so long labels cannot touch the
        # next colour key, including when the cached grob is rendered smaller.
        guides(fill = guide_legend(ncol = 1, byrow = TRUE)) +
        theme(plot.title = element_text(face = "bold", size = 11),
              axis.title.x = element_blank(),
              axis.title.y = element_text(size = 14, margin = margin(r = 14)),
              axis.text  = element_text(size = 14),
              axis.text.y = element_text(margin = margin(r = 6)),
              axis.text.x = element_text(size = 13, lineheight = 0.95, margin = margin(t = 4)),
              legend.position = "bottom", legend.justification = "center",
              legend.box = "vertical",
              legend.text = element_text(size = 10, margin = margin(l = 6)),
              legend.key.height = grid::unit(16, "pt"),
              legend.margin = margin(t = 8, b = 8))
    }

    render_enrichment_panel <- function(enrichment) {
      panel_title <- "Platinum enrichment among aggressive variants"
      if (enrichment$n_aggressive == 0 || enrichment$n_conventional == 0)
        return(ggplot() + annotate("text", x = 0, y = 0,
                                  label = "(both aggressive and conventional patients required)") +
                 theme_void() + labs(title = str_wrap(panel_title, 48)))
      d <- tibble(
        group = factor(c("Aggressive\n(AVPC + NEPC)", "Conventional"),
                       levels = c("Aggressive\n(AVPC + NEPC)", "Conventional")),
        prop  = c(enrichment$p_agg, enrichment$p_conv),
        lo    = c(enrichment$lo_agg, enrichment$lo_conv),
        hi    = c(enrichment$hi_agg, enrichment$hi_conv),
        n     = c(enrichment$n_aggressive, enrichment$n_conventional),
        k     = c(enrichment$k_agg, enrichment$k_conv)
      )
      ymax <- max(enrichment$hi_agg, enrichment$hi_conv) * 1.35
      ggplot(d, aes(group, prop, fill = group)) +
        geom_col(width = 0.55) +
        geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.18,
                      color = COLOR_NEUTRAL_INK, linewidth = 0.7) +
        scale_fill_manual(values = c(COLOR_PLATINUM_POS, "#9a9890"), guide = "none") +
        annotate("text", x = 1.5, y = ymax * 0.97,
                 label = sprintf("OR = %.1f, Fisher's exact p = %.1e",
                                 enrichment$OR, enrichment$p_value),
                 fontface = "bold", size = 3.7, color = COLOR_NEUTRAL_INK) +
        scale_y_continuous(labels = scales::percent, limits = c(0, ymax)) +
        labs(x = NULL, y = "P(platinum+ | subtype group)",
             title = str_wrap(panel_title, 48)) +
        theme_fig() +
        theme(plot.title = element_text(face = "bold", size = 11))
    }

    # 2x2 aggressive/conventional x platinum+/- contrast with Wilson intervals.
    compute_enrichment <- function(labels_all) {
      df <- labels_all %>%
        filter(primary_label %in% c("conventional", "avpc", "nepc")) %>%
        mutate(aggressive = primary_label %in% c("avpc", "nepc"))
      n_excluded <- nrow(labels_all) - nrow(df)
      ct <- matrix(
        c(sum(df$aggressive  &  df$is_platinum), sum(df$aggressive  & !df$is_platinum),
          sum(!df$aggressive &  df$is_platinum), sum(!df$aggressive & !df$is_platinum)),
        nrow = 2, byrow = TRUE,
        dimnames = list(c("aggressive", "conventional"), c("platinum+", "platinum-")))
      print(ct)
      n_aggressive   <- sum(ct["aggressive", ])
      n_conventional <- sum(ct["conventional", ])
      ft <- if (n_aggressive > 0 && n_conventional > 0) {
        fisher.test(ct, alternative = "greater")
      } else list(estimate = NA_real_, p.value = NA_real_)
      k_agg  <- ct["aggressive",   "platinum+"]
      k_conv <- ct["conventional", "platinum+"]
      w_agg  <- wilson_ci(k_agg,  n_aggressive)
      w_conv <- wilson_ci(k_conv, n_conventional)
      list(ct = ct, n_excluded = n_excluded,
           OR = unname(ft$estimate), p_value = ft$p.value,
           n_aggressive = n_aggressive, n_conventional = n_conventional,
           k_agg = k_agg, k_conv = k_conv,
           p_agg = w_agg[1], lo_agg = w_agg[2], hi_agg = w_agg[3],
           p_conv = w_conv[1], lo_conv = w_conv[2], hi_conv = w_conv[3])
    }

    # Figure 2 is drawn on three nested cohorts, each intersected with the
    # classifier labels: every ICD-10 C61 patient, the ADT-exposed subset, and
    # the ADT day-0 prediction cohort. Platinum status is any dated platinum
    # ever (platinum_any_MRN_list.csv), so it is defined for all three.
    platinum_any_path <- file.path(NEPC_PROJ_PATH, "mrn_lists", "platinum_any_MRN_list.csv")
    if (!file.exists(platinum_any_path))
      stop(platinum_any_path, " not found -- re-run compile_COMPASS_cohort_data.py")
    platinum_any_set <- unique(as.character(
      cached_read_csv(platinum_any_path, show_col_types = FALSE)[[ID_COL]]))
    cohort_mrns <- unique(as.character(patient_df[[ID_COL]]))
    stopifnot(base_landmark == 0L, length(cohort_mrns) == nrow(patient_df))
    icd_mrns <- unique(as.character(icd_prostate_mrn_flags[[ID_COL]]))
    adt_exposed_mrns <- unique(as.character(
      icd_prostate_mrn_flags[[ID_COL]][icd_prostate_mrn_flags$ADT_EXPOSED == 1]))
    n_pred_outside_adt <- sum(!cohort_mrns %in% adt_exposed_mrns)
    if (n_pred_outside_adt > 0)
      message(sprintf(
        "Figure 2 v3: %s prediction-cohort MRN(s) are absent from ADT_EXPOSED == 1 in %s",
        format(n_pred_outside_adt, big.mark = ","), ICD_PROSTATE_MRN_FLAGS_CSV))

    render_figure2v3 <- function(variant, mrns, cohort_label) {
      stem <- function(panel) sprintf("figure2v3_%s_%s", variant, panel)
      labels_all <- llm_classifier_labels %>%
        filter(as.character(DFCI_MRN) %in% mrns) %>%
        mutate(is_platinum = as.character(DFCI_MRN) %in% platinum_any_set)
      message(sprintf(
        "Figure 2 v3 [%s] %s: %s total patients; %s labeled/evaluable (%s source labels total)",
        variant, cohort_label, format(length(mrns), big.mark = ","),
        format(nrow(labels_all), big.mark = ","),
        format(nrow(llm_classifier_labels), big.mark = ",")))
      cohort_note <- sprintf("%s (N=%s; %s with an LLM label)", cohort_label,
                             format(length(mrns), big.mark = ","),
                             format(nrow(labels_all), big.mark = ","))

      ## Panel A: 3-class validation of primary_label against manual chart review.
      ## Panels B/C below use the same primary_label for the subtype landscape and
      ## aggressive-variant group.
      merged <- manual_annotations %>% inner_join(labels_all, by = "DFCI_MRN")
      cat(sprintf("figure2v3 [%s] merged_results: %s rows; manual conventional/AVPC/NEPC = %s\n",
                  variant, format(nrow(merged), big.mark = ","),
                  paste(sapply(CLASS_ORDER, function(cls)
                    format(sum(merged$manual_label == cls, na.rm = TRUE), big.mark = ",")),
                    collapse = "/")))
      metrics <- multiclass_metrics(merged$manual_label, merged$primary_label, CLASS_ORDER)
      n_manual <- metrics$cm %>% group_by(truth) %>% summarise(n = sum(n), .groups = "drop")
      caption_a <- str_wrap(sprintf(
        "%s; %s chart-reviewed/labeled patients (manual %s); overall 3-class accuracy %.2f (primary_subtype_labels.parquet).",
        cohort_note, format(metrics$N, big.mark = ","),
        paste(sprintf("%s=%s", CLASS_LABELS[as.character(n_manual$truth)],
                      format(n_manual$n, big.mark = ",")), collapse = ", "),
        metrics$Accuracy), 95)
      pA1 <- render_confusion_panel(metrics, "Subtype classifier agreement") +
        labs(caption = caption_a) +
        theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK))
      pA2 <- render_metric_bar_panel(metrics) +
        labs(title = "Subtype classifier metrics", caption = caption_a) +
        theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK))
      save_fig(pA1, OUT_DIR_V3, stem("confusion_matrix"), 6.2, 5.8)
      save_fig(pA2, OUT_DIR_V3, stem("metric_bar"), 9.0, 5.0)
      if (show) print(pA1)
      if (show) print(pA2)

      ## Panel B -- subtype landscape by platinum status (3-class primary_label).
      # load_llm_strata coerces primary_label values outside the three modeled
      # classes to NA, and those rows would draw an "NA" bar.
      labels_classified <- labels_all %>% filter(!is.na(primary_label))
      n_unclassified <- nrow(labels_all) - nrow(labels_classified)
      if (n_unclassified > 0)
        message(sprintf(
          "figure2v3 [%s] Panel B: dropped %s row(s) without one of the three primary_label classes",
          variant, format(n_unclassified, big.mark = ",")))
      platinum_positive <- labels_classified %>% filter(is_platinum) %>% count_labels() %>%
        mutate(platinum_status = "positive")
      platinum_negative <- labels_classified %>% filter(!is_platinum) %>% count_labels() %>%
        mutate(platinum_status = "negative")
      n_pos <- sum(platinum_positive$count)
      n_neg <- sum(platinum_negative$count)
      caption_b <- sprintf("%s; %s classified%s; platinum+ n=%s, platinum- n=%s.",
                           cohort_note, format(nrow(labels_classified), big.mark = ","),
                           if (n_unclassified > 0)
                             sprintf(" (%s labeled row(s) outside the three classes excluded)",
                                     format(n_unclassified, big.mark = ","))
                           else "",
                           format(n_pos, big.mark = ","), format(n_neg, big.mark = ","))
      pB <- render_landscape_panel(
          bind_rows(platinum_positive, platinum_negative), n_pos, n_neg,
          "Subtype landscape by platinum status") +
        labs(caption = str_wrap(caption_b, 85)) +
        theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK, hjust = 0.5))
      save_fig(pB, OUT_DIR_V3, stem("subtype_landscape"), 9.5, 8)

      ## Panel C -- aggressive (avpc+nepc) vs conventional platinum enrichment.
      enrichment <- compute_enrichment(labels_all)
      cat(sprintf("figure2v3 [%s] enrichment: OR = %.2f, Fisher p = %.3g\n",
                  variant, enrichment$OR, enrichment$p_value))
      caption_c <- sprintf("%s; excludes unclassified labels (%s rows). Error bars are 95%% Wilson intervals. OR=%.1f, one-sided Fisher p=%.1e.",
                           cohort_note, format(enrichment$n_excluded, big.mark = ","),
                           enrichment$OR, enrichment$p_value)
      pC <- render_enrichment_panel(enrichment) + labs(caption = str_wrap(caption_c, 58)) +
        theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK, hjust = 0.5))
      save_fig(pC, OUT_DIR_V3, stem("enrichment"), 6.0, 5.8)
      if (show) print(pB)
      if (show) print(pC)
    }

    render_figure2v3("lm0", cohort_mrns, "ADT day-0 prediction cohort")
  }
  }

  if (IS_CANONICAL_ADT && !is.null(cohort_forest_config)) {
    notify_progress("stage", "PSA/testosterone forest across ADT cohorts")
    sys.source(cohort_forest_config$script, envir = environment())
    landmark <- cohort_forest_config$landmark
    forest <- load_cohort_forest(NEPC_PROJ_PATH, cohort_forest_config$cohorts, ENDPOINT, landmark)
    if (nrow(forest)) {
      p_forest <- plot_cohort_forest(forest, ENDPOINT, landmark)
      # Fixed slide geometry from figure_supplements.R, not a per-stat height:
      # this figure is presented as one full-bleed 16:9 slide.
      save_fig(p_forest, OUT_DIR, sprintf("cohort_forest_%s_landmark%d", ENDPOINT, landmark),
               COHORT_FOREST_SLIDE_SIZE[["width"]], COHORT_FOREST_SLIDE_SIZE[["height"]])
      if (show) print(p_forest)
    } else message("Cohort forest: no PSA/testosterone estimates available for ", ENDPOINT)
  }

  ## ---- Unified ADT/LLM metastatic/regex-stage diagnostics ----
  if (IS_CANONICAL_ADT && EMIT_ENDPOINT_INDEPENDENT && !is.null(metastatic_supplement)) {
    sys.source(metastatic_supplement$r_script, envir = environment())
    stems <- metastatic_supplement_stems()
    complete <- vapply(stems, function(stem) {
      base <- file.path(output_dir_for_stem(stem), COHORT_LEAF)
      figure_file_complete(paste0(base, ".pdf"))
    }, logical(1))
    if (!overwrite && all(complete)) {
      for (stem in stems) notify_progress("panel_skipped", stem)
      message("Metastatic-label supplement: all subpanels complete; skipping preparation")
    } else {
      render_metastatic_supplements(
        metastatic_supplement, patient_df,
        save_panel = function(plot, stem, width, height)
          save_fig(plot, fig_dir("metastatic_labels"), stem, width, height),
        notify = notify_progress, show = show
      )
    }
  }
}
