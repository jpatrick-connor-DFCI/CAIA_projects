# Per-cohort figure-generation pipeline called from 05_figures.Rmd once per
# cohort in the same R session.
#
# Required packages (install once on the cluster):
#   install.packages(c("tidyverse", "survival", "survminer", "broom",
#                       "ggrepel", "jsonlite", "scales", "mgcv"))
#   # optional: install.packages(c("ggpattern", "ragg"))
#   #   ggpattern -> striped baseline bars in Fig 4a
#   #   ragg      -> crisper high-DPI PNG device (falls back to default if absent)
suppressPackageStartupMessages({
  library(tidyverse)   # dplyr, ggplot2, readr, tidyr, purrr, stringr, forcats
  library(survival)
  library(survminer)
  library(broom)
  library(ggrepel)
  library(jsonlite)
  library(scales)
})

# ---- Publication-quality rendering defaults (parallels the Python rcParams) ----
# 600-dpi PNGs on a white background; use the crisper {ragg} device when available.
SAVE_DPI <- 600
HAS_RAGG <- requireNamespace("ragg", quietly = TRUE)

# Shared ggplot theme: publication typography, thin spines, white background.
theme_fig <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
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
    )
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

# Bottom 20% vs top 20%; the middle 60% is dropped rather than plotted. Same
# tie contract as figure_tertiles: equal measurements never land in opposite
# arms. This matters most for post-ADT PSA, where a pile of tied values sits at
# the assay floor and can straddle the 20th-percentile cut -- splitting those by
# row order would manufacture separation between clinically identical patients.
figure_extreme_quintiles <- function(values) {
  values <- suppressWarnings(as.numeric(values))
  valid <- is.finite(values)
  result <- rep(NA_character_, length(values))
  if (sum(valid) < 2L) return(result)
  cuts <- as.numeric(quantile(values[valid], c(0.2, 0.8), names = FALSE))
  if (cuts[1] >= cuts[2]) return(result)
  # Closed at the bottom cut, open at the top: ties at either edge stay whole.
  low <- valid & values <= cuts[1]
  high <- valid & values > cuts[2]
  result[low] <- "Bottom 20%"
  result[high] <- "Top 20%"
  if (!any(low) || !any(high)) return(rep(NA_character_, length(values)))
  attr(result, "cutpoints") <- cuts
  result
}

figure_gleason_groups <- function(values) {
  x <- suppressWarnings(as.numeric(values))
  valid <- is.finite(x) & x == floor(x) & x >= 2 & x <= 10
  out <- rep(NA_character_, length(x))
  out[valid] <- ifelse(x[valid] <= 6, "Gleason ≤6", paste("Gleason", x[valid]))
  out
}

significant_mutation_features <- function(results) {
  if (!all(c("feature", "q_value") %in% names(results))) return(character())
  selected <- !is.na(results$q_value) & results$q_value < .05 &
    grepl("_(SNV|SV|AMP|DEL)$", as.character(results$feature))
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
                                     note = NULL) {
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
  legend_labels <- setNames(sprintf("%s (n=%s; events=%s)", counts$stratum,
                                    counts$n, counts$events), as.character(counts$stratum))
  colors <- setNames(c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")[seq_along(group_order)], group_order)
  logrank <- tryCatch({
    test <- survival::survdiff(survival::Surv(time, event) ~ stratum, data = d)
    pchisq(test$chisq, length(test$n) - 1, lower.tail = FALSE)
  }, error = function(e) NA_real_)
  ggplot(curves, aes(time, survival, color = stratum, fill = stratum)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = .08, color = NA,
                show.legend = FALSE, na.rm = TRUE) +
    geom_step(linewidth = .85) +
    scale_color_manual(values = colors, breaks = group_order, labels = legend_labels, name = NULL) +
    scale_fill_manual(values = colors, guide = "none") +
    coord_cartesian(ylim = c(0, 1.02)) +
    labs(x = paste("Days from", origin), y = "Platinum-free probability", title = title,
         subtitle = if (is.finite(logrank)) sprintf("Log-rank p = %.3g", logrank) else "Log-rank p unavailable",
         caption = note) + theme_fig() +
    guides(color = guide_legend(ncol = 1)) +
    theme(legend.position = "bottom", legend.text = element_text(size = 10))
}

prepare_figure_text <- function(plot, width) {
  if (!inherits(plot, "ggplot")) return(plot)
  # Wrap outside-panel text against the output width, without changing labels
  # embedded in the data. plot alignment gives captions the full device width.
  widths <- c(title = floor(width * 8), subtitle = floor(width * 10), caption = floor(width * 12))
  for (field in names(widths)) {
    value <- plot$labels[[field]]
    if (is.character(value) && length(value) == 1L)
      plot$labels[[field]] <- paste(vapply(strsplit(value, "\n", fixed = TRUE)[[1]],
        function(line) paste(strwrap(line, width = widths[[field]]), collapse = "\n"), character(1)), collapse = "\n")
  }
  plot + theme(plot.title.position = "plot", plot.caption.position = "plot",
               plot.margin = margin(12, 16, 12, 12))
}


# Colorblind-safe categorical palette (fixed assignment), shared by the
# Figure 2 (LLM subtype / platinum enrichment) panels:
#   blue -> platinum+, orange -> platinum-. Kept consistent everywhere.
COLOR_PLATINUM_POS <- "#2a78d6"   # blue
COLOR_PLATINUM_NEG <- "#eb6834"   # orange
COLOR_NEUTRAL_INK  <- "#52514e"   # secondary ink, for annotations/text only

# Root for the newer classifier-derived LLM NEPC labels (LLM_NEPC_classifier_labels.tsv).
# Deliberately a different root than NEPC_PROJ_PATH: this file lives under the
# user's data/LLM_annotations/ tree, not the CAIA/COMPASS project tree, and is
# is the sole LLM label source for Figure 2 v3 and the Section 4-5
# stratifications. Overridable per-call by generate_figures(); this default matches
# the path used everywhere else in the pipeline invocation.
DEFAULT_LLM_ANNOTATIONS_PATH <- "/data/gusev/USERS/jpconnor/data/LLM_annotations/LLM_NEPC_labels"

# Figure 2 uses primary subtypes for landscape/enrichment panels. Longitudinal
# lab figures use only the binary NEPC stratum in FIGURE_LLM_STRATA below.
LLM_STRATA <- list(
  primary_label = list(
    col = "primary_label",
    levels = c("conventional", "avpc", "nepc", "biomarker")
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

# A biomarker primary label is reserved for these prostate-relevant genes.
# Token boundaries prevent partial matches inside unrelated gene/text strings.
BIOMARKER_PRIMARY_GENES <- c("BRCA1", "BRCA2", "PTEN", "TP53", "RB1")
BIOMARKER_PRIMARY_PATTERN <- paste0(
  "(^|[^A-Z0-9])(",
  paste(BIOMARKER_PRIMARY_GENES, collapse = "|"),
  ")([^A-Z0-9]|$)"
)

normalize_classifier_primary_labels <- function(primary_label, reported_biomarkers,
                                                has_nepc, has_avpc) {
  labels <- str_to_lower(str_trim(as.character(primary_label)))
  biomarker_text <- toupper(ifelse(is.na(reported_biomarkers), "",
                                   as.character(reported_biomarkers)))
  qualifying_biomarker <- grepl(BIOMARKER_PRIMARY_PATTERN, biomarker_text, perl = TRUE)
  needs_fallback <- !is.na(labels) & labels == "biomarker" & !qualifying_biomarker

  labels[needs_fallback & !is.na(has_nepc) & has_nepc == 1] <- "nepc"
  labels[needs_fallback & labels == "biomarker" &
           !is.na(has_avpc) & has_avpc == 1] <- "avpc"
  labels[needs_fallback & labels == "biomarker"] <- "conventional"
  labels
}

# load_llm_strata: read LLM_NEPC_classifier_labels.tsv and normalize biomarker
# primary labels using the reported biomarker field. Returns NULL with a
# message if the file is absent, so every downstream stratified-plot loop can
# skip cleanly rather than erroring.
# ---------------------------------------------------------------------------
# Loop-invariant read cache.
#
# 05_figures.Rmd calls generate_figures() once per cohort x endpoint -- 6 x 2 =
# 12 passes. Several inputs do not vary across that loop at all (the LLM label
# set, the platinum MRN list, the ICD flag table) and one varies only with the
# treatment arm (the longitudinal labs CSV, by far the largest file the
# pipeline reads). Without a cache the longitudinal table alone is parsed 24
# times per knit: twice per pass, once for the patient/lab split and again for
# the Figure 7 canonical-lab loader, neither of which knows about the other.
#
# cached_read_csv() memoizes on the resolved path plus the readr arguments that
# affect the RESULT (col_select, col_types, n_max), so a header-only probe and
# a full read of the same file stay distinct entries. Values are returned as-is
# rather than copied: every consumer below treats them as read-only, and dplyr
# verbs copy on modify anyway.
#
# Correctness note: the cache lives for the R session, so a knit that
# regenerates an input mid-run would serve a stale frame. That does not happen
# here -- 01/02/03 finish before this document is knit -- but clear_read_cache()
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

# TSV twin of cached_read_csv, for the LLM classifier label table.
cached_read_tsv <- function(path, ...) {
  args <- list(...)
  resolved <- tryCatch(normalizePath(path, mustWork = TRUE),
                       error = function(e) path)
  key <- paste0("tsv:", resolved, "|",
                paste(names(args), vapply(args, function(a)
                  paste(format(a), collapse = ","), character(1)),
                  sep = "=", collapse = "|"))
  if (!is.null(.read_cache[[key]])) {
    st <- .read_cache[[".stats"]]; st$hits <- st$hits + 1L
    .read_cache[[".stats"]] <- st
    return(.read_cache[[key]])
  }
  value <- read_tsv(path, ...)
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

load_llm_strata <- function(llm_annotations_path) {
  path <- file.path(llm_annotations_path, "LLM_NEPC_classifier_labels.tsv")
  if (!file.exists(path)) {
    message(sprintf("load_llm_strata: %s not found -- skipping LLM-strata plots", path))
    return(NULL)
  }
  strata <- cached_read_tsv(path, show_col_types = FALSE)
  required <- c("DFCI_MRN", "primary_label", "has_nepc", "has_avpc")
  missing <- setdiff(required, names(strata))
  if (length(missing) > 0) {
    stop(sprintf("%s is missing required columns: %s", path, paste(missing, collapse = ", ")))
  }
  biomarker_candidates <- c(
    "biomarker_genes",
    "reported_biomarkers",
    "biomarkers_reported",
    "reported_biomarker",
    "biomarkers",
    "biomarker"
  )
  candidate_index <- match(tolower(biomarker_candidates), tolower(names(strata)))
  candidate_index <- candidate_index[!is.na(candidate_index)]
  biomarker_col <- if (length(candidate_index) > 0) names(strata)[candidate_index[1]] else NULL

  raw_primary <- str_to_lower(str_trim(as.character(strata$primary_label)))
  if (is.null(biomarker_col) && any(raw_primary == "biomarker", na.rm = TRUE)) {
    stop(sprintf(
      paste0(
        "%s contains primary_label='biomarker' but no reported-biomarker column. ",
        "Expected one of: %s"
      ),
      path,
      paste(biomarker_candidates, collapse = ", ")
    ))
  }
  strata$reported_biomarkers <- if (is.null(biomarker_col)) {
    NA_character_
  } else {
    as.character(strata[[biomarker_col]])
  }
  strata <- strata %>%
    select(all_of(required), reported_biomarkers) %>%
    mutate(
      primary_label = str_to_lower(str_trim(as.character(primary_label))),
      primary_label = if_else(
        is.na(primary_label) |
          primary_label %in% c("", "nan", "na", "null", "none"),
        NA_character_,
        primary_label
      ),
      has_nepc = suppressWarnings(as.numeric(has_nepc)),
      has_avpc = suppressWarnings(as.numeric(has_avpc))
    )
  original_primary <- strata$primary_label
  strata$primary_label <- normalize_classifier_primary_labels(
    strata$primary_label,
    strata$reported_biomarkers,
    strata$has_nepc,
    strata$has_avpc
  )
  n_reclassified <- sum(
    !is.na(original_primary) &
      original_primary == "biomarker" &
      strata$primary_label != "biomarker",
    na.rm = TRUE
  )
  message(sprintf(
    paste0(
      "load_llm_strata: retained biomarker only for %s; ",
      "reclassified %s non-qualifying biomarker-primary row(s)"
    ),
    paste(BIOMARKER_PRIMARY_GENES, collapse = "/"),
    format(n_reclassified, big.mark = ",")
  ))
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
OTHER <- c("TSH")
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

cached_canonical_longitudinal <- function(path, labs = names(CATEGORY_MAP)) {
  if (!is.null(getOption("compass.figure_data_manifest")))
    return(figure_prepared_table(path, "canonical") %>% filter(LAB_GROUP %in% labs))
  resolved <- tryCatch(normalizePath(path, mustWork = TRUE),
                       error = function(e) path)
  key <- paste0("canonical-longitudinal:", resolved, ":", paste(sort(unique(labs)), collapse = "|"))
  if (exists(key, envir = .processed_read_cache, inherits = FALSE))
    return(.processed_read_cache[[key]])

  df <- cached_figure_longitudinal(path)
  needed <- c("LAB_NAME", "LAB_VALUE", "t_lab", "DFCI_MRN")
  missing <- setdiff(needed, names(df))
  if (length(missing))
    stop("canonical longitudinal data missing columns: ", paste(missing, collapse = ", "))

  # Classify distinct names, not every row of the measurement table.
  distinct_names <- unique(as.character(df$LAB_NAME))
  raw_names <- tolower(distinct_names)
  canonical_lookup <- setNames(names(CATEGORY_MAP), tolower(names(CATEGORY_MAP)))
  lab_group <- unname(canonical_lookup[raw_names])
  psa_alias <- !is.na(raw_names) & grepl("prostate specific ag", raw_names, fixed = TRUE)
  lab_group[psa_alias] <- "PSA"
  lab_group <- lab_group[match(df$LAB_NAME, distinct_names)]
  keep <- !is.na(lab_group) & lab_group %in% labs
  # Strip repeated patient metadata BEFORE row filtering, joins and binning.
  value <- df %>% select(any_of(c(needed, "PLATINUM", "t_platinum")))
  value <- value[keep, , drop = FALSE]
  value$LAB_GROUP <- lab_group[keep]
  value <- value %>%
    mutate(t_lab = suppressWarnings(as.numeric(t_lab)),
           t_rel = t_lab,
           LAB_VALUE = suppressWarnings(as.numeric(LAB_VALUE))) %>%
    drop_na(t_rel, LAB_VALUE)
  .processed_read_cache[[key]] <- value
  value
}

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
  missing_metrics <- c(auc = NA_real_, cindex = NA_real_, brier = NA_real_)
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
    brier = metric("test_integrated_brier")
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
  if (is.na(info$size) || isTRUE(info$isdir) || info$size < 12) return(FALSE)
  tryCatch({
    con <- file(path, "rb")
    on.exit(close(con))
    if (endsWith(tolower(path), ".png")) {
      header <- readBin(con, "raw", 8L)
      seek(con, info$size - 12, origin = "start")
      footer <- readBin(con, "raw", 12L)
      identical(header, as.raw(c(137, 80, 78, 71, 13, 10, 26, 10))) &&
        identical(footer, as.raw(c(0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130)))
    } else if (endsWith(tolower(path), ".pdf")) {
      header <- readBin(con, "raw", 5L)
      seek(con, max(0, info$size - 1024), origin = "start")
      footer <- readBin(con, "raw", 1024L)
      nonspace <- which(!footer %in% as.raw(c(9, 10, 13, 32)))
      last <- if (length(nonspace)) tail(nonspace, 1L) else 0L
      identical(header, charToRaw("%PDF-")) && last >= 5L &&
        identical(footer[seq.int(last - 4L, last)], charToRaw("%%EOF"))
    } else FALSE
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
  previous <- sub("/by_figure/(main|supplements)/", "/by_figure/", destination)
  if (identical(previous, destination) || !figure_file_complete(previous))
    return(invisible(FALSE))
  dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(".compass-copy-", tmpdir = dirname(destination),
                        fileext = paste0(".", tools::file_ext(destination)))
  on.exit(unlink(temporary))
  if (!file.copy(previous, temporary, overwrite = TRUE) || !figure_file_complete(temporary) ||
      !file.rename(temporary, destination)) stop("Could not reuse previous figure: ", previous)
  message("reused completed figure in new layout: ", destination)
  invisible(TRUE)
}

figure_output_tier <- function(cohort, endpoint, plot_stem) {
  supplemental <- cohort != "adt" || endpoint != "platinum" ||
    grepl("^(adt_intent_|adt_labels_|cohort_forest_|figure1s|pre_adt_coverage_)", plot_stem) ||
    grepl("nepc", plot_stem, ignore.case = TRUE) ||
    plot_stem %in% c("figure2v3_confusion_matrix", "figure2v3_metric_bar")
  if (supplemental) "supplements" else "main"
}

# Render the full COMPASS figure set for one cohort arm and one survival
# endpoint. Mirrors the body of the former figure notebook's per-cohort cells
# (Figures 1-7 + Table 1), so the R Markdown document can call it once per
# (cohort, endpoint) pair in the same R session.
generate_figures <- function(cohort, nepc_proj_path, fig_root,
                             endpoint = "platinum",
                             cohorts = SUPPORTED_COHORTS, show = FALSE,
                             llm_annotations_path = DEFAULT_LLM_ANNOTATIONS_PATH,
                             plot_non_androgen_distributions = FALSE,
                             plot_non_androgen_lab_figures = FALSE,
                             plot_gam_trajectories = TRUE,
                             plot_adt_intent_supplement = TRUE,
                             save_dpi = SAVE_DPI,
                             save_pdf = FALSE,
                             output_mode = "panels",
                             progress = NULL,
                             overwrite = FALSE,
                             metastatic_supplement = NULL,
                             cohort_forest_config = NULL) {
  if (!is.null(progress) && !is.function(progress))
    stop("progress must be NULL or a function(event, detail)")
  notify_progress <- function(event, detail) {
    if (!is.null(progress)) progress(event, detail)
    invisible(NULL)
  }
  # Rscript opens `Rplots.pdf` when any plot is drawn without an explicit device.
  # All intended outputs below use ggsave(), so route any incidental drawing to a
  # temporary null PDF device during non-interactive runs.
  if (!show) {
    previous_device_option <- getOption("device")
    options(device = function(...) grDevices::pdf(file = NULL, ...))
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
  if (!is.numeric(save_dpi) || length(save_dpi) != 1L || is.na(save_dpi) || save_dpi <= 0)
    stop("save_dpi must be one positive number")
  if (!is.logical(save_pdf) || length(save_pdf) != 1L || is.na(save_pdf))
    stop("save_pdf must be one non-missing logical value")
  if (!is.logical(overwrite) || length(overwrite) != 1L || is.na(overwrite))
    stop("overwrite must be one non-missing logical value")
  # Endpoint-independent figures describe the platinum-labelled cohort itself.
  # Emit them only on the platinum pass so a NEPC run does not create a second,
  # misleadingly endpoint-labelled copy.
  EMIT_ENDPOINT_INDEPENDENT <- identical(ENDPOINT, ENDPOINT_INDEPENDENT_FIGURES_ENDPOINT)
  if (!is.logical(plot_non_androgen_distributions) ||
      length(plot_non_androgen_distributions) != 1 ||
      is.na(plot_non_androgen_distributions))
    stop("plot_non_androgen_distributions must be one non-missing logical value")
  if (!is.logical(plot_non_androgen_lab_figures) ||
      length(plot_non_androgen_lab_figures) != 1 ||
      is.na(plot_non_androgen_lab_figures))
    stop("plot_non_androgen_lab_figures must be one non-missing logical value")
  if (!is.logical(plot_gam_trajectories) ||
      length(plot_gam_trajectories) != 1 ||
      is.na(plot_gam_trajectories))
    stop("plot_gam_trajectories must be one non-missing logical value")
  if (!is.logical(plot_adt_intent_supplement) ||
      length(plot_adt_intent_supplement) != 1 ||
      is.na(plot_adt_intent_supplement))
    stop("plot_adt_intent_supplement must be one non-missing logical value")
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

  # Canonical ADT platinum panels are main figures; NEPC, cohort variations,
  # and diagnostic families are supplements. Each subpanel keeps its own
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
  # Layout, every output:
  #   FIG_ROOT/by_figure/<main|supplements>/<group>/<trimmed-name>/<endpoint>__<subset>__<exclusion>.png
  # Per-lab panels keep their category/lab nesting for the same reason as
  # before -- ~40 labs x 4 strata would otherwise dump 160+ entries into one
  # directory -- and still bottom out at <= 12 files per leaf:
  #   FIG_ROOT/by_figure/labs/<category>/<lab>/longitudinal/<endpoint>__<subset>__<exclusion>.png
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

  # Every stem routes to .../by_figure/<main|supplements>/<group>/<trimmed-artifact-name>/, whose leaf holds one
  # file per cohort x endpoint cell. Uniform for numbered and supplemental
  # groups alike: the per-artifact level is what keeps sibling panels in a group
  # from sharing a leaf, and COHORT_LEAF (endpoint + subset + exclusion) is
  # what keeps the twelve cells within a stem from colliding.
  output_dir_for_stem <- function(plot_stem) {
    group <- figure_group(plot_stem)
    tier <- figure_output_tier(COHORT, ENDPOINT, plot_stem)
    file.path(FIG_ROOT, "by_figure", tier, group, artifact_name_for_stem(plot_stem, group))
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
  manual_annotations <- cached_read_csv(file.path(LLM_LABEL_PATH, "baca_lab_annotations.csv"),
                                        show_col_types = FALSE)
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
  # layout, then render only missing formats. Overwrite always regenerates.
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

    png_out <- file.path(output_dir, paste0(output_stem, ".png"))
    pdf_out <- file.path(output_dir, paste0(output_stem, ".pdf"))
    if (!overwrite) {
      reuse_previous_figure_layout(png_out)
      if (save_pdf) reuse_previous_figure_layout(pdf_out)
    }
    need_png <- overwrite || !figure_file_complete(png_out)
    need_pdf <- save_pdf && (overwrite || !figure_file_complete(pdf_out))
    if (!need_png && !need_pdf) {
      message("skipped completed figure: ", stem)
      notify_progress("panel_skipped", stem)
      # Do not force a lazy plot expression when its files already exist.
      return(invisible(NULL))
    }
    if (is.null(progress)) message("rendering ", stem, " ...")
    notify_progress("panel_start", stem)
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    write_format <- function(destination, pdf = FALSE) {
      # A failed/interrupted graphics device cannot leave a completed-looking
      # destination. Keep any previous figure until the replacement is ready.
      temporary <- tempfile(".compass-render-", tmpdir = output_dir,
                            fileext = if (pdf) ".pdf" else ".png")
      on.exit(unlink(temporary))
      if (pdf) {
        ggsave(temporary, plot = plot, width = width, height = height, units = "in",
               bg = "white", device = grDevices::cairo_pdf)
      } else if (HAS_RAGG) {
        ggsave(temporary, plot = plot, width = width, height = height, units = "in",
               dpi = save_dpi, bg = "white", device = ragg::agg_png)
      } else {
        ggsave(temporary, plot = plot, width = width, height = height, units = "in",
               dpi = save_dpi, bg = "white", type = "cairo")
      }
      if (!figure_file_complete(temporary))
        stop("Graphics device produced an incomplete file for ", destination)
      if (!file.rename(temporary, destination))
        stop("Could not publish figure ", destination)
      message("wrote ", destination)
    }
    plot <- prepare_figure_text(plot, width)
    if (need_png) write_format(png_out)
    if (need_pdf) write_format(pdf_out, pdf = TRUE)

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
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    out_base <- file.path(output_dir, COHORT_LEAF)
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

  # Retired: Figure 1 analysis-set-size supplement.
  if (FALSE) {
  ## ---- Figure 1 supplement -- final analysis-set sizes per arm x landmark ----
  # Every downstream model is fit on its own patient set: the univariate lab
  # screen keeps whoever has that feature at the landmark, the somatic/Gleason
  # screens are built on three different index dates, and the multivariate
  # arms report a held-out split. Reviewers ask for the final N and event count
  # behind each of those numbers, so this panel reads them back out of the same
  # result files the analysis figures plot and renders them as one table.
  #
  # It is deliberately read-only over already-written results: an arm whose
  # tree does not exist for this (cohort, endpoint) is reported as absent
  # rather than aborting the figure pass.
  SUPP_OUT_DIR <- fig_dir("figure1s_analysis_sets")

  # Univariate screens report one row per (feature x stat), each with its own
  # complete-case n. Summarise the arm by the maximum -- the fully observed
  # feature set, i.e. the landmark cohort actually carried into the screen --
  # and keep the minimum so the sparsest feature's support stays visible.
  summarise_univariate_set <- function(path, arm, landmark) {
    if (!file.exists(path))
      return(tibble(analysis = "Univariate", arm = arm, landmark_days = landmark,
                    n_patients = NA_integer_, n_events = NA_integer_,
                    detail = "results not found"))
    df <- read_csv(path, show_col_types = FALSE)
    if (!all(c("endpoint", "n_patients_used", "n_events_used") %in% names(df)))
      return(tibble(analysis = "Univariate", arm = arm, landmark_days = landmark,
                    n_patients = NA_integer_, n_events = NA_integer_,
                    detail = "unexpected schema"))
    df <- df %>% filter(tolower(as.character(endpoint)) == ENDPOINT)
    n_used <- suppressWarnings(as.numeric(df$n_patients_used))
    e_used <- suppressWarnings(as.numeric(df$n_events_used))
    keep <- is.finite(n_used) & is.finite(e_used)
    if (!any(keep))
      return(tibble(analysis = "Univariate", arm = arm, landmark_days = landmark,
                    n_patients = NA_integer_, n_events = NA_integer_,
                    detail = sprintf("no %s rows", ENDPOINT)))
    n_used <- n_used[keep]; e_used <- e_used[keep]
    # The max-n row is the one whose counts head the row; report its event
    # count rather than max(events) so N and events describe the same fit.
    top <- which.max(n_used)
    tibble(analysis = "Univariate", arm = arm, landmark_days = landmark,
           n_patients = as.integer(n_used[top]), n_events = as.integer(e_used[top]),
           detail = sprintf("%d feature%s; min n=%s", length(n_used),
                            ifelse(length(n_used) == 1, "", "s"),
                            format(as.integer(min(n_used)), big.mark = ",", trim = TRUE)))
  }

  # Multivariate arms write one metrics row per endpoint carrying the split
  # sizes, in the canonical schema every family now shares
  # (survival_common/metrics_schema.py) -- no per-family spellings to reconcile.
  summarise_multivariate_set <- function(path, arm, landmark) {
    absent <- function(detail)
      tibble(analysis = "Multivariate", arm = arm, landmark_days = landmark,
             n_patients = NA_integer_, n_events = NA_integer_, detail = detail)
    if (is.na(path) || !file.exists(path)) return(absent("results not found"))
    df <- read_csv(path, show_col_types = FALSE)
    if (!("endpoint" %in% names(df))) return(absent("no endpoint column"))
    row <- df %>%
      filter(tolower(as.character(.data[["endpoint"]])) == .env$ENDPOINT)
    if (nrow(row) == 0) return(absent(sprintf("no %s row", ENDPOINT)))
    pick <- function(column) {
      if (!(column %in% names(row))) return(NA_real_)
      suppressWarnings(as.numeric(row[[column]][1]))
    }
    n_train_val <- pick("n_train_val")
    n_test      <- pick("n_test")
    e_train_val <- pick("n_events_train_val")
    e_test      <- pick("n_events_test")
    # Total analysis set = train/val + held-out test. The train-side columns
    # are always present under the canonical schema, but a family that cannot
    # compute them writes NA -- keep reporting the held-out side alone there.
    if (is.finite(n_train_val)) {
      detail <- sprintf("train/val %s + test %s",
                        format(as.integer(n_train_val), big.mark = ","),
                        format(as.integer(n_test), big.mark = ","))
      n_total <- n_train_val + n_test
      e_total <- e_train_val + e_test
    } else {
      detail <- sprintf("held-out test only (%s)",
                        format(as.integer(n_test), big.mark = ","))
      n_total <- n_test
      e_total <- e_test
    }
    tibble(analysis = "Multivariate", arm = arm, landmark_days = landmark,
           n_patients = if (is.finite(n_total)) as.integer(n_total) else NA_integer_,
           n_events   = if (is.finite(e_total)) as.integer(e_total) else NA_integer_,
           detail     = detail)
  }

  SUPP_BASE <- BASE
  # Univariate arms: the lab screen at every landmark, plus the three
  # somatic/Gleason/PRS screens, which are ADT-only and landmark-0 only
  # (SOMATIC_GLEASON_LANDMARKS = (0,) in build_somatic_gleason_inputs.py).
  supp_univariate <- map_dfr(LANDMARKS, function(lm) {
    summarise_univariate_set(
      file.path(SUPP_BASE, "cox", sprintf("landmark_%s", lm), "both",
                "cox_agg_univariate_nobs_adjusted.csv"),
      "Labs", lm)
  })
  SUPP_SG_LABELS <- c(gleason    = "Gleason score",
                      sequencing = "Somatic alterations",
                      prs        = "Polygenic risk scores")
  # Arms evaluated at only a subset of LANDMARKS. Anything absent here is
  # assumed to be run at every landmark, so a blank cell means "not found".
  ARM_LANDMARKS <- setNames(rep(list(0L), length(SUPP_SG_LABELS)),
                            unname(SUPP_SG_LABELS))
  supp_univariate <- bind_rows(
    supp_univariate,
    map_dfr(names(SUPP_SG_LABELS), function(analysis) {
      summarise_univariate_set(
        file.path(SUPP_BASE, "cox_somatic_gleason", "landmark_0", analysis, "both",
                  "cox_agg_univariate_nobs_adjusted.csv"),
        unname(SUPP_SG_LABELS[[analysis]]), 0L)
    })
  )

  # Multivariate arms mirror Figure 4's series exactly, including the age-only
  # baselines (same patient set as their lab twin, but reported separately so a
  # split mismatch would be visible) and Dynamic-DeepHit's cause-only config.
  SUPP_MV_ARMS <- list(
    list(name = "Elastic-Net Cox",
         path = function(lm) file.path(SUPP_BASE, "cox", sprintf("landmark_%s", lm),
                                       "both", "cox_agg_multivariable_metrics.csv")),
    list(name = "Cox baseline (age)",
         path = function(lm) file.path(SUPP_BASE, "cox", sprintf("landmark_%s", lm),
                                       "baseline", "cox_agg_baseline_metrics.csv")),
    list(name = "XGBoost Survival",
         path = function(lm) file.path(SUPP_BASE, "xgboost", sprintf("landmark_%s", lm),
                                       "both", "landmark_xgboost_metrics.csv")),
    list(name = "XGBoost baseline (age)",
         path = function(lm) file.path(SUPP_BASE, "xgboost", sprintf("landmark_%s", lm),
                                       "baseline", "landmark_xgboost_baseline_metrics.csv")),
    list(name = "Dynamic-DeepHit",
         path = function(lm) {
           candidates <- file.path(
             SUPP_BASE, "multivariate_longitudinal",
             c("dynamic_deephit", "dynamic-deephit"), sprintf("landmark_%s", lm),
             ENDPOINT, sprintf("dynamic_deephit_metrics_%s.csv", ENDPOINT)
           )
           existing <- candidates[file.exists(candidates)]
           if (length(existing) == 0) NA_character_ else existing[1]
         })
  )
  supp_multivariate <- map_dfr(SUPP_MV_ARMS, function(arm) {
    map_dfr(LANDMARKS, function(lm) summarise_multivariate_set(arm$path(lm), arm$name, lm))
  })

  supp_sets <- bind_rows(supp_univariate, supp_multivariate) %>%
    mutate(
      analysis = factor(analysis, levels = c("Univariate", "Multivariate")),
      arm = factor(arm, levels = unique(c(
        "Labs", unname(SUPP_SG_LABELS), map_chr(SUPP_MV_ARMS, "name")
      ))),
      landmark_label = factor(
        sprintf("%s%d days", ifelse(landmark_days > 0, "+", ""), landmark_days),
        levels = sprintf("%s%d days", ifelse(sort(unique(landmark_days)) > 0, "+", ""),
                         sort(unique(landmark_days)))
      )
    ) %>%
    # Same restriction the panel applies: a landmark-0-only arm has no row at
    # +90/+180 to report, as opposed to a missing one.
    filter(map2_lgl(as.character(arm), landmark_days, function(a, lm) {
      evaluated <- ARM_LANDMARKS[[a]]
      is.null(evaluated) || lm %in% evaluated
    })) %>%
    arrange(analysis, arm, landmark_days)

  # Persist the numbers alongside the panel: the table is the deliverable a
  # methods section quotes, the plot is the at-a-glance version.
  supp_table <- supp_sets %>%
    transmute(Analysis = as.character(analysis),
              Arm = as.character(arm),
              Landmark = as.character(landmark_label),
              `N patients` = ifelse(is.na(n_patients), "n/a",
                                    format(n_patients, big.mark = ",", trim = TRUE)),
              `N events` = ifelse(is.na(n_events), "n/a",
                                  format(n_events, big.mark = ",", trim = TRUE)),
              `Event rate` = ifelse(is.na(n_patients) | is.na(n_events) | n_patients == 0,
                                    "n/a", sprintf("%.1f%%", 100 * n_events / n_patients)),
              Notes = detail)
  for (p in write_table1(supp_table,
                         file.path(SUPP_OUT_DIR,
                                   sprintf("figure1s_analysis_sets_%s", ENDPOINT))))
    message(sprintf("wrote %s", p))
  if (show) print(supp_table)

  # Heat-table: one tile per (arm x landmark), labelled "N (events)". A tile is
  # grey when that arm has no results for this (cohort, endpoint, landmark),
  # which is itself the informative outcome -- it says the run never produced
  # that cell rather than that it produced a zero.
  render_analysis_set_panel <- function(d, title) {
    if (nrow(d) == 0)
      return(ggplot() + annotate("text", x = 0, y = 0, label = "(no results found)") +
               theme_void() + labs(title = title))
    # An arm evaluated at only some landmarks (the somatic/Gleason/PRS screens
    # are landmark-0 only) should leave the other columns blank rather than
    # drawing a tile that implies a run was attempted and lost.
    d <- d %>%
      filter(map2_lgl(as.character(arm), landmark_days, function(a, lm) {
        evaluated <- ARM_LANDMARKS[[a]]
        is.null(evaluated) || lm %in% evaluated
      })) %>%
      mutate(arm = factor(as.character(arm), levels = rev(levels(droplevels(arm)))))
    ggplot(d, aes(landmark_label, arm)) +
      geom_tile(aes(fill = n_patients), color = "white", linewidth = 1) +
      geom_text(aes(label = ifelse(is.na(n_patients), "\u2014",
                                   sprintf("%s\n(%s events)",
                                           format(n_patients, big.mark = ",", trim = TRUE),
                                           ifelse(is.na(n_events), "?",
                                                  format(n_events, big.mark = ",", trim = TRUE))))),
                size = 3.1, lineheight = 0.95, color = "#0b0b0b") +
      scale_fill_gradient(low = "#eaf1fb", high = "#9dbbe6",
                          na.value = "#e6e6e6", guide = "none") +
      labs(x = NULL, y = NULL, title = title) +
      theme_fig() +
      theme(panel.grid = element_blank(),
            plot.title = element_text(face = "bold", size = 11))
  }

  supp_uni_panel <- render_analysis_set_panel(
    supp_sets %>% filter(analysis == "Univariate"),
    "Panel A — univariate screens")
  supp_mv_panel <- render_analysis_set_panel(
    supp_sets %>% filter(analysis == "Multivariate"),
    "Panel B — multivariate models (train/val + held-out test)")
  save_fig(supp_uni_panel, SUPP_OUT_DIR,
           sprintf("figure1s_analysis_sets_univariate_%s", ENDPOINT), 7.0, 3.6)
  save_fig(supp_mv_panel, SUPP_OUT_DIR,
           sprintf("figure1s_analysis_sets_multivariate_%s", ENDPOINT), 7.0, 4.2)

  if (show) print(supp_uni_panel)
  if (show) print(supp_mv_panel)
  }

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
  ## ---- Figure 2 v3 -- classifier labels over every ADT-exposed patient ----
  # The only Figure 2 variant retained. Earlier variants (v0: unrestricted
  # LLM_v3_labels.tsv; v1: those labels narrowed to the ADT landmark-0
  # prediction cohort; v2: the classifier labels on that same prediction
  # cohort) have been removed. v3 uses LLM_NEPC_classifier_labels.tsv over
  # every MRN with ADT_EXPOSED == 1 in the ICD prostate workflow flags -- the
  # "ADT entry requirement" step of the Figure 1 CONSORT, before any
  # downstream prediction-cohort filtering.
  if (is.null(llm_classifier_labels)) {
    message("Figure 2 v3: llm_classifier_labels unavailable -- skipping.")
  } else {
    OUT_DIR_V3 <- fig_dir("figure2v3_llm")

    drop_cols <- function(df, cols) df %>% select(-any_of(cols))

    # Panels B/C restored from 3c4efc4 (before their removal in fec33f4).
    # Fixed class order groups the two aggressive classes together for readability.
    CLASS_ORDER <- c("conventional", "avpc", "nepc", "biomarker")
    CLASS_LABELS <- c(conventional = "Conventional", avpc = "AVPC", nepc = "NEPC",
                      biomarker = "Biomarker")

    count_labels <- function(df) {
      df %>% count(primary_label, name = "count") %>%
        mutate(frac = count / sum(count))
    }

    # Annotated 2x2 confusion matrix, LLM (rows) vs manual truth (cols).
    render_confusion_panel <- function(metrics,
                                       truth_label = "NEPC",
                                       pred_label = "NEPC",
                                       title = "Panel A1 — confusion matrix") {
      cm <- tibble(
        truth = factor(c(paste0("Non-", truth_label), truth_label,
                         paste0("Non-", truth_label), truth_label),
                       levels = c(paste0("Non-", truth_label), truth_label)),
        pred  = factor(c(paste0("Non-", pred_label), paste0("Non-", pred_label),
                         pred_label, pred_label),
                       levels = c(pred_label, paste0("Non-", pred_label))),
        n     = c(metrics$TN, metrics$FP, metrics$FN, metrics$TP)
      )
      thresh <- max(cm$n) / 2
      ggplot(cm, aes(truth, pred, fill = n)) +
        geom_tile(color = "white", linewidth = 1) +
        geom_text(aes(label = format(n, big.mark = ","),
                      color = n > thresh), size = 5, fontface = "bold") +
        scale_fill_gradient(low = "#eaf1fb", high = COLOR_PLATINUM_POS, guide = "none") +
        scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#0b0b0b"), guide = "none") +
        labs(x = "Manual annotation (truth)", y = "LLM label (prediction)", title = title) +
        theme_fig() +
        theme(panel.grid = element_blank(),
              plot.title = element_text(face = "bold", size = 11))
    }

    # Compact metric bar: Accuracy, Precision, Recall, Specificity (Specificity == TNR).
    render_metric_bar_panel <- function(metrics) {
      d <- tibble(
        metric = factor(c("Accuracy","Precision","Recall","Specificity"),
                        levels = c("Accuracy","Precision","Recall","Specificity")),
        value  = c(metrics$Accuracy, metrics$Precision, metrics$Recall, metrics$TNR)
      )
      ggplot(d, aes(metric, value)) +
        geom_col(fill = COLOR_PLATINUM_POS, width = 0.6) +
        geom_text(aes(label = sprintf("%.2f", value)), vjust = -0.4, size = 3.6, color = "#0b0b0b") +
        coord_cartesian(ylim = c(0, 1.0)) +
        labs(x = NULL, y = "Metric value", title = "Panel A2 — classifier metrics") +
        theme_fig() +
        theme(plot.title = element_text(face = "bold", size = 11))
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
        labs(x = NULL, y = "Fraction within platinum group", title = str_wrap(title, 65)) +
        theme_fig() +
        theme(plot.title = element_text(face = "bold", size = 11),
              axis.title.x = element_blank(),
              axis.title.y = element_text(size = 16),
              axis.text  = element_text(size = 14),
              legend.position = "bottom", legend.justification = "center")
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

    adt_exposed_mrns_v3 <- unique(as.character(
      icd_prostate_mrn_flags[[ID_COL]][icd_prostate_mrn_flags$ADT_EXPOSED == 1]))
    v3_labels_all <- llm_classifier_labels %>%
      filter(as.character(DFCI_MRN) %in% adt_exposed_mrns_v3)
    # Not a stopifnot: a shortfall here means the prediction cohort and the ICD
    # flag table disagree, which is worth surfacing but should not abort the pass.
    cohort_mrns <- unique(as.character(patient_df[[ID_COL]]))
    stopifnot(base_landmark == 0L, length(cohort_mrns) == nrow(patient_df))
    n_pred_outside_v3 <- sum(!cohort_mrns %in% adt_exposed_mrns_v3)
    if (n_pred_outside_v3 > 0)
      message(sprintf(
        "Figure 2 v3: %s prediction-cohort MRN(s) are absent from ADT_EXPOSED == 1 in %s",
        format(n_pred_outside_v3, big.mark = ","), ICD_PROSTATE_MRN_FLAGS_CSV))
    message(sprintf(
      "Figure 2 v3 ADT-exposed universe: %s total patients (%s in the landmark-0 prediction cohort); %s labeled/evaluable (%s source labels total)",
      format(length(adt_exposed_mrns_v3), big.mark = ","),
      format(length(cohort_mrns), big.mark = ","),
      format(nrow(v3_labels_all), big.mark = ","),
      format(nrow(llm_classifier_labels), big.mark = ",")
    ))

    ## Panel A: binary NEPC validation uses has_nepc directly. Panels B/C below
    ## use primary_label for the subtype landscape and aggressive-variant group.
    merged_v3 <- manual_annotations %>%
      drop_cols(c("pathology_details", "manual_platinum_reason")) %>%
      inner_join(v3_labels_all, by = "DFCI_MRN") %>%
      mutate(
        manual_NEPC = simplified_manual_platinum_reason %in% c("nepc", "squamous_transformation")
      )
    cat(sprintf("figure2v3 merged_results: %s rows, %s manual-NEPC positive\n",
                format(nrow(merged_v3), big.mark = ","),
                format(sum(merged_v3$manual_NEPC), big.mark = ",")))
    metrics_v3 <- binary_metrics(merged_v3$manual_NEPC, merged_v3$has_nepc)
    n_total_v3 <- metrics_v3$N
    n_nepc_manual_v3 <- metrics_v3$TP + metrics_v3$FN
    caption_a_v3 <- sprintf("All ADT-exposed patients (N=%s total, no prediction-cohort restriction); %s chart-reviewed/labeled patients; %s manual-NEPC positive (LLM_NEPC_classifier_labels.tsv).",
                            format(length(adt_exposed_mrns_v3), big.mark = ","),
                            format(n_total_v3, big.mark = ","), format(n_nepc_manual_v3, big.mark = ","))
    pA1_v3 <- render_confusion_panel(
      metrics_v3, "NEPC", "NEPC", "NEPC classifier agreement"
    ) + labs(caption = caption_a_v3) +
      theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK))
    pA2_v3 <- render_metric_bar_panel(metrics_v3) +
      labs(title = "NEPC classifier metrics", caption = caption_a_v3) +
      theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK))
    save_fig(pA1_v3, OUT_DIR_V3, "figure2v3_confusion_matrix", 5.8, 5.4)
    save_fig(pA2_v3, OUT_DIR_V3, "figure2v3_metric_bar", 6.2, 5.0)
    if (show) print(pA1_v3)
    if (show) print(pA2_v3)

    ## Panel B -- subtype landscape by platinum status (4-class primary_label).
    # load_llm_strata coerces primary_label values outside the four modeled
    # classes to NA, and those rows would draw an "NA" bar.
    v3_labels_classified <- v3_labels_all %>% filter(!is.na(primary_label))
    n_unclassified_v3 <- nrow(v3_labels_all) - nrow(v3_labels_classified)
    if (n_unclassified_v3 > 0) {
      message(sprintf(
        "figure2v3 Panel B: dropped %s row(s) without one of the four primary_label classes",
        format(n_unclassified_v3, big.mark = ",")
      ))
    }
    platinum_positive_v3 <- v3_labels_classified %>% filter(is_platinum) %>% count_labels() %>%
      mutate(platinum_status = "positive")
    platinum_negative_v3 <- v3_labels_classified %>% filter(!is_platinum) %>% count_labels() %>%
      mutate(platinum_status = "negative")
    label_distributions_v3 <- bind_rows(platinum_positive_v3, platinum_negative_v3)
    n_pos <- sum(platinum_positive_v3$count)
    n_neg <- sum(platinum_negative_v3$count)
    caption_b_v3 <- sprintf("All ADT-exposed patients; %s classified of %s total patients%s; platinum+ n=%s, platinum- n=%s.",
                            format(nrow(v3_labels_classified), big.mark = ","),
                            format(length(adt_exposed_mrns_v3), big.mark = ","),
                            if (n_unclassified_v3 > 0)
                              sprintf(" (%s labeled row(s) outside the four classes excluded)",
                                      format(n_unclassified_v3, big.mark = ","))
                            else "",
                            format(n_pos, big.mark = ","), format(n_neg, big.mark = ","))
    pB_v3 <- render_landscape_panel(
        label_distributions_v3, n_pos, n_neg,
        "Subtype landscape by platinum status") +
      labs(caption = str_wrap(caption_b_v3, 85)) +
      theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK, hjust = 0.5))
    save_fig(pB_v3, OUT_DIR_V3, "figure2v3_subtype_landscape", 6.5, 8)

    ## Panel C -- aggressive (avpc+nepc) vs conventional platinum enrichment.
    enrichment_v3 <- compute_enrichment(v3_labels_all)
    cat(sprintf("figure2v3 enrichment: OR = %.2f, Fisher p = %.3g\n",
                enrichment_v3$OR, enrichment_v3$p_value))
    caption_c_v3 <- sprintf("All ADT-exposed patients; excludes biomarker/unclassified labels (%s rows). Error bars are 95%% Wilson intervals. OR=%.1f, one-sided Fisher p=%.1e.",
                            format(enrichment_v3$n_excluded, big.mark = ","),
                            enrichment_v3$OR, enrichment_v3$p_value)
    pC_v3 <- render_enrichment_panel(enrichment_v3) + labs(caption = str_wrap(caption_c_v3, 58)) +
      theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK, hjust = 0.5))
    save_fig(pC_v3, OUT_DIR_V3, "figure2v3_enrichment", 6.0, 5.8)
    if (show) print(pB_v3)
    if (show) print(pC_v3)
  }
  }

  # ----------------------------- labeling knobs ---------------------------
  TOP_K_PER_PANEL <- 4
  ALWAYS_LABEL    <- c("Hemoglobin", "Albumin", "Alkaline phosphatase")
  PANEL_XLIM      <- c(-1.5, 1.5)
  Y_MAX_CAP       <- 30   # -log10(p) ceiling; values above are drawn at the cap as triangles

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

  plot_volcano_panel <- function(sub, title) {
    sub <- sub %>%
      filter(!lab_name %in% DROP) %>%
      mutate(
        .point_id  = row_number(),
        category  = vapply(lab_name, assign_category, character(1)),
        neglog10p = -log10(pmax(p_value, 1e-300)),
        sig       = q_value < 0.05,
        capped    = neglog10p > Y_MAX_CAP,
        y         = pmin(neglog10p, Y_MAX_CAP),
        label     = sprintf("%s (%s)", lab_name, feature_stat)
      )
    ns  <- sub %>% filter(!sig)
    sigd <- sub %>% filter(sig) %>%
      mutate(category = factor(category, levels = DRAW_ORDER),
             is_hero  = category == "Androgen axis")
    y_max <- if (nrow(sub)) max(sub$y) else 5
    q_y <- q_threshold_neglog10p(sub)
    lab_df <- labels_for_panel(sub, TOP_K_PER_PANEL, ALWAYS_LABEL)
    # Keep every point in the repel calculation, including unlabeled points, so
    # text cannot settle on top of a nearby observation.
    sub <- sub %>% mutate(repel_label = ifelse(.point_id %in% lab_df$.point_id, label, ""))

    n_tested <- nrow(sub); n_sig <- sum(sub$sig)
    breakdown <- sub %>% filter(sig) %>% count(category)
    short <- c("Androgen axis"="Androgen","CBC"="CBC","LFT"="LFT","CMP"="CMP",
               "Vitals"="Vitals")
    bd_str <- paste(vapply(setdiff(LEGEND_ORDER, "Other"), function(c) {
      n <- breakdown$n[match(c, breakdown$category)]; if (is.na(n)) n <- 0
      sprintf("%s %d", short[[c]], n)
    }, character(1)), collapse = "  ")
    footer <- sprintf("%d / %d q<0.05   \u00b7   %s", n_sig, n_tested, bd_str)

    p <- ggplot() +
      geom_vline(xintercept = 0, color = "grey", linewidth = 0.7) +
      geom_vline(xintercept = c(-0.5, 0.5), color = "grey", linetype = "dashed",
                 linewidth = 0.6, alpha = 0.7) +
      { if (!is.na(q_y)) geom_hline(yintercept = q_y, color = "black",
                                    linetype = "dotted", linewidth = 0.9) } +
      geom_point(data = ns, aes(coef_feature, y), size = 1.6 * 1.5, color = NS_COLOR, alpha = 0.7) +
      geom_point(data = sigd %>% filter(!capped),
                 aes(coef_feature, y, color = category, size = is_hero),
                 shape = 21, fill = NA, stroke = 0.9, show.legend = FALSE) +
      geom_point(data = sigd %>% filter(!capped),
                 aes(coef_feature, y, fill = category, size = is_hero),
                 shape = 21, color = "white", stroke = 0.6, alpha = 0.92) +
      geom_point(data = sigd %>% filter(capped),
                 aes(coef_feature, y, fill = category), shape = 24,
                 size = 3.4 * 1.5, color = "white", stroke = 0.6, alpha = 0.92, show.legend = FALSE) +
      ggrepel::geom_text_repel(
        data = sub,
        aes(coef_feature, pmin(neglog10p, Y_MAX_CAP), label = repel_label, color = category),
        size = 3.2, fontface = "plain", segment.color = "#95a5a6", segment.size = 0.3,
        max.overlaps = Inf, min.segment.length = 0, box.padding = 0.55,
        point.padding = 0.8, point.size = 4, force = 2, max.time = 4, max.iter = 100000, seed = 0,
        nudge_x = ifelse(sub$coef_feature < 0, -0.35, 0.35), nudge_y = 0.35,
        show.legend = FALSE) +
      scale_color_manual(values = CATEGORY_COLORS, breaks = LEGEND_ORDER, name = NULL) +
      scale_fill_manual(values = CATEGORY_COLORS, breaks = LEGEND_ORDER, name = NULL) +
      guides(color = "none", fill = guide_legend(nrow = 2, override.aes = list(shape = 21, size = 3, alpha = 1))) +
      scale_size_manual(values = c(`TRUE` = 3.2 * 1.5, `FALSE` = 2.1 * 1.5), guide = "none") +
      coord_cartesian(xlim = range(c(PANEL_XLIM, sub$coef_feature), finite = TRUE) * 1.08,
                      ylim = c(-0.2, max(y_max * 1.20, 5))) +
      labs(x = "Cox log HR per SD", y = expression(-log[10](p)), title = title,
           caption = str_wrap(footer, 88)) +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 14),
            plot.caption = element_text(size = 9, color = "#5d6d7e", family = "sans",
                                        hjust = 0, lineheight = 1.05),
            axis.title = element_text(size = 12),
            axis.text  = element_text(size = 10),
            legend.text = element_text(size = 10),
            legend.position = "bottom", legend.justification = "center")
    p
  }

  # Significance-only coloring for the univariate volcano: no lab-category
  # color, just significant (q<0.05) vs not, with optional caller-supplied
  # labs picked out via a distinct highlight color/outline. `highlight_labs`
  # matches on `lab_name` (e.g. "Hemoglobin"), independent of feature_stat.
  SIG_COLOR   <- COLOR_PLATINUM_POS   # blue, reused from the platinum-status palette
  HIGHLIGHT_COLOR <- "#8e1c2b"        # reused from CATEGORY_COLORS' "Androgen axis" red

  plot_volcano_panel_by_significance <- function(sub, title, highlight_labs = character(0)) {
    sub <- sub %>%
      filter(!lab_name %in% DROP) %>%
      mutate(
        .point_id  = row_number(),
        neglog10p = -log10(pmax(p_value, 1e-300)),
        sig       = q_value < 0.05,
        capped    = neglog10p > Y_MAX_CAP,
        y         = pmin(neglog10p, Y_MAX_CAP),
        label     = sprintf("%s (%s)", lab_name, feature_stat),
        highlighted = lab_name %in% highlight_labs,
        point_group = factor(
          ifelse(highlighted, "Highlighted",
                 ifelse(sig, "Significant (q<0.05)", "Not significant")),
          levels = c("Not significant", "Significant (q<0.05)", "Highlighted"))
      )
    ns   <- sub %>% filter(point_group == "Not significant")
    sigd <- sub %>% filter(point_group == "Significant (q<0.05)")
    hid  <- sub %>% filter(point_group == "Highlighted")
    y_max <- if (nrow(sub)) max(sub$y) else 5
    q_y <- q_threshold_neglog10p(sub)
    lab_df <- labels_for_panel(sub %>% mutate(category = "Other"), TOP_K_PER_PANEL, ALWAYS_LABEL)
    # Always label highlighted points in addition to the usual auto-selection.
    lab_df <- bind_rows(lab_df, sub %>% filter(highlighted)) %>%
      distinct(lab_name, feature_stat, .keep_all = TRUE)
    sub <- sub %>% mutate(repel_label = ifelse(.point_id %in% lab_df$.point_id, label, ""))

    n_tested <- nrow(sub); n_sig <- sum(sub$sig)
    footer <- sprintf("%d / %d q<0.05", n_sig, n_tested)
    if (length(highlight_labs)) footer <- sprintf("%s   ·   %d highlighted", footer, sum(sub$highlighted))

    point_colors <- c(`Not significant` = NS_COLOR, `Significant (q<0.05)` = SIG_COLOR,
                      `Highlighted` = HIGHLIGHT_COLOR)

    p <- ggplot() +
      geom_vline(xintercept = 0, color = "grey", linewidth = 0.7) +
      geom_vline(xintercept = c(-0.5, 0.5), color = "grey", linetype = "dashed",
                 linewidth = 0.6, alpha = 0.7) +
      { if (!is.na(q_y)) geom_hline(yintercept = q_y, color = "black",
                                    linetype = "dotted", linewidth = 0.9) } +
      geom_point(data = ns, aes(coef_feature, y), size = 1.6 * 1.5, color = NS_COLOR, alpha = 0.7) +
      geom_point(data = sigd %>% filter(!capped),
                 aes(coef_feature, y), shape = 21, fill = SIG_COLOR, color = "white",
                 size = 2.1 * 1.5, stroke = 0.6, alpha = 0.92) +
      geom_point(data = sigd %>% filter(capped),
                 aes(coef_feature, y), shape = 24, fill = SIG_COLOR, color = "white",
                 size = 3.4 * 1.5, stroke = 0.6, alpha = 0.92) +
      geom_point(data = hid %>% filter(!capped),
                 aes(coef_feature, y), shape = 21, fill = HIGHLIGHT_COLOR, color = "white",
                 size = 3.2 * 1.5, stroke = 0.9, alpha = 0.98) +
      geom_point(data = hid %>% filter(capped),
                 aes(coef_feature, y), shape = 24, fill = HIGHLIGHT_COLOR, color = "white",
                 size = 4.2 * 1.5, stroke = 0.9, alpha = 0.98) +
      ggrepel::geom_text_repel(
        data = sub,
        aes(coef_feature, pmin(neglog10p, Y_MAX_CAP), label = repel_label, color = point_group),
        size = 3.2, fontface = "plain", segment.color = "#95a5a6", segment.size = 0.3,
        max.overlaps = Inf, min.segment.length = 0, box.padding = 0.55,
        point.padding = 0.8, point.size = 4, force = 2, max.time = 4, max.iter = 100000, seed = 0,
        nudge_x = ifelse(sub$coef_feature < 0, -0.35, 0.35), nudge_y = 0.35,
        show.legend = FALSE) +
      scale_color_manual(values = point_colors, breaks = names(point_colors), name = NULL,
                         guide = guide_legend(override.aes = list(size = 4.5, alpha = 1))) +
      coord_cartesian(xlim = range(c(PANEL_XLIM, sub$coef_feature), finite = TRUE) * 1.08,
                      ylim = c(-0.2, max(y_max * 1.20, 5))) +
      labs(x = "Cox log HR per SD", y = expression(-log[10](p)), title = title,
           caption = footer) +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 14),
            plot.caption = element_text(size = 9, color = "#5d6d7e", family = "sans",
                                        hjust = 0),
            axis.title = element_text(size = 12),
            axis.text  = element_text(size = 10),
            legend.text = element_text(size = 10),
            legend.position = "bottom", legend.justification = "center")
    p
  }

  OUT_DIR <- fig_dir("figure3_univariate")
  notify_progress("stage", "Figure 3: lab associations")

  load_uni <- function(landmark) {
    path <- file.path(BASE, "cox", sprintf("landmark_%s", landmark), "both",
                      "cox_agg_univariate_nobs_adjusted.csv")
    read_csv(path, show_col_types = FALSE) %>% mutate(landmark_days = landmark)
  }

  uni <- map_dfr(LANDMARKS, load_uni) %>%
    filter(tolower(as.character(endpoint)) == ENDPOINT) %>%
    drop_na(coef_feature, p_value, q_value)
  cat(sprintf("%d (lab x stat) rows across landmarks %s\n",
              nrow(uni), paste(sort(unique(uni$landmark_days)), collapse = ", ")))

  # Filter unstable Cox estimates: |log HR| > 4 or CI spans > 2 orders of magnitude.
  COEF_CAP <- 4.0; CI_RATIO_CAP <- 100
  ci_ratio <- uni$ci_upper / uni$ci_lower
  mask <- abs(uni$coef_feature) <= COEF_CAP & ci_ratio < CI_RATIO_CAP
  cat(sprintf("dropping %d / %d unstable rows\n", sum(!mask), nrow(uni)))
  uni <- uni[mask, ]
  cat(sprintf("%d rows remaining\n", nrow(uni)))

  # One solo volcano panel per landmark in this cohort's LANDMARKS.
  panels <- lapply(LANDMARKS, function(lm) {
    list(lm, ifelse(lm > 0, sprintf("+%d days", lm), sprintf("%d days", lm)))
  })
  for (pn in panels) {
    lm <- pn[[1]]; title <- pn[[2]]
    sub <- uni %>% filter(landmark_days == lm)
    if (nrow(sub) == 0) {
      p <- ggplot() + annotate("text", x = 0, y = 0,
                               label = sprintf("(no data for landmark = %dd)", lm),
                               color = "#7f8c8d") + theme_void()
    } else {
      p <- plot_volcano_panel(sub, title)
    }
    save_fig(p, OUT_DIR, sprintf("figure3_univariate_%s_landmark%d", ENDPOINT, lm),
             width = 9, height = 7)
    if (show) print(p)
  }

  ## ---- Figure 3b -- somatic and Gleason-score univariate associations ----
  notify_progress("stage", "Figure 3b: somatic and Gleason-score associations")
  # Separate plots from the lab volcanoes above: these come from
  # build_somatic_gleason_inputs.py / run_somatic_gleason_univariate(), which
  # build three DIFFERENT cohorts with three different index dates
  # (sequencing = specimen collection, gleason = the Gleason score date nearest
  # ADT start, prs = ADT start). Pooling them into one panel would mix index
  # dates and patient sets, so each index analysis gets its own figure.
  #
  # Only ADT is built (the index dates are defined relative to ADT start) and
  # only at landmark 0 (SOMATIC_GLEASON_LANDMARKS = (0,)). A missing tree is
  # reported and skipped rather than aborting the endpoint's figure pass.
  OUT_DIR <- fig_dir("figure3b_somatic_gleason")

  SG_LANDMARK  <- 0L
  SG_ANALYSES  <- c("sequencing", "gleason")
  SG_LABELS <- c(
    sequencing = "Somatic alterations",
    gleason = "Gleason score"
  )
  # Each analysis' prediction time origin, from build_somatic_gleason_inputs.py.
  SG_ORIGINS <- c(
    sequencing = "sequencing specimen collection date",
    gleason = "Gleason score date nearest ADT initiation"
  )
  SG_TOP_N <- 25   # forest plots show at most this many features, ranked by p

  sg_path <- function(analysis, landmark = SG_LANDMARK) {
    file.path(BASE, "cox_somatic_gleason", sprintf("landmark_%s", landmark),
              analysis, "both", "cox_agg_univariate_nobs_adjusted.csv")
  }

  # Same stability filter as the lab volcanoes: drop |log HR| > 4 or a CI
  # spanning more than two orders of magnitude. Binary somatic indicators at
  # low prevalence are the usual source of both.
  load_sg <- function(analysis) {
    path <- sg_path(analysis)
    if (!file.exists(path)) {
      message(sprintf("Figure 3b %s: %s not found -- skipping.", analysis, path))
      return(NULL)
    }
    df <- read_csv(path, show_col_types = FALSE) %>%
      filter(tolower(as.character(endpoint)) == ENDPOINT) %>%
      drop_na(coef_feature, p_value, q_value)
    if (nrow(df) == 0) {
      message(sprintf("Figure 3b %s: no %s rows in %s -- skipping.",
                      analysis, ENDPOINT, path))
      return(NULL)
    }
    n_before <- nrow(df)
    keep <- abs(df$coef_feature) <= COEF_CAP & (df$ci_upper / df$ci_lower) < CI_RATIO_CAP
    df <- df[keep, ]
    cat(sprintf("Figure 3b %s: %d features, dropped %d unstable, %d remaining\n",
                analysis, n_before, sum(!keep), nrow(df)))
    if (nrow(df) == 0) return(NULL)
    df
  }

  # Forest of log HR per SD with 95% CI, ranked by p-value. Used wherever a
  # volcano would be uninformative -- notably Gleason, which contributes a
  # single continuous feature (GLEASON_SCORE) and so would plot as one point.
  plot_sg_forest <- function(sub, title, subtitle, top_n = SG_TOP_N) {
    d <- sub %>%
      arrange(p_value) %>%
      head(top_n) %>%
      mutate(
        sig   = q_value < 0.05,
        # ci_lower/ci_upper are hazard-ratio bounds; the x axis is on the log
        # HR scale to match coef_feature (= log HR per SD).
        lo    = log(ci_lower),
        hi    = log(ci_upper),
        # Somatic/Gleason/PRS rows carry an empty feature_stat, which readr
        # parses as an all-NA logical column when no row has a value; coerce
        # before testing so the label does not collapse to NA.
        .stat = ifelse(is.na(feature_stat), "", as.character(feature_stat)),
        label = ifelse(nzchar(.stat) & .stat != "value",
                       sprintf("%s (%s)", lab_name, .stat),
                       as.character(lab_name))
      ) %>%
      mutate(label = factor(label, levels = rev(unique(label))))
    n_sig <- sum(d$sig)
    x_span <- range(c(d$lo, d$hi, 0), na.rm = TRUE, finite = TRUE)
    ggplot(d, aes(coef_feature, label, color = sig)) +
      geom_vline(xintercept = 0, color = "grey", linewidth = 0.7) +
      geom_errorbar(aes(xmin = lo, xmax = hi), orientation = "y",
                    width = 0.28, linewidth = 0.7) +
      geom_point(size = 2.6) +
      scale_color_manual(
        values = c(`TRUE` = SIG_COLOR, `FALSE` = "#9a9890"),
        breaks = c(TRUE, FALSE),
        labels = c("q<0.05", "n.s."), name = NULL,
        guide = guide_legend(override.aes = list(size = 3.2))) +
      coord_cartesian(xlim = x_span * 1.05) +
      labs(x = "Cox log HR per SD (95% CI)", y = NULL,
           title = title, subtitle = subtitle,
           caption = sprintf("%d / %d features q<0.05%s", n_sig, nrow(sub),
                             if (nrow(sub) > nrow(d))
                               sprintf("; showing the %d smallest p-values", nrow(d))
                             else "")) +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 14),
            plot.subtitle = element_text(size = 10, color = COLOR_NEUTRAL_INK),
            plot.caption = element_text(size = 9, color = COLOR_NEUTRAL_INK, hjust = 0),
            axis.text.y = element_text(size = 10),
            legend.position = "top")
  }

  # Volcano for the analyses that test many features at once (somatic binary
  # indicators, PRS). Reuses the significance-only styling of the lab volcano
  # -- lab CATEGORY_COLORS do not apply to genomic features. Axis limits are
  # data-driven rather than the labs' fixed PANEL_XLIM, because somatic
  # indicators and PRS live on different effect-size scales.
  plot_sg_volcano <- function(sub, title, subtitle, top_k = 8) {
    d <- sub %>%
      mutate(
        .point_id = row_number(),
        neglog10p = -log10(pmax(p_value, 1e-300)),
        sig       = q_value < 0.05,
        capped    = neglog10p > Y_MAX_CAP,
        y         = pmin(neglog10p, Y_MAX_CAP),
        label     = as.character(lab_name)
      )
    # Label the strongest hits: every significant feature when there are few,
    # otherwise the top_k by p-value.
    lab_ids <- d %>% filter(sig) %>% arrange(p_value) %>% head(top_k) %>% pull(.point_id)
    d <- d %>% mutate(repel_label = ifelse(.point_id %in% lab_ids, label, ""))

    q_y <- q_threshold_neglog10p(d)
    x_span <- max(abs(d$coef_feature), na.rm = TRUE) * 1.1
    n_sig <- sum(d$sig)

    ggplot(d, aes(coef_feature, y)) +
      geom_vline(xintercept = 0, color = "grey", linewidth = 0.7) +
      { if (!is.na(q_y)) geom_hline(yintercept = q_y, color = "black",
                                    linetype = "dotted", linewidth = 0.9) } +
      geom_point(data = ~ dplyr::filter(.x, !sig), color = NS_COLOR, size = 2.2, alpha = 0.7) +
      geom_point(data = ~ dplyr::filter(.x, sig & !capped), shape = 21,
                 fill = SIG_COLOR, color = "white", size = 3.0, stroke = 0.6, alpha = 0.92) +
      geom_point(data = ~ dplyr::filter(.x, sig & capped), shape = 24,
                 fill = SIG_COLOR, color = "white", size = 4.0, stroke = 0.6, alpha = 0.92) +
      ggrepel::geom_text_repel(
        aes(label = repel_label), size = 3.2,
        segment.color = "#95a5a6", segment.size = 0.3, max.overlaps = Inf,
        min.segment.length = 0, box.padding = 0.5, point.padding = 0.8,
        point.size = 4, nudge_x = ifelse(d$coef_feature < 0, -1, 1) * x_span * .08,
        nudge_y = .25, force = 2, max.time = 4, max.iter = 100000, seed = 0) +
      coord_cartesian(xlim = c(-x_span, x_span)) +
      labs(x = "Cox log HR per SD", y = expression(-log[10](p)),
           title = title, subtitle = subtitle,
           caption = sprintf("%d / %d features q<0.05", n_sig, nrow(d))) +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 14),
            plot.subtitle = element_text(size = 10, color = COLOR_NEUTRAL_INK),
            plot.caption = element_text(size = 9, color = COLOR_NEUTRAL_INK, hjust = 0))
  }

  if (!IS_ADT) {
    message(paste0("Figure 3b: the sequencing and Gleason index dates are defined ",
                   "relative to ADT start, so these analyses are built for the ADT ",
                   "arm only -- skipping for the ARPI pass."))
  } else {
    for (analysis in SG_ANALYSES) {
      sub <- load_sg(analysis)
      if (is.null(sub) && !(analysis == "gleason" && ENDPOINT == "platinum")) next
      if (!is.null(sub)) {
        subtitle <- sprintf("%s endpoint  ·  time origin: %s  ·  n=%s patients",
                            toupper(ENDPOINT), SG_ORIGINS[[analysis]],
                            format(max(sub$n_patients_used, na.rm = TRUE), big.mark = ","))
        title <- sprintf("%s — univariate Cox", SG_LABELS[[analysis]])

        # Gleason contributes a single continuous feature, so a volcano would be
        # one point; a forest shows the effect size and its CI instead. The
        # many-feature analyses get a volcano plus a forest of the top hits.
        if (identical(analysis, "gleason") || nrow(sub) < 10) {
          p <- plot_sg_forest(sub, title, subtitle)
          save_fig(p, OUT_DIR,
                   sprintf("figure3b_%s_%s_forest_landmark%d", analysis, ENDPOINT, SG_LANDMARK),
                   width = 9, height = max(4.3, 0.36 * nrow(sub) + 2.8))
          if (show) print(p)
        } else {
          p <- plot_sg_volcano(sub, title, subtitle)
          save_fig(p, OUT_DIR,
                   sprintf("figure3b_%s_%s_volcano_landmark%d", analysis, ENDPOINT, SG_LANDMARK),
                   width = 7.5, height = 6)
          if (show) print(p)

          p_forest <- plot_sg_forest(sub, sprintf("%s — top associations", SG_LABELS[[analysis]]),
                                     subtitle)
          n_shown <- min(SG_TOP_N, nrow(sub))
          save_fig(p_forest, OUT_DIR,
                   sprintf("figure3b_%s_%s_forest_landmark%d", analysis, ENDPOINT, SG_LANDMARK),
                   width = 9, height = max(4.3, 0.36 * n_shown + 2.8))
          if (show) print(p_forest)
        }
      }
      if (ENDPOINT == "platinum") {
        index_path <- file.path(INPUTS_DIR, "somatic_gleason", analysis, "aggregated_landmark0.csv")
        if (!file.exists(index_path)) {
          message("Indexed platinum KM skipped: ", index_path, " is missing")
          next
        }
        # Match the displayed significant alterations exactly; untested or
        # missing calls never enter the non-carrier group.
        features <- if (analysis == "gleason") "GLEASON_SCORE" else
          significant_mutation_features(sub)
        if (!length(features)) { message("No significant mutation carriers to plot for ", COHORT); next }
        indexed <- read_csv(index_path, show_col_types = FALSE,
          col_select = any_of(c("DFCI_MRN", "t_platinum", "PLATINUM", features)),
          col_types = cols(.default = col_double(), DFCI_MRN = col_character()))
        for (feature in features) {
          if (!feature %in% names(indexed)) { warning("Indexed KM: missing feature ", feature); next }
          values <- suppressWarnings(as.numeric(indexed[[feature]]))
          if (analysis == "gleason") {
            groups <- figure_gleason_groups(values)
            group_order <- c("Gleason ≤6", "Gleason 7", "Gleason 8", "Gleason 9", "Gleason 10")
            title_km <- "Gleason score: time to platinum"
            note <- "Scores ≤6 are combined; higher scores shown individually. Missing scores excluded. Shading: 95% CI."
          } else {
            if (any(!is.na(values) & !values %in% c(0, 1))) {
              warning("Carrier KM skipped: ", feature, " is not a binary mutation call")
              next
            }
            groups <- ifelse(is.na(values), NA_character_, ifelse(values == 1, "Carrier", "Non-carrier"))
            group_order <- c("Non-carrier", "Carrier")
            title_km <- paste(gsub("_", " ", feature), "carrier status: time to platinum")
            note <- "Descriptive: variant selected using this cohort's Cox q<0.05. Missing calls excluded. Shading: 95% CI."
          }
          d <- figure_platinum_strata(indexed, groups)
          p_km <- plot_stratified_platinum(d, title_km, SG_ORIGINS[[analysis]], group_order,
                                          paste(COHORT_DISPLAY, note))
          if (is.null(p_km)) { message("Indexed KM skipped: fewer than two observed groups for ", feature); next }
          stem <- sprintf("figure3b_%s_platinum_km_%s_landmark0", analysis, lab_stem_slug(feature))
          save_fig(p_km, OUT_DIR, stem, 8.5, if (analysis == "gleason") 7.5 else 6.5)
          if (show) print(p_km)
        }
      }
    }
  }

  OUT_DIR <- fig_dir("figure4_multivariate")
  HAS_GGPATTERN <- requireNamespace("ggpattern", quietly = TRUE)
  notify_progress("stage", "Figure 4: model performance and importance")

  cox_labs <- function(lm) read_endpoint_performance(
    file.path(BASE, "cox", sprintf("landmark_%s", lm), "both", "cox_agg_multivariable_metrics.csv"),
    ENDPOINT)
  cox_baseline <- function(lm) read_endpoint_performance(
    file.path(BASE, "cox", sprintf("landmark_%s", lm), "baseline", "cox_agg_baseline_metrics.csv"),
    ENDPOINT)
  xgb_labs <- function(lm) read_endpoint_performance(
    file.path(BASE, "xgboost", sprintf("landmark_%s", lm), "both", "landmark_xgboost_metrics.csv"),
    ENDPOINT)
  xgb_baseline <- function(lm) read_endpoint_performance(
    file.path(BASE, "xgboost", sprintf("landmark_%s", lm), "baseline", "landmark_xgboost_baseline_metrics.csv"),
    ENDPOINT)
  # Retained with the disabled all-model supplement below. Keep its recursive
  # input discovery unreachable too; otherwise every active figure cell scans
  # the Dynamic-DeepHit tree for a figure that is never emitted.
  if (FALSE) {
  # 03b writes one directory per config in _LONGITUDINAL_CONFIGS_BY_ENDPOINT.
  # The cause-only config leads that tuple and is the arm comparable to
  # Cox/XGBoost here, so it is named for the endpoint itself.
  DEEPHIT_CONFIG <- ENDPOINT
  resolve_dynamic_deephit_metrics <- function(lm) {
    filename <- sprintf("dynamic_deephit_metrics_%s.csv", DEEPHIT_CONFIG)
    preferred <- c(
      file.path(
        BASE, "multivariate_longitudinal", "dynamic_deephit",
        sprintf("landmark_%s", lm), DEEPHIT_CONFIG, filename
      ),
      file.path(
        BASE, "multivariate_longitudinal", "dynamic-deephit",
        sprintf("landmark_%s", lm), DEEPHIT_CONFIG, filename
      )
    )
    existing <- preferred[file.exists(preferred)]
    if (length(existing) > 0) return(existing[1])

    # Accommodate older/custom 03b output roots while still requiring the
    # requested landmark, endpoint config, and exact metrics filename.
    longitudinal_root <- file.path(BASE, "multivariate_longitudinal")
    if (!dir.exists(longitudinal_root)) return(NA_character_)
    discovered <- list.files(
      longitudinal_root, pattern = paste0("^", filename, "$"),
      recursive = TRUE, full.names = TRUE
    )
    normalized <- gsub("\\\\", "/", discovered)
    wanted <- endsWith(
      normalized,
      paste0("/landmark_", lm, "/", DEEPHIT_CONFIG, "/", filename)
    )
    discovered <- discovered[wanted]
    if (length(discovered) > 1) {
      warning(sprintf(
        "Figure 4 supplement: multiple Dynamic-DeepHit files for landmark %s; using %s",
        lm, discovered[1]
      ))
    }
    if (length(discovered) == 0) NA_character_ else discovered[1]
  }
  DEEPHIT_METRIC_PATHS <- setNames(
    map_chr(LANDMARKS, resolve_dynamic_deephit_metrics),
    as.character(LANDMARKS)
  )
  dynamic_deephit <- function(lm) {
    path <- DEEPHIT_METRIC_PATHS[[as.character(lm)]]
    if (is.na(path)) return(c(auc = NA_real_, cindex = NA_real_, brier = NA_real_))
    read_endpoint_performance(path, ENDPOINT)
  }
  } # retired Dynamic-DeepHit discovery

  # (label, loader, color, is_baseline). Age baselines are the lighter, patterned twins.
  DISCRIMINATION_SERIES <- tibble::tribble(
    ~name,                    ~loader,        ~color,     ~baseline,
    "Elastic-Net Cox",        list(cox_labs),     "#4C72B0", FALSE,
    "Cox baseline (age)",     list(cox_baseline), "#9DB3D6", TRUE,
    "XGBoost Survival",       list(xgb_labs),     "#B58900", FALSE,
    "XGBoost baseline (age)", list(xgb_baseline), "#E0CC8A", TRUE
  )
  SERIES_LEVELS <- DISCRIMINATION_SERIES$name
  SERIES_COLORS <- setNames(DISCRIMINATION_SERIES$color, SERIES_LEVELS)

  # Long tidy frame: one row per (series x landmark x metric).
  discrimination_data <- DISCRIMINATION_SERIES %>%
    rowwise() %>%
    do({
      s <- .
      map_dfr(LANDMARKS, function(lm) {
        v <- s$loader[[1]](lm)
        tibble(name = s$name, landmark = lm, auc = v[["auc"]], cindex = v[["cindex"]])
      })
    }) %>% ungroup() %>%
    mutate(name = factor(name, levels = SERIES_LEVELS))

  render_discrimination_panel <- function(metric, ylabel, show_legend = FALSE) {
    d <- discrimination_data %>%
      transmute(name, baseline = name %in% c("Cox baseline (age)","XGBoost baseline (age)"),
                landmark = factor(sprintf("%s%d days", ifelse(landmark > 0, "+", ""), landmark),
                                  levels = sprintf("%s%d days", ifelse(LANDMARKS > 0,"+",""), LANDMARKS)),
                value = .data[[metric]])
    finite_max <- suppressWarnings(max(d$value, na.rm = TRUE))
    ymax <- min(1.0, (if (is.finite(finite_max)) finite_max else 0.7) * 1.12)

    base <- ggplot(d, aes(landmark, value, group = name))
    if (HAS_GGPATTERN) {
      p <- base + ggpattern::geom_col_pattern(
        aes(fill = name, pattern = baseline),
        position = position_dodge(width = 0.85), width = 0.8,
        color = "white", pattern_fill = "white", pattern_density = 0.08,
        pattern_spacing = 0.02, pattern_angle = 45) +
        ggpattern::scale_pattern_manual(values = c(`TRUE` = "stripe", `FALSE` = "none"), guide = "none")
    } else {
      p <- base + geom_col(aes(fill = name, linetype = baseline),
                           position = position_dodge(width = 0.85), width = 0.8,
                           color = "white") +
        scale_linetype_manual(values = c(`TRUE` = "dashed", `FALSE` = "solid"), guide = "none")
    }
    p +
      geom_text(aes(label = ifelse(is.finite(value), sprintf("%.3f", value), "")),
                position = position_dodge(width = 0.85), vjust = -0.4, size = 2.5,
                show.legend = FALSE) +
      geom_hline(yintercept = 0.5, color = "grey", linetype = "dotted", linewidth = 0.9) +
      scale_fill_manual(values = SERIES_COLORS, name = NULL, drop = FALSE) +
      coord_cartesian(ylim = c(0, max(1.04, ymax))) +
      labs(x = NULL, y = ylabel) +
      theme_fig() +
      guides(fill = guide_legend(ncol = 2, byrow = TRUE)) +
      theme(legend.position = if (show_legend) "top" else "none",
            legend.justification = "right",
            legend.box.just = "right",
            legend.direction = "horizontal",
            panel.grid.major.x = element_blank())
  }

  disc_panels <- list(
    list(sprintf("figure4a_discrimination_auc_%s", ENDPOINT),    "auc",    "Test Mean AUC(t)"),
    list(sprintf("figure4a_discrimination_cindex_%s", ENDPOINT), "cindex", "Test C-index")
  )
  for (dp in disc_panels) {
    p <- render_discrimination_panel(dp[[2]], dp[[3]], show_legend = TRUE) +
      labs(title = sprintf("Labs vs. age baseline \u2014 %s (%s)", ENDPOINT, dp[[3]])) +
      theme(plot.title = element_text(face = "bold", size = 11))
    save_fig(p, OUT_DIR, dp[[1]], width = 7.5, height = 5.5)
    if (show) print(p)
  }

  # Source-specific available-case sensitivities. Each source (Gleason or
  # somatic) has its own matched labs comparator, preserving the original
  # ADT-relative outcome clock and held-out split within every landmark.
  sensitivity_metric_path <- function(analysis, feature_set, model, landmark, filename) {
    file.path(BASE, "sensitivity_available_case", analysis, feature_set, model,
              sprintf("landmark_%s", landmark), "both", filename)
  }
  render_available_case_sensitivity <- function(sensitivity_data, series, analysis_label, metric, ylabel) {
    d <- sensitivity_data %>% transmute(
      name,
      landmark = factor(sprintf("%s%d days", ifelse(landmark > 0, "+", ""), landmark),
                        levels = sprintf("%s%d days", ifelse(LANDMARKS > 0, "+", ""), LANDMARKS)),
      value = .data[[metric]]
    )
    ggplot(d, aes(landmark, value, fill = name)) +
      geom_col(position = position_dodge(width = 0.85), width = 0.8, color = "white") +
      geom_text(aes(label = ifelse(is.finite(value), sprintf("%.3f", value), "")),
                position = position_dodge(width = 0.85), vjust = -0.35, size = 2.5, show.legend = FALSE) +
      geom_hline(yintercept = 0.5, color = "grey", linetype = "dotted", linewidth = 0.9) +
      scale_fill_manual(values = setNames(series$color, series$name),
                        name = NULL, drop = FALSE) +
      coord_cartesian(ylim = c(0, 1.04)) +
      labs(x = NULL, y = ylabel,
           title = sprintf("%s vs. labs sensitivity — %s", analysis_label, ENDPOINT),
           subtitle = "Same cases within each landmark; source data available by that landmark") +
      theme_fig() +
      guides(fill = guide_legend(ncol = 2, byrow = TRUE)) +
      theme(legend.position = "top", legend.text = element_text(size = 8),
            panel.grid.major.x = element_blank(),
            plot.title = element_text(face = "bold", size = 11))
  }
  for (sensitivity_analysis in c("gleason", "somatic")) {
    analysis_label <- ifelse(sensitivity_analysis == "gleason", "Gleason", "Somatic")
    source_label <- ifelse(sensitivity_analysis == "gleason", "Gleason", "Somatic")
    SENSITIVITY_SERIES <- tibble::tribble(
      ~name,                         ~loader, ~color,
      "Elastic-Net Cox: labs",      list(function(lm) read_endpoint_performance(sensitivity_metric_path(sensitivity_analysis, "labs", "cox", lm, "cox_agg_multivariable_metrics.csv"), ENDPOINT)), "#4C72B0",
      paste("Elastic-Net Cox:", source_label), list(function(lm) read_endpoint_performance(sensitivity_metric_path(sensitivity_analysis, "somatic-gleason", "cox", lm, "cox_agg_multivariable_metrics.csv"), ENDPOINT)), "#2A9D8F",
      "XGBoost: labs",              list(function(lm) read_endpoint_performance(sensitivity_metric_path(sensitivity_analysis, "labs", "xgboost", lm, "landmark_xgboost_metrics.csv"), ENDPOINT)), "#B58900",
      paste("XGBoost:", source_label), list(function(lm) read_endpoint_performance(sensitivity_metric_path(sensitivity_analysis, "somatic-gleason", "xgboost", lm, "landmark_xgboost_metrics.csv"), ENDPOINT)), "#D55E00"
    )
    sensitivity_data <- SENSITIVITY_SERIES %>% rowwise() %>% do({
      s <- .
      map_dfr(LANDMARKS, function(lm) {
        v <- s$loader[[1]](lm)
        tibble(name = s$name, landmark = lm, auc = v[["auc"]], cindex = v[["cindex"]])
      })
    }) %>% ungroup() %>% mutate(name = factor(name, levels = SENSITIVITY_SERIES$name))
    panel_prefix <- ifelse(sensitivity_analysis == "gleason", "figure4c", "figure4d")
    for (dp in disc_panels) {
      p <- render_available_case_sensitivity(sensitivity_data, SENSITIVITY_SERIES,
                                               analysis_label, dp[[2]], dp[[3]])
      save_fig(p, OUT_DIR,
               sub("figure4a_discrimination_", paste0(panel_prefix, "_sensitivity_", sensitivity_analysis, "_"), dp[[1]]),
               width = 8.5, height = 5.5)
      if (show) print(p)
    }
  }

  # Retired: supplemental all-model comparison.
  if (FALSE) {
  # Supplemental held-out comparison across every directly comparable
  # multivariate lab model. Dynamic-DeepHit's cause-only configuration censors
  # death, matching the Cox/XGBoost endpoint; the competing-risk configuration
  # is intentionally excluded because it estimates a different quantity.
  SUPPLEMENT_SERIES <- tibble::tribble(
    ~name,               ~loader,                 ~color,
    "Elastic-Net Cox",   list(cox_labs),           "#4C72B0",
    "XGBoost Survival",  list(xgb_labs),           "#B58900",
    "Dynamic-DeepHit",   list(dynamic_deephit),    "#2A9D8F"
  )
  supplement_levels <- SUPPLEMENT_SERIES$name
  supplement_colors <- setNames(SUPPLEMENT_SERIES$color, supplement_levels)
  multivariate_supplement_data <- SUPPLEMENT_SERIES %>%
    rowwise() %>%
    do({
      s <- .
      map_dfr(LANDMARKS, function(lm) {
        v <- s$loader[[1]](lm)
        tibble(
          model = s$name, landmark_days = lm,
          mean_auc_t = v[["auc"]], c_index = v[["cindex"]],
          integrated_brier = v[["brier"]]
        )
      })
    }) %>%
    ungroup() %>%
    mutate(model = factor(model, levels = supplement_levels))

  multivariate_supplement_data <- multivariate_supplement_data %>%
    mutate(
      metrics_status = case_when(
        is.finite(mean_auc_t) & is.finite(c_index) & is.finite(integrated_brier) ~ "available",
        is.finite(mean_auc_t) | is.finite(c_index) | is.finite(integrated_brier) ~ "partial",
        TRUE ~ "missing"
      ),
      source_path = if_else(
        model == "Dynamic-DeepHit",
        unname(DEEPHIT_METRIC_PATHS[as.character(landmark_days)]),
        NA_character_
      )
    )
  deephit_landmarks_found <- multivariate_supplement_data %>%
    filter(model == "Dynamic-DeepHit", metrics_status != "missing") %>%
    pull(landmark_days)
  deephit_status <- if (length(deephit_landmarks_found) > 0) {
    sprintf(
      "Dynamic-DeepHit metrics found at landmark(s): %s days",
      paste(deephit_landmarks_found, collapse = ", ")
    )
  } else {
    "Dynamic-DeepHit metrics not found; run 03b_multivariate_longitudinal.ipynb"
  }
  message("Figure 4 supplement: ", deephit_status)
  for (lm in LANDMARKS) {
    path <- DEEPHIT_METRIC_PATHS[[as.character(lm)]]
    message(sprintf(
      "  Dynamic-DeepHit landmark %s: %s",
      lm, if (is.na(path)) "missing" else path
    ))
  }

  supplement_long <- multivariate_supplement_data %>%
    pivot_longer(
      c(mean_auc_t, c_index, integrated_brier),
      names_to = "metric", values_to = "value"
    ) %>%
    mutate(
      metric = factor(
        metric,
        levels = c("mean_auc_t", "c_index", "integrated_brier"),
        labels = c("Mean AUC(t)", "C-index", "Integrated Brier score")
      ),
      landmark = factor(
        sprintf("%s%d days", ifelse(landmark_days > 0, "+", ""), landmark_days),
        levels = sprintf(
          "%s%d days", ifelse(LANDMARKS > 0, "+", ""), LANDMARKS
        )
      ),
      # Nearby model estimates otherwise print directly on top of each other.
      label_vjust = case_when(
        model == "Elastic-Net Cox"  ~ -2.2,
        model == "XGBoost Survival" ~  1.7,
        TRUE                         ~ -0.6
      )
    )

  p_supplement <- ggplot(
    supplement_long,
    aes(landmark, value, color = model, group = model)
  ) +
    geom_hline(
      data = tibble(
        metric = factor(
          c("Mean AUC(t)", "C-index"),
          levels = levels(supplement_long$metric)
        ),
        reference = 0.5
      ),
      aes(yintercept = reference), inherit.aes = FALSE,
      color = "grey65", linetype = "dotted", linewidth = 0.7
    ) +
    geom_line(linewidth = 0.9, na.rm = TRUE) +
    geom_point(size = 2.7, na.rm = TRUE) +
    geom_text(
      aes(label = ifelse(is.finite(value), sprintf("%.3f", value), ""),
          vjust = label_vjust),
      size = 2.5, show.legend = FALSE, na.rm = TRUE
    ) +
    facet_wrap(~metric, scales = "free_y", nrow = 1) +
    scale_color_manual(values = supplement_colors, drop = FALSE, name = NULL) +
    scale_y_continuous(labels = label_number(accuracy = 0.001),
                       expand = expansion(mult = c(0.14, 0.24))) +
    labs(
      title = "Supplementary Figure — Held-out multivariate model performance",
      subtitle = paste(
        sprintf("Death-censored %s endpoint; Dynamic-DeepHit uses longitudinal person-period labs.",
                ENDPOINT),
        deephit_status
      ),
      x = NULL, y = NULL,
      caption = paste0(
        "Higher AUC(t)/C-index and lower integrated Brier score indicate better performance. ",
        "Gaps indicate unavailable metrics."
      )
    ) +
    theme_fig() +
    theme(
      legend.position = "top",
      legend.direction = "horizontal",
      panel.spacing.x = unit(1.2, "lines"),
      panel.grid.major.x = element_blank(),
      strip.text = element_text(face = "bold")
    )

  supplement_stem <- "figure4s_multivariate_all_models"
  save_fig(p_supplement, OUT_DIR, supplement_stem, width = 13, height = 5.5)
  supplement_csv <- file.path(
    output_dir_for_stem(supplement_stem),
    paste0(COHORT_LEAF, "_data.csv")
  )
  write_csv(multivariate_supplement_data, supplement_csv)
  message("wrote ", supplement_csv)
  if (show) print(p_supplement)
  }

  OUT_DIR <- fig_dir("figure4_multivariate")

  load_cox_coefs <- function(landmark) {
    p <- file.path(BASE, "cox", sprintf("landmark_%s", landmark), "both", "cox_agg_multivariable.csv")
    read_csv(p, show_col_types = FALSE) %>%
      filter(tolower(as.character(endpoint)) == ENDPOINT) %>%
      filter(!coalesce(as.logical(is_age_covariate), FALSE)) %>%
      filter(coalesce(coef, 0) != 0)
  }

  load_xgb_importance <- function(landmark) {
    p <- file.path(BASE, "xgboost", sprintf("landmark_%s", landmark), "both", "landmark_xgboost_feature_importance.csv")
    df <- read_csv(p, show_col_types = FALSE) %>%
      filter(tolower(as.character(endpoint)) == ENDPOINT) %>%
      filter(tolower(feature) != "age") %>%
      filter(coalesce(gain, 0) > 0)
    parsed <- t(vapply(df$feature, parse_feature, character(2)))
    df$lab_name <- parsed[, 1]; df$feature_stat <- parsed[, 2]
    df
  }

  render_importance_panel <- function(df, kind, title) {
    if (nrow(df) == 0) {
      return(ggplot() + annotate("text", x = 0, y = 0, label = "(no features to display)",
                                 color = "#7f8c8d") + theme_void() +
               labs(title = title) + theme(plot.title = element_text(face = "bold", size = 11)))
    }
    df <- df %>%
      filter(!lab_name %in% DROP) %>%
      mutate(category = vapply(lab_name, assign_category, character(1)))
    if (nrow(df) == 0) {
      return(ggplot() + annotate("text", x = 0, y = 0, label = "(no features to display)",
                                 color = "#7f8c8d") + theme_void() +
               labs(title = title) + theme(plot.title = element_text(face = "bold", size = 11)))
    }
    if (kind == "cox") {
      df <- df %>% arrange(desc(abs(coef))) %>% head(TOP_N)
      df$value <- df$coef; xlabel <- "log HR coefficient"
    } else {
      df <- df %>% arrange(desc(gain)) %>% head(TOP_N)
      df$value <- df$gain; xlabel <- "XGBoost gain"
    }
    df <- df %>% mutate(
      label = mapply(format_label, lab_name, feature_stat),
      category = factor(category, levels = LEGEND_ORDER))
    df$label <- factor(df$label, levels = rev(df$label))   # top feature at top of barh

    p <- ggplot(df, aes(value, label, fill = category)) +
      geom_col(color = "white", linewidth = 0.5) +
      scale_fill_manual(values = CATEGORY_COLORS, breaks = LEGEND_ORDER, name = NULL, drop = FALSE) +
      labs(x = xlabel, y = NULL, title = title) +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 11),
            axis.text.y = element_text(size = 10))
    if (kind == "cox") p <- p + geom_vline(xintercept = 0, color = "black", linewidth = 0.5)
    p
  }

  IMPORTANCE_MODEL_ROWS <- list(
    list("cox", "Elastic-Net Cox",  load_cox_coefs),
    list("xgb", "XGBoost Survival", load_xgb_importance)
  )

  for (row in IMPORTANCE_MODEL_ROWS) {
    kind <- row[[1]]; model_name <- row[[2]]; loader <- row[[3]]
    for (lm in LANDMARKS) {
      df <- tryCatch(loader(lm), error = function(e) tibble())
      sign <- if (lm > 0) "+" else ""
      title <- sprintf("%s  \u00b7  %s%d days", model_name, sign, lm)
      p <- render_importance_panel(df, kind, title)
      save_fig(p, OUT_DIR, sprintf("figure4b_importance_%s_%s_landmark%d", ENDPOINT, kind, lm),
               width = 7.5, height = 5.5)
      if (show) print(p)
    }
  }

  # Every per-lab panel below is platinum-specific: the KM panels model
  # time-to-platinum, distributions split on PLATINUM, and trajectory panels
  # are descriptive platinum/classifier strata. They are not survival-endpoint
  # comparisons and therefore belong only to the canonical platinum pass.
  if (!EMIT_ENDPOINT_INDEPENDENT) {
    message(sprintf("Per-lab figures: emitted only on the %s endpoint pass; skipping %s",
                    ENDPOINT_INDEPENDENT_FIGURES_ENDPOINT, ENDPOINT))
  } else {
  OUT_DIR <- fig_dir("androgen_supplements")
  notify_progress("stage", "Lab trajectories and pre-ADT coverage")
  # All canonical labs in CATEGORY_MAP (CBC/CMP/LFT/Vitals/Androgen axis/Other),
  # generalizing what used to be the PSA/Testosterone-only ANDROGEN_LABS list.
  ALL_LABS <- names(CATEGORY_MAP)
  LAB_FIGURE_LABS <- if (plot_non_androgen_lab_figures) {
    ALL_LABS
  } else {
    intersect(ALL_LABS, ANDROGEN)
  }
  message(sprintf(
    "Per-lab KM/longitudinal figures: %s (%s labs)",
    if (plot_non_androgen_lab_figures) "all canonical labs" else "androgen axis only",
    length(LAB_FIGURE_LABS)
  ))

  # Case-insensitive match for a `<lab>__mean` column (mirrors find_col in the
  # Python COMPASS diagnostics).
  find_mean_col <- function(columns, substr) {
    hit <- columns[grepl(tolower(substr), tolower(columns), fixed = TRUE) &
                   endsWith(columns, "__mean")]
    if (length(hit)) hit[1] else NA_character_
  }
  # resolve_mean_col: generic replacement for resolve_androgen_columns, over
  # any CATEGORY_MAP lab name. PSA additionally falls back to the raw OMOP
  # "Prostate specific Ag" name, mirroring the original PSA-specific handling.
  resolve_mean_col <- function(columns, lab_name) {
    hit <- find_mean_col(columns, lab_name)
    if (is.na(hit) && identical(lab_name, "PSA")) hit <- find_mean_col(columns, "Prostate specific Ag")
    hit
  }

  # 4-way quartile split; Figure 5 contrasts the lowest and highest quartiles.
  quartile_split <- function(df, col) {
    vals <- suppressWarnings(as.numeric(df[[col]]))
    out <- rep(NA_character_, length(vals))
    qs <- tryCatch(quantile(vals, c(0.25, 0.5, 0.75), na.rm = TRUE, names = FALSE),
                   error = function(e) NULL)
    if (is.null(qs) || any(diff(qs) <= 0)) return(out)
    lvl <- cut(vals, breaks = c(-Inf, qs, Inf),
               labels = c("Low (Q1)", "Q2", "Q3", "High (Q4)"))
    out[!is.na(vals)] <- as.character(lvl)[!is.na(vals)]
    out
  }

  # Multi-stratum KM overlay (analogue of survival_common.plotting.overlay_km).
  # survival_by_label: named list of tibbles each with columns (time, event).
  KM_PALETTE <- c("#1f3a93", "#8e1c2b", "#2e7d32", "#b8860b", "#6a3d9a")
  overlay_km <- function(survival_by_label, xlabel, ylabel, title) {
    curves <- imap_dfr(survival_by_label, function(d, lab) {
      fit <- survfit(Surv(time, event) ~ 1, data = d)
      tibble(label = lab, time = c(0, fit$time), surv = c(1, fit$surv),
             lo = c(1, fit$lower), hi = c(1, fit$upper))
    })
    labs_lvl <- names(survival_by_label)
    pal <- setNames(KM_PALETTE[seq_along(labs_lvl)], labs_lvl)
    ggplot(curves, aes(time, surv, color = label, fill = label)) +
      geom_step(linewidth = 0.9) +
      geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, color = NA,
                  outline.type = "full", show.legend = FALSE) +
      scale_color_manual(values = pal, name = NULL) +
      scale_fill_manual(values = pal, guide = "none") +
      coord_cartesian(ylim = c(0, 1.02)) +
      labs(x = xlabel, y = ylabel, title = title) +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 10),
            legend.position = "bottom", legend.justification = "center")
  }

  aggregated_landmark_cache <- new.env(parent = emptyenv())
  load_aggregated_landmark <- function(landmark) {
    key <- as.character(landmark)
    if (exists(key, envir = aggregated_landmark_cache, inherits = FALSE)) {
      value <- aggregated_landmark_cache[[key]]
      if (inherits(value, "missing_aggregated")) return(NULL)
      return(value)
    }
    path <- file.path(INPUTS_DIR, sprintf("aggregated_landmark%s.csv", landmark))
    if (!file.exists(path)) {
      message(sprintf("skipped landmark %s -- %s not found", landmark, path))
      aggregated_landmark_cache[[key]] <- structure(list(), class = "missing_aggregated")
      return(NULL)
    }
    value <- read_csv(path, show_col_types = FALSE,
      col_select = c(any_of(c("DFCI_MRN", "t_platinum", "PLATINUM")),
        matches("^(PSA|Testosterone|Prostate specific Ag).*__mean$", ignore.case = TRUE)),
      col_types = cols(.default = col_double(), DFCI_MRN = col_character()))
    aggregated_landmark_cache[[key]] <- value
    value
  }

  FIG5_LANDMARKS <- LANDMARKS

  plot_km_androgen_quartile <- function(agg, lab, mean_col, landmark) {
    ttl <- sprintf("%s quartile -- landmark %s%dd", lab, ifelse(landmark > 0, "+", ""), landmark)
    blank <- function(msg) ggplot() + annotate("text", x = 0, y = 0, label = msg, color = "#7f8c8d") +
      theme_void() + labs(title = ttl) + theme(plot.title = element_text(face = "bold", size = 10))
    if (is.na(mean_col) || !all(c("t_platinum","PLATINUM") %in% names(agg))) return(blank("(no data)"))

    d <- agg
    d$stratum <- quartile_split(d, mean_col)
    d <- d %>% filter(!is.na(stratum), !is.na(t_platinum), !is.na(PLATINUM),
                      stratum %in% c("Low (Q1)","High (Q4)"))
    if (nrow(d) == 0 || length(unique(d$stratum)) < 2)
      return(blank("(insufficient data after quartile split)"))

    km <- platinum_km_inputs(d)
    d <- d[km$row_id, , drop = FALSE]
    d$km_time <- km$time; d$km_event <- km$event
    survival_by_label <- split(d %>% transmute(time = km_time, event = km_event), d$stratum)
    ttl2 <- sprintf("%s quartile (%s) -- landmark %s%dd", lab, mean_col,
                    ifelse(landmark > 0, "+", ""), landmark)
    p <- overlay_km(survival_by_label, "Days from landmark", "Platinum-free probability", ttl2)

    n_low <- sum(d$stratum == "Low (Q1)"); n_high <- sum(d$stratum == "High (Q4)")
    ev_low <- sum(d$km_event[d$stratum == "Low (Q1)"])
    ev_high <- sum(d$km_event[d$stratum == "High (Q4)"])
    ann <- sprintf("Low: n=%d, events=%d\nHigh: n=%d, events=%d", n_low, ev_low, n_high, ev_high)
    sd <- survdiff(Surv(km_time, km_event) ~ stratum, data = d)
    pval <- 1 - pchisq(sd$chisq, length(sd$n) - 1)
    ann <- sprintf("%s\nlog-rank p = %.3g", ann, pval)
    p + annotate("text", x = Inf, y = 0, label = ann, hjust = 1.05, vjust = 0,
                 size = 2.6, color = "#5d6d7e", family = "sans")
  }

  # Extreme-quintile contrast, using the same incident risk set and pre-landmark
  # mean features as the fitted models. Only PSA/testosterone are requested.
  notify_progress("stage", "Time to platinum: PSA/testosterone bottom vs top 20%")
  for (landmark in LANDMARKS) {
    agg <- load_aggregated_landmark(landmark)
    if (is.null(agg)) next
    if (!all(c("DFCI_MRN", "t_platinum", "PLATINUM") %in% names(agg))) {
      warning("Tertile KM skipped: missing platinum outcomes at landmark ", landmark)
      next
    }
    eligible <- is.finite(agg$t_platinum) & agg$t_platinum > 0 & agg$PLATINUM %in% c(0, 1)
    agg <- agg[eligible, , drop = FALSE]
    for (lab in ANDROGEN) {
      column <- resolve_mean_col(names(agg), lab)
      if (is.na(column)) { message("Tertile KM: missing ", lab, " mean at landmark ", landmark); next }
      groups <- figure_extreme_quintiles(agg[[column]])
      if (all(is.na(groups))) {
        message("Quintile KM: ", lab, " at landmark ", landmark,
                " cannot form distinct bottom/top 20% groups without splitting tied values; skipped")
        next
      }
      d <- figure_platinum_strata(agg, groups)
      cuts <- attr(groups, "cutpoints")
      note <- sprintf("%s. Pre-landmark mean; bottom 20%% <= %.4g, top 20%% > %.4g (middle 60%% not shown). Equal values stay together. Shading: 95%% CI.",
                      COHORT_DISPLAY, cuts[1], cuts[2])
      p <- plot_stratified_platinum(d, paste(lab, "bottom vs top 20%: time to platinum"),
        sprintf("the +%d-day treatment landmark", landmark),
        c("Bottom 20%", "Top 20%"), note)
      if (is.null(p)) next
      save_fig(p, OUT_DIR, sprintf("km_quintile_%s_landmark%d", lab_stem_slug(lab), landmark), 8, 6.5)
      if (show) print(p)
    }
  }

  # Retired: per-lab quartile KM figures.
  if (FALSE) for (lab in LAB_FIGURE_LABS) {
    for (landmark in FIG5_LANDMARKS) {
      agg <- load_aggregated_landmark(landmark)
      if (is.null(agg)) {
        p <- ggplot() + annotate("text", x = 0, y = 0, label = "(aggregated CSV not found)",
                                 color = "#7f8c8d") + theme_void() +
          labs(title = sprintf("%s quartile -- landmark %dd", lab, landmark))
      } else {
        mean_col <- resolve_mean_col(names(agg), lab)
        if (is.na(mean_col)) {
          message(sprintf("km_quartile_%s: no %s__mean column at landmark %d -- skipping",
                          lab_stem_slug(lab), lab, landmark))
          next
        }
        p <- plot_km_androgen_quartile(agg, lab, mean_col, landmark)
      }
      save_fig(p, OUT_DIR, sprintf("km_quartile_%s_landmark%d", lab_stem_slug(lab), landmark),
               width = 6.5, height = 5.5)
      if (show) print(p)
    }
  }

  # Retired: LLM-stratified KM figures.
  if (FALSE) {
  ## ---- LLM-stratified KM curves: time-to-platinum by has_nepc -----------
  ## ---- (platinum_km_inputs) + overlay_km as the quartile curves above.    ----
  if (is.null(llm_classifier_labels)) {
    message("km_llm_*: llm_classifier_labels unavailable -- skipping.")
  } else {
    cohort_mrns_km <- unique(as.character(patient_df[[ID_COL]]))
    strata_cohort <- llm_classifier_labels %>%
      filter(as.character(DFCI_MRN) %in% cohort_mrns_km)
    message(sprintf("km_llm_*: %s / %s LLM-labeled MRNs in %s cohort",
                    format(nrow(strata_cohort), big.mark = ","),
                    format(nrow(llm_classifier_labels), big.mark = ","), COHORT_DISPLAY))

    plot_km_llm_stratum <- function(scheme_name, landmark) {
      scheme <- FIGURE_LLM_STRATA[[scheme_name]]
      ttl <- sprintf("time-to-platinum by %s -- landmark %s%dd", scheme_name,
                     ifelse(landmark > 0, "+", ""), landmark)
      blank <- function(msg) ggplot() + annotate("text", x = 0, y = 0, label = msg, color = "#7f8c8d") +
        theme_void() + labs(title = ttl) + theme(plot.title = element_text(face = "bold", size = 10))

      agg <- load_aggregated_landmark(landmark)
      if (is.null(agg) || !all(c("t_platinum", "PLATINUM", "DFCI_MRN") %in% names(agg)))
        return(blank("(no data)"))

      d <- agg %>%
        mutate(DFCI_MRN = as.character(DFCI_MRN)) %>%
        inner_join(strata_cohort %>% mutate(DFCI_MRN = as.character(DFCI_MRN)), by = "DFCI_MRN")
      d$stratum <- as.character(d[[scheme$col]])
      if (!is.null(scheme$labels)) {
        d$stratum <- scheme$labels[match(d$stratum, as.character(scheme$levels))]
      }
      d <- d %>% filter(!is.na(stratum), !is.na(t_platinum), !is.na(PLATINUM))
      if (nrow(d) == 0 || length(unique(d$stratum)) < 2)
        return(blank("(insufficient labeled data)"))

      km <- platinum_km_inputs(d)
      d <- d[km$row_id, , drop = FALSE]
      d$km_time <- km$time; d$km_event <- km$event
      survival_by_label <- split(d %>% transmute(time = km_time, event = km_event), d$stratum)
      p <- overlay_km(survival_by_label, "Days from landmark", "Platinum-free probability", ttl)

      n_by_stratum <- table(d$stratum)
      ev_by_stratum <- tapply(d$km_event, d$stratum, sum)
      ann <- paste(sprintf("%s: n=%d, events=%d", names(n_by_stratum), n_by_stratum,
                           ev_by_stratum[names(n_by_stratum)]), collapse = "\n")
      if (length(unique(d$stratum)) >= 2) {
        sd <- tryCatch(survdiff(Surv(km_time, km_event) ~ stratum, data = d), error = function(e) NULL)
        if (!is.null(sd)) {
          pval <- 1 - pchisq(sd$chisq, length(sd$n) - 1)
          ann <- sprintf("%s\nlog-rank p = %.3g", ann, pval)
        }
      }
      p + annotate("text", x = Inf, y = 0, label = ann, hjust = 1.05, vjust = 0,
                   size = 2.6, color = "#5d6d7e", family = "sans")
    }

    for (scheme_name in names(FIGURE_LLM_STRATA)) {
      for (landmark in FIG5_LANDMARKS) {
        p <- plot_km_llm_stratum(scheme_name, landmark)
        save_fig(p, fig_dir("km_llm"), sprintf("km_llm_%s_landmark%d", scheme_name, landmark),
                 width = 6.5, height = 5.5)
        if (show) print(p)
      }
    }
  }
  }

  FIG6_LANDMARKS <- LANDMARKS
  PLAT_COLORS <- c(`0` = "#1f3a93", `1` = "#8e1c2b")
  PLAT_LABELS <- c(`0` = "PLATINUM=0", `1` = "PLATINUM=1")

  plot_androgen_dist_by_platinum <- function(agg, lab, mean_col, landmark, log_scale) {
    ttl <- sprintf("%s -- landmark %s%dd", lab, ifelse(landmark > 0, "+", ""), landmark)
    blank <- function(msg) ggplot() + annotate("text", x = 0, y = 0, label = msg, color = "#7f8c8d") +
      theme_void() + labs(title = ttl) + theme(plot.title = element_text(face = "bold", size = 10))
    if (is.na(mean_col) || !mean_col %in% names(agg) || !"PLATINUM" %in% names(agg))
      return(blank("(no data)"))

    d <- tibble(val = suppressWarnings(as.numeric(agg[[mean_col]])),
                plat = suppressWarnings(as.numeric(agg$PLATINUM))) %>% drop_na()
    if (log_scale) d <- d %>% filter(val >= 0)  # keep valid zeros for log1p
    if (nrow(d) == 0) return(blank("(no data after filtering)"))

    d <- d %>% filter(plat %in% c(0, 1)) %>%
      mutate(plot_val = if (log_scale) log1p(val) else val,
             plat = factor(plat, levels = c(0, 1)))
    if (nrow(d) == 0) return(blank("(no data)"))

    lo <- min(d$plot_val); hi <- max(d$plot_val)
    if (hi <= lo) hi <- lo + 1

    counts <- d %>% count(plat)
    leg <- setNames(sprintf("%s (n=%s)", PLAT_LABELS[as.character(counts$plat)],
                            format(counts$n, big.mark = ",")), as.character(counts$plat))

    xlab <- if (log_scale) sprintf("log1p(%s %s)", lab, mean_col) else sprintf("%s %s", lab, mean_col)

    ann <- d %>% group_by(plat) %>%
      summarise(med = median(val), q1 = quantile(val, .25), q3 = quantile(val, .75), .groups = "drop") %>%
      mutate(line = sprintf("%s: med=%.2f (IQR %.2f-%.2f)", PLAT_LABELS[as.character(plat)], med, q1, q3))
    ann_lines <- paste(ann$line, collapse = "\n")
    g0 <- d$val[d$plat == 0]; g1 <- d$val[d$plat == 1]
    if (length(g0) && length(g1)) {
      pv <- suppressWarnings(wilcox.test(g0, g1)$p.value)
      ann_lines <- sprintf("%s\nMann-Whitney p = %.3g", ann_lines, pv)
    }

    ggplot(d, aes(plot_val, after_stat(density), fill = plat)) +
      geom_histogram(bins = 30, alpha = 0.55, color = "white", linewidth = 0.15,
                     position = "identity") +
      scale_fill_manual(values = PLAT_COLORS, labels = leg, name = NULL) +
      coord_cartesian(xlim = c(lo, hi)) +
      labs(x = xlab, y = "Density", title = ttl) +
      annotate("text", x = Inf, y = Inf, label = ann_lines, hjust = 1.05, vjust = 1.5,
               size = 2.6, color = "#5d6d7e", family = "sans") +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 10),
            legend.position = c(0.98, 0.72), legend.justification = c(1, 1))
  }

  FIG6_SCALE_VARIANTS <- list(list(TRUE, "dist_by_platinum_log"),
                              list(FALSE, "dist_by_platinum_raw"))
  FIG6_LABS <- if (plot_non_androgen_distributions) {
    ALL_LABS
  } else {
    intersect(ALL_LABS, ANDROGEN)
  }
  message(sprintf(
    "Figure 6 distributions: %s (%s labs)",
    if (plot_non_androgen_distributions) "androgen + non-androgen" else "androgen only",
    length(FIG6_LABS)
  ))
  # Retired: per-lab distribution figures.
  if (FALSE) for (variant in FIG6_SCALE_VARIANTS) {
    use_log <- variant[[1]]; out_stem <- variant[[2]]
    for (lab in FIG6_LABS) {
      for (landmark in FIG6_LANDMARKS) {
        agg <- load_aggregated_landmark(landmark)
        if (is.null(agg)) {
          p <- ggplot() + annotate("text", x = 0, y = 0, label = "(aggregated CSV not found)",
                                   color = "#7f8c8d") + theme_void() +
            labs(title = sprintf("%s -- landmark %dd", lab, landmark))
        } else {
          mean_col <- resolve_mean_col(names(agg), lab)
          if (is.na(mean_col)) {
            message(sprintf("%s_%s: no %s__mean column at landmark %d -- skipping",
                            out_stem, lab_stem_slug(lab), lab, landmark))
            next
          }
          p <- plot_androgen_dist_by_platinum(agg, lab, mean_col, landmark, use_log)
        }
        save_fig(p, OUT_DIR, sprintf("%s_%s_landmark%d", out_stem, lab_stem_slug(lab), landmark),
                 width = 6.5, height = 5.0)
        if (show) print(p)
      }
    }
  }

  RANDOM_SEED <- 0
  # Exact base-landmark cohort for the group mean +/- CI panel (binned
  # group-level means, not per-patient traces), so there is no rendering reason
  # to subsample. Set to a finite integer to cap the cohort if needed.
  N_GROUP <- Inf

  # is_canonical_lab: CATEGORY_MAP-driven replacement for the old PSA/
  # Testosterone-only is_androgen_lab regex. Keeps the "Prostate specific Ag"
  # raw-OMOP alias mapping to PSA. match_canonical_lab_name returns the
  # canonical CATEGORY_MAP name (or NA) so LAB_GROUP can carry it directly.
  match_canonical_lab_name <- function(name) {
    n <- tolower(name)
    if (grepl("prostate specific ag", n)) return("PSA")
    hit <- ALL_LABS[tolower(ALL_LABS) == n]
    if (length(hit)) hit[1] else NA_character_
  }
  is_canonical_lab <- function(name) !is.na(vapply(name, match_canonical_lab_name, character(1)))

  load_canonical_longitudinal <- function() {
    if (!file.exists(LONGITUDINAL_CSV)) {
      message(sprintf("Figure 7: skipped -- %s not found", LONGITUDINAL_CSV)); return(NULL)
    }
    # Canonical classification/numeric cleanup is cached at arm level.
    df <- tryCatch(cached_canonical_longitudinal(LONGITUDINAL_CSV, labs = LAB_FIGURE_LABS),
                   error = function(e) {
                     message("Figure 7: skipped -- ", conditionMessage(e)); NULL
                   })
    if (is.null(df)) return(NULL)
    needed <- c("LAB_NAME","LAB_VALUE","t_lab","DFCI_MRN")
    missing <- setdiff(needed, names(df))
    if (length(missing)) {
      message(sprintf("Figure 7: skipped -- missing columns %s", paste(missing, collapse=", ")))
      return(NULL)
    }
    message(sprintf("Figure 7 loader: %s rows / %s patients in CSV",
                    format(nrow(df), big.mark=","),
                    format(length(unique(df$DFCI_MRN)), big.mark=",")))
    if (nrow(df) == 0) { message("Figure 7: skipped -- no canonical-lab rows"); return(NULL) }

    # Use the exact base-landmark analysis cohort. A union across the 0/90/180d
    # files mixes different risk sets and makes nominally different cohort
    # definitions converge on nearly the same longitudinal population.
    cohort_mrns <- unique(as.character(patient_df[[ID_COL]]))
    cohort_mrns <- cohort_mrns[!is.na(cohort_mrns) & nzchar(cohort_mrns)]
    n_before <- length(unique(df$DFCI_MRN))
    df <- df %>% filter(as.character(DFCI_MRN) %in% cohort_mrns)
    message(sprintf("  restricted to exact base-landmark cohort: %s -> %s patients, %s rows",
                    format(n_before, big.mark=","),
                    format(length(unique(df$DFCI_MRN)), big.mark=","),
                    format(nrow(df), big.mark=",")))
    if (nrow(df) == 0) {
      message("Figure 7: skipped -- no canonical lab rows for base-landmark patients")
      return(NULL)
    }

    if (nrow(df) == 0) { message("Figure 7: skipped -- all t_lab/LAB_VALUE NaN"); return(NULL) }

    df$t_platinum_rel <- if (all(c("t_platinum","PLATINUM") %in% names(df)))
      suppressWarnings(as.numeric(df$t_platinum)) else NA_real_
    df
  }

  canonical_long_df <- load_canonical_longitudinal()
  if (!is.null(canonical_long_df))
    message(sprintf("Figure 7: %s canonical-lab rows across %s %s patients, %s distinct labs",
                    format(nrow(canonical_long_df), big.mark=","),
                    format(length(unique(canonical_long_df$DFCI_MRN)), big.mark=","), COHORT_DISPLAY,
                    length(unique(canonical_long_df$LAB_GROUP))))

  ## ---- Figure 7b: group mean +/- 95% CI, binned by time from treatment anchor ----
  BIN_WIDTH_DAYS <- 180
  # Trajectory display/modeling window. The separate coverage diagnostics
  # retain five years of pre-ADT history to explain support near the boundary.
  PRE_DAYS  <- 1 * 365.25   # days BEFORE the treatment anchor
  POST_DAYS <- 5 * 365.25   # days AFTER the treatment anchor
  COVERAGE_PRE_DAYS <- 5 * 365.25

  # Anchor every bin on day 0 so no summary mixes pre- and post-ADT labs. The
  # outermost bins absorb the fractional-year remainder.
  anchored_bin_edges <- function(pre_days, post_days, width_days) {
    pre <- -rev(unique(c(seq(0, pre_days, by = width_days), pre_days)))
    post <- unique(c(seq(0, post_days, by = width_days), post_days))
    sort(unique(c(pre, post)))
  }

  # Require ten patients within each stratum/bin so rare subtype tails do not
  # turn a handful of observations into large apparent swings.
  MIN_BIN_PATIENTS <- 10

  # Collapse repeated measurements to one value per patient/stratum/180-day
  # bin. Both the direct summaries and the R-fitted GAMs consume this table, so
  # patients with dense testing do not dominate either visualization.
  patient_bin_trajectory <- function(df, lab_group, stratum_col = "plat_group",
                                     stratum_values = NULL, log_scale = FALSE) {
    if (!is.null(getOption("compass.figure_data_manifest")))
      return(figure_cached_patient_bins(LONGITUDINAL_CSV, df, lab_group,
                                        stratum_col, stratum_values, log_scale))
    sub <- df %>% filter(LAB_GROUP == lab_group, t_rel >= -PRE_DAYS, t_rel <= POST_DAYS)
    if (isTRUE(log_scale)) {
      # Transform measurements before patient/bin aggregation. This makes both
      # the observed summaries and GAM fits genuinely operate in log space,
      # rather than merely applying a logarithmic display axis afterward.
      sub <- sub %>%
        filter(is.finite(LAB_VALUE), LAB_VALUE >= 0) %>%
        mutate(LAB_VALUE = log1p(LAB_VALUE))
    }
    if (!is.null(stratum_values)) {
      required_lookup_cols <- c("DFCI_MRN", "stratum")
      missing_lookup_cols <- setdiff(required_lookup_cols, names(stratum_values))
      if (length(missing_lookup_cols) > 0) {
        stop(sprintf(
          "patient_bin_trajectory: stratum_values is missing required column(s): %s",
          paste(missing_lookup_cols, collapse = ", ")
        ))
      }
      sub <- sub %>%
        mutate(DFCI_MRN = as.character(DFCI_MRN)) %>%
        select(-any_of("stratum")) %>%
        inner_join(stratum_values %>% select(all_of(required_lookup_cols)),
                   by = "DFCI_MRN")
    } else {
      sub <- sub %>% mutate(
        stratum = if (identical(stratum_col, "plat_group"))
          as.integer(coalesce(suppressWarnings(as.numeric(PLATINUM)), 0))
        else as.character(.data[[stratum_col]])
      )
    }
    if (nrow(sub) == 0) return(NULL)
    edges <- anchored_bin_edges(PRE_DAYS, POST_DAYS, BIN_WIDTH_DAYS)
    sub <- sub %>% mutate(
      # right=FALSE makes day 0 the start of the first post-anchor bin.
      t_bin = cut(t_rel, breaks = edges, include.lowest = TRUE, right = FALSE),
      stratum = as.character(stratum)
    )
    sub <- sub %>% filter(!is.na(stratum))
    if (nrow(sub) == 0) return(NULL)
    mids <- (head(edges, -1) + tail(edges, -1)) / 2
    names(mids) <- levels(sub$t_bin)
    sub %>% drop_na(LAB_VALUE, t_bin) %>%
      group_by(DFCI_MRN, t_bin, stratum) %>%
      summarise(LAB_VALUE = mean(LAB_VALUE), .groups = "drop") %>%
      mutate(t_mid = unname(mids[as.character(t_bin)]))
  }

  # Per-bin group mean +/- 95% CI.
  bin_group_ci <- function(df, lab_group, stratum_col = "plat_group",
                           stratum_values = NULL, log_scale = FALSE) {
    patient_bin <- patient_bin_trajectory(
      df, lab_group, stratum_col, stratum_values, log_scale = log_scale
    )
    if (is.null(patient_bin) || nrow(patient_bin) == 0) return(NULL)
    patient_bin %>% group_by(t_bin, t_mid, stratum) %>%
      summarise(n = n_distinct(DFCI_MRN), mean = mean(LAB_VALUE),
                sem = if (n() > 1) sd(LAB_VALUE) / sqrt(n()) else 0, .groups = "drop") %>%
      filter(n >= MIN_BIN_PATIENTS) %>%
      mutate(ci_lo = mean - 1.96 * sem, ci_hi = mean + 1.96 * sem,
             # Log1p values (and raw androgen values) cannot be negative;
             # normal-approximation intervals can otherwise cross below zero.
             ci_lo = if (isTRUE(log_scale) || lab_group %in% ANDROGEN)
               pmax(0, ci_lo) else ci_lo)
  }

  plot_group_ci_panel <- function(df, lab_group, title, stratum_col = "plat_group",
                                  stratum_values = NULL, stratum_legend = NULL,
                                  stratum_colors = NULL, log_scale = FALSE) {
    binned <- bin_group_ci(
      df, lab_group, stratum_col, stratum_values, log_scale = log_scale
    )
    if (is.null(binned) || nrow(binned) == 0)
      return(ggplot() + annotate("text", x = 0, y = 0, label = "(no data)", color = "#7f8c8d") +
               theme_void() + labs(title = title))
    if (is.null(stratum_legend)) stratum_legend <- c(`0` = "Non-platinum", `1` = "Platinum")
    if (is.null(stratum_colors)) stratum_colors <- setNames(PLAT_COLORS, c("0","1"))
    binned <- binned %>% mutate(stratum = factor(stratum)) %>% arrange(t_mid)
    ggplot(binned, aes(t_mid, mean, color = stratum, fill = stratum)) +
      geom_vline(xintercept = 0, color = "#2c3e50", linetype = "dotted", linewidth = 1, alpha = 0.6) +
      geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.2, color = NA) +
      geom_line(linewidth = 0.8) + geom_point(size = 1.6) +
      scale_color_manual(values = stratum_colors, labels = stratum_legend, name = NULL) +
      scale_fill_manual(values = stratum_colors, guide = "none") +
      scale_x_continuous(
        breaks = seq(-PRE_DAYS, POST_DAYS, by = 365.25),
        labels = function(d) sprintf("%g", round(d / 365.25))
      ) +
      labs(x = sprintf("Years from %s (binned, %dd windows)",
                       ANCHOR_LABEL, BIN_WIDTH_DAYS),
           y = sprintf("%s%s (mean +/- 95%% CI)",
                       if (isTRUE(log_scale)) "log1p(" else "",
                       if (isTRUE(log_scale)) paste0(lab_group, ")") else lab_group),
           title = title,
           caption = sprintf("Only bins with at least %d patients in a stratum are shown.",
                             MIN_BIN_PATIENTS)) +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 11),
            plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK, hjust = 0))
  }

  if (is.null(canonical_long_df)) {
    message("Figure 7b: skipped -- no canonical longitudinal data available")
  } else {
    patients <- unique(na.omit(canonical_long_df$DFCI_MRN))
    if (length(patients) > N_GROUP) {
      set.seed(RANDOM_SEED)
      patients <- sample(patients, N_GROUP)
    }
    group_df <- canonical_long_df %>% filter(DFCI_MRN %in% patients)
    labs_present <- intersect(LAB_FIGURE_LABS, unique(group_df$LAB_GROUP))

    # LLM-strata lookups (DFCI_MRN -> stratum value), restricted to this
    # cohort's labeled MRNs; each scheme reports labeled/total when used below.
    llm_lookup <- NULL
    if (!is.null(llm_classifier_labels)) {
      cohort_mrns_long <- unique(as.character(patient_df[[ID_COL]]))
      llm_lookup <- llm_classifier_labels %>%
        filter(as.character(DFCI_MRN) %in% cohort_mrns_long) %>%
        mutate(DFCI_MRN = as.character(DFCI_MRN))
    }

    # ---------------------------------------------------------------------
    # Pre-ADT PSA/testosterone coverage diagnostics. Unlike trajectory plots,
    # these explicitly retain patients with zero measurements in the
    # denominator. This distinguishes sparse testing from a low lab value and
    # makes the thin pre-anchor positive-stratum trajectories interpretable.
    # ---------------------------------------------------------------------
    if (IS_ADT && any(canonical_long_df$LAB_GROUP %in% ANDROGEN)) {
      coverage_labs <- intersect(ANDROGEN, unique(canonical_long_df$LAB_GROUP))
      cohort_mrns_long <- unique(as.character(patient_df[[ID_COL]]))
      cohort_mrns_long <- cohort_mrns_long[
        !is.na(cohort_mrns_long) & nzchar(cohort_mrns_long)
      ]

      platinum_value <- if ("PLATINUM" %in% names(patient_df)) {
        coalesce(suppressWarnings(as.numeric(patient_df$PLATINUM)), 0)
      } else {
        as.numeric(as.character(patient_df[[ID_COL]]) %in% as.character(platinum_set))
      }
      platinum_strata <- tibble(
        DFCI_MRN = as.character(patient_df[[ID_COL]]),
        stratification = "Platinum status",
        stratum = if_else(platinum_value == 1, "Platinum", "Non-platinum")
      ) %>%
        filter(DFCI_MRN %in% cohort_mrns_long) %>%
        distinct(DFCI_MRN, stratification, .keep_all = TRUE)

      coverage_strata <- platinum_strata
      if (!is.null(llm_lookup)) {
        nepc_strata <- llm_lookup %>%
          transmute(
            DFCI_MRN = as.character(DFCI_MRN),
            stratification = "NEPC status",
            stratum = case_when(
              suppressWarnings(as.numeric(has_nepc)) == 0 ~ "has_nepc=0",
              suppressWarnings(as.numeric(has_nepc)) == 1 ~ "has_nepc=1",
              TRUE ~ NA_character_
            )
          ) %>%
          filter(!is.na(stratum)) %>%
          distinct(DFCI_MRN, stratification, .keep_all = TRUE)
        coverage_strata <- bind_rows(coverage_strata, nepc_strata)
      }

      pre_androgen_df <- canonical_long_df %>%
        select(DFCI_MRN, LAB_GROUP, t_rel) %>%
        filter(LAB_GROUP %in% coverage_labs,
               t_rel >= -COVERAGE_PRE_DAYS, t_rel < 0)
      pre_lab_summary <- if (!is.null(getOption("compass.figure_data_manifest"))) {
        figure_prepared_table(LONGITUDINAL_CSV, "coverage_patient") %>%
          filter(DFCI_MRN %in% cohort_mrns_long, LAB_GROUP %in% coverage_labs)
      } else pre_androgen_df %>%
        group_by(DFCI_MRN = as.character(DFCI_MRN), LAB_GROUP) %>%
        summarise(
          n_pre = n(),
          n_pre_180 = sum(t_rel >= -BIN_WIDTH_DAYS),
          .groups = "drop"
        )
      coverage_patient <- crossing(
        coverage_strata,
        LAB_GROUP = coverage_labs
      ) %>%
        left_join(pre_lab_summary, by = c("DFCI_MRN", "LAB_GROUP")) %>%
        mutate(
          n_pre = coalesce(as.integer(n_pre), 0L),
          n_pre_180 = coalesce(as.integer(n_pre_180), 0L)
        )

      coverage_colors <- c(
        "Non-platinum" = PLAT_COLORS[["0"]],
        "Platinum" = PLAT_COLORS[["1"]],
        "has_nepc=0" = KM_PALETTE[[1]],
        "has_nepc=1" = KM_PALETTE[[2]]
      )

      # 1) Direct answer: how many patients have any pre-ADT result, and how
      # many have one close enough to ADT initiation to characterize baseline?
      coverage_any <- coverage_patient %>%
        transmute(
          stratification, stratum, LAB_GROUP,
          `Any in prior 5 years` = n_pre > 0,
          `Within 180 days before ADT` = n_pre_180 > 0
        ) %>%
        pivot_longer(
          cols = c(`Any in prior 5 years`, `Within 180 days before ADT`),
          names_to = "window", values_to = "covered"
        ) %>%
        group_by(stratification, stratum, LAB_GROUP, window) %>%
        summarise(n_patients = n(), n_covered = sum(covered), .groups = "drop") %>%
        mutate(ci = map2(n_covered, n_patients, wilson_ci)) %>%
        unnest_wider(ci)
      p_coverage_any <- ggplot(
        coverage_any,
        aes(stratum, phat, fill = window, group = window)
      ) +
        geom_col(position = position_dodge(width = 0.78), width = 0.68) +
        geom_errorbar(
          aes(ymin = lo, ymax = hi),
          position = position_dodge(width = 0.78), width = 0.18, linewidth = 0.45
        ) +
        geom_text(
          aes(y = hi + .025, label = sprintf("%d/%d", n_covered, n_patients)),
          position = position_dodge(width = 0.78), vjust = -0.45, size = 2.7
        ) +
        facet_grid(LAB_GROUP ~ stratification, scales = "free_x", space = "free_x") +
        scale_y_continuous(labels = percent_format(accuracy = 1), breaks = seq(0, 1, .25)) +
        coord_cartesian(ylim = c(0, 1.12), clip = "off") +
        scale_fill_manual(values = c("#4863a0", "#d9903d"), name = NULL) +
        labs(
          x = NULL, y = "Patients with at least one measurement",
          title = sprintf("Pre-ADT PSA and testosterone coverage — %s cohort", COHORT_DISPLAY),
          subtitle = "Bars use the full stratum denominator; error bars are 95% Wilson intervals.",
          caption = "NEPC panels include classifier-labeled patients only. Day 0 is excluded."
        ) +
        theme_fig() +
        theme(axis.text.x = element_text(angle = 20, hjust = 1),
              legend.position = "top")
      save_fig(
        p_coverage_any, OUT_DIR, "pre_adt_coverage_any_psa_testosterone",
        width = 11, height = 7.2
      )
      if (show) print(p_coverage_any)

      # Retired: patient-level count-burden figure. The patient table remains
      # the input to the retained any/recent coverage diagnostic.
      if (FALSE) {
      # 2) Patient-level test burden, including the zero-count category that
      # disappears from ordinary longitudinal plots.
      coverage_counts <- coverage_patient %>%
        mutate(
          count_group = case_when(
            n_pre == 0 ~ "0",
            n_pre == 1 ~ "1",
            n_pre <= 3 ~ "2-3",
            n_pre <= 9 ~ "4-9",
            TRUE ~ "10+"
          ),
          count_group = factor(count_group, levels = c("0", "1", "2-3", "4-9", "10+"))
        ) %>%
        count(stratification, stratum, LAB_GROUP, count_group, name = "n") %>%
        group_by(stratification, stratum, LAB_GROUP) %>%
        mutate(fraction = n / sum(n)) %>%
        ungroup()
      p_coverage_counts <- ggplot(
        coverage_counts, aes(stratum, fraction, fill = count_group)
      ) +
        geom_col(width = 0.7, color = "white", linewidth = 0.2) +
        facet_grid(LAB_GROUP ~ stratification, scales = "free_x", space = "free_x") +
        scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
        scale_fill_manual(
          values = c("0" = "#d73027", "1" = "#fc8d59", "2-3" = "#fee08b",
                     "4-9" = "#91cf60", "10+" = "#1a9850"),
          name = "Measurements in prior 5 years"
        ) +
        labs(
          x = NULL, y = "Patients",
          title = sprintf("Pre-ADT androgen-lab measurement burden — %s cohort", COHORT_DISPLAY),
          subtitle = "Each bar includes patients with no pre-ADT PSA or testosterone measurement.",
          caption = "NEPC panels include classifier-labeled patients only."
        ) +
        theme_fig() +
        theme(axis.text.x = element_text(angle = 20, hjust = 1),
              legend.position = "top")
      save_fig(
        p_coverage_counts, OUT_DIR, "pre_adt_coverage_counts_psa_testosterone",
        width = 11, height = 7.2
      )
      if (show) print(p_coverage_counts)
      }

      # 3) Availability in each zero-anchored 180-day bin. The denominator is
      # fixed within a stratum, so falling lines reflect thinning observation
      # coverage rather than changing cohort composition.
      pre_edges <- anchored_bin_edges(COVERAGE_PRE_DAYS, 0, BIN_WIDTH_DAYS)
      pre_bin_factor <- cut(
        numeric(0), breaks = pre_edges, include.lowest = TRUE, right = FALSE
      )
      pre_bin_levels <- levels(pre_bin_factor)
      pre_bin_mids <- (head(pre_edges, -1) + tail(pre_edges, -1)) / 2
      names(pre_bin_mids) <- pre_bin_levels
      pre_bin_denominators <- coverage_strata %>%
        count(stratification, stratum, name = "n_patients")
      pre_bin_observations <- if (!is.null(getOption("compass.figure_data_manifest"))) {
        figure_prepared_table(LONGITUDINAL_CSV, "coverage_bins") %>%
          filter(DFCI_MRN %in% cohort_mrns_long, LAB_GROUP %in% coverage_labs) %>%
          transmute(DFCI_MRN, LAB_GROUP, t_bin = pre_bin_levels[bin_id + 1L])
      } else pre_androgen_df %>%
        transmute(
          DFCI_MRN = as.character(DFCI_MRN), LAB_GROUP,
          t_bin = as.character(cut(
            t_rel, breaks = pre_edges, include.lowest = TRUE, right = FALSE
          ))
        )
      pre_bin_counts <- pre_bin_observations %>%
        # Availability is binary per patient/lab/bin. Collapse repeated tests
        # before expanding each patient into platinum and NEPC strata.
        distinct(DFCI_MRN, LAB_GROUP, t_bin) %>%
        inner_join(coverage_strata, by = "DFCI_MRN") %>%
        distinct(DFCI_MRN, LAB_GROUP, t_bin, stratification, stratum) %>%
        count(stratification, stratum, LAB_GROUP, t_bin, name = "n_covered")
      coverage_by_bin <- crossing(
        pre_bin_denominators,
        LAB_GROUP = coverage_labs,
        t_bin = pre_bin_levels
      ) %>%
        left_join(
          pre_bin_counts,
          by = c("stratification", "stratum", "LAB_GROUP", "t_bin")
        ) %>%
        mutate(
          n_covered = coalesce(n_covered, 0L),
          fraction = n_covered / n_patients,
          t_years = unname(pre_bin_mids[t_bin]) / 365.25
        )
      p_coverage_bins <- ggplot(
        coverage_by_bin,
        aes(t_years, fraction, color = stratum, group = stratum)
      ) +
        geom_vline(xintercept = 0, color = "#2c3e50", linetype = "dotted",
                   linewidth = 0.8, alpha = 0.7) +
        geom_line(linewidth = 0.9) +
        geom_point(size = 1.7) +
        facet_grid(LAB_GROUP ~ stratification) +
        scale_color_manual(values = coverage_colors, name = NULL) +
        scale_x_continuous(breaks = seq(-5, 0, by = 1), limits = c(-5, 0)) +
        scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
        labs(
          x = "Years before ADT initiation (180-day bins)",
          y = "Patients with a measurement in bin",
          title = sprintf("Pre-ADT lab availability over time — %s cohort", COHORT_DISPLAY),
          subtitle = "The denominator is fixed to all patients in each platinum or labeled NEPC stratum.",
          caption = "Each patient contributes at most once per bin. Day 0 begins the post-ADT period."
        ) +
        theme_fig() +
        theme(legend.position = "top")
      save_fig(
        p_coverage_bins, OUT_DIR, "pre_adt_coverage_by_bin_psa_testosterone",
        width = 11, height = 7.2
      )
      if (show) print(p_coverage_bins)
    }

    for (lab_group in labs_present) {
      lab_window_mrns <- group_df %>%
        filter(LAB_GROUP == lab_group, t_rel >= -PRE_DAYS, t_rel <= POST_DAYS) %>%
        distinct(DFCI_MRN) %>%
        pull(DFCI_MRN)
      n_pat <- length(lab_window_mrns)
      slug <- lab_stem_slug(lab_group)

      for (log_scale in c(FALSE, TRUE)) {
        scale_suffix <- if (log_scale) "_log" else ""

        # Platinum-status stratum (existing behavior, now over every lab).
        ttl <- sprintf("%s (n=%s)", COHORT_DISPLAY,
                       format(n_pat, big.mark = ","))
        p <- plot_group_ci_panel(group_df, lab_group, ttl, log_scale = log_scale)
        save_fig(p, OUT_DIR, sprintf("longitudinal_platinum_%s%s", slug, scale_suffix),
                 width = 9.5, height = 5.5)
        if (show) print(p)

        if (!is.null(llm_lookup)) {
          for (scheme_name in names(FIGURE_LLM_STRATA)) {
            scheme <- FIGURE_LLM_STRATA[[scheme_name]]
            stratum_values <- llm_lookup %>%
              transmute(DFCI_MRN, stratum = as.character(.data[[scheme$col]]))
            if (!is.null(scheme$labels))
              stratum_values$stratum <- scheme$labels[match(stratum_values$stratum, as.character(scheme$levels))]
            stratum_values <- stratum_values %>% filter(!is.na(stratum))
            n_labeled <- length(intersect(unique(stratum_values$DFCI_MRN),
                                          lab_window_mrns))
            message(sprintf("longitudinal_%s_%s%s: %d / %d cohort patients labeled",
                            scheme_name, slug, scale_suffix, n_labeled, n_pat))
            stratum_legend <- if (!is.null(scheme$labels)) setNames(scheme$labels, scheme$labels) else NULL
            stratum_colors <- setNames(KM_PALETTE[seq_along(scheme$levels)],
                                       if (!is.null(scheme$labels)) scheme$labels else as.character(scheme$levels))
            ttl_s <- sprintf("%s (n=%s/%s labeled)", COHORT_DISPLAY,
                             format(n_labeled, big.mark = ","),
                             format(n_pat, big.mark = ","))
            p_s <- plot_group_ci_panel(
              group_df, lab_group, ttl_s, stratum_col = scheme$col,
              stratum_values = stratum_values, stratum_legend = stratum_legend,
              stratum_colors = stratum_colors, log_scale = log_scale
            )
            save_fig(p_s, OUT_DIR,
                     sprintf("longitudinal_%s_%s%s", scheme_name, slug, scale_suffix),
                     width = 9.5, height = 5.5)
            if (show) print(p_s)
          }
        }
      }
    }
  }

  # -----------------------------------------------------------------------
  # Windowed plotting GAMs, fit directly in R. These are descriptive
  # smooths and are independent of the GAM feature/model pipeline. One patient
  # contributes at most one value per 180-day bin. mgcv selects the smoothing
  # penalty by fast REML; select=TRUE adds shrinkage so unsupported nonlinear
  # structure can collapse toward zero effective degrees of freedom.
  # -----------------------------------------------------------------------
  fit_plotting_gam <- function(patient_bins, lab_group, log_scale = FALSE) {
    if (!requireNamespace("mgcv", quietly = TRUE))
      stop("mgcv is required for R-fitted trajectory GAM figures")
    predictions <- list(); diagnostics <- list()
    for (level in sort(unique(patient_bins$stratum))) {
      d <- patient_bins %>%
        filter(stratum == level, is.finite(LAB_VALUE), is.finite(t_mid)) %>%
        mutate(t_years = t_mid / 365.25)
      n_times <- n_distinct(d$t_mid)
      if (nrow(d) < 20L || n_times < 4L) {
        message(sprintf("  plotting GAM %s / %s skipped: %d patient-bin rows across %d bins",
                        lab_group, level, nrow(d), n_times))
        next
      }
      # k is an upper bound on wiggliness; fREML selects the effective
      # smoothness within that basis and the extra select penalty can shrink it.
      k_use <- min(15L, n_times - 1L)
      fit <- mgcv::bam(
        LAB_VALUE ~ s(t_years, bs = "tp", k = k_use),
        data = d, method = "fREML", select = TRUE, discrete = TRUE
      )
      grid <- tibble(t_years = seq(min(d$t_years), max(d$t_years), length.out = 240L))
      pred <- predict(fit, newdata = grid, se.fit = TRUE, type = "response")
      edf <- tryCatch(sum(summary(fit)$s.table[, "edf"]), error = function(e) NA_real_)
      predictions[[level]] <- grid %>% transmute(
        stratum = level, t_mid = t_years * 365.25,
        fit = as.numeric(pred$fit),
        ci_lo = as.numeric(pred$fit - 1.96 * pred$se.fit),
        ci_hi = as.numeric(pred$fit + 1.96 * pred$se.fit),
        ci_lo = if (isTRUE(log_scale) || lab_group %in% ANDROGEN)
          pmax(0, ci_lo) else ci_lo,
        fit = if (isTRUE(log_scale) || lab_group %in% ANDROGEN)
          pmax(0, fit) else fit
      )
      diagnostics[[level]] <- tibble(
        stratum = level, n_patients = n_distinct(d$DFCI_MRN),
        n_patient_bins = nrow(d), k = k_use, edf = edf,
        reml_score = unname(fit$gcv.ubre)
      )
    }
    list(predictions = bind_rows(predictions), diagnostics = bind_rows(diagnostics))
  }

  plot_group_gam_panel <- function(df, lab_group, title,
                                   stratum_col = "plat_group", stratum_values = NULL,
                                   stratum_legend = NULL, stratum_colors = NULL,
                                   log_scale = FALSE) {
    patient_bins <- patient_bin_trajectory(
      df, lab_group, stratum_col, stratum_values, log_scale = log_scale
    )
    if (is.null(patient_bins) || nrow(patient_bins) == 0)
      return(ggplot() + annotate("text", x = 0, y = 0, label = "(no data)") +
               theme_void() + labs(title = title))
    result <- fit_plotting_gam(patient_bins, lab_group, log_scale = log_scale)
    if (nrow(result$predictions) == 0)
      return(ggplot() + annotate("text", x = 0, y = 0,
                                 label = "(insufficient data for GAM)") +
               theme_void() + labs(title = title))
    observed <- patient_bins %>%
      group_by(t_mid, stratum) %>%
      summarise(n = n_distinct(DFCI_MRN), mean = mean(LAB_VALUE), .groups = "drop") %>%
      filter(n >= MIN_BIN_PATIENTS)
    if (is.null(stratum_legend)) stratum_legend <- c(`0` = "Non-platinum", `1` = "Platinum")
    if (is.null(stratum_colors)) stratum_colors <- setNames(PLAT_COLORS, c("0", "1"))
    diagnostics_text <- result$diagnostics %>%
      mutate(label = sprintf("%s: n=%s, EDF=%.1f", stratum,
                             format(n_patients, big.mark = ","), edf)) %>%
      pull(label) %>% paste(collapse = "   |   ")
    ggplot(result$predictions, aes(t_mid, fit, color = stratum, fill = stratum)) +
      geom_vline(xintercept = 0, color = "#2c3e50", linetype = "dotted",
                 linewidth = 1, alpha = 0.6) +
      geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.18, color = NA) +
      geom_line(linewidth = 1) +
      geom_point(data = observed, aes(t_mid, mean, color = stratum),
                 inherit.aes = FALSE, size = 1.6, alpha = 0.65) +
      scale_color_manual(values = stratum_colors, labels = stratum_legend, name = NULL) +
      scale_fill_manual(values = stratum_colors, guide = "none") +
      scale_x_continuous(
        breaks = seq(-PRE_DAYS, POST_DAYS, by = 365.25),
        labels = function(d) sprintf("%g", round(d / 365.25))
      ) +
      labs(
        x = sprintf("Years from %s", ANCHOR_LABEL),
        y = sprintf("GAM-smoothed %s", if (isTRUE(log_scale))
                    paste0("log1p(", lab_group, ")") else lab_group),
        title = title,
        subtitle = diagnostics_text,
        caption = paste0(
          "R/mgcv GAM on patient-level 180-day bin means",
          if (isTRUE(log_scale)) " after log1p transformation" else "",
          "; smoothness selected by fREML with shrinkage. ",
          "Points are observed bin means (n >= ", MIN_BIN_PATIENTS, ")."
        )
      ) +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 11),
            plot.subtitle = element_text(size = 8.5),
            plot.caption = element_text(size = 7.5, color = COLOR_NEUTRAL_INK))
  }

  if (!plot_gam_trajectories) {
    message("R-fitted GAM trajectories: disabled; skipping")
  } else if (!is.null(canonical_long_df)) {
    for (lab_group in labs_present) {
      slug <- lab_stem_slug(lab_group)
      lab_window_mrns <- group_df %>%
        filter(LAB_GROUP == lab_group, t_rel >= -PRE_DAYS, t_rel <= POST_DAYS) %>%
        distinct(DFCI_MRN) %>%
        pull(DFCI_MRN)
      n_pat <- length(lab_window_mrns)
      for (log_scale in c(FALSE, TRUE)) {
        scale_suffix <- if (log_scale) "_log" else ""
        p_gam <- plot_group_gam_panel(
          group_df, lab_group,
          sprintf("%s (n=%s)", COHORT_DISPLAY,
                  format(n_pat, big.mark = ",")),
          log_scale = log_scale
        )
        save_fig(p_gam, OUT_DIR,
                 sprintf("gam_longitudinal_platinum_%s%s", slug, scale_suffix),
                 width = 9.5, height = 5.5)
        if (show) print(p_gam)

        if (!is.null(llm_lookup)) {
          scheme <- FIGURE_LLM_STRATA[["has_nepc"]]
          nepc_values <- llm_lookup %>%
            transmute(DFCI_MRN, stratum = as.character(.data[[scheme$col]]))
          nepc_values$stratum <- scheme$labels[
            match(nepc_values$stratum, as.character(scheme$levels))
          ]
          nepc_values <- nepc_values %>% filter(!is.na(stratum))
          n_labeled <- length(intersect(unique(nepc_values$DFCI_MRN),
                                        lab_window_mrns))
          p_gam_nepc <- plot_group_gam_panel(
            group_df, lab_group,
            sprintf("%s (n=%s/%s labeled)", COHORT_DISPLAY,
                    format(n_labeled, big.mark = ","),
                    format(n_pat, big.mark = ",")),
            stratum_col = scheme$col, stratum_values = nepc_values,
            stratum_legend = setNames(scheme$labels, scheme$labels),
            stratum_colors = setNames(KM_PALETTE[seq_along(scheme$levels)], scheme$labels),
            log_scale = log_scale
          )
          save_fig(p_gam_nepc, OUT_DIR,
                   sprintf("gam_longitudinal_has_nepc_%s%s", slug, scale_suffix),
                   width = 9.5, height = 5.5)
          if (show) print(p_gam_nepc)
        }
      }
    }
  }

  # -----------------------------------------------------------------------
  # Retired precomputed feature-extraction GAM figures. Kept unreachable for
  # one release so old output schemas remain documented; plotting now uses the
  # one-year-pre/five-year-post R fits above rather than Python/Stage-A curves.
  if (FALSE) {
  # GAM-smoothed trajectories by platinum exposure and classifier NEPC call.
  # gam_trajectory_features.R writes one fitted value per patient x lab x
  # trailing-window grid point. Bands below summarize between-patient
  # variation in those fitted curves; they are not mgcv coefficient intervals.
  # -----------------------------------------------------------------------
  summarize_gam_curves <- function(curves, stratum_lookup) {
    curves %>%
      mutate(DFCI_MRN = as.character(DFCI_MRN)) %>%
      inner_join(
        stratum_lookup %>% mutate(DFCI_MRN = as.character(DFCI_MRN)),
        by = "DFCI_MRN"
      ) %>%
      filter(is.finite(t_lab), is.finite(GAM_FITTED), !is.na(stratum)) %>%
      group_by(t_lab, stratum) %>%
      summarise(
        n_patients = n_distinct(DFCI_MRN),
        mean_fitted = mean(GAM_FITTED),
        sem = if (n() > 1) sd(GAM_FITTED) / sqrt(n()) else 0,
        .groups = "drop"
      ) %>%
      mutate(
        ci_lo = mean_fitted - 1.96 * sem,
        ci_hi = mean_fitted + 1.96 * sem
      )
  }

  plot_gam_curve_panel <- function(summary_df, title, palette, landmark) {
    if (is.null(summary_df) || nrow(summary_df) == 0) {
      return(
        ggplot() +
          annotate("text", x = 0, y = 0, label = "(no labeled GAM curves)", color = "#7f8c8d") +
          theme_void() + labs(title = title)
      )
    }
    summary_df <- summary_df %>%
      mutate(stratum = factor(as.character(stratum), levels = names(palette))) %>%
      arrange(stratum, t_lab)
    n_by_group <- summary_df %>%
      group_by(stratum) %>%
      summarise(n = max(n_patients), .groups = "drop")
    subtitle <- paste(
      sprintf("%s n=%s", n_by_group$stratum, format(n_by_group$n, big.mark = ",")),
      collapse = "   |   "
    )

    ggplot(summary_df, aes(t_lab, mean_fitted, color = stratum, fill = stratum)) +
      geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.18, color = NA) +
      geom_line(linewidth = 1) +
      geom_vline(xintercept = 0, color = "#2c3e50", linetype = "dashed", linewidth = 0.7) +
      { if (landmark != 0) geom_vline(
          xintercept = landmark, color = "#2c3e50", linetype = "dotted", linewidth = 0.7
        ) } +
      scale_color_manual(values = palette, drop = FALSE, name = NULL) +
      scale_fill_manual(values = palette, drop = FALSE, guide = "none") +
      labs(
        x = sprintf("Days from %s", ANCHOR_LABEL),
        y = "Mean GAM-fitted lab value",
        title = title,
        subtitle = subtitle,
        caption = "Ribbon: 95% CI across patient-specific fitted trajectories"
      ) +
      theme_fig() +
      theme(
        plot.title = element_text(face = "bold", size = 11),
        plot.subtitle = element_text(size = 8.5),
        plot.caption = element_text(size = 7.5, color = COLOR_NEUTRAL_INK)
      )
  }

  platinum_palette <- c(
    "Non-platinum" = COLOR_PLATINUM_NEG,
    "Platinum" = COLOR_PLATINUM_POS
  )
  nepc_palette <- c(
    "NEPC-negative" = COLOR_NEUTRAL_INK,
    "NEPC-positive" = "#8e1c2b"
  )

  for (landmark in LANDMARKS) {
    curve_path <- file.path(
      INPUTS_DIR, sprintf("gam_trajectory_curves_landmark%d.csv", landmark)
    )
    agg_path <- file.path(INPUTS_DIR, sprintf("aggregated_landmark%d.csv", landmark))
    if (!file.exists(curve_path)) {
      message(sprintf("GAM trajectories: %s absent -- skipping landmark %d", curve_path, landmark))
      next
    }
    if (!file.exists(agg_path)) {
      message(sprintf("GAM trajectories: %s absent -- skipping landmark %d", agg_path, landmark))
      next
    }

    gam_curves <- read_csv(curve_path, show_col_types = FALSE) %>%
      mutate(
        DFCI_MRN = as.character(DFCI_MRN),
        LAB_GROUP = vapply(LAB_NAME, match_canonical_lab_name, character(1)),
        t_lab = suppressWarnings(as.numeric(t_lab)),
        GAM_FITTED = suppressWarnings(as.numeric(GAM_FITTED))
      ) %>%
      filter(!is.na(LAB_GROUP))
    # t_platinum joins the read so pre-landmark platinum exposure can be
    # dropped here too: the NEPC/AVPC endpoints keep those patients in their
    # aggregated CSVs, and plotting them would label a patient "Platinum" for
    # treatment that preceded the anchor.
    agg_raw <- load_aggregated_landmark(landmark) %>%
      select(any_of(c("DFCI_MRN", "PLATINUM", "t_platinum")))
    agg_platinum <- suppressWarnings(as.numeric(agg_raw$PLATINUM))
    agg_t_platinum <- if ("t_platinum" %in% names(agg_raw)) {
      suppressWarnings(as.numeric(agg_raw$t_platinum))
    } else {
      message(sprintf(
        "  [warn] %s lacks t_platinum; GAM panels cannot exclude pre-landmark platinum exposure",
        basename(agg_path)))
      rep(NA_real_, nrow(agg_raw))
    }
    agg_pre_anchor <- agg_platinum %in% 1 & !is.na(agg_t_platinum) & agg_t_platinum <= 0
    if (any(agg_pre_anchor))
      message(sprintf(
        "  GAM trajectories: dropped %d patient(s) with platinum at or before landmark %d",
        length(unique(agg_raw$DFCI_MRN[agg_pre_anchor])), landmark))
    aggregated_status <- agg_raw[!agg_pre_anchor, , drop = FALSE] %>%
      transmute(
        DFCI_MRN = as.character(DFCI_MRN),
        stratum = if_else(
          suppressWarnings(as.numeric(PLATINUM)) == 1,
          "Platinum", "Non-platinum"
        )
      ) %>%
      distinct(DFCI_MRN, .keep_all = TRUE)

    # Every GAM stratum is restricted to the same platinum-eligible cohort, so
    # the NEPC panel and the platinum panel describe one patient set.
    eligible_gam_mrns <- aggregated_status$DFCI_MRN
    gam_curves <- gam_curves %>% filter(DFCI_MRN %in% eligible_gam_mrns)

    nepc_status <- if (is.null(llm_classifier_labels)) {
      NULL
    } else {
      llm_classifier_labels %>%
        transmute(
          DFCI_MRN = as.character(DFCI_MRN),
          has_nepc = suppressWarnings(as.numeric(has_nepc)),
          stratum = case_when(
            has_nepc == 1 ~ "NEPC-positive",
            has_nepc == 0 ~ "NEPC-negative",
            TRUE ~ NA_character_
          )
        ) %>%
        filter(!is.na(stratum)) %>%
        distinct(DFCI_MRN, .keep_all = TRUE)
    }

    labs_present <- intersect(LAB_FIGURE_LABS, unique(gam_curves$LAB_GROUP))
    for (lab_group in labs_present) {
      lab_curves <- gam_curves %>% filter(LAB_GROUP == lab_group)
      platinum_summary <- summarize_gam_curves(lab_curves, aggregated_status)
      nepc_summary <- if (is.null(nepc_status)) NULL else {
        summarize_gam_curves(lab_curves, nepc_status)
      }
      p_platinum <- plot_gam_curve_panel(
        platinum_summary, "By platinum status", platinum_palette, landmark
      )
      p_nepc <- plot_gam_curve_panel(
        nepc_summary, "By classifier NEPC status", nepc_palette, landmark
      )
      save_fig(
        p_platinum,
        fig_dir("gam_trajectories"),
        sprintf("gam_trajectory_platinum_%s_landmark%d", lab_stem_slug(lab_group), landmark),
        width = 6.5,
        height = 5.5
      )
      save_fig(
        p_nepc,
        fig_dir("gam_trajectories"),
        sprintf("gam_trajectory_has_nepc_%s_landmark%d", lab_stem_slug(lab_group), landmark),
        width = 6.5,
        height = 5.5
      )
      if (show) print(p_platinum)
      if (show) print(p_nepc)
    }
  }
  } # retired precomputed GAM figure block
  }

  if (IS_CANONICAL_ADT && !is.null(cohort_forest_config)) {
    notify_progress("stage", "PSA/testosterone forest across ADT cohorts")
    sys.source(cohort_forest_config$script, envir = environment())
    landmark <- cohort_forest_config$landmark
    forest <- load_cohort_forest(NEPC_PROJ_PATH, cohort_forest_config$cohorts, ENDPOINT, landmark)
    if (nrow(forest)) {
      p_forest <- plot_cohort_forest(forest, ENDPOINT, landmark)
      save_fig(p_forest, OUT_DIR, sprintf("cohort_forest_%s_landmark%d", ENDPOINT, landmark),
               12, max(6, 3 * length(unique(forest$feature_stat))))
      if (show) print(p_forest)
    } else message("Cohort forest: no PSA/testosterone estimates available for ", ENDPOINT)
  }

  ## ---- Unified ADT/LLM metastatic/regex-stage diagnostics ----
  if (IS_CANONICAL_ADT && EMIT_ENDPOINT_INDEPENDENT && !is.null(metastatic_supplement)) {
    sys.source(metastatic_supplement$r_script, envir = environment())
    stems <- metastatic_supplement_stems()
    complete <- vapply(stems, function(stem) {
      base <- file.path(output_dir_for_stem(stem), COHORT_LEAF)
      figure_file_complete(paste0(base, ".png")) &&
        (!save_pdf || figure_file_complete(paste0(base, ".pdf")))
    }, logical(1))
    if (!overwrite && all(complete)) {
      for (stem in stems) notify_progress("panel_skipped", stem)
      message("Metastatic-label supplement: all subpanels complete; skipping preparation")
    } else {
      render_metastatic_supplements(
        metastatic_supplement, patient_df,
        cached_canonical_longitudinal(LONGITUDINAL_CSV, labs = LAB_FIGURE_LABS),
        save_panel = function(plot, stem, width, height)
          save_fig(plot, fig_dir("metastatic_labels"), stem, width, height),
        notify = notify_progress, show = show
      )
    }
  }

  ## ---- Supplement -- localized-adjuvant vs metastatic ADT-intent strata ----
  ## Reads the CSVs written by COMPASS/survival_analysis/adt_intent_comparison.py.
  ## That module is read-only over the `local_runs_adt_{localized,metastatic}*`
  ## trees; nothing here refits a model. These global comparisons are emitted
  ## only by the canonical ADT cell, since restricted cohorts do not alter them.
  ADT_INTENT_DIR <- file.path(NEPC_PROJ_PATH, "survival_analysis",
                              "adt_intent_comparison")
  adt_intent_table <- function(filename) {
    path <- file.path(ADT_INTENT_DIR, filename)
    if (!file.exists(path)) return(NULL)
    frame <- suppressWarnings(read_csv(path, show_col_types = FALSE))
    # The module writes empty frames deliberately, to distinguish "no
    # comparable runs" from "this stage never ran". Treat both as nothing
    # to plot, but only the missing file is worth a message.
    if (nrow(frame) == 0) return(NULL)
    frame
  }

  if (IS_CANONICAL_ADT && !plot_adt_intent_supplement)
    message("ADT-intent supplement: disabled; skipping")

  if (IS_CANONICAL_ADT && plot_adt_intent_supplement) {
    intent_counts <- adt_intent_table("adt_intent_cohort_counts.csv")
    intent_overlap <- adt_intent_table("adt_intent_cohort_overlap.csv")
    intent_association <- adt_intent_table("adt_intent_univariate_heterogeneity.csv")
    intent_association_summary <- adt_intent_table("adt_intent_univariate_summary.csv")
    intent_performance <- adt_intent_table("adt_intent_performance_delta.csv")

    # Disjointness is already a hard assertion in adt_intent_comparison.py, so
    # this is a read-side audit rather than a gate: it re-states the check in
    # the figure log, where a stale CSV from an older run would otherwise be
    # plotted without comment.
    if (!is.null(intent_overlap)) {
      overlapping <- intent_overlap %>% filter(.data$n_overlap > 0)
      if (nrow(overlapping) > 0) {
        warning(sprintf(
          "ADT-intent strata overlap in %d endpoint/landmark cell(s); the comparison CSVs are stale",
          nrow(overlapping)
        ))
      }
    }

    if (is.null(intent_counts) && is.null(intent_association) &&
        is.null(intent_performance)) {
      message(sprintf(
        "ADT-intent supplement: no comparison tables under %s; run adt_intent_comparison.py first",
        ADT_INTENT_DIR
      ))
    }

    # Keyed by the `cohort` values adt_intent_comparison.py writes, i.e. the
    # COHORT_SPECS keys -- "metastatic_adt", not the "metastatic" column suffix.
    INTENT_PALETTE <- c(localized = "#0b6ba8", metastatic_adt = "#c1272d")
    INTENT_LABELS <- c(localized = "Localized-adjuvant", metastatic_adt = "Metastatic")
    landmark_label <- function(landmark) {
      ifelse(landmark == 0, "0-day", sprintf("+%d-day", landmark))
    }

    ## Panel 1 -- modelled cohort and event counts per stratum x landmark.
    if (!is.null(intent_counts)) {
      counts_endpoint <- intent_counts %>%
        filter(.data$endpoint == ENDPOINT, .data$status == "ok")
      if (nrow(counts_endpoint) > 0) {
        counts_plot <- counts_endpoint %>%
          mutate(
            cohort_label = unname(INTENT_LABELS[.data$cohort]),
            landmark_lab = factor(landmark_label(.data$landmark_days),
                                  levels = landmark_label(sort(unique(.data$landmark_days)))),
            sparse = as.logical(.data$sparse_events)
          )
        p_counts <- ggplot(counts_plot,
                           aes(x = .data$landmark_lab, y = .data$n_events,
                               fill = .data$cohort)) +
          geom_col(position = position_dodge(width = 0.75), width = 0.68) +
          geom_text(
            aes(label = sprintf("%d/%d", .data$n_events, .data$n_patients)),
            position = position_dodge(width = 0.75),
            vjust = -0.35, size = 3
          ) +
          # Sparse cells are annotated rather than dropped: an underpowered
          # stratum is a result the reader needs to see, not a missing bar.
          geom_point(
            data = ~ dplyr::filter(.x, .data$sparse),
            aes(y = 0), position = position_dodge(width = 0.75),
            shape = 8, size = 2, color = "#1a1a1a", show.legend = FALSE
          ) +
          scale_fill_manual(values = INTENT_PALETTE, labels = INTENT_LABELS,
                            name = "ADT intent") +
          scale_y_continuous(expand = expansion(mult = c(0, 0.14))) +
          labs(
            title = sprintf("%s events by ADT-intent stratum", toupper(ENDPOINT)),
            subtitle = paste0(
              "Labels are events/patients in the modelled landmark cohort. ",
              "* marks fewer than 50 events."
            ),
            x = "Landmark", y = "Incident events"
          ) +
          theme_fig()
        save_fig(p_counts, fig_dir("supplement_adt_intent"),
                 sprintf("adt_intent_cohort_counts_%s", ENDPOINT),
                 width = 8, height = 5)
        if (show) print(p_counts)
      }
    }

    ## Panel 2 -- univariate log-HR concordance between the two strata.
    ## This is the panel notebook 10 rendered inline and never saved.
    if (!is.null(intent_association)) {
      concordance <- intent_association %>%
        filter(.data$endpoint == ENDPOINT) %>%
        filter(!is.na(.data$coef_feature_localized),
               !is.na(.data$coef_feature_metastatic))
      if (nrow(concordance) > 0) {
        concordance <- concordance %>%
          mutate(
            fdr_either = (!is.na(.data$q_value_localized) & .data$q_value_localized < 0.05) |
                         (!is.na(.data$q_value_metastatic) & .data$q_value_metastatic < 0.05),
            landmark_lab = factor(landmark_label(.data$landmark_days),
                                  levels = landmark_label(sort(unique(.data$landmark_days))))
          )
        limit <- max(
          abs(c(concordance$coef_feature_localized, concordance$coef_feature_metastatic)),
          na.rm = TRUE
        ) * 1.08
        limit <- max(limit, 0.1)
        # Per-facet concordance statistics come from the summary table rather
        # than being recomputed here, so the figure and the CSV cannot disagree.
        concordance_labels <- NULL
        if (!is.null(intent_association_summary)) {
          concordance_labels <- intent_association_summary %>%
            filter(.data$endpoint == ENDPOINT) %>%
            mutate(
              landmark_lab = factor(landmark_label(.data$landmark_days),
                                    levels = levels(concordance$landmark_lab)),
              label = sprintf("rho = %.2f\n%d/%d same direction",
                              .data$spearman_log_hr, .data$n_same_direction,
                              .data$n_features_both)
            ) %>%
            filter(!is.na(.data$landmark_lab))
        }

        p_concordance <- ggplot(
            concordance,
            aes(x = .data$coef_feature_localized, y = .data$coef_feature_metastatic)
          ) +
          geom_hline(yintercept = 0, linewidth = 0.4, color = "#b0b0b0") +
          geom_vline(xintercept = 0, linewidth = 0.4, color = "#b0b0b0") +
          geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                      linewidth = 0.5, color = "#1a1a1a", alpha = 0.6) +
          geom_point(aes(color = .data$fdr_either, size = .data$fdr_either,
                         alpha = .data$fdr_either)) +
          scale_color_manual(values = c(`FALSE` = "#8a8a8a", `TRUE` = "#6a3d9a"),
                             labels = c(`FALSE` = "q >= 0.05", `TRUE` = "q < 0.05"),
                             name = "FDR in either stratum") +
          scale_size_manual(values = c(`FALSE` = 1.1, `TRUE` = 2.1), guide = "none") +
          scale_alpha_manual(values = c(`FALSE` = 0.35, `TRUE` = 0.85), guide = "none") +
          (if (is.null(concordance_labels) || nrow(concordance_labels) == 0) NULL else
             geom_text(data = concordance_labels,
                       aes(x = -limit * 0.95, y = limit * 0.95, label = .data$label),
                       hjust = 0, vjust = 1, size = 3, color = "#52514e",
                       inherit.aes = FALSE)) +
          coord_equal(xlim = c(-limit, limit), ylim = c(-limit, limit)) +
          facet_wrap(~ landmark_lab) +
          labs(
            title = sprintf("%s univariate effect concordance across ADT-intent strata",
                            toupper(ENDPOINT)),
            subtitle = paste0(
              "Each point is one feature. The dashed line is equality; ",
              "distance from it is the between-stratum log-HR difference."
            ),
            x = "Localized-adjuvant log HR per SD",
            y = "Metastatic log HR per SD"
          ) +
          theme_fig()
        save_fig(p_concordance, fig_dir("supplement_adt_intent"),
                 sprintf("adt_intent_loghr_concordance_%s", ENDPOINT),
                 width = 11, height = 4.6)
        if (show) print(p_concordance)

        ## Panel 3 -- the strongest between-stratum heterogeneity signals.
        heterogeneity <- concordance %>%
          filter(!is.na(.data$q_heterogeneity)) %>%
          arrange(.data$q_heterogeneity) %>%
          group_by(.data$landmark_days) %>%
          slice_head(n = TOP_N) %>%
          ungroup()
        if (nrow(heterogeneity) > 0) {
          # Within-facet ordering without tidytext: suffix the label with the
          # facet, order on the shared effect scale, then strip the suffix
          # back off at the axis. Keeps the dependency list unchanged.
          p_heterogeneity <- heterogeneity %>%
            mutate(
              feature_facet = paste(.data$feature, .data$landmark_lab, sep = "___"),
              feature_facet = forcats::fct_reorder(.data$feature_facet,
                                                   .data$delta_log_hr_met_minus_loc)
            ) %>%
            ggplot(aes(x = .data$delta_log_hr_met_minus_loc, y = .data$feature_facet,
                       fill = .data$q_heterogeneity < 0.05)) +
            geom_vline(xintercept = 0, linewidth = 0.4, color = "#b0b0b0") +
            geom_col(width = 0.7) +
            scale_y_discrete(labels = function(x) sub("___.*$", "", x)) +
            scale_fill_manual(values = c(`FALSE` = "#c4c4c4", `TRUE` = "#6a3d9a"),
                              labels = c(`FALSE` = "q >= 0.05", `TRUE` = "q < 0.05"),
                              name = "Heterogeneity FDR") +
            facet_wrap(~ landmark_lab, scales = "free_y") +
            labs(
              title = sprintf("%s: largest between-stratum effect differences",
                              toupper(ENDPOINT)),
              subtitle = paste0(
                "Metastatic minus localized log HR per SD. Positive values are ",
                "stronger in the metastatic stratum. Approximate Wald test; ",
                "BH-adjusted within each landmark."
              ),
              x = "Delta log HR (metastatic - localized)", y = NULL
            ) +
            theme_fig()
          save_fig(p_heterogeneity, fig_dir("supplement_adt_intent"),
                   sprintf("adt_intent_heterogeneity_%s", ENDPOINT),
                   width = 12, height = 6.2)
          if (show) print(p_heterogeneity)
        }
      }
    }

    ## Panel 4 -- held-out performance deltas, metastatic minus localized.
    if (!is.null(intent_performance)) {
      performance_endpoint <- intent_performance %>%
        filter(.data$endpoint == ENDPOINT)
      delta_cols <- intersect(
        c("delta_c_index_met_minus_loc", "delta_mean_auc_t_met_minus_loc",
          "delta_integrated_brier_met_minus_loc"),
        names(performance_endpoint)
      )
      if (nrow(performance_endpoint) > 0 && length(delta_cols) > 0) {
        METRIC_LABELS <- c(
          delta_c_index_met_minus_loc = "C-index",
          delta_mean_auc_t_met_minus_loc = "Mean AUC(t)",
          delta_integrated_brier_met_minus_loc = "Integrated Brier"
        )
        performance_long <- performance_endpoint %>%
          select(all_of(c("landmark", "model", "config", delta_cols))) %>%
          tidyr::pivot_longer(all_of(delta_cols), names_to = "metric",
                              values_to = "delta") %>%
          filter(!is.na(.data$delta)) %>%
          mutate(
            metric = factor(unname(METRIC_LABELS[.data$metric]),
                            levels = unname(METRIC_LABELS[delta_cols])),
            model_config = paste(.data$model, .data$config, sep = " / "),
            landmark_lab = factor(landmark_label(.data$landmark),
                                  levels = landmark_label(sort(unique(.data$landmark))))
          )
        if (nrow(performance_long) > 0) {
          p_performance <- ggplot(
              performance_long,
              aes(x = .data$delta, y = .data$model_config, fill = .data$delta > 0)
            ) +
            geom_vline(xintercept = 0, linewidth = 0.4, color = "#b0b0b0") +
            geom_col(width = 0.65) +
            scale_fill_manual(values = c(`FALSE` = "#0b6ba8", `TRUE` = "#c1272d"),
                              guide = "none") +
            facet_grid(rows = vars(.data$landmark_lab), cols = vars(.data$metric),
                       scales = "free_x") +
            labs(
              title = sprintf("%s held-out performance: metastatic minus localized",
                              toupper(ENDPOINT)),
              subtitle = paste0(
                "Positive favors metastatic for C-index and AUC(t); negative\n",
                "favors metastatic for integrated Brier. Test cohorts differ, ",
                "so these are descriptive contrasts, not paired tests."
              ),
              x = "Delta (metastatic - localized)", y = NULL
            ) +
            theme_fig()
          save_fig(p_performance, fig_dir("supplement_adt_intent"),
                   sprintf("adt_intent_performance_delta_%s", ENDPOINT),
                   width = 11, height = 6)
          if (show) print(p_performance)
        }
      }
    }
  }
}
