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

# Notebook adapter: Rscript cohort_forest_figures.R input.csv output_root landmark dpi overwrite
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
