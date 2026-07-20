# Per-cohort figure-generation pipeline for the COMPASS R notebook, extracted
# from COMPASS_generate_figures_R.ipynb so the notebook can call it once per
# cohort in the same R session instead of re-executing itself via Rscript per
# cohort.
#
# Required packages (install once on the cluster):
#   install.packages(c("tidyverse", "survival", "survminer", "broom",
#                       "ggrepel", "patchwork", "jsonlite", "scales"))
#   # optional: install.packages(c("ggpattern", "ragg"))
#   #   ggpattern -> striped baseline bars in Fig 4a
#   #   ragg      -> crisper high-DPI PNG device (falls back to default if absent)
suppressPackageStartupMessages({
  library(tidyverse)   # dplyr, ggplot2, readr, tidyr, purrr, stringr, forcats
  library(survival)
  library(survminer)
  library(broom)
  library(ggrepel)
  library(patchwork)
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
      plot.margin        = margin(6, 8, 6, 6)
    )
}
theme_set(theme_fig())

# Colorblind-safe categorical palette (fixed assignment), shared by the
# Figure 2 (LLM subtype / platinum enrichment) panels:
#   blue -> platinum+, orange -> platinum-. Kept consistent everywhere.
COLOR_PLATINUM_POS <- "#2a78d6"   # blue
COLOR_PLATINUM_NEG <- "#eb6834"   # orange
COLOR_NEUTRAL_INK  <- "#52514e"   # secondary ink, for annotations/text only

# ---- helpers ported from the Python notebook ----

# binary_metrics: confusion-matrix-derived classification metrics for a 0/1
# (or logical) truth/pred pair. Returns a named list (mirrors the pd.Series).
binary_metrics <- function(y_true, y_pred) {
  y_true <- as.integer(y_true); y_pred <- as.integer(y_pred)
  tp <- sum(y_true == 1 & y_pred == 1)
  tn <- sum(y_true == 0 & y_pred == 0)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  safe <- function(num, den) if (den > 0) num / den else 0
  list(
    Accuracy  = (tp + tn) / (tp + tn + fp + fn),
    Precision = safe(tp, tp + fp),
    Recall    = safe(tp, tp + fn),
    TPR       = safe(tp, tp + fn),
    FPR       = safe(fp, fp + tn),
    TNR       = safe(tn, tn + fp),
    FNR       = safe(fn, fn + tp),
    TP = tp, FP = fp, TN = tn, FN = fn
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
NS_COLOR <- "#d5d8dc"

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
    list(lab_name = parts[1], feature_stat = paste(parts[-1], collapse = "__"))
  } else {
    list(lab_name = name, feature_stat = "")
  }
}


COHORTS <- c(
  "icd_arpi",
  "vte_arpi",
  "icd_or_vte_arpi",
  "icd_allow_other_primaries_arpi",
  "vte_allow_other_primaries_arpi",
  "icd_or_vte_allow_other_primaries_arpi"
)

# Render the full COMPASS figure set for one cohort arm. Mirrors the body of
# COMPASS_generate_figures_R.ipynb's per-cohort cells (Figures 1-7 + Table 1)
# so the notebook can call this once per cohort in the same R session instead
# of re-executing itself via Rscript.
generate_figures <- function(cohort, nepc_proj_path, fig_root, cohorts = COHORTS, show = FALSE) {
  NEPC_PROJ_PATH <- nepc_proj_path
  COHORT <- cohort
  if (!COHORT %in% cohorts)
    stop(sprintf("Unknown cohort=%s; expected one of %s",
                 COHORT, paste(cohorts, collapse = ", ")))
  message(sprintf("Generating figures for cohort: %s", COHORT))

  BASE <- file.path(NEPC_PROJ_PATH, "survival_analysis", paste0("local_runs_", COHORT))
  LONGITUDINAL_CSV <- file.path(NEPC_PROJ_PATH, "longitudinal_prediction_data.csv")
  INPUTS_DIR <- file.path(NEPC_PROJ_PATH, "survival_analysis", paste0("prediction_inputs_", COHORT))

  FIG_ROOT <- fig_root
  # Figure-first, panel-second layout:
  # FIG_ROOT/<figure>/<plot-stem>/R/<cohort>_<plot-stem>.png
  FIG_LANG <- "R"
  figure_group <- function(plot_stem) {
    if (startsWith(plot_stem, "figure1") || startsWith(plot_stem, "table1")) return("figure1")
    if (startsWith(plot_stem, "figure2")) return("figure2")
    if (startsWith(plot_stem, "figure3")) return("figure3")
    if (startsWith(plot_stem, "figure4")) return("figure4")
    if (startsWith(plot_stem, "km_")) return("figure5")
    if (startsWith(plot_stem, "androgen_dist_")) return("figure6")
    if (startsWith(plot_stem, "androgen_longitudinal_")) return("figure7")
    stop(sprintf("Unmapped figure output stem: %s", plot_stem))
  }
  # Compatibility value passed by existing save_fig call sites; actual output
  # routing is determined from each exact `stem` inside save_fig/write_table1.
  fig_dir <- function(plot_stem) {
    file.path(FIG_ROOT, plot_stem, FIG_LANG)
  }

  LANDMARKS <- c(0, 90)
  TOP_N <- 15

  LLM_LABEL_PATH <- file.path(NEPC_PROJ_PATH, "LLM_NEPC_labels")
  manual_annotations <- read_csv(file.path(LLM_LABEL_PATH, "baca_lab_annotations.csv"),
                                 show_col_types = FALSE)
  nepc_annotations <- read_tsv(file.path(LLM_LABEL_PATH, "LLM_v3_labels.tsv"),
                               show_col_types = FALSE)
  platinum_mrns <- read_csv(file.path(NEPC_PROJ_PATH, "mrn_lists/platinum_MRN_list.csv"),
                            show_col_types = FALSE)
  platinum_set <- unique(platinum_mrns$DFCI_MRN)
  nepc_annotations <- nepc_annotations %>%
    mutate(is_platinum = DFCI_MRN %in% platinum_set)
  cat(sprintf("nepc_annotations: %s rows (%s platinum+, %s platinum-)\n",
              format(nrow(nepc_annotations), big.mark = ","),
              format(sum(nepc_annotations$is_platinum), big.mark = ","),
              format(sum(!nepc_annotations$is_platinum), big.mark = ",")))

  # save_fig: write one high-resolution PNG to the figure/panel directory.
  save_fig <- function(plot, out_dir, stem, width, height) {
    # `out_dir` is retained for call-site compatibility. Outputs are grouped by
    # exact plot stem so every cohort version of a panel sits in one directory.
    panel_dir <- file.path(FIG_ROOT, figure_group(stem), stem, FIG_LANG)
    dir.create(panel_dir, recursive = TRUE, showWarnings = FALSE)
    output_stem <- paste0(COHORT, "_", stem)

    png_out <- file.path(panel_dir, paste0(output_stem, ".png"))
    if (HAS_RAGG) {
      ggsave(png_out, plot = plot, width = width, height = height, units = "in",
             dpi = SAVE_DPI, bg = "white", device = ragg::agg_png)
    } else {
      ggsave(png_out, plot = plot, width = width, height = height, units = "in",
             dpi = SAVE_DPI, bg = "white", type = "cairo")
    }
    message("wrote ", png_out)

    invisible(plot)
  }

  COHORT_LABEL <- "PROFILE"

  load_profile_patient_and_labs <- function(path, id_col = "DFCI_MRN") {
    df <- read_csv(path, show_col_types = FALSE, guess_max = 100000)
    date_cols <- c("DIAGNOSIS_DATE", "TREATMENT_ANCHOR_DATE", "PLATINUM_DATE",
                   "LAST_CONTACT_DATE", "LAB_DATE", "FIRST_RECORD_DATE")
    for (col in intersect(date_cols, names(df)))
      df[[col]] <- suppressWarnings(as.Date(df[[col]]))
    if (all(c("DIAGNOSIS_DATE", "TREATMENT_ANCHOR_DATE") %in% names(df)))
      df$t_dx_to_anchor <- as.numeric(df$TREATMENT_ANCHOR_DATE - df$DIAGNOSIS_DATE)

    patient_level <- c(id_col, "AGE_AT_TREATMENTSTART", "FIRST_RECORD_DATE", "DIAGNOSIS_DATE",
                       "TREATMENT_ANCHOR_DATE", "LAST_CONTACT_DATE", "DEATH",
                       "PLATINUM_MEDICATION", "PLATINUM_DATE", "PLATINUM",
                       "t_diagnosis", "t_first_treatment", "t_platinum",
                       "t_last_contact", "t_death", "t_dx_to_anchor")
    pat_cols <- intersect(patient_level, names(df))
    patient_df <- df %>% select(all_of(pat_cols)) %>%
      distinct(.data[[id_col]], .keep_all = TRUE)
    lab_cols <- intersect(c(id_col, "LAB_NAME", "LAB_VALUE", "LAB_UNIT", "LAB_DATE", "t_lab"),
                          names(df))
    labs_df <- df %>% filter(!is.na(LAB_NAME)) %>% select(all_of(lab_cols))
    list(patient_df = patient_df, labs_df = labs_df)
  }

  restrict_to_base_landmark_cohort <- function(patient_df, labs_df, inputs_dir,
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
    lab_ids <- suppressWarnings(as.numeric(labs_df[[id_col]]))
    list(patient_df = patient_df[patient_ids %in% cohort_ids, , drop = FALSE],
         labs_df = labs_df[lab_ids %in% cohort_ids, , drop = FALSE])
  }

  record_span_days <- function(labs_df, id_col, date_col = "LAB_DATE") {
    labs_df %>% group_by(.data[[id_col]]) %>%
      summarise(record_span_days = as.numeric(max(.data[[date_col]], na.rm = TRUE) -
                                              min(.data[[date_col]], na.rm = TRUE)),
                .groups = "drop") %>% pull(record_span_days)
  }

  load_attrition <- function(inputs_dir) {
    landmark_path <- file.path(inputs_dir, "landmark_attrition.json")
    if (!file.exists(landmark_path))
      stop(landmark_path, " not found -- re-run build_prediction_inputs.py")
    fromJSON(landmark_path, simplifyVector = FALSE)
  }

  render_consort_panel <- function(attrition) {
    downstream <- attrition[["downstream_cohort_filters"]]
    steps <- list(
      c("Selected MRN arm", downstream[["n_before_downstream_cohort_filters"]]),
      c(sprintf("+ >=%s PSA records", downstream[["min_psa_count"]]),
        downstream[["n_after_psa_count_filter"]]),
      c("- PARPi-exposed excluded", downstream[["n_after_parpi_exclusion"]])
    )
    elig <- attrition[["eligible_by_landmark"]]
    for (k in sort(as.integer(names(elig)))) {
      sign <- if (k > 0) "+" else ""
      steps[[length(steps) + 1]] <- c(sprintf("Eligible at landmark %s%sd", sign, k),
                                      elig[[as.character(k)]])
    }
    steps[[length(steps) + 1]] <- c("Common cohort (all landmarks)",
                                     attrition[["n_common_across_landmarks"]])
    n_steps <- length(steps)
    df <- tibble(i = seq_len(n_steps) - 1,
                 label = vapply(steps, `[`, character(1), 1),
                 n = as.numeric(vapply(steps, `[`, character(1), 2))) %>%
      mutate(ytop = n_steps - i, ycen = ytop - 0.5,
             text = sprintf("%s\nn = %s", label, format(n, big.mark = ",")),
             xmin = 0.07, xmax = 0.93, ymin = ycen - 0.31, ymax = ycen + 0.31)
    arrows <- df %>% filter(i < n_steps - 1) %>%
      mutate(y = ymin, yend = df$ymax[match(i + 1, df$i)], x = 0.5, xend = 0.5)
    split_sizes <- attrition[["split_sizes"]]
    footer <- paste(sprintf("%s: n=%s", names(split_sizes),
                            format(as.numeric(unlist(split_sizes)), big.mark = ",")),
                    collapse = "  ·  ")
    ggplot(df) +
      geom_rect(aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
                fill = "#eef1f5", color = "#5d6d7e", linewidth = 0.4) +
      geom_text(aes(x = 0.5, y = ycen, label = text), size = 3.1, lineheight = 0.9) +
      geom_segment(data = arrows, aes(x = x, xend = xend, y = y, yend = yend),
                   arrow = arrow(length = unit(0.12, "cm"), type = "closed"),
                   color = "#5d6d7e", linewidth = 0.45) +
      annotate("text", x = 0.5, y = n_steps + 0.25,
               label = paste0("Train/valid/test split — ", footer),
               size = 2.9, fontface = "italic", color = "#5d6d7e") +
      labs(title = "Cohort attrition (CONSORT)") + xlim(0, 1) + ylim(0, n_steps + 0.6) +
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
    lab <- sprintf("%s %s (n=%s)", COHORT_LABEL, COHORT, format(nrow(d), big.mark = ","))
    gg <- ggsurvplot(fit, data = d, conf.int = TRUE, censor = FALSE,
                     palette = "#1f3a93", legend = "none",
                     xlab = "Days from treatment anchor",
                     ylab = "Platinum-free probability", title = "Platinum-free survival",
                     ggtheme = theme_fig())
    gg$plot + coord_cartesian(ylim = c(0, 1.02)) +
      annotate("text", x = Inf, y = Inf, label = lab, hjust = 1.05, vjust = 1.5, size = 3)
  }

  overlay_hist <- function(series, xlab, title, bins = 50) {
    v <- suppressWarnings(as.numeric(series)); v <- v[is.finite(v)]
    if (!length(v)) return(ggplot() + annotate("text", x = 0, y = 0, label = "(no data)") +
                             theme_void() + labs(title = title))
    lab <- sprintf("%s (n=%s)", COHORT_LABEL, format(length(v), big.mark = ","))
    ggplot(tibble(v = v), aes(v)) +
      geom_histogram(aes(y = after_stat(density)), bins = bins,
                     fill = "#1f3a93", color = "white", linewidth = 0.15, alpha = 0.75) +
      labs(x = xlab, y = "Density", title = title) +
      annotate("text", x = Inf, y = Inf, label = lab, hjust = 1.05, vjust = 1.5, size = 3)
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
    add("Median follow-up from treatment anchor, days (IQR)",
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
    panel_dir <- file.path(FIG_ROOT, figure_group(stem), stem, FIG_LANG)
    dir.create(panel_dir, recursive = TRUE, showWarnings = FALSE)
    out_base <- file.path(panel_dir, paste0(COHORT, "_", stem))
    csv <- paste0(out_base, ".csv"); md_p <- paste0(out_base, ".md")
    write_csv(table1, csv)
    writeLines(to_markdown_table(table1), md_p)
    c(csv, md_p)
  }

  OUT_DIR <- fig_dir("figure1_cohort")
  ID_COL <- "DFCI_MRN"

  message(sprintf("Loading cohort-specific attrition counts from %s ...", INPUTS_DIR))
  attrition <- load_attrition(INPUTS_DIR)
  message(sprintf("Loading %s ...", LONGITUDINAL_CSV))
  split <- load_profile_patient_and_labs(LONGITUDINAL_CSV, id_col = ID_COL)
  base_landmark <- min(as.integer(names(attrition[["eligible_by_landmark"]])))
  split <- restrict_to_base_landmark_cohort(split$patient_df, split$labs_df, INPUTS_DIR,
                                            ID_COL, base_landmark)
  patient_df <- split$patient_df; labs_df <- split$labs_df
  expected_n <- as.integer(attrition[["eligible_by_landmark"]][[as.character(base_landmark)]])
  stopifnot(nrow(patient_df) == expected_n)
  message(sprintf("  selected base-landmark cohort: patients=%s  labs=%s",
                  format(nrow(patient_df), big.mark = ","), format(nrow(labs_df), big.mark = ",")))

  n_landmarks <- length(attrition[["eligible_by_landmark"]])
  pA <- render_consort_panel(attrition)
  save_fig(pA, OUT_DIR, "figure1a_consort", 7.5, 0.62 * (n_landmarks + 8) + 1)
  pB <- render_km_panel(patient_df)
  save_fig(pB, OUT_DIR, "figure1b_km", 6.5, 4.8)

  span_series <- record_span_days(labs_df, ID_COL)
  km_inputs <- platinum_km_inputs(patient_df)
  event_rows <- km_inputs$row_id[km_inputs$event == 1]
  timing_panels <- list(
    overlay_hist(span_series, "Record span (days)", "Per-patient lab record span", 50),
    overlay_hist(patient_df$t_dx_to_anchor, "Days: diagnosis → treatment anchor",
                 "Diagnosis → treatment anchor", 50),
    overlay_hist(patient_df$t_platinum[event_rows],
                 "Days from treatment anchor to platinum (events only)",
                 "Time to platinum (event patients only)", 40)
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

  fig1 <- (pA | pB) / wrap_plots(timing_panels, nrow = 1) +
    plot_layout(heights = c(1.25, 1)) +
    plot_annotation(title = sprintf("Figure 1 — COMPASS cohort overview (%s)", COHORT),
                    tag_levels = "A")
  save_fig(fig1, OUT_DIR, "figure1_cohort_overview", 16, 10)
  if (show) print(fig1)

  ## Panel A -- LLM validation (NEPC-vs-rest classifier)
  OUT_DIR <- fig_dir("figure2_llm")

  drop_cols <- function(df, cols) df %>% select(-any_of(cols))

  merged_results <- manual_annotations %>%
    drop_cols(c("pathology_details", "manual_platinum_reason")) %>%
    inner_join(
      nepc_annotations %>% drop_cols(c("has_nepc","has_avpc","has_molecular_avpc",
                                       "avpc_criteria","visceral_met_pattern","num_snippets")),
      by = "DFCI_MRN"
    ) %>%
    mutate(
      manual_NEPC = simplified_manual_platinum_reason %in% c("nepc","squamous_transformation"),
      LLM_NEPC    = primary_label == "nepc"
    )

  cat(sprintf("merged_results: %s rows, %s manual-NEPC positive\n",
              format(nrow(merged_results), big.mark = ","),
              format(sum(merged_results$manual_NEPC), big.mark = ",")))

  # manual = truth, LLM = pred (correct argument order for binary_metrics).
  metrics <- binary_metrics(merged_results$manual_NEPC, merged_results$LLM_NEPC)

  # Sanity check: confusion-matrix counts must reconstruct N and the manual-NEPC total.
  stopifnot(metrics$TP + metrics$FP + metrics$TN + metrics$FN == nrow(merged_results))
  stopifnot(metrics$TP + metrics$FN == sum(merged_results$manual_NEPC))

  if (show) print(as_tibble(metrics))

  # Annotated 2x2 confusion matrix, LLM (rows) vs manual truth (cols).
  render_confusion_panel <- function(metrics) {
    cm <- tibble(
      truth = factor(c("Non-NEPC","NEPC","Non-NEPC","NEPC"),
                     levels = c("Non-NEPC","NEPC")),
      pred  = factor(c("Non-NEPC","Non-NEPC","NEPC","NEPC"),
                     levels = c("NEPC","Non-NEPC")),   # reversed so NEPC row is on top
      n     = c(metrics$TN, metrics$FP, metrics$FN, metrics$TP)
    )
    thresh <- max(cm$n) / 2
    ggplot(cm, aes(truth, pred, fill = n)) +
      geom_tile(color = "white", linewidth = 1) +
      geom_text(aes(label = format(n, big.mark = ","),
                    color = n > thresh), size = 5, fontface = "bold") +
      scale_fill_gradient(low = "#eaf1fb", high = COLOR_PLATINUM_POS, guide = "none") +
      scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#0b0b0b"), guide = "none") +
      labs(x = "Manual annotation (truth)", y = "LLM label (prediction)",
           title = "Panel A1 \u2014 confusion matrix") +
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
      labs(x = NULL, y = "Metric value", title = "Panel A2 \u2014 classifier metrics") +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 11))
  }

  n_total       <- nrow(merged_results)
  n_nepc_manual <- sum(merged_results$manual_NEPC)
  caption_a <- sprintf("N = %s chart-reviewed patients; %s manual-NEPC positive.",
                       format(n_total, big.mark = ","),
                       format(n_nepc_manual, big.mark = ","))

  pA1 <- render_confusion_panel(metrics) +
    labs(caption = caption_a) + theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK))
  save_fig(pA1, OUT_DIR, "figure2a1_confusion_matrix", width = 4.2, height = 4.2)
  if (show) print(pA1)

  pA2 <- render_metric_bar_panel(metrics) +
    labs(caption = caption_a) + theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK))
  save_fig(pA2, OUT_DIR, "figure2a2_metric_bar", width = 5.0, height = 4.2)
  if (show) print(pA2)

  ## Panel B -- subtype landscape (descriptive, 4-class)
  # platinum_positive_labels := is_platinum == TRUE  (n=200);  status "positive"
  # platinum_negative_labels := is_platinum == FALSE (n=1682); status "negative"
  count_labels <- function(df) {
    df %>% count(primary_label, name = "count") %>%
      mutate(frac = count / sum(count))
  }
  platinum_positive_labels <- nepc_annotations %>% filter(is_platinum)  %>% count_labels() %>%
    mutate(platinum_status = "positive")
  platinum_negative_labels <- nepc_annotations %>% filter(!is_platinum) %>% count_labels() %>%
    mutate(platinum_status = "negative")

  label_distributions <- bind_rows(platinum_positive_labels, platinum_negative_labels)

  # --- Sanity check: guards against the swap regressing silently. ---
  expected_pos_counts <- c(avpc = 118, nepc = 36, conventional = 30, biomarker = 16)
  expected_neg_counts <- c(conventional = 1090, avpc = 456, biomarker = 72, nepc = 64)
  named_counts <- function(df) { v <- df$count; names(v) <- df$primary_label; v }
  pos_counts <- named_counts(platinum_positive_labels)
  neg_counts <- named_counts(platinum_negative_labels)

  validate_expected_counts <- function(observed, expected, label) {
    aligned <- observed[names(expected)]
    matches <- !anyNA(aligned) &&
      identical(names(aligned), names(expected)) &&
      all(as.numeric(aligned) == as.numeric(expected))
    if (!matches) {
      fmt <- function(x) paste(sprintf("%s=%s", names(x), as.numeric(x)), collapse = ", ")
      stop(sprintf("%s counts do not match expected {%s}; got {%s}",
                   label, fmt(expected), fmt(observed)), call. = FALSE)
    }
    invisible(TRUE)
  }
  validate_expected_counts(pos_counts, expected_pos_counts, "platinum-positive")
  validate_expected_counts(neg_counts, expected_neg_counts, "platinum-negative")
  stopifnot(sum(platinum_positive_labels$count) == 200)
  stopifnot(sum(platinum_negative_labels$count) == 1682)

  if (show) print(label_distributions)

  # Fixed class order groups the two aggressive classes together for readability.
  CLASS_ORDER <- c("conventional", "avpc", "nepc", "biomarker")
  n_pos <- sum(platinum_positive_labels$count)
  n_neg <- sum(platinum_negative_labels$count)

  render_landscape_panel <- function(title = "Panel B \u2014 subtype landscape by platinum status (descriptive)") {
    d <- label_distributions %>%
      mutate(primary_label   = factor(primary_label, levels = CLASS_ORDER),
             platinum_status = factor(platinum_status, levels = c("positive","negative")))
    ggplot(d, aes(primary_label, frac, fill = platinum_status)) +
      geom_col(position = position_dodge(width = 0.8), width = 0.72) +
      scale_fill_manual(
        values = c(positive = COLOR_PLATINUM_POS, negative = COLOR_PLATINUM_NEG),
        labels = c(sprintf("Platinum+ (n=%s)", format(n_pos, big.mark = ",")),
                   sprintf("Platinum- (n=%s)", format(n_neg, big.mark = ","))),
        name = NULL) +
      coord_cartesian(ylim = c(0, 1.0)) +
      labs(x = "LLM primary label", y = "Fraction within platinum group", title = title) +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 11),
            legend.position = c(0.98, 0.98), legend.justification = c(1, 1))
  }

  caption_b <- paste0("Fractions computed within each platinum group separately; groups ",
                      "differ greatly in size and are not a random sample of the same ",
                      "population -- see Panel C for the base-rate-robust enrichment statistic.")
  pB <- render_landscape_panel() +
    labs(caption = caption_b) +
    theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK, hjust = 0.5))
  save_fig(pB, OUT_DIR, "figure2b_subtype_landscape", width = 7.5, height = 5)
  if (show) print(pB)

  df <- nepc_annotations

  # Exclude 'biomarker' (and anything outside the 3 core classes) from the contrast.
  n_before <- nrow(df)
  df <- df %>% filter(primary_label %in% c("conventional","avpc","nepc"))
  n_excluded <- n_before - nrow(df)
  cat(sprintf("Excluded %s rows outside {conventional, avpc, nepc} (e.g. 'biomarker') from the enrichment contrast; %s rows remain.\n",
              format(n_excluded, big.mark = ","), format(nrow(df), big.mark = ",")))

  df <- df %>% mutate(aggressive = primary_label %in% c("avpc","nepc"))

  # 2x2 table: rows = aggressive/conventional, cols = platinum TRUE/FALSE.
  ct <- matrix(
    c(sum(df$aggressive   & df$is_platinum),  sum(df$aggressive   & !df$is_platinum),
      sum(!df$aggressive  & df$is_platinum),  sum(!df$aggressive  & !df$is_platinum)),
    nrow = 2, byrow = TRUE,
    dimnames = list(c("aggressive","conventional"), c("platinum+","platinum-"))
  )
  print(ct)

  # Sanity check against the plan's known margins.
  stopifnot(ct["aggressive","platinum+"]   == 154)
  stopifnot(ct["conventional","platinum+"] == 30)
  stopifnot(ct["aggressive","platinum-"]   == 520)
  stopifnot(ct["conventional","platinum-"] == 1090)

  ft <- fisher.test(ct, alternative = "greater")
  OR <- unname(ft$estimate); p_value <- ft$p.value
  cat(sprintf("\nOdds ratio (aggressive vs conventional, platinum+ vs platinum-): OR = %.2f\n", OR))
  cat(sprintf("Fisher's exact p-value (one-sided, OR > 1): p = %.3g\n", p_value))

  stopifnot(OR > 1)

  # P(platinum+ | aggressive) vs P(platinum+ | conventional), with Wilson CIs.
  n_aggressive   <- sum(ct["aggressive", ])
  n_conventional <- sum(ct["conventional", ])
  platinum_given_aggressive   <- ct["aggressive",   "platinum+"]
  platinum_given_conventional <- ct["conventional", "platinum+"]

  w_agg  <- wilson_ci(platinum_given_aggressive,   n_aggressive)    # c(phat, lo, hi)
  w_conv <- wilson_ci(platinum_given_conventional, n_conventional)
  p_agg <- w_agg[1];  lo_agg <- w_agg[2];  hi_agg <- w_agg[3]
  p_conv <- w_conv[1]; lo_conv <- w_conv[2]; hi_conv <- w_conv[3]

  cat(sprintf("P(platinum+ | aggressive)   = %d/%d = %.1f%%  (95%% CI %.1f%%-%.1f%%)\n",
              platinum_given_aggressive, n_aggressive, 100*p_agg, 100*lo_agg, 100*hi_agg))
  cat(sprintf("P(platinum+ | conventional) = %d/%d = %.1f%%  (95%% CI %.1f%%-%.1f%%)\n",
              platinum_given_conventional, n_conventional, 100*p_conv, 100*lo_conv, 100*hi_conv))

  render_enrichment_panel <- function() {
    d <- tibble(
      group = factor(c("Aggressive\n(AVPC + NEPC)", "Conventional"),
                     levels = c("Aggressive\n(AVPC + NEPC)", "Conventional")),
      prop  = c(p_agg, p_conv), lo = c(lo_agg, lo_conv), hi = c(hi_agg, hi_conv),
      n     = c(n_aggressive, n_conventional),
      k     = c(platinum_given_aggressive, platinum_given_conventional)
    )
    ymax <- max(hi_agg, hi_conv) * 1.35
    ggplot(d, aes(group, prop, fill = group)) +
      geom_col(width = 0.55) +
      geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.18,
                    color = COLOR_NEUTRAL_INK, linewidth = 0.7) +
      geom_text(aes(y = hi + ymax * 0.04,
                    label = sprintf("%.1f%%\n(%d/%d)", 100*prop, k, n)),
                size = 3.3, lineheight = 0.9) +
      scale_fill_manual(values = c(COLOR_PLATINUM_POS, "#9a9890"), guide = "none") +
      annotate("text", x = 1.5, y = ymax * 0.97,
               label = sprintf("OR = %.1f, Fisher's exact p = %.1e", OR, p_value),
               fontface = "bold", size = 3.7, color = COLOR_NEUTRAL_INK) +
      scale_y_continuous(labels = scales::percent, limits = c(0, ymax)) +
      labs(x = NULL, y = "P(platinum+ | subtype group)",
           title = "Panel C \u2014 platinum enrichment among aggressive variants") +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 11))
  }

  caption_c <- sprintf(paste0("Excludes 'biomarker' primary_label (%s rows) from the ",
    "aggressive/conventional contrast. Error bars are 95%% Wilson score intervals. OR and ",
    "p from Fisher's exact test (one-sided, OR > 1) on the 2x2 aggressive/conventional x ",
    "platinum+/- table."), format(n_excluded, big.mark = ","))
  pC <- render_enrichment_panel() +
    labs(caption = caption_c) +
    theme(plot.caption = element_text(size = 8, color = COLOR_NEUTRAL_INK, hjust = 0.5))
  save_fig(pC, OUT_DIR, "figure2c_enrichment", width = 6, height = 5.5)
  if (show) print(pC)

  # Layout mirrors the Python subplot_mosaic [[A1, A2, B],[C, C, B]]:
  # left column stacks A1/A2 over C; right column is B spanning full height.
  left  <- (render_confusion_panel(metrics) + render_metric_bar_panel(metrics)) /
           render_enrichment_panel()
  right <- render_landscape_panel("Panel B \u2014 subtype landscape (descriptive)")

  full_caption <- sprintf(paste0(
    "(A) NEPC-vs-rest classifier, LLM vs Baca-lab manual annotation (N=%s, %s manual-NEPC+). ",
    "(B) Descriptive subtype landscape, platinum+ (n=%s) vs platinum- (n=%s); not itself the ",
    "enrichment claim. (C) Platinum+ rate among aggressive (AVPC+NEPC) vs conventional ",
    "patients (OR=%.1f, Fisher p=%.1e)."),
    format(n_total, big.mark = ","), format(n_nepc_manual, big.mark = ","),
    format(n_pos, big.mark = ","), format(n_neg, big.mark = ","), OR, p_value)

  fig2 <- (left | right) +
    plot_layout(widths = c(2, 1.3)) +
    plot_annotation(
      title = paste0("Figure 2 \u2014 LLM-extracted prostate subtypes validate against ",
                     "manual annotation and reveal platinum enrichment for aggressive variants"),
      caption = str_wrap(full_caption, 130),
      theme = theme(plot.title = element_text(face = "bold", size = 13),
                    plot.caption = element_text(size = 8.5, color = COLOR_NEUTRAL_INK)))
  save_fig(fig2, OUT_DIR, "figure2_llm_subtype_platinum", width = 15, height = 9)
  if (show) print(fig2)

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
    androgen_rows <- sig %>% filter(category == "Androgen axis")
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

    n_tested <- nrow(sub); n_sig <- sum(sub$sig)
    breakdown <- sub %>% filter(sig) %>% count(category)
    short <- c("Androgen axis"="Androgen","CBC"="CBC","LFT"="LFT","CMP"="CMP","Vitals"="Vitals")
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
      geom_point(data = ns, aes(coef_feature, y), size = 1.6, color = NS_COLOR, alpha = 0.45) +
      geom_point(data = sigd %>% filter(!capped),
                 aes(coef_feature, y, color = category, size = is_hero),
                 shape = 21, fill = NA, stroke = 0.9) +
      geom_point(data = sigd %>% filter(!capped),
                 aes(coef_feature, y, fill = category, size = is_hero),
                 shape = 21, color = "white", stroke = 0.6, alpha = 0.92) +
      geom_point(data = sigd %>% filter(capped),
                 aes(coef_feature, y, fill = category), shape = 24,
                 size = 3.4, color = "white", stroke = 0.6, alpha = 0.92) +
      ggrepel::geom_text_repel(
        data = lab_df, aes(coef_feature, pmin(neglog10p, Y_MAX_CAP), label = label, color = category),
        size = 3, fontface = "bold", segment.color = "#95a5a6", segment.size = 0.3,
        max.overlaps = Inf, min.segment.length = 0, show.legend = FALSE) +
      scale_color_manual(values = CATEGORY_COLORS, breaks = LEGEND_ORDER, name = NULL) +
      scale_fill_manual(values = CATEGORY_COLORS, breaks = LEGEND_ORDER, name = NULL) +
      scale_size_manual(values = c(`TRUE` = 3.2, `FALSE` = 2.1), guide = "none") +
      coord_cartesian(xlim = PANEL_XLIM, ylim = c(-0.2, max(y_max * 1.10, 5))) +
      labs(x = "Cox log HR per SD", y = expression(-log[10](p)), title = title) +
      annotate("text", x = PANEL_XLIM[2], y = 0, label = footer, hjust = 1, vjust = 0,
               size = 2.9, color = "#5d6d7e", family = "mono") +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 12.5),
            legend.position = c(0.02, 0.98), legend.justification = c(0, 1))
    p
  }

  OUT_DIR <- fig_dir("figure3_univariate")

  load_uni <- function(landmark) {
    p <- file.path(BASE, "cox", sprintf("landmark_%s", landmark), "both",
                   "cox_agg_univariate_nobs_adjusted.csv")
    read_csv(p, show_col_types = FALSE) %>% mutate(landmark_days = landmark)
  }

  uni <- map_dfr(LANDMARKS, load_uni) %>%
    filter(endpoint == "platinum") %>%
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

  # One solo volcano panel per landmark.
  panels <- list(list(0, "0 days"), list(90, "+90 days"))
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
    save_fig(p, OUT_DIR, sprintf("figure3_univariate_platinum_landmark%d", lm),
             width = 7.5, height = 6)
    if (show) print(p)
  }

  OUT_DIR <- fig_dir("figure4_multivariate")
  HAS_GGPATTERN <- requireNamespace("ggpattern", quietly = TRUE)

  # (mean_auc_t, c_index) for the platinum endpoint, or (NA, NA) if missing.
  read_metrics <- function(path, auc_col, cindex_col) {
    if (!file.exists(path)) return(c(auc = NA_real_, cindex = NA_real_))
    df <- read_csv(path, show_col_types = FALSE)
    row <- df %>% filter(endpoint == "platinum")
    if (nrow(row) == 0) return(c(auc = NA_real_, cindex = NA_real_))
    c(auc = as.numeric(row[[auc_col]][1]), cindex = as.numeric(row[[cindex_col]][1]))
  }
  cox_labs     <- function(lm) read_metrics(file.path(BASE,"cox",sprintf("landmark_%s",lm),"both","cox_agg_multivariable_metrics.csv"), "test_mean_auc_t","test_c_index")
  cox_baseline <- function(lm) read_metrics(file.path(BASE,"cox",sprintf("landmark_%s",lm),"baseline","cox_agg_baseline_metrics.csv"), "test_mean_auc_t","test_c_index")
  xgb_labs     <- function(lm) read_metrics(file.path(BASE,"xgboost",sprintf("landmark_%s",lm),"both","landmark_xgboost_metrics.csv"), "mean_auc_t","c_index")
  xgb_baseline <- function(lm) read_metrics(file.path(BASE,"xgboost",sprintf("landmark_%s",lm),"baseline","landmark_xgboost_baseline_metrics.csv"), "mean_auc_t","c_index")

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
      geom_text(aes(fill = name, label = ifelse(is.finite(value), sprintf("%.3f", value), "")),
                position = position_dodge(width = 0.85), vjust = -0.4, size = 2.5,
                show.legend = FALSE) +
      geom_hline(yintercept = 0.5, color = "grey", linetype = "dotted", linewidth = 0.9) +
      scale_fill_manual(values = SERIES_COLORS, name = NULL, drop = FALSE) +
      coord_cartesian(ylim = c(0.45, ymax)) +
      labs(x = NULL, y = ylabel) +
      theme_fig() +
      theme(legend.position = if (show_legend) c(0.99, 0.99) else "none",
            legend.justification = c(1, 1),
            panel.grid.major.x = element_blank())
  }

  disc_panels <- list(
    list("figure4a_discrimination_auc_platinum",    "auc",    "Test Mean AUC(t)"),
    list("figure4a_discrimination_cindex_platinum", "cindex", "Test C-index")
  )
  for (dp in disc_panels) {
    p <- render_discrimination_panel(dp[[2]], dp[[3]], show_legend = TRUE) +
      labs(title = sprintf("Labs vs. age baseline \u2014 platinum (%s)", dp[[3]])) +
      theme(plot.title = element_text(face = "bold", size = 11))
    save_fig(p, OUT_DIR, dp[[1]], width = 7.5, height = 5.5)
    if (show) print(p)
  }

  OUT_DIR <- fig_dir("figure4_multivariate")

  load_cox_coefs <- function(landmark) {
    p <- file.path(BASE, "cox", sprintf("landmark_%s", landmark), "both", "cox_agg_multivariable.csv")
    read_csv(p, show_col_types = FALSE) %>%
      filter(endpoint == "platinum") %>%
      filter(!coalesce(as.logical(is_age_covariate), FALSE)) %>%
      filter(coalesce(coef, 0) != 0)
  }

  load_xgb_importance <- function(landmark) {
    p <- file.path(BASE, "xgboost", sprintf("landmark_%s", landmark), "both", "landmark_xgboost_feature_importance.csv")
    df <- read_csv(p, show_col_types = FALSE) %>%
      filter(endpoint == "platinum") %>%
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
    df <- df %>% mutate(category = vapply(lab_name, assign_category, character(1)))
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
            axis.text.y = element_text(size = 8.5))
    if (kind == "cox") p <- p + geom_vline(xintercept = 0, color = "black", linewidth = 0.5)
    p
  }

  IMPORTANCE_MODEL_ROWS <- list(
    list("cox", "Elastic-Net Cox",  load_cox_coefs),
    list("xgb", "XGBoost Survival", load_xgb_importance)
  )

  importance_panels <- list()
  for (row in IMPORTANCE_MODEL_ROWS) {
    kind <- row[[1]]; model_name <- row[[2]]; loader <- row[[3]]
    for (lm in LANDMARKS) {
      df <- tryCatch(loader(lm), error = function(e) tibble())
      sign <- if (lm > 0) "+" else ""
      title <- sprintf("%s  \u00b7  %s%d days", model_name, sign, lm)
      p <- render_importance_panel(df, kind, title)
      save_fig(p, OUT_DIR, sprintf("figure4b_importance_platinum_%s_landmark%d", kind, lm),
               width = 7.5, height = 5.5)
      if (show) print(p)
      importance_panels[[sprintf("%s_%d", kind, lm)]] <- p
    }
  }

  disc_row <- render_discrimination_panel("auc", "Test Mean AUC(t)", show_legend = TRUE) |
              render_discrimination_panel("cindex", "Test C-index", show_legend = FALSE)

  imp_grid <- (importance_panels[["cox_0"]] | importance_panels[["cox_90"]]) /
              (importance_panels[["xgb_0"]] | importance_panels[["xgb_90"]])

  fig4 <- disc_row / imp_grid +
    plot_layout(heights = c(1, 2)) +
    plot_annotation(
      title = "Figure 4 \u2014 Multivariate model performance (labs vs. age baseline)",
      theme = theme(plot.title = element_text(face = "bold", size = 13)))
  save_fig(fig4, fig_dir("figure4_multivariate"), "figure4_multivariate_performance",
           width = 15, height = 13)
  if (show) print(fig4)

  OUT_DIR <- fig_dir("androgen_supplements")
  ANDROGEN_LABS <- c("Testosterone", "PSA")

  # Case-insensitive match for a `<lab>__mean` column (mirrors find_col in the
  # Python COMPASS diagnostics).
  find_mean_col <- function(columns, substr) {
    hit <- columns[grepl(tolower(substr), tolower(columns), fixed = TRUE) &
                   endsWith(columns, "__mean")]
    if (length(hit)) hit[1] else NA_character_
  }
  resolve_androgen_columns <- function(columns) {
    psa <- find_mean_col(columns, "PSA")
    if (is.na(psa)) psa <- find_mean_col(columns, "Prostate specific Ag")
    list(Testosterone = find_mean_col(columns, "Testosterone"), PSA = psa)
  }

  # 3-way tertile split into Low (T1) / Mid (T2) / High (T3); NA where undefined.
  tertile_split <- function(df, col) {
    vals <- suppressWarnings(as.numeric(df[[col]]))
    out <- rep(NA_character_, length(vals))
    qs <- tryCatch(quantile(vals, c(1/3, 2/3), na.rm = TRUE, names = FALSE),
                   error = function(e) NULL)
    if (is.null(qs) || qs[1] == qs[2]) return(out)
    lvl <- cut(vals, breaks = c(-Inf, qs[1], qs[2], Inf),
               labels = c("Low (T1)", "Mid (T2)", "High (T3)"))
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
            legend.position = c(0.02, 0.02), legend.justification = c(0, 0))
  }

  load_aggregated_landmark <- function(landmark) {
    path <- file.path(INPUTS_DIR, sprintf("aggregated_landmark%s.csv", landmark))
    if (!file.exists(path)) {
      message(sprintf("skipped landmark %s -- %s not found", landmark, path)); return(NULL)
    }
    read_csv(path, show_col_types = FALSE, guess_max = 100000)
  }

  FIG5_LANDMARKS <- c(0, 90)

  plot_km_androgen_tertile <- function(agg, lab, mean_col, landmark) {
    ttl <- sprintf("%s tertile -- landmark %s%dd", lab, ifelse(landmark > 0, "+", ""), landmark)
    blank <- function(msg) ggplot() + annotate("text", x = 0, y = 0, label = msg, color = "#7f8c8d") +
      theme_void() + labs(title = ttl) + theme(plot.title = element_text(face = "bold", size = 10))
    if (is.na(mean_col) || !all(c("t_platinum","PLATINUM") %in% names(agg))) return(blank("(no data)"))

    d <- agg
    d$stratum <- tertile_split(d, mean_col)
    d <- d %>% filter(!is.na(stratum), !is.na(t_platinum), !is.na(PLATINUM),
                      stratum %in% c("Low (T1)","High (T3)"))
    if (nrow(d) == 0 || length(unique(d$stratum)) < 2)
      return(blank("(insufficient data after tertile split)"))

    km <- platinum_km_inputs(d)
    d <- d[km$row_id, , drop = FALSE]
    d$km_time <- km$time; d$km_event <- km$event
    survival_by_label <- split(d %>% transmute(time = km_time, event = km_event), d$stratum)
    ttl2 <- sprintf("%s tertile (%s) -- landmark %s%dd", lab, mean_col,
                    ifelse(landmark > 0, "+", ""), landmark)
    p <- overlay_km(survival_by_label, "Days from landmark", "Platinum-free probability", ttl2)

    n_low <- sum(d$stratum == "Low (T1)"); n_high <- sum(d$stratum == "High (T3)")
    ev_low <- sum(d$km_event[d$stratum == "Low (T1)"])
    ev_high <- sum(d$km_event[d$stratum == "High (T3)"])
    ann <- sprintf("Low: n=%d, events=%d\nHigh: n=%d, events=%d", n_low, ev_low, n_high, ev_high)
    sd <- survdiff(Surv(km_time, km_event) ~ stratum, data = d)
    pval <- 1 - pchisq(sd$chisq, length(sd$n) - 1)
    ann <- sprintf("%s\nlog-rank p = %.3g", ann, pval)
    p + annotate("text", x = Inf, y = 0, label = ann, hjust = 1.05, vjust = 0,
                 size = 2.6, color = "#5d6d7e", family = "mono")
  }

  for (lab in ANDROGEN_LABS) {
    for (landmark in FIG5_LANDMARKS) {
      agg <- load_aggregated_landmark(landmark)
      if (is.null(agg)) {
        p <- ggplot() + annotate("text", x = 0, y = 0, label = "(aggregated CSV not found)",
                                 color = "#7f8c8d") + theme_void() +
          labs(title = sprintf("%s tertile -- landmark %dd", lab, landmark))
      } else {
        mean_cols <- resolve_androgen_columns(names(agg))
        p <- plot_km_androgen_tertile(agg, lab, mean_cols[[lab]], landmark)
      }
      save_fig(p, OUT_DIR, sprintf("km_%s_tertile_platinum_landmark%d", tolower(lab), landmark),
               width = 6.5, height = 5.5)
      if (show) print(p)
    }
  }

  FIG6_LANDMARKS <- c(0, 90)
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
               size = 2.6, color = "#5d6d7e", family = "mono") +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 10),
            legend.position = c(0.98, 0.72), legend.justification = c(1, 1))
  }

  FIG6_SCALE_VARIANTS <- list(list(TRUE, "androgen_dist_by_platinum_log"),
                              list(FALSE, "androgen_dist_by_platinum_raw"))
  for (variant in FIG6_SCALE_VARIANTS) {
    use_log <- variant[[1]]; out_stem <- variant[[2]]
    for (lab in ANDROGEN_LABS) {
      for (landmark in FIG6_LANDMARKS) {
        agg <- load_aggregated_landmark(landmark)
        if (is.null(agg)) {
          p <- ggplot() + annotate("text", x = 0, y = 0, label = "(aggregated CSV not found)",
                                   color = "#7f8c8d") + theme_void() +
            labs(title = sprintf("%s -- landmark %dd", lab, landmark))
        } else {
          mean_cols <- resolve_androgen_columns(names(agg))
          p <- plot_androgen_dist_by_platinum(agg, lab, mean_cols[[lab]], landmark, use_log)
        }
        save_fig(p, OUT_DIR, sprintf("%s_%s_landmark%d", out_stem, tolower(lab), landmark),
                 width = 6.5, height = 5.0)
        if (show) print(p)
      }
    }
  }

  RANDOM_SEED <- 0
  # Full ARPI cohort for the group mean +/- CI panel (binned group-level means, not
  # per-patient traces), so there is no rendering reason to subsample -- using
  # everyone maximizes per-bin N and tightens the CIs. Set to a finite integer to
  # cap (the random subsample below only triggers when length(patients) > N_GROUP).
  N_GROUP <- Inf

  is_androgen_lab <- function(name) {
    n <- tolower(name)
    grepl("testosterone", n) | grepl("psa", n) | grepl("prostate specific ag", n)
  }

  # Union of patient IDs across the aggregated landmark CSVs, so Figure 7 traces
  # describe exactly the modeled population. Returns NULL if none are found (caller
  # then skips the figure). Mirrors _aggregated_landmark_mrns() in
  # the Python figures notebook.
  aggregated_landmark_mrns <- function(id_col = "DFCI_MRN") {
    mrns <- character(0); found <- character(0)
    for (lm in LANDMARKS) {
      p <- file.path(INPUTS_DIR, sprintf("aggregated_landmark%s.csv", lm))
      if (!file.exists(p)) next
      hdr <- names(read_csv(p, n_max = 0, show_col_types = FALSE))
      col <- if (id_col %in% hdr) id_col else if ("DFCI_MRN" %in% hdr) "DFCI_MRN" else NA
      if (is.na(col)) {
        message(sprintf("  [warn] %s has no %s/DFCI_MRN column; skipping", basename(p), id_col)); next
      }
      ids <- read_csv(p, col_select = all_of(col), show_col_types = FALSE)[[1]]
      ids <- unique(ids[!is.na(ids)])
      mrns <- union(mrns, as.character(ids))
      found <- c(found, sprintf("%s (%d)", basename(p), length(ids)))
    }
    if (!length(found)) return(NULL)
    message(sprintf("  aggregated-landmark cohort: %s unique MRNs from %s",
                    format(length(mrns), big.mark=","), paste(found, collapse=", ")))
    mrns
  }

  load_androgen_longitudinal <- function() {
    if (!file.exists(LONGITUDINAL_CSV)) {
      message(sprintf("Figure 7: skipped -- %s not found", LONGITUDINAL_CSV)); return(NULL)
    }
    df <- read_csv(LONGITUDINAL_CSV, show_col_types = FALSE, guess_max = 100000)
    needed <- c("LAB_NAME","LAB_VALUE","t_lab","DFCI_MRN")
    missing <- setdiff(needed, names(df))
    if (length(missing)) {
      message(sprintf("Figure 7: skipped -- missing columns %s", paste(missing, collapse=", ")))
      return(NULL)
    }
    message(sprintf("Figure 7 loader: %s rows / %s patients in CSV",
                    format(nrow(df), big.mark=","),
                    format(length(unique(df$DFCI_MRN)), big.mark=",")))
    df <- df %>% filter(is_androgen_lab(LAB_NAME))
    if (nrow(df) == 0) { message("Figure 7: skipped -- no Testosterone/PSA rows"); return(NULL) }

    # Restrict to patients present in the aggregated landmark CSVs so the trajectory
    # panels describe exactly the modeled (aggregated-landmark) population. If no
    # aggregated CSV is found, leave the cohort unrestricted.
    landmark_mrns <- aggregated_landmark_mrns("DFCI_MRN")
    if (!is.null(landmark_mrns)) {
      n_before <- length(unique(df$DFCI_MRN))
      df <- df %>% filter(as.character(DFCI_MRN) %in% landmark_mrns)
      message(sprintf("  restricted to aggregated-landmark cohort: %s -> %s patients, %s rows",
                      format(n_before, big.mark=","),
                      format(length(unique(df$DFCI_MRN)), big.mark=","),
                      format(nrow(df), big.mark=",")))
      if (nrow(df) == 0) {
        message("Figure 7: skipped -- no androgen rows for aggregated-landmark patients"); return(NULL)
      }
    } else {
      message(sprintf("Figure 7: skipped -- no aggregated_landmark*.csv found under %s", INPUTS_DIR))
      return(NULL)
    }

    df <- df %>% mutate(
      t_lab = suppressWarnings(as.numeric(t_lab)),
      t_rel = t_lab,
      LAB_VALUE = suppressWarnings(as.numeric(LAB_VALUE))
    ) %>% drop_na(t_rel, LAB_VALUE)
    if (nrow(df) == 0) { message("Figure 7: skipped -- all t_lab/LAB_VALUE NaN"); return(NULL) }

    df$t_platinum_rel <- if (all(c("t_platinum","PLATINUM") %in% names(df)))
      suppressWarnings(as.numeric(df$t_platinum)) else NA_real_
    df$LAB_GROUP <- ifelse(grepl("testosterone", tolower(df$LAB_NAME)), "Testosterone", "PSA")
    df
  }

  androgen_long_df <- load_androgen_longitudinal()
  if (!is.null(androgen_long_df))
    message(sprintf("Figure 7: %s androgen-axis rows across %s %s patients",
                    format(nrow(androgen_long_df), big.mark=","),
                    format(length(unique(androgen_long_df$DFCI_MRN)), big.mark=","), COHORT))

  ## ---- Figure 7b: group mean +/- 95% CI, binned by time from treatment anchor ----
  BIN_WIDTH_DAYS <- 60
  # Asymmetric window: 1 year of pre-landmark history, 2 years of follow-up.
  PRE_DAYS  <- 365   # days BEFORE the treatment anchor
  POST_DAYS <- 730   # days AFTER the treatment anchor

  bin_group_ci <- function(df, lab_group) {
    sub <- df %>% filter(LAB_GROUP == lab_group, t_rel >= -PRE_DAYS, t_rel <= POST_DAYS)
    if (nrow(sub) == 0) return(NULL)
    edges <- seq(-PRE_DAYS, POST_DAYS, by = BIN_WIDTH_DAYS)
    sub <- sub %>% mutate(
      t_bin = cut(t_rel, breaks = edges, include.lowest = TRUE),
      plat_group = if ("PLATINUM" %in% names(sub))
        as.integer(coalesce(suppressWarnings(as.numeric(PLATINUM)), 0)) else 0L)
    mids <- (head(edges, -1) + tail(edges, -1)) / 2
    names(mids) <- levels(sub$t_bin)
    patient_bin <- sub %>% drop_na(LAB_VALUE, t_bin) %>%
      group_by(DFCI_MRN, t_bin, plat_group) %>%
      summarise(LAB_VALUE = mean(LAB_VALUE), .groups = "drop")
    patient_bin %>% group_by(t_bin, plat_group) %>%
      summarise(n = n_distinct(DFCI_MRN), mean = mean(LAB_VALUE),
                sem = if (n() > 1) sd(LAB_VALUE) / sqrt(n()) else 0, .groups = "drop") %>%
      mutate(t_mid = mids[as.character(t_bin)],
             ci_lo = mean - 1.96 * sem, ci_hi = mean + 1.96 * sem)
  }

  plot_group_ci_panel <- function(df, lab_group, title) {
    binned <- bin_group_ci(df, lab_group)
    if (is.null(binned) || nrow(binned) == 0)
      return(ggplot() + annotate("text", x = 0, y = 0, label = "(no data)", color = "#7f8c8d") +
               theme_void() + labs(title = title))
    tot <- binned %>% group_by(plat_group) %>%
      summarise(n_min = min(n), n_max = max(n), .groups = "drop")
    leg <- setNames(sprintf("%s (n patients/bin=%s–%s)",
                            PLAT_LABELS[as.character(tot$plat_group)],
                            format(tot$n_min, big.mark = ","),
                            format(tot$n_max, big.mark = ",")),
                    as.character(tot$plat_group))
    binned <- binned %>% mutate(plat_group = factor(plat_group)) %>% arrange(t_mid)
    ggplot(binned, aes(t_mid, mean, color = plat_group, fill = plat_group)) +
      geom_vline(xintercept = 0, color = "#2c3e50", linetype = "dotted", linewidth = 1, alpha = 0.6) +
      geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.2, color = NA) +
      geom_line(linewidth = 0.8) + geom_point(size = 1.6) +
      scale_color_manual(values = setNames(PLAT_COLORS, c("0","1")), labels = leg, name = NULL) +
      scale_fill_manual(values = setNames(PLAT_COLORS, c("0","1")), guide = "none") +
      labs(x = "Days from treatment anchor (binned, 60d windows)",
           y = sprintf("%s (mean +/- 95%% CI)", lab_group), title = title) +
      theme_fig() +
      theme(plot.title = element_text(face = "bold", size = 11))
  }

  if (is.null(androgen_long_df)) {
    message("Figure 7b: skipped -- no androgen longitudinal data available")
  } else {
    patients <- unique(na.omit(androgen_long_df$DFCI_MRN))
    if (length(patients) > N_GROUP) {
      set.seed(RANDOM_SEED)
      patients <- sample(patients, N_GROUP)
    }
    group_df <- androgen_long_df %>% filter(DFCI_MRN %in% patients)
    for (lab_group in c("PSA", "Testosterone")) {
      n_pat <- group_df %>% filter(LAB_GROUP == lab_group) %>% summarise(n = n_distinct(DFCI_MRN)) %>% pull(n)
      ttl <- sprintf("%s -- group mean +/- 95%% CI vs. days from treatment anchor (n=%s patients)",
                     lab_group, format(n_pat, big.mark = ","))
      p <- plot_group_ci_panel(group_df, lab_group, ttl)
      save_fig(p, OUT_DIR, sprintf("androgen_longitudinal_group_ci_%s", tolower(lab_group)),
               width = 7.0, height = 5.5)
      if (show) print(p)
    }
  }
}
