# Run with COMPASS_REVIEW_DIR=/tmp/compass-revised Rscript tests/test_figure_legibility_and_km.R
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(tidyr); library(stringr); library(survival); library(readr)})
source_exprs <- parse("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
find_assignment <- function(expr, name) {
  if (missing(expr) || !is.call(expr)) return(NULL)
  if (identical(expr[[1]], as.name("<-")) && identical(expr[[2]], as.name(name))) return(expr)
  for (child in as.list(expr)[-1]) {
    result <- find_assignment(child, name)
    if (!is.null(result)) return(result)
  }
  NULL
}
for (name in c("theme_fig", "figure_tertiles", "figure_extreme_quintiles",
               "figure_gleason_groups", "figure_platinum_strata",
               "significant_mutation_features", "plot_stratified_platinum", "prepare_figure_text",
               "plot_model_discrimination", "render_discrimination_panel", "render_available_case_sensitivity",
               "read_endpoint_performance", "plot_volcano_panel", "labels_for_panel",
               "q_threshold_neglog10p", "assign_category", "plot_sg_forest")) {
  for (expr in source_exprs) {
    assignment <- find_assignment(expr, name)
    if (!is.null(assignment)) eval(assignment)
  }
}
source("COMPASS/survival_analysis/figure_supplements.R")

# Ties are preserved; degenerate distributions cannot be claimed to have tertiles.
groups <- figure_tertiles(c(1,1,2,3,4,5,6,7,8,NA,Inf))
stopifnot(groups[1] == groups[2], length(unique(na.omit(groups))) == 3,
          all(is.na(tail(groups, 2))), all(is.na(figure_tertiles(rep(0, 20)))))
# Bottom/top 20% only: the middle 60% is dropped, and ties at a cut stay whole.
q <- figure_extreme_quintiles(1:100)
stopifnot(identical(sort(unique(na.omit(q))), c("Bottom 20%", "Top 20%")),
          sum(q == "Bottom 20%", na.rm = TRUE) == 20,
          sum(q == "Top 20%", na.rm = TRUE) == 20,
          sum(is.na(q)) == 60)
stopifnot(all(is.na(figure_extreme_quintiles(rep(0, 20)))),
          all(is.na(figure_extreme_quintiles(c(1, NA, Inf)))))
# A pile of tied values straddling the bottom cut is never split across arms.
tied <- figure_extreme_quintiles(c(rep(2, 40), 3:62))
stopifnot(length(unique(tied[1:40])) == 1L)
stopifnot(identical(figure_gleason_groups(c(2:10,0,11,NA,6.5,Inf,-Inf)),
  c(rep("Gleason ≤7", 6), rep("Gleason ≥8", 3), rep(NA_character_, 6))),
  identical(figure_gleason_groups(c("6", "7", "8", "10", "unknown")),
            c("Gleason ≤7", "Gleason ≤7", "Gleason ≥8", "Gleason ≥8", NA_character_)))
results <- tibble(feature = c("TP53_SNV", "PTEN_DEL", "RB1_SNV", "PSA__mean", "ALK_SNV"),
                  q_value = c(.01,.04,.051,.001,NA))
stopifnot(identical(significant_mutation_features(results), c("TP53_SNV", "PTEN_DEL")))
frame <- tibble(DFCI_MRN = 1:8, t_platinum = c(100,200,300,0,-30,NA,400,500),
                PLATINUM = c(1,0,1,1,1,0,NA,0), t_death = rep(1,8))
g <- c("Carrier","Non-carrier",NA,"Carrier","Carrier","Carrier","Carrier","Non-carrier")
d <- figure_platinum_strata(frame,g)
stopifnot(identical(d$time, c(100,200,500)), identical(d$event,c(1,0,0)))
# The source clock is authoritative: never subtract the landmark twice or use
# original ADT t_death when plotting an indexed sequencing/Gleason duration.
stopifnot(!any(d$time == 1))
stopifnot(inherits(tryCatch(figure_platinum_strata(bind_rows(frame, frame[1,]), c(g,"Carrier")),
                           error = identity), "error"))

plots <- list()
set.seed(30)
f <- tibble(DFCI_MRN = 1:90, t_platinum = rexp(90, .001) + 1, PLATINUM = rep(c(1,0,0),30))
g <- figure_tertiles(1:90)
plots$tertiles <- plot_stratified_platinum(figure_platinum_strata(f,g), "PSA tertiles: time to platinum",
  "the +180-day treatment landmark", c("Low tertile","Middle tertile","High tertile"),
  "Synthetic data. Equal values stay together. Shading: 95% CI.")
qg <- figure_extreme_quintiles(1:90)
plots$quintiles <- plot_stratified_platinum(figure_platinum_strata(f,qg),
  "PSA bottom vs top 20%: time to platinum", "the +180-day treatment landmark",
  c("Bottom 20%","Top 20%"),
  "Synthetic data. Equal values stay together. Shading: 95% CI.")
stopifnot(!is.null(plots$quintiles),
          nrow(figure_platinum_strata(f,qg)) == sum(!is.na(qg)))
plots$carriers <- plot_stratified_platinum(figure_platinum_strata(f,rep(c("Carrier","Non-carrier"),45)),
  "TP53 SNV carrier status: time to platinum", "sequencing specimen collection date")
gleason_data <- figure_platinum_strata(f, figure_gleason_groups(rep(6:10,18)))
plots$gleason <- plot_stratified_platinum(gleason_data,
  "Gleason score: time to platinum", "Gleason score date nearest ADT initiation",
  c("Gleason ≤7", "Gleason ≥8"),
  "Scores grouped as ≤7 versus ≥8. Missing or invalid scores excluded. Shading: 95% CI.")
stopifnot(nrow(gleason_data) == nrow(f), sum(gleason_data$event) == sum(f$PLATINUM),
          identical(sort(unique(plots$gleason$data$stratum)), c("Gleason ≤7", "Gleason ≥8")),
          identical(unname(plots$gleason$scales$get_scales("colour")$labels),
                    c("Gleason ≤7 (n=36; events=12)", "Gleason ≥8 (n=54; events=18)")))
gleason_logrank <- survival::survdiff(survival::Surv(time, event) ~ stratum, data = gleason_data)
stopifnot(identical(plots$gleason$labels$subtitle,
                    sprintf("Log-rank p = %.3g", pchisq(gleason_logrank$chisq, 1, lower.tail = FALSE))),
          is.null(plot_stratified_platinum(filter(gleason_data, stratum == "Gleason ≤7"),
                                           "Gleason", "index date")))

forest <- expand_grid(cohort = names(cohort_forest_labels), endpoint = c("platinum","nepc"),
  lab_name = c("PSA","Testosterone"), feature_stat = c("mean","min","max","last","delta","n_observations")) %>%
  mutate(feature = paste(lab_name,feature_stat,sep="__"), landmark_days = 180,
         hazard_ratio_per_sd = exp(seq(-1,1,length.out=n())), ci_lower = hazard_ratio_per_sd / 1.3,
         ci_upper = hazard_ratio_per_sd * 1.3, q_value = rep(c(.01,.2), length.out=n()))
forest$ci_upper[1] <- 15
for (endpoint in c("platinum","nepc")) {
  d <- prepare_cohort_forest(forest, endpoint)
  stopifnot(nrow(d) == 48, !any(tolower(as.character(d$feature_stat)) %in% c("delta","n_observations")),
            all(d$endpoint == endpoint), all(c("PSA","Testosterone") %in% d$analyte))
  plots[[paste0("forest_",endpoint)]] <- plot_cohort_forest(d,endpoint)
  built <- ggplot_build(plots[[paste0("forest_",endpoint)]])
  stopifnot(sum(is.finite(built$data[[3]]$x)) == 48)
}

# Compare low and perfect metrics: neither bars nor value labels get clipped.
LANDMARKS <- c(0,90,180)
SERIES_COLORS <- c(a="#0072B2",b="#D55E00")
HAS_GGPATTERN <- FALSE
discrimination_data <- expand_grid(name=c("a","b"),landmark=LANDMARKS) %>% mutate(auc=c(.2,.4,.45,.5,.99,1))
plots$performance <- render_discrimination_panel("auc","Test mean AUC(t)",TRUE)
b <- ggplot_build(plots$performance)
stopifnot(b$layout$panel_params[[1]]$y.range[1] <= 0, b$layout$panel_params[[1]]$y.range[2] > 1)

# Sensitivity counts come from the same endpoint/model rows as the metrics, not
# from the full cohort or a sum across the four repeated model populations.
ENDPOINT <- "platinum"
series <- tibble(name = c("Elastic-Net Cox: labs", "Elastic-Net Cox: Gleason",
                          "XGBoost: labs", "XGBoost: Gleason"),
                 color = c("#4C72B0", "#2A9D8F", "#B58900", "#D55E00"))
sensitivity <- expand_grid(name = series$name, landmark = LANDMARKS) %>%
  mutate(auc = .7, cindex = .65, n_train_val = 960 - landmark, n_test = 240)
plots$sensitivity <- render_available_case_sensitivity(sensitivity, series, "Gleason", "auc", "Test mean AUC(t)")
stopifnot(identical(plots$sensitivity$labels$title,
  "Gleason vs. labs sensitivity — platinum\n0d: n=1,200; +90d: n=1,110; +180d: n=1,020"))
counts_path <- tempfile(fileext = ".csv")
readr::write_csv(tibble(endpoint = c("nepc", "platinum"), test_mean_auc_t = c(.6, .7),
                       n_train_val = c(80, 960), n_test = c(20, 240)), counts_path)
metrics <- read_endpoint_performance(counts_path, "platinum")
stopifnot(metrics[["n_train_val"]] == 960, metrics[["n_test"]] == 240,
          metrics[["auc"]] == .7)
unlink(counts_path)
sensitivity$n_train_val[sensitivity$landmark == 90] <- NA_real_
sensitivity$n_test[1] <- 241
sensitivity$auc[sensitivity$landmark == 180] <- NA_real_
unavailable <- render_available_case_sensitivity(sensitivity, series, "Gleason", "auc", "AUC")
stopifnot(grepl("0d: n=1,200–1,201 (varies by model)", unavailable$labels$title, fixed = TRUE),
          grepl("+90d: n=unavailable; +180d: n=unavailable", unavailable$labels$title, fixed = TRUE))
sensitivity$n_test[2] <- -1
sensitivity$n_train_val[3] <- 1.5
invisible(ggplot_build(render_available_case_sensitivity(sensitivity, series, "Gleason", "cindex", "C-index")))

# Dense volcano: full x-range, one external legend, readable labels, all points retained.
DROP <- character(); TOP_K_PER_PANEL <- 4; ALWAYS_LABEL <- "Hemoglobin"
PANEL_XLIM <- c(-1.5,1.5); Y_MAX_CAP <- 30; NS_COLOR <- "#9ba4ae"
CATEGORY_MAP <- c(PSA="Androgen axis",Testosterone="Androgen axis",Hemoglobin="CBC",Albumin="LFT")
CATEGORY_COLORS <- c("Androgen axis"="#8e1c2b",CBC="#159b83",LFT="#df8300",Other="#666666")
DRAW_ORDER <- LEGEND_ORDER <- names(CATEGORY_COLORS)
v <- expand_grid(lab_name=names(CATEGORY_MAP),feature_stat=c("mean","min","max","last","delta")) %>%
  mutate(coef_feature=c(-2.5,2.4,seq(-.8,.8,length.out=n()-2)), p_value=10^-seq(2,15,length.out=n()), q_value=.01)
plots$volcano <- plot_volcano_panel(v,"Lab associations at +180 days")
b <- ggplot_build(plots$volcano)
stopifnot(b$layout$panel_params[[1]]$x.range[1] < -2.5, b$layout$panel_params[[1]]$x.range[2] > 2.4,
          plots$volcano$theme$legend.position == "bottom")

review <- Sys.getenv("COMPASS_REVIEW_DIR", "")
if (nzchar(review)) {
  dir.create(review, recursive=TRUE, showWarnings=FALSE)
  for (name in names(plots)) {
    is_forest <- startsWith(name,"forest_")
    w <- if (is_forest) COHORT_FOREST_SLIDE_SIZE[["width"]] else if (name == "sensitivity") 8.5 else 9
    p <- prepare_figure_text(plots[[name]],w)
    ggsave(file.path(review,paste0(name,".png")),p,width=w,
           height=if(is_forest)COHORT_FOREST_SLIDE_SIZE[["height"]] else if(name == "sensitivity")5.5 else 7.5,
           dpi=120,device=ragg::agg_png,bg="white")
  }
}
cat("Legibility, tertile ties, indexed times, mutation selection, Gleason groups, and endpoint forests passed.\n")
