# Run with COMPASS_REVIEW_DIR=/tmp/compass-revised Rscript tests/test_figure_legibility_and_km.R
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(tidyr); library(stringr); library(survival)})
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
for (name in c("theme_fig", "figure_tertiles", "figure_gleason_groups", "figure_platinum_strata",
               "significant_mutation_features", "plot_stratified_platinum", "prepare_figure_text",
               "render_discrimination_panel", "plot_volcano_panel", "labels_for_panel",
               "q_threshold_neglog10p", "assign_category", "plot_sg_forest")) {
  for (expr in source_exprs) {
    assignment <- find_assignment(expr, name)
    if (!is.null(assignment)) eval(assignment)
  }
}
source("COMPASS/survival_analysis/cohort_forest_figures.R")

# Ties are preserved; degenerate distributions cannot be claimed to have tertiles.
groups <- figure_tertiles(c(1,1,2,3,4,5,6,7,8,NA,Inf))
stopifnot(groups[1] == groups[2], length(unique(na.omit(groups))) == 3,
          all(is.na(tail(groups, 2))), all(is.na(figure_tertiles(rep(0, 20)))))
stopifnot(identical(figure_gleason_groups(c(6,7,8,9,10,0,11,NA,6.5)),
  c("Gleason ≤6", "Gleason 7", "Gleason 8", "Gleason 9", "Gleason 10", rep(NA_character_,4))))
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
plots$carriers <- plot_stratified_platinum(figure_platinum_strata(f,rep(c("Carrier","Non-carrier"),45)),
  "TP53 SNV carrier status: time to platinum", "sequencing specimen collection date")
plots$gleason <- plot_stratified_platinum(figure_platinum_strata(f,figure_gleason_groups(rep(6:10,18))),
  "Gleason score: time to platinum", "Gleason score date nearest ADT initiation")

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
    w <- if (is_forest) 12 else 9
    p <- prepare_figure_text(plots[[name]],w)
    ggsave(file.path(review,paste0(name,".png")),p,width=w,height=if(is_forest)12 else 7.5,
           dpi=120,device=ragg::agg_png,bg="white")
  }
}
cat("Legibility, tertile ties, indexed times, mutation selection, Gleason groups, and endpoint forests passed.\n")
