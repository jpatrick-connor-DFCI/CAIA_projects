source("COMPASS/figures/COMPASS_generate_figures_pipeline.R")
source("COMPASS/figures/figure_workflow.R")

# Figure 5's one-model accuracy panel uses vertical landmark axes, preserves
# the supplied values/intervals, and has enough room for the endpoint strips.
metrics <- tidyr::expand_grid(endpoint=c("platinum","avpc","nepc"),landmark_days=c(0,90,180)) %>%
  mutate(bundle="xgb",model_family="xgboost_cox",config="both",
    n_patients=1000,n_events=30,mean_auc=.7,mean_auc_lo=.6,mean_auc_hi=.8)
fig <- manuscript_federated_transfer_no_dfci(metrics,include=character(),
  model_families="xgboost_cox",probability=TRUE)
accuracy <- fig$panels[[1]]
stopifnot(inherits(accuracy,"ggplot"),!inherits(accuracy$coordinates,"CoordFlip"),
  identical(accuracy$labels$x,"Landmark after ADT initiation"),
  identical(accuracy$labels$y,"Mean AUC(t)"),
  identical(as.character(accuracy$data$landmark),rep(c("0 days","+90 days","+180 days"),3)),
  all(accuracy$data$value==.7),all(accuracy$data$lo==.6),all(accuracy$data$hi==.8),
  fig$spec$row_heights[1]>=3)

# One top-decile group ends follow-up before five years: it must stay on the
# axis with an unavailable label, not disappear or acquire a fabricated risk.
groups <- bind_rows(lapply(names(FEDERATED_TRANSFER_GRADIENT[FEDERATED_TRANSFER_RISK_STRATIFIERS]),
  function(s) tibble(stratifier=s,group=FEDERATED_TRANSFER_GRADIENT[[s]]$levels)))
gradient <- groups %>% mutate(endpoint="platinum",model_family="xgboost_cox",landmark_days=180L,
  clock="sequencing",n=30L,events=5L,event_free=.8,lo=.7,hi=.9,
  event_free=if_else(group=="Top 10%",NA_real_,event_free))
tables <- list(gradient=gradient)
p <- plot_federated_transfer_horizon_risk(tables,"xgboost_cox",180L)
score <- p$data %>% filter(stratifier=="score_decile") %>% arrange(group)
stopifnot(identical(as.character(score$group),c("Bottom 10%","Top 10%")),
  is.finite(score$probability[1]),is.na(score$probability[2]),
  is.na(score$probability_lo[2]),is.na(score$probability_hi[2]),score$n[2]==30L)
built <- suppressMessages(ggplot2::ggplot_build(p))
stopifnot(all(c("Bottom 10%","Top 10%") %in% built$layout$panel_params[[1]]$x$get_labels()),
  "Not estimable" %in% built$data[[4]]$label)

# An absent top group is also shown, explicitly as having no patients.
tables$gradient <- filter(gradient,group!="Top 10%")
p <- plot_federated_transfer_horizon_risk(tables,"xgboost_cox",180L)
score <- filter(p$data,stratifier=="score_decile")
stopifnot(setequal(as.character(score$group),c("Bottom 10%","Top 10%")),
  is.na(score$n[score$group=="Top 10%"]),
  "No patients" %in% suppressMessages(ggplot2::ggplot_build(p))$data[[4]]$label)
cat("Figure 5 vertical accuracy and visible score deciles passed.\n")
