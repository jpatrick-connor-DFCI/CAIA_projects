source("COMPASS/figures/COMPASS_generate_figures_pipeline.R")
source("COMPASS/figures/figure_workflow.R")

metrics <- expand_grid(endpoint=c("platinum","avpc","nepc"),landmark_days=c(0,90,180),
  model_family=c("xgboost_cox","elastic_net_cox"),config=c("both","baseline")) %>%
  mutate(bundle=model_family,n_patients=1000-landmark_days,n_events=30,
    mean_auc=if_else(config=="both",.7,.55),mean_auc_lo=mean_auc-.05,mean_auc_hi=mean_auc+.05)
clinical <- filter(metrics,config=="both") %>% select(-config) %>%
  crossing(subpopulation=c("gleason","trio"),model=c("transferred_labs","clinical_cox","combined_cox")) %>%
  mutate(mean_auc=case_when(model=="combined_cox"~.75,model=="clinical_cox"~.65,TRUE~.7),
    mean_auc_lo=mean_auc-.04,mean_auc_hi=mean_auc+.04)
prepared <- prepare_federated_transfer_clinical(clinical,"gleason")
plot <- plot_federated_transfer_intervals(prepared,clinical=TRUE)
stopifnot(nrow(plot$data)==45,length(unique(plot$data$comparison))==5,
  is.null(plot$mapping$shape),all(plot$data$value==prepared$mean_auc),
  all(plot$data$lo==prepared$mean_auc_lo),all(plot$data$hi==prepared$mean_auc_hi))
invisible(ggplot_build(plot))
for(layout in c("interval","scatter")) {
  result <- manuscript_federated_transfer_no_dfci(metrics,clinical,include=c("gleason","trio"),
    settings=list(layout=layout,landmarks=180L))
  stopifnot(inherits(result$plot,"gtable"),length(result$spec$titles)==3,
    all(result$data$landmark_days==180),all(result$data$config=="both"),
    grepl("day-180 patients/events",result$legend,fixed=TRUE),
    grepl("180d 820/30",result$legend,fixed=TRUE))
}
Sys.setenv(COMPASS_TRANSFER_LAYOUT="interval",COMPASS_TRANSFER_LANDMARKS="180")
stopifnot(identical(manuscript_transfer_settings(),list(layout="interval",landmarks=180L)),
  grepl("Horizontal intervals",manuscript_transfer_caption(),fixed=TRUE),
  grepl("day-180 patients",manuscript_transfer_caption(),fixed=TRUE))
Sys.setenv(COMPASS_TRANSFER_LANDMARKS="bogus")
stopifnot(inherits(tryCatch(manuscript_transfer_settings(),error=identity),"error"))
cat("Transfer intervals preserve estimates, deduplicate clinical comparators, and respect landmark counts/captions.\n")
