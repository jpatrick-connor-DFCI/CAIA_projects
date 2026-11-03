source("COMPASS/figures/COMPASS_generate_figures_pipeline.R")
source("COMPASS/figures/figure_workflow.R")

metrics <- expand_grid(endpoint=c("platinum","avpc","nepc"),landmark_days=c(0,90,180),
  model_family=c("xgboost_cox","elastic_net_cox"),config=c("both","baseline")) %>%
  mutate(bundle=model_family,n_patients=1000-landmark_days,n_events=30,
    mean_auc=if_else(config=="baseline",.5,.6+landmark_days/900),
    mean_auc_lo=mean_auc-.05,mean_auc_hi=mean_auc+.05)
clinical <- filter(metrics,config=="both") %>% select(-config) %>%
  crossing(subpopulation=c("gleason","trio","trio_gleason"),
    model=c("transferred_labs","clinical_cox","combined_cox")) %>%
  mutate(mean_auc=mean_auc+case_when(model=="clinical_cox"~-.05,model=="combined_cox"~.05,TRUE~0))

mean <- summarise_transfer_landmarks(prepare_federated_transfer(filter(metrics,config=="both")))
stopifnot(nrow(mean)==6,all(abs(mean$mean_auc-.7)<1e-12),all(mean$n_landmarks==3),
  all(mean$n_patients_day0==1000),all(mean$n_patients_day180==820),
  all(mean$value_day0==.6),!any(c("lo","hi") %in% names(mean)))
comparison <- prepare_federated_transfer_clinical(clinical,"trio_gleason")
means <- summarise_transfer_landmarks(comparison,TRUE)
stopifnot(nrow(means)==15,all(means$n_landmarks==3),
  all(abs(means$mean_auc[means$comparison=="Clinical only"]-.65)<1e-12))

# A missing/invalid landmark remains unavailable, rather than silently using two.
missing <- filter(comparison,!(endpoint=="nepc" & landmark_days==180 & score=="combined_cox"))
missing$value[missing$endpoint=="avpc" & missing$landmark_days==90 & missing$score=="clinical_cox"] <- NA_real_
incomplete <- summarise_transfer_landmarks(missing,TRUE)
stopifnot(sum(is.na(incomplete$mean_auc))==3,
  all(incomplete$n_landmarks[is.na(incomplete$mean_auc)]==2))
stopifnot(inherits(tryCatch(summarise_transfer_landmarks(bind_rows(comparison,comparison[1,]),TRUE),
  error=identity),"error"))

for(set in 1:2) {
  result <- manuscript_federated_transfer_means(metrics,clinical,set)
  stopifnot(inherits(result$plot,"gtable"),length(result$spec$titles)==if(set==1) 3 else 2,
    setequal(result$data$subpopulation,if(set==1) c("full","gleason","trio") else c("full","trio_gleason")),
    nrow(result$data)==if(set==1) 36 else 21,
    grepl("manuscript",manuscript_folder("ADT",result$spec$key),fixed=TRUE),
    result$spec$key %in% names(manuscript_captions()))
}
stopifnot(is.null(manuscript_federated_transfer_means(metrics,filter(clinical,subpopulation!="trio_gleason"),2)))
p <- plot_transfer_landmark_mean(means,c(.4,1))
stopifnot(!any(vapply(p$layers,function(l) inherits(l$geom,"GeomLinerange"),logical(1))))
invisible(ggplot_build(p))
cat("Landmark means, missing-data handling, deduplication and both manuscript sets passed.\n")
