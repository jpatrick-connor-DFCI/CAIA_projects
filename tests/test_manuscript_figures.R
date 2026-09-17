source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_data_cache.R")

# Styling a manuscript copy must not mutate a layer shared by the standalone.
p <- ggplot(tibble(x=1,y=2,label="Unchanged"),aes(x,y,label=label)) +
  geom_text(size=6) + theme_classic() +
  theme(axis.title.y=element_text(size=20),axis.text.x=element_text(size=18))
q <- manuscript_style(p,"Copy","A")
stopifnot(p$layers[[1]]$aes_params$size==6,q$layers[[1]]$aes_params$size==2.5,
  q$theme$axis.title.y$size==8,q$theme$axis.title.y$face=="bold",q$theme$axis.text.x$size==7,
  identical(p$data,q$data),is.null(q$labels$title),identical(q$labels$tag,"a"))
flow <- manuscript_style(p+theme(axis.text=element_blank()),"Flow","A")
stopifnot(inherits(flow$theme$axis.text.x,"element_blank"),inherits(flow$theme$axis.text.y,"element_blank"))
cohort_spec <- manuscript_specs()[[1]]
stopifnot(cohort_spec$key=="01_cohort_overview",cohort_spec$height==7.2)
dfci_spec <- manuscript_specs()[[3]]
stopifnot(dfci_spec$key=="03_dfci_univariable",identical(dfci_spec$row_heights,c(1.5,.18,1,1)),
  identical(dfci_spec$shared_legend_panels,1:3),all(dfci_spec$layout[2,]==8))
multivariable_spec <- manuscript_specs()[[4]]
stopifnot(multivariable_spec$key=="04_multivariable_labs",length(multivariable_spec$members)==8,
  multivariable_spec$height==11.8,identical(multivariable_spec$row_heights,c(.9,.9,1,1)))

# Captions mirror the paired Markdown/text layout used by the clinical text
# embedding project and retain figure-specific audit language.
captions <- manuscript_captions()
stopifnot(length(captions)==9,identical(names(captions),sprintf("%02d_%s",1:9,c(
  "cohort_overview","llm_nepc_avpc","dfci_univariable","multivariable_labs",
  "gleason_sensitivity","cohort_sensitivity_incidence","cohort_sensitivity_associations",
  "federated_psa_testosterone","federated_xgboost"))),
  all(startsWith(unname(captions),paste0("Figure ",1:9,". "))),
  grepl("person_id",captions[["09_federated_xgboost"]],fixed=TRUE))
caption_root <- tempfile("manuscript-caption-test-")
caption_paths <- manuscript_write_captions(caption_root)
stopifnot(length(caption_paths)==18,all(file.exists(caption_paths)))
for(i in 1:9) stopifnot(identical(
  readLines(file.path(caption_root,"captions",paste0("figure",i,".md")),warn=FALSE),
  readLines(file.path(caption_root,"captions",paste0("figure",i,".txt")),warn=FALSE)))
unlink(caption_root,recursive=TRUE)

# Endpoint selection does not combine event definitions or invent missing cells.
incidence <- tibble(cohort=c("adt","adt","adt_noprecastrate"),
  endpoint=c("platinum","nepc","platinum"),landmark_days=180,
  n_patients=c(100,100,NA),n_events=c(10,30,NA),status=c("ok","ok","missing"))
result <- manuscript_incidence(incidence)
stopifnot(nrow(result$data)==2,all(result$data$endpoint=="platinum"),
  result$data$n_events[1]==10,!result$data$available[2],
  grepl("not censoring-adjusted",result$legend,fixed=TRUE))

# Preserve every supplied lab estimate and q-value across two analytes, four
# statistics, three landmarks and sites; q-values are not recomputed here.
labs <- tidyr::expand_grid(lab_name=c("PSA","Testosterone"),feature_stat=c("mean","min","max","last"),
  landmark_days=c(0,90,180),source=c("Dana-Farber","Federated*")) %>%
  mutate(hazard_ratio_per_sd=seq(.6,1.3,length.out=n()),ci_lower=hazard_ratio_per_sd-.1,
    ci_upper=hazard_ratio_per_sd+.1,valid_estimate=TRUE,valid_ci=TRUE,p_value=.02,q_value=.09,
    source_kind=if_else(source=="Federated*","across_sites","within_site"),
    significance="Nominal p < 0.05 only",population_note="Membership inferred; verify source.")
forest <- manuscript_federated_labs(labs)
for(column in c("hazard_ratio_per_sd","ci_lower","ci_upper","p_value","q_value"))
  stopifnot(identical(forest$data[[column]],labs[[column]]))
stopifnot(nrow(forest$data)==48,grepl("Membership inferred",forest$legend,fixed=TRUE))
duplicate <- tryCatch(manuscript_federated_labs(bind_rows(labs,labs[1,])),error=identity)
stopifnot(inherits(duplicate,"error"))

# A flagged identifier remains in the supplied importance input and audit note.
metrics <- expand_grid(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
  mutate(test_mean_auc_t=.7,test_c_index=.65,n_test=20,n_events_test=4,
    input_audit_note="CAUTION: person_id has nonzero gain; audit before interpretation.")
importance <- expand_grid(landmark_days=c(0,90,180),feature=c("PSA__max","person_id")) %>%
  mutate(lab_name=if_else(feature=="person_id","person_id","PSA"),feature_stat="max",
    gain=10,displayed=TRUE,identifier_feature=feature=="person_id",
    input_audit_note=metrics$input_audit_note[1])
xgb <- manuscript_federated_xgb(metrics,importance)
stopifnot(grepl("person_id",xgb$legend,fixed=TRUE),
  xgb$spec$height==10.2,
  identical(xgb$data$test_mean_auc_t,metrics$test_mean_auc_t),
  sum(importance$identifier_feature)==3)
original <- plot_federated_comparison(labs,"PSA","Test")
stopifnot(identical(original$scales$get_scales("colour")$palette(2),
  c(`Dana-Farber`="#0072B2",`Federated*`="#222222")))
cat("Manuscript data fidelity, missing cells, audit notes and isolated styling passed.\n")
