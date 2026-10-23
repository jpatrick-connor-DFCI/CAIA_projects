source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_workflow.R")

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
  identical(dfci_spec$shared_legend_panels,1:3),all(dfci_spec$layout[2,]==8),
  identical(manuscript_dfci_volcano_ylim,c(-.2,20)),
  manuscript_dfci_km_landmark==180,
  grepl("landmark180",dfci_spec$members[5],fixed=TRUE),
  grepl("landmark180",dfci_spec$members[7],fixed=TRUE),
  identical(manuscript_dfci_landmark_headers,c("+0 days","+90 days","+180 days")),
  identical(manuscript_km_y_label,"Platinum-free\nprobability"),
  identical(ALBUMIN_DAY180_NUDGE,c(x=-.8,y=1.7)))
multivariable_spec <- manuscript_specs()[[4]]
stopifnot(multivariable_spec$key=="04_multivariable_labs",length(multivariable_spec$members)==7,
  multivariable_spec$members[1]=="figure4a_discrimination_auc_platinum",
  !any(grepl("cindex",multivariable_spec$members,fixed=TRUE)),
  multivariable_spec$width==8.5,multivariable_spec$height==9,
  identical(multivariable_spec$row_heights,c(.9,1,1)),
  manuscript_multivariable_legend_height==.14)
landmark_header <- manuscript_center_landmark_header(ggplot()+labs(subtitle="+90 days"))
stopifnot(landmark_header$theme$plot.subtitle$face=="bold",
  landmark_header$theme$plot.subtitle$hjust==.5,
  landmark_header$theme$plot.subtitle$size==8)

# Captions mirror the paired Markdown/text layout used by the clinical text
# embedding project and retain figure-specific audit language.
captions <- manuscript_captions()
federated_keys <- c("federated_03_univariable","federated_04_multivariable","federated_05_transfer_no_dfci")
# Figure 5 (internal Gleason sensitivity) was removed; Gleason is compared in transfer.
stopifnot(length(captions)==10,identical(names(captions),c(sprintf("%02d_%s",c(1:4,6:8),c(
  "cohort_overview","llm_nepc_avpc","dfci_univariable","multivariable_labs",
  "dfci_cohort_sensitivity","federated_incidence_associations",
  "federated_multivariable")),federated_keys)),
  all(startsWith(unname(captions[1:7]),paste0("Figure ",c(1:4,6:8),". "))),
  all(startsWith(unname(captions[federated_keys]),paste0("Federated Figure ",3:5,". "))),
  grepl("22,138 patients (664 platinum events)",captions[["federated_03_univariable"]],fixed=TRUE),
  grepl("without Dana-Farber",captions[["federated_05_transfer_no_dfci"]],fixed=TRUE),
  grepl("Gleason score and age (b)",captions[["federated_05_transfer_no_dfci"]],fixed=TRUE),
  grepl("AUC(t)",captions[["federated_05_transfer_no_dfci"]],fixed=TRUE),
  !any(grepl("C-index",unname(captions),fixed=TRUE)),
  grepl("180-day landmark",captions[["03_dfci_univariable"]],fixed=TRUE),
  grepl("≤0.2453",captions[["03_dfci_univariable"]],fixed=TRUE),
  !grepl("day-0 landmark",captions[["03_dfci_univariable"]],fixed=TRUE),
  grepl("4,429/133",captions[["08_federated_multivariable"]],fixed=TRUE),
  grepl("Elastic-Net",captions[["08_federated_multivariable"]],fixed=TRUE),
  !grepl("person_id",captions[["08_federated_multivariable"]],fixed=TRUE),
  grepl("9,939/308",captions[["07_federated_incidence_associations"]],fixed=TRUE))
caption_root <- tempfile("manuscript-caption-test-")
caption_paths <- manuscript_write_captions(caption_root)
stopifnot(length(caption_paths)==20,all(file.exists(caption_paths)))
# The federated set is numbered on its own and never overwrites local figure3-5;
# each caption sits in the manuscript folder of its figure's section.
caption_md <- caption_paths[endsWith(caption_paths,".md")]
stopifnot(setequal(basename(caption_md),paste0(c(paste0("figure",c(1:4,6:8)),paste0("federated_figure",3:5)),".md")))
for(path in caption_md) stopifnot(identical(readLines(path,warn=FALSE),
  readLines(sub("[.]md$",".txt",path),warn=FALSE)))
caption_file <- function(section,name) file.path(caption_root,section,"manuscript","captions",paste0(name,".md"))
stopifnot(startsWith(readLines(caption_file("03_univariate/internal","figure3"),n=1,warn=FALSE),"Figure 3. "),
  startsWith(readLines(caption_file("03_univariate/federated","federated_figure3"),n=1,warn=FALSE),"Federated Figure 3. "),
  file.exists(caption_file("01_cohort","figure1")),file.exists(caption_file("02_llm","figure2")),
  !file.exists(caption_file("04_multivariate/internal","figure5")),file.exists(caption_file("04_multivariate/internal","figure4")),file.exists(caption_file("06_cohort_sensitivity","figure6")),
  file.exists(caption_file("04_multivariate/federated","figure8")),file.exists(caption_file("05_transfer","federated_figure5")),
  setequal(names(MANUSCRIPT_SECTIONS),names(captions)))
unlink(caption_root,recursive=TRUE)

site_incidence <- tibble(site_name=c("dana_farber_caia_1_1","fred_hutch_caia_1_1","jhu_caia_1_1"),
  site=c("Dana-Farber","Fred Hutch","Johns Hopkins"),landmark_days=0L,endpoint="platinum",
  n_patients=c(200,100,50),n_events=c(20,8,2),available=TRUE)
federated_incidence <- manuscript_federated_incidence(site_incidence)
stopifnot(nrow(federated_incidence$data)==3,
  identical(federated_incidence$data$n_patients,c(200,100,50)),
  identical(federated_incidence$data$n_events,c(20,8,2)),
  !grepl("compiled",federated_incidence$legend,fixed=TRUE))

# Endpoint selection does not combine event definitions or invent missing cells.
incidence <- tibble(cohort=c("adt","adt","adt_noprecastrate"),
  endpoint=c("platinum","nepc","platinum"),landmark_days=180,
  n_patients=c(100,100,NA),n_events=c(10,30,NA),status=c("ok","ok","missing"))
result <- manuscript_incidence(incidence)
stopifnot(nrow(result$data)==3,setequal(result$data$endpoint,c("platinum","nepc")),
  result$data$n_events[1]==10,!result$data$available[3],
  grepl("NEPC",result$legend,fixed=TRUE),
  grepl("not censoring-adjusted",result$legend,fixed=TRUE))

association_plot <- ggplot(tibble(hazard_ratio_per_sd=c(.8,1.2),ci_lower=c(.7,1),
  ci_upper=c(.9,1.4),cohort=c("adt","adt_noprecastrate")),
  aes(hazard_ratio_per_sd,cohort)) + geom_point()
dfci_combined <- manuscript_dfci_cohort_sensitivity(result,association_plot)
stopifnot(dfci_combined$spec$key=="06_dfci_cohort_sensitivity",
  identical(dfci_combined$spec$layout,rbind(c(1,2),c(3,3))),
  setequal(unique(dfci_combined$data$figure_section),
    c("incidence_and_counts","laboratory_associations")))

# Figure 7 labels the transformed horizontal coordinate directly as ln(HR),
# retaining the same estimates and null position as the HR-scale source plot.
log_axis_source <- ggplot(tibble(hazard_ratio_per_sd=c(.7,1.2),ci_lower=c(.6,1.0),
  ci_upper=c(.9,1.4),cohort=c("a","b")),aes(hazard_ratio_per_sd,cohort)) + geom_point()
log_axis_plot <- manuscript_log_hazard_axis(log_axis_source)
log_axis_scale <- log_axis_plot$scales$get_scales("x")
stopifnot(log_axis_plot$labels$x=="Log hazard ratio per SD (95% CI)",
  any(abs(log(log_axis_scale$breaks))<sqrt(.Machine$double.eps)))

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
psa_axis <- forest$panels[[1]]$scales$get_scales("x")
stopifnot(nrow(forest$data)==48,grepl("Membership inferred",forest$legend,fixed=TRUE),
  length(psa_axis$breaks)==3,any(abs(log(psa_axis$breaks))<sqrt(.Machine$double.eps)))
federated_combined <- manuscript_federated_incidence_associations(federated_incidence,forest)
stopifnot(federated_combined$spec$key=="07_federated_incidence_associations",
  identical(federated_combined$spec$layout,rbind(c(1,2),c(3,3),c(4,4))),
  setequal(unique(federated_combined$data$figure_section),
    c("incidence_and_counts","laboratory_associations")))
duplicate <- tryCatch(manuscript_federated_labs(bind_rows(labs,labs[1,])),error=identity)
stopifnot(inherits(duplicate,"error"))

# Figure 8 mirrors local Figure 4 from the captured federated CSVs. A flagged
# identifier remains in the supplied importance input and audit note.
performance <- bind_rows(lapply(c("elastic_net","xgboost"),function(family)
  expand_grid(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(test_mean_auc_t=.7,test_c_index=.65,n_test=20,n_events_test=4) %>%
    prepare_federated_performance(family) %>%
    mutate(name=as.character(name),input_audit_note=if(family=="xgboost")
      "CAUTION: person_id has nonzero gain; audit before interpretation." else NA_character_)))
coefficients <- expand_grid(landmark_days=c(0,90,180),feature=c("PSA__max","Testosterone__mean")) %>%
  mutate(lab_name=sub("__.*$","",feature),feature_stat=sub("^.*__","",feature),
    coefficient=rep(c(.2,-.3),3),coef=coefficient,displayed=TRUE,identifier_feature=FALSE,input_audit_note=NA)
importance <- expand_grid(landmark_days=c(0,90,180),feature=c("PSA__max","person_id")) %>%
  mutate(lab_name=if_else(feature=="person_id","person_id","PSA"),feature_stat="max",
    gain=10,displayed=TRUE,identifier_feature=feature=="person_id",
    input_audit_note=performance$input_audit_note[7])
federated_models <- manuscript_federated_multivariable(performance,coefficients,importance)
stopifnot(grepl("person_id",federated_models$legend,fixed=TRUE),
  federated_models$spec$key=="08_federated_multivariable",
  federated_models$spec$width==8.5,federated_models$spec$height==9,
  length(federated_models$spec$titles)==3,
  inherits(federated_models$plot,"gtable"),nrow(federated_models$data)==12,
  identical(levels(federated_models$data$name),names(FEDERATED_MODEL_COLORS)),
  grepl("Day 90, Cox baseline (age)",federated_models$legend,fixed=TRUE),
  identical(manuscript_federated_feature_label(c("Alkaline phosphatase (max)",
    "Neutrophils absolute (n_observations)","Diastolic blood pressure (min)")),
    c("ALP (max)","Neutrophils abs. (n obs.)","DBP (min)")))
unknown_series <- tryCatch(manuscript_federated_multivariable(
  mutate(performance,name=if_else(row_number()==1,"Other",name)),coefficients,importance),error=identity)
stopifnot(inherits(unknown_series,"error"))
original <- plot_federated_comparison(labs,"PSA","Test")
stopifnot(identical(original$scales$get_scales("colour")$palette(2),
  c(`Dana-Farber`="#0072B2",`Federated*`="#222222")))
clinical_fixture <- function() tidyr::expand_grid(endpoint=c("NEPC","avpc","platinum"),landmark_days=c(0,180),
    subpopulation=c("gleason","trio","trio_gleason","other"),
    bundle=c("xgboost_federated_model_adt","cox_federated_elasticnet_model_adt"),
    model=c("transferred_labs","transferred_age","clinical_cox","combined_cox")) %>%
  mutate(model_family=if_else(startsWith(bundle,"xgboost"),"xgboost_cox","elastic_net_cox"),
    n_patients=if_else(subpopulation=="gleason",400,150),n_events=20,n_scored=n_patients,n_scored_events=n_events,
    c_index=case_when(model=="combined_cox"~.74,model=="clinical_cox"~.7,model=="transferred_labs"~.62,TRUE~.55),
    c_index_lo=c_index-.05,c_index_hi=c_index+.05,
    delta_c_vs_transferred=if_else(model=="transferred_labs",NA_real_,c_index-.62),
    delta_lo=delta_c_vs_transferred-.04,delta_hi=delta_c_vs_transferred+.04,
    mean_auc=c_index+.02,mean_auc_lo=mean_auc-.05,mean_auc_hi=mean_auc+.05,
    delta_auc_vs_transferred=if_else(model=="transferred_labs",NA_real_,mean_auc-.64),
    delta_auc_lo=delta_auc_vs_transferred-.04,delta_auc_hi=delta_auc_vs_transferred+.04,
    delta_auc_vs_clinical=if_else(model=="combined_cox",mean_auc-.72,NA_real_),
    delta_auc_vs_clinical_lo=delta_auc_vs_clinical-.03,delta_auc_vs_clinical_hi=delta_auc_vs_clinical+.03,
    c_index_apparent=NA_real_,mean_auc_apparent=NA_real_,n_folds=5,n_failed_folds=0,n_convergence_warnings=0,
    note=NA_character_)
# Federated Figure 5: a = full-cohort discrimination, b/c/d = vs Gleason, TP53/RB1/PTEN and both.
transfer_metrics <- tidyr::expand_grid(endpoint=c("nepc","avpc","platinum"),
    bundle=c("xgboost_federated_model_adt","cox_federated_elasticnet_model_adt"),
    landmark_days=c(0,180),config=c("both","baseline")) %>%
  mutate(model_family=if_else(startsWith(bundle,"xgboost"),"xgboost_cox","elastic_net_cox"),
    n_patients=1000,n_events=30,c_index=.6,c_index_lo=.55,c_index_hi=.65,
    delta_c_vs_baseline=if_else(config=="both",.05,NA_real_),delta_c_lo=.01,delta_c_hi=.09,
    mean_auc=.62,mean_auc_lo=.57,mean_auc_hi=.67,
    delta_auc_vs_baseline=if_else(config=="both",.06,NA_real_),delta_auc_lo=.02,delta_auc_hi=.1)
transfer_clinical <- mutate(filter(clinical_fixture(),subpopulation!="other"),endpoint=tolower(endpoint))
fig5 <- manuscript_federated_transfer_no_dfci(transfer_metrics,transfer_clinical)
stopifnot(fig5$spec$key=="federated_05_transfer_no_dfci",inherits(fig5$plot,"gtable"),
  identical(fig5$spec$titles,c("Discrimination","Patients with Gleason score","Sequenced patients",
    "Sequenced patients with Gleason score")),
  fig5$spec$height==10.4,nrow(fig5$spec$layout)==4,
  grepl("b: patients with a Gleason score",fig5$legend,fixed=TRUE),
  grepl("c: sequenced patients,",fig5$legend,fixed=TRUE),
  grepl("d: sequenced patients with a Gleason score",fig5$legend,fixed=TRUE),
  grepl("combined Cox",fig5$legend,fixed=TRUE),!grepl("burden",fig5$legend,fixed=TRUE),
  grepl("PLATINUM: 0d 400/20",fig5$legend,fixed=TRUE),!grepl("ΔAUC",fig5$legend,fixed=TRUE),
  !grepl("ΔC",fig5$legend,fixed=TRUE))
fig5_trio <- manuscript_federated_transfer_no_dfci(transfer_metrics,filter(transfer_clinical,subpopulation=="trio"))
stopifnot(identical(fig5_trio$spec$titles,c("Discrimination","Sequenced patients")),
  grepl("b: sequenced patients,",fig5_trio$legend,fixed=TRUE))
fallback_message <- NULL
fig5_a <- withCallingHandlers(manuscript_federated_transfer_no_dfci(transfer_metrics,NULL),
  message=function(m) {fallback_message <<- c(fallback_message,conditionMessage(m)); invokeRestart("muffleMessage")})
stopifnot(length(fig5_a$spec$titles)==1,fig5_a$spec$height==3.2,
  any(grepl("clinical comparison unavailable",fallback_message,fixed=TRUE)))
# Federated Figure 5 e-h: patient-level transfer scores -> aggregate KM tables.
km_patients_fixture <- function(n=900) {
  set.seed(11)
  score <- rnorm(n); score[1:60] <- 0
  gleason <- sample(c(6:10,NA),n,TRUE)
  sequenced <- seq_len(n)%%2==0
  trio <- matrix(rbinom(3*n,1,.3),n,dimnames=list(NULL,c("TP53_ANY","RB1_ANY","PTEN_ANY")))
  bind_rows(lapply(c(adt="adt",sequencing="sequencing"),function(clock) {
    rows <- if(clock=="adt") seq_len(n) else which(sequenced)
    rates <- c(platinum=4e-4,nepc=2e-4,avpc=1e-3)
    bind_rows(lapply(names(rates),function(endpoint) {
      t <- rexp(n,rates[[endpoint]]*exp(score)*ifelse(!is.na(gleason)&gleason>=8,2,1)); censor <- runif(n,300,3000)
      bind_rows(lapply(c(elastic_net_cox="cox_federated_elasticnet_model_adt",xgboost_cox="xgboost_federated_model_adt"),
        function(bundle) tibble(clock=clock,endpoint=toupper(endpoint),bundle=bundle,
          model_family=if(startsWith(bundle,"xgboost")) "xgboost_cox" else "elastic_net_cox",landmark_days=180,
          person_id=seq_len(n),risk_score=if(startsWith(bundle,"xgboost")) round(score) else score,
          duration_days=pmin(t,censor),event=as.integer(t<=censor),GLEASON_SCORE=gleason,
          HAS_SEQUENCING=if(clock=="sequencing") "True" else NA_character_,
          TP53_ANY=if(clock=="sequencing") trio[,1] else NA_real_,RB1_ANY=if(clock=="sequencing") trio[,2] else NA_real_,
          PTEN_ANY=if(clock=="sequencing") trio[,3] else NA_real_,SOMATIC_N_GENES_ALTERED=NA_real_)[rows,]))
    }))
  }))
}
km_dir <- tempfile("transfer-km-"); dir.create(km_dir)
readr::write_csv(km_patients_fixture(),file.path(km_dir,FEDERATED_TRANSFER_KM_FILE))
km_input <- load_federated_transfer_km(km_dir)
km_groups <- federated_transfer_km_groups(km_input)
# One cutpoint per clock x bundle x landmark: a patient's score group never depends on the endpoint.
shared <- km_groups %>% group_by(clock,bundle,person_id) %>%
  summarise(n_extreme=n_distinct(score_extreme),n_half=n_distinct(score_half),
    n_quintile=n_distinct(score_quintile),.groups="drop")
tied <- filter(km_groups,clock=="adt",bundle=="cox_federated_elasticnet_model_adt",person_id<=60)
stopifnot(all(shared$n_extreme==1),all(shared$n_half==1),all(shared$n_quintile==1),
  n_distinct(tied$score_extreme)==1,n_distinct(tied$score_half)==1,n_distinct(tied$score_quintile)==1,
  all(is.na(km_groups$trio_group[km_groups$clock=="adt"])),
  all(is.na(km_groups$gleason_group[km_groups$clock=="sequencing"])),
  all(!is.na(km_groups$trio_group[km_groups$clock=="sequencing"])),
  setequal(na.omit(km_groups$score_gleason),federated_transfer_cross(FEDERATED_TRANSFER_GLEASON)))
en <- filter(km_groups,clock=="adt",bundle=="cox_federated_elasticnet_model_adt",endpoint=="nepc")
stopifnot(abs(mean(en$score_extreme %in% "Top 20%")-.2)<.02,
  identical(sort(unique(na.omit(en$score_quintile))),paste0("Q",1:5)))
km_tables <- federated_transfer_km_tables(km_input)
stopifnot(identical(names(km_tables),names(FEDERATED_TRANSFER_KM_TABLES)),
  setequal(unique(km_tables$summary$view),names(FEDERATED_TRANSFER_KM_VIEWS)),
  all(km_tables$summary$clock[km_tables$summary$view %in% c("trio","score_trio")]=="sequencing"),
  all(km_tables$hr$clock[km_tables$hr$view=="gleason"]=="adt"),
  setequal(unique(km_tables$gradient$stratifier),names(FEDERATED_TRANSFER_GRADIENT)),
  all(km_tables$gradient$event_free>=0 & km_tables$gradient$event_free<=1,na.rm=TRUE),
  nrow(filter(km_tables$hr,is.finite(hr)))>0,
  # Rounded XGBoost scores tie across quintile edges: no quintile gradient rather than mislabeled bins.
  !any(km_tables$gradient$stratifier=="score_quintile" & km_tables$gradient$model_family=="xgboost_cox"))
score_hr <- filter(km_tables$hr,view=="score",endpoint=="avpc",model_family=="elastic_net_cox")
stopifnot(score_hr$hr>1,score_hr$n_reference+score_hr$n_compared<nrow(en))
# Panels below the event floor are dropped, not drawn.
stopifnot(is.null(federated_transfer_km_fit(mutate(en,event=0L),"score_extreme",c("Bottom 20%","Top 20%"))))
p_km <- plot_federated_transfer_km(km_tables,"score_gleason","nepc","elastic_net_cox",180)
stopifnot(inherits(p_km,"ggplot"),identical(p_km$labels$y,"NEPC-free probability"),
  grepl("Log-rank",p_km$labels$subtitle,fixed=TRUE))
fig5_km <- manuscript_federated_transfer_no_dfci(transfer_metrics,transfer_clinical,km_tables)
stopifnot(inherits(fig5_km$plot,"gtable"),length(fig5_km$spec$titles)==8,
  identical(fig5_km$spec$titles[1:4],fig5$spec$titles),
  nrow(fig5_km$spec$layout)==6,ncol(fig5_km$spec$layout)==2,
  all(fig5_km$spec$layout[1:4,1]==fig5_km$spec$layout[1:4,2]),
  identical(as.vector(t(fig5_km$spec$layout[5:6,])),5:8),
  abs(fig5_km$spec$height-(10.4+2*MANUSCRIPT_TRANSFER_KM_ROW_HEIGHT))<1e-9,
  grepl("e: Kaplan-Meier NEPC-free survival",fig5_km$legend,fixed=TRUE),
  grepl("f: Kaplan-Meier NEPC-free survival from the 180-day landmark after the sequencing date",fig5_km$legend,fixed=TRUE),
  grepl("h: Univariable Cox hazard ratios",fig5_km$legend,fixed=TRUE))
# Without the clinical comparison the KM panels follow panel a.
fig5_km_a <- suppressMessages(manuscript_federated_transfer_no_dfci(transfer_metrics,NULL,km_tables))
stopifnot(length(fig5_km_a$spec$titles)==5,grepl("b: Kaplan-Meier",fig5_km_a$legend,fixed=TRUE),
  identical(fig5_km_a$spec$layout[4,],c(5L,NA)))
# Standalone pages land in 05_transfer/km/ with aggregate (not patient-level) sidecars.
km_root <- tempfile("transfer-km-out-")
render_federated_transfer_km(km_root,km_dir,dpi=40,overwrite=TRUE)
km_out <- file.path(km_root,FIGURE_SECTIONS[["transfer"]],"km")
stopifnot(file.exists(file.path(km_out,c("transfer_km_score_gleason_nepc_elasticnet_landmark180__platinum.png",
    "transfer_km_score_trio_avpc_xgboost_landmark180__platinum.png","transfer_km_grid_elasticnet_landmark180__platinum.png",
    "transfer_km_gradient_elasticnet_landmark180__platinum.png","transfer_km_hr_xgboost_landmark180__platinum.png"))),
  file.exists(file.path(km_out,"data",paste0(FEDERATED_TRANSFER_KM_TABLES,"__platinum.csv"))),
  !any(grepl("person_id",unlist(lapply(file.path(km_out,"data",paste0(FEDERATED_TRANSFER_KM_TABLES,"__platinum.csv")),
    readLines,n=1L)),fixed=TRUE)))
missing_km <- tryCatch(load_federated_transfer_km(tempdir()),warning=identity)
stopifnot(inherits(missing_km,"warning"))
cat("Manuscript data fidelity, missing cells, audit notes and isolated styling passed.\n")
