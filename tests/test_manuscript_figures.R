source("COMPASS/figures/COMPASS_generate_figures_pipeline.R")
source("COMPASS/figures/figure_workflow.R")
source("COMPASS/figures/checklist_supplement.R")

# Styling a manuscript copy must not mutate a layer shared by the standalone.
p <- ggplot(tibble(x=1,y=2,label="Unchanged"),aes(x,y,label=label)) +
  geom_text(size=6) + theme_classic() +
  theme(axis.title.y=element_text(size=20),axis.text.x=element_text(size=18))
q <- manuscript_style(p,"Copy","A")
stopifnot(p$layers[[1]]$aes_params$size==6,q$layers[[1]]$aes_params$size==FIGURE_FONT_SIZES[["body"]]/ggplot2::.pt,
  q$theme$axis.title.y$size==9,q$theme$axis.title.y$face=="bold",q$theme$axis.text.x$size==8,
  identical(p$data,q$data),is.null(q$labels$title),identical(q$labels$tag,"a"))
flow <- manuscript_style(p+theme(axis.text=element_blank()),"Flow","A")
stopifnot(inherits(flow$theme$axis.text.x,"element_blank"),inherits(flow$theme$axis.text.y,"element_blank"))
specs <- manuscript_specs()
spec_keys <- vapply(specs,`[[`,character(1),"key")
spec_by_key <- function(key) specs[[match(key,spec_keys)]]
# Figure 1 has panels a-e only (no data-availability or federated site-count panels).
cohort_specs <- specs[spec_keys=="01_cohort_overview"]
stopifnot(length(cohort_specs)==1,cohort_specs[[1]]$height==7.1,
  identical(cohort_specs[[1]]$members,MANUSCRIPT_COHORT_MEMBERS),
  max(cohort_specs[[1]]$layout)==5,length(cohort_specs[[1]]$titles)==5)
# Figure 2: one spec per cohort (the ADT day-0 cohort), keyed with a variant suffix.
stopifnot(identical(spec_keys[startsWith(spec_keys,"02_")],paste0("02_llm_nepc_avpc__",names(FIGURE2_COHORTS))),
  identical(names(FIGURE2_COHORTS),"lm0"),
  identical(spec_by_key("02_llm_nepc_avpc__lm0")$members,figure2v3_stems("lm0")),
  all(vapply(specs[startsWith(spec_keys,"02_")], `[[`, numeric(1), "width")==10),
  identical(figure2v3_variant(c("figure2v3_icd_metric_bar","figure2v3_lm0_enrichment","figure2v3_metric_bar",
    "figure2v3_xyz_enrichment")),c(NA,"lm0",NA,NA)),
  identical(figure2v3_panel("figure2v3_lm0_subtype_landscape"),"subtype_landscape"),
  manuscript_base_key("02_llm_nepc_avpc__lm0")=="02_llm_nepc_avpc",
  identical(manuscript_figure2_header("lm0","ADT day-0 (N=4,000; 1,234 with an LLM label)"),
    "Cohort: ADT day-0 prediction cohort with LLM label (n = 1,234)"),
  identical(manuscript_figure2_header("lm0",NULL),"Cohort: ADT day-0 prediction cohort with LLM label"))
fed3_spec <- spec_by_key("federated_03_univariable")
stopifnot(fed3_spec$height==8.9,fed3_spec$width==10,identical(fed3_spec$row_heights,c(1.5,.32,1,1)),
  identical(fed3_spec$shared_legend_panels,1:3),all(fed3_spec$layout[2,]==8),
  identical(manuscript_dfci_volcano_ylim,c(-.2,36)),
  manuscript_dfci_km_landmark==180,
  grepl("landmark180",fed3_spec$members[5],fixed=TRUE),
  grepl("landmark180",fed3_spec$members[7],fixed=TRUE),
  identical(manuscript_dfci_landmark_headers,c("+0 days","+90 days","+180 days")),
  identical(manuscript_km_y_label,"Platinum-free\nprobability"),
  !any(c("03_dfci_univariable","04_multivariable_labs") %in% spec_keys))
# The HR direction guide sits under the shared volcano legend.
direction <- hr_direction_grob()
direction_text <- unname(unlist(lapply(direction$children,function(x) if(inherits(x,"text")) x$label)))
stopifnot(inherits(direction,"gTree"),identical(direction_text,c("1","Larger association with platinum hazard in the labeled direction")),
  identical(MANUSCRIPT_UNIVARIABLE_KEYS,"federated_03_univariable"))
landmark_header <- manuscript_center_landmark_header(ggplot()+labs(subtitle="+90 days"))
stopifnot(landmark_header$theme$plot.subtitle$face=="bold",
  landmark_header$theme$plot.subtitle$hjust==.5,
  landmark_header$theme$plot.subtitle$size==9)

# Captions mirror the paired Markdown/text layout used by the clinical text
# embedding project and retain figure-specific audit language.
captions <- manuscript_captions()
grid_keys <- paste0("federated_06_checklist_no_dfci__lm",rep(c(0,90,180),each=2),"_h",c(365,1825))
federated_keys <- c("federated_03_univariable","federated_04_multivariable","federated_04_multivariable__full_features",
  "federated_05_transfer_no_dfci__top_bottom_20_all_endpoints","federated_05_transfer_no_dfci__no_d",
  "federated_06_checklist_no_dfci",grid_keys)
local_keys <- c("01_cohort_overview","02_llm_nepc_avpc__lm0","03_cohort_sensitivity_supplement")
stopifnot(length(captions)==15,setequal(names(captions),c(local_keys,federated_keys)),
  all(startsWith(unname(captions[local_keys]),paste0("Figure ",c(1,2,3),". "))),
  all(grepl("^Federated Figure ([3-6])[. ]",unname(captions[federated_keys[c(1,2,5,6)]]))),
  setequal(unique(manuscript_base_key(names(captions))),names(MANUSCRIPT_SECTIONS)),
  grepl("any dated platinum",captions[["02_llm_nepc_avpc__lm0"]],fixed=TRUE),
  !grepl("(f) DFCI patients",captions[["01_cohort_overview"]],fixed=TRUE),
  grepl("bottom and top 10%",captions[["federated_03_univariable"]],fixed=TRUE),
  !grepl("Kaplan-Meier",captions[["federated_05_transfer_no_dfci__no_d"]],fixed=TRUE),
  !grepl("Gleason score (d)",captions[["federated_05_transfer_no_dfci__no_d"]],fixed=TRUE),
  grepl("22,138 patients (664 platinum events)",captions[["federated_03_univariable"]],fixed=TRUE),
  grepl("Gleason score and age (b)",captions[["federated_05_transfer_no_dfci__no_d"]],fixed=TRUE),
  grepl("(c-e) Kaplan-Meier",captions[["federated_06_checklist_no_dfci"]],fixed=TRUE),
  grepl("same sequenced patients with a Gleason score",captions[["federated_06_checklist_no_dfci"]],fixed=TRUE),
  !any(grepl("C-index",unname(captions),fixed=TRUE)),
  grepl("4,429/133",captions[["federated_04_multivariable"]],fixed=TRUE))
caption_root <- tempfile("manuscript-caption-test-")
caption_paths <- manuscript_write_captions(caption_root)
stopifnot(length(caption_paths)==30,all(file.exists(caption_paths)))
# The federated set is numbered on its own; each caption sits in the
# manuscript folder of its figure's section.
caption_md <- caption_paths[endsWith(caption_paths,".md")]
stopifnot(setequal(basename(caption_md),paste0(c("figure1","figure2_lm0","figure3_supplement",
  paste0("federated_figure",c(3,4,6)),"federated_figure4_full_features","federated_figure5_no_d",
  "federated_figure5_top_bottom_20_all_endpoints",
  paste0("federated_figure6_lm",rep(c(0,90,180),each=2),"_h",c(365,1825))),".md")))
for(path in caption_md) stopifnot(identical(readLines(path,warn=FALSE),
  readLines(sub("[.]md$",".txt",path),warn=FALSE)))
caption_file <- function(section,name) file.path(caption_root,section,
  "manuscript","captions",paste0(name,".md"))
stopifnot(startsWith(readLines(caption_file("03_univariate/federated","federated_figure3"),n=1,warn=FALSE),"Federated Figure 3. "),
  file.exists(caption_file("01_cohort","figure1")),!file.exists(caption_file("02_llm","figure2")),
  file.exists(caption_file("02_llm","figure2_lm0")),
  file.exists(caption_file("05_transfer_survival","federated_figure5_no_d")),
  startsWith(readLines(caption_file("06_transfer_checklists","federated_figure6"),n=1,warn=FALSE),"Federated Figure 6. "),
  file.exists(caption_file("03_univariate/cohort_sensitivity","figure3_supplement")),
  file.exists(caption_file("04_multivariate/federated","federated_figure4")),
  !dir.exists(file.path(caption_root,"03_univariate","internal")),
  !dir.exists(file.path(caption_root,"04_multivariate","internal")),
  setequal(names(MANUSCRIPT_SECTIONS),manuscript_base_key(names(captions))))
unlink(caption_root,recursive=TRUE)

site_incidence <- tibble(site_name=c("dana_farber_caia_1_1","fred_hutch_caia_1_1","jhu_caia_1_1"),
  site=c("Dana-Farber","Fred Hutch","Johns Hopkins"))
# Figure 1 site panel: every site with a day-0 result, events/patients labels,
# a dagger under 25 events; sites without a day-0 row are dropped.
fig1_sites <- tibble(site_name=c(site_incidence$site_name,"msk_caia_1_1"),
    site=c(site_incidence$site,"MSK"),landmark_days=c(0,0,0,90),endpoint="platinum",
    n_patients=c(200,100,50,80),n_events=c(30,8,2,5))
fig1_site_plot <- figure1_site_counts_plot(fig1_sites)
stopifnot(inherits(fig1_site_plot,"ggplot"),nrow(fig1_site_plot$data)==3,
  identical(fig1_site_plot$data$count_label[order(fig1_site_plot$data$site)],
    c("30/200","8/100†","2/50†")),
  grepl("fewer than 25 events",fig1_site_plot$labels$subtitle,fixed=TRUE))

# Federated extreme KM summaries: schema v3 carries its fraction; legacy
# quintile exports (q20/q80) read as 20%.
legacy <- federated_km_summary_cutoffs(tibble(q20=1,q80=9))
current <- federated_km_summary_cutoffs(tibble(q_low=1,q_high=9,extreme_fraction=.1))
from_manifest <- federated_km_summary_cutoffs(tibble(q_low=1,q_high=9),fraction=.1)
stopifnot(legacy$extreme_fraction==.2,identical(c(legacy$q_low,legacy$q_high),c(1,9)),
  current$extreme_fraction==.1,from_manifest$extreme_fraction==.1,
  inherits(tryCatch(federated_km_summary_cutoffs(tibble(q_low=1,q_high=9)),error=identity),"error"),
  inherits(tryCatch(federated_km_summary_cutoffs(tibble(q=1)),error=identity),"error"))
fed_km_fixture <- function(fraction) {
  labels <- figure_extreme_labels(fraction)
  list(km_summary=tibble(scope="pooled",site_name=NA_character_,cutoff_scope="pooled",landmark_days=180,
      lab="PSA",status="ok",n_bottom=50,n_top=50,events_bottom=2,events_top=12,logrank_p=.001,
      q_low=.2,q_high=9.6,extreme_fraction=fraction),
    km=tidyr::expand_grid(stratum=labels,time=c(0,100,200)) %>%
      mutate(scope="pooled",site_name=NA_character_,cutoff_scope="pooled",landmark_days=180,lab="PSA",
        survival=1-time/1000*(stratum==labels[2]),lower=survival-.02,upper=pmin(1,survival+.02)))
}
fed_km <- plot_federated_lab_km(fed_km_fixture(.1),"PSA",180)
fed_km_legacy <- plot_federated_lab_km(fed_km_fixture(.2),"PSA",180)
stopifnot(inherits(fed_km,"ggplot"),grepl("bottom vs top 10%",fed_km$labels$title,fixed=TRUE),
  grepl("middle 80% not shown",fed_km$labels$caption,fixed=TRUE),
  grepl("bottom vs top 20%",fed_km_legacy$labels$title,fixed=TRUE),
  federated_extreme_pct(fed_km_fixture(.1))==10)

# Figure 1 is panels a-e only; Figure 2
# gets a cohort header from the captured caption.
blank <- function(caption=NULL) list(plot=ggplot(tibble(x=1,y=1),aes(x,y))+geom_point()+labs(caption=caption))
cohort_items <- setNames(lapply(MANUSCRIPT_COHORT_MEMBERS,function(stem) blank()),MANUSCRIPT_COHORT_MEMBERS)
figure2_items <- setNames(lapply(figure2v3_stems("lm0"),function(stem)
  blank("ADT day-0 prediction cohort (N=2,239; 2,100 with an LLM label)")),figure2v3_stems("lm0"))
built <- manuscript_build(c(cohort_items,figure2_items),character(),tempdir())
stopifnot(setequal(names(built),c("01_cohort_overview","02_llm_nepc_avpc__lm0")),
  max(built[["01_cohort_overview"]]$spec$layout)==5,
  identical(built[["02_llm_nepc_avpc__lm0"]]$spec$header,
    "Cohort: ADT day-0 prediction cohort with LLM label (n = 2,100)"),
  inherits(built[["02_llm_nepc_avpc__lm0"]]$plot,"gtable"))

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
stopifnot(dfci_combined$spec$key=="03_cohort_sensitivity_supplement",
  identical(dfci_combined$spec$layout,rbind(c(1,2),c(3,3))),
  setequal(unique(dfci_combined$data$figure_section),
    c("incidence_and_counts","laboratory_associations")))

# The cohort-sensitivity forest labels the transformed horizontal coordinate directly as ln(HR),
# retaining the same estimates and null position as the HR-scale source plot.
log_axis_source <- ggplot(tibble(hazard_ratio_per_sd=c(.7,1.2),ci_lower=c(.6,1.0),
  ci_upper=c(.9,1.4),cohort=c("a","b")),aes(hazard_ratio_per_sd,cohort)) + geom_point()
log_axis_plot <- manuscript_log_hazard_axis(log_axis_source)
log_axis_scale <- log_axis_plot$scales$get_scales("x")
stopifnot(log_axis_plot$labels$x=="Log hazard ratio per SD (95% CI)",
  any(abs(log(log_axis_scale$breaks))<sqrt(.Machine$double.eps)))

# Federated Figure 4 from the captured federated CSVs. A flagged
# identifier remains in the supplied importance input and audit note.
performance <- bind_rows(lapply(c("elastic_net","xgboost"),function(family)
  expand_grid(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(test_mean_auc_t=.7,test_c_index=.65,n_test=20,n_events_test=4) %>%
    prepare_federated_performance(family) %>%
    mutate(name=as.character(name),input_audit_note=if(.env$family=="xgboost")
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
  federated_models$spec$key=="federated_04_multivariable",
  federated_models$spec$width==8.5,federated_models$spec$height==9,
  length(federated_models$spec$titles)==3,
  inherits(federated_models$plot,"gtable"),nrow(federated_models$data)==12,
  identical(levels(federated_models$data$name),names(FEDERATED_MODEL_COLORS)),
  grepl("Day 90, Cox baseline (age)",federated_models$legend,fixed=TRUE),
  identical(manuscript_federated_feature_label(c("Alkaline phosphatase (max)",
    "Neutrophils absolute (n_observations)","Diastolic blood pressure (min)")),
    c("ALP (max)","Neutrophils abs. (n obs.)","DBP (min)")))
# Main ranking excludes auxiliary features before applying the 15-feature limit.
extra <- tibble(landmark_days=0, feature=c("PSA__n_observations","PSA__mean__missing","age","Body_height__mean","PSA__mean"),
  lab_name=c("PSA","PSA","age","Body height","PSA"),
  feature_stat=c("n_observations","mean missing","","mean","mean"),
  coefficient=c(9,8,7,6,.1),coef=coefficient,displayed=FALSE,identifier_feature=FALSE,input_audit_note=NA_character_)
focused_coefficients <- bind_rows(coefficients,extra)
focused <- manuscript_federated_multivariable(performance,focused_coefficients,importance,
  key="federated_04_multivariable")
full <- manuscript_federated_multivariable(performance,focused_coefficients,importance,
  key="federated_04_multivariable__full_features",full=TRUE)
stopifnot(!"XGBoost baseline (age)" %in% as.character(focused$data$name),
  "XGBoost baseline (age)" %in% as.character(full$data$name),
  "PSA__mean" %in% focused$panels[[2]]$data$feature,
  !any(extra$feature[1:4] %in% focused$panels[[2]]$data$feature),
  all(extra$feature %in% full$panels[[2]]$data$feature),
  "person_id" %in% full$panels[[5]]$data$feature,
  !"person_id" %in% focused$panels[[5]]$data$feature)
unknown_series <- tryCatch(manuscript_federated_multivariable(
  mutate(performance,name=if_else(row_number()==1,"Other",name)),coefficients,importance),error=identity)
stopifnot(inherits(unknown_series,"error"))
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
fig5_strips <- levels(prepare_federated_transfer(transfer_metrics,
  federated_transfer_strip_counts(transfer_metrics))$endpoint_label)
stopifnot(fig5$spec$key=="federated_05_transfer_no_dfci",inherits(fig5$plot,"gtable"),
  identical(fig5$spec$titles,c("Discrimination","Patients with Gleason score","Sequenced patients",
    "Sequenced patients with Gleason score")),
  abs(fig5$spec$height-11.2)<1e-9,identical(dim(fig5$spec$layout),c(4L,6L)),
  all(apply(fig5$spec$layout,1,function(row) length(unique(row))==1)),
  grepl("b: patients with a Gleason score",fig5$legend,fixed=TRUE),
  grepl("c: sequenced patients,",fig5$legend,fixed=TRUE),
  grepl("d: sequenced patients with a Gleason score",fig5$legend,fixed=TRUE),
  grepl("combined Cox",fig5$legend,fixed=TRUE),!grepl("burden",fig5$legend,fixed=TRUE),
  grepl("PLATINUM: 0d 400/20",fig5$legend,fixed=TRUE),!grepl("ΔAUC",fig5$legend,fixed=TRUE),
  !grepl("ΔC",fig5$legend,fixed=TRUE),
  # Endpoints run platinum, AVPC, NEPC; strips carry the day-0 counts.
  identical(names(FEDERATED_TRANSFER_ENDPOINTS),c("platinum","avpc","nepc")),
  identical(fig5_strips,paste0(unname(FEDERATED_TRANSFER_ENDPOINTS),"\nDay 0: n = 1,000, 30 events")))
fig5_no_d <- manuscript_federated_transfer_no_dfci(transfer_metrics,transfer_clinical,
  include=c("gleason","trio"),key="federated_05_transfer_no_dfci__no_d")
stopifnot(fig5_no_d$spec$key=="federated_05_transfer_no_dfci__no_d",
  identical(fig5_no_d$spec$titles,fig5$spec$titles[1:3]),
  !grepl("d: sequenced patients with a Gleason score",fig5_no_d$legend,fixed=TRUE))
fig5_trio <- manuscript_federated_transfer_no_dfci(transfer_metrics,filter(transfer_clinical,subpopulation=="trio"))
stopifnot(identical(fig5_trio$spec$titles,c("Discrimination","Sequenced patients")),
  grepl("b: sequenced patients,",fig5_trio$legend,fixed=TRUE))
fallback_message <- NULL
fig5_a <- withCallingHandlers(manuscript_federated_transfer_no_dfci(transfer_metrics,NULL),
  message=function(m) {fallback_message <<- c(fallback_message,conditionMessage(m)); invokeRestart("muffleMessage")})
stopifnot(length(fig5_a$spec$titles)==1,fig5_a$spec$height==3.4,
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
        function(bundle) { xgb <- startsWith(bundle,"xgboost"); seq_clock <- clock=="sequencing"
          scale <- c(platinum=1,nepc=1+1e-9,avpc=1.5)[[endpoint]]
          tibble(clock=clock,endpoint=toupper(endpoint),bundle=bundle,
          model_family=if(xgb) "xgboost_cox" else "elastic_net_cox",landmark_days=180,
          person_id=seq_len(n),risk_score=if(xgb) round(score) else score*scale,
          duration_days=pmin(t,censor),event=as.integer(t<=censor),GLEASON_SCORE=gleason,
          HAS_SEQUENCING=if(seq_clock) "True" else NA_character_,
          TP53_ANY=if(seq_clock) trio[,1] else NA_real_,RB1_ANY=if(seq_clock) trio[,2] else NA_real_,
          PTEN_ANY=if(seq_clock) trio[,3] else NA_real_,SOMATIC_N_GENES_ALTERED=NA_real_)[rows,]}))
    }))
  }))
}
km_dir <- tempfile("transfer-km-"); dir.create(km_dir)
readr::write_csv(km_patients_fixture(),file.path(km_dir,FEDERATED_TRANSFER_KM_FILE))
km_input <- load_federated_transfer_km(km_dir)
km_groups <- federated_transfer_km_groups(km_input)
# Scores differ by endpoint (as transferred scores do); cutpoints are per clock x endpoint x bundle x landmark.
top <- km_groups %>% filter(model_family=="elastic_net_cox") %>% group_by(clock,endpoint,bundle) %>%
  summarise(top=mean(score_extreme %in% "Top 20%"),decile=mean(score_decile %in% "Top 10%"),
    high=mean(score_half %in% "high score"),.groups="drop")
tied <- filter(km_groups,clock=="adt",bundle=="cox_federated_elasticnet_model_adt",person_id<=60)
seq_rows <- filter(km_groups,clock=="sequencing")
# The 60 tied zero scores sit at the median and go low, so the half split is ~47% high.
stopifnot(nrow(top)==6,all(abs(top$top-.2)<.02),all(abs(top$decile-.1)<.02),all(abs(top$high-.5)<.04),
  n_distinct(tied$score_extreme)==1,n_distinct(tied$score_half)==1,n_distinct(tied$score_quartile)==1,
  all(is.na(km_groups$trio_group[km_groups$clock=="adt"])),
  all(is.na(km_groups$trio_pattern[km_groups$clock=="adt"])),
  identical(seq_rows$gleason_group,figure_gleason_groups(seq_rows$GLEASON_SCORE)),
  all(!is.na(seq_rows$trio_group)),
  identical(unname(FEDERATED_TRANSFER_GRADIENT$trio_pattern$levels),
    c("WT","TP53","PTEN","RB1","2 MT","3 MT")))
trio_n <- seq_rows$TP53_ANY+seq_rows$RB1_ANY+seq_rows$PTEN_ANY
stopifnot(all(seq_rows$trio_pattern[trio_n==0]=="WT"),
  all(seq_rows$trio_pattern[trio_n==1 & seq_rows$TP53_ANY==1]=="TP53"),
  all(seq_rows$trio_pattern[trio_n==1 & seq_rows$PTEN_ANY==1]=="PTEN"),
  all(seq_rows$trio_pattern[trio_n==1 & seq_rows$RB1_ANY==1]=="RB1"),
  all(seq_rows$trio_pattern[trio_n==2]=="2 MT"),
  all(seq_rows$trio_pattern[trio_n==3]=="3 MT"),
  setequal(unique(na.omit(seq_rows$trio_pattern)),FEDERATED_TRANSFER_GRADIENT$trio_pattern$levels),
  all(seq_rows$trio_group[trio_n==0]==FEDERATED_TRANSFER_TRIO[1]))
en <- filter(km_groups,clock=="adt",bundle=="cox_federated_elasticnet_model_adt",endpoint=="nepc")
stopifnot(abs(mean(en$score_extreme %in% "Top 20%")-.2)<.02,
  identical(sort(unique(na.omit(en$score_quartile))),paste0("Q",1:4)),
  all(abs(table(en$score_quartile)/nrow(en)-.25)<.03))
# Tied scores never straddle quartiles: a collapsed quartile means no quartile groups at all.
stopifnot(all(is.na(federated_transfer_score_quantile(c(rep(0,8),1,2)))),
  identical(federated_transfer_score_quantile(1:8),paste0("Q",rep(1:4,each=2))),
  identical(federated_transfer_score_quantile(c(1:5,NA,6:8))[6],NA_character_))
km_tables <- federated_transfer_km_tables(km_input)
stopifnot(identical(names(km_tables),names(FEDERATED_TRANSFER_KM_TABLES)),
  identical(names(FEDERATED_TRANSFER_KM_TABLES),c("curves","summary","gradient")),
  setequal(unique(km_tables$summary$view),setdiff(names(FEDERATED_TRANSFER_KM_VIEWS),"checklist")),
  !any(c("score_gleason","hr") %in% names(FEDERATED_TRANSFER_KM_VIEWS)),
  all(km_tables$summary$clock[km_tables$summary$view %in% c("trio","score_trio")]=="sequencing"),
  setequal(unique(km_tables$gradient$stratifier),names(FEDERATED_TRANSFER_GRADIENT)),
  all(km_tables$gradient$clock=="sequencing"),
  setequal(unique(km_tables$gradient$group[km_tables$gradient$stratifier=="trio_pattern"]),
    FEDERATED_TRANSFER_GRADIENT$trio_pattern$levels),
  all(km_tables$gradient$horizon_days==1825),
  all(km_tables$gradient$event_free>=0 & km_tables$gradient$event_free<=1,na.rm=TRUE),
  identical(federated_transfer_horizon_label(),"5-year"))
mixed_km <- km_tables
mixed_row <- which(mixed_km$gradient$model_family=="xgboost_cox" &
  mixed_km$gradient$landmark_days==180)[[1]]
mixed_km$gradient$clock[[mixed_row]] <- "adt"
stopifnot(inherits(tryCatch(plot_federated_transfer_gradient(mixed_km,"xgboost_cox",180),
  error=identity),"error"))
# A group whose every patient has the event before 5 years is 0, not missing;
# follow-up censored before 5 years is not estimable.
stopifnot(federated_transfer_event_free(tibble(duration_days=c(100,400,900),event=1L))[["event_free"]]==0,
  is.na(federated_transfer_event_free(tibble(duration_days=c(100,400,900),event=c(1L,0L,0L)))[["event_free"]]))
# Panels below the event floor are dropped, not drawn.
stopifnot(is.null(federated_transfer_km_fit(mutate(en,event=0L),"score_extreme",c("Bottom 20%","Top 20%"))))
p_km <- plot_federated_transfer_km(km_tables,"score_decile","nepc","elastic_net_cox",180)
stopifnot(inherits(p_km,"ggplot"),identical(p_km$labels$y,"NEPC-free probability"),
  grepl("Log-rank",p_km$labels$subtitle,fixed=TRUE))
# a-d, then one row of platinum/AVPC/NEPC KM panels (XGBoost, day 180), then the gradient.
fig5_km <- manuscript_federated_transfer_no_dfci(transfer_metrics,transfer_clinical,km_tables)
panel_h <- manuscript_federated_transfer_km(km_tables)$gradient
stopifnot(FEDERATED_TRANSFER_KM_FIGURE$model_family=="xgboost_cox",
  FEDERATED_TRANSFER_KM_FIGURE$landmark_days==180L,
  FEDERATED_TRANSFER_KM_HORIZON==1825,
  inherits(panel_h,"ggplot"),
  all(panel_h$data$model_family=="xgboost_cox"),
  all(panel_h$data$landmark_days==180L),
  all(panel_h$data$horizon_days==1825),
  all(panel_h$data$clock=="sequencing"))
stopifnot(inherits(fig5_km$plot,"gtable"),length(fig5_km$spec$titles)==8,
  identical(fig5_km$spec$titles[1:4],fig5$spec$titles),
  identical(fig5_km$spec$titles[5:8],c("Platinum: score extremes","AVPC: score extremes","NEPC: score extremes",
    "5-year event-free probability")),
  identical(dim(fig5_km$spec$layout),c(6L,6L)),
  identical(fig5_km$spec$layout[5,],rep(5:7,each=2)),all(fig5_km$spec$layout[6,]==8),
  abs(fig5_km$spec$height-(11.2+MANUSCRIPT_TRANSFER_ROW_HEIGHTS[["km"]]+MANUSCRIPT_TRANSFER_ROW_HEIGHTS[["gradient"]]))<1e-9,
  grepl("e: Kaplan-Meier platinum-free survival from the 180-day landmark",fig5_km$legend,fixed=TRUE),
  grepl("f: Kaplan-Meier AVPC-free survival",fig5_km$legend,fixed=TRUE),
  grepl("g: Kaplan-Meier NEPC-free survival",fig5_km$legend,fixed=TRUE),
  grepl("top vs bottom 10% of the transferred XGBoost score",fig5_km$legend,fixed=TRUE),
  grepl("h: Kaplan-Meier 5-year event-free probability (95% CI) from 180 days after the sequencing date",fig5_km$legend,fixed=TRUE),
  grepl("All use the sequencing clock (sequenced patients).",fig5_km$legend,fixed=TRUE),
  !grepl("hazard ratio",fig5_km$legend,ignore.case=TRUE))
fig5_km_no_d <- manuscript_federated_transfer_no_dfci(transfer_metrics,transfer_clinical,km_tables,
  include=c("gleason","trio"),key="federated_05_transfer_no_dfci__no_d")
stopifnot(length(fig5_km_no_d$spec$titles)==7,identical(fig5_km_no_d$spec$layout[4,],rep(4:6,each=2)),
  grepl("d: Kaplan-Meier platinum-free",fig5_km_no_d$legend,fixed=TRUE),
  grepl("g: Kaplan-Meier 5-year",fig5_km_no_d$legend,fixed=TRUE))
# Without the clinical comparison the KM panels follow panel a.
fig5_km_a <- suppressMessages(manuscript_federated_transfer_no_dfci(transfer_metrics,NULL,km_tables))
stopifnot(length(fig5_km_a$spec$titles)==5,grepl("b: Kaplan-Meier",fig5_km_a$legend,fixed=TRUE),
  identical(fig5_km_a$spec$layout[2,],rep(2:4,each=2)),all(fig5_km_a$spec$layout[3,]==5))
# Main Figure 5: XGBoost only, the KM panels, then 5-year platinum by score decile, Gleason and any alteration.
fig5_main <- manuscript_federated_transfer_no_dfci(transfer_metrics,NULL,km_tables,include=character(),
  model_families="xgboost_cox",probability=TRUE)
risk <- plot_federated_transfer_horizon_risk(km_tables,"xgboost_cox",180L)
stopifnot(inherits(fig5_main$plot,"gtable"),length(fig5_main$spec$titles)==5,
  tail(fig5_main$spec$titles,1)=="5-year platinum probability",
  all(fig5_main$data$model_family=="xgboost_cox"),
  fig5_main$spec$row_heights[1]==MANUSCRIPT_TRANSFER_ROW_HEIGHTS[["single"]],
  identical(fig5_main$spec$layout[2,],rep(2:4,each=2)),all(fig5_main$spec$layout[3,]==5),
  grepl("e: Kaplan-Meier 5-year platinum probability",fig5_main$legend,fixed=TRUE),
  inherits(risk,"ggplot"),all(risk$data$endpoint=="platinum"),all(risk$data$clock=="sequencing"),
  setequal(unique(risk$data$stratifier),FEDERATED_TRANSFER_RISK_STRATIFIERS),
  setequal(as.character(risk$data$group),c(FEDERATED_TRANSFER_GRADIENT$score_decile$levels,
    FEDERATED_TRANSFER_GLEASON,FEDERATED_TRANSFER_TRIO_COUNTS)),
  identical(risk$labels$y,"5-year platinum probability"),
  # 1 - KM swaps the interval: the event-free upper bound gives the lower probability bound.
  all(abs(risk$data$probability-(1-risk$data$event_free))<1e-12,na.rm=TRUE),
  all(abs(risk$data$probability_lo-(1-risk$data$hi))<1e-12,na.rm=TRUE),
  all(abs(risk$data$probability_hi-(1-risk$data$lo))<1e-12,na.rm=TRUE),
  all(risk$data$probability_lo<=risk$data$probability & risk$data$probability<=risk$data$probability_hi,na.rm=TRUE))
fig5_twenty <- manuscript_federated_transfer_no_dfci(transfer_metrics,NULL,km_tables,include=character(),
  model_families="xgboost_cox",probability=TRUE,extreme_percent=20L)
twenty <- manuscript_federated_transfer_km(km_tables,probability=TRUE,extreme_percent=20L)
score_twenty <- filter(twenty$gradient$data,stratifier=="score_extreme")
stopifnot(length(twenty$km)==3L,
  all(vapply(twenty$km,function(p) setequal(as.character(p$data$stratum),c("Bottom 20%","Top 20%")),logical(1))),
  setequal(as.character(score_twenty$group),c("Bottom 20%","Top 20%")),
  all(score_twenty$clock=="sequencing"),
  grepl("top vs bottom 20%",fig5_twenty$legend,fixed=TRUE),
  grepl("middle 60%",manuscript_transfer_xgboost_caption(20L),fixed=TRUE))
fig5_endpoints <- manuscript_federated_transfer_no_dfci(transfer_metrics,NULL,km_tables,include=character(),
  model_families="xgboost_cox",probability=TRUE,extreme_percent=20L,probability_endpoints=TRUE)
stopifnot(length(fig5_endpoints$panels)==7L,
  identical(fig5_endpoints$spec$layout[3,],rep(5:7,each=2)),
  identical(vapply(fig5_endpoints$panels[5:7],function(p) unique(p$data$endpoint),character(1)),
    names(FEDERATED_TRANSFER_ENDPOINTS)),
  all(vapply(fig5_endpoints$panels[5:7],function(p)
    setequal(unique(p$data$stratifier),c("score_extreme","gleason_group","trio_group")),logical(1))),
  grepl("(e-g)",manuscript_captions()$federated_05_transfer_no_dfci__top_bottom_20_all_endpoints,fixed=TRUE))
wrong_clock <- km_tables; wrong_clock$gradient$clock <- "adt"
clock_error <- tryCatch(plot_federated_transfer_horizon_risk(wrong_clock,"xgboost_cox",180L),error=conditionMessage)
stopifnot(grepl("sequencing-landmark",clock_error,fixed=TRUE))
# Figure 6: checklist card and rule accuracy above KM curves by checklist status.
checklist_dir <- tempfile("transfer-checklist-"); dir.create(checklist_dir)
set.seed(66)
checklist_patients <- tidyr::expand_grid(endpoint=c("PLATINUM","AVPC","NEPC"),person_id=seq_len(300)) %>%
  mutate(bundle="predictive_checklist_no_dfci",landmark_days=180,model_horizon_days=365,
    risk_score=rep(0:4,length.out=n()),flag=as.integer(risk_score>=2),
    duration_days=rexp(n(),.001*(1+flag)),event=rbinom(n(),1L,.4))
readr::write_csv(checklist_patients,file.path(checklist_dir,FEDERATED_CHECKLIST_PATIENTS_FILE))
readr::write_csv(tidyr::expand_grid(endpoint=names(FEDERATED_TRANSFER_ENDPOINTS),rule=names(FEDERATED_CHECKLIST_RULES)) %>%
  mutate(bundle="predictive_checklist_no_dfci",landmark_days=180,horizon_days=365,n_patients=400,n_events=30,
    n_positive=120,auc=.6,auc_lo=.52,auc_hi=.68,delta_auc_vs_checklist=if_else(rule=="checklist",0,-.02),
    delta_auc_lo=-.1,delta_auc_hi=.06),file.path(checklist_dir,FEDERATED_CHECKLIST_COMPARISON_FILE))
checklist_items <- tibble(landmark_days=180,horizon_days=365,config="both",position=1:5,
  display_item=paste("Lab item",1:5),M=2L,N=5L)
checklist_metrics <- tibble(landmark_days=180,horizon_days=365,config=c("both","baseline"),
  model=c("checklist","age_checklist"),status="ok",n=1200,n_cases=90,auc=c(.66,.55),auc_lower=c(.61,.5),
  auc_upper=c(.71,.6),brier=.07,sensitivity=.6,specificity=.7,ppv=.15,npv=.95)
real_load_federated_checklists <- load_federated_checklists
load_federated_checklists <- function(federated_path,variant) list(
  items=bind_rows(checklist_items,mutate(checklist_items,landmark_days=90,M=3L)),
  metrics=bind_rows(checklist_metrics,mutate(checklist_metrics,landmark_days=90)))
checklist_root <- tempfile("transfer-checklist-out-")
checklist_tables <- write_federated_checklist_tables(checklist_root,"unused",checklist_dir)
checklist_data <- setNames(file.path(checklist_root,FIGURE_SECTIONS[["transfer_checklists"]],"data",
  paste0(FEDERATED_CHECKLIST_TABLES,"__platinum.csv")),names(FEDERATED_CHECKLIST_TABLES))
stopifnot(all(file.exists(checklist_data)),
  !any(grepl("person_id",unlist(lapply(checklist_data,readLines,n=1L)),fixed=TRUE)),
  identical(as.integer(checklist_tables$items$position),1:5),all(checklist_tables$items$landmark_days==180),
  all(checklist_tables$items$horizon_days==365),all(checklist_tables$comparison$horizon_days==365),
  all(checklist_tables$summary$view=="checklist"),all(checklist_tables$summary$clock=="adt"),
  setequal(unique(checklist_tables$summary$endpoint),names(FEDERATED_TRANSFER_ENDPOINTS)),
  setequal(checklist_tables$summary$stratum,FEDERATED_CHECKLIST_GROUPS))
stopifnot(FEDERATED_CHECKLIST_FIGURE$landmark_days==180,
  FEDERATED_CHECKLIST_FIGURE$horizon_days==365,
  FEDERATED_FEATURE_SET_FIGURE$horizon_days==1825)
stale_patient <- tryCatch(validate_federated_checklist_cell(
  mutate(checklist_patients,model_horizon_days=1825),FEDERATED_CHECKLIST_FIGURE,
  "model_horizon_days","patients"),error=conditionMessage)
stale_comparison <- tryCatch(validate_federated_checklist_cell(
  mutate(checklist_tables$comparison,horizon_days=1825),FEDERATED_CHECKLIST_FIGURE,
  "horizon_days","comparison"),error=conditionMessage)
stopifnot(grepl("expected landmark 180 and horizon 365",stale_patient,fixed=TRUE),
  grepl("expected landmark 180 and horizon 365",stale_comparison,fixed=TRUE))
# Built from the sidecars, as manuscript_build does.
fig6 <- manuscript_federated_checklist(lapply(checklist_data,readr::read_csv,show_col_types=FALSE))
fig6_card <- federated_checklist_card_lines(checklist_tables$items,checklist_tables$metrics)
stopifnot(fig6$spec$key=="federated_06_checklist_no_dfci",inherits(fig6$plot,"gtable"),
  length(fig6$spec$titles)==5,
  identical(fig6$spec$layout[1,],rep(1:2,each=3L)),identical(fig6$spec$layout[2,],rep(3:5,each=2L)),
  length(fig6$spec$row_heights)==2,fig6$spec$row_heights[2]==MANUSCRIPT_TRANSFER_ROW_HEIGHTS[["km"]],
  grepl("1-year platinum",fig6$legend,fixed=TRUE),grepl("1-year IPCW AUC",fig6$legend,fixed=TRUE),
  any(grepl("Flag when at least 2 of 5 items",fig6_card$text,fixed=TRUE)),
  sum(startsWith(fig6_card$text,"[ ] "))==5,
  grepl("positive at 2 of 5 items",fig6$legend,fixed=TRUE),
  grepl("Platinum Gleason ≥8 120, 0.60 (0.52-0.68), -0.02 (-0.10 to 0.06)",fig6$legend,fixed=TRUE),
  !grepl("Checklist positive 120, 0.60 (0.52-0.68),",fig6$legend,fixed=TRUE),
  grepl("c: Kaplan-Meier platinum-free survival",fig6$legend,fixed=TRUE),
  grepl("e: Kaplan-Meier NEPC-free survival",fig6$legend,fixed=TRUE))
# Checklist transfer supplement: one figure per landmark x horizon, cut from the checklist_grid/ tables.
grid_dir <- file.path(checklist_dir,FEDERATED_CHECKLIST_GRID_DIR); dir.create(grid_dir)
readr::write_csv(bind_rows(checklist_patients,mutate(checklist_patients,landmark_days=90,flag=as.integer(risk_score>=3))),
  file.path(grid_dir,FEDERATED_CHECKLIST_PATIENTS_FILE))
grid_comparison <- readr::read_csv(file.path(checklist_dir,FEDERATED_CHECKLIST_COMPARISON_FILE),show_col_types=FALSE)
readr::write_csv(bind_rows(grid_comparison,mutate(grid_comparison,landmark_days=90)),
  file.path(grid_dir,FEDERATED_CHECKLIST_COMPARISON_FILE))
grid_root <- tempfile("transfer-checklist-grid-")
suppressWarnings(write_federated_checklist_grid_tables(grid_root,"unused",checklist_dir))
grid_cell <- federated_checklist_cell(90L,365L)
grid_tables <- lapply(setNames(file.path(grid_root,FIGURE_SECTIONS[["transfer_checklists"]],"data",
  paste0(FEDERATED_CHECKLIST_TABLES,federated_checklist_cell_suffix(grid_cell),"__platinum.csv")),
  names(FEDERATED_CHECKLIST_TABLES)),readr::read_csv,show_col_types=FALSE)
grid_figure <- manuscript_federated_checklist(grid_tables,federated_checklist_grid_key(grid_cell),grid_cell)
stopifnot(length(list.files(file.path(grid_root,FIGURE_SECTIONS[["transfer_checklists"]],"data"),
    pattern="_lm[0-9]+_h[0-9]+__platinum[.]csv$"))==length(FEDERATED_CHECKLIST_TABLES)*2L,
  nrow(FEDERATED_CHECKLIST_GRID)==6L,
  grid_figure$spec$key=="federated_06_checklist_no_dfci__lm90_h365",
  all(grid_tables$summary$landmark_days==90),
  any(grepl("Flag when at least 3 of 5 items",federated_checklist_card_lines(grid_tables$items,grid_tables$metrics,grid_cell)$text,fixed=TRUE)),
  grepl("from the 90-day ADT landmark",grid_figure$legend,fixed=TRUE),
  identical(manuscript_checklist_caption(),manuscript_checklist_caption_text()),
  grepl("5-year platinum checklist from 0 days after ADT initiation",
    captions[["federated_06_checklist_no_dfci__lm0_h1825"]],fixed=TRUE),
  grepl("within 5 years",captions[["federated_06_checklist_no_dfci__lm0_h1825"]],fixed=TRUE),
  !grepl("180|1-year|1 year",captions[["federated_06_checklist_no_dfci__lm0_h1825"]]),
  identical(manuscript_folder("root","federated_06_checklist_no_dfci__lm0_h365"),
    manuscript_folder("root","federated_06_checklist_no_dfci")))
# manuscript_renders/: main, supplement, and one checklist figure per landmark x horizon.
renders_plan <- manuscript_renders_plan()
stopifnot(nrow(renders_plan)==5L+1L+2L+2L+6L+6L,!anyDuplicated(renders_plan$target),
  setequal(dirname(renders_plan$target),c("main","supplement",file.path("supplement","federated_checklist_transfer"),
    file.path("supplement","generalized_federated_checklist_transfer"))),
  sum(dirname(renders_plan$target)=="main")==5L,
  all(c("supplement/03_cohort_sensitivity_supplement.pdf","supplement/adt_vs_llm__platinum__all__incl.pdf") %in%
    renders_plan$target),
  sum(dirname(renders_plan$target)==file.path("supplement","federated_checklist_transfer"))==6L,
  sum(dirname(renders_plan$target)==file.path("supplement","generalized_federated_checklist_transfer"))==6L,
  "supplement/generalized_federated_checklist_transfer/federated_06_checklist_guideline_no_dfci__lm0_h1825.pdf" %in% renders_plan$target,
  identical(manuscript_folder("root","federated_06_checklist_guideline_no_dfci__lm0_h365"),
    manuscript_folder("root","federated_06_checklist_no_dfci")),
  "main/federated_05_transfer_no_dfci__top_bottom_20_all_endpoints.pdf" %in% renders_plan$target,
  "supplement/federated_checklist_transfer/federated_06_checklist_no_dfci__lm0_h1825.pdf" %in% renders_plan$target,
  all(renders_plan$source %in% readLines("COMPASS/figures/retained_exports.txt")))
renders_root <- tempfile("renders-")
for(source in renders_plan$source[-1L]) {
  dir.create(dirname(file.path(renders_root,"ADT",source)),recursive=TRUE,showWarnings=FALSE)
  writeLines(source,file.path(renders_root,"ADT",source))
}
suppressMessages(figure_publish_manuscript_renders(renders_root))
published <- file.path(renders_root,"ADT",MANUSCRIPT_RENDERS_DIR,renders_plan$target)
stopifnot(identical(file.exists(published),c(FALSE,rep(TRUE,nrow(renders_plan)-1L))),
  identical(readLines(published[2L]),renders_plan$source[2L]))
# The drawn items must be the rule that assigned the patient groups.
readr::write_csv(mutate(checklist_patients,flag=1L-flag),file.path(checklist_dir,FEDERATED_CHECKLIST_PATIENTS_FILE))
flag_error <- tryCatch(write_federated_checklist_tables(checklist_root,"unused",checklist_dir),error=conditionMessage)
stopifnot(grepl("2-of-5 checklist",flag_error,fixed=TRUE),
  is.null(suppressWarnings(write_federated_checklist_tables(checklist_root,"unused",tempfile()))))
load_federated_checklists <- real_load_federated_checklists
# Feature-set supplement: real files on the delivery layout, regular rows beside them.
fs_root <- tempfile("feature-set-"); dir.create(fs_root)
fs_fed <- file.path(fs_root,"cox_federated_univariate_adt.csv")
fs_put <- function(d,...) {
  path <- file.path(fs_root,...)
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  readr::write_csv(d,path)
}
fs_survival <- function(model,auc,n_test=1000) tidyr::expand_grid(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
  mutate(analysis_label="adt",endpoint="platinum",model=model,cohort="all",test_mean_auc_t=auc,test_c_index=auc,
    n_test=n_test,n_events_test=40)
# Standalone pages land in 05_transfer_survival/km/ with aggregate (not patient-level) sidecars.
km_root <- tempfile("transfer-km-out-")
render_federated_transfer_km(km_root,km_dir,dpi=40,overwrite=TRUE)
km_out <- file.path(km_root,FIGURE_SECTIONS[["transfer"]],"km")
stopifnot(file.exists(file.path(km_out,c("transfer_km_score_decile_nepc_elasticnet_landmark180__platinum.pdf",
    "transfer_km_score_trio_avpc_xgboost_landmark180__platinum.pdf","transfer_km_grid_elasticnet_landmark180__platinum.pdf",
    "transfer_km_gradient_xgboost_landmark180__platinum.pdf"))),
  !any(grepl("^transfer_km_(hr|score_gleason)_",list.files(km_out))),
  file.exists(file.path(km_out,"data",paste0(FEDERATED_TRANSFER_KM_TABLES,"__platinum.csv"))),
  !any(grepl("person_id",unlist(lapply(file.path(km_out,"data",paste0(FEDERATED_TRANSFER_KM_TABLES,"__platinum.csv")),
    readLines,n=1L)),fixed=TRUE)))
missing_km <- tryCatch(load_federated_transfer_km(tempdir()),warning=identity)
stopifnot(inherits(missing_km,"warning"))
cat("Manuscript data fidelity, missing cells, audit notes and isolated styling passed.\n")
