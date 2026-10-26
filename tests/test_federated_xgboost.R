source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_workflow.R")
source("COMPASS/survival_analysis/checklist_supplement.R")
fails <- function(expr,pattern) {
  e <- tryCatch(force(expr),error=identity)
  stopifnot(inherits(e,"error"),grepl(pattern,conditionMessage(e)))
}
# Compound missingness suffixes keep their statistic; unknown suffixes fall
# back to the last component, and names without "__" have no statistic.
stopifnot(identical(federated_feature_stat(c(
  "Leukocytes____volume__in_Blood_by_Automated_count__delta__missing",
  "Platelets____volume__in_Blood_by_Automated_count__n_observations__missing",
  "Heart_rate__min","Testosterone__Mass_volume__in_Serum_or_Plasma__missing","age")),
  c("delta missing","n_observations missing","min","missing","")))
stopifnot(identical(federated_lab_label(c("aPTT_in_Blood_by_Coagulation_assay__max",
  "Lactate_dehydrogenase__Enzymatic_activity_volume__in_Serum_or_Plasma__last")),c("aPTT","LDH")))
local({
  root <- tempfile("federated-xgb-",tmpdir="/private/tmp"); dir.create(root)
  on.exit(unlink(root,recursive=TRUE))
  path <- file.path(root,"input.csv")
  metrics <- expand_grid(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(analysis_label="adt",endpoint="platinum",model="xgboost_cox",cohort="all",
      test_c_index=if_else(config=="both",.78,.61),test_mean_auc_t=if_else(config=="both",.8,.62),
      train_val_c_index=.99,train_val_mean_auc_t=.99,n_test=100,n_events_test=10)
  extras <- bind_rows(mutate(metrics,analysis_label="arpi"),mutate(metrics,endpoint="nepc"),
    mutate(metrics,cohort="other"),mutate(metrics,model="elastic_net_cox"))
  write_csv(bind_rows(metrics,extras),path)
  loaded <- load_federated_multivariable(path,"xgboost","metrics")
  stopifnot(nrow(loaded)==6,identical(loaded$test_c_index,metrics$test_c_index))
  d <- prepare_federated_performance(loaded,"xgboost")
  stopifnot(all(d$auc==d$test_mean_auc_t),all(d$cindex==d$test_c_index),!any(d$auc==.99),
    all(d$model=="xgboost_cox"),all(d$family=="xgboost"))
  colors <- c("XGBoost Survival"="#B58900","XGBoost baseline (age)"="#E0CC8A")
  stopifnot(identical(FEDERATED_MODEL_COLORS[names(colors)],colors))
  p <- plot_model_discrimination(d,"auc","Test Mean AUC(t)",colors,show_legend=TRUE)
  stopifnot(nrow(p$data)==6,all(p$data$value %in% c(.8,.62)),
    identical(p$scales$get_scales("fill")$palette(2),colors))
  partial <- prepare_federated_performance(loaded[-1,],"xgboost")
  stopifnot(nrow(partial)==6,sum(is.na(partial$auc))==1)
  load_metrics <- function() load_federated_multivariable(path,"xgboost","metrics")
  write_csv(bind_rows(metrics,metrics[1,]),path); fails(load_metrics(),"Duplicate")
  write_csv(mutate(metrics,test_c_index=1.1),path); fails(load_metrics(),"Invalid")
  write_csv(mutate(metrics,n_events_test=101),path); fails(load_metrics(),"Invalid")
  write_csv(select(metrics,-endpoint),path); fails(load_metrics(),"missing")
  features <- c("Prostate_specific_Ag__Mass_volume__in_Serum_or_Plasma__max",
    "Erythrocyte__DistWidth__Entitic_volume__by_Automated_count__last",
    "Platelets____volume__in_Blood_by_Automated_count__n_observations",
    "Testosterone__Mass_volume__in_Serum_or_Plasma__missing","age","person_id","Body_height__mean")
  importance <- expand_grid(landmark_days=c(0,90,180),feature=features) %>%
    mutate(analysis_label="adt",endpoint="platinum",gain=seq_len(n()))
  write_csv(importance,path)
  load_importance <- function() load_federated_multivariable(path,"xgboost","features")
  imp <- load_importance()
  stopifnot(identical(imp$gain,as.numeric(importance$gain)),
    identical(imp$lab_name[1:3],c("PSA","RDW","Platelets")),
    identical(imp$feature_stat[1:4],c("max","last","n_observations","missing")),
    sum(imp$identifier_feature)==3,grepl("person_id",federated_input_audit_note(imp,"xgboost")))
  panel <- plot_model_importance(filter(imp,landmark_days==0,feature!="age"),"xgb","Test")
  stopifnot(nrow(panel$data)==5,!any(panel$data$lab_name=="Body height"),
    "person_id" %in% panel$data$feature,all(diff(panel$data$gain)<=0),
    as.character(panel$data$category[panel$data$lab_name=="PSA"])=="Androgen axis")
  many <- tibble(lab_name=paste("Lab",1:20),feature_stat="mean",gain=1:20)
  top <- plot_model_importance(many,"xgb","Test")
  stopifnot(nrow(top$data)==15,identical(top$data$gain,20:6))
  write_csv(bind_rows(importance,importance[1,]),path); fails(load_importance(),"Duplicate")
  write_csv(mutate(importance,gain=-1),path); fails(load_importance(),"Invalid")
  stopifnot(is.null(suppressWarnings(load_federated_multivariable(file.path(root,"absent.csv"),"xgboost"))))
  # Either optional delivery can render independently. Missing landmarks are
  # explicit, and figure data/receipts are never written into the output tree.
  xdir <- file.path(root,"federated_xgboost"); dir.create(xdir)
  write_csv(filter(importance,landmark_days==0),file.path(xdir,"xgboost_federated_importance_adt.csv"))
  output <- file.path(root,"figures","ADT"); dir.create(output,recursive=TRUE)
  scenes <- prepare_figure_scenes(file.path(root,"data"),"test",function()
    suppressWarnings(render_federated_multivariable(output,file.path(root,"fed.csv"),"xgboost",60,TRUE)))
  stopifnot(length(scenes$scenes)==1,scenes$scenes[[1]]$stem=="xgboost_importance")
  exported <- read_csv(file.path(output,"04_multivariate","federated","data","xgboost_importance__platinum.csv"),show_col_types=FALSE)
  stopifnot(nrow(exported)==7,sum(exported$displayed)==5,
    all(exported$input_audit_note==federated_input_audit_note(filter(imp,landmark_days==0),"xgboost")))
  metric_root <- file.path(root,"metrics-only")
  dir.create(file.path(metric_root,"federated_xgboost"),recursive=TRUE)
  write_csv(metrics,file.path(metric_root,"federated_xgboost","xgboost_federated_metrics_adt.csv"))
  metric_scenes <- prepare_figure_scenes(file.path(root,"metrics-data"),"test",function()
    suppressWarnings(render_federated_multivariable(output,file.path(metric_root,"fed.csv"),"xgboost",60,TRUE)))
  stopifnot(length(metric_scenes$scenes)==1,metric_scenes$scenes[[1]]$stem=="xgboost_performance",
    length(list.files(output,pattern="rds$",recursive=TRUE))==0)
})
local({
  # Elastic-net Cox shares the schema: signed coefficients, not gains.
  root <- tempfile("federated-enet-",tmpdir="/private/tmp"); dir.create(root)
  on.exit(unlink(root,recursive=TRUE))
  cox_dir <- file.path(root,"federated_cox_multivariate"); dir.create(cox_dir)
  paths <- federated_multivariable_files(file.path(root,"fed.csv"),"elastic_net")
  stopifnot(identical(unname(basename(unlist(paths))),c("cox_federated_elasticnet_metrics_adt.csv",
    "cox_federated_elasticnet_coefficients_adt.csv")),identical(names(paths),c("metrics","features")))
  metrics <- expand_grid(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(analysis_label="adt",endpoint="platinum",model="elastic_net_cox",cohort="all",
      test_c_index=if_else(config=="both",.75,.62),test_mean_auc_t=if_else(config=="both",.77,.6),
      n_test=100,n_events_test=10,note="fit_ok")
  write_csv(bind_rows(metrics,mutate(metrics,model="xgboost_cox",test_c_index=.5)),paths[["metrics"]])
  loaded <- load_federated_multivariable(paths[["metrics"]],"elastic_net","metrics")
  stopifnot(nrow(loaded)==6,all(loaded$test_c_index %in% c(.75,.62)))
  d <- prepare_federated_performance(loaded,"elastic_net")
  stopifnot(identical(levels(d$name),c("Elastic-Net Cox","Cox baseline (age)")),
    all(as.character(d$name[d$config=="baseline"])=="Cox baseline (age)"))
  features <- c("Testosterone__Mass_volume__in_Serum_or_Plasma__mean",
    "Leukocytes____volume__in_Blood_by_Automated_count__delta__missing",
    "Heart_rate__min","age","Body_height__mean")
  coefficients <- expand_grid(landmark_days=c(0,90,180),feature=features) %>%
    mutate(analysis_label="adt",endpoint="platinum",
      coefficient=rep(c(-.2,-.1,.15,.3,.05),3),hazard_ratio=exp(coefficient))
  write_csv(coefficients,paths[["features"]])
  coefs <- load_federated_multivariable(paths[["features"]],"elastic_net","features")
  stopifnot(identical(coefs$coef,coefs$coefficient),
    identical(coefs$lab_name[1:5],c("Testosterone","WBC","Heart rate","age","Body height")),
    identical(coefs$feature_stat[1:5],c("mean","delta missing","min","","mean")),
    identical(federated_input_audit_note(coefs,"elastic_net"),""))
  panel <- plot_model_importance(filter(coefs,landmark_days==0,feature!="age"),"cox","Test")
  stopifnot(identical(as.character(panel$data$label),
    c("Testosterone (mean)","Heart rate (min)","WBC (delta missing)")))
  write_csv(mutate(coefficients,coefficient=Inf),paths[["features"]])
  fails(load_federated_multivariable(paths[["features"]],"elastic_net","features"),"Invalid")
  write_csv(select(coefficients,-coefficient),paths[["features"]])
  fails(load_federated_multivariable(paths[["features"]],"elastic_net","features"),"missing")
  write_csv(mutate(coefficients,feature=if_else(feature=="age","person_id",feature)),paths[["features"]])
  flagged <- load_federated_multivariable(paths[["features"]],"elastic_net","features")
  stopifnot(grepl("person_id",federated_input_audit_note(flagged,"elastic_net")),
    grepl("coefficients",federated_input_audit_note(NULL,"elastic_net")))
  write_csv(coefficients,paths[["features"]])
  output <- file.path(root,"figures"); dir.create(output)
  scenes <- prepare_figure_scenes(file.path(root,"data"),"test",function()
    render_federated_multivariable(output,file.path(root,"fed.csv"),"elastic_net",60,TRUE))
  stopifnot(identical(vapply(scenes$scenes,`[[`,character(1),"stem"),
    c("elasticnet_performance","elasticnet_coefficients")))
  exported <- read_csv(file.path(output,"04_multivariate","federated","data","elasticnet_coefficients__platinum.csv"),show_col_types=FALSE)
  stopifnot(nrow(exported)==15,sum(exported$displayed)==9,
    !any(exported$displayed[exported$feature %in% c("age","Body_height__mean")]))
  performance <- read_csv(file.path(output,"04_multivariate","federated","data","elasticnet_performance__platinum.csv"),show_col_types=FALSE)
  stopifnot(nrow(performance)==6,setequal(performance$name,c("Elastic-Net Cox","Cox baseline (age)")))
})
local({
  # rhino_data_v2: elastic-net Cox in one directory per variant; XGBoost and DeepHit
  # in one Rhino package per landmark (lm_<days>/{w,no}_DFCI/, optionally under adt/).
  root <- tempfile("federated-v2-",tmpdir="/private/tmp"); dir.create(root)
  on.exit(unlink(root,recursive=TRUE))
  fed <- file.path(root,"cox_federated_univariate_adt.csv")
  put <- function(d,...) {
    path <- file.path(root,...); dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
    write_csv(d,path,na=""); path
  }
  # Landmark XGBoost exports only the full model; lm_90 lacks a C-index.
  xgb <- function(lm,variant) tibble(analysis_label="adt",endpoint="platinum",model="xgboost_cox",
    cohort="all",landmark_days=lm,config="both",test_c_index=if(lm==90) NA_real_ else .7,
    test_mean_auc_t=.7+lm/1000,n_test=100,n_events_test=10)
  for(variant in c("w_DFCI","no_DFCI")) for(lm in c(0,90,180)) {
    sub <- if(lm==0) c("adt","xgboost_federated_metrics_adt.csv") else "xgboost_federated_metrics_adt.csv"
    do.call(put,c(list(xgb(lm,variant)),"federated_xgboost",paste0("lm_",lm),variant,as.list(sub)))
    put(tibble(analysis_label="adt",endpoint="platinum",landmark_days=lm,feature=c("PSA__mean","age"),gain=c(2,1)),
      "federated_xgboost",paste0("lm_",lm),variant,"xgboost_federated_importance_adt.csv")
    put(expand_grid(landmark_days=lm,training_arm=c("landmark","dynamic"),config=c("both","baseline")) %>%
      mutate(analysis_label="adt",endpoint="platinum",test_mean_auc_t=.72,test_c_index=.7,n_test=100,n_events_test=10),
      "federated_deephit",paste0("lm_",lm),variant,"deephit_federated_metrics_adt.csv")
  }
  enet <- put(expand_grid(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(analysis_label="adt",endpoint="platinum",model="elastic_net_cox",cohort="all",
      test_c_index=.7,test_mean_auc_t=.75,n_test=100,n_events_test=10),
    "federated_elastic_net_cox_w_DFCI","cox_federated_elasticnet_metrics_adt.csv")
  for(variant in names(FEDERATED_MULTIVARIABLE_VARIANTS)) {
    marker <- FEDERATED_MULTIVARIABLE_VARIANTS[[variant]]$marker
    paths <- federated_multivariable_files(fed,"xgboost",variant)
    stopifnot(length(paths$metrics)==3L,length(paths$features)==3L,
      all(basename(dirname(sub("/adt/","/",paths$metrics,fixed=TRUE)))==marker))
    metrics <- load_federated_multivariable(paths$metrics,"xgboost","metrics")
    stopifnot(nrow(metrics)==3L,is.numeric(metrics$test_c_index),is.na(metrics$test_c_index[metrics$landmark_days==90]))
    d <- prepare_federated_performance(metrics,"xgboost")
    stopifnot(nrow(d)==6L,all(is.na(d$auc[d$config=="baseline"])),all(d$auc[d$config=="both"]>.69))
    stopifnot(nrow(load_federated_multivariable(paths$features,"xgboost","features"))==6L)
    dh <- load_federated_dynamic_deephit(fed,variant)
    stopifnot(nrow(dh)==3L,all(dh$training_arm=="dynamic"),grepl("; ",dh$source_path[1],fixed=TRUE),
      length(federated_deephit_metric_paths(fed,variant))==7L)
  }
  stopifnot(identical(federated_multivariable_files(fed,"elastic_net")$metrics,enet))
  # No no-DFCI elastic-net export: the legacy path is reported missing.
  missing <- federated_multivariable_files(fed,"elastic_net","no_dfci")$metrics
  stopifnot(identical(missing,file.path(root,"federated_cox_multivariate_no_DFCI","cox_federated_elasticnet_metrics_adt.csv")),
    is.null(suppressWarnings(load_federated_multivariable(missing,"elastic_net","metrics"))))
  output <- file.path(root,"figures"); dir.create(output)
  render_federated_multivariable(output,fed,"xgboost",60,TRUE,"no_dfci")
  performance <- read_csv(file.path(output,"04_multivariate","federated","data",
    "xgboost_performance_no_dfci__platinum.csv"),show_col_types=FALSE)
  stopifnot(nrow(performance)==6L,sum(is.finite(performance$auc))==3L)
  # Two copies of a landmark's export fail instead of silently doubling rows.
  put(xgb(0,"w_DFCI"),"federated_xgboost","lm_0","w_DFCI","xgboost_federated_metrics_adt.csv")
  fails(load_federated_multivariable(federated_multivariable_files(fed,"xgboost")$metrics,"xgboost","metrics"),"Duplicate")
  # A legacy pooled export takes precedence over the per-landmark files.
  legacy <- put(xgb(0,"w_DFCI"),"federated_xgboost","xgboost_federated_metrics_adt.csv")
  stopifnot(identical(federated_multivariable_files(fed,"xgboost")$metrics,legacy))
})
local({
  # rhino_data_v2 predictive checklists: federated_predictive_checklists/lm_<days>/{w,no}_DFCI/.
  root <- tempfile("federated-checklist-",tmpdir="/private/tmp"); dir.create(root)
  on.exit(unlink(root,recursive=TRUE))
  fed <- file.path(root,"cox_federated_univariate_adt.csv")
  put <- function(d,...) {
    path <- file.path(root,"federated_predictive_checklists",...)
    dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
    write_csv(d,path,na=""); path
  }
  # A 2-year horizon is exported but not shown; the age checklist is one item.
  items <- function(lm) bind_rows(
    expand_grid(landmark_days=lm,horizon_days=c(365,730,1825),config="both",position=1:2) %>%
      mutate(feature=c("Prostate_specific_Ag__Mass_volume__in_Serum_or_Plasma__max","Respiratory_rate__last")[position],
        op=c(">=","is_missing")[position],threshold_value=c(72.1234,NA)[position],M=1,N=2),
    expand_grid(landmark_days=lm,horizon_days=c(365,730,1825),config="baseline",position=1) %>%
      mutate(feature="age_at_anchor",op="<=",threshold_value=58.06,M=1,N=1))
  metrics <- function(lm,auc) expand_grid(landmark_days=lm,horizon_days=c(365,730,1825),
    config=c("both","baseline"),dataset=c("train_val","test")) %>%
    mutate(status="ok",n=1000,n_cases=20,auc=if_else(config=="both",auc,.5),auc_lower=auc-.1,
      auc_upper=auc+.1,brier=.01,sensitivity=.6,specificity=.8,ppv=.05,npv=.99,
      across(c(n,n_cases),~if_else(dataset=="test",.x/4,.x)))
  for(lm in c(0,90,180)) for(variant in c("w_DFCI","no_DFCI")) {
    sub <- if(lm==0) "adt" else character()
    do.call(put,c(list(items(lm)),paste0("lm_",lm),variant,as.list(sub),"predictive_checklist_checklists_adt.csv"))
    do.call(put,c(list(metrics(lm,if(variant=="w_DFCI") .7 else .6)),paste0("lm_",lm),variant,as.list(sub),
      "predictive_checklist_metrics_adt.csv"))
  }
  stopifnot(all(lengths(federated_checklist_files(fed))==3L),all(lengths(federated_checklist_files(fed,"no_dfci"))==3L))
  loaded <- load_federated_checklists(fed)
  stopifnot(nrow(loaded$metrics)==12L,all(loaded$metrics$dataset=="test"),all(loaded$metrics$n==250),
    all(loaded$metrics$auc[loaded$metrics$config=="both"]==.7),
    identical(sort(unique(loaded$metrics$model)),c("age_checklist","checklist")),
    nrow(loaded$items)==18L,!any(loaded$items$horizon_days==730))
  stopifnot(identical(sort(unique(loaded$items$display_item)),
    c("Age <= 58.1","PSA (max) >= 72.1","Respiratory rate (last) is missing")))
  no_dfci <- load_federated_checklists(fed,"no_dfci")$metrics
  stopifnot(nrow(no_dfci)==12L,all(no_dfci$auc[no_dfci$config=="both"]==.6))
  output <- file.path(root,"figures"); dir.create(output)
  render_federated_checklist_cards(output,fed,60,TRUE)
  out <- file.path(output,"04_multivariate","federated")
  stopifnot(file.exists(file.path(out,"checklist_supplement__platinum.png")))
  sidecar <- read_csv(file.path(out,"data","checklist_supplement_metrics__platinum.csv"),show_col_types=FALSE)
  stopifnot(nrow(sidecar)==12L,nrow(read_csv(file.path(out,"data","checklist_supplement_items__platinum.csv"),
    show_col_types=FALSE))==18L)
  # Fewer items than N, a duplicated landmark, out-of-range AUC and a missing export.
  put(filter(items(90),!(horizon_days==365 & config=="both" & position==2)),
    "lm_90","w_DFCI","predictive_checklist_checklists_adt.csv")
  fails(render_federated_checklist_cards(output,fed,60,TRUE),"landmark 90, 1-year lists 1 items but N = 2")
  put(items(90),"lm_90","w_DFCI","predictive_checklist_checklists_adt.csv")
  put(metrics(0,.7),"lm_0","w_DFCI","predictive_checklist_metrics_adt.csv")
  fails(load_federated_checklists(fed),"Duplicate federated checklist")
  put(metrics(0,1.2),"lm_0","w_DFCI","predictive_checklist_metrics_adt.csv")
  unlink(file.path(root,"federated_predictive_checklists","lm_0","w_DFCI","adt"),recursive=TRUE)
  fails(load_federated_checklists(fed),"Invalid federated checklist auc")
  unlink(Sys.glob(file.path(root,"federated_predictive_checklists","lm_*","w_DFCI","predictive_checklist_metrics_adt.csv")))
  stopifnot(is.null(suppressWarnings(load_federated_checklists(fed))))
})
cat("Federated elastic-net/XGBoost: test-only metrics, isolation, feature parsing, shared styling, audit, missing inputs, rhino_data_v2 layout, predictive checklists and validation passed.\n")
