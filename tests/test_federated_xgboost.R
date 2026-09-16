source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_data_cache.R")
source("COMPASS/survival_analysis/federated_no_msk_figures.R")
local({
  root <- tempfile("federated-xgb-",tmpdir="/private/tmp"); dir.create(root)
  on.exit(unlink(root,recursive=TRUE))
  path <- file.path(root,"input.csv")
  fails <- function(expr,pattern) {
    e <- tryCatch(force(expr),error=identity)
    stopifnot(inherits(e,"error"),grepl(pattern,conditionMessage(e)))
  }
  metrics <- expand_grid(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(analysis_label="adt",endpoint="platinum",model="xgboost_cox",cohort="all",
      test_c_index=if_else(config=="both",.78,.61),test_mean_auc_t=if_else(config=="both",.8,.62),
      train_val_c_index=.99,train_val_mean_auc_t=.99,n_test=100,n_events_test=10)
  extras <- bind_rows(mutate(metrics,analysis_label="arpi"),mutate(metrics,endpoint="nepc"),
    mutate(metrics,cohort="other"),mutate(metrics,model="cox"))
  write_csv(bind_rows(metrics,extras),path)
  loaded <- load_federated_xgboost(path)
  stopifnot(nrow(loaded)==6,identical(loaded$test_c_index,metrics$test_c_index))
  d <- prepare_federated_xgboost_performance(loaded)
  stopifnot(all(d$auc==d$test_mean_auc_t),all(d$cindex==d$test_c_index),!any(d$auc==.99))
  colors <- c("XGBoost Survival"="#B58900","XGBoost baseline (age)"="#E0CC8A")
  p <- plot_model_discrimination(d,"auc","Test Mean AUC(t)",colors,show_legend=TRUE)
  stopifnot(nrow(p$data)==6,all(p$data$value %in% c(.8,.62)),
    identical(p$scales$get_scales("fill")$palette(2),colors))
  partial <- prepare_federated_xgboost_performance(loaded[-1,])
  stopifnot(nrow(partial)==6,sum(is.na(partial$auc))==1)
  write_csv(bind_rows(metrics,metrics[1,]),path); fails(load_federated_xgboost(path),"Duplicate")
  write_csv(mutate(metrics,test_c_index=1.1),path); fails(load_federated_xgboost(path),"Invalid")
  write_csv(mutate(metrics,n_events_test=101),path); fails(load_federated_xgboost(path),"Invalid")
  write_csv(select(metrics,-endpoint),path); fails(load_federated_xgboost(path),"missing")
  features <- c("Prostate_specific_Ag__Mass_volume__in_Serum_or_Plasma__max",
    "Erythrocyte__DistWidth__Entitic_volume__by_Automated_count__last",
    "Platelets____volume__in_Blood_by_Automated_count__n_observations",
    "Testosterone__Mass_volume__in_Serum_or_Plasma__missing","age","person_id","Body_height__mean")
  importance <- expand_grid(landmark_days=c(0,90,180),feature=features) %>%
    mutate(analysis_label="adt",endpoint="platinum",gain=seq_len(n()))
  write_csv(importance,path)
  imp <- load_federated_xgboost(path,"importance")
  stopifnot(identical(imp$gain,as.numeric(importance$gain)),
    identical(imp$lab_name[1:3],c("PSA","RDW","Platelets")),
    identical(imp$feature_stat[1:4],c("max","last","n_observations","missing")),
    sum(imp$identifier_feature)==3,grepl("person_id",federated_xgboost_audit_note(imp)))
  panel <- plot_model_importance(filter(imp,landmark_days==0,feature!="age"),"xgb","Test")
  stopifnot(nrow(panel$data)==5,!any(panel$data$lab_name=="Body height"),
    "person_id" %in% panel$data$feature,all(diff(panel$data$gain)<=0),
    as.character(panel$data$category[panel$data$lab_name=="PSA"])=="Androgen axis")
  many <- tibble(lab_name=paste("Lab",1:20),feature_stat="mean",gain=1:20)
  top <- plot_model_importance(many,"xgb","Test")
  stopifnot(nrow(top$data)==15,identical(top$data$gain,20:6))
  write_csv(bind_rows(importance,importance[1,]),path); fails(load_federated_xgboost(path,"importance"),"Duplicate")
  write_csv(mutate(importance,gain=-1),path); fails(load_federated_xgboost(path,"importance"),"Invalid")
  stopifnot(is.null(suppressWarnings(load_federated_xgboost(file.path(root,"absent.csv")))))
  # Either optional delivery can render independently. Missing landmarks are
  # explicit, and figure data/receipts are never written into the output tree.
  xdir <- file.path(root,"federated_xgboost"); dir.create(xdir)
  write_csv(filter(importance,landmark_days==0),file.path(xdir,"xgboost_federated_importance_adt.csv"))
  output <- file.path(root,"figures"); dir.create(output)
  scenes <- prepare_figure_scenes(file.path(root,"data"),"test",function()
    suppressWarnings(render_federated_xgboost(output,file.path(root,"fed.csv"),60,TRUE)))
  stopifnot(length(scenes$scenes)==1,scenes$scenes[[1]]$stem=="xgboost_importance")
  exported <- read_csv(file.path(output,"xgboost_importance__platinum.csv"),show_col_types=FALSE)
  stopifnot(nrow(exported)==7,sum(exported$displayed)==5,
    all(exported$input_audit_note==federated_xgboost_audit_note(filter(imp,landmark_days==0))))
  metric_root <- file.path(root,"metrics-only")
  dir.create(file.path(metric_root,"federated_xgboost"),recursive=TRUE)
  write_csv(metrics,file.path(metric_root,"federated_xgboost","xgboost_federated_metrics_adt.csv"))
  metric_scenes <- prepare_figure_scenes(file.path(root,"metrics-data"),"test",function()
    suppressWarnings(render_federated_xgboost(output,file.path(metric_root,"fed.csv"),60,TRUE)))
  stopifnot(length(metric_scenes$scenes)==1,metric_scenes$scenes[[1]]$stem=="xgboost_performance",
    length(list.files(output,pattern="rds$",recursive=TRUE))==0)
})
cat("Federated XGBoost: test-only metrics, isolation, feature parsing, shared styling, audit, missing inputs and validation passed.\n")
