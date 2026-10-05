source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_workflow.R")
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
  output <- file.path(root,"figures","ADT","federated"); dir.create(output,recursive=TRUE)
  scenes <- prepare_figure_scenes(file.path(root,"data"),"test",function()
    suppressWarnings(render_federated_multivariable(output,file.path(root,"fed.csv"),"xgboost",60,TRUE)))
  stopifnot(length(scenes$scenes)==1,scenes$scenes[[1]]$stem=="xgboost_importance")
  exported <- read_csv(file.path(output,"data","xgboost_importance__platinum.csv"),show_col_types=FALSE)
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
  stopifnot(identical(unname(basename(paths)),c("cox_federated_elasticnet_metrics_adt.csv",
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
  exported <- read_csv(file.path(output,"data","elasticnet_coefficients__platinum.csv"),show_col_types=FALSE)
  stopifnot(nrow(exported)==15,sum(exported$displayed)==9,
    !any(exported$displayed[exported$feature %in% c("age","Body_height__mean")]))
  performance <- read_csv(file.path(output,"data","elasticnet_performance__platinum.csv"),show_col_types=FALSE)
  stopifnot(nrow(performance)==6,setequal(performance$name,c("Elastic-Net Cox","Cox baseline (age)")))
})
cat("Federated elastic-net/XGBoost: test-only metrics, isolation, feature parsing, shared styling, audit, missing inputs and validation passed.\n")
