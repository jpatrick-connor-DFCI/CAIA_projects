source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_workflow.R")

# The correction includes hidden height/unstable estimates, excludes non-mean
# statistics, and remains separate by endpoint and landmark.
d <- tibble(lab_name=c("Testosterone","PSA","Body height","Albumin","PSA","Sodium"),
  feature_stat=c(rep("mean",4),"max","mean"),
  coef_feature=c(-log(2),log(1.2),5,-.1,1,.1),
  ci_lower=c(.4,1.1,20,.8,2,.8),ci_upper=c(.7,1.3,200,1.1,4,1.2),
  p_value=c(.01,.02,.04,.2,1e-30,NA),q_value=.99,
  endpoint="platinum",landmark_days=0)
m <- prepare_mean_univariable(d)
stopifnot(nrow(m)==5,all(m$feature_stat=="mean"),all(m$n_mean_tests==4),
  isTRUE(all.equal(m$q_value[1:4],c(.04,.04, .04*4/3,.2))),is.na(m$q_value[5]),
  all(m$q_value_original==.99),identical(m$p_value,d$p_value[c(1:4,6)]),
  m$risk_label[1]=="Lower Testosterone",m$risk_label[2]=="Higher PSA",
  m$risk_hr[1]==2,m$risk_ci_lower[1]==1/.7,m$risk_ci_upper[1]==1/.4,
  m$risk_ci_lower[2]==1.1,m$risk_ci_upper[2]==1.3)
filtered <- prepare_mean_univariable(filter(m,lab_name=="Testosterone"))
stopifnot(filtered$q_value==.04,filtered$n_mean_tests==4)
separate <- prepare_mean_univariable(bind_rows(d,mutate(d,landmark_days=180,p_value=p_value*2),
  mutate(d,endpoint="nepc",p_value=p_value*3)))
stopifnot(separate$q_value[separate$landmark_days==180 & separate$lab_name=="Testosterone"]==.08,
  separate$q_value[separate$endpoint=="nepc" & separate$lab_name=="Testosterone"]==.12)
stopifnot(inherits(tryCatch(prepare_mean_univariable(bind_rows(d,d[1,])),error=identity),"error"))

# Plotting and export preparation cannot change the family after visual filters.
p <- plot_volcano_panel(m,"0 days")
repel <- Filter(function(layer) inherits(layer$geom,"GeomTextRepel"),p$layers)[[1]]$data
stopifnot(nrow(repel)==3,all(repel$risk_coef>=0),!"Body height" %in% repel$lab_name,
  all(repel$n_mean_tests==4),repel$q_value[repel$lab_name=="Testosterone"]==.04)
invisible(ggplot_build(p))
path <- tempfile(fileext=".csv")
readr::write_csv(d,path)
export <- federated_volcano_data(path)
unlink(path)
stopifnot(nrow(export)==3,all(export$n_mean_tests==4),export$q_value[1]==.04)
cat("Mean-only BH families, direction/CI reversal, audit columns and display exclusions passed.\n")
