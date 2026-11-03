source("COMPASS/figures/COMPASS_generate_figures_pipeline.R")
source("COMPASS/figures/figure_workflow.R")

local({
  root <- tempfile("retired-transfer-checklists-"); dir.create(root)
  on.exit(unlink(root,recursive=TRUE))
  config <- list(fig_root=file.path(root,"figures"),cache_root=file.path(root,"cache"))
  folder <- file.path(config$fig_root,"ADT_figures","06_transfer_checklists")
  data_folder <- file.path(config$fig_root,"ADT_figure_data","06_transfer_checklists")
  dir.create(folder,recursive=TRUE); dir.create(data_folder,recursive=TRUE)
  retired <- c(file.path(folder,c("transfer_clinical_checklist_xgboost_cox_avpc__platinum.pdf",
    "transfer_clinical_checklist_elastic_net_cox_nepc__platinum.png")),
    file.path(data_folder,"transfer_clinical_checklist_metrics__platinum.csv"))
  keep <- file.path(folder,c("transfer_predictive_checklist_supplement__platinum.pdf",
    "transfer_clinical_checklist_other_avpc__platinum.pdf"))
  for(path in c(retired,keep)) writeLines(basename(path),path)
  # A non-federated run must not touch these files.
  figure_archive_retired_transfer_checklists(config,list())
  stopifnot(all(file.exists(c(retired,keep))))
  prepared <- list(federated=list(scenes=list(list(destination="other"))))
  moved <- figure_archive_retired_transfer_checklists(config,prepared)
  stopifnot(setequal(moved,retired),!any(file.exists(retired)),all(file.exists(keep)))
  archived <- list.files(file.path(config$cache_root,"previous_exports"),recursive=TRUE,full.names=TRUE)
  stopifnot(setequal(basename(archived),basename(retired)),
    all(vapply(archived,function(path) identical(readLines(path),basename(path)),logical(1))))
  stopifnot(length(figure_archive_retired_transfer_checklists(config,prepared))==0L)
})
cat("Retired transfer checklist exports archived without affecting frozen checklists.\n")
