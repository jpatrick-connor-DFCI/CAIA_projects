source("COMPASS/survival_analysis/figure_pipeline/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_pipeline/checklist_supplement.R")
source("COMPASS/survival_analysis/figure_pipeline/figure_workflow.R")

local({
  root <- tempfile("transfer-checklist-")
  dir.create(root)
  on.exit(unlink(root,recursive=TRUE))
  input <- file.path(root,"clinical_checklists")
  dir.create(input)
  cells <- tidyr::expand_grid(feature_set="xgboost_cox",endpoint="avpc",
    landmark_days=c(0,90,180),horizon_days=c(365,1825)) %>%
    mutate(population="sequenced_with_gleason",status="ok",reason=NA_character_,
      threshold=2L,n_items=3L,n_test=60L,n_events_total=80L,n_events_test=20L)
  items <- tidyr::crossing(cells[c("feature_set","endpoint","landmark_days","horizon_days")],
    tibble(item_order=1:3,name=c("FEDERATED_SCORE >= 0.12","GLEASON_SCORE >= 8","TP53_ANY >= 1")))
  metrics <- cells %>% mutate(model="checklist",dataset="test",auc=.72,auc_lower=.61,auc_upper=.83,
    brier=.12,sensitivity=.65,specificity=.8,ppv=.5,npv=.9)
  write <- function() {
    readr::write_csv(cells,file.path(input,"checklist_cells.csv"))
    readr::write_csv(items,file.path(input,"checklist_items_long.csv"))
    readr::write_csv(metrics,file.path(input,"checklist_metrics.csv"))
  }
  write()
  captured <- list()
  original <- save_federated_panel
  on.exit(assign("save_federated_panel",original,envir=.GlobalEnv),add=TRUE)
  assign("save_federated_panel",function(p,path,width,height,dpi,overwrite) {
    stopifnot(inherits(p,"gtable"),isTRUE(all.equal(width,15.6)),height>8)
    captured[[length(captured)+1L]] <<- basename(path)
    ggplot2::ggsave(file.path(root,basename(path)),p,width=width,height=height,device=figure_pdf_device)
  },envir=.GlobalEnv)
  withCallingHandlers(render_transfer_clinical_checklists(root,input,60,TRUE),
    warning=function(w) stop(w))
  stopifnot(identical(captured[[1]],"transfer_clinical_checklist_xgboost_cox_avpc__platinum.pdf"))
  lines <- checklist_card_lines(cells[1,],filter(items,landmark_days==0,horizon_days==365),
    metrics[1,],1,origin="sequencing")
  stopifnot(grepl("sequencing",lines$text[1]),!any(grepl("ADT",lines$text)))
  sidecar <- figure_sidecar_path(federated_output_path(root,"transfer_clinical_checklist_metrics",".csv"))
  stopifnot(file.exists(sidecar))
  # Empty/unsupported cells still render all cards and preserve failure rows in sidecars.
  cells <- mutate(cells,status="underpowered",reason="Empty training or test risk set",threshold=NA,n_items=NA)
  items <- items[FALSE,]
  metrics <- mutate(metrics,status="unsupported",auc=NA_real_,brier=NA_real_)
  write()
  withCallingHandlers(render_transfer_clinical_checklists(root,input,60,TRUE),
    warning=function(w) stop(w))
  stopifnot(length(captured)==2L,nrow(readr::read_csv(sidecar,show_col_types=FALSE))==6L)
  stopifnot(is.null(render_transfer_clinical_checklists(root,file.path(root,"absent"),60,TRUE)))
})
