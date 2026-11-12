# Real R entry points: independent preparation, render from snapshot, PDF/SVG,
# recoverable reuse, checksum validation and no patient files in figure exports.
local({
  root <- tempfile("checklist-augmentation-",tmpdir=Sys.getenv("COMPASS_TEST_OUTPUT_ROOT",tempdir()))
  dir.create(root)
  if(Sys.getenv("COMPASS_TEST_KEEP_OUTPUT")=="1") message("Synthetic augmentation outputs: ",root)
  else on.exit(unlink(root,recursive=TRUE))
  input <- file.path(root,"data","checklist_augmentation")
  aggregate <- file.path(input,"aggregate")
  dir.create(aggregate,recursive=TRUE)
  metrics <- expand.grid(cohort=c("gleason","genomic"),endpoint=c("platinum","avpc"),
    model=c("clinical","age_labs","augmented","L"),stringsAsFactors=FALSE)
  metrics$clock <- ifelse(metrics$cohort=="gleason","adt","sequencing")
  metrics$auc <- ifelse(metrics$model=="augmented",.74,.68)
  metrics$auc_lo <- metrics$auc-.06; metrics$auc_hi <- metrics$auc+.06
  metrics$n_patients <- ifelse(metrics$cohort=="gleason",400,120)
  metrics$n_events_horizon <- ifelse(metrics$endpoint=="platinum",42,30)
  contrast <- expand.grid(cohort=c("gleason","genomic"),endpoint=c("platinum","avpc"),stringsAsFactors=FALSE)
  contrast$model <- "augmented"; contrast$reference <- "clinical"
  contrast$delta_auc <- .06; contrast$delta_auc_lo <- -.02; contrast$delta_auc_hi <- .13
  risks <- expand.grid(endpoint=c("platinum","avpc"),gleason_high=0:1,flag=0:1,stringsAsFactors=FALSE)
  risks$n_patients <- 100; risks$n_events_horizon <- 10
  risks$risk <- .1+.05*risks$gleason_high+.06*risks$flag
  risks$risk_lo <- risks$risk-.05; risks$risk_hi <- risks$risk+.05
  risks$estimand <- "net event risk (death censored)"
  items <- data.frame(feature=c("Alanine_aminotransferase__min","Alkaline_phosphatase__max","Testosterone__max"),
    M=2,N=3,threshold_value=c(15,100,550),op=c("<=",">=","<="),units=c("U/L","U/L","ng/dL"))
  tables <- list(metrics=metrics,contrasts=contrast,four_groups=risks,frozen_items=items,
    reproduction=data.frame(check="synthetic",status="ok"))
  for(name in names(tables)) write.csv(tables[[name]],file.path(aggregate,paste0(name,".csv")),row.names=FALSE)
  publish_manifest <- function(directory=input,landmark=90L) {
    paths <- list.files(file.path(directory,"aggregate"),full.names=TRUE,pattern="[.]csv$")
    jsonlite::write_json(list(schema_version=1L,status="complete",inference="SYNTHETIC TEST — conditional diagnostic",
      settings=list(landmark=landmark,horizon=1825L),
      files=data.frame(name=basename(paths),md5=unname(tools::md5sum(paths)))),
      file.path(directory,"run_manifest.json"),auto_unbox=TRUE,pretty=TRUE)
  }
  publish_manifest()
  private <- file.path(input,"patient_level"); dir.create(private)
  writeLines("person_id,score\n12345,2",file.path(private,"predictions.csv"))
  Sys.setenv(COMPASS_DATA_ROOT=file.path(root,"data"),COMPASS_FIG_ROOT=file.path(root,"exports"),
    COMPASS_FIGURE_DATA_ROOT=file.path(root,"cache"),COMPASS_MANUSCRIPT_FIG_ROOT=file.path(root,"figures_repo"),
    COMPASS_CHECKLIST_AUGMENTATION_ROOT=input,COMPASS_FIGURE_GROUPS="checklist_augmentation",
    COMPASS_AUGMENTATION_LANDMARKS="90")
  entry <- function(name, success=TRUE) {
    log <- file.path(root,paste0(name,".log"))
    result <- system2(file.path(R.home("bin"),"Rscript"),file.path("COMPASS/notebooks/figures",paste0(name,".R")),stdout=log,stderr=log)
    if(success && result!=0L) stop(paste(readLines(log),collapse="\n"))
    if(!success) stopifnot(result!=0L)
  }
  entry("final_figure_data_prep")
  output_root <- file.path(root,"figures_repo","main","alternative_06")
  output <- file.path(output_root,"lm90_h1825")
  stopifnot(!dir.exists(output_root))
  # Cached render remains independent of patient/source availability.
  stopifnot(file.rename(input,paste0(input,".hidden")))
  entry("run_figure_render")
  paths <- list.files(output,pattern="[.](pdf|svg)$",full.names=TRUE)
  stopifnot(length(paths)==8L,all(file.info(paths)$size>1000))
  before <- file.info(paths)$mtime
  entry("run_figure_render")
  stopifnot(identical(before,file.info(paths)$mtime))
  stopifnot(!any(grepl("patient|prediction|fold",list.files(output,recursive=TRUE))))
  # A truncated SVG is repaired even when a receipt exists.
  svg <- paths[grepl("[.]svg$",paths)][1]
  writeLines("broken",svg)
  entry("run_figure_render")
  stopifnot(file.info(svg)$size>1000)
  stopifnot(file.rename(paste0(input,".hidden"),input))
  # Stale source mutation cannot pass preparation by relying on old results.
  write("",file.path(aggregate,"metrics.csv"),append=TRUE)
  entry("final_figure_data_prep",success=FALSE)
  # Missing genomic analysis becomes a truthful placeholder, not fabricated points.
  tables$metrics <- metrics[metrics$cohort=="gleason",]
  write.csv(tables$metrics,file.path(aggregate,"metrics.csv"),row.names=FALSE)
  publish_manifest()
  entry("final_figure_data_prep")
  entry("run_figure_render")
  snapshot <- readRDS(file.path(root,"cache","checklist_augmentation","prepared.rds"))
  stopifnot(grepl("Unavailable",snapshot$plots$panel_c_genomic$plot$layers[[1]]$data$label))
  primary_times <- file.info(paths)$mtime
  for(lm in c(0L,180L)) {
    cell <- sprintf("lm%d_h1825",lm)
    cell_dir <- file.path(input,cell)
    dir.create(file.path(cell_dir,"aggregate"),recursive=TRUE)
    cell_tables <- tables
    cell_tables$metrics <- metrics
    cell_tables$frozen_items <- items[seq_len(if(lm==0L) 3L else 2L),]
    cell_tables$frozen_items$M <- 1L
    cell_tables$frozen_items$N <- nrow(cell_tables$frozen_items)
    cell_tables$frozen_items$threshold_value[1] <- 20
    for(name in names(cell_tables)) {
      table <- cell_tables[[name]]; table$landmark_days <- lm; table$horizon_days <- 1825L
      write.csv(table,file.path(cell_dir,"aggregate",paste0(name,".csv")),row.names=FALSE)
    }
    publish_manifest(cell_dir,lm)
  }
  Sys.setenv(COMPASS_AUGMENTATION_LANDMARKS="90,0,180")
  entry("final_figure_data_prep")
  entry("run_figure_render")
  stopifnot(length(list.files(output_root,pattern="[.](pdf|svg)$",recursive=TRUE))==24L,
    !length(list.files(output_root,pattern="[.](pdf|svg)$")),
    setequal(list.files(output_root),paste0("lm",c(0,90,180),"_h1825")))
  stopifnot(identical(primary_times,file.info(paths)$mtime))
  for(lm in c(0L,180L)) {
    cell <- sprintf("lm%d_h1825",lm)
    snapshot <- readRDS(file.path(root,"cache","checklist_augmentation",cell,"prepared.rds"))
    stopifnot(grepl(sprintf("ADT +%d days",lm),snapshot$plots$panel_b_gleason$plot$labels$subtitle,fixed=TRUE),
      snapshot$manifest$settings$landmark==lm,snapshot$tables$frozen_items$M[1]==1L,
      file.exists(file.path(output_root,cell,"06_checklist_augmentation.pdf")))
  }
  # A mislabeled receipt cannot substitute a different landmark's results.
  publish_manifest(file.path(input,"lm0_h1825"),180L)
  Sys.setenv(COMPASS_AUGMENTATION_LANDMARKS="0")
  entry("final_figure_data_prep",success=FALSE)
  publish_manifest(file.path(input,"lm0_h1825"),0L)
  message("Checklist augmentation R workflow passed")
})
