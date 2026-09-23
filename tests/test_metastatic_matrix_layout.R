source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_workflow.R")
source("COMPASS/survival_analysis/figure_supplements.R")
local({
  root <- tempfile("matrix-layout-",tmpdir="/private/tmp"); dir.create(root)
  on.exit(unlink(root,recursive=TRUE))
  old_options <- options(compass.figure_data_manifest=list(metastatic_labels=file.path(root,"labels.parquet")))
  on.exit(options(old_options),add=TRUE)
  old_extra <- Sys.getenv("COMPASS_METASTATIC_EXTRA_PANELS",unset=NA_character_)
  Sys.setenv(COMPASS_METASTATIC_EXTRA_PANELS="0")
  on.exit(if(is.na(old_extra)) Sys.unsetenv("COMPASS_METASTATIC_EXTRA_PANELS") else
    Sys.setenv(COMPASS_METASTATIC_EXTRA_PANELS=old_extra),add=TRUE)
  labels <- tibble(DFCI_MRN=as.character(1:8),
    ADT_LABEL=rep(c("Local","Metastatic"),4),
    LLM_LABEL=c("Local","Local","Metastatic","Metastatic","Local",NA,"Metastatic","Local"),
    REGEX_MAX_ANY=c("Local","Metastatic","Local","Metastatic",NA,"Local","Metastatic","Local"),
    ANCHOR_DELTA_DAYS=0,ANALYSIS_ANCHOR_DATE="2020-01-01")
  nanoparquet::write_parquet(labels,file.path(root,"labels.parquet"))
  output <- Sys.getenv("COMPASS_MATRIX_REVIEW_DIR",file.path(root,"figures"))
  dir.create(output,recursive=TRUE,showWarnings=FALSE)
  captured <- list()
  scenes <- prepare_figure_scenes(file.path(root,"data"),"square-matrices",function()
    render_metastatic_supplements(list(),tibble(DFCI_MRN=1:8),tibble(),
      save_panel=function(p,stem,width,height) {
        captured[[stem]] <<- p
        stopifnot(width==7,height==6)
        pair <- metastatic_label_pairs()[[sub("^adt_labels_","",stem)]]
        counts <- metastatic_pair_counts(labels,pair[1],pair[2])
        stopifnot(nrow(p$data)==4,sum(p$data$n)==counts$paired,
          identical(p$data$fraction,counts$counts$fraction))
        built <- ggplot_build(p)
        stopifnot(isTRUE(all.equal(p$coordinates$aspect(built$layout$panel_params[[1]]),1)))
        getOption("compass.figure_capture")(p,file.path(output,stem),width,height,stem)
      },notify=function(...) NULL))
  stopifnot(length(scenes$scenes)==3,setequal(names(captured),metastatic_supplement_stems()))
  for(scene in scenes$scenes) {
    stopifnot(scene$width==7,scene$height==6)
    render_figure_scene(scene,scenes$signature,120,FALSE)
  }
  stopifnot(length(list.files(output,pattern="png$"))==3,
    length(list.files(output,pattern="rds$",recursive=TRUE))==0)
})
cat("Label matrices: 7x6 exports, square cells, unchanged counts/fractions and cached rendering passed.\n")
