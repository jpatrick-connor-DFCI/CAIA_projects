source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_data_cache.R")
local({
  root <- tempfile("volcano-exports-",tmpdir="/private/tmp"); dir.create(root)
  on.exit(unlink(root,recursive=TRUE))
  output <- Sys.getenv("COMPASS_VOLCANO_REVIEW_DIR",file.path(root,"figures"))
  d <- expand_grid(lab_name=c("PSA","Testosterone","Hemoglobin","Albumin",
    "Alkaline phosphatase","Sodium","Body weight","Platelets","WBC","Creatinine"),
    feature_stat=c("mean","min","max","last","delta")) %>%
    mutate(coef_feature=rep(c(.3,-.5,-.25,-.4,.6,.2,-.3,.45,.2,-.1),each=5)+rep(seq(-.08,.08,length.out=5),10),
      p_value=10^-seq(.3,18,length.out=n()),q_value=if_else(p_value<.01,.02,.3))
  source_plots <- list()
  observed <- list()
  # Observe the actual ggplots entering the compiler, before text is converted
  # into gtables; no production plotting behavior is replaced.
  original_grob <- figure_grob
  assign("figure_grob",function(plot,width,height,directory=tempdir()) {
    if(inherits(plot,"ggplot")) observed[[length(observed)+1L]] <<- plot
    original_grob(plot,width,height,directory)
  },envir=.GlobalEnv)
  on.exit(assign("figure_grob",original_grob,envir=.GlobalEnv),add=TRUE)
  build <- function() for(endpoint in c("platinum","nepc")) for(lm in c(0,90,180)) {
    data <- mutate(d,coef_feature=coef_feature*(1+lm/90),landmark_days=lm)
    # An unavailable landmark must not silently disappear from the comparison.
    p <- if(endpoint=="nepc" && lm==180) ggplot()+theme_void()+
      annotate("text",x=0,y=0,label="No data at 180 days") else plot_volcano_panel(data,paste(lm,"days"))
    stem <- paste0("figure3_univariate_",endpoint,"_landmark",lm)
    source_plots[[stem]] <<- p
    destination <- file.path(output,"ADT","by_figure","figure3",paste0("univariate_",endpoint,"_landmark",lm),
      paste0(endpoint,"__all__incl"))
    getOption("compass.figure_capture")(p,destination,9,7,stem)
  }
  m <- prepare_figure_scenes(file.path(root,"data"),"volcano-test",build)
  stopifnot(length(m$scenes)==8,!anyDuplicated(vapply(m$scenes,`[[`,character(1),"destination")))
  singles <- Filter(function(s) startsWith(s$stem,"figure3_univariate_"),m$scenes)
  combined <- Filter(function(s) s$stem=="labs_univariate_all_landmarks",m$scenes)
  stopifnot(length(singles)==6,length(combined)==2,
    all(vapply(singles,function(s) s$width==9 && s$height==7,logical(1))),
    all(vapply(combined,function(s) s$width==24 && s$height==9 && s$landmark=="0,90,180",logical(1))),
    all(grepl("/associations/",vapply(m$scenes,`[[`,character(1),"destination"))))
  compiled_panels <- Filter(function(p) !is.null(p$labels$title) &&
    startsWith(p$labels$title,"Landmark ") && length(p$coordinates$limits$x)==2,observed)
  stopifnot(length(compiled_panels)>=5)
  for(p in compiled_panels) {
    lm <- as.numeric(gsub("[^0-9]","",p$labels$title))
    repel <- Filter(function(layer) inherits(layer$geom,"GeomTextRepel"),p$layers)[[1]]
    original <- source_plots[[paste0("figure3_univariate_platinum_landmark",lm)]]
    source_repel <- Filter(function(layer) inherits(layer$geom,"GeomTextRepel"),original$layers)[[1]]
    stopifnot(p$theme$axis.text$size>=12,p$theme$plot.caption$size>=11,repel$aes_params$size>=4.2,
      nrow(repel$data)==nrow(d),identical(repel$data,source_repel$data),
      identical(p$labels$caption,original$labels$caption),
      identical(p$scales$get_scales("fill")$limits,LEGEND_ORDER))
  }
  # First endpoint's three compiled panels (plus its shared-legend measurement)
  # share limits covering every original individual plot.
  shared_x <- compiled_panels[[1]]$coordinates$limits$x
  for(p in compiled_panels[1:4]) stopifnot(identical(p$coordinates$limits$x,shared_x))
  for(p in source_plots[1:3]) stopifnot(shared_x[1]<=p$coordinates$limits$x[1],shared_x[2]>=p$coordinates$limits$x[2])
  for(scene in m$scenes) render_figure_scene(scene,m$signature,120,FALSE)
  stopifnot(length(list.files(output,pattern="png$",recursive=TRUE))==8,
    length(list.files(output,pattern="rds$",recursive=TRUE))==0)
  # Warm preparation and rendering reuse both output forms.
  reused <- prepare_figure_scenes(file.path(root,"data"),"volcano-test",function() stop("Unexpected rebuild"))
  stopifnot(length(reused$scenes)==8)
  for(scene in reused$scenes) stopifnot(render_figure_scene(scene,reused$signature,120,FALSE)$rendered==0)
})
cat("Univariate figures: individual + combined exports, endpoint isolation, missing landmarks, common axes, large labels, unchanged estimates and resume passed.\n")
