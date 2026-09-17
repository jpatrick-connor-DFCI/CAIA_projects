# Reassemble downloaded PNG exports without requiring clinical inputs or fits.
# Rscript reformat_downloaded_figures.R <downloaded-ADT-dir> <new-parent>/ADT [0|90|180]
# Always writes a separate tree. The input PNGs and their plotted data are unchanged.
suppressPackageStartupMessages({library(png); library(grid); library(gridExtra)})

download_read_panel <- function(path) {
  p <- png::readPNG(path,native=TRUE)
  # nativeRaster storage is row-major, unlike an ordinary R matrix.
  matrix(as.integer(p),nrow=nrow(p),byrow=TRUE)
}

download_native_raster <- function(p) {
  structure(as.integer(t(p)),dim=dim(p),class="nativeRaster")
}

download_text_bands <- function(p) {
  rows <- which(rowSums(p != -1L)>2)
  groups <- split(rows,cumsum(c(TRUE,diff(rows)>1)))
  t(vapply(groups,range,numeric(2)))
}

download_remove_header <- function(p, lines=2L, retain=integer()) {
  bands <- download_text_bands(p)
  stopifnot(nrow(bands)>lines)
  cut <- floor(mean(c(bands[lines,2],bands[lines+1L,1])))
  # Refuse an unexpected input layout instead of cropping a plot/data region.
  stopifnot(cut < nrow(p)*.3)
  result <- p[seq.int(cut+1L,nrow(p)),,drop=FALSE]
  if(length(retain)) {
    kept <- p[seq.int(bands[min(retain),1]-3L,bands[max(retain),2]+3L),,drop=FALSE]
    result <- rbind(result,matrix(-1L,nrow=24L,ncol=ncol(p)),kept,
      matrix(-1L,nrow=12L,ncol=ncol(p)))
  }
  result
}

download_draw_panel <- function(p, x,y,width,height,heading=NULL,canvas_width,canvas_height) {
  grid::pushViewport(grid::viewport(x=x,y=y,width=width,height=height))
  header <- if(length(heading)) .34 else 0
  available_w <- width*canvas_width
  available_h <- height*canvas_height-header
  scale <- min(available_w/ncol(p),available_h/nrow(p))
  if(length(heading)) grid::grid.text(heading,
    x=grid::unit((available_w-ncol(p)*scale)/2,"in")+grid::unit(8,"pt"),
    y=grid::unit(1,"npc")-grid::unit(8,"pt"),
    just=c("left","top"),gp=grid::gpar(fontsize=17,fontface="bold"))
  grid::grid.raster(download_native_raster(p),x=.5,y=grid::unit(1,"npc")-grid::unit(header,"in"),
    just=c("center","top"),width=grid::unit(ncol(p)*scale,"in"),height=grid::unit(nrow(p)*scale,"in"),
    interpolate=FALSE)
  grid::popViewport()
}

reformat_downloaded_figures <- function(input, output, landmark=0L) {
  stopifnot(landmark %in% c(0L,90L,180L),dir.exists(input),!file.exists(output),basename(output)=="ADT")
  dir.create(output,recursive=TRUE)
  stopifnot(all(file.copy(list.files(input,full.names=TRUE),output,recursive=TRUE)))
  files <- list.files(input,pattern="\\.png$",recursive=TRUE)
  pattern <- paste0("^(figure1_cohort|figure2v3_llm|classifier_validation|subtype_platinum|",
    "metastatic_label_agreement|labs_univariate_all_landmarks|somatic_carrier_km_page[0-9]+|",
    "discrimination|sensitivity_(gleason|somatic)|importance_(cox|xgb)|km_extremes|",
    "xgboost_(performance|importance)|event_incidence_lm[0-9]+|metastatic_label_overlap)__")
  selected <- files[grepl(pattern,basename(files)) & !startsWith(files,"compiled/")]
  for(relative in selected) {
    p <- download_read_panel(file.path(input,relative))
    stem <- sub("__.*$","",basename(relative))
    count_line <- startsWith(stem,"sensitivity_")
    incidence <- startsWith(stem,"event_incidence_")
    overlap <- stem=="metastatic_label_overlap"
    p <- download_remove_header(p,lines=if(count_line || incidence) 3L else 2L,
      retain=if(incidence) 2:3 else if(count_line || overlap || startsWith(stem,"xgboost_")) 2L else integer())
    destination <- file.path(output,"compiled",relative)
    dir.create(dirname(destination),recursive=TRUE,showWarnings=FALSE)
    png::writePNG(download_native_raster(p),destination,dpi=200)
    unlink(file.path(output,relative)) # Only the new copy, never the source.
  }
  identity <- "platinum__all__incl"
  path <- function(relative) file.path(input,paste0(relative,"__",identity,".png"))
  volcano <- download_remove_header(download_read_panel(path("associations/labs_univariate_all_landmarks")))
  # Replace the row's three landmark headings with lettered headings below.
  volcano <- download_remove_header(volcano,lines=1L)
  panels <- list()
  for(lab in c("psa","testosterone")) {
    trajectory <- download_remove_header(download_read_panel(path(paste0("trajectories/",lab,"/longitudinal_platinum_log"))),lines=1L)
    km <- download_remove_header(download_read_panel(path(paste0("trajectories/",lab,"/km_extremes"))))
    column <- match(landmark,c(0L,90L,180L))
    columns <- seq.int(floor((column-1)*ncol(km)/3)+1L,floor(column*ncol(km)/3))
    km <- download_remove_header(km[,columns,drop=FALSE],lines=1L)
    panels <- c(panels,list(trajectory,km))
  }
  stem <- paste0("dfci_labs_overview_km_lm",landmark,"__",identity)
  target <- file.path(output,"compiled","associations",stem)
  width <- 24; height <- 22
  draw <- function() {
    grid::grid.newpage()
    for(i in 1:3) grid::grid.text(paste(letters[i],"  DFCI — landmark",c(0,90,180)[i],"days"),
      x=grid::unit((i-1)*8+.12,"in"),y=grid::unit(height-.12,"in"),just=c("left","top"),
      gp=grid::gpar(fontsize=17,fontface="bold"))
    download_draw_panel(volcano,.5,1-(.4+7.6/2)/height,1,7.6/height,
      canvas_width=width,canvas_height=height)
    headings <- c("d   PSA trajectory (log1p)",paste0("e   PSA bottom vs top 20% — day ",landmark),
      "f   Testosterone trajectory (log1p)",paste0("g   Testosterone bottom vs top 20% — day ",landmark))
    for(i in 1:4) {
      row <- ceiling(i/2); col <- (i-1) %% 2
      download_draw_panel(panels[[i]],.25+.5*col,1-(8+(row-.5)*7)/height,.5,7/height,
        headings[i],width,height)
    }
  }
  ragg::agg_png(paste0(target,".png"),width=width,height=height,units="in",res=200,background="white")
  tryCatch(draw(),finally=grDevices::dev.off())
  grDevices::cairo_pdf(paste0(target,".pdf"),width=width,height=height)
  tryCatch(draw(),finally=grDevices::dev.off())
  metadata <- readr::read_csv(file.path(output,"manifest.csv"),show_col_types=FALSE,
    col_types=readr::cols(.default="c"))
  moved <- metadata$path %in% selected
  metadata$path[moved] <- paste0("compiled/",metadata$path[moved])
  metadata$topic[moved] <- "compiled"
  for(format in c("png","pdf")) {
    row <- metadata[1,,drop=FALSE]
    row$path <- paste0("compiled/associations/",stem,".",format)
    row$topic <- "compiled"; row$figure_id <- sub("__.*$","",stem)
    row$endpoint <- "platinum"; row$cohort <- "all__incl"; row$landmark <- "0,90,180"
    row$title <- paste0("DFCI lab associations, trajectories and extreme-quintile survival (KM day ",landmark,")")
    row$format <- format
    metadata <- rbind(metadata,row)
  }
  readr::write_csv(metadata,file.path(output,"manifest.csv"))
  figure_write_catalog(list(fig_root=dirname(output)),list())
  invisible(list(selected=selected,new_figure=target))
}

if(sys.nframe()==0L) {
  args <- commandArgs(trailingOnly=TRUE)
  if(length(args)<2L) stop("Supply input arm directory and a new output directory; optional KM landmark 0/90/180.")
  script <- sub("^--file=","",commandArgs()[startsWith(commandArgs(),"--file=")])
  source(file.path(dirname(normalizePath(script)),"figure_publication.R"))
  reformat_downloaded_figures(args[1],args[2],if(length(args)>2L) as.integer(args[3]) else 0L)
}
