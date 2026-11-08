# Notebook 07 writes aggregate tables; these R stages never execute Python.
# A dedicated snapshot allows this group to run without notebook 05 or unrelated
# federated deliveries. Patient inputs and OOF predictions are never read here.

augmentation_placeholder <- function(title, detail) {
  ggplot2::ggplot() + ggplot2::annotate("text", x=0, y=0, label=detail,
    family=FIGURE_FONT, size=5, lineheight=1.2) + ggplot2::xlim(-1,1) + ggplot2::ylim(-1,1) +
    ggplot2::theme_void() + ggplot2::labs(title=title) + figure_font_theme() +
    ggplot2::theme(axis.text=ggplot2::element_blank(),axis.title=ggplot2::element_blank(),
      axis.ticks=ggplot2::element_blank(),axis.line=ggplot2::element_blank())
}

augmentation_models <- c(clinical="Age + Gleason", age_labs="Age + lab count",
  augmented="Age + Gleason + lab count", L="Frozen lab count")

augmentation_auc_panel <- function(tables, cohort, title, genomic=FALSE, landmark=90L) {
  d <- tables$metrics
  d <- d[d$cohort==cohort & d$endpoint %in% c("platinum","avpc") &
    d$model %in% names(augmentation_models),,drop=FALSE]
  if(!nrow(d)) return(augmentation_placeholder(title,
    "Unavailable: required inputs or eligible patients absent.\nSee availability and cohort-flow tables."))
  labels <- augmentation_models
  if(genomic) labels[c("clinical","augmented")] <- c("Age + Gleason + TSG", "Age + Gleason + TSG + lab count")
  d$model_label <- factor(unname(labels[d$model]), levels=rev(unname(labels)))
  d$endpoint_label <- paste0(toupper(d$endpoint), "   N=",d$n_patients,"; events ≤5y=",d$n_events_horizon)
  d$endpoint_label <- factor(d$endpoint_label,levels=unique(d$endpoint_label[order(match(d$endpoint,c("platinum","avpc")))]))
  delta <- tables$contrasts
  delta <- delta[delta$cohort==cohort & delta$endpoint %in% c("platinum","avpc") &
    delta$model=="augmented" & delta$reference=="clinical",,drop=FALSE]
  format_estimate <- function(x,lo,hi) if(is.finite(x)) {
    if(is.finite(lo) && is.finite(hi)) sprintf("%.3f [%.3f, %.3f]",x,lo,hi) else sprintf("%.3f [CI unavailable]",x)
  } else "not estimable"
  contrast_text <- if(nrow(delta)) paste(vapply(seq_len(nrow(delta)),function(i)
    paste0(toupper(delta$endpoint[i])," ΔAUC: ",format_estimate(delta$delta_auc[i],
      delta$delta_auc_lo[i],delta$delta_auc_hi[i])),character(1)),collapse="; ") else "Paired contrast unavailable"
  d$estimate <- vapply(seq_len(nrow(d)),function(i) format_estimate(d$auc[i],d$auc_lo[i],d$auc_hi[i]),character(1))
  clock <- if(genomic) sprintf("Sequencing report +%d days (after ADT)",landmark) else sprintf("ADT +%d days",landmark)
  ggplot2::ggplot(d,ggplot2::aes(x=auc,y=model_label)) +
    ggplot2::geom_vline(xintercept=.5,linetype=2,color="grey65") +
    ggplot2::geom_errorbar(ggplot2::aes(xmin=auc_lo,xmax=auc_hi),orientation="y",width=.15,na.rm=TRUE) +
    ggplot2::geom_point(ggplot2::aes(shape=model=="L"),size=3,na.rm=TRUE) +
    ggplot2::geom_text(ggplot2::aes(x=1.03,label=estimate),hjust=0,size=3.8) +
    ggplot2::scale_shape_manual(values=c(`FALSE`=16,`TRUE`=17),guide="none") +
    ggplot2::facet_wrap(~endpoint_label,ncol=1) +
    ggplot2::scale_x_continuous(limits=c(0,1.46),breaks=seq(0,1,.25)) +
    ggplot2::labs(title=title,subtitle=paste0(clock,"; five-year horizon"),
      x="IPCW AUC at five years",y=NULL,caption=contrast_text) +
    ggplot2::theme_bw() + figure_font_theme() +
    ggplot2::theme(panel.grid.minor=ggplot2::element_blank(),plot.caption=ggplot2::element_text(hjust=0))
}

augmentation_risk_panel <- function(tables, landmark=90L) {
  d <- tables$four_groups
  d <- d[d$endpoint %in% c("platinum","avpc") & d$estimand=="net event risk (death censored)",,drop=FALSE]
  title <- "d  Risk within Gleason strata"
  if(!nrow(d)) return(augmentation_placeholder(title,"No eligible Gleason cohort"))
  d$group <- factor(paste0(ifelse(d$gleason_high==1,"Gleason ≥8","Gleason <8")," / ",
    ifelse(d$flag==1,"lab flag +","lab flag −")), levels=rev(c("Gleason <8 / lab flag −",
      "Gleason <8 / lab flag +","Gleason ≥8 / lab flag −","Gleason ≥8 / lab flag +")))
  d$counts <- paste0("N=",d$n_patients,"; events ≤5y=",d$n_events_horizon,
    ifelse(is.finite(d$risk),"","; not estimable"))
  d$endpoint <- factor(d$endpoint,levels=c("platinum","avpc"))
  ggplot2::ggplot(d,ggplot2::aes(x=risk,y=group,color=factor(flag))) +
    ggplot2::geom_errorbar(ggplot2::aes(xmin=risk_lo,xmax=risk_hi),orientation="y",width=.15,na.rm=TRUE) +
    ggplot2::geom_point(size=3,na.rm=TRUE) +
    ggplot2::geom_text(ggplot2::aes(x=1.03,label=counts),hjust=0,size=3.8,color="black") +
    ggplot2::facet_wrap(~endpoint,ncol=1,labeller=ggplot2::labeller(endpoint=toupper)) +
    ggplot2::scale_color_manual(values=c(`0`="#417999",`1`="#b34a42"),guide="none") +
    ggplot2::scale_x_continuous(limits=c(0,1.65),breaks=seq(0,1,.25),labels=function(x) paste0(100*x,"%")) +
    ggplot2::labs(title=title,subtitle=sprintf("ADT +%d days; five-year net risk (Kaplan–Meier)",landmark),
      x="Five-year event risk",y=NULL,caption="Death is censored; competing-death sensitivity is separate.\nWithin-stratum contrasts do not establish an interaction.") +
    ggplot2::theme_bw() + figure_font_theme() + ggplot2::theme(panel.grid.minor=ggplot2::element_blank())
}

augmentation_plots <- function(tables, manifest) {
  landmark <- manifest$settings$landmark
  items <- tables$frozen_items
  if(!nrow(items) || length(unique(items$M))!=1L || any(items$N!=nrow(items)) ||
     any(items$M<1L | items$M>items$N)) stop("Invalid frozen M-of-N checklist table")
  m <- items$M[1]; n <- nrow(items)
  item_labels <- federated_checklist_item_label(items$feature,items$op,items$threshold_value,
                                               federated_checklist_units(items))
  card <- augmentation_placeholder("a  Frozen checklist and candidate additions",paste(
    sprintf("Frozen day-%d screen: at least %d of %d criteria",landmark,m,n),
    paste(item_labels,collapse="\n"), "",
    "All measurements strictly before the prediction landmark;",
    "no lower lookback bound. Missing values use frozen training medians.", "",
    sprintf("L = number of criteria met; B = I(L ≥%d)",m),
    sprintf("Candidate only: add Gleason ≥8 (%d items), then any qualifying TSG (%d).",n+1L,n+2L),
    "Optional additions are local development; no clinical extension is endorsed.",sep="\n"))
  b <- augmentation_auc_panel(tables,"gleason","b  Increment beyond Gleason",landmark=landmark)
  c <- augmentation_auc_panel(tables,"genomic","c  Increment beyond Gleason and TSG",TRUE,landmark)
  d <- augmentation_risk_panel(tables,landmark)
  main <- gridExtra::arrangeGrob(card,b,c,d,ncol=2,
    bottom=grid::textGrob(paste(sprintf("Day %d (%s); five years. Local models: nested patient-level cross-validation.",
      landmark,if(landmark==90L) "primary" else "additional landmark"),
      manifest$inference,"| NEPC exploratory; full transfer and sensitivities in source tables."),
      gp=grid::gpar(fontsize=12,fontfamily=FIGURE_FONT)))
  list(`06_checklist_augmentation`=list(plot=main,width=24,height=19),
       panel_b_gleason=list(plot=b,width=13,height=8),
       panel_c_genomic=list(plot=c,width=13,height=8),
       panel_d_four_groups=list(plot=d,width=13,height=8))
}

run_checklist_augmentation_cell <- function(config, stage, script_dir,
    prepare_overwrite=FALSE, render_overwrite=FALSE, dpi=200, landmark=90L) {
  stage <- match.arg(stage,c("all","prepare","render"))
  input <- config$checklist_augmentation_path
  if(is.null(input)) input <- file.path(config$data_root,"checklist_augmentation")
  output_root <- config$manuscript_fig_root
  if(is.null(output_root)) output_root <- file.path(config$fig_root,"ADT_figures")
  cell <- sprintf("lm%d_h1825",landmark)
  output <- file.path(output_root,"main","alternative_06",cell)
  cache <- file.path(config$cache_root,"checklist_augmentation")
  if(landmark!=90L) {
    input <- file.path(input,cell); cache <- file.path(cache,cell)
  }
  absolute <- function(x) normalizePath(x,mustWork=FALSE)
  nested <- function(x,y) identical(absolute(x),absolute(y)) || startsWith(absolute(x),paste0(absolute(y),"/"))
  if(nested(cache,output_root) || nested(output_root,cache) || nested(input,output_root) || nested(output_root,input))
    stop("Augmentation inputs/caches and figure exports must be separate, non-nested directories")
  dir.create(cache,recursive=TRUE,showWarnings=FALSE)
  snapshot <- file.path(cache,"prepared.rds")
  if(stage!="render") {
    receipt_path <- file.path(input,"run_manifest.json")
    if(!file.exists(receipt_path)) stop("Run 07_checklist_test.ipynb first: missing ",receipt_path)
    manifest <- jsonlite::fromJSON(receipt_path)
    if(!identical(manifest$status,"complete") || manifest$schema_version!=1L)
      stop("Notebook 07 has no complete compatible run; finish its audit and analysis first")
    if(!identical(as.integer(manifest$settings$landmark),as.integer(landmark)) ||
       !identical(as.integer(manifest$settings$horizon),1825L)) stop("Mismatched landmark/horizon in ",receipt_path)
    names <- manifest$files$name
    if(any(basename(names)!=names) || anyDuplicated(names) || any(!grepl("^[a-z_]+[.]csv$",names)) ||
       any(names %in% c("predictions.csv","folds.csv"))) stop("Invalid aggregate-file manifest")
    paths <- file.path(input,"aggregate",names)
    if(any(!file.exists(paths)) || !identical(unname(tools::md5sum(paths)),manifest$files$md5))
      stop("Notebook 07 aggregate tables changed; complete notebook 07 again")
    required <- c("metrics.csv","contrasts.csv","four_groups.csv","frozen_items.csv","reproduction.csv")
    if(!all(required %in% names)) stop("Incomplete notebook 07 tables")
    signature <- list(inputs=manifest$files,manifest=unname(tools::md5sum(receipt_path)),
      code=unname(tools::md5sum(file.path(script_dir,c("checklist_augmentation.R","figure_typography.R")))),
      typography=FIGURE_TYPOGRAPHY,font=FIGURE_FONT)
    old <- if(file.exists(snapshot)) readRDS(snapshot) else NULL
    if(prepare_overwrite || is.null(old) || !identical(old$signature,signature)) {
      tables <- stats::setNames(lapply(paths,function(p) utils::read.csv(p,check.names=FALSE,stringsAsFactors=FALSE)),
                               tools::file_path_sans_ext(names))
      for(table in tables) {
        if("landmark_days" %in% names(table) && any(is.na(table$landmark_days) | table$landmark_days!=landmark))
          stop("Mixed or incorrect landmark in augmentation tables")
        if("horizon_days" %in% names(table) && any(is.na(table$horizon_days) | table$horizon_days!=1825L))
          stop("Mixed or incorrect prediction horizon in augmentation tables")
      }
      if(any(tables$reproduction$status!="ok")) stop("Unresolved Figure 6 reproduction checks")
      prepared <- list(signature=signature,manifest=manifest,tables=tables,plots=augmentation_plots(tables,manifest))
      temporary <- tempfile("augmentation-",tmpdir=cache)
      saveRDS(prepared,temporary)
      if(!file.rename(temporary,snapshot)) stop("Could not publish augmentation snapshot")
    }
  }
  if(!file.exists(snapshot)) stop("No prepared augmentation scene; run final_figure_data_prep.R first")
  prepared <- readRDS(snapshot)
  if(stage!="prepare") {
    dir.create(output,recursive=TRUE,showWarnings=FALSE)
    receipt <- file.path(cache,"rendered.rds")
    signature <- list(prepared=prepared$signature,output=absolute(output),dpi=dpi)
    previous <- if(file.exists(receipt)) readRDS(receipt) else NULL
    for(name in names(prepared$plots)) for(format in c("pdf","svg")) {
      plot <- prepared$plots[[name]]
      path <- file.path(output,paste0(name,".",format))
      complete <- figure_render_complete(path)
      key <- basename(path)
      unchanged <- complete && identical(previous$files[[key]],unname(tools::md5sum(path)))
      if(!render_overwrite && identical(previous$signature,signature) && unchanged) next
      temporary <- tempfile(".augmentation-",tmpdir=output,fileext=paste0(".",format))
      ggplot2::ggsave(temporary,plot$plot,width=plot$width,height=plot$height,dpi=dpi,bg="white",
        device=if(format=="pdf") figure_pdf_device else figure_svg_device,limitsize=FALSE)
      if(!file.rename(temporary,path)) stop("Could not publish ",path)
    }
    data_out <- file.path(output,"data")
    dir.create(data_out,recursive=TRUE,showWarnings=FALSE)
    for(name in names(prepared$tables)) utils::write.csv(prepared$tables[[name]],file.path(data_out,paste0(name,".csv")),row.names=FALSE,na="")
    writeLines(c("Figure 6 augmentation analysis",prepared$manifest$inference,
      "Frozen lab criteria and imputation are transported unchanged. Cox augmentations are locally developed.",
      "Patient-grouped five-fold outer validation; training-only preprocessing, inner ridge tuning and baseline hazards.",
      "AUC: manuscript fixed-day IPCW estimator, censoring estimated on each evaluation population.",
      "Intervals with inadequate bootstrap support are omitted; see bootstrap and fit diagnostics.",
      sprintf("Sequencing report +%d is a distinct application; the ADT+%d genomic sensitivity is separate.",landmark,landmark),
      "Negative genes require assessability evidence. Candidate N+1/N+2-item scores remain investigational.",
      "Probabilities are net event risks. Competing-death CIF is labeled separately when available.",
      "NEPC is exploratory, with frozen scores and optional platinum-trained transfer only."),file.path(data_out,"methods.txt"))
    files <- unlist(lapply(names(prepared$plots),function(name) file.path(output,paste0(name,c(".pdf",".svg")))))
    saveRDS(list(signature=signature,files=as.list(stats::setNames(unname(tools::md5sum(files)),basename(files)))),receipt)
    message("Augmentation figures: ",output)
  }
  invisible(list(snapshot=snapshot,output=output))
}

run_checklist_augmentation_workflow <- function(config, stage, script_dir,
    prepare_overwrite=FALSE, render_overwrite=FALSE, dpi=200) {
  values <- trimws(strsplit(Sys.getenv("COMPASS_AUGMENTATION_LANDMARKS","90,0,180"),",",fixed=TRUE)[[1]])
  if(!length(values) || any(!values %in% c("0","90","180")) || anyDuplicated(values))
    stop("COMPASS_AUGMENTATION_LANDMARKS must be distinct values from 0,90,180")
  results <- lapply(as.integer(values),function(landmark)
    run_checklist_augmentation_cell(config,stage,script_dir,prepare_overwrite,render_overwrite,dpi,landmark))
  invisible(stats::setNames(results,paste0("lm",values,"_h1825")))
}
