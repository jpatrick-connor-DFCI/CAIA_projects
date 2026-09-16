# Presentation-only compilation. Statistical inputs and cached RDS stay in data.
figure_slug <- function(x) tolower(gsub("^_|_$", "", gsub("[^A-Za-z0-9]+", "_", x)))

figure_public_path <- function(path) {
  if (!grepl("/by_figure/", path, fixed = TRUE)) return(path)
  root <- sub("/by_figure/.*$", "", path)
  parts <- strsplit(sub("^.*?/by_figure/", "", path), "/", fixed = TRUE)[[1]]
  group <- parts[1]; leaf <- tail(parts, 1)
  artifact <- paste(parts[seq.int(2, length(parts)-1)], collapse = "_")
  topic <- switch(group, figure1 = "cohort", cohort_comparison = "cohort",
    androgen_pre_adt_coverage = "cohort", supplement_adt_intent = "cohort", figure1s_analysis_sets = "cohort",
    figure2v3_llm = "labels", metastatic_labels = "labels",
    figure3 = "associations", figure3b = "associations", figure4 = "prediction",
    federated_no_msk = "federated_no_msk", "associations")
  if (group == "labs") {
    topic <- file.path("trajectories", figure_slug(parts[3]))
    artifact <- paste(parts[4:(length(parts)-1)], collapse = "_")
  }
  artifact <- sub("^[abcd]_", "", artifact)
  artifact <- sub("^sequencing_", "somatic_", artifact)
  artifact <- sub("gleason_platinum_km_gleason_score", "gleason_km", artifact)
  artifact <- gsub("(^|_)(platinum|nepc)(_|$)", "_", artifact)
  # Restore strata names: an endpoint in a trajectory artifact is the grouping,
  # not redundant endpoint metadata.
  if (group == "labs") artifact <- paste(parts[4:(length(parts)-1)], collapse = "_")
  artifact <- gsub("landmark([0-9]+)", "lm\\1", artifact)
  if (group == "figure3") artifact <- paste0("labs_", artifact)
  if (group == "cohort_comparison" && grepl("^_?lm", artifact)) artifact <- paste0("cohort_forest_", artifact)
  if (group == "androgen_pre_adt_coverage") artifact <- paste0("pre_adt_coverage_", artifact)
  artifact <- figure_slug(artifact)
  if(artifact=="time_to") artifact <- "time_to_event"
  if(artifact=="km") artifact <- "event_free_survival"
  file.path(root, topic, paste0(artifact, "__", leaf))
}

figure_compilation_spec <- function(stem) {
  if (grepl("^figure3_univariate_(platinum|nepc)_landmark[0-9]+$", stem))
    return(list(key="labs_univariate_all_landmarks", title="Univariate lab associations",
      width=24, height=9, cols=3, keep_individuals=TRUE,
      shared_caption=FALSE, shared_legend=TRUE))
  if (stem %in% c("figure2v3_confusion_matrix", "figure2v3_metric_bar"))
    return(list(key="classifier_validation", title="NEPC classifier validation", width=12, height=6, cols=2,
      shared_caption=TRUE, shared_legend=FALSE))
  if (stem %in% c("figure2v3_subtype_landscape", "figure2v3_enrichment"))
    return(list(key="subtype_platinum", title="Subtype landscape and platinum enrichment", width=15, height=7, cols=2,
      shared_caption=FALSE, shared_legend=FALSE))
  if (grepl("^figure4[acd]_", stem)) {
    key <- sub("_(auc|cindex)_[^_]+$", "", stem)
    title <- if (grepl("gleason", key)) "Gleason vs. labs sensitivity" else
      if (grepl("somatic", key)) "Somatic vs. labs sensitivity" else "Labs vs. age baseline"
    return(list(key=sub("^figure4[a-z]_", "", key), title=title, width=16, height=6.5, cols=2,
      shared_caption=TRUE, shared_legend=TRUE))
  }
  if (grepl("^figure4b_importance_", stem)) {
    model <- sub("^.*_(cox|xgb)_landmark.*$", "\\1", stem)
    return(list(key=paste0("importance_",model), title=paste(if(model=="cox") "Elastic-Net Cox" else "XGBoost", "feature importance"),
      width=19, height=8, cols=3, shared_caption=TRUE, shared_legend=TRUE))
  }
  if (grepl("^km_quintile_", stem))
    return(list(key=sub("_landmark[0-9]+$", "", stem), title="Lab extremes: time to platinum",
      width=19, height=7, cols=3, shared_caption=TRUE, shared_legend=FALSE))
  if (grepl("^figure3b_sequencing_.*_km_", stem))
    return(list(key="somatic_carrier_km", title="Mutation carrier status: time to platinum",
      width=16, height=12, cols=2, page_size=4, shared_caption=TRUE, shared_legend=FALSE))
  NULL
}

figure_measure_device <- function(path, width, height) {
  if (requireNamespace("ragg", quietly=TRUE)) ragg::agg_png(path, width=width, height=height, units="in", res=144)
  else grDevices::png(path, width=width, height=height, units="in", res=144, type="cairo")
}

figure_grob <- function(plot, width, height, directory=tempdir()) {
  path <- tempfile(".measure-", tmpdir=directory, fileext=".png")
  figure_measure_device(path, width, height)
  on.exit({grDevices::dev.off(); unlink(path)}, add=TRUE)
  if (inherits(plot,"ggplot")) ggplot2::ggplotGrob(prepare_figure_text(plot,width)) else plot
}

figure_combine <- function(items, spec, directory) {
  is_volcano <- identical(spec$key,"labs_univariate_all_landmarks")
  # Compare landmarks on common axes, without dropping observations or changing
  # the source plots used by the individual exports. Empty panels stay explicit.
  if(is_volcano) {
    axes <- lapply(items,function(x) x$plot$coordinates$limits)
    available <- vapply(axes,function(x) length(x$x)==2 && length(x$y)==2,logical(1))
    if(any(available)) {
      xlim <- range(unlist(lapply(axes[available],`[[`,"x")),finite=TRUE)
      ylim <- range(unlist(lapply(axes[available],`[[`,"y")),finite=TRUE)
      for(i in which(available)) items[[i]]$plot <- items[[i]]$plot +
        ggplot2::coord_cartesian(xlim=xlim,ylim=ylim)
    }
  }
  captions <- lapply(items, function(x) if(inherits(x$plot,"ggplot")) x$plot$labels$caption else NULL)
  same_caption <- isTRUE(spec$shared_caption) && length(captions) > 1 &&
    all(vapply(captions, identical, logical(1), captions[[1]]))
  legends <- list(); grobs <- list()
  for (i in seq_along(items)) {
    p <- items[[i]]$plot; stem <- items[[i]]$stem
    if (inherits(p,"ggplot")) {
      title <- p$labels$title
      if (is_volcano) {
        title <- paste0("Landmark ",sub("^.*landmark","",stem)," days")
      } else if (grepl("^figure4[acd]_",stem)) {
        # Preserve the previously requested landmark n's, once in the shared title.
        if (i==1 && length(title) && grepl("\n",title))
          spec$title <- paste(spec$title, sub("^[^\n]*\n", "",title),sep="\n")
        title <- if(grepl("cindex",stem)) "C-index" else "Mean AUC(t)"
      } else if (grepl("^figure4b_|^km_quintile_",stem)) {
        title <- paste0("Landmark ", sub("^.*landmark", "",stem), " days")
        if (grepl("^km_quintile_",stem)) {
          lab <- sub("^km_quintile_(.*)_landmark.*$", "\\1",stem)
          spec$title <- paste(if(lab=="psa") "PSA" else tools::toTitleCase(gsub("_"," ",lab)), "extremes: time to platinum")
        }
      } else if (grepl("^figure3b_.*_km_",stem)) {
        title <- sub(" carrier status: time to platinum$", "", title)
      } else if (stem=="figure2v3_confusion_matrix") {
        title <- "Agreement with chart review"
      } else if (stem=="figure2v3_metric_bar") {
        title <- "Classification metrics"
      } else if (stem=="figure2v3_subtype_landscape") {
        title <- "Subtype composition"
      } else if (stem=="figure2v3_enrichment") {
        title <- "Platinum enrichment"
      }
      p <- p + ggplot2::labs(title=title)
      if (same_caption) p <- p + ggplot2::labs(caption=NULL)
      # Shared captions/legends do not remove per-panel n/events in KM legends.
      if (isTRUE(spec$shared_legend)) {
        if(!length(legends)) {
          g <- figure_grob(p, spec$width/spec$cols, spec$height-1, directory)
          boxes <- g$grobs[grepl("^guide-box",g$layout$name)]
          boxes <- Filter(function(x) inherits(x,"gtable"), boxes)
          if (length(boxes)) legends <- boxes[1]
        }
        p <- p + ggplot2::theme(legend.position="none")
      }
      # Long category names need space at the smaller compiled panel width.
      if(!is_volcano) p <- p + ggplot2::theme(plot.title=ggplot2::element_text(size=11,face="bold"),
        axis.text=ggplot2::element_text(size=9), legend.text=ggplot2::element_text(size=9),
        plot.caption=ggplot2::element_text(size=8))
    }
    grobs[[i]] <- figure_grob(p,spec$width/spec$cols,spec$height/ceiling(length(items)/spec$cols),directory)
  }
  caption <- if(same_caption) captions[[1]] else NULL
  identity <- basename(items[[1]]$destination)
  fields <- strsplit(identity,"__",fixed=TRUE)[[1]]
  context <- paste(c(toupper(basename(sub("/by_figure/.*$","",items[[1]]$destination))),
    if(length(fields)) paste(fields[1],"endpoint"),
    if(length(fields)>1) if(fields[2]=="all") "all patients" else gsub("_"," ",fields[2]),
    if(length(fields)>2) if(fields[3]=="incl") "pre-ADT castrate included" else gsub("_"," ",fields[3])),collapse=" · ")
  if (!is.null(spec$context)) context <- spec$context
  title <- paste(spec$title, context, sep="\n")
  bottom <- list()
  if(length(legends)) bottom <- c(bottom,legends)
  if(length(caption) && nzchar(caption)) bottom <- c(bottom,list(grid::textGrob(
    paste(unlist(lapply(strsplit(caption,"\n",fixed=TRUE)[[1]],strwrap,
      width=floor(spec$width*12))),collapse="\n"),gp=grid::gpar(fontsize=8))))
  # Reserve the actual legend/caption heights. Equal null rows can overlap or
  # clip multi-line captions when this footer is nested as arrangeGrob's bottom.
  footer <- if(length(bottom)) gridExtra::arrangeGrob(grobs=bottom,ncol=1,
    heights=do.call(grid::unit.c,lapply(bottom,function(g)
      (if(inherits(g,"gtable")) sum(g$heights) else grid::grobHeight(g)) + grid::unit(4,"pt")))) else NULL
  if(!is.null(footer)) footer <- gtable::gtable_add_rows(footer,grid::unit(12,"pt"),pos=-1)
  gridExtra::arrangeGrob(grobs=grobs,ncol=spec$cols,
    top=grid::textGrob(title,gp=grid::gpar(fontsize=if(is_volcano) 18 else 14,fontface="bold")),
    bottom=footer,
    padding=grid::unit(2,"lines"))
}

figure_html_escape <- function(x) {
  x <- gsub("&","&amp;",x,fixed=TRUE); x <- gsub("<","&lt;",x,fixed=TRUE)
  x <- gsub(">","&gt;",x,fixed=TRUE); gsub('"',"&quot;",x,fixed=TRUE)
}

figure_href <- function(path) paste(vapply(strsplit(path,"/",fixed=TRUE)[[1]],
  utils::URLencode,character(1),reserved=TRUE),collapse="/")

figure_write_catalog <- function(config, prepared) {
  scenes <- unlist(lapply(prepared,function(m) m$scenes),recursive=FALSE)
  for (arm in c("ADT","ARPI")) {
    root <- file.path(config$fig_root,arm)
    if(!dir.exists(root)) next
    paths <- list.files(root,recursive=TRUE,full.names=TRUE,pattern="\\.(png|pdf|csv|md)$")
    paths <- paths[!grepl("/by_figure/|/main/|/supplements/",paths) & basename(paths)!="manifest.csv"]
    if(!length(paths)) next
    rel <- substring(paths,nchar(root)+2)
    records <- lapply(seq_along(paths),function(i) {
      stem <- tools::file_path_sans_ext(basename(paths[i]))
      fields <- strsplit(stem,"__",fixed=TRUE)[[1]]
      matches <- Filter(function(s) identical(s$destination,tools::file_path_sans_ext(paths[i])),scenes)
      title <- if(length(matches)) matches[[1]]$title else gsub("_"," ",fields[1])
      data.frame(path=rel[i],topic=strsplit(rel[i],"/",fixed=TRUE)[[1]][1],figure_id=fields[1],
        endpoint=if(length(fields)>1) fields[2] else "",cohort=if(length(fields)>2) paste(fields[-c(1,2)],collapse="__") else "",
        landmark=if(length(matches)) matches[[1]]$landmark else "",title=title,
        format=tools::file_ext(paths[i]),stringsAsFactors=FALSE)
    })
    registry <- do.call(rbind,records)
    # Keep metadata for previously rendered cohorts outside this run's scope.
    old_path <- file.path(root,"manifest.csv")
    if(file.exists(old_path)) {
      previous <- readr::read_csv(old_path,show_col_types=FALSE,col_types=readr::cols(.default="c"))
      for(i in seq_len(nrow(registry))) {
        old <- match(registry$path[i],previous$path)
        if(!is.na(old) && !any(vapply(scenes,function(s) identical(paste0(s$destination,".",registry$format[i]),paths[i]),logical(1)))) {
          registry$title[i] <- previous$title[old]; registry$landmark[i] <- previous$landmark[old]
        }
      }
    }
    readr::write_csv(registry,old_path)
    cards <- vapply(which(registry$format=="png"),function(i) {
      row <- registry[i,]; stem <- tools::file_path_sans_ext(row$path)
      sidecars <- registry$path[tools::file_path_sans_ext(registry$path)==stem & registry$format!="png"]
      href <- figure_href
      paste0('<article><a href="',href(row$path),'"><img loading="lazy" src="',href(row$path),
        '" alt="',figure_html_escape(row$title),'"></a><h3>',figure_html_escape(row$title),'</h3><p>',
        figure_html_escape(paste(row$topic,row$endpoint,row$cohort,row$landmark,sep=" · ")),'</p>',
        paste(vapply(c(row$path,sidecars),function(p) paste0('<a href="',href(p),'">',toupper(tools::file_ext(p)),'</a>'),character(1)),collapse=" · "),'</article>')
    },character(1))
    tables <- registry$path[registry$format %in% c("csv","md")]
    png_topics <- registry$topic[registry$format=="png"]
    sections <- unlist(lapply(unique(png_topics),function(topic) c(
      paste0('<h2 id="',figure_html_escape(topic),'">',figure_html_escape(tools::toTitleCase(gsub("_"," ",topic))),'</h2><main>'),
      cards[png_topics==topic],'</main>')))
    html <- c('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
      paste0('<title>',arm,' figure index</title>'),
      '<style>body{font:16px system-ui;margin:2rem;background:#f7f8fa;color:#20242a}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}article{background:white;padding:1rem;border:1px solid #ddd;border-radius:8px}img{width:100%;height:260px;object-fit:contain}h3{font-size:1rem}p{color:#555}a{color:#165b91}</style></head><body>',
      paste0('<h1>',arm,' figures</h1><p>Click a figure for full resolution. <a href="manifest.csv">Download manifest</a>.</p>'),
      paste(vapply(unique(png_topics),function(topic) paste0('<a href="#',topic,'">',figure_html_escape(topic),'</a>'),character(1)),collapse=" · "),sections,
      '<h2>Tables</h2><ul>',vapply(tables,function(p) paste0('<li><a href="',figure_href(p),'">',figure_html_escape(p),'</a></li>'),character(1)),
      '</ul></body></html>')
    writeLines(html,file.path(root,"index.html"))
  }
}

figure_archive_old_exports <- function(config, prepared) {
  # Archive only successfully regenerated run identities. Unselected cohorts and
  # unknown files are untouched; copies are verified before removing originals.
  successful <- names(Filter(function(m) length(m$scenes)>0 && is.null(m$error),prepared))
  paths <- character()
  for(arm in c("ADT","ARPI")) {
    root <- file.path(config$fig_root,arm,"by_figure")
    if(!dir.exists(root)) next
    candidates <- list.files(root,recursive=TRUE,full.names=TRUE,pattern="\\.(png|pdf|csv|md)$")
    for(path in candidates) {
      rel <- substring(path,nchar(root)+2); group <- strsplit(rel,"/",fixed=TRUE)[[1]][1]
      special <- if(group=="federated_no_msk") "federated" else if(group=="cohort_comparison") "cohort_overview" else NULL
      keep <- if(length(special)) special %in% successful else any(vapply(successful,function(job) {
        bits <- strsplit(job,"__",fixed=TRUE)[[1]]
        length(bits)==2 && toupper(cohort_arm(bits[1]))==arm &&
          startsWith(basename(path),paste0(bits[2],"__",cohort_leaf_slug(bits[1]),"."))
      },logical(1)))
      if(keep) paths <- c(paths,path)
    }
  }
  archive <- tempfile("export-",tmpdir=file.path(config$cache_root,"previous_exports"))
  for(path in paths) {
    target <- file.path(archive,substring(path,nchar(config$fig_root)+2))
    dir.create(dirname(target),recursive=TRUE,showWarnings=FALSE)
    if(!file.copy(path,target,copy.date=TRUE) || !identical(unname(tools::md5sum(path)),unname(tools::md5sum(target))))
      stop("Cannot verify archived export; original retained: ",path)
  }
  for(path in paths) if(unlink(path)!=0) stop("Cannot remove verified legacy export: ",path)
  for(arm in c("ADT","ARPI")) {
    root <- file.path(config$fig_root,arm,"by_figure")
    if(!dir.exists(root)) next
    dirs <- list.dirs(root,recursive=TRUE)
    for(dir in dirs[order(nchar(dirs),decreasing=TRUE)])
      if(!length(list.files(dir,all.files=TRUE,no..=TRUE))) unlink(dir,recursive=TRUE)
  }
  if(length(paths)) message("Archived ",length(paths)," superseded exports; recoverable at ",archive)
}
