# COMPASS figure workflow: presentation helpers, federated and manuscript
# figures, the cached prepare/render bridge used by 05_figures.Rmd, and the
# downloaded-PNG reformatter. Source after COMPASS_generate_figures_pipeline.R.
# Standalone CLI: Rscript figure_workflow.R <downloaded-ADT-dir> <new-parent>/ADT [0|90|180]

# ============================================================================
# ---- publication -----------------------------------------------------------
# ============================================================================
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
    federated = "federated", "associations")
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

figure_compiled_path <- function(path, arm_root) {
  public <- figure_public_path(path)
  prefix <- paste0(arm_root,"/")
  stopifnot(startsWith(public,prefix))
  relative <- substring(public,nchar(prefix)+1L)
  if(startsWith(relative,"compiled/")) return(public)
  file.path(arm_root,"compiled",relative)
}

figure_overview_km_landmark <- function() {
  value <- Sys.getenv("COMPASS_COMBINED_KM_LANDMARK","180")
  if(!value %in% c("0","90","180"))
    stop("COMPASS_COMBINED_KM_LANDMARK must be 0, 90, or 180.")
  as.integer(value)
}

figure_dfci_overview_spec <- function(landmark=figure_overview_km_landmark()) {
  stopifnot(landmark %in% c(0,90,180))
  list(key=paste0("dfci_labs_overview_km_lm",landmark),
    title=paste0("DFCI lab associations, trajectories and extreme-quintile survival (KM day ",landmark,")"),
    width=24, height=22, cols=6, row_heights=c(8,7,7),
    layout_matrix=rbind(c(1,1,2,2,3,3),c(4,4,4,5,5,5),c(6,6,6,7,7,7)),
    members=c(paste0("figure3_univariate_platinum_landmark",c(0,90,180)),
      "longitudinal_platinum_psa_log",paste0("km_quintile_psa_landmark",landmark),
      "longitudinal_platinum_testosterone_log",paste0("km_quintile_testosterone_landmark",landmark)),
    panel_tags=letters[1:7], keep_individuals=TRUE, endpoint="platinum",
    shared_caption=FALSE, shared_legend=FALSE)
}

figure_compilation_spec <- function(stem) {
  cohort_panels <- c("figure1a_consort", "figure1b_km", "figure1c_span",
    "figure1c_dx_to_tx", "figure1c_time_to_platinum")
  if (stem %in% cohort_panels)
    return(list(key="figure1_cohort", title="Cohort selection and follow-up",
      width=22.5, height=11, cols=3, keep_individuals=TRUE,
      members=cohort_panels, panel_tags=letters[1:5],
      layout_matrix=rbind(c(1,2,3),c(1,4,5)),
      shared_caption=FALSE, shared_legend=FALSE))
  metastatic_panels <- paste0("adt_labels_", c("llm_vs_regex_max_any",
    "adt_vs_regex_max_any", "adt_vs_llm"))
  if (stem %in% metastatic_panels)
    return(list(key="metastatic_label_agreement", title="Metastatic label agreement",
      width=21, height=7, cols=3, keep_individuals=TRUE,
      members=metastatic_panels, panel_tags=letters[1:3],
      shared_caption=FALSE, shared_legend=FALSE))
  if (grepl("^figure3_univariate_(platinum|nepc)_landmark[0-9]+$", stem))
    return(list(key="labs_univariate_all_landmarks", title="Univariate lab associations",
      width=24, height=9, cols=3, keep_individuals=TRUE,
      shared_caption=FALSE, shared_legend=TRUE))
  if (stem %in% c("figure2v3_confusion_matrix", "figure2v3_metric_bar"))
    return(list(key="classifier_validation", title="NEPC classifier validation", width=12, height=6, cols=2,
      keep_individuals=TRUE, shared_caption=TRUE, shared_legend=FALSE))
  if (stem %in% c("figure2v3_subtype_landscape", "figure2v3_enrichment"))
    return(list(key="subtype_platinum", title="Subtype landscape and platinum enrichment", width=15, height=7, cols=2,
      keep_individuals=TRUE, shared_caption=FALSE, shared_legend=FALSE))
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
      width=24, height=7.5, cols=3, keep_individuals=TRUE, shared_caption=TRUE, shared_legend=FALSE))
  if (grepl("^figure3b_sequencing_.*_km_", stem))
    return(list(key="somatic_carrier_km", title="Mutation carrier status: time to platinum",
      width=16, height=12, cols=2, page_size=4, shared_caption=TRUE, shared_legend=FALSE))
  NULL
}

# A panel can belong to both an existing paired export and a complete figure.
figure_compilation_specs <- function(stem) {
  spec <- figure_compilation_spec(stem)
  specs <- if(is.null(spec)) list() else list(spec)
  llm_panels <- c("figure2v3_confusion_matrix", "figure2v3_metric_bar",
    "figure2v3_subtype_landscape", "figure2v3_enrichment")
  if(stem %in% llm_panels) specs <- c(specs, list(list(
    key="figure2v3_llm", title="LLM annotation validation and subtype landscape",
    width=16, height=12, cols=2, keep_individuals=TRUE,
    members=llm_panels, panel_tags=letters[1:4],
    shared_caption=FALSE, shared_legend=FALSE)))
  overview <- figure_dfci_overview_spec()
  if(stem %in% overview$members) specs <- c(specs,list(overview))
  specs
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
  is_overview <- startsWith(spec$key,"dfci_labs_overview_")
  volcano_panels <- vapply(items,function(x) grepl("^figure3_univariate_",x$stem),logical(1))
  # Compare landmarks on common axes, without dropping observations or changing
  # the source plots used by the individual exports. Empty panels stay explicit.
  if(is_volcano || is_overview) {
    axes <- lapply(items,function(x) x$plot$coordinates$limits)
    available <- volcano_panels & vapply(axes,function(x) length(x$x)==2 && length(x$y)==2,logical(1))
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
      if (volcano_panels[i]) {
        title <- paste0("Landmark ",sub("^.*landmark","",stem)," days")
      } else if (grepl("^figure4[acd]_",stem)) {
        # Counts formerly in the supertitle belong to the panel subtitle now.
        if (length(title) && grepl("\n",title)) p <- p + ggplot2::labs(
          subtitle=paste(c(p$labels$subtitle,sub("^[^\n]*\n", "",title)),collapse="\n"))
        title <- if(grepl("cindex",stem)) "C-index" else "Mean AUC(t)"
      } else if (grepl("^figure4b_|^km_quintile_",stem)) {
        title <- paste0("Landmark ", sub("^.*landmark", "",stem), " days")
        if (grepl("^km_quintile_",stem)) {
          lab <- sub("^km_quintile_(.*)_landmark.*$", "\\1",stem)
          spec$title <- paste(if(lab=="psa") "PSA" else tools::toTitleCase(gsub("_"," ",lab)), "extremes: time to platinum")
          if(is_overview) title <- paste(if(lab=="psa") "PSA" else "Testosterone",
            "bottom vs top 20%",paste0("(day ",sub("^.*landmark","",stem),")"))
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
      } else if (startsWith(stem,"adt_labels_")) {
        title <- switch(stem, adt_labels_llm_vs_regex_max_any="LLM vs regex stage",
          adt_labels_adt_vs_regex_max_any="ADT intent vs regex stage",
          adt_labels_adt_vs_llm="ADT intent vs LLM", title)
      }
      p <- p + ggplot2::labs(title=NULL) # retain letter tags; descriptions live in the catalog/legends
      tag <- if(length(spec$panel_tags)) spec$panel_tags[i] else letters[i]
      p <- p + ggplot2::labs(tag=tag) +
        ggplot2::theme(plot.tag=ggplot2::element_text(size=18,face="bold"),
          axis.title=ggplot2::element_text(face="bold"))
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
      if(!volcano_panels[i]) p <- p + ggplot2::theme(plot.title=ggplot2::element_text(size=13,face="bold"),
        axis.title=ggplot2::element_text(size=12,face="bold"), axis.text=ggplot2::element_text(size=11),
        legend.text=ggplot2::element_text(size=10), plot.caption=ggplot2::element_text(size=9))
      # The flowchart has no axes; resizing must not revive theme_void labels.
      if(stem=="figure1a_consort") p <- p + ggplot2::labs(x=NULL,y=NULL) +
        ggplot2::theme(axis.text=ggplot2::element_blank())
    }
    panel_width <- spec$width/spec$cols
    panel_height <- spec$height/ceiling(length(items)/spec$cols)
    if(!is.null(spec$layout_matrix)) {
      cells <- which(spec$layout_matrix==i,arr.ind=TRUE)
      panel_width <- spec$width * length(unique(cells[,"col"])) / ncol(spec$layout_matrix)
      heights <- if(is.null(spec$row_heights)) rep(1,nrow(spec$layout_matrix)) else spec$row_heights
      panel_height <- spec$height * sum(heights[unique(cells[,"row"])]) / sum(heights)
    }
    grobs[[i]] <- figure_grob(p,panel_width,panel_height,directory)
  }
  caption <- if(same_caption) captions[[1]] else NULL
  # Identity/title remain in filenames and the catalog, not in a supertitle.
  # Explicit population caveats (federated comparisons) must remain visible.
  if(length(spec$context)) caption <- paste(c(caption,spec$context),collapse="\n")
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
  combined <- do.call(gridExtra::arrangeGrob,c(list(grobs=grobs,ncol=spec$cols,
    bottom=footer,
    padding=grid::unit(1,"lines")),
    if(!is.null(spec$layout_matrix)) list(layout_matrix=spec$layout_matrix),
    if(!is.null(spec$row_heights)) list(heights=spec$row_heights)))
  attr(combined,"compass_compiled") <- TRUE
  combined
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
  # The former shallow compiled location is retired only after replacement
  # images were actually rendered. Other identities/formats are untouched.
  for(m in prepared) for(scene in m$scenes) if(length(scene$previous_destination)) {
    for(format in c("png","pdf")) {
      previous <- paste0(scene$previous_destination,".",format)
      replacement <- paste0(scene$destination,".",format)
      if(file.exists(previous) && file.exists(replacement)) paths <- c(paths,previous)
    }
  }
  for(arm in c("ADT","ARPI")) {
    root <- file.path(config$fig_root,arm,"by_figure")
    if(!dir.exists(root)) next
    candidates <- list.files(root,recursive=TRUE,full.names=TRUE,pattern="\\.(png|pdf|csv|md)$")
    for(path in candidates) {
      rel <- substring(path,nchar(root)+2); group <- strsplit(rel,"/",fixed=TRUE)[[1]][1]
      special <- if(group=="federated") "federated" else if(group=="cohort_comparison") "cohort_overview" else NULL
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

# ============================================================================
# ---- federated figures -----------------------------------------------------
# ============================================================================
# Federated PSA/testosterone forests, site counts, and XGBoost Figure 4 reports.
# Covers all participating sites, including MSK.
# Significance uses the supplied
# p/q values, never an FDR recalculation on this selected subset.
federated_lab_label <- function(raw) {
  short <- gsub("_", " ", sub("__.*$", "", raw))
  aliases <- c("Alanine aminotransferase" = "ALT", "Aspartate aminotransferase" = "AST",
    "Bilirubin direct" = "Direct bilirubin", "Bilirubin total" = "Total bilirubin",
    "Carbon dioxide" = "CO2", "Erythrocytes" = "RBC", "Leukocytes" = "WBC",
    "Prostate specific Ag" = "PSA", "Protein" = "Total protein",
    "Prothrombin time" = "PT", "Thyrotropin" = "TSH", "Urea nitrogen" = "BUN",
    "Basophils" = "Basophils absolute", "Eosinophils" = "Eosinophils absolute",
    "Lymphocytes" = "Lymphocytes absolute", "Monocytes" = "Monocytes absolute",
    "Neutrophils" = "Neutrophils absolute")
  mapped <- unname(aliases[short])
  short[!is.na(mapped)] <- mapped[!is.na(mapped)]
  short[grepl("^Erythrocyte__DistWidth", raw)] <- "RDW"
  short
}

FEDERATED_SITE_LABELS <- c(dana_farber_caia_1_1 = "Dana-Farber", fred_hutch_caia_1_1 = "Fred Hutch",
                           jhu_caia_1_1 = "Johns Hopkins", msk_caia_prod_1 = "MSK")

federated_site_label <- function(site_name) {
  coalesce(unname(FEDERATED_SITE_LABELS[site_name]), site_name)
}

validate_federated_sites <- function(site_name) {
  if (any(is.na(site_name) | !nzchar(trimws(site_name)))) stop("Missing federated site name")
  unknown <- setdiff(unique(site_name), names(FEDERATED_SITE_LABELS))
  if (length(unknown)) warning("Unlabeled federated site(s), shown by raw name: ",
    paste(unknown, collapse = ", "))
}

load_federated_forest <- function(path, within_site = FALSE) {
  d <- readr::read_csv(path, show_col_types = FALSE)
  needed <- c("landmark_days", "endpoint", "feature", "ci_lower", "ci_upper", "p_value", "q_value")
  if (within_site) needed <- c(needed, "site_name")
  missing <- setdiff(needed, names(d))
  if (length(missing)) stop("Federated forest input is missing: ", paste(missing, collapse = ", "))
  if (!"hazard_ratio_per_sd" %in% names(d)) {
    if (!"coef_feature" %in% names(d)) stop("Federated forest needs hazard_ratio_per_sd or coef_feature")
    d$hazard_ratio_per_sd <- suppressWarnings(exp(as.numeric(d$coef_feature)))
  }
  for (column in c("hazard_ratio_per_sd", "ci_lower", "ci_upper", "p_value", "q_value", "landmark_days"))
    d[[column]] <- suppressWarnings(as.numeric(d[[column]]))
  if (!"lab_name" %in% names(d)) d$lab_name <- sub("__[^_]+$", "", d$feature)
  if (!"feature_stat" %in% names(d)) d$feature_stat <- sub("^.*__", "", d$feature)
  d$raw_lab_name <- d$lab_name
  d$lab_name <- federated_lab_label(d$lab_name)
  if ("analysis_label" %in% names(d)) d <- filter(d, tolower(analysis_label) == "adt")
  d <- d %>% mutate(feature_stat = tolower(trimws(feature_stat))) %>%
    filter(tolower(endpoint) == "platinum", landmark_days %in% c(0, 90, 180),
           lab_name %in% c("PSA", "Testosterone"), feature_stat %in% c("mean", "min", "max", "last"))
  keys <- c(if (within_site) "site_name", "landmark_days", "lab_name", "feature_stat")
  if (anyDuplicated(d[keys])) stop("Duplicate site/analyte/statistic/landmark rows in federated forest input")
  if (within_site) {
    validate_federated_sites(d$site_name)
    d$site <- federated_site_label(d$site_name)
  }
  d %>% mutate(
    nominal_significant = if_else(is.finite(p_value) & p_value >= 0 & p_value <= 1, p_value < .05, NA),
    fdr_significant = if_else(is.finite(q_value) & q_value >= 0 & q_value <= 1, q_value < .05, NA),
    significance = case_when(
      fdr_significant ~ "FDR q < 0.05",
      nominal_significant & !is.na(fdr_significant) ~ "Nominal p < 0.05 only",
      !is.na(nominal_significant) & !is.na(fdr_significant) ~ "Not significant",
      TRUE ~ "Significance unavailable"),
    valid_estimate = is.finite(hazard_ratio_per_sd) & hazard_ratio_per_sd > 0,
    valid_ci = is.finite(ci_lower) & ci_lower > 0 & is.finite(ci_upper) &
      ci_upper >= ci_lower & ci_lower <= hazard_ratio_per_sd & ci_upper >= hazard_ratio_per_sd)
}

plot_federated_forest <- function(d, landmark, site_name = NULL) {
  scope <- "Across sites"
  if (!is.null(site_name)) {
    d <- filter(d, .data$site_name == .env$site_name)
    scope <- paste("Within site:", federated_site_label(site_name))
  } else if ("site_name" %in% names(d)) {
    stop("Select one site for a within-site forest; do not pool site rows")
  }
  stats <- c("mean", "min", "max", "last")
  stat_labels <- c(mean = "Mean", min = "Minimum", max = "Maximum", last = "Last")
  sub <- filter(d, landmark_days == landmark) %>%
    complete(lab_name = c("PSA", "Testosterone"), feature_stat = stats) %>%
    mutate(lab_name = factor(lab_name, levels = c("PSA", "Testosterone")),
           row_key = paste(lab_name, feature_stat, sep = "__"),
           significance = coalesce(significance, "Significance unavailable"),
           valid_estimate = coalesce(valid_estimate, FALSE), valid_ci = coalesce(valid_ci, FALSE))
  sub$row_key <- factor(sub$row_key, levels = rev(as.vector(t(outer(
    c("PSA", "Testosterone"), stats, paste, sep = "__")))))
  fmt <- function(x) ifelse(is.finite(x) & x >= 0 & x <= 1, format.pval(x, digits = 2, eps = 1e-300), "NA")
  row_labels <- setNames(paste0(stat_labels[sub$feature_stat], "\np=", fmt(sub$p_value), "; q=", fmt(sub$q_value),
    ifelse(!sub$valid_estimate, "\nEstimate unavailable", ifelse(!sub$valid_ci, "\nCI unavailable", ""))), sub$row_key)
  limits <- range(c(1, sub$hazard_ratio_per_sd[sub$valid_estimate],
    sub$ci_lower[sub$valid_estimate & sub$valid_ci], sub$ci_upper[sub$valid_estimate & sub$valid_ci]), finite = TRUE)
  limits <- exp(log(limits) + c(-1, 1)*max(diff(log(limits))*.12, .1))
  ggplot(sub, aes(hazard_ratio_per_sd, row_key, color = lab_name)) +
    geom_vline(xintercept = 1, color = "grey55", linetype = "dashed", linewidth = .5) +
    geom_blank(aes(x = 1)) +
    geom_errorbar(data = filter(sub, valid_estimate, valid_ci), aes(xmin = ci_lower, xmax = ci_upper),
                  orientation = "y", width = .17, linewidth = .8) +
    geom_point(data = filter(sub, valid_estimate), aes(shape = significance), size = 3.5, stroke = 1,
               show.legend = TRUE) +
    scale_shape_manual(values = c("Not significant" = 1, "Nominal p < 0.05 only" = 16,
      "FDR q < 0.05" = 18, "Significance unavailable" = 4),
      limits = c("Not significant", "Nominal p < 0.05 only", "FDR q < 0.05", "Significance unavailable"),
      breaks = c("Not significant", "Nominal p < 0.05 only", "FDR q < 0.05",
                 if (any(sub$significance == "Significance unavailable")) "Significance unavailable"),
      drop = FALSE, name = NULL) +
    scale_color_manual(values = c(PSA = "#0072B2", Testosterone = "#D55E00"), guide = "none") +
    scale_x_log10(limits = limits, labels = scales::label_number(accuracy = .01)) +
    scale_y_discrete(labels = row_labels, expand = expansion(add = .7)) +
    facet_wrap(~lab_name, nrow = 1, scales = "free_y", drop = FALSE) +
    labs(x = "Hazard ratio per SD (95% CI; log scale)", y = NULL,
      title = sprintf("Federated | %s | ADT platinum | +%d days", scope, landmark),
      subtitle = "PSA and testosterone: mean, minimum, maximum, and last value",
      caption = paste("Nominal significance: p < 0.05; FDR significance: supplied q < 0.05.",
        "Delta and observation-count features excluded. q-values are not recomputed for this subset.",
        "Open circle: not significant; filled circle: nominal only; diamond: FDR significant.", sep = "\n")) +
    theme_classic(base_size = 11) +
    theme(strip.background = element_blank(), strip.text = element_text(face = "bold", size = 12),
      axis.text.y = element_text(size = 10), legend.position = "bottom",
      plot.caption = element_text(hjust = 0, size = 9), plot.title.position = "plot",
      plot.caption.position = "plot", panel.spacing = grid::unit(1.5, "lines"),
      plot.margin = margin(12, 16, 12, 12))
}

# The within-site bundle directory was renamed between exports
# (nvflare_within_site_cox_univariate -> nvflare_within_site_univariate_cox).
# Accept either, preferring whichever actually exists.
FEDERATED_WITHIN_SITE_DIRS <- c("nvflare_within_site_univariate_cox",
                                "nvflare_within_site_cox_univariate")

federated_within_site_file <- function(federated_path, filename) {
  candidates <- file.path(dirname(federated_path), FEDERATED_WITHIN_SITE_DIRS, filename)
  existing <- candidates[file.exists(candidates)]
  if (length(existing)) existing[1] else candidates[1]
}

load_federated_sites <- function(federated_path) {
  path <- federated_within_site_file(federated_path, "cox_within_site_all_sites_cohort.csv")
  if (!file.exists(path)) {
    warning("Federated site counts unavailable; missing: ", path)
    return(NULL)
  }
  d <- readr::read_csv(path, show_col_types = FALSE)
  needed <- c("site_name", "analysis_label", "landmark_days", "n_patients", "n_events")
  if (length(setdiff(needed, names(d)))) stop("Federated site counts missing required columns: ",
    paste(setdiff(needed, names(d)), collapse = ", "))
  # This bundle's ADT cohort counts describe the platinum endpoint. If an
  # endpoint column is supplied by a newer writer, select it explicitly.
  if ("endpoint" %in% names(d)) d <- filter(d, tolower(endpoint) == "platinum")
  d <- filter(d, tolower(analysis_label) == "adt", landmark_days %in% c(0, 90, 180))
  if (!nrow(d)) stop("No ADT platinum site counts at landmarks 0/90/180")
  validate_federated_sites(d$site_name)
  if (anyDuplicated(d[c("site_name", "landmark_days")])) stop("Duplicate federated site/landmark counts")
  for (column in c("n_patients", "n_events")) {
    d[[column]] <- suppressWarnings(as.numeric(d[[column]]))
    if (any(!is.finite(d[[column]]) | d[[column]] < 0 | d[[column]] != floor(d[[column]])))
      stop("Invalid federated site counts: ", column)
  }
  if (any(d$n_events > d$n_patients)) stop("Federated events exceed patients")
  d %>% mutate(endpoint = "platinum", site = federated_site_label(site_name),
    event_incidence_pct = if_else(n_patients > 0, 100 * n_events / n_patients, NA_real_)) %>%
    arrange(landmark_days, site)
}

prepare_federated_site_incidence <- function(d) {
  # Preserve a blank row for sites with no day-0 result, never borrow day 90/180.
  d <- distinct(d, site_name, site) %>% left_join(
    filter(d, landmark_days == 0) %>% select(-site), by = "site_name") %>%
    mutate(landmark_days = 0, endpoint = "platinum") %>% arrange(site)
  z <- 1.96
  d %>% mutate(available = !is.na(n_patients) & n_patients > 0,
    p = if_else(available, n_events / n_patients, NA_real_),
    denominator = 1 + z^2 / n_patients,
    centre = (p + z^2 / (2*n_patients)) / denominator,
    half = z * sqrt(p*(1-p)/n_patients + z^2/(4*n_patients^2)) / denominator,
    event_incidence_pct = 100*p, ci_lower_pct = 100*pmax(0, centre-half),
    ci_upper_pct = 100*pmin(1, centre+half), thin = available & n_events < 25) %>%
    select(-p, -denominator, -centre, -half)
}

plot_federated_sites <- function(d) {
  d <- prepare_federated_site_incidence(d) %>% mutate(y = n():1)
  good <- filter(d, available)
  rate_max <- max(c(1, good$ci_upper_pct), na.rm = TRUE)
  count_max <- max(c(1, d$n_patients), na.rm = TRUE)
  yscale <- function(labels = TRUE) scale_y_continuous(breaks = d$y,
    labels = if (labels) d$site else NULL, limits = c(.4, nrow(d)+.6))
  style <- theme_classic(base_size = 11) + theme(
    panel.grid.major.x = element_line(color = "grey92", linewidth = .3),
    axis.ticks.y = element_blank(), axis.line.y = element_blank(),
    plot.title = element_text(face = "bold", size = 12),
    plot.margin = margin(12, 20, 12, 12))
  a <- ggplot(good, aes(y = y)) +
    geom_rect(aes(xmin = 0, xmax = event_incidence_pct, ymin = y-.20, ymax = y+.20), fill = "#2a78d6")
  hatch <- bind_rows(lapply(which(good$thin & good$event_incidence_pct > 0), function(i) {
    width <- rate_max/25
    starts <- seq(-width, good$event_incidence_pct[i], by = width*.6)
    x <- pmax(0, starts); xend <- pmin(good$event_incidence_pct[i], starts+width)
    tibble(x = x, xend = xend, y = good$y[i]-.20+(x-starts)/width*.40,
           yend = good$y[i]-.20+(xend-starts)/width*.40) %>% filter(xend > x)
  }))
  if (nrow(hatch)) a <- a + geom_segment(data = hatch, aes(x=x, xend=xend, y=y, yend=yend),
    inherit.aes = FALSE, color = "white", linewidth = .35)
  a <- a + geom_errorbar(aes(xmin = ci_lower_pct, xmax = ci_upper_pct), orientation = "y",
      width = .12, color = "grey30", linewidth = .5) +
    geom_text(aes(x = ci_upper_pct, label = sprintf("%.2f%s", event_incidence_pct, if_else(thin, "*", ""))),
      hjust = -.2, size = 3, color = "grey30") +
    geom_text(data = filter(d, !available), aes(x = 0, label = "Unavailable"), hjust = 0, size = 3) +
    yscale() + scale_x_continuous(limits = c(0, rate_max*1.25), expand = expansion(mult = c(0,.01))) +
    labs(x = "Event rate (%)", y = NULL, title = "(a) incidence at landmark 0d (95% Wilson CI)") + style
  b <- ggplot(filter(d, !is.na(n_patients)), aes(y = y)) +
    geom_rect(aes(xmin = 0, xmax = n_patients, ymin = y-.20, ymax = y+.20), fill = "#2a78d6", alpha = .28) +
    geom_rect(aes(xmin = 0, xmax = n_events, ymin = y-.20, ymax = y+.20), fill = "#2a78d6") +
    geom_text(aes(x = n_patients, label = paste0(scales::comma(n_events), "/", scales::comma(n_patients))),
      hjust = -.1, size = 3, color = "grey30") +
    geom_text(data = filter(d, is.na(n_patients)), aes(x = 0, label = "Unavailable"), hjust = 0, size = 3) +
    yscale(FALSE) + scale_x_continuous(limits = c(0, count_max*1.32), expand = expansion(mult = c(0,.01))) +
    labs(x = "Patients (events overlaid)", y = NULL, title = "(b) cohort size and event count",
      caption = "Pale = patients; solid = events; label = events/patients") + style
  a <- ggplotGrob(a); b <- ggplotGrob(b)
  a$heights <- b$heights <- grid::unit.pmax(a$heights, b$heights)
  gridExtra::arrangeGrob(a, b, ncol = 2, top = grid::textGrob(paste(
    "Federated | ADT platinum incidence by site | landmark 0d",
    sprintf("Hatched/* = <25 events (%d of %d available sites); read as power, not biology", sum(good$thin), nrow(good)),
    sep = "\n"), gp = grid::gpar(fontsize = 14)),
    bottom = grid::textGrob("Observed events / analyzed patients during follow-up; not fixed-horizon cumulative incidence.",
      gp = grid::gpar(fontsize = 10)), padding = grid::unit(2.3, "lines"))
}

save_federated_panel <- function(plot, path, width, height, dpi, overwrite) {
  capture <- getOption("compass.figure_capture")
  if (is.function(capture)) {
    capture(plot, sub("\\.png$", "", path), width, height, sub("__.*$","",basename(path)))
    return(invisible(TRUE))
  }
  if (is.null(plot)) return(FALSE)
  complete <- exists("figure_file_complete", mode = "function") && figure_file_complete(path)
  if (!overwrite && complete) { message("Skipped ", path); return(TRUE) }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temp <- tempfile(".federated-", tmpdir = dirname(path), fileext = ".png")
  on.exit(unlink(temp), add = TRUE)
  ggplot2::ggsave(temp, plot, width = width, height = height, dpi = dpi, bg = "white",
                  device = if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png else "png")
  if (!file.rename(temp, path)) stop("Could not publish federated supplement panel: ", path)
  message("Wrote ", path)
  TRUE
}

federated_population_note <- function(across, cohorts) {
  unknown <- "Federated*: participating sites unverified; not assumed to include all displayed sites."
  if(is.null(cohorts) || !all(c("n_patients_used","n_events_used") %in% names(across))) return(unknown)
  targets <- distinct(across,landmark_days,n_patients_used,n_events_used)
  if(anyDuplicated(targets$landmark_days) || anyNA(targets)) return(unknown)
  sites <- sort(unique(cohorts$site_name))
  if(!length(sites) || length(sites)>10) return(unknown)
  candidates <- unlist(lapply(seq_along(sites),function(k) combn(sites,k,simplify=FALSE)),recursive=FALSE)
  matches <- Filter(function(selected) {
    all(vapply(seq_len(nrow(targets)),function(i) {
      rows <- filter(cohorts,site_name %in% selected,landmark_days==targets$landmark_days[i])
      nrow(rows)==length(selected) && sum(rows$n_patients)==targets$n_patients_used[i] &&
        sum(rows$n_events)==targets$n_events_used[i]
    },logical(1)))
  },candidates)
  if(length(matches)!=1) return(unknown)
  paste0("Federated*: patient/event counts match ",paste(federated_site_label(matches[[1]]),collapse=" + "),
         "; membership inferred, not verified.")
}

prepare_federated_comparison <- function(across, within=NULL) {
  across <- mutate(across,source="Federated*",source_kind="across_sites")
  if(is.null(within)) return(across)
  bind_rows(across,mutate(within,source=site,source_kind="within_site"))
}

# Widescreen slide canvas; shared by both analytes and the cached renderer.
FEDERATED_FOREST_SLIDE_SIZE <- c(width=16, height=9)

plot_federated_comparison <- function(estimates, analyte, population_note) {
  sources <- c(sort(setdiff(unique(estimates$source),"Federated*")),"Federated*")
  d <- filter(estimates,lab_name==analyte) %>%
    complete(source=sources,feature_stat=c("mean","min","max","last"),landmark_days=c(0,90,180)) %>%
    mutate(source=factor(source,levels=rev(sources)),
      feature_stat=factor(feature_stat,levels=c("mean","min","max","last"),labels=c("Mean","Minimum","Maximum","Last")),
      landmark=factor(landmark_days,levels=c(0,90,180),labels=c("0 days","+90 days","+180 days")),
      valid_estimate=coalesce(valid_estimate,FALSE),valid_ci=coalesce(valid_ci,FALSE),
      significance=coalesce(significance,"Significance unavailable"))
  limits <- range(c(1,d$hazard_ratio_per_sd[d$valid_estimate],
    d$ci_lower[d$valid_estimate & d$valid_ci],d$ci_upper[d$valid_estimate & d$valid_ci]),finite=TRUE)
  limits <- exp(log(limits)+c(-1,1)*max(.12,diff(log(limits))*.12))
  palette <- setNames(c("#0072B2","#D55E00","#009E73","#CC79A7","#E69F00")[seq_along(sources)],sources)
  palette["Federated*"] <- "#222222"
  counts <- filter(estimates,source_kind=="across_sites",lab_name==analyte)
  count_note <- ""
  if(all(c("n_patients_used","n_patients_observed") %in% names(counts))) {
    rows <- distinct(counts,landmark_days,n_patients_used,n_patients_observed) %>% arrange(landmark_days)
    count_note <- paste0("Federated modeled / observed n: ",paste(sprintf("%dd: %s / %s",rows$landmark_days,
      scales::comma(rows$n_patients_used),scales::comma(rows$n_patients_observed)),collapse="; "),".")
  }
  ggplot(d,aes(hazard_ratio_per_sd,source,color=source)) +
    geom_vline(xintercept=1,linetype="dashed",color="grey65",linewidth=.4) +
    geom_blank(aes(x=1)) +
    geom_errorbar(data=filter(d,valid_estimate,valid_ci),aes(xmin=ci_lower,xmax=ci_upper),
      orientation="y",width=.12,linewidth=.65) +
    geom_point(data=filter(d,valid_estimate),aes(shape=significance),size=3.2,stroke=.8) +
    geom_text(data=filter(d,!valid_estimate),aes(x=1,label="Unavailable"),color="grey50",size=2.6) +
    geom_text(data=filter(d,valid_estimate,!valid_ci),aes(x=limits[2],label="CI unavailable"),
      hjust=1,color="grey50",size=2.3) +
    scale_color_manual(values=palette,guide="none") +
    scale_shape_manual(name=NULL,values=c("Not significant"=1,"Nominal p < 0.05 only"=16,
      "FDR q < 0.05"=18,"Significance unavailable"=4),
      breaks=c("Not significant","Nominal p < 0.05 only","FDR q < 0.05",
        if(any(d$significance=="Significance unavailable")) "Significance unavailable"),drop=FALSE) +
    scale_x_log10(limits=limits,labels=scales::label_number(accuracy=.01)) +
    scale_y_discrete(drop=FALSE,expand=expansion(add=.65)) +
    facet_grid(landmark ~ feature_stat,drop=FALSE) +
    labs(title=paste(analyte,"associations across sites"),
      subtitle="ADT · platinum endpoint · all federated sites · hazard ratios per SD with 95% confidence intervals",
      x="Hazard ratio per SD (log scale)",y=NULL,
      caption=paste(population_note,count_note,
        "Supplied FDR q values (not recalculated). Exact estimates, counts and fitting metadata: accompanying CSV. No delta features.",sep="\n")) +
    theme_classic(base_size=14) +
    theme(strip.background=element_blank(),strip.text=element_text(face="bold"),
      panel.spacing=grid::unit(.65,"lines"),legend.position="bottom",
      legend.text=element_text(margin=margin(l=5,r=10)),
      plot.caption=element_text(hjust=0,size=10.5),plot.title.position="plot",plot.caption.position="plot",
      plot.margin=margin(12,18,12,12))
}

load_federated_xgboost <- function(path, kind=c("metrics","importance")) {
  kind <- match.arg(kind)
  if (!file.exists(path)) {
    warning("Federated XGBoost ",kind," unavailable; missing: ",path)
    return(NULL)
  }
  d <- readr::read_csv(path,show_col_types=FALSE)
  needed <- c("analysis_label","endpoint","landmark_days",if(kind=="metrics")
    c("model","cohort","config","test_c_index","test_mean_auc_t","n_test","n_events_test") else c("feature","gain"))
  missing <- setdiff(needed,names(d))
  if(length(missing)) stop("Federated XGBoost ",kind," input is missing: ",paste(missing,collapse=", "))
  d <- filter(d,tolower(analysis_label)=="adt",tolower(endpoint)=="platinum",landmark_days %in% c(0,90,180))
  if(kind=="metrics") {
    d <- filter(d,tolower(model)=="xgboost_cox",tolower(cohort)=="all",config %in% c("both","baseline"))
    keys <- c("landmark_days","config")
    for(column in c("test_c_index","test_mean_auc_t")) {
      if(!is.numeric(d[[column]]) && !all(is.na(d[[column]]))) stop("Non-numeric XGBoost metric: ",column)
      if(any(!is.na(d[[column]]) & (!is.finite(d[[column]]) | d[[column]]<0 | d[[column]]>1)))
        stop("Invalid XGBoost metric: ",column)
    }
    if(any(!is.na(d$n_test) & (!is.finite(d$n_test) | d$n_test<0)) ||
       any(!is.na(d$n_events_test) & (!is.finite(d$n_events_test) | d$n_events_test<0 | d$n_events_test>d$n_test),na.rm=TRUE))
      stop("Invalid XGBoost test counts")
  } else {
    keys <- c("landmark_days","feature")
    if(any(is.na(d$feature) | !nzchar(trimws(d$feature)))) stop("Missing XGBoost feature name")
    if(!is.numeric(d$gain) || any(!is.finite(d$gain) | d$gain<0)) stop("Invalid XGBoost gain")
    # OMOP names contain internal double underscores: the statistic is the LAST
    # suffix, unlike the short local names. Keep original names/gains in CSV.
    d <- mutate(d,lab_name=federated_lab_label(feature),
      feature_stat=if_else(grepl("__",feature,fixed=TRUE),sub("^.*__","",feature),""),
      identifier_feature=tolower(feature) %in% c("person_id","patient_id","dfci_mrn","mrn"))
  }
  if(anyDuplicated(d[keys])) stop("Duplicate landmark/",if(kind=="metrics") "config" else "feature",
    " rows in federated XGBoost ",kind," input")
  if(!nrow(d)) warning("No ADT/platinum rows in federated XGBoost ",kind," input: ",path)
  d
}

prepare_federated_xgboost_performance <- function(metrics) {
  metrics %>% complete(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(name=factor(if_else(config=="both","XGBoost Survival","XGBoost baseline (age)"),
      levels=c("XGBoost Survival","XGBoost baseline (age)")),
      landmark=landmark_days,auc=test_mean_auc_t,cindex=test_c_index)
}

federated_xgboost_audit_note <- function(importance) {
  if(is.null(importance)) return("Feature-input audit unavailable: importance file missing.")
  ids <- filter(importance,identifier_feature,gain>0)
  if(!nrow(ids)) return("")
  paste0("CAUTION: identifier feature(s) ",paste(unique(ids$feature),collapse=", "),
    " have nonzero gain at landmark(s) ",paste(sort(unique(ids$landmark_days)),collapse=", "),
    " days; audit model inputs before interpreting performance or importance.")
}

render_federated_xgboost <- function(root,federated_path,dpi,overwrite) {
  directory <- file.path(dirname(federated_path),"federated_xgboost")
  metrics <- load_federated_xgboost(file.path(directory,"xgboost_federated_metrics_adt.csv"),"metrics")
  importance <- load_federated_xgboost(file.path(directory,"xgboost_federated_importance_adt.csv"),"importance")
  audit <- federated_xgboost_audit_note(importance)
  if(startsWith(audit,"CAUTION")) warning(audit)
  capture <- getOption("compass.figure_table_capture")
  write_table <- function(data,stem) {
    path <- file.path(root,paste0(stem,"__platinum.csv"))
    readr::write_csv(data,path)
    if(is.function(capture)) capture(path)
  }
  context <- "ADT · platinum endpoint · federated (all sites)"
  if(!is.null(metrics) && nrow(metrics)) {
    d <- prepare_federated_xgboost_performance(metrics)
    colors <- c("XGBoost Survival"="#B58900","XGBoost baseline (age)"="#E0CC8A")
    count_note <- d %>% arrange(landmark_days,config) %>%
      transmute(note=sprintf("%dd %s: n=%s, events=%s",landmark_days,
        if_else(config=="both","labs","baseline"),scales::comma(n_test),scales::comma(n_events_test))) %>%
      pull(note) %>% paste(collapse="; ")
    caption <- paste("Held-out test metrics (not training or tuning CV). Gaps indicate unavailable metrics.",
      paste0("Test counts — ",count_note,"."),audit,sep="\n")
    items <- lapply(c("auc","cindex"),function(metric) {
      stem <- paste0("figure4a_discrimination_",metric,"_platinum")
      list(plot=plot_model_discrimination(d,metric,if(metric=="auc") "Test Mean AUC(t)" else "Test C-index",
        colors,show_legend=TRUE)+labs(caption=caption),stem=stem,destination=file.path(root,"platinum"))
    })
    spec <- figure_compilation_spec(items[[1]]$stem)
    spec$title <- "Federated XGBoost: labs vs. age baseline"; spec$context <- context
    p <- figure_combine(items,spec,tempdir())
    write_table(mutate(d,input_audit_note=audit),"xgboost_performance")
    save_federated_panel(p,file.path(root,"xgboost_performance__platinum.png"),spec$width,spec$height,dpi,overwrite)
  }
  if(!is.null(importance) && nrow(importance)) {
    caption <- paste("Top 15 positive-gain features per landmark; age and body height excluded from display, as in local plots.",
      "Gains are supplied model split gains, not signed effects or SHAP values. All source features remain in the CSV.",audit,sep="\n")
    items <- lapply(c(0,90,180),function(lm) {
      d <- filter(importance,landmark_days==lm,tolower(feature)!="age",gain>0)
      stem <- paste0("figure4b_importance_platinum_xgb_landmark",lm)
      p <- plot_model_importance(d,"xgb",paste("Landmark",lm,"days"))+labs(caption=caption)
      if(!any(importance$landmark_days==lm)) p <- p+labs(subtitle="Importance unavailable")
      list(plot=p,stem=stem,destination=file.path(root,"platinum"))
    })
    shown <- bind_rows(tibble(landmark_days=numeric(),feature=character()),
      Filter(is.data.frame,lapply(items,function(item) item$plot$data)))
    exported <- importance %>% mutate(displayed=paste(landmark_days,feature) %in% paste(shown$landmark_days,shown$feature),
      input_audit_note=audit)
    spec <- figure_compilation_spec(items[[1]]$stem)
    spec$title <- "Federated XGBoost feature importance"; spec$context <- context
    p <- figure_combine(items,spec,tempdir())
    write_table(exported,"xgboost_importance")
    save_federated_panel(p,file.path(root,"xgboost_importance__platinum.png"),spec$width,spec$height,dpi,overwrite)
  }
  invisible(NULL)
}

render_federated_supplement <- function(data_root, fig_root, federated_path,
                                                dpi = 200, overwrite = FALSE) {
  d <- load_federated_forest(federated_path)
  root <- file.path(fig_root, "ADT", "federated")
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  capture <- getOption("compass.figure_table_capture")
  within <- NULL
  within_path <- federated_within_site_file(federated_path, "cox_within_site_all_sites_results.csv")
  if (file.exists(within_path)) {
    within <- load_federated_forest(within_path, within_site = TRUE)
  } else warning("Within-site federated forests unavailable; missing: ", within_path)
  sites <- load_federated_sites(federated_path)
  estimates <- prepare_federated_comparison(d,within)
  note <- federated_population_note(d,sites)
  estimates$population_note <- ifelse(estimates$source_kind=="across_sites",note,"Within-site model")
  for(analyte in c("PSA","Testosterone")) {
    base <- file.path(root,paste0(tolower(analyte),"_forest__platinum"))
    readr::write_csv(filter(estimates,lab_name==analyte),paste0(base,".csv"))
    if(is.function(capture)) capture(paste0(base,".csv"))
    save_federated_panel(plot_federated_comparison(estimates,analyte,note),paste0(base,".png"),
      FEDERATED_FOREST_SLIDE_SIZE[["width"]],FEDERATED_FOREST_SLIDE_SIZE[["height"]],dpi,overwrite)
  }
  if (!is.null(sites)) {
    site_path <- file.path(root, "site_incidence_lm000__platinum.csv")
    dir.create(dirname(site_path), recursive = TRUE, showWarnings = FALSE)
    readr::write_csv(prepare_federated_site_incidence(sites), site_path)
    if (is.function(capture)) capture(site_path)
    save_federated_panel(plot_federated_sites(sites),
      file.path(root, "site_incidence_lm000__platinum.png"), 14, 5.5, dpi, overwrite)
  }
  render_federated_xgboost(root,federated_path,dpi,overwrite)
  invisible(d)
}

# ============================================================================
# ---- manuscript figures ----------------------------------------------------
# ============================================================================
# Manuscript exports are additional views of the original statistics/plots.
# Never refit, recompute q-values, or serialize patient-bearing ggplot objects.
manuscript_dfci_volcano_ylim <- c(-.2,20)
manuscript_dfci_landmark_headers <- c("+0 days","+90 days","+180 days")
manuscript_dfci_km_landmark <- 180
manuscript_km_y_label <- "Platinum-free\nprobability"
manuscript_multivariable_legend_height <- .14
manuscript_specs <- function() {
  spec <- function(key, members, height, layout, titles, row_heights=NULL,
                   shared_legend_panels=integer(),width=7.2) list(key=key, members=members,
    width=width, height=height, layout=layout, titles=titles,
    row_heights=row_heights, shared_legend_panels=shared_legend_panels)
  list(
    spec("01_cohort_overview", c("figure1a_consort","figure1b_km","figure1c_span",
      "figure1c_dx_to_tx","figure1c_time_to_platinum"), 7.2,
      rbind(c(1,1,2,2),c(1,1,3,3),c(4,4,5,5)),
      c("Cohort selection","Platinum-free survival","Lab record span","Diagnosis to ADT","ADT to platinum")),
    spec("02_llm_nepc_avpc", c("figure2v3_confusion_matrix","figure2v3_metric_bar",
      "figure2v3_subtype_landscape","figure2v3_enrichment"), 6.2, rbind(1:2,3:4),
      c("NEPC: chart-review agreement","NEPC classification metrics","NEPC/AVPC proxy subtypes","Platinum enrichment")),
    spec("03_dfci_univariable", figure_dfci_overview_spec(manuscript_dfci_km_landmark)$members, 8.6,
      rbind(c(1,1,2,2,3,3),rep(8,6),c(4,4,4,5,5,5),c(6,6,6,7,7,7)),
      c("Day 0","Day 90","Day 180","PSA trajectory","PSA: extreme quintiles",
        "Testosterone trajectory","Testosterone: extreme quintiles"),
      row_heights=c(1.5,.18,1,1), shared_legend_panels=1:3),
    spec("04_multivariable_labs", c(
      paste0("figure4a_discrimination_",c("auc","cindex"),"_platinum"),
      paste0("figure4b_importance_platinum_cox_landmark",c(0,90,180)),
      paste0("figure4b_importance_platinum_xgb_landmark",c(0,90,180))),
      11.8, matrix(1:4,4), c("Mean AUC(t)","C-index","Elastic-Net coefficients","XGBoost feature importance"),
      row_heights=c(.9,.9,1,1),width=8.5),
    spec("05_gleason_sensitivity", paste0("figure4c_sensitivity_gleason_",c("auc","cindex"),"_platinum"),
      5.6, matrix(1:2,2), c("Mean AUC(t)","C-index")))
}

manuscript_captions <- function() {
  caption <- function(...) paste(c(...),collapse="\n\n")
  c(
    `01_cohort_overview`=caption(
      "Figure 1. Cohort assembly, follow-up, and platinum treatment after androgen-deprivation therapy.",
      "(a) Cumulative selection of the PROFILE prostate cancer cohort, from patients identified by International Classification of Diseases codes through the final laboratory-analysis cohort. (b) Kaplan-Meier estimate of platinum-free survival from androgen-deprivation therapy (ADT) initiation; shading denotes the pointwise 95% confidence interval. (c) Distribution of each patient's laboratory-record span. (d) Distribution of time from prostate cancer diagnosis to ADT initiation, displayed through the 99th percentile. (e) Distribution of time from ADT initiation to platinum treatment among patients who received platinum.",
      "Panels b and c include 2,239 patients. Panel d displays 2,216 of 2,239 patients, and panel e displays 162 of 198 platinum-treated patients. ADT, androgen-deprivation therapy; PARPi, poly(ADP-ribose) polymerase inhibitor; PSA, prostate-specific antigen."),
    `02_llm_nepc_avpc`=caption(
      "Figure 2. Large language model classification of neuroendocrine and aggressive-variant prostate cancer proxies.",
      "(a) Confusion matrix comparing large language model (LLM) neuroendocrine prostate cancer (NEPC) labels with manual chart-review annotations. (b) Accuracy, precision, recall, and specificity in the chart-reviewed set. (c) Distribution of conventional, aggressive-variant prostate cancer (AVPC), NEPC, and biomarker subtypes within platinum-positive and platinum-negative groups. (d) Observed platinum-positive fraction for aggressive subtypes, defined as AVPC or NEPC, versus conventional prostate cancer; error bars denote 95% Wilson confidence intervals. The displayed odds ratio and one-sided Fisher exact-test P value compare the aggressive and conventional groups.",
      "Panels a and b include 41 chart-reviewed patients, of whom 12 were manually annotated as NEPC. Panel c includes 3,754 classified patients (213 platinum positive and 3,541 platinum negative) from 3,869 ADT-exposed patients; seven labeled rows outside the four displayed classes were excluded. Panel d excludes 382 biomarker or unclassified rows. OR, odds ratio."),
    `03_dfci_univariable`=caption(
      "Figure 3. Univariable laboratory associations with platinum treatment and longitudinal PSA and testosterone patterns.",
      "(a-c) Univariable Cox associations at ADT initiation (a), 90 days (b), and 180 days (c). The horizontal axis gives the log hazard ratio per standard-deviation increase and the vertical axis gives -log10(P), displayed through 20. The solid vertical line denotes no association, dashed vertical lines mark log hazard ratios of -0.5 and 0.5, and the dotted horizontal line is the landmark-specific threshold corresponding to a Benjamini-Hochberg false-discovery rate below 0.05. Colors identify laboratory categories. At the three landmarks, 31 of 235, 72 of 244, and 76 of 245 tested features, respectively, had q < 0.05. (d,f) Mean log1p-transformed PSA (d) and testosterone (f) trajectories by subsequent platinum status; points and shading denote the mean and 95% confidence interval, and only bins containing at least 10 patients per stratum are shown. The dotted vertical line denotes ADT initiation. (e,g) Kaplan-Meier platinum-free survival from the 180-day landmark comparing the bottom and top 20% of the pre-landmark mean PSA (e) or testosterone (g) distribution. Middle quintiles are omitted, equal values remain together, shading denotes pointwise 95% confidence intervals, and P values are from log-rank tests.",
      "The trajectory analyses include 2,217 patients for PSA and 2,059 for testosterone. For PSA, the bottom group was ≤0.2453 (n=403; 11 events) and the top group was >9.626 (n=403; 58 events); log-rank P=2.75 × 10^-13. For testosterone, the bottom group was ≤10 (n=317; 41 events) and the top group was >294.3 (n=313; 19 events); log-rank P=6.92 × 10^-6. ADT, androgen-deprivation therapy; CBC, complete blood count; CMP, comprehensive metabolic panel; HR, hazard ratio; LFT, liver-function test; PSA, prostate-specific antigen; SD, standard deviation."),
    `04_multivariable_labs`=caption(
      "Figure 4. Multivariable laboratory models for prediction of platinum treatment.",
      "(a,b) Held-out test mean time-dependent area under the receiver-operating-characteristic curve, AUC(t) (a), and Harrell concordance index (b) for Elastic-Net Cox and XGBoost survival models at 0, 90, and 180 days after ADT initiation. Laboratory models are compared with age-only baselines; the dotted horizontal line marks 0.5. (c) Nonzero Elastic-Net Cox coefficients at each landmark. Positive coefficients indicate higher platinum hazard and negative coefficients indicate lower hazard, conditional on the other selected features. (d) XGBoost split-gain feature importance at each landmark. Colors in c and d identify laboratory categories.",
      "Performance values are from held-out test sets rather than training or tuning cross-validation. Elastic-Net coefficients are signed log hazard-ratio coefficients. XGBoost gains are unsigned measures of split improvement and do not encode effect direction. ADT, androgen-deprivation therapy; AUC, area under the curve; PSA, prostate-specific antigen."),
    `05_gleason_sensitivity`=caption(
      "Figure 5. Gleason-score sensitivity analysis of platinum-treatment prediction.",
      "Held-out test mean AUC(t) (a) and Harrell C-index (b) for Elastic-Net Cox and XGBoost survival models at 0, 90, and 180 days after ADT initiation. Within each model class, laboratory-feature models are compared with models using Gleason score. Values are printed above the bars, and the dotted horizontal line marks 0.5.",
      "The source-data cohorts contained 1,144 patients at day 0, 1,173 at day 90, and 1,171 at day 180, including training/validation and test partitions. ADT, androgen-deprivation therapy; AUC(t), time-dependent area under the receiver-operating-characteristic curve; C-index, concordance index."),
    `06_dfci_cohort_sensitivity`=caption(
      "Figure 6. Cohort-size, endpoint-incidence, and laboratory-association sensitivity analyses within Dana-Farber.",
      "Patients eligible at the 180-day androgen-deprivation therapy (ADT) landmark are shown across the full ADT cohort, cohorts excluding prior castration, metastatic cohorts defined from ADT treatment intent, and metastatic cohorts defined by the large language model (LLM). (a) Observed platinum and neuroendocrine prostate cancer (NEPC) event fractions during follow-up with 95% Wilson confidence intervals. (b) Pale bars show eligible patients and solid bars show observed events; labels give events/patients. Blue denotes platinum and orange denotes NEPC in a and b; an asterisk denotes fewer than 25 events. (c) Natural-log hazard ratios per standard-deviation increase and 95% confidence intervals for the mean, minimum, maximum, and last PSA or testosterone value through the landmark. The dashed line at 0 marks the null; filled points indicate Benjamini-Hochberg q < 0.05.",
      "Cohort definitions overlap. Event fractions are observed proportions rather than censoring-adjusted cumulative incidence estimates. Change-from-baseline and observation-count features are excluded from c. 'No prior castrate' denotes the source no-previous-castration restriction. CI, confidence interval; PSA, prostate-specific antigen; SD, standard deviation."),
    `07_federated_incidence_associations`=caption(
      "Figure 7. Site-specific platinum incidence and federated PSA and testosterone associations.",
      "The Dana-Farber, Fred Hutch, and Johns Hopkins ADT cohorts are shown at treatment initiation. (a) Observed platinum-event fractions during follow-up with 95% Wilson confidence intervals. (b) Pale bars show analyzed patients and solid bars show observed platinum events; labels give events/patients. (c,d) Natural-log hazard ratios per standard-deviation increase and 95% confidence intervals for PSA (c) and testosterone (d) at the three sites and in the supplied federated analysis. Columns show the mean, minimum, maximum, and last observed value; rows show landmarks at 0, 90, and 180 days. Dashed lines at 0 mark the null. Open circles indicate nonsignificant associations, filled circles indicate nominal P < 0.05 without false-discovery-rate significance, and diamonds indicate supplied q < 0.05.",
      "Site-specific fractions are observed event proportions rather than censoring-adjusted cumulative incidence estimates. For PSA, federated modeled/observed sample sizes were 6,346/4,250 at day 0, 6,272/5,516 at day 90, and 6,125/5,753 at day 180. Corresponding testosterone sample sizes were 6,346/734, 6,272/1,419, and 6,125/1,567. Federated patient and event counts match the combined Fred Hutch and Johns Hopkins counts, but membership was inferred and not independently verified. Supplied q values were not recalculated. ADT, androgen-deprivation therapy; PSA, prostate-specific antigen; SD, standard deviation."),
    `08_federated_xgboost`=caption(
      "Figure 8. Federated XGBoost performance and feature importance for prediction of platinum treatment.",
      "(a,b) Held-out test mean AUC(t) (a) and Harrell C-index (b) for the XGBoost survival model and age-only XGBoost baseline at 0, 90, and 180 days after ADT initiation; the dotted horizontal line marks 0.5. (c-e) Positive split-gain feature importances at day 0 (c), day 90 (d), and day 180 (e), with up to 15 features displayed per landmark. Colors identify laboratory categories.",
      "Test-set sample sizes/events were 1,270/25 at day 0, 1,255/23 at day 90, and 1,226/22 at day 180 for both model configurations. Performance values are from held-out test sets rather than training or tuning cross-validation. Gains are unsigned measures of split improvement and are not signed effects or SHAP values. The supplied model-input audit reports nonzero gain for person_id at days 90 and 180; model inputs require audit before performance or importance is interpreted. ADT, androgen-deprivation therapy; ALP, alkaline phosphatase; AUC(t), time-dependent area under the receiver-operating-characteristic curve; BUN, blood urea nitrogen; C-index, concordance index; MCV, mean corpuscular volume; PSA, prostate-specific antigen; RDW, red-cell distribution width; SHAP, Shapley additive explanations.")
  )
}

manuscript_write_captions <- function(root,keys=names(manuscript_captions()),capture=NULL) {
  captions <- manuscript_captions()
  unknown <- setdiff(keys,names(captions))
  if(length(unknown)) stop("No manuscript caption defined for: ",paste(unknown,collapse=", "))
  caption_root <- file.path(root,"captions")
  dir.create(caption_root,recursive=TRUE,showWarnings=FALSE)
  paths <- character()
  for(key in keys) {
    number <- match(key,names(captions))
    for(extension in c("md","txt")) {
      path <- file.path(caption_root,sprintf("figure%d.%s",number,extension))
      writeLines(captions[[key]],path,useBytes=TRUE)
      paths <- c(paths,path)
      if(is.function(capture)) capture(path)
    }
  }
  invisible(paths)
}

manuscript_theme <- function() ggplot2::theme(
  text=ggplot2::element_text(family="sans",size=8,color="#222222"),
  plot.title=ggplot2::element_text(size=9,face="bold",margin=ggplot2::margin(b=5)),
  plot.subtitle=ggplot2::element_text(size=7,margin=ggplot2::margin(b=4)),
  plot.tag=ggplot2::element_text(size=11,face="bold"),
  axis.text=ggplot2::element_text(size=7,color="#222222"),
  axis.text.x=ggplot2::element_text(size=7,color="#222222"),
  axis.text.y=ggplot2::element_text(size=7,color="#222222"),
  axis.title=ggplot2::element_text(size=8,face="bold"),
  axis.title.x=ggplot2::element_text(size=8,face="bold"),
  axis.title.y=ggplot2::element_text(size=8,face="bold"),
  strip.text=ggplot2::element_text(size=8,face="bold"),
  legend.text=ggplot2::element_text(size=7,margin=ggplot2::margin(l=2,r=4)),
  legend.title=ggplot2::element_text(size=7), legend.position="bottom",
  legend.justification="center",legend.box.just="center",
  legend.key.size=grid::unit(9,"pt"),legend.spacing.x=grid::unit(3,"pt"),
  legend.key.width=grid::unit(9,"pt"),legend.key.height=grid::unit(9,"pt"),
  legend.margin=ggplot2::margin(0,0,0,0),legend.box.margin=ggplot2::margin(0,0,0,0),
  panel.spacing=grid::unit(6,"pt"),plot.title.position="plot",
  plot.margin=ggplot2::margin(5,5,5,5))

manuscript_style <- function(p, title=NULL, tag=NULL) {
  if(!inherits(p,"ggplot")) return(p)
  # Clone layers: ggproto layers are references also held by the individual plot.
  p$layers <- lapply(p$layers,function(layer) {
    copy <- ggplot2::ggproto(NULL,layer)
    if(inherits(layer$geom,"GeomText") || inherits(layer$geom,"GeomLabel") ||
       inherits(layer$geom,"GeomTextRepel") || inherits(layer$geom,"GeomLabelRepel"))
      copy$aes_params$size <- min(2.5,if(is.null(layer$aes_params$size)) 2.5 else layer$aes_params$size)
    if(inherits(layer$geom,"GeomPoint") && !is.null(layer$aes_params$size))
      copy$aes_params$size <- min(1.8,layer$aes_params$size)
    copy
  })
  # Panel descriptions belong in the legend; only letter tags remain above panels.
  result <- p + ggplot2::labs(title=NULL,tag=if(is.null(tag)) NULL else tolower(tag),caption=NULL) + manuscript_theme()
  for(field in c("axis.text","axis.text.x","axis.text.y","axis.title.x","axis.title.y"))
    if(inherits(p$theme[[field]],"element_blank") ||
       (startsWith(field,"axis.text.") && inherits(p$theme$axis.text,"element_blank")))
      result$theme[[field]] <- ggplot2::element_blank()
  result
}

manuscript_grob <- function(p,width,height) {
  path <- tempfile(fileext=".png")
  figure_measure_device(path,width,height)
  on.exit({grDevices::dev.off();unlink(path)})
  if(inherits(p,"ggplot")) ggplot2::ggplotGrob(p) else p
}

manuscript_combine <- function(plots,spec,shared_legend=FALSE,tag_offset=0L) {
  legends <- list(); grobs <- list()
  inline_legend_panels <- if(is.null(spec$shared_legend_panels)) integer() else
    spec$shared_legend_panels
  for(i in seq_along(plots)) {
    p <- manuscript_style(plots[[i]],spec$titles[i],
      if(length(plots)>1) letters[i+tag_offset] else NULL)
    if(identical(spec$key,"03_dfci_univariable") && i<=3) p <- p + ggplot2::theme(
      plot.subtitle=ggplot2::element_text(size=8,face="bold",hjust=.5,
        margin=ggplot2::margin(b=2)),
      plot.margin=ggplot2::margin(4,1,4,1))
    if(inherits(p,"ggplot")) {
      if(shared_legend || i %in% inline_legend_panels) {
        g <- manuscript_grob(p,spec$width,spec$height)
        boxes <- Filter(function(x) inherits(x,"gtable"),g$grobs[grepl("^guide-box",g$layout$name)])
        if(!length(legends) && length(boxes)) legends <- boxes[1]
        p <- p + ggplot2::theme(legend.position="none")
      }
      # Counts and long explanatory subtitles are retained in the sidecar legend.
      if(startsWith(spec$key,"04_") || startsWith(spec$key,"05_"))
        p <- p + ggplot2::labs(subtitle=NULL)
    }
    cells <- which(spec$layout==i,arr.ind=TRUE)
    row_heights <- if(is.null(spec$row_heights)) rep(1,nrow(spec$layout)) else spec$row_heights
    grobs[[i]] <- manuscript_grob(p,spec$width*length(unique(cells[,2]))/ncol(spec$layout),
      spec$height*sum(row_heights[unique(cells[,1])])/sum(row_heights))
  }
  if(length(inline_legend_panels)) {
    legend_cell <- setdiff(unique(as.vector(spec$layout)),seq_along(plots))
    if(length(legend_cell)!=1 || !length(legends))
      stop("Inline shared legend requires exactly one legend layout cell and one extractable legend")
    grobs[[legend_cell]] <- legends[[1]]
  }
  footer <- if(length(legends) && !length(inline_legend_panels)) legends[[1]] else NULL
  gridExtra::arrangeGrob(grobs=grobs,layout_matrix=spec$layout,heights=row_heights,bottom=footer,
    padding=grid::unit(3,"pt"))
}

manuscript_legend <- function(p,width,height) {
  g <- manuscript_grob(p,width,height)
  boxes <- Filter(function(x) inherits(x,"gtable"),g$grobs[grepl("^guide-box",g$layout$name)])
  if(length(boxes)) boxes[[1]] else NULL
}

manuscript_center_landmark_header <- function(p) p + ggplot2::theme(
  plot.subtitle=ggplot2::element_text(size=8,face="bold",hjust=.5,
    margin=ggplot2::margin(b=3)))

manuscript_multivariable_labs <- function(plots,spec) {
  stopifnot(length(plots)==8)
  performance <- vector("list",2)
  model_legend <- NULL
  for(i in 1:2) {
    p <- manuscript_style(plots[[i]] + ggplot2::labs(subtitle=NULL),spec$titles[i],letters[i]) +
      ggplot2::guides(fill=ggplot2::guide_legend(nrow=1,byrow=TRUE))
    if(is.null(model_legend)) model_legend <- manuscript_legend(p,spec$width,spec$height)
    performance[[i]] <- manuscript_grob(p + ggplot2::theme(legend.position="none"),
      spec$width,spec$height*.18)
  }
  landmarks <- c("0 days","+90 days","+180 days")
  importance <- vector("list",6)
  category_legend <- NULL
  for(i in seq_len(6)) {
    p <- manuscript_style(plots[[i+2]] + ggplot2::labs(subtitle=landmarks[(i-1)%%3+1]),NULL,NULL) +
      ggplot2::guides(color="none",fill=ggplot2::guide_legend(nrow=1,byrow=TRUE,
        override.aes=list(alpha=1)))
    p <- manuscript_center_landmark_header(p)
    if(is.null(category_legend)) category_legend <- manuscript_legend(p,spec$width,spec$height)
    importance[[i]] <- manuscript_grob(p + ggplot2::theme(legend.position="none"),
      spec$width/3,spec$height*.24)
  }
  tag <- function(letter) grid::textGrob(tolower(letter),x=grid::unit(0,"npc"),hjust=0,
    gp=grid::gpar(fontfamily="sans",fontsize=11,fontface="bold",col="#222222"))
  coefficient_row <- gridExtra::arrangeGrob(grobs=importance[1:3],ncol=3,top=tag("C"),
    padding=grid::unit(1,"pt"))
  xgboost_row <- gridExtra::arrangeGrob(grobs=importance[4:6],ncol=3,top=tag("D"),
    padding=grid::unit(1,"pt"))
  grobs <- performance
  heights <- spec$row_heights[1:2]
  if(!is.null(model_legend)) {
    grobs <- c(grobs,list(model_legend))
    heights <- c(heights,manuscript_multivariable_legend_height)
  }
  grobs <- c(grobs,list(coefficient_row))
  heights <- c(heights,spec$row_heights[3])
  if(!is.null(category_legend)) {
    grobs <- c(grobs,list(category_legend))
    heights <- c(heights,manuscript_multivariable_legend_height)
  }
  grobs <- c(grobs,list(xgboost_row))
  heights <- c(heights,spec$row_heights[4])
  gridExtra::arrangeGrob(grobs=grobs,ncol=1,heights=heights,
    padding=grid::unit(3,"pt"))
}

manuscript_cohort_labels <- c(adt="All ADT",adt_noprecastrate="All ADT; no prior castrate",
  adt_metastatic_adt="Metastatic (ADT intent)",
  adt_metastatic_adt_noprecastrate="Metastatic (ADT intent);\nno prior castrate",
  adt_metastatic_llm="Metastatic (LLM)",
  adt_metastatic_llm_noprecastrate="Metastatic (LLM);\nno prior castrate")

manuscript_incidence <- function(d) {
  d <- d[tolower(d$endpoint) %in% c("platinum","nepc"),,drop=FALSE]
  d$endpoint <- tolower(d$endpoint)
  stopifnot(nrow(d)>0,!anyDuplicated(d[c("cohort","endpoint")]))
  cohort_order <- c(intersect(names(manuscript_cohort_labels),unique(d$cohort)),
    setdiff(unique(d$cohort),names(manuscript_cohort_labels)))
  d$label <- unname(manuscript_cohort_labels[d$cohort])
  d$label[is.na(d$label)] <- d$cohort[is.na(d$label)]
  d$available <- d$status=="ok" & !is.na(d$n_patients) & d$n_patients>0
  stopifnot(all(d$n_events[d$available]>=0),all(d$n_events[d$available]<=d$n_patients[d$available]))
  good <- d[d$available,,drop=FALSE]
  z <- 1.96; p <- good$n_events/good$n_patients; den <- 1+z^2/good$n_patients
  centre <- (p+z^2/(2*good$n_patients))/den
  half <- z*sqrt(p*(1-p)/good$n_patients+z^2/(4*good$n_patients^2))/den
  good$rate <- 100*p; good$lo <- 100*pmax(0,centre-half);good$hi <- 100*pmin(1,centre+half)
  good$endpoint_label <- factor(good$endpoint,levels=c("platinum","nepc"),labels=c("Platinum","NEPC"))
  good$y <- length(cohort_order)+1-match(good$cohort,cohort_order)+
    ifelse(good$endpoint=="platinum",.18,-.18)
  colors <- c(Platinum="#2a78d6",NEPC="#eb6834")
  labels <- unname(manuscript_cohort_labels[cohort_order])
  labels[is.na(labels)] <- cohort_order[is.na(labels)]
  yscale <- function(show=TRUE) ggplot2::scale_y_continuous(
    breaks=rev(seq_along(cohort_order)),labels=if(show) labels else NULL,
    limits=c(.45,length(cohort_order)+.55),expand=ggplot2::expansion(mult=0))
  a <- ggplot2::ggplot(good,ggplot2::aes(y=y,fill=endpoint_label)) +
    ggplot2::geom_rect(ggplot2::aes(xmin=0,xmax=rate,ymin=y-.14,ymax=y+.14),color="white",linewidth=.2) +
    ggplot2::geom_errorbar(ggplot2::aes(xmin=lo,xmax=hi),orientation="y",width=.09,linewidth=.4,color="#454545") +
    ggplot2::geom_text(ggplot2::aes(x=hi,label=sprintf("%.1f%%%s",rate,ifelse(n_events<25,"*",""))),
      hjust=-.18,size=2.5,color="#52514e") +
    ggplot2::scale_fill_manual(values=colors,drop=FALSE) + yscale(TRUE) +
    ggplot2::scale_x_continuous(limits=c(0,max(good$hi)*1.22),expand=ggplot2::expansion(mult=c(0,.01))) +
    ggplot2::labs(x="Observed events (%)",y=NULL,fill=NULL) + ggplot2::theme_classic() +
    ggplot2::theme(panel.grid.major.x=ggplot2::element_line(color="grey92",linewidth=.3),
      axis.ticks.y=ggplot2::element_blank(),axis.line.y=ggplot2::element_blank())
  b <- ggplot2::ggplot(good,ggplot2::aes(y=y,fill=endpoint_label)) +
    ggplot2::geom_rect(ggplot2::aes(xmin=0,xmax=n_patients,ymin=y-.14,ymax=y+.14),alpha=.28) +
    ggplot2::geom_rect(ggplot2::aes(xmin=0,xmax=n_events,ymin=y-.14,ymax=y+.14)) +
    ggplot2::geom_text(ggplot2::aes(x=n_patients,label=paste0(scales::comma(n_events),"/",scales::comma(n_patients))),
      hjust=-.1,size=2.5,color="#52514e") +
    ggplot2::scale_fill_manual(values=colors,drop=FALSE) + yscale(FALSE) +
    ggplot2::scale_x_continuous(limits=c(0,max(good$n_patients)*1.27),expand=ggplot2::expansion(mult=c(0,.01)),labels=scales::comma) +
    ggplot2::labs(x="Patients (events overlaid)",y=NULL,fill=NULL) + ggplot2::theme_classic() +
    ggplot2::theme(panel.grid.major.x=ggplot2::element_line(color="grey92",linewidth=.3),
      axis.text.y=ggplot2::element_blank(),axis.ticks.y=ggplot2::element_blank(),axis.line.y=ggplot2::element_blank())
  # Explicit missing rows, never substituted with zero events.
  missing <- d[!d$available,,drop=FALSE]
  missing$y <- length(cohort_order)+1-match(missing$cohort,cohort_order)+
    ifelse(missing$endpoint=="platinum",.18,-.18)
  if(nrow(missing)) for(which in c("a","b")) {
    p <- get(which)+ggplot2::geom_text(data=missing,ggplot2::aes(x=0,y=y,label="Unavailable"),
      inherit.aes=FALSE,hjust=0,size=2.5)
    assign(which,p)
  }
  spec <- list(key="06_dfci_cohort_sensitivity",width=7.2,height=3.6,
    layout=matrix(1:2,1),titles=c("Endpoint incidence","Cohort size and event count"))
  list(plot=manuscript_combine(list(a,b),spec,shared_legend=TRUE),panels=list(a,b),spec=spec,data=d,
    legend=paste("Landmark day",paste(unique(d$landmark_days),collapse=", "),
      ". a: observed platinum and NEPC event fractions during follow-up with 95% Wilson confidence intervals.",
      "b: pale bars show eligible patients; solid bars show observed events; labels are events/patients.",
      "Blue denotes platinum and orange denotes NEPC; * marks fewer than 25 events.",
      "These are not censoring-adjusted cumulative incidences or risks by the landmark day. Cohorts overlap.",
      "'No prior castrate' denotes the source noprecastrate cohort restriction."))
}

manuscript_log_hazard_axis <- function(p,outer_breaks=FALSE) {
  stopifnot(inherits(p,"ggplot"),all(c("hazard_ratio_per_sd","ci_lower","ci_upper") %in% names(p$data)))
  values <- unlist(p$data[c("hazard_ratio_per_sd","ci_lower","ci_upper")],use.names=FALSE)
  values <- values[is.finite(values) & values>0]
  stopifnot(length(values)>0)
  log_limits <- range(c(0,log(values)))
  log_limits <- log_limits+c(-1,1)*max(.04,diff(log_limits)*.04)
  ticks <- pretty(log_limits,n=5)
  ticks <- ticks[ticks>=log_limits[1] & ticks<=log_limits[2]]
  if(!any(abs(ticks)<sqrt(.Machine$double.eps))) ticks <- sort(unique(c(ticks,0)))
  if(isTRUE(outer_breaks)) ticks <- sort(unique(c(range(ticks),0)))
  p + ggplot2::scale_x_log10(limits=exp(log_limits),breaks=exp(ticks),
    labels=function(x) scales::label_number(accuracy=.1)(log(x))) +
    ggplot2::labs(x="Log hazard ratio per SD (95% CI)")
}

manuscript_dfci_cohort_sensitivity <- function(incidence,association) {
  stopifnot(is.list(incidence),inherits(association,"ggplot"))
  association <- manuscript_log_hazard_axis(association) +
    ggplot2::scale_y_discrete(labels=function(x) unname(manuscript_cohort_labels[x]))
  association_grob <- manuscript_grob(manuscript_style(association,NULL,"c"),7.2,5)
  spec <- list(key="06_dfci_cohort_sensitivity",width=7.2,height=8.7,
    layout=rbind(c(1,2),c(3,3)),titles=c("Endpoint incidence","Cohort size and event count",
      "PSA and testosterone associations"),row_heights=c(3.6,5.1))
  combined <- gridExtra::arrangeGrob(grobs=list(incidence$plot,association_grob),ncol=1,
    heights=spec$row_heights,padding=grid::unit(3,"pt"))
  incidence_data <- incidence$data;incidence_data$figure_section <- "incidence_and_counts"
  association_data <- association$data;association_data$figure_section <- "laboratory_associations"
  list(plot=combined,spec=spec,data=dplyr::bind_rows(incidence_data,association_data),
    legend=paste(incidence$legend,
      "c: natural-log hazard ratios per SD and 95% CIs for PSA and testosterone at day 180; dashed lines at 0 mark the null. Filled points: q < 0.05; open points: q >= 0.05 or unavailable.",
      sep="\n"))
}

manuscript_federated_incidence <- function(d) {
  d <- d[tolower(d$endpoint)=="platinum" & d$landmark_days==0,,drop=FALSE]
  keys <- c("dana_farber_caia_1_1","fred_hutch_caia_1_1","jhu_caia_1_1")
  selected <- d[match(keys,d$site_name),,drop=FALSE]
  stopifnot(nrow(selected)==3,!anyNA(selected$site_name),!anyDuplicated(selected$site_name),
    all(selected$available %in% TRUE),all(selected$n_events>=0),
    all(selected$n_events<=selected$n_patients))
  out <- data.frame(site_key=keys,
    site=c("Dana-Farber","Fred Hutch","Johns Hopkins"),
    landmark_days=0L,endpoint="platinum",
    n_patients=selected$n_patients,n_events=selected$n_events,available=TRUE,
    stringsAsFactors=FALSE)
  z <- 1.96;p <- out$n_events/out$n_patients;den <- 1+z^2/out$n_patients
  centre <- (p+z^2/(2*out$n_patients))/den
  half <- z*sqrt(p*(1-p)/out$n_patients+z^2/(4*out$n_patients^2))/den
  out$event_incidence_pct <- 100*p
  out$ci_lower_pct <- 100*pmax(0,centre-half)
  out$ci_upper_pct <- 100*pmin(1,centre+half)
  out$y <- rev(seq_len(nrow(out)))
  yscale <- function(show=TRUE) ggplot2::scale_y_continuous(breaks=out$y,
    labels=if(show) out$site else NULL,limits=c(.5,nrow(out)+.5),expand=ggplot2::expansion(mult=0))
  a <- ggplot2::ggplot(out,ggplot2::aes(y=y)) +
    ggplot2::geom_rect(ggplot2::aes(xmin=0,xmax=event_incidence_pct,ymin=y-.22,ymax=y+.22),fill="#2a78d6") +
    ggplot2::geom_errorbar(ggplot2::aes(xmin=ci_lower_pct,xmax=ci_upper_pct),orientation="y",
      width=.12,color="#454545",linewidth=.4) +
    ggplot2::geom_text(ggplot2::aes(x=ci_upper_pct,label=sprintf("%.2f%%",event_incidence_pct)),
      hjust=-.18,size=2.5,color="#52514e") + yscale(TRUE) +
    ggplot2::scale_x_continuous(limits=c(0,max(out$ci_upper_pct)*1.25),
      expand=ggplot2::expansion(mult=c(0,.01))) +
    ggplot2::labs(x="Observed platinum events (%)",y=NULL) + ggplot2::theme_classic() +
    ggplot2::theme(panel.grid.major.x=ggplot2::element_line(color="grey92",linewidth=.3),
      axis.ticks.y=ggplot2::element_blank(),axis.line.y=ggplot2::element_blank())
  b <- ggplot2::ggplot(out,ggplot2::aes(y=y)) +
    ggplot2::geom_rect(ggplot2::aes(xmin=0,xmax=n_patients,ymin=y-.22,ymax=y+.22),fill="#2a78d6",alpha=.28) +
    ggplot2::geom_rect(ggplot2::aes(xmin=0,xmax=n_events,ymin=y-.22,ymax=y+.22),fill="#2a78d6") +
    ggplot2::geom_text(ggplot2::aes(x=n_patients,label=paste0(scales::comma(n_events),"/",scales::comma(n_patients))),
      hjust=-.1,size=2.5,color="#52514e") + yscale(FALSE) +
    ggplot2::scale_x_continuous(limits=c(0,max(out$n_patients)*1.28),
      expand=ggplot2::expansion(mult=c(0,.01)),labels=scales::comma) +
    ggplot2::labs(x="Patients (events overlaid)",y=NULL) + ggplot2::theme_classic() +
    ggplot2::theme(panel.grid.major.x=ggplot2::element_line(color="grey92",linewidth=.3),
      axis.text.y=ggplot2::element_blank(),axis.ticks.y=ggplot2::element_blank(),axis.line.y=ggplot2::element_blank())
  spec <- list(key="07_federated_incidence_associations",width=7.2,height=3,
    layout=matrix(1:2,1),titles=c("Platinum incidence","Cohort size and event count"))
  list(plot=manuscript_combine(list(a,b),spec),panels=list(a,b),spec=spec,data=out,
    legend=paste("Landmark day 0. a: observed platinum-event fraction during follow-up with 95% Wilson confidence intervals.",
      "b: pale bars show analyzed patients; solid bars show platinum events; labels are events/patients.",
      "Observed proportions are not censoring-adjusted cumulative incidences."))
}

manuscript_federated_labs <- function(d) {
  d <- d[d$feature_stat %in% c("mean","min","max","last") &
    d$landmark_days %in% c(0,90,180),,drop=FALSE]
  stopifnot(!anyDuplicated(d[c("lab_name","landmark_days","source","feature_stat")]))
  note <- paste(unique(d$population_note[d$source=="Federated*"]),collapse="\n")
  # Reuse the standalone forests: source colors, landmark/statistic facets and
  # significance symbols retain their original meaning; relabel the transformed
  # horizontal coordinate directly as the natural-log hazard ratio.
  plots <- lapply(c("PSA","Testosterone"),function(analyte)
    manuscript_log_hazard_axis(
      plot_federated_comparison(d,analyte,note) + ggplot2::labs(subtitle=NULL),
      outer_breaks=analyte=="PSA"))
  spec <- list(key="07_federated_incidence_associations",width=7.2,height=8.4,
    layout=matrix(1:2,2),titles=c("PSA","Testosterone"))
  list(plot=manuscript_combine(plots,spec,shared_legend=TRUE),panels=plots,spec=spec,data=d,
    legend=paste("a: PSA. b: testosterone. Platinum endpoint; original site colors and landmark/statistic facets.",
      "Columns: mean, minimum, maximum and last value. Rows: days 0, 90 and 180.",
      "Horizontal coordinates are natural-log hazard ratios per SD; dashed lines at 0 mark the null.",
      "Open circles: not significant; filled circles: nominal p < 0.05 only; diamonds: supplied FDR q < 0.05.",
      paste(vapply(plots,function(p) p$labels$caption,character(1)),collapse="\n"),sep="\n"))
}

manuscript_federated_incidence_associations <- function(incidence,labs) {
  stopifnot(is.list(incidence),is.list(labs),length(labs$panels)==2)
  lab_spec <- labs$spec
  lab_spec$height <- 8.4
  lab_grob <- manuscript_combine(labs$panels,lab_spec,shared_legend=TRUE,tag_offset=2L)
  spec <- list(key="07_federated_incidence_associations",width=7.2,height=11.5,
    layout=rbind(c(1,2),c(3,3),c(4,4)),titles=c("Platinum incidence","Cohort size and event count",
      "PSA associations","Testosterone associations"),row_heights=c(3,4.25,4.25))
  combined <- gridExtra::arrangeGrob(grobs=list(incidence$plot,lab_grob),ncol=1,
    heights=c(spec$row_heights[1],sum(spec$row_heights[-1])),padding=grid::unit(3,"pt"))
  incidence_data <- incidence$data;incidence_data$figure_section <- "incidence_and_counts"
  lab_data <- labs$data;lab_data$figure_section <- "laboratory_associations"
  list(plot=combined,spec=spec,data=dplyr::bind_rows(incidence_data,lab_data),
    legend=paste(incidence$legend,labs$legend,sep="\n"))
}

manuscript_federated_xgb <- function(metrics,importance) {
  d <- prepare_federated_xgboost_performance(metrics)
  colors <- c("XGBoost Survival"="#B58900","XGBoost baseline (age)"="#E0CC8A")
  plots <- lapply(c("auc","cindex"),function(metric)
    plot_model_discrimination(d,metric,if(metric=="auc") "Test mean AUC(t)" else "Test C-index",
      colors,show_legend=TRUE))
  for(lm in c(0,90,180)) {
    selected <- importance[importance$landmark_days==lm & importance$displayed %in% TRUE,,drop=FALSE]
    plots[[length(plots)+1L]] <- plot_model_importance(selected,"xgb",NULL) +
      ggplot2::scale_y_discrete(labels=function(x) {
        x <- gsub("Alkaline phosphatase","ALP",x,fixed=TRUE)
        gsub("Neutrophils absolute","Neutrophils (abs.)",x,fixed=TRUE)
      }) + ggplot2::guides(fill=ggplot2::guide_legend(nrow=1))
  }
  spec <- list(key="08_federated_xgboost",width=7.2,height=10.2,
    layout=rbind(c(1,1,2,2),c(3,3,3,3),c(4,4,4,4),c(5,5,5,5)),
    row_heights=c(.85,.65,1,1.2),
    titles=c("Test mean AUC(t)","Test C-index","Feature importance: day 0",
      "Feature importance: day 90","Feature importance: day 180"))
  # Share the category legend across importance panels. Keep the performance
  # legend in A/B because its colors encode model configuration, not lab class.
  for(i in 3:4) plots[[i]] <- plots[[i]] + ggplot2::guides(fill="none")
  p <- manuscript_combine(plots,spec)
  audit <- unique(c(metrics$input_audit_note,importance$input_audit_note))
  audit <- audit[!is.na(audit)&nzchar(audit)]
  counts <- paste(sprintf("Day %d, %s: test n=%s; events=%s",metrics$landmark_days,
    metrics$config,metrics$n_test,metrics$n_events_test),collapse="\n")
  list(plot=p,spec=spec,data=metrics,legend=paste(
    "a: test mean AUC(t). b: test C-index. c–e: feature importance at days 0, 90 and 180.",
    "Held-out test metrics, not training or tuning CV. Original gold model/baseline bars and lab-category importance colors.",
    "Up to 15 positive-gain features per landmark, using the original display exclusions. ALP: alkaline phosphatase; abs.: absolute count.",
    "Gains are supplied split gains, not signed effects or SHAP values.",counts,paste(audit,collapse="\n"),sep="\n"))
}

manuscript_build <- function(items,tables,root) {
  result <- list()
  for(spec in manuscript_specs()) {
    if(!all(spec$members %in% names(items))) next
    plots <- lapply(items[spec$members],`[[`,"plot")
    legend <- unlist(lapply(seq_along(plots),function(i) {
      p <- plots[[i]]
      paste0(letters[i],". ",paste(c(p$labels$title,p$labels$subtitle,p$labels$caption),collapse="\n"))
    }))
    if(spec$key=="04_multivariable_labs") {
      legend <- c(
        paste0("a. ",paste(c(plots[[1]]$labels$title,plots[[1]]$labels$subtitle),collapse="\n")),
        paste0("b. ",paste(c(plots[[2]]$labels$title,plots[[2]]$labels$subtitle),collapse="\n")),
        "c. Elastic-Net model coefficients at landmark days 0, 90, and 180.",
        "d. XGBoost feature importances at landmark days 0, 90, and 180. Gains are unsigned split gains.")
      result[[spec$key]] <- list(plot=manuscript_multivariable_labs(plots,spec),spec=spec,
        legend=paste(legend,collapse="\n\n"),data=NULL)
      next
    }
    if(spec$key=="02_llm_nepc_avpc") {
      plots[[3]] <- plots[[3]] + ggplot2::labs(y="Fraction within\nplatinum group")
      plots[[4]] <- plots[[4]] + ggplot2::labs(y="Platinum-positive fraction")
    }
    if(spec$key=="03_dfci_univariable") {
      axes <- lapply(plots[1:3],function(p) p$coordinates$limits)
      if(all(vapply(axes,function(x) length(x$x)==2,logical(1))))
        for(i in 1:3) plots[[i]] <- plots[[i]] + ggplot2::coord_cartesian(
          xlim=range(unlist(lapply(axes,`[[`,"x"))),ylim=manuscript_dfci_volcano_ylim)
      for(i in 1:3) {
        plots[[i]]$layers <- lapply(plots[[i]]$layers,function(layer) {
          copy <- ggplot2::ggproto(NULL,layer)
          if(inherits(layer$geom,"GeomTextRepel")) {
            copy$aes_params$size <- 2.2
            copy$aes_params$point.size <- 1
            copy$geom_params$box.padding <- grid::unit(.18,"lines")
            copy$geom_params$point.padding <- grid::unit(.12,"lines")
          }
          if(inherits(layer$geom,"GeomPoint")) copy$aes_params$stroke <- .3
          copy
        })
        plots[[i]] <- plots[[i]] + ggplot2::scale_size_manual(
          values=c(`TRUE`=1.8,`FALSE`=1.2),guide="none")
      }
      for(i in 1:3) {
        plots[[i]] <- plots[[i]] + ggplot2::labs(
          subtitle=manuscript_dfci_landmark_headers[i],y=if(i==1) expression(-log[10](p)) else NULL) +
          ggplot2::guides(color="none",fill=ggplot2::guide_legend(nrow=1,byrow=TRUE,
            override.aes=list(shape=21,size=1.8,alpha=1)))
        if(i>1) plots[[i]] <- plots[[i]] + ggplot2::theme(
          axis.text.y=ggplot2::element_blank(),axis.ticks.y=ggplot2::element_blank())
      }
      for(i in c(4,6)) plots[[i]] <- plots[[i]] + ggplot2::labs(subtitle=NULL,
        x="Years from ADT initiation",y=paste0("Mean log1p(",if(i==4) "PSA" else "testosterone", ")\n(95% CI)"))
      for(i in c(5,7)) {
        logrank <- plots[[i]]$labels$subtitle
        plots[[i]] <- plots[[i]] + ggplot2::labs(subtitle=NULL,
          x=paste0("Days from day-",manuscript_dfci_km_landmark," landmark"),y=manuscript_km_y_label)
        if(is.character(logrank) && length(logrank)==1 && nzchar(logrank))
          plots[[i]] <- plots[[i]] + ggplot2::annotate("text",x=Inf,y=.05,label=logrank,
            hjust=1.08,vjust=0,size=2.3,color="#52514e")
      }
      legend <- c(legend,
        "Panels a-c share a -log10(p) display range through 20 and centered landmark headers.")
    }
    if(spec$key=="01_cohort_overview") {
      # Patient counts stay in the external legend; omit them from panels B-E.
      for(i in 2:5) plots[[i]] <- plots[[i]] + ggplot2::labs(subtitle=NULL)
      plots[[1]] <- plots[[1]] + ggplot2::labs(x=NULL,y=NULL) + ggplot2::theme(axis.text=ggplot2::element_blank())
      if(all(c("ycen","ymin","ymax") %in% names(plots[[1]]$data))) {
        plots[[1]]$data$ymin <- plots[[1]]$data$ycen-.4
        plots[[1]]$data$ymax <- plots[[1]]$data$ycen+.4
        plots[[1]]$layers <- lapply(plots[[1]]$layers,function(layer) {
          copy <- ggplot2::ggproto(NULL,layer)
          if(inherits(copy$geom,"GeomSegment") && is.data.frame(copy$data)) {
            copy$data$y <- copy$data$y-.09;copy$data$yend <- copy$data$yend+.09
          }
          copy
        })
      }
      plots[[2]] <- plots[[2]] + ggplot2::labs(y="Platinum-free probability") + ggplot2::guides(color="none",fill="none")
      plots[[4]] <- plots[[4]] + ggplot2::labs(x="Days from diagnosis to ADT")
      plots[[5]] <- plots[[5]] + ggplot2::labs(x="Days from ADT to platinum")
    }
    result[[spec$key]] <- list(plot=manuscript_combine(plots,spec,
      shared_legend=startsWith(spec$key,"04_")||startsWith(spec$key,"05_")),spec=spec,
      legend=paste(legend,collapse="\n\n"),
      data=NULL)
  }
  read <- function(pattern) {
    paths <- tables[grepl(pattern,basename(tables))]
    if(length(paths)!=1) return(NULL)
    readr::read_csv(paths,show_col_types=FALSE)
  }
  incidence <- read("^event_incidence_lm180__platinum__all__incl[.]csv$")
  association_key <- "cohort_forest_platinum_landmark180"
  if(!is.null(incidence)) {
    if(association_key %in% names(items)) result[["06"]] <- manuscript_dfci_cohort_sensitivity(
      manuscript_incidence(incidence),items[[association_key]]$plot)
    else message("Skipping manuscript Figure 6: missing captured ",association_key)
  }
  psa <- read("^psa_forest__platinum[.]csv$"); testosterone <- read("^testosterone_forest__platinum[.]csv$")
  site_incidence <- read("^site_incidence_lm000__platinum[.]csv$")
  required_sites <- c("dana_farber_caia_1_1","fred_hutch_caia_1_1","jhu_caia_1_1")
  if(!is.null(psa)&&!is.null(testosterone)&&!is.null(site_incidence) &&
     all(required_sites %in% site_incidence$site_name))
    result[["07"]] <- manuscript_federated_incidence_associations(
      manuscript_federated_incidence(site_incidence),
      manuscript_federated_labs(dplyr::bind_rows(psa,testosterone)))
  metrics <- read("^xgboost_performance__platinum[.]csv$"); importance <- read("^xgboost_importance__platinum[.]csv$")
  if(!is.null(metrics)&&!is.null(importance)) result[["08"]] <- manuscript_federated_xgb(metrics,importance)
  result
}

# ============================================================================
# ---- cached prepare/render workflow ----------------------------------------
# ============================================================================
# Persistent bridge: Polars tables -> R statistics/graphics objects -> PNG/PDF.
# Preparation serializes gtable objects, not ggplot environments (which can
# otherwise retain the entire patient data frame in every saved RDS).
figure_read_parquet <- function(path) {
  if (requireNamespace("arrow", quietly = TRUE)) return(as_tibble(arrow::read_parquet(path)))
  if (requireNamespace("nanoparquet", quietly = TRUE)) return(as_tibble(nanoparquet::read_parquet(path)))
  stop('Prepared figure data require R package "nanoparquet" or "arrow".')
}

figure_prepared_table <- function(source, table) {
  manifest <- getOption("compass.figure_data_manifest")
  if (is.null(manifest)) return(NULL) # direct legacy helper/test compatibility
  resolved <- normalizePath(source, mustWork = FALSE)
  arms <- Filter(function(arm) identical(arm$source, resolved), manifest$arms)
  if (length(arms) != 1L) stop("No prepared figure data for ", source)
  path <- file.path(arms[[1]]$directory, paste0(table, ".parquet"))
  key <- paste0("polars:", path)
  if (!exists(key, envir = .processed_read_cache, inherits = FALSE))
    .processed_read_cache[[key]] <- figure_read_parquet(path)
  .processed_read_cache[[key]]
}

figure_cached_patient_bins <- function(source, df, lab_group, stratum_col,
                                        stratum_values, log_scale) {
  d <- figure_prepared_table(source, "patient_bins")
  if (is.null(d)) return(NULL)
  d <- d %>% filter(LAB_GROUP == lab_group, DFCI_MRN %in% df$DFCI_MRN)
  value_col <- if (log_scale) "value_log" else "value_raw"
  count_col <- if (log_scale) "n_log" else "n_raw"
  d <- d %>% filter(.data[[count_col]] > 0)
  if (!is.null(stratum_values)) {
    lookup <- figure_trajectory_lookup(stratum_values, d$DFCI_MRN)
    d <- inner_join(d, lookup, by = "DFCI_MRN")
  } else if (identical(stratum_col, "plat_group")) {
    d$stratum <- as.character(as.integer(coalesce(d$PLATINUM, 0)))
  } else {
    lookup <- df %>% transmute(DFCI_MRN, stratum = as.character(.data[[stratum_col]])) %>% distinct()
    d <- inner_join(d, lookup, by = "DFCI_MRN")
  }
  if (!nrow(d)) return(tibble())
  edges <- unlist(getOption("compass.figure_data_manifest")$bins$trajectory_edges)
  levels <- levels(cut(numeric(), edges, include.lowest = TRUE, right = FALSE))
  # If measurement-level platinum status differed, recombine its partial means
  # with observation weights before applying an independent classifier stratum.
  d %>% filter(!is.na(stratum)) %>% group_by(DFCI_MRN, bin_id, t_mid, stratum) %>%
    summarise(LAB_VALUE = sum(.data[[value_col]] * .data[[count_col]]) / sum(.data[[count_col]]), .groups = "drop") %>%
    mutate(t_bin = factor(levels[bin_id + 1L], levels = levels)) %>%
    select(DFCI_MRN, t_bin, stratum, LAB_VALUE, t_mid)
}

figure_atomic_rds <- function(value, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(".figure-", tmpdir = dirname(path))
  on.exit(unlink(temporary))
  saveRDS(value, temporary, compress = FALSE)
  if (!file.rename(temporary, path)) stop("Cannot publish ", path)
}

figure_file_identity <- function(paths) {
  info <- file.info(paths)
  data.frame(path = paths, size = info$size, mtime = as.numeric(info$mtime), stringsAsFactors = FALSE)
}

figure_scene_manifest <- function(directory, signature) {
  path <- file.path(directory, "complete.rds")
  if (!file.exists(path)) return(NULL)
  tryCatch({
    m <- readRDS(path)
    if (!identical(m$signature, signature) || !length(m$scenes)) return(NULL)
    paths <- vapply(m$scenes, `[[`, character(1), "path")
    if (!identical(m$files, figure_file_identity(paths)) || anyNA(m$files$size)) return(NULL)
    if (length(m$tables) && !identical(m$table_files, figure_file_identity(m$tables))) return(NULL)
    m
  }, error = function(e) NULL)
}

prepare_figure_scenes <- function(directory, signature, build, force = FALSE, manuscript_root = NULL) {
  m <- if (!force) figure_scene_manifest(directory, signature) else NULL
  if (!is.null(m)) { message("Reused prepared figures: ", basename(dirname(directory))); return(m) }
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  generation <- tempfile("generation-", tmpdir = directory)
  dir.create(generation)
  scenes <- list(); tables <- character(); bundles <- list(); manuscript_items <- list()
  publish <- function(plot, destination, width, height, stem, title = NULL, landmarks = "") {
    previous_destination <- NULL
    if(isTRUE(attr(plot,"compass_compiled"))) {
      previous_destination <- figure_public_path(destination)
      arm_root <- if(grepl("/by_figure/",destination,fixed=TRUE)) sub("/by_figure/.*$","",destination) else
        sub("/(ADT|ARPI)/.*$","/\\1",destination)
      destination <- figure_compiled_path(previous_destination,arm_root)
    }
    path <- file.path(generation, sprintf("panel-%04d.rds", length(scenes) + 1L))
    if(is.null(title)) title <- if(inherits(plot,"ggplot") && length(plot$labels$title))
      paste(plot$labels$title,collapse=" ") else gsub("_"," ",stem)
    grob <- figure_grob(plot,width,height,generation)
    if(!nzchar(landmarks) && inherits(plot,"ggplot")) {
      for(column in c("landmark_days","landmark")) if(column %in% names(plot$data)) {
        values <- unique(gsub("[^0-9]","",as.character(plot$data[[column]])))
        landmarks <- paste(values[nzchar(values)],collapse=",")
        break
      }
    }
    if(!nzchar(landmarks) && grepl("lm[0-9]+",stem)) landmarks <- sub("^.*lm([0-9]+).*$","\\1",stem)
    figure_atomic_rds(grob, path)
    scenes[[length(scenes) + 1L]] <<- list(path = path, destination = figure_public_path(destination),
      width = width, height = height, stem = stem, title = title, landmark = landmarks,
      previous_destination = previous_destination)
    scenes[[length(scenes)]]$manuscript <<- grepl("/manuscript figures/",destination,fixed=TRUE)
  }
  capture <- function(plot, destination, width, height, stem) {
    # Resolve all aesthetics/stats now; renderer needs neither raw data nor fits.
    if (is.null(plot)) return(invisible(NULL))
    if(!is.null(manuscript_root) && inherits(plot,"ggplot") &&
       basename(destination)=="platinum__all__incl")
      manuscript_items[[stem]] <<- list(plot=plot)
    specs <- figure_compilation_specs(stem)
    specs <- Filter(function(spec) is.null(spec$endpoint) ||
      startsWith(basename(destination),paste0(spec$endpoint,"__")),specs)
    for(spec in specs) {
      # Lab artifacts have an extra landmark directory; bundle by arm/identity
      # and logical key instead of their legacy folder depth.
      key <- paste(sub("/by_figure/.*$","",destination),basename(destination),spec$key,sep="|")
      bundles[[key]] <<- c(bundles[[key]],list(list(plot=plot,destination=destination,stem=stem,spec=spec)))
    }
    if(!length(specs) || any(vapply(specs,function(spec) isTRUE(spec$keep_individuals),logical(1))))
      publish(plot,destination,width,height,stem,landmarks=if(grepl("landmark[0-9]+",stem))
      sub("^.*landmark([0-9]+).*$","\\1",stem) else "")
    invisible(NULL)
  }
  manuscript_capture <- function(plot,stem) {
    if(!is.null(manuscript_root) && inherits(plot,"ggplot"))
      manuscript_items[[stem]] <<- list(plot=plot)
    invisible(NULL)
  }
  table_capture <- function(paths) {
    for(path in paths) {
      target <- figure_public_path(path)
      if(!identical(path,target)) {
        dir.create(dirname(target),recursive=TRUE,showWarnings=FALSE)
        if(!file.copy(path,target,overwrite=TRUE) ||
           !identical(unname(tools::md5sum(path)),unname(tools::md5sum(target)))) stop("Cannot publish figure table: ",path)
        # The old path is archived after a successful render, not removed here.
      }
      tables <<- unique(c(tables,target))
    }
  }
  old <- options(compass.figure_capture = capture,
                 compass.manuscript_capture = manuscript_capture,
                 compass.figure_table_capture = table_capture,
                 device = function(...) grDevices::pdf(file = NULL, ...))
  on.exit(options(old), add = TRUE)
  # Build all nested gtables on the raster renderer too; PDF-metric sizing caused
  # the clipped labels and touching legend keys in the downloaded cluster PNGs.
  measure <- tempfile(".build-device-",tmpdir=generation,fileext=".png")
  figure_measure_device(measure,16,10)
  on.exit({grDevices::dev.off(); unlink(measure)},add=TRUE)
  build()
  for(items in bundles) {
    spec <- items[[1]]$spec
    if(length(spec$members)) {
      stems <- vapply(items,`[[`,character(1),"stem")
      missing <- setdiff(spec$members,stems)
      if(length(missing)) {
        message("Skipping incomplete combined figure ",spec$key,": missing ",paste(missing,collapse=", "))
        next
      }
      items <- items[match(spec$members,stems)]
    }
    page_size <- if(is.null(spec$page_size)) length(items) else spec$page_size
    pages <- split(seq_along(items),ceiling(seq_along(items)/page_size))
    for(page in seq_along(pages)) {
      selected <- items[pages[[page]]]
      compiled <- figure_combine(selected,spec,generation)
      destination <- figure_public_path(selected[[1]]$destination)
      identity <- basename(selected[[1]]$destination)
      key <- paste0(spec$key,if(length(pages)>1) paste0("_page",page))
      # Lab name is already the folder; omit it from the compiled filename.
      if(startsWith(key,"km_quintile_")) key <- "km_extremes"
      destination <- file.path(dirname(destination),paste0(key,"__",identity))
      landmarks <- unique(unlist(lapply(selected,function(x) if(grepl("landmark[0-9]+",x$stem))
        sub("^.*landmark([0-9]+).*$","\\1",x$stem) else character())))
      if(!length(landmarks)) landmarks <- unique(unlist(lapply(selected,function(x) {
        if(inherits(x$plot,"ggplot") && "landmark" %in% names(x$plot$data))
          gsub("[^0-9]","",as.character(x$plot$data$landmark)) else character()
      })))
      title <- spec$title
      if(startsWith(spec$key,"km_quintile_")) title <- paste(if(grepl("_psa$",spec$key)) "PSA" else
        tools::toTitleCase(gsub("_"," ",sub("^km_quintile_","",spec$key))),"extremes: time to platinum")
      publish(compiled,destination,spec$width,spec$height,key,title,paste(landmarks[nzchar(landmarks)],collapse=","))
    }
  }
  if(!is.null(manuscript_root)) {
    manuscripts <- manuscript_build(manuscript_items,tables,manuscript_root)
    for(item in manuscripts) {
      destination <- file.path(manuscript_root,item$spec$key)
      dir.create(manuscript_root,recursive=TRUE,showWarnings=FALSE)
      legend_path <- paste0(destination,".md")
      writeLines(c(paste0("# ",gsub("_"," ",item$spec$key)),"",
        sprintf("Source-rendered manuscript figure: %.1f-inch width; 600-dpi PNG and vector PDF.",
          item$spec$width),"",item$legend),legend_path)
      table_capture(legend_path)
      if(is.data.frame(item$data)) {
        data_path <- paste0(destination,".csv")
        readr::write_csv(item$data,data_path); table_capture(data_path)
      }
      publish(item$plot,destination,item$spec$width,item$spec$height,item$spec$key)
    }
    manuscript_write_captions(manuscript_root,
      vapply(manuscripts,function(item) item$spec$key,character(1)),table_capture)
  }
  paths <- vapply(scenes, `[[`, character(1), "path")
  m <- list(signature = signature, scenes = scenes, files = figure_file_identity(paths),
            tables = tables, table_files = figure_file_identity(tables))
  figure_atomic_rds(m, file.path(directory, "complete.rds"))
  m
}

figure_receipt_path <- function(scene, format) {
  # Scene objects live under COMPASS_FIGURE_DATA_ROOT, never beside exports.
  paste0(scene$path, ".", format, ".receipt.rds")
}

figure_flatten_export_layout <- function(fig_root) {
  moves <- list(); roots <- character()
  for (arm in c("ADT", "ARPI")) {
    base <- file.path(fig_root, arm, "by_figure")
    for (tier in c("main", "supplements")) {
      old <- file.path(base, tier)
      if (!dir.exists(old)) next
      roots <- c(roots, old)
      paths <- list.files(old, recursive = TRUE, full.names = TRUE, all.files = TRUE, no.. = TRUE)
      for (path in paths) moves[[length(moves)+1L]] <- c(from = path,
        to = file.path(base, substring(path, nchar(old)+2L)))
    }
    # Notebook 07 previously wrote standalone cohort-comparison panels here.
    old <- file.path(fig_root, arm, "supplements")
    if (dir.exists(old)) {
      roots <- c(roots, old)
      paths <- list.files(old, recursive = TRUE, full.names = TRUE, all.files = TRUE, no.. = TRUE)
      for (path in paths) moves[[length(moves)+1L]] <- c(from = path,
        to = file.path(base, "cohort_comparison", substring(path, nchar(old)+2L)))
    }
  }
  if (!length(roots)) return(invisible(0L))
  # Preflight all collisions before moving anything. Different images/tables
  # with the same destination need an explicit choice, not a silent overwrite.
  if (length(moves)) {
    from <- vapply(moves, `[[`, character(1), "from")
    to <- vapply(moves, `[[`, character(1), "to")
    if (any(grepl("\\.rds$", from, ignore.case = TRUE)))
      stop("Unrecognized RDS files remain in the old figure layout; move them to data before flattening.")
    for (destination in unique(to)) {
      candidates <- c(from[to == destination], if (file.exists(destination)) destination)
      hashes <- unname(tools::md5sum(candidates))
      if (anyNA(hashes) || length(unique(hashes)) != 1L)
        stop("Figure layout collision; no files moved. Resolve differing copies for: ", destination)
    }
    for (i in seq_along(from)) {
      dir.create(dirname(to[i]), recursive = TRUE, showWarnings = FALSE)
      if (file.exists(to[i])) {
        if (unlink(from[i]) != 0L) stop("Could not remove identical legacy copy: ", from[i])
      } else if (!file.rename(from[i], to[i])) stop("Could not move figure: ", from[i])
    }
  }
  for (root in roots) {
    directories <- list.dirs(root, recursive = TRUE, full.names = TRUE)
    for (directory in directories[order(nchar(directories), decreasing = TRUE)])
      if (!length(list.files(directory, all.files = TRUE, no.. = TRUE))) unlink(directory, recursive = TRUE)
  }
  if (length(moves)) message("Flattened ", length(moves),
    " figure/table files into by_figure families; removed empty main/supplements directories. Identical duplicate copies were consolidated.")
  invisible(length(moves))
}

figure_archive_legacy_receipts <- function(fig_root, cache_root) {
  if (!dir.exists(fig_root)) return(invisible(0L))
  paths <- list.files(fig_root, pattern = "\\.cache\\.rds$", recursive = TRUE, full.names = TRUE)
  moved <- 0L
  for (path in paths) {
    # Only migrate this pipeline's known receipt schema, not arbitrary user RDS.
    receipt <- tryCatch(readRDS(path), error = function(e) NULL)
    if (!is.list(receipt) || !is.list(receipt$identity) || is.null(receipt$identity$signature) ||
        !is.data.frame(receipt$file) || !all(c("path", "size", "mtime") %in% names(receipt$file))) next
    archive <- file.path(cache_root, "legacy_render_receipts",
      paste0(figure_object_hash(list(figure_absolute_path(path), unname(tools::md5sum(path)))), ".rds"))
    dir.create(dirname(archive), recursive = TRUE, showWarnings = FALSE)
    # Copy/verify before removal: figure and data roots can be different filesystems.
    if (!file.exists(archive) && !file.copy(path, archive, copy.date = TRUE))
      stop("Could not move legacy render receipt into data cache: ", path)
    if (!identical(unname(tools::md5sum(path)), unname(tools::md5sum(archive))))
      stop("Legacy receipt copy failed verification; original retained: ", path)
    if (unlink(path) != 0L) stop("Receipt copied to data cache but could not remove original: ", path)
    moved <- moved + 1L
  }
  if (moved) message("Moved ", moved, " legacy render receipts from figure exports into ",
                     file.path(cache_root, "legacy_render_receipts"), " (recoverable there).")
  invisible(moved)
}

render_figure_scene <- function(scene, signature, dpi, pdf, overwrite = FALSE) {
  if(isTRUE(scene$manuscript)) { dpi <- max(600,dpi); pdf <- TRUE }
  grob <- NULL
  rendered <- 0L
  for (format in c("png", if (pdf) "pdf")) {
    path <- paste0(scene$destination, ".", format)
    receipt <- figure_receipt_path(scene, format)
    identity <- list(signature = signature, dpi = if (format == "png") dpi else NULL,
                     width = scene$width, height = scene$height)
    fresh <- if (file.exists(receipt)) tryCatch({
      r <- readRDS(receipt)
      identical(r$identity, identity) && identical(r$file, figure_file_identity(path))
    }, error = function(e) FALSE) else FALSE
    if (!overwrite && fresh && figure_file_complete(path)) next
    if (is.null(grob)) grob <- readRDS(scene$path)
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    temporary <- tempfile(".figure-render-", tmpdir = dirname(path), fileext = paste0(".", format))
    tryCatch({
      device <- if (format == "pdf") grDevices::cairo_pdf else if (HAS_RAGG) ragg::agg_png else "png"
      ggplot2::ggsave(temporary, grob, width = scene$width, height = scene$height,
                     dpi = dpi, bg = "white", device = device)
      if (!figure_file_complete(temporary) || !file.rename(temporary, path)) stop("Cannot publish ", path)
      figure_atomic_rds(list(identity = identity, file = figure_file_identity(path)), receipt)
    }, finally = unlink(temporary))
    rendered <- rendered + 1L
  }
  list(stem = scene$stem, rendered = rendered)
}

figure_scene_complete <- function(scene, signature, dpi, pdf) {
  if(isTRUE(scene$manuscript)) { dpi <- max(600,dpi); pdf <- TRUE }
  all(vapply(c("png", if (pdf) "pdf"), function(format) {
    path <- paste0(scene$destination, ".", format)
    receipt <- figure_receipt_path(scene, format)
    if (!file.exists(receipt) || !figure_file_complete(path)) return(FALSE)
    identity <- list(signature = signature, dpi = if (format == "png") dpi else NULL,
                     width = scene$width, height = scene$height)
    tryCatch({
      r <- readRDS(receipt)
      identical(r$identity, identity) && identical(r$file, figure_file_identity(path))
    }, error = function(e) FALSE)
  }, logical(1)))
}

figure_object_hash <- function(value) {
  path <- tempfile()
  on.exit(unlink(path))
  saveRDS(value, path, compress = FALSE, version = 2)
  unname(tools::md5sum(path))
}

figure_absolute_path <- function(path) {
  # normalizePath leaves missing destinations unresolved; their identity would
  # otherwise change after the first render creates them (notably /var -> /private/var).
  if (file.exists(path)) return(normalizePath(path, mustWork = TRUE))
  file.path(figure_absolute_path(dirname(path)), basename(path))
}

figure_parallel <- function(items, fun, workers) {
  if (!length(items)) return(list())
  safe <- function(item) tryCatch(fun(item), error = function(e) list(error = conditionMessage(e)))
  # macOS graphics/font libraries are not fork-safe after gtable preparation.
  # Keep Mac/Windows sequential; Linux cluster workers retain shared caches.
  result <- if (workers > 1L && .Platform$OS.type != "windows" && Sys.info()[["sysname"]] != "Darwin") {
    parallel::mclapply(items, safe, mc.cores = min(workers, length(items)), mc.preschedule = FALSE)
  } else lapply(items, safe)
  lapply(result, function(x) if (is.null(x) || inherits(x, "try-error"))
    list(error = "worker exited without a result") else x)
}

figure_notebook_manifest <- function(config, check_sources = TRUE) {
  guidance <- "Run 04_prep_figure_data.ipynb with the requested settings, then rerun 05_figures.Rmd."
  fail <- function(detail) stop(detail, "\n", guidance, call. = FALSE)
  path <- file.path(config$cache_root, "manifest.json")
  if (!file.exists(path)) fail(paste("Missing notebook-prepared manifest:", path))
  manifest <- tryCatch(jsonlite::read_json(path, simplifyVector = FALSE),
    error = function(e) fail(paste("Cannot read prepared manifest:", conditionMessage(e))))
  if (!identical(manifest$schema_version, 2L)) fail("Prepared manifest has an unsupported schema.")
  prepared <- manifest$config
  values <- function(x) as.character(unlist(x, use.names = FALSE))
  same <- function(a, b) identical(sort(values(a)), sort(values(b)))
  same_path <- function(a, b) length(a) == 1L && length(b) == 1L &&
    identical(figure_absolute_path(a), figure_absolute_path(b))
  for (key in c("data_root", "cache_root"))
    if (!same_path(prepared[[key]], config[[key]])) fail(paste("Prepared path does not match:", key))
  federated_only <- identical(config$scope, "federated")
  if (!federated_only) {
    if (identical(prepared$scope, "federated")) fail("Only federated data were prepared.")
    for (key in c("cohorts", "endpoints"))
      if (!all(values(config[[key]]) %in% values(prepared[[key]])))
        fail(paste("Notebook did not prepare requested", key))
    for (key in c("labs", "gam", "adt_intent", "metastatic", "metastatic_extra",
                  "forest_cohorts", "forest_landmark"))
      if (!same(prepared[[key]], config[[key]])) fail(paste("Prepared setting does not match:", key))
    if (!same_path(prepared$classifier_path, config$classifier_path))
      fail("Prepared classifier path does not match.")
    if (isTRUE(config$metastatic)) for (key in c("intent", "stage", "llm", "icd"))
      if (!same_path(prepared$metastatic_sources[[key]], config$metastatic_sources[[key]]))
        fail(paste("Prepared metastatic source does not match:", key))
  }
  if (isTRUE(config$federated)) {
    if (!isTRUE(prepared$federated) || !same_path(prepared$federated_path, config$federated_path))
      fail("Requested federated results were not prepared.")
    registered_inputs <- c(file.path("nvflare_within_site_cox_univariate",
      c("cox_within_site_all_sites_cohort.csv", "cox_within_site_all_sites_results.csv")),
      file.path("federated_xgboost",c("xgboost_federated_metrics_adt.csv","xgboost_federated_importance_adt.csv")))
    for (filename in registered_inputs) {
      site_path <- file.path(dirname(config$federated_path), filename)
      if (!any(vapply(manifest$federated_sources, function(item)
        identical(figure_absolute_path(item$path), figure_absolute_path(site_path)), logical(1))))
        fail(paste("Federated input was not registered by the notebook:", filename))
    }
  }
  # R only checks metadata. It never launches Python or reconstructs its tables.
  # Render-only deliberately uses the prepared snapshot with sources offline.
  fingerprints <- if (federated_only) {
    if (check_sources) manifest$federated_sources else list()
  } else c(manifest$outputs, if (check_sources) manifest$source_fingerprints)
  for (item in fingerprints) {
    exists <- file.exists(item$path)
    if (isTRUE(item$missing)) {
      if (exists) fail(paste("A previously missing source is now present:", item$path))
    } else {
      info <- file.info(item$path)
      if (!exists || is.na(info$size) || info$size != item$bytes ||
          abs(as.numeric(info$mtime) - item$mtime_ns / 1e9) > 1e-6)
        fail(paste("Prepared data/source is missing or changed:", item$path))
    }
  }
  manifest
}

run_cached_figure_workflow <- function(config, pipeline_path, stage = "all",
    prepare_workers = 1L, render_workers = 2L, dpi = 200, pdf = FALSE,
    prepare_overwrite = FALSE, render_overwrite = FALSE,
    metastatic_config = NULL, forest_config = NULL, federated_config = NULL) {
  stage <- match.arg(stage, c("all", "prepare", "render"))
  started <- proc.time()[["elapsed"]]
  for (name in c("data_root", "cache_root", "fig_root"))
    config[[name]] <- figure_absolute_path(config[[name]])
  if (identical(config$cache_root, config$fig_root) ||
      startsWith(config$cache_root, paste0(config$fig_root, "/")) ||
      startsWith(config$fig_root, paste0(config$cache_root, "/")))
    stop("COMPASS_FIGURE_DATA_ROOT and COMPASS_FIG_ROOT must be separate, non-nested directories; keep caches in data.")
  dir.create(config$cache_root, recursive = TRUE, showWarnings = FALSE)
  manifest <- figure_notebook_manifest(config, check_sources = stage != "render")
  figure_archive_legacy_receipts(config$fig_root, config$cache_root)
  figure_flatten_export_layout(config$fig_root)
  old <- options(compass.figure_data_manifest = manifest)
  on.exit(options(old), add = TRUE)
  runtime <- list(R = as.character(getRversion()), packages = vapply(
    c("ggplot2", "survival", "survminer", "mgcv", "dplyr", "tidyr", "readr", "ggrepel", "scales", "stringr"),
    function(x) as.character(utils::packageVersion(x)), character(1)),
    code = unname(tools::md5sum(list.files(dirname(pipeline_path), pattern = "\\.R$", full.names = TRUE))),
    fig_root = config$fig_root, combined_km_landmark = figure_overview_km_landmark())
  jobs <- list()
  if (!identical(config$scope, "federated")) {
    for (cohort in config$cohorts) for (endpoint in config$endpoints) {
      name <- paste(cohort, endpoint, sep = "__")
      signature <- figure_object_hash(list(manifest$cells[[name]], runtime))
      jobs[[name]] <- list(name = name, cohort = cohort, endpoint = endpoint, signature = signature,
        directory = file.path(config$cache_root, "scenes", name, signature))
    }
  }
  federated_enabled <- isTRUE(config$federated) && (identical(config$scope, "federated") ||
    ("adt" %in% config$cohorts && "platinum" %in% config$endpoints))
  if (federated_enabled) {
    signature <- figure_object_hash(list(manifest$federated, runtime))
    jobs$federated <- list(name = "federated", signature = signature,
      directory = file.path(config$cache_root, "scenes", "federated", signature))
  }
  if (!identical(config$scope, "federated") && "adt" %in% config$cohorts && "platinum" %in% config$endpoints) {
    # Figure 6 also consumes the canonical ADT/platinum cohort-forest inputs,
    # so changes to either the overview tables or those model results must
    # invalidate this job's prepared manuscript scene.
    signature <- figure_object_hash(list(manifest$cohort_overview$key,
      manifest$cells[["adt__platinum"]],runtime))
    jobs$cohort_overview <- list(name = "cohort_overview", signature = signature,
      directory = file.path(config$cache_root, "scenes", "cohort_overview", signature))
  }
  build_job <- function(job) {
    if (!is.null(manifest$errors[[job$name]])) stop(manifest$errors[[job$name]])
    # Optional missing federation inputs produce a reported skip, not an empty
    # completion marker. A later delivery is picked up by source fingerprints.
    if (job$name == "cohort_overview") {
      source(file.path(dirname(pipeline_path), "figure_supplements.R"), local = TRUE)
      build <- function() render_cohort_overview(manifest, config, forest_config)
    } else if (job$name == "federated") {
      inputs <- federated_config$results
      if (stage != "render" && any(!file.exists(inputs))) {
        warning("Federated supplement skipped; missing: ", paste(inputs[!file.exists(inputs)], collapse = ", "))
        return(list(skipped = TRUE))
      }
      build <- function() render_federated_supplement(config$data_root, config$fig_root,
        federated_config$results, dpi = dpi, overwrite = TRUE)
    } else {
      build <- function() {
        message("Preparing figure set: ", job$name)
        generate_figures(job$cohort, config$data_root, config$fig_root,
          endpoint = job$endpoint, llm_annotations_path = config$classifier_path,
          plot_non_androgen_lab_figures = length(config$labs) > 2L,
          plot_gam_trajectories = config$gam, plot_adt_intent_supplement = config$adt_intent,
          overwrite = TRUE, save_pdf = FALSE, show = FALSE,
          metastatic_supplement = if (config$metastatic) metastatic_config else NULL,
          cohort_forest_config = forest_config,
          progress = function(event, detail) {
            if (event %in% c("stage", "panel_done")) message(job$name, ": ", detail)
          })
      }
    }
    if (stage == "render") {
      m <- figure_scene_manifest(job$directory, job$signature)
      if (is.null(m)) stop("Missing/stale prepared figures for ", job$name, "; run all or prepare.")
      return(m)
    }
    prepare_figure_scenes(job$directory, job$signature, build, force = prepare_overwrite,
      manuscript_root=if(job$name %in% c("adt__platinum","cohort_overview","federated"))
        file.path(config$fig_root,"ADT","manuscript figures") else NULL)
  }
  # Only pending jobs warm tables. A resumed all-mode knit reads no patient
  # Parquet at all when every scene snapshot is current.
  if (stage != "render") {
    pending <- Filter(function(job) !job$name %in% c("federated", "cohort_overview") &&
      is.null(manifest$errors[[job$name]]) &&
      (prepare_overwrite || is.null(figure_scene_manifest(job$directory, job$signature))), jobs)
    for (arm in unique(vapply(pending, function(job) cohort_arm(job$cohort), character(1)))) {
      source_path <- manifest$arms[[arm]]$source
      invisible(figure_prepared_table(source_path, "patients"))
      if (any(vapply(pending, function(job) cohort_arm(job$cohort) == arm && job$endpoint == "platinum", logical(1))))
        for (table in c("canonical", "patient_bins", "coverage_patient", "coverage_bins"))
          invisible(figure_prepared_table(source_path, table))
    }
    invisible(gc())
  }
  prepared <- figure_parallel(jobs, build_job, prepare_workers)
  names(prepared) <- names(jobs)
  failures <- unlist(lapply(names(prepared), function(name) {
    if (!is.null(prepared[[name]]$error)) paste(name, prepared[[name]]$error, sep = ": ")
  }))
  clear_read_cache()
  invisible(gc())
  render_results <- list()
  if (stage != "prepare") {
    tasks <- unlist(lapply(prepared, function(m) {
      if (is.null(m$scenes)) return(list())
      lapply(m$scenes, function(scene) list(scene = scene, signature = m$signature))
    }), recursive = FALSE)
    message("Rendering/checking ", length(tasks), " panels with ", render_workers, " workers ...")
    completed <- if (render_overwrite) rep(FALSE, length(tasks)) else
      vapply(tasks, function(task) figure_scene_complete(task$scene, task$signature, dpi, pdf), logical(1))
    render_results <- lapply(tasks, function(task) list(stem = task$scene$stem, rendered = 0L))
    message(sum(completed), " panels already current; ", sum(!completed), " need rendering.")
    render_results[!completed] <- figure_parallel(tasks[!completed], function(task) {
      result <- render_figure_scene(task$scene, task$signature, dpi, pdf, render_overwrite)
      message(if (result$rendered) "Saved " else "Reused ", result$stem)
      result
    }, render_workers)
    failures <- c(failures, unlist(lapply(render_results, `[[`, "error")))
  }
  message(sprintf("Figure workflow finished in %.1fs (%s).", proc.time()[["elapsed"]] - started, stage))
  if (length(failures)) stop("Figure workflow failures:\n", paste(failures, collapse = "\n"))
  if(stage != "prepare") {
    figure_archive_old_exports(config,prepared)
    figure_write_catalog(config,prepared)
  }
  invisible(list(prepared = prepared, rendered = render_results))
}

# ============================================================================
# ---- reformat downloaded PNG exports (CLI) ---------------------------------
# ============================================================================
# Reassemble downloaded PNG exports without requiring clinical inputs or fits.
# Rscript figure_workflow.R <downloaded-ADT-dir> <new-parent>/ADT [0|90|180]
# Always writes a separate tree. The input PNGs and their plotted data are unchanged.

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
  reformat_downloaded_figures(args[1],args[2],if(length(args)>2L) as.integer(args[3]) else 0L)
}
