# COMPASS figure workflow: presentation helpers, federated and manuscript
# figures, the cached prepare/render bridge used by 07_figures/all.R, and the
# downloaded-PNG reformatter. Source after COMPASS_generate_figures_pipeline.R.
# Standalone CLI: Rscript figure_workflow.R <downloaded-ADT-dir> <new-parent>/ADT [0|90|180]

# ============================================================================
# ---- publication -----------------------------------------------------------
# ============================================================================
# Presentation-only compilation. Statistical inputs and cached RDS stay in data.
figure_slug <- function(x) tolower(gsub("^_|_$", "", gsub("[^A-Za-z0-9]+", "_", x)))

# Tables, legends, manifests and the HTML index sit in a "data" folder beside
# the figures they describe, so each export folder lists only PNG/PDF figures.
FIGURE_SIDECAR_DIR <- "data"
figure_sidecar_path <- function(path) {
  if (basename(dirname(path)) == FIGURE_SIDECAR_DIR) return(path)
  file.path(dirname(path), FIGURE_SIDECAR_DIR, basename(path))
}

# Export sections under each arm root, numbered so file browsers and the HTML
# index list them in reading order. Univariate and multivariate results are
# split into DFCI-only ("internal") and pooled-site ("federated") halves;
# models refit without DFCI count as multivariate, and only their scoring at
# DFCI is a transfer. Every non-canonical cohort is a cohort sensitivity.
FIGURE_SECTIONS <- c(cohort = "01_cohort", llm = "02_llm",
  univariate_internal = "03_univariate/internal", univariate_federated = "03_univariate/federated",
  multivariate_internal = "04_multivariate/internal", multivariate_federated = "04_multivariate/federated",
  transfer = "05_transfer", cohort_sensitivity = "06_cohort_sensitivity")
FIGURE_CANONICAL_COHORT <- "all__incl"
# Pre-restructure top-level folders, archived once their job re-renders.
FIGURE_RETIRED_FOLDERS <- c("cohort", "labels", "associations", "prediction",
  "trajectories", "federated", "compiled", "manuscript figures")

figure_public_path <- function(path) {
  if (!grepl("/by_figure/", path, fixed = TRUE)) return(path)
  root <- sub("/by_figure/.*$", "", path)
  parts <- strsplit(sub("^.*?/by_figure/", "", path), "/", fixed = TRUE)[[1]]
  group <- parts[1]; leaf <- tail(parts, 1)
  artifact <- paste(parts[seq.int(2, length(parts)-1)], collapse = "_")
  section <- switch(group, figure1 = , androgen_pre_adt_coverage = , figure1s_analysis_sets = "cohort",
    figure2v3_llm = , km_llm = "llm",
    cohort_comparison = , supplement_adt_intent = , metastatic_labels = "cohort_sensitivity",
    # Co-mutation KM curves are somatic univariate associations; the rest of
    # Figure 4 is multivariable models and their sensitivity analyses.
    figure4 = if (grepl("^h_comutation", artifact)) "univariate_internal" else "multivariate_internal",
    federated = "univariate_federated", "univariate_internal")
  topic <- FIGURE_SECTIONS[[section]]
  # Univariate KM curves and lab trajectories have their own folders; per-lab
  # panels are nested by lab beneath them.
  if (group == "labs") {
    topic <- file.path(topic, if (startsWith(parts[4], "km_")) "km" else "trajectories", figure_slug(parts[3]))
    artifact <- paste(parts[4:(length(parts)-1)], collapse = "_")
  }
  cohort <- paste(strsplit(leaf, "__", fixed = TRUE)[[1]][-1], collapse = "__")
  if (nzchar(cohort) && cohort != FIGURE_CANONICAL_COHORT && section != "cohort_sensitivity")
    topic <- file.path(FIGURE_SECTIONS[["cohort_sensitivity"]], cohort, topic)
  artifact <- sub("^[a-h]_", "", artifact)
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
  if (section == "univariate_internal" && group != "labs") {
    if (group == "KM_curves" || grepl("(^|_)km(_|$)", artifact) || startsWith(artifact, "comutation_"))
      topic <- file.path(topic, "km")
    else if (group %in% c("gam_trajectories", "androgen_trajectories", "androgen_distributions"))
      topic <- file.path(topic, "trajectories")
  }
  file.path(root, topic, paste0(artifact, "__", leaf))
}

# Federated outputs are written by stem rather than through by_figure.
figure_federated_section <- function(stem) {
  section <- if (grepl("^transfer_", stem)) "transfer" else
    if (grepl("^(elasticnet|xgboost|deephit|checklist)_", stem)) "multivariate_federated" else
    if (grepl("^(site_incidence|federated_cohort_km)", stem)) "cohort" else "univariate_federated"
  FIGURE_SECTIONS[[section]]
}
# Pooled KM curves and trajectories mirror the local km/<lab>/ and trajectories/<lab>/ folders.
federated_univariate_folder <- function(stem) {
  if (grepl("^transfer_km_", stem)) return("km")
  kind <- if (grepl("^federated_km_", stem)) "km" else if (grepl("^federated_trajectory_", stem)) "trajectories" else
    return(character())
  lab <- sub("^federated_(km_quintile|trajectory)_(.+?)(_landmark[0-9]+)?(_by_site)?$", "\\2", stem, perl = TRUE)
  if (lab == "summary") kind else file.path(kind, lab)
}
federated_output_path <- function(arm_root, stem, extension = "") {
  do.call(file.path, as.list(c(arm_root, figure_federated_section(stem), federated_univariate_folder(stem),
    paste0(stem, "__platinum", extension))))
}

# Manuscript figures sit in a "manuscript" folder of the section they summarize.
# A key may carry a "__<variant>" suffix (alternative versions of one figure);
# variants share their base key's section.
MANUSCRIPT_SECTIONS <- c(`01_cohort_overview` = "cohort", `02_llm_nepc_avpc` = "llm",
  `03_dfci_univariable` = "univariate_internal", `04_multivariable_labs` = "multivariate_internal",
  `06_dfci_cohort_sensitivity` = "cohort_sensitivity",
  `07_federated_incidence_associations` = "univariate_federated",
  `08_federated_multivariable` = "multivariate_federated",
  federated_03_univariable = "univariate_federated", federated_04_multivariable = "multivariate_federated",
  federated_05_transfer_no_dfci = "transfer")
manuscript_base_key <- function(key) sub("__.*$", "", key)
manuscript_folder <- function(arm_root, key) {
  base <- manuscript_base_key(key)
  if (!base %in% names(MANUSCRIPT_SECTIONS)) stop("No manuscript section defined for: ", key)
  file.path(arm_root, FIGURE_SECTIONS[[MANUSCRIPT_SECTIONS[[base]]]], "manuscript")
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
    title=paste0("DFCI lab associations, trajectories and extreme-decile survival (KM day ",landmark,")"),
    width=24, height=22, cols=6, row_heights=c(8,7,7),
    layout_matrix=rbind(c(1,1,2,2,3,3),c(4,4,4,5,5,5),c(6,6,6,7,7,7)),
    members=c(paste0("figure3_univariate_platinum_landmark",c(0,90,180)),
      "longitudinal_platinum_psa_log",paste0("km_quintile_psa_landmark",landmark),
      "longitudinal_platinum_testosterone_log",paste0("km_quintile_testosterone_landmark",landmark)),
    panel_tags=letters[1:7], keep_individuals=TRUE, endpoint="platinum",
    shared_caption=FALSE, shared_legend=FALSE)
}

# Supplementary non-androgen DFCI labs: each row pairs the trajectory (log1p for
# the right-skewed enzymes, raw otherwise) with the extreme-decile KM.
DFCI_SUPPLEMENT_LAB_SCALES <- c(LDH="log1p",Hemoglobin="raw",Albumin="raw",`Alkaline phosphatase`="log1p")
figure_dfci_supplement_labs_spec <- function(landmark=figure_overview_km_landmark(),
                                             scales=DFCI_SUPPLEMENT_LAB_SCALES) {
  stopifnot(landmark %in% c(0,90,180))
  slugs <- figure_slug(names(scales))
  n <- length(slugs)
  list(key=paste0("dfci_supplement_labs_lm",landmark),
    title=paste0("DFCI supplementary lab trajectories and extreme-decile survival (KM day ",landmark,")"),
    width=16, height=5.5*n, cols=2, layout_matrix=matrix(seq_len(2*n),ncol=2,byrow=TRUE),
    members=as.vector(rbind(paste0("longitudinal_platinum_",slugs,ifelse(scales=="log1p","_log","")),
      sprintf("km_quintile_%s_landmark%d",slugs,landmark))),
    panel_tags=letters[seq_len(2*n)], keep_individuals=TRUE, endpoint="platinum", multi_lab=TRUE,
    shared_caption=FALSE, shared_legend=FALSE)
}

# Figure 2 is drawn once per cohort; stems are figure2v3_<variant>_<panel>.
FIGURE2_COHORTS <- c(icd="ICD-10 prostate cancer", adt="ADT-exposed", lm0="ADT day-0 prediction cohort")
FIGURE2_PANELS <- c("confusion_matrix", "metric_bar", "subtype_landscape", "enrichment")
figure2v3_variant <- function(stem) {
  pattern <- sprintf("^figure2v3_(%s)_(%s)$", paste(names(FIGURE2_COHORTS), collapse="|"),
    paste(FIGURE2_PANELS, collapse="|"))
  ifelse(grepl(pattern, stem), sub(pattern, "\\1", stem), NA_character_)
}
figure2v3_panel <- function(stem) sub("^figure2v3_[^_]+_", "", stem)
figure2v3_stems <- function(variant) paste0("figure2v3_", variant, "_", FIGURE2_PANELS)

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
  llm_variant <- figure2v3_variant(stem)
  if (!is.na(llm_variant) && figure2v3_panel(stem) %in% c("confusion_matrix", "metric_bar"))
    return(list(key=paste0("classifier_validation_", llm_variant),
      title=paste("Subtype classifier validation:", FIGURE2_COHORTS[[llm_variant]]), width=12, height=6, cols=2,
      keep_individuals=TRUE, shared_caption=TRUE, shared_legend=FALSE))
  if (!is.na(llm_variant) && figure2v3_panel(stem) %in% c("subtype_landscape", "enrichment"))
    return(list(key=paste0("subtype_platinum_", llm_variant),
      title=paste("Subtype landscape and platinum enrichment:", FIGURE2_COHORTS[[llm_variant]]), width=15, height=7, cols=2,
      keep_individuals=TRUE, shared_caption=FALSE, shared_legend=FALSE))
  if (grepl("^figure4[acd]_", stem)) {
    key <- sub("_(auc|cindex)_[^_]+$", "", stem)
    title <- if (grepl("somatic", key)) "Somatic vs. labs sensitivity" else "Labs vs. age baseline"
    # One discrimination panel (mean AUC(t)) per figure; the C-index panel was retired.
    return(list(key=sub("^figure4[a-z]_", "", key), title=title, width=8.5, height=6.5, cols=1,
      shared_caption=TRUE, shared_legend=TRUE))
  }
  # Risk-score stratification KMs: one page per scheme x source x landmark,
  # and one per clinical stratifier for the within-level panels.
  if (grepl("^figure4f_riskstrat_", stem)) {
    key <- sub("^figure4f_(riskstrat_(test|cv_oof)_(full|gleason_somatic_(cox|xgboost)|gleason_(cox|xgboost)))_.*_landmark([0-9]+)$",
               "\\1_lm\\6", stem, perl=TRUE)
    return(list(key=key, title="Labs risk score vs. clinical stratifiers: KM", width=21, page_row_height=6.5,
      cols=3, page_size=9, shared_caption=TRUE, shared_legend=FALSE))
  }
  if (grepl("^figure4g_withinstrat_", stem)) {
    # Stratifier slugs are known; level slugs vary, so match the stratifier.
    strat <- sub("^(gleason|stage|tp53_rb1|tp53|pten|rb1|trio_burden|trio_combinations)_.*$", "\\1",
      sub("^figure4g_withinstrat_(test|cv_oof)_", "", stem), perl=TRUE)
    scheme <- sub("^figure4g_withinstrat_(test|cv_oof)_.*$", "\\1", stem)
    landmark <- sub("^.*_landmark([0-9]+)$", "\\1", stem)
    return(list(key=paste0("withinstrat_", scheme, "_", strat, "_lm", landmark),
      title="Labs risk within clinical strata: KM", width=21, page_row_height=6.5,
      cols=3, page_size=9, shared_caption=TRUE, shared_legend=FALSE))
  }
  if (grepl("^figure4b_importance_", stem)) {
    model <- sub("^.*_(cox|xgb)_landmark.*$", "\\1", stem)
    return(list(key=paste0("importance_",model), title=paste(if(model=="cox") "Elastic-Net Cox" else "XGBoost", "feature importance"),
      width=19, height=8, cols=3, shared_caption=TRUE, shared_legend=TRUE))
  }
  if (grepl("^km_quintile_", stem))
    return(list(key=sub("_landmark[0-9]+$", "", stem), title="Lab extremes: time to platinum",
      width=24, height=7.5, cols=3, keep_individuals=TRUE, shared_caption=TRUE, shared_legend=FALSE))
  if (grepl("^figure3b_sequencing_(snv|any)_.*_km_", stem)) {
    variant_mode <- sub("^figure3b_sequencing_(snv|any)_.*$", "\\1", stem)
    return(list(key=paste0("somatic_", variant_mode, "_carrier_km"),
      title=paste(if (variant_mode == "snv") "SNV" else "Any-alteration", "carrier status: time to platinum"),
      width=16, height=12, cols=2, page_size=4, shared_caption=TRUE, shared_legend=FALSE))
  }
  NULL
}

# A panel can belong to both an existing paired export and a complete figure.
figure_compilation_specs <- function(stem) {
  spec <- figure_compilation_spec(stem)
  specs <- if(is.null(spec)) list() else list(spec)
  llm_variant <- figure2v3_variant(stem)
  if(!is.na(llm_variant)) specs <- c(specs, list(list(
    key=paste0("figure2v3_llm_", llm_variant),
    title=paste("LLM annotation validation and subtype landscape:", FIGURE2_COHORTS[[llm_variant]]),
    width=16, height=12, cols=2, keep_individuals=TRUE,
    members=figure2v3_stems(llm_variant), panel_tags=letters[1:4],
    shared_caption=FALSE, shared_legend=FALSE)))
  for(compiled in list(figure_dfci_overview_spec(),figure_dfci_supplement_labs_spec()))
    if(stem %in% compiled$members) specs <- c(specs,list(compiled))
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
            sprintf("bottom vs top %d%%",round(100*LAB_EXTREME_FRACTION)),paste0("(day ",sub("^.*landmark","",stem),")"))
        }
      } else if (grepl("^figure4[fg]_",stem)) {
        # Compiled titles are dropped; the stratifier/level must stay on the panel.
        p <- p + ggplot2::labs(subtitle=paste(c(title,p$labels$subtitle),collapse=" \u2014 "))
      } else if (grepl("^figure3b_.*_km_",stem)) {
        title <- sub(" carrier status: time to platinum$", "", title)
      } else if (!is.na(figure2v3_variant(stem))) {
        title <- switch(figure2v3_panel(stem), confusion_matrix="Agreement with chart review",
          metric_bar="Classification metrics", subtype_landscape="Subtype composition",
          enrichment="Platinum enrichment")
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

# Move tables/notes left beside figures by earlier exports into the sidecar
# folder. When both copies exist the newer one wins. by_figure/ is left to the
# archiver, and manuscript captions already have their own folder.
figure_relocate_sidecars <- function(root) {
  paths <- list.files(root,recursive=TRUE,full.names=TRUE,pattern="\\.(csv|md|txt|html)$")
  rel <- substring(paths,nchar(root)+2)
  stray <- !grepl("(^|/)(by_figure|main|supplements)/",rel) &
    !basename(dirname(paths)) %in% c(FIGURE_SIDECAR_DIR,"captions")
  for(path in paths[stray]) {
    target <- figure_sidecar_path(path)
    dir.create(dirname(target),recursive=TRUE,showWarnings=FALSE)
    if(file.exists(target) && file.mtime(target) >= file.mtime(path)) {
      if(unlink(path)!=0) stop("Cannot remove superseded table: ",path)
    } else if(!file.rename(path,target)) stop("Cannot move table into ",FIGURE_SIDECAR_DIR,"/: ",path)
  }
  invisible(paths[stray])
}

figure_write_catalog <- function(config, prepared) {
  scenes <- unlist(lapply(prepared,function(m) m$scenes),recursive=FALSE)
  for (arm in c("ADT","ARPI")) {
    root <- file.path(config$fig_root,arm)
    if(!dir.exists(root)) next
    figure_relocate_sidecars(root)
    paths <- list.files(root,recursive=TRUE,full.names=TRUE,pattern="\\.(png|pdf|csv|md)$")
    paths <- paths[!grepl("/by_figure/|/main/|/supplements/",paths) & basename(paths)!="manifest.csv"]
    if(!length(paths)) next
    rel <- substring(paths,nchar(root)+2)
    # Index sections are the export sections, internal/federated kept apart.
    split_sections <- unique(dirname(FIGURE_SECTIONS[grepl("/",FIGURE_SECTIONS)]))
    records <- lapply(seq_along(paths),function(i) {
      stem <- tools::file_path_sans_ext(basename(paths[i]))
      segments <- strsplit(rel[i],"/",fixed=TRUE)[[1]]
      topic <- if(segments[1] %in% split_sections && length(segments)>2)
        paste(segments[1:2],collapse="/") else segments[1]
      fields <- strsplit(stem,"__",fixed=TRUE)[[1]]
      matches <- Filter(function(s) identical(s$destination,tools::file_path_sans_ext(paths[i])),scenes)
      title <- if(length(matches)) matches[[1]]$title else gsub("_"," ",fields[1])
      data.frame(path=rel[i],topic=topic,figure_id=fields[1],
        endpoint=if(length(fields)>1) fields[2] else "",cohort=if(length(fields)>2) paste(fields[-c(1,2)],collapse="__") else "",
        landmark=if(length(matches)) matches[[1]]$landmark else "",title=title,
        format=tools::file_ext(paths[i]),stringsAsFactors=FALSE)
    })
    registry <- do.call(rbind,records)
    # Keep metadata for previously rendered cohorts outside this run's scope.
    old_path <- file.path(root,FIGURE_SIDECAR_DIR,"manifest.csv")
    if(file.exists(old_path)) {
      previous <- readr::read_csv(old_path,show_col_types=FALSE,col_types=readr::cols(.default="c"))
      for(i in seq_len(nrow(registry))) {
        old <- match(registry$path[i],previous$path)
        if(!is.na(old) && !any(vapply(scenes,function(s) identical(paste0(s$destination,".",registry$format[i]),paths[i]),logical(1)))) {
          registry$title[i] <- previous$title[old]; registry$landmark[i] <- previous$landmark[old]
        }
      }
    }
    dir.create(dirname(old_path),showWarnings=FALSE)
    readr::write_csv(registry,old_path)
    cards <- vapply(which(registry$format=="png"),function(i) {
      row <- registry[i,]; stem <- tools::file_path_sans_ext(row$path)
      figure_stem <- sub(paste0("(^|/)",FIGURE_SIDECAR_DIR,"/([^/]+)$"),"\\1\\2",
        tools::file_path_sans_ext(registry$path))
      sidecars <- registry$path[figure_stem==stem & registry$format!="png"]
      # The index lives in <arm>/data/, one level below the figures it links.
      href <- function(p) paste0("../",figure_href(p))
      paste0('<article><a href="',href(row$path),'"><img loading="lazy" src="',href(row$path),
        '" alt="',figure_html_escape(row$title),'"></a><h3>',figure_html_escape(row$title),'</h3><p>',
        figure_html_escape(paste(row$topic,row$endpoint,row$cohort,row$landmark,sep=" · ")),'</p>',
        paste(vapply(c(row$path,sidecars),function(p) paste0('<a href="',href(p),'">',toupper(tools::file_ext(p)),'</a>'),character(1)),collapse=" · "),'</article>')
    },character(1))
    tables <- registry$path[registry$format %in% c("csv","md")]
    png_topics <- registry$topic[registry$format=="png"]
    sections <- unlist(lapply(unique(png_topics),function(topic) c(
      paste0('<h2 id="',figure_html_escape(topic),'">',figure_html_escape(tools::toTitleCase(gsub("/"," / ",gsub("_"," ",topic)))),'</h2><main>'),
      cards[png_topics==topic],'</main>')))
    html <- c('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
      paste0('<title>',arm,' figure index</title>'),
      '<style>body{font:16px system-ui;margin:2rem;background:#f7f8fa;color:#20242a}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}article{background:white;padding:1rem;border:1px solid #ddd;border-radius:8px}img{width:100%;height:260px;object-fit:contain}h3{font-size:1rem}p{color:#555}a{color:#165b91}</style></head><body>',
      paste0('<h1>',arm,' figures</h1><p>Click a figure for full resolution. <a href="manifest.csv">Download manifest</a>.</p>'),
      paste(vapply(unique(png_topics),function(topic) paste0('<a href="#',topic,'">',figure_html_escape(topic),'</a>'),character(1)),collapse=" · "),sections,
      '<h2>Tables</h2><ul>',vapply(tables,function(p) paste0('<li><a href="../',figure_href(p),'">',figure_html_escape(p),'</a></li>'),character(1)),
      '</ul></body></html>')
    writeLines(html,file.path(root,FIGURE_SIDECAR_DIR,"index.html"))
  }
}

figure_archive_old_exports <- function(config, prepared) {
  # Archive only successfully regenerated run identities. Unselected cohorts and
  # unknown files are untouched; copies are verified before removing originals.
  successful <- names(Filter(function(m) length(m$scenes)>0 && is.null(m$error),prepared))
  paths <- character()
  # A cohort job owns exports whose name ends in its <endpoint>__<leaf> identity.
  cohort_job_owns <- function(path,arm) {
    name <- sub("[.][^.]+$","",basename(path))
    any(vapply(successful,function(job) {
      bits <- strsplit(job,"__",fixed=TRUE)[[1]]
      if(length(bits)!=2 || toupper(cohort_arm(bits[1]))!=arm) return(FALSE)
      identity <- paste0(bits[2],"__",cohort_leaf_slug(bits[1]))
      name==identity || endsWith(name,paste0("__",identity))
    },logical(1)))
  }
  # Section compiled/ folders (compiled figures now sit beside their panels, or
  # only in manuscript/ when they duplicate a manuscript figure).
  section_compiled <- function(arm) {
    root <- file.path(config$fig_root,arm)
    dirs <- if(dir.exists(root)) list.dirs(root,recursive=TRUE) else character()
    dirs[basename(dirs)=="compiled" & dirname(dirs)!=root & !grepl("/by_figure(/|$)",dirs)]
  }
  # Folders whose KM curves and trajectories moved into km/ and trajectories/,
  # and the retired internal Gleason sensitivity (figure and manuscript Figure 5).
  normalize <- function(x) file.path(normalizePath(dirname(x),mustWork=FALSE),basename(x))
  current <- normalize(as.character(unlist(lapply(prepared[successful],function(m) vapply(m$scenes,`[[`,character(1),"destination")))))
  section_moved <- function(arm) {
    root <- file.path(config$fig_root,arm)
    dirs <- if(dir.exists(root)) list.dirs(root,recursive=TRUE) else character()
    rel <- substring(dirs,nchar(root)+2)
    list(moved=dirs[grepl("^(06_cohort_sensitivity/[^/]+/)?03_univariate/(internal(/trajectories/[^/]+)?|federated)$",rel)],
      gleason=dirs[grepl("^(06_cohort_sensitivity/[^/]+/)?04_multivariate/internal(/manuscript)?(/data|/captions)?$",rel)])
  }
  for(arm in c("ADT","ARPI")) {
    folders <- section_moved(arm)
    for(folder in folders$moved)
      for(path in list.files(folder,full.names=TRUE,pattern="\\.(png|pdf)$")) {
        if(normalize(tools::file_path_sans_ext(path)) %in% current) next
        keep <- if(endsWith(folder,"/03_univariate/federated")) "federated" %in% successful else cohort_job_owns(path,arm)
        if(keep) paths <- c(paths,path)
      }
    for(folder in folders$gleason)
      for(path in list.files(folder,full.names=TRUE,pattern="^(sensitivity_gleason_|05_gleason_sensitivity|figure5[.])")) {
        keep <- if(grepl("/manuscript(/|$)",folder)) "adt__platinum" %in% successful else cohort_job_owns(path,arm)
        if(keep) paths <- c(paths,path)
      }
    for(folder in section_compiled(arm))
      for(path in list.files(folder,recursive=TRUE,full.names=TRUE,pattern="\\.(png|pdf|csv|md)$")) {
        name <- tools::file_path_sans_ext(basename(path))
        keep <- if(grepl("^(cohort_forest_|event_incidence_|metastatic_label_overlap|cohort_km_cascade)",name))
          "cohort_overview" %in% successful else cohort_job_owns(path,arm)
        if(keep) paths <- c(paths,path)
      }
    root <- file.path(config$fig_root,arm,"by_figure")
    candidates <- if(dir.exists(root)) list.files(root,recursive=TRUE,full.names=TRUE,pattern="\\.(png|pdf|csv|md)$") else character()
    for(path in candidates) {
      rel <- substring(path,nchar(root)+2); group <- strsplit(rel,"/",fixed=TRUE)[[1]][1]
      special <- if(group=="federated") "federated" else if(group=="cohort_comparison") "cohort_overview" else NULL
      keep <- if(length(special)) special %in% successful else cohort_job_owns(path,arm)
      if(keep) paths <- c(paths,path)
    }
    # The pre-section layout (topic folders, top-level compiled/ and
    # "manuscript figures/") is retired per job once that job has re-rendered.
    for(folder in FIGURE_RETIRED_FOLDERS) {
      root <- file.path(config$fig_root,arm,folder)
      if(!dir.exists(root)) next
      for(path in list.files(root,recursive=TRUE,full.names=TRUE,pattern="\\.(png|pdf|csv|md|txt)$")) {
        segments <- strsplit(substring(path,nchar(file.path(config$fig_root,arm))+2),"/",fixed=TRUE)[[1]]
        if(segments[1]=="compiled") segments <- segments[-1]
        name <- tools::file_path_sans_ext(basename(path))
        job <- if(folder=="manuscript figures") {
          if(grepl("^(federated|0[78]_|figure[78]$)",name)) "federated" else
            if(grepl("^(06_|figure6$)",name)) "cohort_overview" else "adt__platinum"
        } else if(segments[1]=="federated") "federated" else
          if(segments[1]=="cohort" && grepl("^(cohort_forest_|event_incidence_|metastatic_label_overlap)",name)) "cohort_overview" else NULL
        keep <- if(length(job)) job %in% successful else cohort_job_owns(path,arm)
        if(keep) paths <- c(paths,path)
      }
    }
  }
  paths <- unique(paths)
  archive <- tempfile("export-",tmpdir=file.path(config$cache_root,"previous_exports"))
  for(path in paths) {
    target <- file.path(archive,substring(path,nchar(config$fig_root)+2))
    dir.create(dirname(target),recursive=TRUE,showWarnings=FALSE)
    if(!file.copy(path,target,copy.date=TRUE) || !identical(unname(tools::md5sum(path)),unname(tools::md5sum(target))))
      stop("Cannot verify archived export; original retained: ",path)
  }
  for(path in paths) if(unlink(path)!=0) stop("Cannot remove verified legacy export: ",path)
  for(arm in c("ADT","ARPI")) for(folder in c(file.path(config$fig_root,arm,c("by_figure",FIGURE_RETIRED_FOLDERS)),
                                                  section_compiled(arm))) {
    if(!dir.exists(folder)) next
    dirs <- list.dirs(folder,recursive=TRUE)
    for(dir in dirs[order(nchar(dirs),decreasing=TRUE)])
      if(!length(list.files(dir,all.files=TRUE,no..=TRUE))) unlink(dir,recursive=TRUE)
  }
  if(length(paths)) message("Archived ",length(paths)," superseded exports; recoverable at ",archive)
}

# ============================================================================
# ---- federated figures -----------------------------------------------------
# ============================================================================
# Federated PSA/testosterone forests, site counts, and elastic-net/XGBoost Figure 4 reports.
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
  short[grepl("^aPTT_", raw)] <- "aPTT"
  short[short == "Lactate dehydrogenase"] <- "LDH"
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
# (nvflare_within_site_cox_univariate -> nvflare_within_site_univariate_cox ->
# within_site_cox_univariate in rhino_data_v2). Accept any, preferring whichever
# actually exists.
FEDERATED_WITHIN_SITE_DIRS <- c("within_site_cox_univariate",
                                "nvflare_within_site_univariate_cox",
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
  targets <- arrange(distinct(across,landmark_days,n_patients_used,n_events_used),landmark_days)
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
  if(!length(matches)) {
    # A pooled-only site (e.g. MSK without a within-site export) leaves a
    # consistent excess over every exported site at each landmark.
    excess <- lapply(seq_len(nrow(targets)),function(i) {
      rows <- filter(cohorts,landmark_days==targets$landmark_days[i])
      if(!setequal(rows$site_name,sites) || nrow(rows)!=length(sites)) return(NULL)
      c(targets$n_patients_used[i]-sum(rows$n_patients),targets$n_events_used[i]-sum(rows$n_events))
    })
    if(all(vapply(excess,function(x) length(x)==2 && x[1]>0 && x[2]>=0,logical(1))))
      return(paste0("Federated*: pooled counts exceed ",paste(federated_site_label(sites),collapse=" + "),
        " by ",paste(vapply(seq_along(excess),function(i) sprintf("%s/%s (%gd)",
          scales::comma(excess[[i]][1]),scales::comma(excess[[i]][2]),targets$landmark_days[i]),
          character(1)),collapse=", "),
        "\npatients/events from site(s) without within-site results; membership inferred, not verified."))
  }
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

# Federated multivariable models (elastic-net Cox and XGBoost) share one export
# schema: held-out test metrics per landmark/config plus per-feature values.
FEDERATED_MULTIVARIABLE <- list(
  elastic_net=list(label="Elastic-net Cox",directory="federated_cox_multivariate",
    files=c(metrics="cox_federated_elasticnet_metrics_adt.csv",
      features="cox_federated_elasticnet_coefficients_adt.csv"),
    model="elastic_net_cox",value="coefficient",kind="cox",
    stems=c(metrics="elasticnet_performance",features="elasticnet_coefficients"),
    series=c(both="Elastic-Net Cox",baseline="Cox baseline (age)")),
  xgboost=list(label="XGBoost",directory="federated_xgboost",
    files=c(metrics="xgboost_federated_metrics_adt.csv",
      features="xgboost_federated_importance_adt.csv"),
    model="xgboost_cox",value="gain",kind="xgb",
    stems=c(metrics="xgboost_performance",features="xgboost_importance"),
    series=c(both="XGBoost Survival",baseline="XGBoost baseline (age)")))
# Local Figure 4 series colors; age baselines are the lighter twins.
FEDERATED_MODEL_COLORS <- c("Elastic-Net Cox"="#4C72B0","Cox baseline (age)"="#9DB3D6",
  "XGBoost Survival"="#B58900","XGBoost baseline (age)"="#E0CC8A",
  "Dynamic DeepHit"="#2A9D8F")
FEDERATED_FEATURE_STATS <- c("mean","min","max","last","delta","n_observations")

# The same bundles refit without Dana-Farber sit in "<directory>_no_DFCI";
# rhino_data_v2 marks both variants (marker: "w_DFCI" or "no_DFCI").
FEDERATED_MULTIVARIABLE_VARIANTS <- list(
  pooled=list(suffix="",marker="w_DFCI",stem="",label="",
    note="Trained on pooled Dana-Farber, Fred Hutch, Johns Hopkins and MSK data."),
  no_dfci=list(suffix="_no_DFCI",marker="no_DFCI",stem="_no_dfci",label=" (trained without Dana-Farber)",
    note="Refit without Dana-Farber; trained and tested on Fred Hutch, Johns Hopkins and MSK data."))
# Darker twins of the pooled full-model colors, used where both variants share a panel.
FEDERATED_NO_DFCI_MODEL_COLORS <- c("Elastic-Net Cox, no DFCI"="#1F3D73",
  "XGBoost Survival, no DFCI"="#6E5200",
  "Dynamic DeepHit, no DFCI"="#17675D")

# rhino_data_v2 exports beside the univariate CSV, for one variant: a directory per
# variant (federated_elastic_net_cox_w_DFCI/) or a Rhino package per landmark
# (<model>/lm_<days>/w_DFCI/), either with an optional adt/ subfolder. A family's
# per-landmark files load together; the duplicate-key checks catch overlaps.
# Mirrors federated_v2_export_files() in prepare_figure_data.py.
federated_v2_export_files <- function(federated_path, file, variant="pooled") {
  marker <- FEDERATED_MULTIVARIABLE_VARIANTS[[variant]]$marker
  root <- dirname(federated_path)
  sort(unique(Sys.glob(c(file.path(root,paste0("*_",marker),c(file,file.path("adt",file))),
    file.path(root,"*","lm_*",marker,c(file,file.path("adt",file)))))))
}

# Per-landmark files are read as text and typed together, so a column that is
# empty in one landmark's file still binds with the others.
read_federated_csvs <- function(paths) {
  if(length(paths)==1L) return(readr::read_csv(paths,show_col_types=FALSE))
  d <- dplyr::bind_rows(lapply(paths,readr::read_csv,col_types=readr::cols(.default="c")))
  readr::type_convert(d,col_types=readr::cols())
}

FEDERATED_DEEPHIT_METRIC_DIRS <- c("fast_federated_deephit", "federated_deephit")
FEDERATED_DEEPHIT_METRIC_FILE <- "deephit_federated_metrics_adt.csv"
# Candidate exports in order: each legacy file alone, then the v2 files together.
federated_deephit_metric_sources <- function(federated_path, variant="pooled") {
  legacy <- unlist(lapply(paste0(FEDERATED_DEEPHIT_METRIC_DIRS,FEDERATED_MULTIVARIABLE_VARIANTS[[variant]]$suffix),
    function(directory) file.path(dirname(federated_path),directory,
      c(FEDERATED_DEEPHIT_METRIC_FILE,file.path("adt",FEDERATED_DEEPHIT_METRIC_FILE)))),use.names=FALSE)
  v2 <- setdiff(federated_v2_export_files(federated_path,FEDERATED_DEEPHIT_METRIC_FILE,variant),legacy)
  c(as.list(legacy),if(length(v2)) list(v2))
}
federated_deephit_metric_paths <- function(federated_path, variant="pooled")
  unlist(federated_deephit_metric_sources(federated_path,variant),use.names=FALSE)

load_federated_dynamic_deephit <- function(federated_path,variant="pooled") {
  found <- FALSE
  for(paths in federated_deephit_metric_sources(federated_path,variant)) {
    if(!all(file.exists(paths))) next
    d <- read_federated_csvs(paths)
    path <- paste(paths,collapse="; ")
    # The older DeepHit exporter contains only the landmark arm.
    if(!"training_arm" %in% names(d)) next
    needed <- c("analysis_label","endpoint","landmark_days","config","training_arm",
      "test_mean_auc_t","n_test","n_events_test")
    if(length(missing <- setdiff(needed,names(d))))
      stop("Federated DeepHit metrics are missing: ",paste(missing,collapse=", ")," in ",path)
    d <- dplyr::filter(d,tolower(analysis_label)=="adt",tolower(endpoint)=="platinum",
      training_arm=="dynamic",config=="both",landmark_days %in% c(0,90,180))
    if(nrow(d)) { found <- TRUE; break }
  }
  if(!found) return(NULL)
  if(anyDuplicated(d$landmark_days)) stop("Duplicate dynamic DeepHit landmark rows in ",path)
  if(!is.numeric(d$test_mean_auc_t) && !all(is.na(d$test_mean_auc_t)))
    stop("Non-numeric dynamic DeepHit AUC in ",path)
  if(any(!is.na(d$test_mean_auc_t) & (!is.finite(d$test_mean_auc_t) |
      d$test_mean_auc_t<0 | d$test_mean_auc_t>1)) ||
      any(!is.finite(d$n_test) | !is.finite(d$n_events_test) |
        d$n_test<0 | d$n_events_test<0 | d$n_events_test>d$n_test))
    stop("Invalid dynamic DeepHit held-out metrics in ",path)
  if(!any(is.finite(d$test_mean_auc_t))) return(NULL)
  if(!"test_c_index" %in% names(d)) d$test_c_index <- NA_real_
  d <- d %>% tidyr::complete(landmark_days=c(0,90,180)) %>%
    dplyr::mutate(name=if(variant=="pooled") "Dynamic DeepHit" else "Dynamic DeepHit, no DFCI",
      landmark=landmark_days,auc=test_mean_auc_t,cindex=test_c_index,
      family="dynamic_deephit",input_audit_note=NA_character_,source_path=path)
  d
}

# The legacy "<directory><suffix>/<file>" when present, else the rhino_data_v2
# files (federated_v2_export_files), else the legacy path, reported as missing.
federated_multivariable_files <- function(federated_path, family, variant="pooled") {
  spec <- FEDERATED_MULTIVARIABLE[[family]]
  legacy <- file.path(dirname(federated_path),
    paste0(spec$directory,FEDERATED_MULTIVARIABLE_VARIANTS[[variant]]$suffix),spec$files)
  setNames(lapply(seq_along(legacy),function(i) {
    v2 <- federated_v2_export_files(federated_path,spec$files[[i]],variant)
    if(file.exists(legacy[[i]]) || !length(v2)) legacy[[i]] else v2
  }),names(spec$files))
}

# OMOP names contain internal double underscores, and missingness indicators
# follow the statistic ("__delta__missing" -> "delta missing"). Unknown
# suffixes fall back to the last "__" component.
federated_feature_stat <- function(feature) {
  pattern <- paste0("^.*__(",paste(FEDERATED_FEATURE_STATS,collapse="|"),")(__missing)?$")
  stat <- sub("__missing$"," missing",sub(pattern,"\\1\\2",feature))
  if_else(grepl(pattern,feature),stat,
    if_else(grepl("__",feature,fixed=TRUE),sub("^.*__","",feature),""))
}

load_federated_multivariable <- function(path, family=names(FEDERATED_MULTIVARIABLE),
                                         kind=c("metrics","features")) {
  family <- match.arg(family); kind <- match.arg(kind)
  spec <- FEDERATED_MULTIVARIABLE[[family]]
  what <- paste("Federated",spec$label,if(kind=="metrics") "metrics" else
    if(spec$kind=="cox") "coefficients" else "importance")
  if (!all(file.exists(path))) {
    warning(what," unavailable; missing: ",paste(path[!file.exists(path)],collapse=", "))
    return(NULL)
  }
  d <- read_federated_csvs(path)
  needed <- c("analysis_label","endpoint","landmark_days",if(kind=="metrics")
    c("model","cohort","config","test_mean_auc_t","n_test","n_events_test") else c("feature",spec$value))
  missing <- setdiff(needed,names(d))
  if(length(missing)) stop(what," input is missing: ",paste(missing,collapse=", "))
  # The C-index is kept in the sidecar when present but is no longer plotted.
  if(kind=="metrics" && !"test_c_index" %in% names(d)) d$test_c_index <- NA_real_
  d <- filter(d,tolower(analysis_label)=="adt",tolower(endpoint)=="platinum",landmark_days %in% c(0,90,180))
  if(kind=="metrics") {
    d <- filter(d,tolower(model)==spec$model,tolower(cohort)=="all",config %in% c("both","baseline"))
    keys <- c("landmark_days","config")
    for(column in c("test_c_index","test_mean_auc_t")) {
      if(!is.numeric(d[[column]]) && !all(is.na(d[[column]]))) stop("Non-numeric ",spec$label," metric: ",column)
      if(any(!is.na(d[[column]]) & (!is.finite(d[[column]]) | d[[column]]<0 | d[[column]]>1)))
        stop("Invalid ",spec$label," metric: ",column)
    }
    if(any(!is.na(d$n_test) & (!is.finite(d$n_test) | d$n_test<0)) ||
       any(!is.na(d$n_events_test) & (!is.finite(d$n_events_test) | d$n_events_test<0 | d$n_events_test>d$n_test),na.rm=TRUE))
      stop("Invalid ",spec$label," test counts")
  } else {
    keys <- c("landmark_days","feature")
    if(any(is.na(d$feature) | !nzchar(trimws(d$feature)))) stop("Missing ",spec$label," feature name")
    value <- d[[spec$value]]
    if(!is.numeric(value) || any(!is.finite(value)) || (spec$kind=="xgb" && any(value<0)))
      stop("Invalid ",spec$label," ",spec$value)
    # Keep original names and values in the CSV; labels are display-only.
    d <- mutate(d,lab_name=federated_lab_label(feature),feature_stat=federated_feature_stat(feature),
      identifier_feature=tolower(feature) %in% c("person_id","patient_id","dfci_mrn","mrn"))
    if(spec$kind=="cox") d$coef <- d$coefficient
  }
  if(anyDuplicated(d[keys])) stop("Duplicate landmark/",if(kind=="metrics") "config" else "feature",
    " rows in ",what," input")
  if(!nrow(d)) warning("No ADT/platinum rows in ",what," input: ",paste(path,collapse=", "))
  d
}

prepare_federated_performance <- function(metrics, family) {
  series <- FEDERATED_MULTIVARIABLE[[family]]$series
  metrics %>% complete(landmark_days=c(0,90,180),config=c("both","baseline")) %>%
    mutate(family=.env$family,name=factor(unname(series[config]),levels=unname(series)),
      landmark=landmark_days,auc=test_mean_auc_t,cindex=test_c_index)
}

federated_input_audit_note <- function(features, family) {
  spec <- FEDERATED_MULTIVARIABLE[[family]]
  noun <- if(spec$kind=="cox") "coefficients" else "importance"
  if(is.null(features)) return(paste0("Feature-input audit unavailable: ",spec$label," ",noun," file missing."))
  ids <- features[features$identifier_feature & features[[spec$value]]!=0,,drop=FALSE]
  if(!nrow(ids)) return("")
  paste0("CAUTION: identifier feature(s) ",paste(unique(ids$feature),collapse=", "),
    " have nonzero ",spec$label," ",if(spec$kind=="cox") "coefficients" else "gain",
    " at landmark(s) ",paste(sort(unique(ids$landmark_days)),collapse=", "),
    " days; audit model inputs before interpreting performance or ",noun,".")
}

render_federated_multivariable <- function(root,federated_path,family,dpi,overwrite,variant="pooled") {
  spec <- FEDERATED_MULTIVARIABLE[[family]]
  version <- FEDERATED_MULTIVARIABLE_VARIANTS[[variant]]
  spec$stems <- setNames(paste0(spec$stems,version$stem),names(spec$stems))
  paths <- federated_multivariable_files(federated_path,family,variant)
  metrics <- load_federated_multivariable(paths[["metrics"]],family,"metrics")
  features <- load_federated_multivariable(paths[["features"]],family,"features")
  audit <- federated_input_audit_note(features,family)
  if(startsWith(audit,"CAUTION")) warning(audit)
  capture <- getOption("compass.figure_table_capture")
  write_table <- function(data,stem) {
    path <- figure_sidecar_path(federated_output_path(root,stem,".csv"))
    dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
    readr::write_csv(data,path)
    if(is.function(capture)) capture(path)
  }
  context <- paste0("ADT · platinum endpoint · federated ",
    if(variant=="pooled") "(all sites)" else "(without Dana-Farber)")
  if(!is.null(metrics) && nrow(metrics)) {
    d <- prepare_federated_performance(metrics,family)
    colors <- FEDERATED_MODEL_COLORS[unname(spec$series)]
    count_note <- d %>% arrange(landmark_days,config) %>%
      transmute(note=sprintf("%dd %s: n=%s, events=%s",landmark_days,
        if_else(config=="both","labs","baseline"),scales::comma(n_test),scales::comma(n_events_test))) %>%
      pull(note) %>% paste(collapse="; ")
    caption <- paste(version$note,"Held-out test metrics (not training or tuning CV). Gaps indicate unavailable metrics.",
      paste0("Test counts — ",count_note,"."),audit,sep="\n")
    items <- list(list(plot=plot_model_discrimination(d,"auc","Test Mean AUC(t)",colors,show_legend=TRUE)+
      labs(caption=caption),stem="figure4a_discrimination_auc_platinum",destination=file.path(root,"platinum")))
    figure <- figure_compilation_spec(items[[1]]$stem)
    figure$title <- paste0("Federated ",spec$label,version$label,": labs vs. age baseline"); figure$context <- context
    p <- figure_combine(items,figure,tempdir())
    write_table(mutate(d,input_audit_note=audit),spec$stems[["metrics"]])
    save_federated_panel(p,federated_output_path(root,spec$stems[["metrics"]],".png"),
      figure$width,figure$height,dpi,overwrite)
  }
  if(!is.null(features) && nrow(features)) {
    caption <- if(spec$kind=="cox") paste(
      "Top 15 nonzero coefficients by absolute value per landmark; age and body height excluded from display, as in local plots.",
      "Coefficients are signed log hazard-ratio coefficients, conditional on the other selected features. All source features remain in the CSV.",
      version$note,audit,sep="\n") else paste(
      "Top 15 positive-gain features per landmark; age and body height excluded from display, as in local plots.",
      "Gains are supplied model split gains, not signed effects or SHAP values. All source features remain in the CSV.",
      version$note,audit,sep="\n")
    items <- lapply(c(0,90,180),function(lm) {
      d <- features[features$landmark_days==lm & tolower(features$feature)!="age" &
        features[[spec$value]]!=0,,drop=FALSE]
      stem <- paste0("figure4b_importance_platinum_",spec$kind,"_landmark",lm)
      p <- plot_model_importance(d,spec$kind,paste("Landmark",lm,"days"))+labs(caption=caption)
      if(!any(features$landmark_days==lm))
        p <- p+labs(subtitle=paste(if(spec$kind=="cox") "Coefficients" else "Importance","unavailable"))
      list(plot=p,stem=stem,destination=file.path(root,"platinum"))
    })
    shown <- bind_rows(tibble(landmark_days=numeric(),feature=character()),
      Filter(is.data.frame,lapply(items,function(item) item$plot$data)))
    exported <- features %>% mutate(displayed=paste(landmark_days,feature) %in% paste(shown$landmark_days,shown$feature),
      input_audit_note=audit)
    figure <- figure_compilation_spec(items[[1]]$stem)
    figure$title <- paste0(if(spec$kind=="cox") "Federated elastic-net Cox coefficients" else
      "Federated XGBoost feature importance",version$label)
    figure$context <- context
    p <- figure_combine(items,figure,tempdir())
    write_table(exported,spec$stems[["features"]])
    save_federated_panel(p,federated_output_path(root,spec$stems[["features"]],".png"),
      figure$width,figure$height,dpi,overwrite)
  }
  invisible(NULL)
}

render_federated_deephit_performance <- function(root,federated_path,dpi,overwrite,variant="pooled") {
  d <- load_federated_dynamic_deephit(federated_path,variant)
  if(is.null(d)) return(invisible(NULL))
  stem <- paste0("deephit_performance",FEDERATED_MULTIVARIABLE_VARIANTS[[variant]]$stem)
  path <- figure_sidecar_path(federated_output_path(root,stem,".csv"))
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  readr::write_csv(d,path)
  capture <- getOption("compass.figure_table_capture")
  if(is.function(capture)) capture(path)
  colors <- c(FEDERATED_MODEL_COLORS,FEDERATED_NO_DFCI_MODEL_COLORS)[unique(d$name)]
  p <- plot_model_discrimination(d,"auc","Test Mean AUC(t)",colors,show_legend=TRUE) +
    ggplot2::labs(title=paste0("Federated dynamic DeepHit",
      FEDERATED_MULTIVARIABLE_VARIANTS[[variant]]$label),
      caption="Held-out test metrics from the dynamic arm; missing landmarks are shown as gaps.")
  save_federated_panel(p,federated_output_path(root,stem,".png"),7.5,5.5,dpi,overwrite)
  invisible(d)
}

# rhino_data_v2 predictive checklists (federated_predictive_checklists/lm_<days>/<variant>/):
# a unit-weight M-of-N checklist per landmark x horizon ("both") and a one-item
# age checklist ("baseline"), with metrics on the sites' pooled held-out test patients.
FEDERATED_CHECKLIST_FILES <- c(items="predictive_checklist_checklists_adt.csv",
  metrics="predictive_checklist_metrics_adt.csv")
FEDERATED_CHECKLIST_HORIZONS <- c(`1-year`=365,`5-year`=1825)
FEDERATED_CHECKLIST_MODELS <- c(checklist="Checklist",age_checklist="Age-only checklist")
FEDERATED_CHECKLIST_COLORS <- c(Checklist="#6A3D9A",`Age-only checklist`="#9F86C0")

federated_checklist_files <- function(federated_path,variant="pooled")
  lapply(FEDERATED_CHECKLIST_FILES,function(file) federated_v2_export_files(federated_path,file,variant))

# "RDW (mean) >= 13.2"; thresholds shown to 3 significant digits (exact in the CSV).
federated_checklist_item_label <- function(feature,op,threshold) {
  name <- manuscript_federated_feature_label(if_else(feature=="age_at_anchor","Age",
    format_label(federated_lab_label(feature),federated_feature_stat(feature))))
  if_else(op=="is_missing",paste(name,"is missing"),
    paste(name,op,trimws(formatC(signif(threshold,3),format="fg",digits=3))))
}

load_federated_checklists <- function(federated_path,variant="pooled") {
  paths <- federated_checklist_files(federated_path,variant)
  if(!all(lengths(paths))) {
    warning("Federated predictive checklists unavailable; no ",
      paste(FEDERATED_CHECKLIST_FILES[!lengths(paths)],collapse=", ")," under ",dirname(federated_path))
    return(NULL)
  }
  items <- read_federated_csvs(paths$items)
  metrics <- read_federated_csvs(paths$metrics)
  needed <- list(items=c("landmark_days","horizon_days","config","position","feature","op",
    "threshold_value","M","N"),metrics=c("landmark_days","horizon_days","config","dataset","status",
    "n","n_cases","auc","auc_lower","auc_upper","brier","sensitivity","specificity","ppv","npv"))
  for(part in names(needed)) if(length(missing <- setdiff(needed[[part]],names(list(items=items,metrics=metrics)[[part]]))))
    stop("Federated checklist ",part," are missing: ",paste(missing,collapse=", "))
  keep <- function(d) dplyr::filter(d,landmark_days %in% c(0,90,180),
    horizon_days %in% FEDERATED_CHECKLIST_HORIZONS,config %in% c("both","baseline"))
  items <- keep(items)
  metrics <- dplyr::filter(keep(metrics),dataset=="test")
  if(anyDuplicated(items[c("landmark_days","horizon_days","config","position")]) ||
     anyDuplicated(metrics[c("landmark_days","horizon_days","config")]))
    stop("Duplicate federated checklist rows in ",paste(unlist(paths),collapse="; "))
  for(column in c("auc","auc_lower","auc_upper"))
    if((!is.numeric(metrics[[column]]) && !all(is.na(metrics[[column]]))) ||
       any(!is.na(metrics[[column]]) & (metrics[[column]]<0 | metrics[[column]]>1)))
      stop("Invalid federated checklist ",column)
  list(items=dplyr::mutate(items,display_item=federated_checklist_item_label(feature,op,threshold_value)),
    metrics=dplyr::mutate(metrics,model=if_else(config=="both","checklist","age_checklist")))
}

# Held-out AUC(t) per landmark (a) above the 1-year (b) and 5-year (c) checklists,
# one column per landmark. Cards come from checklist_supplement.R (sourced by 07).
render_federated_checklist_cards <- function(root,federated_path,dpi,overwrite,variant="pooled") {
  loaded <- load_federated_checklists(federated_path,variant)
  if(is.null(loaded) || !nrow(loaded$metrics)) return(invisible(NULL))
  version <- FEDERATED_MULTIVARIABLE_VARIANTS[[variant]]
  stem <- paste0("checklist_supplement",version$stem)
  landmark_levels <- sprintf("Landmark ADT +%d days",c(0,90,180))
  metrics <- loaded$metrics %>% mutate(
    series=factor(unname(FEDERATED_CHECKLIST_MODELS[model]),levels=unname(FEDERATED_CHECKLIST_MODELS)),
    horizon=factor(names(FEDERATED_CHECKLIST_HORIZONS)[match(horizon_days,FEDERATED_CHECKLIST_HORIZONS)],
      levels=names(FEDERATED_CHECKLIST_HORIZONS)),
    landmark_label=factor(sprintf("Landmark ADT +%d days",landmark_days),levels=landmark_levels))
  capture <- getOption("compass.figure_table_capture")
  for(part in c("items","metrics")) {
    path <- figure_sidecar_path(federated_output_path(root,paste0(stem,"_",part),".csv"))
    dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
    readr::write_csv(loaded[[part]],path)
    if(is.function(capture)) capture(path)
  }
  dodge <- ggplot2::position_dodge(width=.45)
  auc <- ggplot2::ggplot(metrics,ggplot2::aes(horizon,auc,color=series,shape=series)) +
    ggplot2::geom_hline(yintercept=.5,color="grey65",linetype="dotted") +
    ggplot2::geom_linerange(ggplot2::aes(ymin=auc_lower,ymax=auc_upper),position=dodge,
      linewidth=.65,na.rm=TRUE) +
    ggplot2::geom_point(position=dodge,size=2.6,na.rm=TRUE) +
    ggplot2::facet_wrap(~landmark_label,nrow=1,drop=FALSE) +
    ggplot2::scale_x_discrete(drop=FALSE) +
    ggplot2::scale_color_manual(values=FEDERATED_CHECKLIST_COLORS,name=NULL,drop=FALSE) +
    ggplot2::scale_shape_manual(values=c(Checklist=16,`Age-only checklist`=1),name=NULL,drop=FALSE) +
    ggplot2::coord_cartesian(ylim=c(0,1)) +
    ggplot2::labs(title="a  Held-out test AUC(t)",x="Checklist horizon",y="AUC(t)") +
    federated_transfer_theme()
  cards <- list()
  for(h in names(FEDERATED_CHECKLIST_HORIZONS)) for(lm in c(0,90,180)) {
    days <- FEDERATED_CHECKLIST_HORIZONS[[h]]
    m <- dplyr::filter(metrics,landmark_days==lm,horizon_days==days)
    full <- dplyr::filter(m,config=="both")
    items <- loaded$items %>% dplyr::filter(landmark_days==lm,horizon_days==days,config=="both") %>%
      dplyr::arrange(position)
    if(nrow(items) && nrow(items)!=items$N[1L])
      stop("Federated checklist at landmark ",lm,", ",h," lists ",nrow(items)," items but N = ",items$N[1L])
    fitted <- nrow(items)>0L && nrow(full)>0L && identical(full$status,"ok")
    cell <- data.frame(landmark_days=lm,status=if(fitted) "ok" else "missing",
      reason=if(nrow(full)) paste("export status", full$status) else "No checklist exported",
      threshold=if(nrow(items)) items$M[1L] else NA,n_items=nrow(items),
      n_test=if(nrow(m)) m$n[1L] else NA,n_events_test=if(nrow(m)) m$n_cases[1L] else NA)
    lines <- checklist_card_lines(cell,dplyr::transmute(items,item_order=position,name=display_item),
      m,round(days/365),FEDERATED_CHECKLIST_MODELS)
    lines$landmark_label <- factor(sprintf("Landmark ADT +%d days",lm),levels=landmark_levels)
    cards[[h]][[length(cards[[h]])+1L]] <- lines
  }
  # Every card row gets the same height, spaced for the longest card.
  line_in <- .17
  panel_in <- line_in*(max(vapply(unlist(cards,recursive=FALSE),nrow,integer(1)))+1.5)
  rows <- lapply(seq_along(cards),function(i) {
    d <- dplyr::bind_rows(lapply(cards[[i]],function(l) {
      l$y <- 1-.02-(seq_len(nrow(l))-1L)*line_in/panel_in
      l
    }))
    plot_checklist_cards(d,sprintf("%s  %s checklists",letters[i+1L],names(cards)[i]),caption=NULL)
  })
  # Matching gtable widths line the AUC facets up with the card columns.
  grobs <- lapply(c(list(auc),rows),ggplot2::ggplotGrob)
  if(length(unique(vapply(grobs,function(g) length(g$widths),integer(1))))==1L) {
    widths <- do.call(grid::unit.pmax,lapply(grobs,function(g) g$widths))
    for(i in seq_along(grobs)) grobs[[i]]$widths <- widths
  }
  heights <- c(3.2,rep(panel_in+.6,length(rows)))
  caption <- paste(version$note,
    "Unit-weight M-of-N checklists; the age-only checklist is a single age item fit on the same development patients.",
    "IPCW-weighted AUC(t), Brier and flag operating points on held-out test patients at each horizon; bars and parentheses are 95% intervals.",
    sep="\n")
  p <- gridExtra::arrangeGrob(grobs=grobs,ncol=1,heights=grid::unit(heights,"in"),
    top=grid::textGrob(paste0("Federated predictive checklists",version$label,": ADT, platinum endpoint"),
      x=grid::unit(10,"pt"),hjust=0,gp=grid::gpar(fontsize=14,fontface="bold")),
    bottom=grid::textGrob(caption,x=grid::unit(10,"pt"),hjust=0,gp=grid::gpar(fontsize=10,col="grey30")))
  save_federated_panel(p,federated_output_path(root,stem,".png"),15.6,sum(heights)+1.2,dpi,overwrite)
  invisible(loaded)
}

# Platinum-trained federated bundles refit without Dana-Farber, scored on local
# endpoints by 04a_transfer_updated.ipynb (`federated_transfer_adt_no_DFCI/`).
# No Dana-Farber patient was in training, so this is an external validation.
FEDERATED_TRANSFER_FILES <- c("transfer_metrics.csv","transfer_auc_t.csv")
# 04 in the patients with a Gleason score (ADT clock) / sequenced patients /
# sequenced patients with a Gleason score (sequencing-date clock): the
# transferred labs score vs a local out-of-fold clinical Cox and a combined Cox
# of the transferred score + the clinical covariates without age (the transferred
# full model already includes age; federated_clinical_comparison.py).
FEDERATED_TRANSFER_CLINICAL_FILE <- "transfer_clinical_comparison.csv"
# 04's patient-level transferred labs scores with Gleason and TP53/RB1/PTEN
# status on each clock; only aggregate KM / event-free / HR tables leave R.
FEDERATED_TRANSFER_KM_FILE <- "transfer_km_patients.csv"
FEDERATED_TRANSFER_NO_DFCI_FILES <- c(FEDERATED_TRANSFER_FILES,FEDERATED_TRANSFER_CLINICAL_FILE,
  FEDERATED_TRANSFER_KM_FILE)
FEDERATED_TRANSFER_CLINICAL <- c(gleason="Gleason + age",trio="TP53/RB1/PTEN + age",
  trio_gleason="Gleason + TP53/RB1/PTEN + age")
FEDERATED_TRANSFER_CLINICAL_COLORS <- c(`Gleason + age`="#1baf7a",`TP53/RB1/PTEN + age`="#8e44ad",
  `Gleason + TP53/RB1/PTEN + age`="#d6336c")
# Somatic subpopulations run on a sequencing-date clock; the rest on ADT initiation.
FEDERATED_TRANSFER_CLINICAL_ORIGIN <- c(gleason="ADT initiation",trio="sequencing date",
  trio_gleason="sequencing date")
FEDERATED_TRANSFER_CLINICAL_SCORES <- c("transferred_labs","transferred_age","clinical_cox","combined_cox")
# Plotted scores (shape); color is the bundle, or the clinical model for the local Cox.
FEDERATED_TRANSFER_CLINICAL_TYPES <- c(transferred_labs="Federated score",clinical_cox="Local clinical Cox",
  combined_cox="Federated score + clinical (no age)")
# What the combined Cox adds to the transferred score; names its shape when one subpopulation is drawn.
FEDERATED_TRANSFER_CLINICAL_COMBINED <- c(gleason="Gleason",trio="TP53/RB1/PTEN",
  trio_gleason="Gleason + TP53/RB1/PTEN")
FEDERATED_TRANSFER_CLINICAL_SHAPES <- c(transferred_labs=16,clinical_cox=18,combined_cox=17)
# combined_cox - clinical_cox, absent from CSVs written before 04 fitted the combined Cox.
FEDERATED_TRANSFER_CLINICAL_OPTIONAL <- c("delta_auc_vs_clinical","delta_auc_vs_clinical_lo","delta_auc_vs_clinical_hi")
FEDERATED_TRANSFER_ENDPOINTS <- c(platinum="Platinum (training endpoint)",avpc="AVPC",nepc="NEPC")
FEDERATED_TRANSFER_MODELS <- c(xgboost_cox="Federated XGBoost",elastic_net_cox="Federated elastic-net Cox")
FEDERATED_TRANSFER_SPEC <- list(stem="transfer_discrimination_no_dfci",auc_stem="transfer_auc_t_no_dfci",
  title="Federated platinum models trained without Dana-Farber, applied to local Dana-Farber endpoints",
  subtitle="External validation: no Dana-Farber patient contributed to training",
  source="Platinum-trained federated bundles refit without Dana-Farber and scored on the local ADT cohort; every endpoint, platinum included, is out of sample.")

# Path helpers mirror their namesakes in prepare_figure_data.py. Every federated
# input defaults to the rhino_data_v2 delivery under the COMPASS data root.
federated_config_path <- function(config, key, default) {
  path <- config[[key]]
  if(length(path)==1L && !is.na(path) && nzchar(path)) path else default
}
federated_rhino_root <- function(config)
  federated_config_path(config,"federated_rhino_root",file.path(config$data_root,"rhino_data_v2"))
federated_transfer_no_dfci_dir <- function(config)
  federated_config_path(config,"federated_transfer_no_dfci_path",
    file.path(federated_rhino_root(config),"federated_transfer_adt_no_DFCI"))
federated_figure_data_dir <- function(config)
  federated_config_path(config,"federated_figure_data_path",
    file.path(federated_rhino_root(config),"figure_data"))

load_federated_transfer <- function(directory, kind=c("metrics","auc_t")) {
  kind <- match.arg(kind)
  path <- file.path(directory,paste0("transfer_",kind,".csv"))
  if(!file.exists(path)) {
    warning("Federated transfer ",kind," unavailable; missing: ",path)
    return(NULL)
  }
  d <- readr::read_csv(path,show_col_types=FALSE)
  needed <- c("endpoint","bundle","landmark_days","config",if(kind=="metrics")
    c("model_family","n_patients","n_events","mean_auc") else c("horizon_days","auc","n_cases"))
  missing <- setdiff(needed,names(d))
  if(length(missing)) stop("Federated transfer ",kind," input is missing: ",paste(missing,collapse=", "))
  # Bootstrap CIs and the age-only comparison are absent when 04 ran without
  # them (or predates the mean AUC(t) bootstrap); keep the columns so plots
  # show gaps, not errors. Harrell's C stays in the CSV but is not plotted.
  optional <- if(kind=="metrics") c("mean_auc_lo","mean_auc_hi","mean_auc_baseline","delta_auc_vs_baseline",
    "delta_auc_lo","delta_auc_hi","c_index","c_index_lo","c_index_hi","c_index_baseline",
    "delta_c_vs_baseline","delta_c_lo","delta_c_hi") else character()
  for(column in setdiff(optional,names(d))) d[[column]] <- NA_real_
  d <- mutate(d,endpoint=tolower(endpoint),config=as.character(config)) %>%
    filter(endpoint %in% names(FEDERATED_TRANSFER_ENDPOINTS))
  bounded <- if(kind=="metrics") c("mean_auc","mean_auc_lo","mean_auc_hi","mean_auc_baseline",
    "c_index","c_index_lo","c_index_hi","c_index_baseline") else "auc"
  for(column in bounded) {
    if(!is.numeric(d[[column]]) && !all(is.na(d[[column]]))) stop("Non-numeric transfer metric: ",column)
    if(any(!is.na(d[[column]]) & (!is.finite(d[[column]]) | d[[column]]<0 | d[[column]]>1)))
      stop("Invalid transfer metric: ",column)
  }
  if(kind=="metrics" && (any(!is.finite(d$n_patients) | d$n_patients<0) ||
     any(!is.finite(d$n_events) | d$n_events<0 | d$n_events>d$n_patients)))
    stop("Invalid transfer patient/event counts")
  keys <- c("endpoint","bundle","landmark_days","config",if(kind=="auc_t") "horizon_days")
  if(anyDuplicated(d[keys])) stop("Duplicate ",paste(keys,collapse="/")," rows in federated transfer ",kind," input")
  if(!nrow(d)) warning("No platinum/NEPC/AVPC rows in federated transfer ",kind," input: ",path)
  d
}

# A transfer CSV written before 04 bootstrapped mean AUC(t) lacks the AUC
# columns the figures plot: warn and skip it rather than stop the whole run.
federated_transfer_stale <- function(d, needed, path) {
  missing <- setdiff(needed,names(d))
  if(!length(missing)) return(FALSE)
  warning("Federated transfer input predates the mean AUC(t) columns (rerun 04a_transfer_updated.ipynb); missing ",
    paste(missing,collapse=", "),": ",path)
  TRUE
}

# 04's clinical comparison: one row per endpoint x landmark x subpopulation x
# bundle x score. The local Cox rows repeat for each bundle (each carries its
# paired ΔAUC(t) vs that bundle's transferred labs score, kept in the table only);
# the combined Cox (that bundle's score + the clinical covariates but age) is per bundle.
load_federated_transfer_clinical <- function(directory) {
  path <- file.path(directory,FEDERATED_TRANSFER_CLINICAL_FILE)
  if(!file.exists(path)) {
    warning("Federated transfer Gleason/TP53-RB1-PTEN comparison unavailable; missing: ",path)
    return(NULL)
  }
  d <- readr::read_csv(path,show_col_types=FALSE)
  needed <- c("endpoint","landmark_days","subpopulation","bundle","model_family","model",
    "n_patients","n_events","n_scored","n_scored_events")
  missing <- setdiff(needed,names(d))
  if(length(missing)) stop("Federated transfer clinical input is missing: ",paste(missing,collapse=", "))
  metrics <- c("mean_auc","mean_auc_lo","mean_auc_hi","delta_auc_vs_transferred","delta_auc_lo","delta_auc_hi")
  if(federated_transfer_stale(d,metrics,path)) return(NULL)
  for(column in setdiff(FEDERATED_TRANSFER_CLINICAL_OPTIONAL,names(d))) d[[column]] <- NA_real_
  metrics <- c(metrics,FEDERATED_TRANSFER_CLINICAL_OPTIONAL)
  d <- mutate(d,endpoint=tolower(endpoint),subpopulation=as.character(subpopulation),model=as.character(model)) %>%
    filter(endpoint %in% names(FEDERATED_TRANSFER_ENDPOINTS),subpopulation %in% names(FEDERATED_TRANSFER_CLINICAL))
  unknown <- setdiff(unique(d$model),FEDERATED_TRANSFER_CLINICAL_SCORES)
  if(length(unknown)) stop("Unknown federated transfer clinical score: ",paste(unknown,collapse=", "))
  for(column in metrics) {
    if(!is.numeric(d[[column]]) && !all(is.na(d[[column]]))) stop("Non-numeric transfer metric: ",column)
    low <- if(startsWith(column,"delta")) -1 else 0
    if(any(!is.na(d[[column]]) & (!is.finite(d[[column]]) | d[[column]]<low | d[[column]]>1)))
      stop("Invalid transfer metric: ",column)
  }
  if(any(!is.finite(d$n_events) | d$n_events<0 | d$n_events>d$n_patients) ||
     any(!is.finite(d$n_scored) | d$n_scored>d$n_patients | d$n_scored_events>d$n_scored))
    stop("Invalid transfer clinical patient/event counts")
  keys <- c("endpoint","landmark_days","subpopulation","bundle","model")
  if(anyDuplicated(d[keys])) stop("Duplicate ",paste(keys,collapse="/")," rows in federated transfer clinical input")
  if(!nrow(d)) warning("No platinum/NEPC/AVPC rows in federated transfer clinical input: ",path)
  d
}

# Mean AUC(t) of each transferred full model, the local Cox (one point; it does
# not depend on the bundle) and each bundle's combined Cox (transferred score +
# clinical covariates but age). The paired ΔAUC(t) columns stay in the exported
# table. series (color) is the bundle, or the clinical model for the local Cox;
# score_type (shape) is the score, the combined one named for its covariates
# when a single subpopulation is drawn.
prepare_federated_transfer_clinical <- function(d, subpopulations=names(FEDERATED_TRANSFER_CLINICAL), counts=NULL) {
  d <- filter(rename(d,score=model),subpopulation %in% subpopulations)
  d <- label_federated_transfer(d,sort(unique(d$landmark_days)),counts)
  clinical <- unname(FEDERATED_TRANSFER_CLINICAL[subpopulations])
  labels <- FEDERATED_TRANSFER_CLINICAL_TYPES
  if(length(subpopulations)==1L)
    labels[["combined_cox"]] <- paste("Federated score +",FEDERATED_TRANSFER_CLINICAL_COMBINED[[subpopulations]])
  types <- unname(labels[intersect(names(labels),d$score)])
  cox <- filter(d,score=="clinical_cox")
  bind_rows(
    mutate(filter(d,score %in% c("transferred_labs","combined_cox")),series=as.character(model)),
    mutate(distinct(cox,endpoint,landmark_days,subpopulation,.keep_all=TRUE),
      series=unname(FEDERATED_TRANSFER_CLINICAL[subpopulation]))) %>%
    mutate(value=mean_auc,lo=mean_auc_lo,hi=mean_auc_hi,
      series=factor(series,levels=c(levels(d$model),clinical)),
      score_type=factor(unname(labels[score]),levels=types),
      metric=factor("Mean AUC(t)"),
      subpopulation_label=factor(unname(FEDERATED_TRANSFER_CLINICAL[subpopulation]),levels=clinical))
}

# counts: optional per-endpoint strip line (federated_transfer_strip_counts).
label_federated_transfer <- function(d, landmarks=sort(unique(d$landmark_days)), counts=NULL) {
  known <- unname(FEDERATED_TRANSFER_MODELS)
  endpoints <- FEDERATED_TRANSFER_ENDPOINTS
  shown <- intersect(names(counts),names(endpoints))
  endpoints[shown] <- paste0(endpoints[shown],"\n",counts[shown])
  mutate(d,
    endpoint_label=factor(unname(endpoints[endpoint]),levels=unname(endpoints)),
    model=coalesce(unname(FEDERATED_TRANSFER_MODELS[tolower(model_family)]),bundle),
    model=factor(model,levels=c(intersect(known,model),sort(setdiff(unique(model),known)))),
    landmark=factor(sprintf("%s%d days",if_else(landmark_days>0,"+",""),landmark_days),
      levels=sprintf("%s%d days",ifelse(landmarks>0,"+",""),landmarks)))
}

# Patients/events per endpoint at one landmark, as facet-strip lines. Rows are
# per bundle/config (or per score) but share one cohort per endpoint.
federated_transfer_strip_counts <- function(d, landmark=0) {
  d <- d[d$landmark_days==landmark,,drop=FALSE]
  if(!nrow(d)) return(NULL)
  d <- d %>% group_by(endpoint) %>% summarise(n=max(n_patients),events=max(n_events),.groups="drop")
  setNames(sprintf("Day %d: n = %s, %s events",as.integer(landmark),scales::comma(d$n),scales::comma(d$events)),
    d$endpoint)
}

prepare_federated_transfer <- function(metrics, counts=NULL) {
  d <- label_federated_transfer(metrics,counts=counts) %>% mutate(
    features=factor(if_else(config=="baseline","Age only","Full model"),levels=c("Full model","Age only")))
  mutate(d,metric=factor("Mean AUC(t)"),value=mean_auc,lo=mean_auc_lo,hi=mean_auc_hi)
}

federated_transfer_count_note <- function(metrics) {
  metrics %>% distinct(endpoint,landmark_days,n_patients,n_events) %>%
    mutate(endpoint=factor(endpoint,levels=names(FEDERATED_TRANSFER_ENDPOINTS))) %>%
    arrange(endpoint,landmark_days) %>%
    group_by(endpoint) %>%
    summarise(note=paste0(toupper(first(as.character(endpoint))),": ",paste(sprintf("%dd %s/%s",
      as.integer(landmark_days),scales::comma(n_patients),scales::comma(n_events)),collapse=", ")),.groups="drop") %>%
    pull(note) %>% paste(collapse="; ")
}

federated_transfer_colors <- function(models) {
  palette <- c(`Federated XGBoost`="#B58900",`Federated elastic-net Cox`="#2a78d6")
  extra <- setdiff(models,names(palette))
  c(palette,setNames(rep_len(c("#1baf7a","#eb6834","#7f8c8d"),length(extra)),extra))
}

federated_transfer_theme <- function() {
  theme_fig() +
    theme(strip.placement="outside",strip.background=element_blank(),strip.text=element_text(face="bold"),
      legend.position="top",legend.justification="left",panel.grid.minor=element_blank(),
      panel.spacing=grid::unit(.8,"lines"),plot.title.position="plot",plot.caption.position="plot",
      plot.caption=element_text(hjust=0,size=10))
}

plot_federated_transfer <- function(d, caption, title=FEDERATED_TRANSFER_SPEC$title,
                                    subtitle=FEDERATED_TRANSFER_SPEC$subtitle) {
  references <- tibble(metric=factor(levels(d$metric),levels=levels(d$metric)),reference=.5)
  dodge <- position_dodge(width=.7)
  ggplot(d,aes(landmark,value,color=model,shape=features,group=interaction(model,features))) +
    geom_hline(data=references,aes(yintercept=reference),inherit.aes=FALSE,color="grey55",linetype="dotted",linewidth=.8) +
    geom_linerange(aes(ymin=lo,ymax=hi),position=dodge,linewidth=.8,na.rm=TRUE) +
    geom_point(position=dodge,size=2.8,stroke=1.1,fill="white",na.rm=TRUE) +
    scale_color_manual(values=federated_transfer_colors(levels(d$model)),name=NULL,drop=FALSE) +
    scale_shape_manual(values=c(`Full model`=16,`Age only`=21),name=NULL,drop=FALSE) +
    facet_grid(metric~endpoint_label,scales="free_y",switch="y") +
    labs(x="Landmark after ADT start",y=NULL,caption=caption,title=title,subtitle=subtitle) +
    federated_transfer_theme()
}

plot_federated_transfer_clinical <- function(d, caption, title=NULL, subtitle=NULL) {
  references <- tibble(metric=factor(levels(d$metric),levels=levels(d$metric)),reference=.5)
  series <- levels(d$series)
  colors <- c(federated_transfer_colors(setdiff(series,FEDERATED_TRANSFER_CLINICAL)),
    FEDERATED_TRANSFER_CLINICAL_COLORS)[series]
  shapes <- setNames(FEDERATED_TRANSFER_CLINICAL_SHAPES[d$score[match(levels(d$score_type),d$score_type)]],
    levels(d$score_type))
  rows <- if(length(unique(d$subpopulation_label))>1) vars(subpopulation_label,metric) else vars(metric)
  dodge <- position_dodge(width=.75)
  ggplot(d,aes(landmark,value,color=series,shape=score_type,group=interaction(series,score_type))) +
    geom_hline(data=references,aes(yintercept=reference),inherit.aes=FALSE,color="grey55",linetype="dotted",linewidth=.8) +
    geom_linerange(aes(ymin=lo,ymax=hi),position=dodge,linewidth=.8,na.rm=TRUE) +
    geom_point(position=dodge,size=3,stroke=1.1,na.rm=TRUE) +
    scale_color_manual(values=colors,name=NULL,drop=FALSE,guide=guide_legend(override.aes=list(shape=15))) +
    scale_shape_manual(values=shapes,name=NULL,drop=FALSE) +
    facet_grid(rows=rows,cols=vars(endpoint_label),scales="free_y",switch="y") +
    labs(x="Landmark after ADT start",y=NULL,caption=caption,title=title,subtitle=subtitle) +
    federated_transfer_theme()
}

federated_transfer_clinical_count_note <- function(clinical, subpopulation) {
  d <- clinical[clinical$subpopulation==subpopulation,,drop=FALSE]
  if(!nrow(d)) return("none")
  federated_transfer_count_note(d)
}

write_federated_transfer_table <- function(root, data, stem) {
  path <- figure_sidecar_path(federated_output_path(root,stem,".csv"))
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  readr::write_csv(data,path)
  capture <- getOption("compass.figure_table_capture")
  if(is.function(capture)) capture(path)
}

render_federated_transfer <- function(root, directory, dpi, overwrite) {
  spec <- FEDERATED_TRANSFER_SPEC
  metrics <- load_federated_transfer(directory,"metrics")
  if(is.null(metrics) || !nrow(metrics)) return(invisible(NULL))
  auc_t <- load_federated_transfer(directory,"auc_t")
  caption <- paste(spec$source,
    "Mean AUC(t): IPCW cumulative/dynamic AUC over the estimable 1-, 2-, 3- and 5-year horizons. Bars: 95% bootstrap CIs over patients.",
    paste0("Patients/events by landmark — ",federated_transfer_count_note(metrics),"."),sep="\n")
  d <- prepare_federated_transfer(metrics)
  write_federated_transfer_table(root,metrics,spec$stem)
  if(!is.null(auc_t) && nrow(auc_t)) write_federated_transfer_table(root,auc_t,spec$auc_stem)
  save_federated_panel(plot_federated_transfer(d,caption,spec$title,spec$subtitle),
    federated_output_path(root,spec$stem,".png"),13,6,dpi,overwrite)
  invisible(metrics)
}

# The all-model transfer table has one checklist model per training horizon.
# Keep it in a separate supplement so the main transfer figure retains its
# original multi-horizon Cox/XGBoost estimand and clinical comparison panels.
render_federated_checklist_supplement <- function(root,directory,dpi,overwrite) {
  path <- file.path(directory,"transfer_metrics.csv")
  if(!file.exists(path)) return(invisible(NULL))
  d <- readr::read_csv(path,show_col_types=FALSE)
  needed <- c("endpoint","bundle","model_family","landmark_days","config",
    "model_horizon_days","evaluation_horizon_days","mean_auc")
  if(length(missing <- setdiff(needed,names(d))))
    stop("All-model transfer metrics are missing: ",paste(missing,collapse=", ")," in ",path)
  d <- dplyr::filter(d,model_family=="predictive_checklist",
    endpoint %in% names(FEDERATED_TRANSFER_ENDPOINTS),
    !is.na(model_horizon_days),model_horizon_days==evaluation_horizon_days,
    config %in% c("both","baseline"))
  if(!nrow(d)) return(invisible(NULL))
  if(anyDuplicated(d[c("endpoint","bundle","landmark_days","config","model_horizon_days")]))
    stop("Duplicate predictive checklist transfer rows in ",path)
  if((!is.numeric(d$mean_auc) && !all(is.na(d$mean_auc))) || any(!is.na(d$mean_auc) &
      (!is.finite(d$mean_auc) | d$mean_auc<0 | d$mean_auc>1)))
    stop("Invalid predictive checklist AUC in ",path)
  for(column in c("mean_auc_lo","mean_auc_hi","brier"))
    if(!column %in% names(d)) d[[column]] <- NA_real_
  stem <- "transfer_predictive_checklist_supplement"
  write_federated_transfer_table(root,d,stem)
  if(!any(is.finite(d$mean_auc))) return(invisible(d))
  shown <- d %>% dplyr::mutate(
    endpoint_label=factor(unname(FEDERATED_TRANSFER_ENDPOINTS[endpoint]),
      levels=unname(FEDERATED_TRANSFER_ENDPOINTS)),
    horizon_label=factor(paste0(as.integer(model_horizon_days/365),"-year checklist"),
      levels=unique(paste0(as.integer(sort(unique(model_horizon_days))/365),"-year checklist"))),
    landmark=factor(paste0("+",landmark_days," days"),
      levels=paste0("+",sort(unique(landmark_days))," days")),
    config_label=factor(if_else(config=="both","Checklist","Age-only checklist"),
      levels=c("Checklist","Age-only checklist")))
  palette <- setNames(rep_len(c("#6A3D9A","#A6761D","#1B9E77"),length(unique(d$bundle))),
    unique(d$bundle))
  p <- ggplot2::ggplot(shown,ggplot2::aes(landmark,mean_auc,color=bundle,shape=config_label,
    group=interaction(bundle,config_label,model_horizon_days))) +
    ggplot2::geom_hline(yintercept=.5,color="grey65",linetype="dotted") +
    ggplot2::geom_line(linewidth=.75,na.rm=TRUE) +
    ggplot2::geom_linerange(ggplot2::aes(ymin=mean_auc_lo,ymax=mean_auc_hi),
      linewidth=.65,na.rm=TRUE) +
    ggplot2::geom_point(size=2.5,na.rm=TRUE) +
    ggplot2::facet_grid(endpoint_label~horizon_label) +
    ggplot2::scale_color_manual(values=palette,name="Bundle") +
    ggplot2::scale_shape_manual(values=c("Checklist"=16,"Age-only checklist"=1),name=NULL) +
    ggplot2::coord_cartesian(ylim=c(0,1)) +
    ggplot2::labs(title="Supplementary: predictive checklist transfer to Dana-Farber",
      subtitle="Platinum-trained models without Dana-Farber; horizon-matched external validation",
      x="Landmark after ADT start",y="Horizon-specific AUC(t)",
      caption="Points are IPCW AUC(t); bars are patient-bootstrap 95% intervals when available. Frozen checklist Brier scores and counts are in the accompanying CSV.") +
    federated_transfer_theme()
  save_federated_panel(p,federated_output_path(root,stem,".png"),11,8,dpi,overwrite)
  invisible(d)
}

# A shared DFCI cohort and horizon permits a direct external comparison of
# Cox, XGBoost, both DeepHit arms and the horizon-specific checklist.
render_federated_all_model_transfer_supplement <- function(root,directory,dpi,overwrite) {
  metrics_path <- file.path(directory,"transfer_metrics.csv")
  auc_path <- file.path(directory,"transfer_auc_t.csv")
  if(!file.exists(metrics_path) || !file.exists(auc_path)) return(invisible(NULL))
  metrics <- readr::read_csv(metrics_path,show_col_types=FALSE)
  auc <- readr::read_csv(auc_path,show_col_types=FALSE)
  if(length(missing <- setdiff(c("bundle","model_family"),names(metrics))))
    stop("All-model metrics are missing: ",paste(missing,collapse=", "))
  if(length(missing <- setdiff(c("endpoint","bundle","landmark_days","config",
      "horizon_days","auc"),names(auc))))
    stop("All-model AUC table is missing: ",paste(missing,collapse=", "))
  for(column in c("training_arm","model_horizon_days","evaluation_horizon_days"))
    if(!column %in% names(auc)) auc[[column]] <- NA
  family <- dplyr::distinct(metrics,bundle,model_family)
  if(anyDuplicated(family$bundle)) stop("Bundle has multiple model families in ",metrics_path)
  d <- dplyr::left_join(auc,family,by="bundle") %>%
    dplyr::filter(config=="both",endpoint %in% names(FEDERATED_TRANSFER_ENDPOINTS),
      is.na(evaluation_horizon_days) | evaluation_horizon_days==horizon_days,
      is.na(model_horizon_days) | model_horizon_days==horizon_days)
  if(!nrow(d)) return(invisible(NULL))
  checklist_horizons <- unique(d$horizon_days[d$model_family=="predictive_checklist"])
  if(length(checklist_horizons))
    d <- dplyr::filter(d,horizon_days %in% checklist_horizons)
  if(!is.numeric(d$auc) || any(!is.na(d$auc) & (!is.finite(d$auc) | d$auc<0 | d$auc>1)))
    stop("Invalid all-model transfer AUC in ",auc_path)
  d <- d %>% dplyr::mutate(series=dplyr::case_when(
    model_family=="elastic_net_cox" ~ "Elastic-Net Cox",
    model_family=="xgboost_cox" ~ "XGBoost",
    model_family=="dynamic_deephit" & training_arm=="dynamic" ~ "DeepHit dynamic",
    model_family=="dynamic_deephit" ~ "DeepHit landmark",
    model_family=="predictive_checklist" ~ "Predictive checklist",
    TRUE ~ as.character(bundle)))
  if(anyDuplicated(d[c("endpoint","landmark_days","horizon_days","series")]))
    d$series <- paste(d$series,d$bundle,sep=" · ")
  stem <- "transfer_all_models_supplement"
  write_federated_transfer_table(root,d,stem)
  if(!any(is.finite(d$auc))) return(invisible(d))
  shown <- d %>% dplyr::mutate(
    endpoint_label=factor(unname(FEDERATED_TRANSFER_ENDPOINTS[endpoint]),
      levels=unname(FEDERATED_TRANSFER_ENDPOINTS)),
    horizon_label=factor(paste0(as.integer(horizon_days)," days"),
      levels=paste0(sort(unique(as.integer(horizon_days)))," days")),
    landmark=factor(paste0("+",landmark_days," days"),
      levels=paste0("+",sort(unique(landmark_days))," days")))
  palette <- setNames(rep_len(c("#4C72B0","#B58900","#2A9D8F","#17675D","#6A3D9A"),
    length(unique(shown$series))),unique(shown$series))
  p <- ggplot2::ggplot(shown,ggplot2::aes(landmark,auc,color=series,group=series)) +
    ggplot2::geom_hline(yintercept=.5,color="grey65",linetype="dotted") +
    ggplot2::geom_line(linewidth=.75,na.rm=TRUE) +
    ggplot2::geom_point(size=2.5,na.rm=TRUE) +
    ggplot2::facet_grid(endpoint_label~horizon_label) +
    ggplot2::scale_color_manual(values=palette,name=NULL) +
    ggplot2::coord_cartesian(ylim=c(0,1)) +
    ggplot2::labs(title="Supplementary: no-DFCI models transferred to Dana-Farber",
      subtitle="Same endpoint-specific patients at each landmark and horizon; checklist is horizon-specific",
      x="Landmark after ADT start",y="IPCW AUC(t)",
      caption="Platinum-trained frozen models. The companion CSV records each model's arm, training horizon, case count and AUC(t).") +
    federated_transfer_theme()
  save_federated_panel(p,federated_output_path(root,stem,".png"),12,8,dpi,overwrite)
  invisible(d)
}

render_federated_transfer_clinical <- function(root, directory, dpi, overwrite) {
  clinical <- load_federated_transfer_clinical(directory)
  if(is.null(clinical) || !nrow(clinical)) return(invisible(NULL))
  caption <- paste(FEDERATED_TRANSFER_SPEC$source,
    "Transferred full labs models scored unchanged. Local unpenalized Cox models (Gleason score + age; TP53, RB1, PTEN any-alteration indicators + age; Gleason + TP53/RB1/PTEN + age) and combined Cox models (each transferred score, standardized, + the same clinical covariates without age, which the transferred full models already include) are scored out-of-fold (stratified K-fold) on the same patients.",
    "Gleason: ADT-initiation clock. TP53/RB1/PTEN with or without Gleason: sequencing-date clock (sequenced patients; transferred models rescored on labs before sequencing + landmark). Clinical data had to be available before the landmark. Mean AUC(t) over the horizons estimable in each subpopulation. Bars: 95% bootstrap CIs over patients.",
    paste0("Gleason subpopulation patients/events — ",federated_transfer_clinical_count_note(clinical,"gleason"),"."),
    paste0("Sequenced subpopulation patients/events — ",federated_transfer_clinical_count_note(clinical,"trio"),"."),
    paste0("Sequenced with Gleason subpopulation patients/events — ",
      federated_transfer_clinical_count_note(clinical,"trio_gleason"),"."),sep="\n")
  write_federated_transfer_table(root,clinical,"transfer_clinical_comparison_no_dfci")
  save_federated_panel(plot_federated_transfer_clinical(prepare_federated_transfer_clinical(clinical),caption,
      "Transferred federated labs models alone and combined with Gleason and TP53/RB1/PTEN",
      "Federated models trained without Dana-Farber; local and combined Cox models fitted on Dana-Farber patients and scored out-of-fold"),
    federated_output_path(root,"transfer_clinical_comparison_no_dfci",".png"),13,10,dpi,overwrite)
  invisible(clinical)
}

# Time-to-event views of the transferred labs score (Federated Figure 5 e-h and
# 05_transfer/km/). Score groups use Dana-Farber cutpoints computed within each
# clock x endpoint x bundle x landmark cohort: each endpoint has its own at-risk
# patients and scores. Gleason views run on the ADT clock and TP53/RB1/PTEN
# views on the sequencing clock (no somatic data on the ADT clock).
FEDERATED_TRANSFER_KM_CLOCKS <- c(adt="ADT",sequencing="sequencing")
FEDERATED_TRANSFER_KM_YLAB <- c(platinum="Platinum-free probability",nepc="NEPC-free probability",
  avpc="AVPC-free probability")
FEDERATED_TRANSFER_ENDPOINT_COLORS <- c(`Platinum (training endpoint)`="#0072B2",AVPC="#009E73",NEPC="#D55E00")
FEDERATED_TRANSFER_KM_HORIZON <- 1825
# Panels with fewer events are not drawn.
FEDERATED_TRANSFER_KM_MIN_EVENTS <- 5
# Manuscript KM panels (top vs bottom 10% of this bundle's score, one per
# endpoint) and gradient; other bundles, landmarks and views are in 05_transfer/km/.
FEDERATED_TRANSFER_KM_FIGURE <- list(model_family="xgboost_cox",landmark_days=180)
FEDERATED_TRANSFER_GLEASON <- c("Gleason ≤7","Gleason ≥8")
FEDERATED_TRANSFER_TRIO <- c("TP53/RB1/PTEN wild-type","TP53/RB1/PTEN altered")
federated_transfer_cross <- function(groups)
  paste(rep(groups,each=2),c("low score","high score"),sep=" · ")
# Literal levels: figure_extreme_labels(.2) and (.1), without needing the pipeline at source time.
FEDERATED_TRANSFER_KM_VIEWS <- list(
  score=list(clock="adt",column="score_extreme",levels=c("Bottom 20%","Top 20%"),label="Federated score"),
  score_decile=list(clock="adt",column="score_decile",levels=c("Bottom 10%","Top 10%"),label="Federated score"),
  gleason=list(clock="adt",column="gleason_group",levels=FEDERATED_TRANSFER_GLEASON,label="Gleason"),
  trio=list(clock="sequencing",column="trio_group",levels=FEDERATED_TRANSFER_TRIO,label="TP53/RB1/PTEN"),
  score_trio=list(clock="sequencing",column="score_trio",levels=federated_transfer_cross(FEDERATED_TRANSFER_TRIO),
    label="Federated score × TP53/RB1/PTEN"))
FEDERATED_TRANSFER_GRADIENT <- list(
  score_quartile=list(clock="adt",levels=paste0("Q",1:4),label="Federated score quartile"),
  gleason_group=list(clock="adt",levels=FEDERATED_TRANSFER_GLEASON,label="Gleason"),
  trio_pattern=list(clock="sequencing",
    levels=c("0 altered","TP53 altered","PTEN altered","RB1 altered","2 altered","3 altered"),
    label="TP53/PTEN/RB1 alteration"))
FEDERATED_TRANSFER_KM_TABLES <- c(curves="transfer_km_curves_no_dfci",summary="transfer_km_summary_no_dfci",
  gradient="transfer_km_gradient_no_dfci")
federated_transfer_horizon_label <- function(horizon=FEDERATED_TRANSFER_KM_HORIZON)
  sprintf("%d-year",as.integer(round(horizon/365)))

load_federated_transfer_km <- function(directory) {
  path <- file.path(directory,FEDERATED_TRANSFER_KM_FILE)
  if(!file.exists(path)) {
    warning("Federated transfer KM patients unavailable; missing: ",path)
    return(NULL)
  }
  d <- readr::read_csv(path,show_col_types=FALSE,
    col_types=readr::cols(HAS_SEQUENCING=readr::col_character(),.default=readr::col_guess()))
  needed <- c("clock","endpoint","bundle","model_family","landmark_days","person_id","risk_score",
    "duration_days","event","GLEASON_SCORE","HAS_SEQUENCING","TP53_ANY","RB1_ANY","PTEN_ANY")
  missing <- setdiff(needed,names(d))
  if(length(missing)) stop("Federated transfer KM input is missing: ",paste(missing,collapse=", "))
  d <- mutate(d,clock=as.character(clock),endpoint=tolower(endpoint),model_family=as.character(model_family),
      # pandas writes the True/NaN flag as "1.0" once it is concatenated with the all-NaN ADT-clock column.
      sequenced=toupper(HAS_SEQUENCING) %in% c("TRUE","T") | suppressWarnings(as.numeric(HAS_SEQUENCING)) %in% 1) %>%
    filter(endpoint %in% names(FEDERATED_TRANSFER_ENDPOINTS),clock %in% names(FEDERATED_TRANSFER_KM_CLOCKS))
  if(any(!d$event %in% c(0,1)) || any(!is.finite(d$duration_days) | d$duration_days<0))
    stop("Invalid federated transfer KM outcomes")
  keys <- c("clock","endpoint","bundle","landmark_days","person_id")
  if(anyDuplicated(d[keys])) stop("Duplicate ",paste(keys,collapse="/")," rows in federated transfer KM input")
  if(!nrow(d)) warning("No platinum/NEPC/AVPC rows in federated transfer KM input: ",path)
  d
}

# Ties at the median go low, matching the closed-bottom figure_extreme_quintiles.
federated_transfer_score_half <- function(x) {
  out <- rep(NA_character_,length(x)); ok <- is.finite(x)
  if(any(ok)) out[ok] <- ifelse(x[ok]>median(x[ok]),"high score","low score")
  out
}
# Score Q1..Qk. Tied scores stay in one group; if ties collapse a group there is no gradient.
federated_transfer_score_quantile <- function(x, k=4L) {
  out <- rep(NA_character_,length(x)); ok <- is.finite(x)
  if(sum(ok)<k) return(out)
  breaks <- unique(quantile(x[ok],seq(0,1,1/k),names=FALSE))
  if(length(breaks)!=k+1L) return(out)
  out[ok] <- paste0("Q",cut(x[ok],breaks,include.lowest=TRUE,labels=FALSE))
  out
}

federated_transfer_km_groups <- function(d) {
  sequenced <- d$clock=="sequencing" & d$sequenced
  trio_n <- d$TP53_ANY+d$RB1_ANY+d$PTEN_ANY
  d %>% group_by(clock,endpoint,bundle,landmark_days) %>%
    mutate(score_extreme=as.vector(figure_extreme_quintiles(risk_score)),
      score_decile=as.vector(figure_extreme_quintiles(risk_score,.1)),
      score_half=federated_transfer_score_half(risk_score),
      score_quartile=federated_transfer_score_quantile(risk_score,4L)) %>% ungroup() %>%
    mutate(gleason_group=if_else(clock=="adt",figure_gleason_groups(GLEASON_SCORE),NA_character_),
      trio_group=if_else(sequenced,FEDERATED_TRANSFER_TRIO[1+(trio_n>0)],NA_character_),
      trio_pattern=case_when(
        sequenced & trio_n==0 ~ "0 altered",
        sequenced & trio_n==1 & TP53_ANY==1 ~ "TP53 altered",
        sequenced & trio_n==1 & PTEN_ANY==1 ~ "PTEN altered",
        sequenced & trio_n==1 & RB1_ANY==1 ~ "RB1 altered",
        sequenced & trio_n==2 ~ "2 altered",
        sequenced & trio_n==3 ~ "3 altered",
        TRUE ~ NA_character_),
      score_trio=if_else(is.na(trio_group)|is.na(score_half),NA_character_,
        paste(trio_group,score_half,sep=" · ")))
}

federated_transfer_km_fit <- function(z, column, levels) {
  z <- z[z[[column]] %in% levels,,drop=FALSE]
  present <- levels[levels %in% z[[column]]]
  if(length(present)<2L || sum(z$event)<FEDERATED_TRANSFER_KM_MIN_EVENTS) return(NULL)
  curves <- bind_rows(lapply(present,function(group) {
    fit <- survival::survfit(survival::Surv(duration_days,event)~1,data=z[z[[column]]==group,])
    tibble(stratum=group,time=c(0,fit$time),survival=c(1,fit$surv),lower=c(1,fit$lower),upper=c(1,fit$upper))
  }))
  z$stratum <- z[[column]]
  logrank <- tryCatch({
    test <- survival::survdiff(survival::Surv(duration_days,event)~stratum,data=z)
    pchisq(test$chisq,length(test$n)-1,lower.tail=FALSE)
  },error=function(e) NA_real_)
  counts <- z %>% group_by(stratum) %>% summarise(n=n(),events=sum(event),.groups="drop") %>%
    arrange(match(stratum,levels)) %>% mutate(logrank_p=logrank)
  list(curves=curves,counts=counts)
}

federated_transfer_event_free <- function(z, horizon=FEDERATED_TRANSFER_KM_HORIZON) {
  none <- c(event_free=NA_real_,lo=NA_real_,hi=NA_real_)
  if(!nrow(z)) return(none)
  fit <- survival::survfit(survival::Surv(duration_days,event)~1,data=z)
  # Follow-up ending before the horizon is unestimable unless every patient had the event.
  if(max(z$duration_days)<horizon && utils::tail(fit$surv,1)>0) return(none)
  s <- summary(fit,times=horizon,extend=TRUE)
  if(length(s$surv)!=1L) return(none)
  c(event_free=s$surv,lo=s$lower,hi=s$upper)
}

# Aggregate tables for every endpoint x bundle x landmark: KM curves and
# per-stratum counts with the log-rank p, and event-free probabilities at
# FEDERATED_TRANSFER_KM_HORIZON.
federated_transfer_km_tables <- function(km) {
  d <- federated_transfer_km_groups(km)
  keys <- distinct(d,endpoint,bundle,model_family,landmark_days) %>%
    arrange(match(endpoint,names(FEDERATED_TRANSFER_ENDPOINTS)),model_family,landmark_days)
  out <- list(curves=list(),summary=list(),gradient=list())
  for(i in seq_len(nrow(keys))) {
    key <- keys[i,]
    cell <- semi_join(d,key,by=names(key))
    tag <- function(x,...) bind_cols(key[rep(1L,nrow(x)),],tibble(...,.rows=nrow(x)),x)
    for(view in names(FEDERATED_TRANSFER_KM_VIEWS)) {
      v <- FEDERATED_TRANSFER_KM_VIEWS[[view]]
      z <- cell[cell$clock==v$clock,,drop=FALSE]
      fit <- federated_transfer_km_fit(z,v$column,v$levels)
      if(!is.null(fit)) {
        out$curves[[length(out$curves)+1]] <- tag(fit$curves,view=view,clock=v$clock)
        out$summary[[length(out$summary)+1]] <- tag(fit$counts,view=view,clock=v$clock)
      }
    }
    for(stratifier in names(FEDERATED_TRANSFER_GRADIENT)) {
      g <- FEDERATED_TRANSFER_GRADIENT[[stratifier]]
      z <- cell[cell$clock==g$clock,,drop=FALSE]
      rows <- bind_rows(lapply(g$levels,function(group) {
        zz <- z[z[[stratifier]] %in% group,,drop=FALSE]
        if(!nrow(zz)) return(NULL)
        tibble(group=group,n=nrow(zz),events=sum(zz$event),!!!as.list(federated_transfer_event_free(zz)))
      }))
      if(nrow(rows)) out$gradient[[length(out$gradient)+1]] <- tag(rows,stratifier=stratifier,
        stratifier_label=g$label,clock=g$clock,horizon_days=FEDERATED_TRANSFER_KM_HORIZON)
    }
  }
  lapply(out,bind_rows)
}

federated_transfer_km_pick <- function(d, endpoint=NULL, model_family, landmark_days, view=NULL) {
  keep <- d$model_family==model_family & d$landmark_days==landmark_days
  if(!is.null(endpoint)) keep <- keep & d$endpoint==endpoint
  if(!is.null(view)) keep <- keep & d$view==view
  d[keep,,drop=FALSE]
}

federated_transfer_km_colors <- function(levels)
  setNames(if(length(levels)==4L) c("#92c5de","#2166ac","#f4a582","#b2182b") else c("#2166ac","#b2182b"),levels)

federated_transfer_model_label <- function(model_family)
  coalesce(unname(FEDERATED_TRANSFER_MODELS[model_family]),model_family)

plot_federated_transfer_km <- function(tables, view, endpoint, model_family, landmark, title=NULL, note=NULL) {
  v <- FEDERATED_TRANSFER_KM_VIEWS[[view]]
  curves <- federated_transfer_km_pick(tables$curves,endpoint,model_family,landmark,view)
  counts <- federated_transfer_km_pick(tables$summary,endpoint,model_family,landmark,view)
  if(!nrow(curves) || !nrow(counts)) return(NULL)
  if(is.null(title)) title <- paste0(FEDERATED_TRANSFER_ENDPOINTS[[endpoint]],": ",v$label)
  plot_km_curves(select(curves,stratum,time,survival,lower,upper),select(counts,stratum,n,events),
    counts$logrank_p[[1]],title,sprintf("the +%d-day %s landmark",as.integer(landmark),
      FEDERATED_TRANSFER_KM_CLOCKS[[v$clock]]),v$levels,note,federated_transfer_km_colors(v$levels),
    ci=length(v$levels)==2L,ylab=FEDERATED_TRANSFER_KM_YLAB[[endpoint]],
    legend_ncol=if(length(v$levels)==4L) 2 else 1,zoom_y=TRUE)
}

# Head-to-head: rows are endpoints; columns the score (ADT clock, all scored
# patients), Gleason (ADT clock, patients with Gleason) and TP53/RB1/PTEN
# (sequencing clock, sequenced patients). Each column is its own cohort.
plot_federated_transfer_km_grid <- function(tables, model_family, landmark) {
  cells <- list()
  for(endpoint in names(FEDERATED_TRANSFER_ENDPOINTS)) for(view in c("score","gleason","trio")) {
    p <- plot_federated_transfer_km(tables,view,endpoint,model_family,landmark,
      title=paste0(FEDERATED_TRANSFER_ENDPOINTS[[endpoint]],": ",FEDERATED_TRANSFER_KM_VIEWS[[view]]$label))
    cells[[length(cells)+1]] <- if(is.null(p)) grid::nullGrob() else p
  }
  if(!any(vapply(cells,inherits,logical(1),"ggplot"))) return(NULL)
  gridExtra::arrangeGrob(grobs=cells,ncol=3,top=grid::textGrob(sprintf(
    "%s trained without Dana-Farber: Dana-Farber event-free survival from the +%d-day landmark",
    federated_transfer_model_label(model_family),as.integer(landmark)),gp=grid::gpar(fontsize=15,fontface="bold")),
    bottom=grid::textGrob(paste(
      "Score: top vs bottom 20% of the transferred labs score (Dana-Farber cutpoints within each endpoint), ADT clock.",
      "Gleason ≤7 vs ≥8 (available before the landmark), ADT clock. TP53/RB1/PTEN: any alteration vs none, sequenced patients, sequencing clock.",
      sep="\n"),gp=grid::gpar(fontsize=10)),padding=grid::unit(1,"lines"))
}

plot_federated_transfer_gradient <- function(tables, model_family, landmark, title=NULL, caption=NULL) {
  d <- federated_transfer_km_pick(tables$gradient,NULL,model_family,landmark)
  d <- d[is.finite(d$event_free),,drop=FALSE]
  if(!nrow(d)) return(NULL)
  groups <- unlist(lapply(FEDERATED_TRANSFER_GRADIENT,`[[`,"levels"),use.names=FALSE)
  labels <- vapply(FEDERATED_TRANSFER_GRADIENT,`[[`,character(1),"label")
  d <- mutate(d,group=factor(group,levels=unique(groups)),
    stratifier_label=factor(stratifier_label,levels=labels),
    endpoint_label=factor(FEDERATED_TRANSFER_ENDPOINTS[endpoint],levels=FEDERATED_TRANSFER_ENDPOINTS))
  dodge <- position_dodge(width=.5)
  ggplot(d,aes(group,event_free,color=endpoint_label,group=endpoint_label)) +
    geom_line(data=function(x) x[x$stratifier!="trio_pattern",,drop=FALSE],
      position=dodge,linewidth=.6,alpha=.6) +
    geom_linerange(aes(ymin=lo,ymax=hi),position=dodge,linewidth=.7,na.rm=TRUE) +
    geom_point(position=dodge,size=2.4) +
    scale_color_manual(values=FEDERATED_TRANSFER_ENDPOINT_COLORS,name=NULL,drop=TRUE) +
    scale_x_discrete(labels=function(x) sub(" altered$","\\naltered",x)) +
    facet_grid(.~stratifier_label,scales="free_x",space="free_x") +
    labs(x=NULL,y=paste(federated_transfer_horizon_label(),"event-free probability"),
      title=title,caption=caption) +
    federated_transfer_theme()
}

FEDERATED_FAMILY_SLUGS <- c(elastic_net_cox="elasticnet",xgboost_cox="xgboost")
federated_transfer_family_slug <- function(model_family)
  if(model_family %in% names(FEDERATED_FAMILY_SLUGS)) FEDERATED_FAMILY_SLUGS[[model_family]] else
    gsub("[^a-z0-9]+","",tolower(model_family))

render_federated_transfer_km <- function(root, directory, dpi, overwrite) {
  km <- load_federated_transfer_km(directory)
  if(is.null(km) || !nrow(km)) return(invisible(NULL))
  tables <- federated_transfer_km_tables(km)
  for(name in names(FEDERATED_TRANSFER_KM_TABLES))
    write_federated_transfer_table(root,tables[[name]],FEDERATED_TRANSFER_KM_TABLES[[name]])
  save <- function(p, stem, width, height)
    if(!is.null(p)) save_federated_panel(p,federated_output_path(root,stem,".png"),width,height,dpi,overwrite)
  cutoffs <- "Score cutpoints: Dana-Farber patients at risk for this endpoint at this landmark (not federated training cutoffs)."
  cells <- distinct(bind_rows(tables$summary,tables$gradient),model_family,landmark_days)
  for(i in seq_len(nrow(cells))) {
    family <- cells$model_family[[i]]; landmark <- cells$landmark_days[[i]]
    model <- federated_transfer_model_label(family)
    suffix <- sprintf("%s_landmark%d",federated_transfer_family_slug(family),as.integer(landmark))
    horizon <- federated_transfer_horizon_label()
    for(endpoint in names(FEDERATED_TRANSFER_ENDPOINTS)) {
      save(plot_federated_transfer_km(tables,"score_trio",endpoint,family,landmark,
          title=paste0(FEDERATED_TRANSFER_ENDPOINTS[[endpoint]],": ",model," score × TP53/RB1/PTEN"),
          note=paste(cutoffs,"Low/high score: at or below / above the median.",
            "Sequenced patients; landmark from the sequencing date.")),
        sprintf("transfer_km_score_trio_%s_%s",endpoint,suffix),9,7)
      save(plot_federated_transfer_km(tables,"score_decile",endpoint,family,landmark,
          title=paste0(FEDERATED_TRANSFER_ENDPOINTS[[endpoint]],": ",model," score, top vs bottom 10%"),
          note=paste(cutoffs,"ADT clock.")),
        sprintf("transfer_km_score_decile_%s_%s",endpoint,suffix),9,7)
    }
    save(plot_federated_transfer_km_grid(tables,family,landmark),sprintf("transfer_km_grid_%s",suffix),16,15)
    save(plot_federated_transfer_gradient(tables,family,landmark,
        title=sprintf("%s: observed %s event-free probability from the +%d-day landmark",model,horizon,as.integer(landmark)),
        caption=paste(cutoffs,"Kaplan-Meier estimates with 95% CIs; score quartiles and Gleason on the ADT clock,",
          sprintf("Six exclusive TP53/PTEN/RB1 alteration groups on the sequencing clock (sequenced patients). Missing points: no follow-up to %s.",
            sub("-year"," years",horizon)))),
      sprintf("transfer_km_gradient_%s",suffix),12,5.5)
  }
  invisible(tables)
}

# Aggregate KM curves and binned trajectories exported by the federated run
# (rhino_data_v2/figure_data): pooled across sites and within each site,
# no patient-level rows. Drawn with the local plotters so the panels match.
FEDERATED_FIGURE_DATA_FILES <- c(manifest="manifest.json",cohort_km="cohort_km.csv",
  km="lab_extreme_km.csv",km_summary="lab_extreme_summary.csv",
  trajectories="lab_trajectories.csv",trajectory_cohort="lab_trajectory_cohort.csv",
  within_km="within_site_lab_extreme_km.csv",within_km_summary="within_site_lab_extreme_summary.csv",
  within_trajectories="within_site_lab_trajectories.csv",
  within_trajectory_cohort="within_site_lab_trajectory_cohort.csv")
FEDERATED_FIGURE_DATA_COLUMNS <- list(
  cohort_km=c("landmark_days","time","n_risk","events","survival","lower","upper"),
  km=c("cutoff_scope","landmark_days","lab","stratum","time","survival","lower","upper"),
  km_summary=c("cutoff_scope","landmark_days","lab","status",
    "n_bottom","n_top","events_bottom","events_top","logrank_p"),
  trajectories=c("lab_name","scale","platinum_status","t_mid","n_patients","mean","ci_lower","ci_upper"),
  trajectory_cohort=c("lab_name","scale","platinum_status","n_patients"))
FEDERATED_FIGURE_LABS <- c("PSA","Testosterone")

# Schema v3 exports carry q_low/q_high and extreme_fraction; v2 exports were
# quintile-only (q20/q80), so their fraction is 0.2.
federated_km_summary_cutoffs <- function(d, fraction=NULL) {
  if(!all(c("q_low","q_high") %in% names(d))) {
    if(!all(c("q20","q80") %in% names(d)))
      stop("Federated KM summary needs q_low/q_high (or legacy q20/q80) cutoffs")
    d <- rename(d,q_low=q20,q_high=q80)
    fraction <- 0.2
  }
  if(!"extreme_fraction" %in% names(d)) d$extreme_fraction <- if(is.null(fraction)) NA_real_ else fraction
  d$extreme_fraction <- as.numeric(d$extreme_fraction)
  if(nrow(d) && anyNA(d$extreme_fraction)) stop("Federated KM summary has no extreme_fraction")
  d
}

load_federated_figure_data <- function(directory) {
  paths <- setNames(file.path(directory,FEDERATED_FIGURE_DATA_FILES),names(FEDERATED_FIGURE_DATA_FILES))
  if(!file.exists(paths[["manifest"]])) {
    warning("Federated KM/trajectory data unavailable; missing: ",paths[["manifest"]])
    return(NULL)
  }
  manifest <- jsonlite::read_json(paths[["manifest"]],simplifyVector=FALSE)
  fraction <- manifest$config$extreme_fraction
  read <- function(name) {
    kind <- sub("^within_","",name)
    needed <- c("scope","site_name",FEDERATED_FIGURE_DATA_COLUMNS[[kind]])
    if(!file.exists(paths[[name]])) {
      # An empty table with the expected columns: dependent panels are skipped.
      warning("Federated figure data unavailable; missing: ",paths[[name]])
      if(kind=="km_summary") needed <- c(needed,"q_low","q_high","extreme_fraction")
      return(as_tibble(setNames(rep(list(logical()),length(needed)),needed)))
    }
    d <- readr::read_csv(paths[[name]],show_col_types=FALSE,
      col_types=readr::cols(site_name=readr::col_character(),.default=readr::col_guess()))
    missing <- setdiff(needed,names(d))
    if(length(missing)) stop("Federated figure data ",basename(paths[[name]])," is missing: ",
      paste(missing,collapse=", "))
    if("analysis_label" %in% names(d)) d <- filter(d,tolower(analysis_label)=="adt")
    if("endpoint" %in% names(d)) d <- filter(d,tolower(endpoint)=="platinum")
    if(any(d$scope=="within_site" & is.na(d$site_name))) stop("Within-site figure data without a site name")
    if(kind=="km_summary") d <- federated_km_summary_cutoffs(d,fraction)
    d
  }
  tables <- lapply(setdiff(names(FEDERATED_FIGURE_DATA_FILES),"manifest"),read)
  names(tables) <- setdiff(names(FEDERATED_FIGURE_DATA_FILES),"manifest")
  join <- function(kind) bind_rows(tables[[kind]],tables[[paste0("within_",kind)]])
  config <- manifest$config
  list(config=list(pre_days=as.numeric(config$pre_days),post_days=as.numeric(config$post_days),
      bin_width=as.numeric(config$bin_width_days),min_bin_patients=as.numeric(config$min_bin_patients)),
    sites=as.character(unlist(manifest$sites)),site_stats=manifest$site_cohort_stats,
    cohort_km=tables$cohort_km,km=join("km"),km_summary=join("km_summary"),
    trajectories=join("trajectories"),trajectory_cohort=join("trajectory_cohort"))
}

# NULL site = the pooled (all-site) rows; never pool within-site rows here.
# The run-wide extreme percentage (one fraction per export).
federated_extreme_pct <- function(fig) {
  fraction <- unique(stats::na.omit(fig$km_summary$extreme_fraction))
  if(length(fraction)!=1L) fraction <- LAB_EXTREME_FRACTION
  round(100*fraction)
}

federated_figure_scope <- function(d, site=NULL) {
  if(is.null(site)) filter(d,scope=="pooled") else filter(d,scope=="within_site",site_name==site)
}
federated_figure_scope_label <- function(site=NULL)
  if(is.null(site)) "Federated (all sites)" else paste("Within site:",federated_site_label(site))

plot_federated_trajectory <- function(fig, lab, site=NULL, scale="log1p") {
  rows <- federated_figure_scope(fig$trajectories,site) %>%
    filter(lab_name==lab,.data$scale==.env$scale,n_patients>=fig$config$min_bin_patients,is.finite(mean))
  if(!nrow(rows)) return(NULL)
  binned <- transmute(rows,t_mid,mean,ci_lo=ci_lower,ci_hi=ci_upper,stratum=as.character(platinum_status))
  n <- federated_figure_scope(fig$trajectory_cohort,site) %>%
    filter(lab_name==lab,.data$scale==.env$scale) %>% pull(n_patients) %>% sum()
  plot_trajectory_ci(binned,lab,sprintf("%s trajectories by platinum status",lab),
    log_scale=scale=="log1p",pre_days=fig$config$pre_days,post_days=fig$config$post_days,
    bin_width=fig$config$bin_width,min_bin_patients=fig$config$min_bin_patients) +
    labs(subtitle=sprintf("%s · n=%s · %s scale",federated_figure_scope_label(site),
      scales::comma(n),scale))
}

# Within-site panels default to site-specific extreme cutoffs; the export also
# carries each site's patients split at the pooled cutoffs (cutoff_scope="pooled").
plot_federated_lab_km <- function(fig, lab, landmark, site=NULL,
                                  cutoff_scope=if(is.null(site)) "pooled" else "within_site") {
  pick <- function(d) federated_figure_scope(d,site) %>%
    filter(.data$lab==.env$lab,landmark_days==landmark,.data$cutoff_scope==.env$cutoff_scope)
  summary <- pick(fig$km_summary)
  curves <- pick(fig$km)
  if(nrow(summary)!=1L || !identical(summary$status,"ok") || !nrow(curves)) return(NULL)
  groups <- figure_extreme_labels(summary$extreme_fraction)
  pct <- round(100*summary$extreme_fraction)
  counts <- tibble(stratum=groups,n=c(summary$n_bottom,summary$n_top),
    events=c(summary$events_bottom,summary$events_top))
  note <- sprintf("%s. Pre-landmark mean; %s cutoffs: bottom %d%% at %.4g, top %d%% at %.4g (middle %d%% not shown). Shading: 95%% CI.",
    federated_figure_scope_label(site),if(cutoff_scope=="pooled") "pooled" else "site-specific",
    pct,summary$q_low,pct,summary$q_high,100-2L*pct)
  plot_km_curves(select(curves,stratum,time,survival,lower,upper),counts,summary$logrank_p,
    sprintf("%s bottom vs top %d%%: time to platinum",lab,pct),sprintf("the +%d-day treatment landmark",landmark),
    groups,note,zoom_y=TRUE)
}

plot_federated_cohort_km <- function(fig, site=NULL) {
  curves <- federated_figure_scope(fig$cohort_km,site)
  if(!nrow(curves)) return(NULL)
  curves <- mutate(curves,stratum=sprintf("Day %d landmark",as.integer(landmark_days)))
  counts <- curves %>% group_by(stratum) %>%
    summarise(n=n_risk[which.min(time)],events=sum(events),.groups="drop")
  plot_km_curves(select(curves,stratum,time,survival,lower,upper),counts,NA_real_,
    "Platinum-free survival by landmark","each treatment landmark",
    sprintf("Day %d landmark",c(0,90,180)),federated_figure_scope_label(site)) +
    labs(subtitle=NULL)
}

# Pooled univariable Cox results drawn as the local Figure 3 volcanoes, with the
# local display filters (finite estimates, |log HR| <= 4, CI ratio < 100).
federated_volcano_data <- function(federated_path) {
  d <- readr::read_csv(federated_path,show_col_types=FALSE)
  needed <- c("landmark_days","endpoint","lab_name","feature_stat","coef_feature",
    "ci_lower","ci_upper","p_value","q_value")
  missing <- setdiff(needed,names(d))
  if(length(missing)) stop("Federated univariable input is missing: ",paste(missing,collapse=", "))
  for(column in c("landmark_days","coef_feature","ci_lower","ci_upper","p_value","q_value"))
    d[[column]] <- suppressWarnings(as.numeric(d[[column]]))
  if("analysis_label" %in% names(d)) d <- filter(d,tolower(analysis_label)=="adt")
  d %>% filter(tolower(endpoint)=="platinum",landmark_days %in% c(0,90,180)) %>%
    mutate(raw_lab_name=lab_name,lab_name=federated_lab_label(lab_name)) %>%
    filter(is.finite(coef_feature),is.finite(p_value),is.finite(q_value),
      abs(coef_feature)<=4,is.finite(ci_upper/ci_lower),ci_upper/ci_lower<100)
}

# Within-site panels for one lab, one page with a panel per site.
federated_site_grid <- function(fig, make, title) {
  plots <- lapply(fig$sites,function(site) {
    p <- make(site)
    if(is.null(p)) return(NULL)
    p + labs(title=federated_site_label(site))
  })
  plots <- Filter(Negate(is.null),plots)
  if(!length(plots)) return(NULL)
  gridExtra::arrangeGrob(grobs=lapply(plots,ggplot2::ggplotGrob),ncol=2,
    top=grid::textGrob(title,gp=grid::gpar(fontsize=15,fontface="bold")),
    padding=grid::unit(1.5,"lines"))
}

render_federated_figure_data <- function(root, federated_path, figure_data_dir, dpi, overwrite) {
  manuscript <- getOption("compass.manuscript_capture")
  keep <- function(plot, stem) if(is.function(manuscript) && !is.null(plot)) manuscript(plot,stem)
  publish <- function(plot, stem, data, width, height) {
    if(is.null(plot)) return(invisible(FALSE))
    write_federated_transfer_table(root,data,stem)
    save_federated_panel(plot,federated_output_path(root,stem,".png"),width,height,dpi,overwrite)
  }
  volcano <- federated_volcano_data(federated_path)
  for(lm in c(0,90,180)) {
    sub <- filter(volcano,landmark_days==lm)
    if(!nrow(sub)) { warning("No federated univariable rows at landmark ",lm); next }
    p <- plot_volcano_panel(sub,if(lm>0) sprintf("+%d days",lm) else "0 days")
    publish(p,paste0("federated_volcano_landmark",lm),sub,9,7)
    keep(p,paste0("fed3_volcano_landmark",lm))
  }
  fig <- load_federated_figure_data(figure_data_dir)
  if(is.null(fig)) return(invisible(NULL))
  publish(plot_federated_cohort_km(fig),"federated_cohort_km",
    federated_figure_scope(fig$cohort_km),8,6.5)
  for(lab in FEDERATED_FIGURE_LABS) {
    slug <- figure_slug(lab)
    rows <- filter(fig$trajectories,lab_name==lab,scale=="log1p")
    p <- plot_federated_trajectory(fig,lab)
    publish(p,paste0("federated_trajectory_",slug),filter(rows,scope=="pooled"),9.5,5.5)
    keep(p,paste0("fed3_trajectory_",slug))
    publish(federated_site_grid(fig,function(site) plot_federated_trajectory(fig,lab,site),
        paste(lab,"trajectories by platinum status, within each site (log1p scale)")),
      paste0("federated_trajectory_",slug,"_by_site"),filter(rows,scope=="within_site"),19,11)
    for(lm in c(0,90,180)) {
      stem <- sprintf("federated_km_quintile_%s_landmark%d",slug,lm)
      curves <- filter(fig$km,lab==.env$lab,landmark_days==lm)
      p <- plot_federated_lab_km(fig,lab,lm)
      publish(p,stem,filter(curves,scope=="pooled"),8,6.5)
      if(lm==manuscript_dfci_km_landmark) keep(p,sprintf("fed3_km_quintile_%s_landmark%d",slug,lm))
      publish(federated_site_grid(fig,function(site) plot_federated_lab_km(fig,lab,lm,site),
          sprintf("%s bottom vs top %d%% within each site (site-specific cutoffs), +%d-day landmark",
            lab,federated_extreme_pct(fig),lm)),
        paste0(stem,"_by_site"),filter(curves,scope=="within_site"),16,13)
    }
  }
  if(nrow(fig$km_summary)) write_federated_transfer_table(root,fig$km_summary,"federated_km_quintile_summary")
  invisible(fig)
}

render_federated_supplement <- function(data_root, fig_root, federated_path,
                                                dpi = 200, overwrite = FALSE,
                                                transfer_no_dfci_dir = NULL, figure_data_dir = NULL) {
  d <- load_federated_forest(federated_path)
  # Arm root; each output is routed to its section by federated_output_path().
  root <- file.path(fig_root, "ADT")
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
    base <- federated_output_path(root,paste0(tolower(analyte),"_forest"))
    table <- figure_sidecar_path(paste0(base,".csv"))
    dir.create(dirname(table),recursive=TRUE,showWarnings=FALSE)
    readr::write_csv(filter(estimates,lab_name==analyte),table)
    if(is.function(capture)) capture(table)
    save_federated_panel(plot_federated_comparison(estimates,analyte,note),paste0(base,".png"),
      FEDERATED_FOREST_SLIDE_SIZE[["width"]],FEDERATED_FOREST_SLIDE_SIZE[["height"]],dpi,overwrite)
  }
  if (!is.null(sites)) {
    site_path <- figure_sidecar_path(federated_output_path(root, "site_incidence_lm000", ".csv"))
    dir.create(dirname(site_path), recursive = TRUE, showWarnings = FALSE)
    readr::write_csv(prepare_federated_site_incidence(sites), site_path)
    if (is.function(capture)) capture(site_path)
    save_federated_panel(plot_federated_sites(sites),
      federated_output_path(root, "site_incidence_lm000", ".png"), 14, 5.5, dpi, overwrite)
  }
  for(variant in names(FEDERATED_MULTIVARIABLE_VARIANTS)) for(family in names(FEDERATED_MULTIVARIABLE))
    render_federated_multivariable(root,federated_path,family,dpi,overwrite,variant)
  for(variant in names(FEDERATED_MULTIVARIABLE_VARIANTS))
    render_federated_deephit_performance(root,federated_path,dpi,overwrite,variant)
  # The four-site checklists; the no-DFCI refits appear in the transfer supplement.
  render_federated_checklist_cards(root,federated_path,dpi,overwrite)
  if (!is.null(figure_data_dir))
    render_federated_figure_data(root, federated_path, figure_data_dir, dpi, overwrite)
  if (!is.null(transfer_no_dfci_dir)) {
    no_dfci <- render_federated_transfer(root, transfer_no_dfci_dir, dpi, overwrite)
    if (!is.null(no_dfci)) {
      render_federated_transfer_clinical(root, transfer_no_dfci_dir, dpi, overwrite)
      render_federated_transfer_km(root, transfer_no_dfci_dir, dpi, overwrite)
    }
    render_federated_checklist_supplement(root, file.path(transfer_no_dfci_dir,"all_models"),dpi,overwrite)
    render_federated_all_model_transfer_supplement(root,
      file.path(transfer_no_dfci_dir,"all_models"),dpi,overwrite)
  }
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
MANUSCRIPT_COHORT_MEMBERS <- c("figure1a_consort","figure1b_km","figure1c_span",
  "figure1c_dx_to_tx","figure1c_time_to_platinum")
MANUSCRIPT_COHORT_TITLES <- c("Cohort selection","Platinum-free survival","Lab record span",
  "Diagnosis to ADT","ADT to platinum")
manuscript_specs <- function() {
  spec <- function(key, members, height, layout, titles, row_heights=NULL,
                   shared_legend_panels=integer(),width=7.2) list(key=key, members=members,
    width=width, height=height, layout=layout, titles=titles,
    row_heights=row_heights, shared_legend_panels=shared_legend_panels)
  c(list(
    spec("01_cohort_overview", c(MANUSCRIPT_COHORT_MEMBERS,"figure1d_dfci_data","figure1e_site_counts"), 9.4,
      rbind(c(1,1,2,2),c(1,1,3,3),c(4,4,5,5),c(6,6,7,7)),
      c(MANUSCRIPT_COHORT_TITLES,"Data availability (DFCI)","Federated sites: events/patients")),
    # Without the federated within-site export, Figure 1 keeps the DFCI panels.
    spec("01_cohort_overview", c(MANUSCRIPT_COHORT_MEMBERS,"figure1d_dfci_data"), 9.4,
      rbind(c(1,1,2,2),c(1,1,3,3),c(4,4,5,5),c(6,6,6,6)),
      c(MANUSCRIPT_COHORT_TITLES,"Data availability (DFCI)"))),
    # Figure 2 per cohort; manuscript_build adds the cohort header.
    lapply(names(FIGURE2_COHORTS),function(variant) spec(paste0("02_llm_nepc_avpc__",variant),
      figure2v3_stems(variant), 6.4, rbind(1:2,3:4),
      c("Subtype: chart-review agreement","Subtype classification metrics","NEPC/AVPC proxy subtypes","Platinum enrichment"))),
    list(
    spec("03_dfci_univariable", figure_dfci_overview_spec(manuscript_dfci_km_landmark)$members, 8.9,
      rbind(c(1,1,2,2,3,3),rep(8,6),c(4,4,4,5,5,5),c(6,6,6,7,7,7)),
      c("Day 0","Day 90","Day 180","PSA trajectory","PSA: extreme deciles",
        "Testosterone trajectory","Testosterone: extreme deciles"),
      row_heights=c(1.5,.32,1,1), shared_legend_panels=1:3),
    spec("04_multivariable_labs", c(
      "figure4a_discrimination_auc_platinum",
      paste0("figure4b_importance_platinum_cox_landmark",c(0,90,180)),
      paste0("figure4b_importance_platinum_xgb_landmark",c(0,90,180))),
      9, matrix(1:3,3), c("Mean AUC(t)","Elastic-Net coefficients","XGBoost feature importance"),
      row_heights=c(.9,1,1),width=8.5),
    # Federated Figure 3: the local Figure 3 layout drawn from pooled federated results.
    spec("federated_03_univariable", c(paste0("fed3_volcano_landmark",c(0,90,180)),
      "fed3_trajectory_psa",paste0("fed3_km_quintile_psa_landmark",manuscript_dfci_km_landmark),
      "fed3_trajectory_testosterone",paste0("fed3_km_quintile_testosterone_landmark",manuscript_dfci_km_landmark)), 8.9,
      rbind(c(1,1,2,2,3,3),rep(8,6),c(4,4,4,5,5,5),c(6,6,6,7,7,7)),
      c("Day 0","Day 90","Day 180","PSA trajectory","PSA: extreme deciles",
        "Testosterone trajectory","Testosterone: extreme deciles"),
      row_heights=c(1.5,.32,1,1), shared_legend_panels=1:3)))
}

# Keys sharing the Figure 3 layout and restyling (manuscript_univariable_restyle).
MANUSCRIPT_UNIVARIABLE_KEYS <- c("03_dfci_univariable","federated_03_univariable")

# Reading guide under the volcano legend: log HR per SD > 0 means higher risk.
hr_direction_grob <- function(fontsize=7, col="#52514e") {
  gp <- grid::gpar(fontsize=fontsize,col=col,fontfamily="sans")
  line <- grid::gpar(col=col,fill=col,lwd=.8)
  y <- grid::unit(.68,"npc"); tick <- grid::unit(3,"pt")
  grid::gTree(name="hr_direction",children=grid::gList(
    grid::segmentsGrob(.5,y,c(.22,.78),y,arrow=grid::arrow(length=grid::unit(4,"pt"),type="closed"),gp=line),
    grid::segmentsGrob(.5,y-tick,.5,y+tick,gp=line),
    grid::textGrob("0",.5,y+tick+grid::unit(1,"pt"),vjust=0,gp=gp),
    grid::textGrob("Lower risk / slower time to platinum",.48,y-tick-grid::unit(2,"pt"),
      hjust=1,vjust=1,gp=gp),
    grid::textGrob("Higher risk / faster time to platinum",.52,y-tick-grid::unit(2,"pt"),
      hjust=0,vjust=1,gp=gp)))
}

# Federated Figure 5 with (no_d=FALSE) or without the sequenced-with-Gleason
# panel; later panel letters shift by one without it.
manuscript_transfer_caption <- function(no_d=FALSE) {
  caption <- function(...) paste(c(...),collapse="\n\n")
  clinical <- if(no_d) "(b,c)" else "(b-d)"
  km <- if(no_d) c("d","e","f") else c("e","f","g")
  gradient <- if(no_d) "g" else "h"
  seq_panels <- if(no_d) "c" else "c and d"
  caption(
    paste0(if(no_d) "Federated Figure 5 (without the sequenced-with-Gleason panel). " else "Federated Figure 5. ",
      "External validation at Dana-Farber of federated platinum models trained without Dana-Farber, alone and combined with Gleason score and TP53/RB1/PTEN status, and time-to-event risk separation by the transferred score."),
    paste0("Elastic-Net Cox and XGBoost survival models trained for platinum treatment on Fred Hutch, Johns Hopkins, and Memorial Sloan Kettering data were applied unchanged to the Dana-Farber ADT cohort and evaluated against three local endpoints: platinum treatment (the training endpoint), aggressive-variant prostate cancer (AVPC), and neuroendocrine prostate cancer (NEPC). (a) Mean time-dependent area under the receiver-operating-characteristic curve, AUC(t), of the full laboratory models at 0, 90, and 180 days after ADT initiation. ",
      clinical," The same full laboratory models compared with unpenalized Cox models fitted locally at Dana-Farber on clinical covariates alone and on the transferred score plus the same covariates other than age (combined models), among patients with a Gleason score available before the landmark (b) and patients with tumor sequencing, with landmarks measured from the sequencing date (c)",
      if(no_d) "" else ", and sequenced patients who also had a Gleason score (d)",
      ". The clinical covariates were Gleason score and age (b) or any alteration (mutation, deletion, amplification, or structural variant) in TP53, RB1, and PTEN and age (c)",
      if(no_d) "" else ", or all of these (d)",
      "; because the transferred full models already include age, the combined models added only Gleason score (b) or the TP53, RB1, and PTEN indicators (c)",
      if(no_d) "" else ", or both (d)",
      ". Circles denote the transferred models, diamonds the local clinical Cox models, and triangles the combined models. In ",
      if(no_d) "a-c" else "a-d",", bars denote 95% bootstrap confidence intervals over patients, dotted lines mark 0.5, and column strips give the day-0 patients and events for each endpoint. ",
      sprintf("(%s-%s) Kaplan-Meier platinum-free (%s), AVPC-free (%s), and NEPC-free (%s) survival from the 180-day landmark after ADT initiation comparing the top and bottom 10%% of the transferred XGBoost score; the middle 80%% is omitted, shading denotes pointwise 95%% confidence intervals, P values are from log-rank tests, and the vertical axes are truncated below 1. ",
        km[1],km[3],km[1],km[2],km[3]),
      sprintf("(%s) Kaplan-Meier 5-year event-free probability (95%% confidence interval) from the 180-day landmark for each endpoint by transferred XGBoost score quartile, Gleason group (≤7 vs. ≥8), and six mutually exclusive TP53, PTEN, and RB1 alteration groups: none, TP53 only, PTEN only, RB1 only, any two, or all three.",gradient)),
    paste0("No Dana-Farber patient contributed to federated model training, so every endpoint, including platinum, is out of sample. Mean AUC(t) is the inverse-probability-of-censoring-weighted cumulative/dynamic AUC averaged over 1, 2, 3, and 5 years after the landmark, with censoring estimated in the evaluated Dana-Farber patients; horizons without an observed event or later follow-up were omitted. ",
      sprintf("Score groups in %s-%s use cutpoints computed among the Dana-Farber patients at risk for each endpoint at the landmark; they are not federated training cutoffs. Score and Gleason groups use the ADT-initiation clock and TP53/RB1/PTEN groups the sequencing-date clock (sequenced patients). ",km[1],gradient),
      "Elastic-Net versions, the 0- and 90-day landmarks, score × TP53/RB1/PTEN Kaplan-Meier curves, and head-to-head Kaplan-Meier grids are provided as supplementary figures. The local and combined Cox models were fitted on the Dana-Farber patients they were evaluated on and were scored with stratified 5-fold cross-validated (out-of-fold) predictions; in the combined models, the transferred score was standardized within the evaluated patients. Gleason scores were used only if available before the landmark. ",
      sprintf("For %s, the transferred models were rescored on laboratory values before the sequencing date plus the landmark, outcomes were measured from that point, the sequencing date was the report date of the tumor sequencing test reported closest to ADT initiation, and age was age at sequencing. ",seq_panels),
      "Patients/events in panel a were 2,267/182 at day 0, 2,233/166 at day 90, and 2,197/158 at day 180 for platinum; 2,237/197, 2,208/181, and 2,165/171 for AVPC; and 2,231/50, 2,212/46, and 2,172/41 for NEPC. ADT, androgen-deprivation therapy; AUC(t), time-dependent area under the receiver-operating-characteristic curve."))
}

manuscript_figure2_caption <- function(variant) {
  cohort <- c(icd="all patients with an ICD-10 prostate cancer (C61) diagnosis",
    adt="all patients with androgen-deprivation therapy (ADT) on or after prostate cancer diagnosis",
    lm0="the ADT day-0 prediction cohort")[[variant]]
  paste(c(
    paste0("Figure 2. Large language model classification of neuroendocrine and aggressive-variant prostate cancer proxies: ",
      FIGURE2_COHORTS[[variant]]," cohort."),
    paste0("The cohort is ",cohort,", restricted to patients with a large language model (LLM) subtype label; the header gives its size. ",
      "(a) Confusion matrix comparing LLM subtype labels (conventional, aggressive-variant prostate cancer [AVPC], and neuroendocrine prostate cancer [NEPC]) on the x axis with manual chart-review annotations on the y axis, among chart-reviewed patients in this cohort. ",
      "(b) One-vs-rest accuracy, precision, recall, and specificity for each subtype in the chart-reviewed set. ",
      "(c) Distribution of conventional, AVPC, and NEPC subtypes within platinum-positive and platinum-negative groups; platinum-positive means any dated platinum treatment in the record. ",
      "(d) Observed platinum-positive fraction for aggressive subtypes, defined as AVPC or NEPC, versus conventional prostate cancer; error bars denote 95% Wilson confidence intervals. The displayed odds ratio and one-sided Fisher exact-test P value compare the aggressive and conventional groups."),
    "Patients without a resolved subtype label were excluded from panels c and d; panel counts are given in each panel note. OR, odds ratio."),
    collapse="\n\n")
}

manuscript_captions <- function() {
  caption <- function(...) paste(c(...),collapse="\n\n")
  c(
    `01_cohort_overview`=caption(
      "Figure 1. Cohort assembly, follow-up, and platinum treatment after androgen-deprivation therapy.",
      "(a) Cumulative selection of the PROFILE prostate cancer cohort, from patients identified by International Classification of Diseases codes through the final laboratory-analysis cohort. (b) Kaplan-Meier estimate of platinum-free survival from androgen-deprivation therapy (ADT) initiation; shading denotes the pointwise 95% confidence interval. (c) Distribution of each patient's laboratory-record span. (d) Distribution of time from prostate cancer diagnosis to ADT initiation, displayed through the 99th percentile. (e) Distribution of time from ADT initiation to platinum treatment among patients who received platinum. (f) DFCI patients with each data type: ICD-10 prostate cancer, ADT exposure, Gleason grade recorded before ADT initiation, tumor sequencing, and the ADT day-0 prediction cohort; Gleason and sequencing counts are restricted to the day-0 cohort, and sequenced patients must be at risk at the sequencing report. (g) Observed platinum events and analyzed patients per federated site in the ADT cohort at day 0; pale bars show patients, solid bars show events, labels are events/patients, and a dagger marks sites with fewer than 25 events.",
      "Panels b and c include 2,239 patients. Panel d displays 2,216 of 2,239 patients, and panel e displays 162 of 198 platinum-treated patients. ADT, androgen-deprivation therapy; PARPi, poly(ADP-ribose) polymerase inhibitor; PSA, prostate-specific antigen."),
    `02_llm_nepc_avpc__icd`=manuscript_figure2_caption("icd"),
    `02_llm_nepc_avpc__adt`=manuscript_figure2_caption("adt"),
    `02_llm_nepc_avpc__lm0`=manuscript_figure2_caption("lm0"),
    `03_dfci_univariable`=caption(
      "Figure 3. Univariable laboratory associations with platinum treatment and longitudinal PSA and testosterone patterns.",
      "(a-c) Univariable Cox associations at ADT initiation (a), 90 days (b), and 180 days (c). The horizontal axis gives the log hazard ratio per standard-deviation increase and the vertical axis gives -log10(P), displayed through 20. The solid vertical line denotes no association, dashed vertical lines mark log hazard ratios of -0.5 and 0.5, and the dotted horizontal line is the landmark-specific threshold corresponding to a Benjamini-Hochberg false-discovery rate below 0.05. Colors identify laboratory categories. The arrow beneath the color key marks direction: estimates to the right of 0 indicate higher risk and faster time to platinum, and estimates to the left indicate lower risk and slower time to platinum. At the three landmarks, 31 of 235, 72 of 244, and 76 of 245 tested features, respectively, had q < 0.05. (d,f) Mean log1p-transformed PSA (d) and testosterone (f) trajectories by subsequent platinum status; points and shading denote the mean and 95% confidence interval, and only bins containing at least 10 patients per stratum are shown. The dotted vertical line denotes ADT initiation. (e,g) Kaplan-Meier platinum-free survival from the 180-day landmark comparing the bottom and top 10% of the pre-landmark mean PSA (e) or testosterone (g) distribution. The middle 80% is omitted, equal values remain together, shading denotes pointwise 95% confidence intervals, and P values are from log-rank tests. Because platinum treatment is rare, the vertical axis is truncated below 1 to show separation between curves.",
      "The trajectory analyses include 2,217 patients for PSA and 2,059 for testosterone. For PSA, the bottom group was ≤0.2453 (n=403; 11 events) and the top group was >9.626 (n=403; 58 events); log-rank P=2.75 × 10^-13. For testosterone, the bottom group was ≤10 (n=317; 41 events) and the top group was >294.3 (n=313; 19 events); log-rank P=6.92 × 10^-6. ADT, androgen-deprivation therapy; CBC, complete blood count; CMP, comprehensive metabolic panel; HR, hazard ratio; LFT, liver-function test; PSA, prostate-specific antigen; SD, standard deviation."),
    `04_multivariable_labs`=caption(
      "Figure 4. Multivariable laboratory models for prediction of platinum treatment.",
      "(a) Held-out test mean time-dependent area under the receiver-operating-characteristic curve, AUC(t), for Elastic-Net Cox and XGBoost survival models and, when available, dynamic DeepHit at 0, 90, and 180 days after ADT initiation. Laboratory models are compared with age-only baselines; the dotted horizontal line marks 0.5. (b) Nonzero Elastic-Net Cox coefficients at each landmark. Positive coefficients indicate higher platinum hazard and negative coefficients indicate lower hazard, conditional on the other selected features. (c) XGBoost split-gain feature importance at each landmark. Colors in b and c identify laboratory categories.",
      "Performance values are from held-out test sets rather than training or tuning cross-validation. Elastic-Net coefficients are signed log hazard-ratio coefficients. XGBoost gains are unsigned measures of split improvement and do not encode effect direction. When available, dynamic DeepHit is a neural cause-specific model with no feature-level coefficients, so it appears in a only; it has no age-only baseline and is scored with the same inverse-probability-of-censoring-weighted horizon grid. ADT, androgen-deprivation therapy; AUC(t), time-dependent area under the receiver-operating-characteristic curve; PSA, prostate-specific antigen."),
    `06_dfci_cohort_sensitivity`=caption(
      "Figure 6. Cohort-size, endpoint-incidence, and laboratory-association sensitivity analyses within Dana-Farber.",
      "Patients eligible at the 180-day androgen-deprivation therapy (ADT) landmark are shown across the full ADT cohort, cohorts excluding prior castration, metastatic cohorts defined from ADT treatment intent, and metastatic cohorts defined by the large language model (LLM). (a) Observed platinum and neuroendocrine prostate cancer (NEPC) event fractions during follow-up with 95% Wilson confidence intervals. (b) Pale bars show eligible patients and solid bars show observed events; labels give events/patients. Blue denotes platinum and orange denotes NEPC in a and b; an asterisk denotes fewer than 25 events. (c) Natural-log hazard ratios per standard-deviation increase and 95% confidence intervals for the mean, minimum, maximum, and last PSA or testosterone value through the landmark. The dashed line at 0 marks the null; filled points indicate Benjamini-Hochberg q < 0.05.",
      "Cohort definitions overlap. Event fractions are observed proportions rather than censoring-adjusted cumulative incidence estimates. Change-from-baseline and observation-count features are excluded from c. 'No prior castrate' denotes the source no-previous-castration restriction. CI, confidence interval; PSA, prostate-specific antigen; SD, standard deviation."),
    `07_federated_incidence_associations`=caption(
      "Figure 7. Site-specific platinum incidence and federated PSA and testosterone associations.",
      "The Dana-Farber, Fred Hutch, and Johns Hopkins ADT cohorts are shown at treatment initiation. (a) Observed platinum-event fractions during follow-up with 95% Wilson confidence intervals. (b) Pale bars show analyzed patients and solid bars show observed platinum events; labels give events/patients. (c,d) Natural-log hazard ratios per standard-deviation increase and 95% confidence intervals for PSA (c) and testosterone (d) at the three sites with within-site results and in the supplied pooled federated analysis, which also includes Memorial Sloan Kettering (MSK). Columns show the mean, minimum, maximum, and last observed value; rows show landmarks at 0, 90, and 180 days. Dashed lines at 0 mark the null. Open circles indicate nonsignificant associations, filled circles indicate nominal P < 0.05 without false-discovery-rate significance, and diamonds indicate supplied q < 0.05.",
      "Site-specific fractions are observed event proportions rather than censoring-adjusted cumulative incidence estimates. For PSA, federated modeled/observed sample sizes were 22,138/16,840 at day 0, 21,791/20,707 at day 90, and 21,216/20,643 at day 180. Corresponding testosterone sample sizes were 22,138/9,483, 21,791/14,206, and 21,216/14,359. Federated patient/event counts exceed the combined Dana-Farber, Fred Hutch, and Johns Hopkins counts by 9,939/308 at day 0, 9,741/277 at day 90, and 9,395/252 at day 180, attributed to MSK, which supplied no within-site results; site membership was not independently verified. Supplied q values were not recalculated. ADT, androgen-deprivation therapy; PSA, prostate-specific antigen; SD, standard deviation."),
    `08_federated_multivariable`=caption(
      "Figure 8. Federated multivariable laboratory models for prediction of platinum treatment.",
      "(a) Held-out test mean time-dependent area under the receiver-operating-characteristic curve, AUC(t), for federated Elastic-Net Cox and XGBoost survival models at 0, 90, and 180 days after ADT initiation. Laboratory models are compared with age-only baselines; the dotted horizontal line marks 0.5. (b) Nonzero Elastic-Net Cox coefficients at each landmark, with up to 15 coefficients of largest absolute value displayed. Positive coefficients indicate higher platinum hazard and negative coefficients indicate lower hazard, conditional on the other selected features. (c) XGBoost split-gain feature importance at each landmark, with up to 15 positive-gain features displayed. Colors in b and c identify laboratory categories.",
      "Models were trained on pooled Dana-Farber, Fred Hutch, Johns Hopkins, and Memorial Sloan Kettering data. Test-set sample sizes/events were 4,429/133 at day 0, 4,360/121 at day 90, and 4,245/113 at day 180 for all model configurations. Performance values are from held-out test sets rather than training or tuning cross-validation. Age and body height are not displayed. Feature labels give the measurement and summary statistic; \"missing\" marks an indicator that the statistic was unavailable. Elastic-Net coefficients are signed log hazard-ratio coefficients. XGBoost gains are unsigned measures of split improvement and are not signed effects or SHAP values. abs., absolute; ADT, androgen-deprivation therapy; ALP, alkaline phosphatase; ALT, alanine aminotransferase; AST, aspartate aminotransferase; AUC(t), time-dependent area under the receiver-operating-characteristic curve; DBP, diastolic blood pressure; MCH, mean corpuscular hemoglobin; MCV, mean corpuscular volume; n obs., number of observations; PSA, prostate-specific antigen; PT, prothrombin time; RBC, red blood cell count; RDW, red-cell distribution width; SBP, systolic blood pressure; SHAP, Shapley additive explanations; TSH, thyroid-stimulating hormone; WBC, white blood cell count."),
    # The federated manuscript set is numbered separately (federated_figure<N>).
    federated_03_univariable=caption(
      "Federated Figure 3. Pooled univariable laboratory associations with platinum treatment and longitudinal PSA and testosterone patterns across four sites.",
      "Results pool the Dana-Farber, Fred Hutch, Johns Hopkins, and Memorial Sloan Kettering (MSK) ADT cohorts. (a-c) Federated univariable Cox associations at ADT initiation (a), 90 days (b), and 180 days (c). The horizontal axis gives the log hazard ratio per standard-deviation increase and the vertical axis gives -log10(P); values above 30 are drawn at 30 as triangles. The solid vertical line denotes no association, dashed vertical lines mark log hazard ratios of -0.5 and 0.5, and the dotted horizontal line is the landmark-specific threshold corresponding to a Benjamini-Hochberg false-discovery rate below 0.05. Colors identify laboratory categories. The arrow beneath the color key marks direction: estimates to the right of 0 indicate higher risk and faster time to platinum, and estimates to the left indicate lower risk and slower time to platinum. At the three landmarks, 93 of 261, 121 of 258, and 126 of 260 displayed features, respectively, had q < 0.05. (d,f) Pooled mean log1p-transformed PSA (d) and testosterone (f) trajectories by subsequent platinum status; points and shading denote the mean and 95% confidence interval, and only bins containing at least 10 patients per stratum are shown. The dotted vertical line denotes ADT initiation. (e,g) Pooled Kaplan-Meier platinum-free survival from the 180-day landmark comparing the bottom and top 10% of the pre-landmark mean PSA (e) or testosterone (g) distribution, with cutoffs computed across all four sites. The middle 80% is omitted, equal values remain together, shading denotes pointwise 95% confidence intervals, and P values are from log-rank tests. Because platinum treatment is rare, the vertical axis is truncated below 1 to show separation between curves.",
      "Panels a-c display features with finite estimates, absolute log hazard ratios of at most 4, and confidence-interval upper-to-lower ratios below 100; supplied q values were not recalculated. Panels d-g were drawn from aggregate site exports (binned means and Kaplan-Meier curves) without patient-level data. Cohorts included 22,138 patients (664 platinum events) at day 0, 21,791 (608) at day 90, and 21,216 (564) at day 180; by site at day 0, Dana-Farber contributed 5,853 (231 events), Fred Hutch 3,277 (90), Johns Hopkins 3,069 (35), and MSK 9,939 (308). The trajectory analyses include 22,035 patients for PSA and 17,937 for testosterone. Cutoffs, group sizes, events, and log-rank P values for e and g are reported in federated_km_quintile_summary.csv. Within-site versions are provided as supplementary figures. ADT, androgen-deprivation therapy; CBC, complete blood count; CMP, comprehensive metabolic panel; LFT, liver-function test; PSA, prostate-specific antigen; SD, standard deviation."),
    federated_04_multivariable=caption(
      "Federated Figure 4. Federated multivariable laboratory models for prediction of platinum treatment, with and without Dana-Farber training data.",
      "(a) Held-out test mean time-dependent area under the receiver-operating-characteristic curve, AUC(t), at 0, 90, and 180 days after ADT initiation for Elastic-Net Cox and XGBoost survival models trained on all four sites, the same full models refit without Dana-Farber (darker bars, \"no DFCI\"), and age-only baselines (hatched). When available, the dynamic DeepHit arm from each training-site set is also shown in a. The dotted horizontal line marks 0.5. (b) Nonzero Elastic-Net Cox coefficients of the four-site model at each landmark, with up to 15 coefficients of largest absolute value displayed. Positive coefficients indicate higher platinum hazard and negative coefficients indicate lower hazard, conditional on the other selected features. (c) XGBoost split-gain feature importance of the four-site model at each landmark, with up to 15 positive-gain features displayed. Colors in b and c identify laboratory categories.",
      "Four-site models were trained on pooled Dana-Farber, Fred Hutch, Johns Hopkins, and Memorial Sloan Kettering (MSK) data; test-set sample sizes/events were 4,429/133 at day 0, 4,360/121 at day 90, and 4,245/113 at day 180. Models without Dana-Farber were trained and tested on Fred Hutch, Johns Hopkins, and MSK data; test-set sample sizes/events were 3,258/87, 3,204/78, and 3,105/72. Because the test sets differ, the two model versions are not compared on the same patients here; Federated Figure 5 evaluates the models without Dana-Farber on Dana-Farber patients. Performance values are from held-out test sets rather than training or tuning cross-validation. Age and body height are not displayed. Elastic-Net coefficients are signed log hazard-ratio coefficients. XGBoost gains are unsigned measures of split improvement and are not signed effects or SHAP values. When shown, dynamic DeepHit has no age-only baseline or feature-level coefficients; its held-out metrics use the same horizon grid. ADT, androgen-deprivation therapy; AUC(t), time-dependent area under the receiver-operating-characteristic curve; DFCI, Dana-Farber Cancer Institute; SHAP, Shapley additive explanations."),
    federated_05_transfer_no_dfci=manuscript_transfer_caption(no_d=FALSE),
    federated_05_transfer_no_dfci__no_d=manuscript_transfer_caption(no_d=TRUE)
  )
}

# Captions go to captions/ inside each figure's manuscript folder under the arm root.
manuscript_write_captions <- function(root,keys=names(manuscript_captions()),capture=NULL) {
  captions <- manuscript_captions()
  unknown <- setdiff(keys,names(captions))
  if(length(unknown)) stop("No manuscript caption defined for: ",paste(unknown,collapse=", "))
  paths <- character()
  for(key in keys) {
    caption_root <- file.path(manuscript_folder(root,key),"captions")
    dir.create(caption_root,recursive=TRUE,showWarnings=FALSE)
    # Keys carry their figure number ("06_..." -> figure6, "federated_03_..." ->
    # federated_figure3), so retired figures leave a gap rather than renumbering;
    # a "__<variant>" suffix is appended ("..._no_dfci__no_d" -> federated_figure5_no_d).
    name <- paste0(if(startsWith(key,"federated_")) "federated_figure" else "figure",
      as.integer(sub("^(federated_)?0*([0-9]+)_.*$","\\2",key)),
      if(grepl("__",key,fixed=TRUE)) paste0("_",sub("^.*?__","",key,perl=TRUE)))
    for(extension in c("md","txt")) {
      path <- file.path(caption_root,paste0(name,".",extension))
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
    if(spec$key %in% MANUSCRIPT_UNIVARIABLE_KEYS && i<=3) p <- p + ggplot2::theme(
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
      if(startsWith(spec$key,"04_"))
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
    if(spec$key %in% MANUSCRIPT_UNIVARIABLE_KEYS)
      grobs[[legend_cell]] <- gridExtra::arrangeGrob(legends[[1]],hr_direction_grob(),ncol=1,
        heights=grid::unit.c(sum(legends[[1]]$heights),grid::unit(1,"null")))
  }
  footer <- if(length(legends) && !length(inline_legend_panels)) legends[[1]] else NULL
  header <- if(length(spec$header)) grid::textGrob(spec$header,
    gp=grid::gpar(fontsize=9,fontface="bold",fontfamily="sans",col="#222222")) else NULL
  gridExtra::arrangeGrob(grobs=grobs,layout_matrix=spec$layout,heights=row_heights,bottom=footer,
    top=header,padding=grid::unit(3,"pt"))
}

manuscript_legend <- function(p,width,height) {
  g <- manuscript_grob(p,width,height)
  boxes <- Filter(function(x) inherits(x,"gtable"),g$grobs[grepl("^guide-box",g$layout$name)])
  if(length(boxes)) boxes[[1]] else NULL
}

manuscript_center_landmark_header <- function(p) p + ggplot2::theme(
  plot.subtitle=ggplot2::element_text(size=8,face="bold",hjust=.5,
    margin=ggplot2::margin(b=3)))

# Discrimination panel(s) first, then three Elastic-Net and three XGBoost
# importance panels; the importance rows take the letters after them.
manuscript_multivariable_labs <- function(plots,spec,legend_nrow=1) {
  n_perf <- length(plots)-6
  stopifnot(n_perf>=1)
  performance <- vector("list",n_perf)
  model_legend <- NULL
  for(i in seq_len(n_perf)) {
    p <- manuscript_style(plots[[i]] + ggplot2::labs(subtitle=NULL),spec$titles[i],letters[i]) +
      ggplot2::guides(fill=ggplot2::guide_legend(nrow=legend_nrow,byrow=TRUE))
    if(is.null(model_legend)) model_legend <- manuscript_legend(p,spec$width,spec$height)
    performance[[i]] <- manuscript_grob(p + ggplot2::theme(legend.position="none"),
      spec$width,spec$height*.18)
  }
  landmarks <- c("0 days","+90 days","+180 days")
  importance <- vector("list",6)
  category_legend <- NULL
  for(i in seq_len(6)) {
    p <- manuscript_style(plots[[i+n_perf]] + ggplot2::labs(subtitle=landmarks[(i-1)%%3+1]),NULL,NULL) +
      ggplot2::guides(color="none",fill=ggplot2::guide_legend(nrow=1,byrow=TRUE,
        override.aes=list(alpha=1)))
    p <- manuscript_center_landmark_header(p)
    if(is.null(category_legend)) category_legend <- manuscript_legend(p,spec$width,spec$height)
    importance[[i]] <- manuscript_grob(p + ggplot2::theme(legend.position="none"),
      spec$width/3,spec$height*.24)
  }
  tag <- function(letter) grid::textGrob(tolower(letter),x=grid::unit(0,"npc"),hjust=0,
    gp=grid::gpar(fontfamily="sans",fontsize=11,fontface="bold",col="#222222"))
  coefficient_row <- gridExtra::arrangeGrob(grobs=importance[1:3],ncol=3,top=tag(letters[n_perf+1]),
    padding=grid::unit(1,"pt"))
  xgboost_row <- gridExtra::arrangeGrob(grobs=importance[4:6],ncol=3,top=tag(letters[n_perf+2]),
    padding=grid::unit(1,"pt"))
  grobs <- performance
  heights <- spec$row_heights[seq_len(n_perf)]
  if(!is.null(model_legend)) {
    grobs <- c(grobs,list(model_legend))
    heights <- c(heights,manuscript_multivariable_legend_height)
  }
  grobs <- c(grobs,list(coefficient_row))
  heights <- c(heights,spec$row_heights[n_perf+1])
  if(!is.null(category_legend)) {
    grobs <- c(grobs,list(category_legend))
    heights <- c(heights,manuscript_multivariable_legend_height)
  }
  grobs <- c(grobs,list(xgboost_row))
  heights <- c(heights,spec$row_heights[n_perf+2])
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

# Pale bars are analyzed patients, solid bars platinum events; labels read
# events/patients, with a dagger when a site has fewer than 25 events.
federated_site_counts_plot <- function(out,site_labels=TRUE) {
  thin <- if("thin" %in% names(out)) out$thin %in% TRUE else rep(FALSE,nrow(out))
  out$count_label <- paste0(scales::comma(out$n_events),"/",scales::comma(out$n_patients),
    ifelse(thin,"\u2020",""))
  p <- ggplot2::ggplot(out,ggplot2::aes(y=y)) +
    ggplot2::geom_rect(ggplot2::aes(xmin=0,xmax=n_patients,ymin=y-.22,ymax=y+.22),fill="#2a78d6",alpha=.28) +
    ggplot2::geom_rect(ggplot2::aes(xmin=0,xmax=n_events,ymin=y-.22,ymax=y+.22),fill="#2a78d6") +
    ggplot2::geom_text(ggplot2::aes(x=n_patients,label=count_label),hjust=-.1,size=2.5,color="#52514e") +
    ggplot2::scale_y_continuous(breaks=out$y,labels=if(site_labels) out$site else NULL,
      limits=c(.5,nrow(out)+.5),expand=ggplot2::expansion(mult=0)) +
    ggplot2::scale_x_continuous(limits=c(0,max(out$n_patients)*1.28),
      expand=ggplot2::expansion(mult=c(0,.01)),labels=scales::comma) +
    ggplot2::labs(x="Patients (events overlaid)",y=NULL) + ggplot2::theme_classic() +
    ggplot2::theme(panel.grid.major.x=ggplot2::element_line(color="grey92",linewidth=.3),
      axis.ticks.y=ggplot2::element_blank(),axis.line.y=ggplot2::element_blank())
  if(!site_labels) p <- p + ggplot2::theme(axis.text.y=ggplot2::element_blank())
  p
}

# Figure 1 site panel: day-0 ADT platinum counts for every site with a result.
figure1_site_counts_plot <- function(sites) {
  out <- prepare_federated_site_incidence(sites) %>% filter(available) %>%
    mutate(y=rev(seq_len(n())))
  if(!nrow(out)) stop("No federated sites with day-0 platinum counts")
  federated_site_counts_plot(out) + ggplot2::labs(title="Federated sites: platinum events/patients",
    subtitle=if(any(out$thin)) "ADT cohort at day 0; \u2020 fewer than 25 events" else "ADT cohort at day 0") +
    ggplot2::theme(plot.title=ggplot2::element_text(face="bold",size=12.5))
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
  b <- federated_site_counts_plot(out,site_labels=FALSE)
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

manuscript_federated_feature_label <- function(x) {
  x <- gsub("Alkaline phosphatase","ALP",x,fixed=TRUE)
  x <- gsub("Diastolic blood pressure","DBP",x,fixed=TRUE)
  x <- gsub("Systolic blood pressure","SBP",x,fixed=TRUE)
  x <- gsub(" absolute"," abs.",x,fixed=TRUE)
  gsub("n_observations","n obs.",x,fixed=TRUE)
}

# Mirrors local Figure 4: four-series mean AUC(t) (a), elastic-net
# coefficients (b) and XGBoost gains (c), from the captured supplement CSVs.
manuscript_federated_multivariable <- function(performance,coefficients,importance,
                                               key="08_federated_multivariable",colors=FEDERATED_MODEL_COLORS,
                                               legend_nrow=1,note=NULL) {
  colors <- colors[names(colors) %in% unique(as.character(performance$name))]
  d <- performance %>% mutate(name=factor(name,levels=names(colors)))
  stopifnot(!anyNA(d$name),!anyDuplicated(d[c("name","landmark")]))
  plots <- list(plot_model_discrimination(d,"auc","Test mean AUC(t)",colors,show_legend=TRUE))
  for(kind in c("cox","xgb")) for(lm in c(0,90,180)) {
    features <- if(kind=="cox") coefficients else importance
    selected <- features[features$landmark_days==lm & features$displayed %in% TRUE,,drop=FALSE]
    plots[[length(plots)+1L]] <- plot_model_importance(selected,kind,NULL) +
      ggplot2::scale_y_discrete(labels=manuscript_federated_feature_label)
  }
  spec <- list(key=key,width=8.5,height=9,layout=matrix(1:3,3),
    row_heights=c(.9,1,1),
    titles=c("Test mean AUC(t)","Elastic-Net coefficients","XGBoost feature importance"))
  p <- manuscript_multivariable_labs(plots,spec,legend_nrow)
  audit <- unique(as.character(c(performance$input_audit_note,coefficients$input_audit_note,
    importance$input_audit_note)))
  audit <- audit[!is.na(audit)&nzchar(audit)]
  d <- arrange(d,name,landmark)
  counts <- paste(sprintf("Day %s, %s: test n=%s; events=%s",d$landmark,as.character(d$name),
    d$n_test,d$n_events_test),collapse="\n")
  list(plot=p,spec=spec,data=d,legend=paste(
    "a: test mean AUC(t). b: Elastic-Net coefficients at days 0, 90 and 180. c: XGBoost feature importance at days 0, 90 and 180.",
    "Held-out test metrics, not training or tuning CV. Local Figure 4 model colors (hatched age baselines) and lab-category colors.",
    "Up to 15 nonzero coefficients (by absolute value) or positive-gain features per landmark, using the local display exclusions (age, body height).",
    "Coefficients are signed log hazard ratios; gains are supplied split gains, not signed effects or SHAP values.",
    note,counts,paste(audit,collapse="\n"),sep="\n"))
}

# Federated Figure 4: the pooled four-site models of Figure 8, with the full
# models refit without Dana-Farber added to the discrimination rows. Each
# variant is scored on its own held-out test sites, so the sets differ.
FEDERATED_FIG4_LEVELS <- c("Elastic-Net Cox","Elastic-Net Cox, no DFCI","Cox baseline (age)",
  "XGBoost Survival","XGBoost Survival, no DFCI","XGBoost baseline (age)",
  "Dynamic DeepHit","Dynamic DeepHit, no DFCI")
manuscript_federated_multivariable_no_dfci <- function(performance,no_dfci_performance,coefficients,importance) {
  renamed <- c(`Elastic-Net Cox`="Elastic-Net Cox, no DFCI",`XGBoost Survival`="XGBoost Survival, no DFCI",
    `Dynamic DeepHit`="Dynamic DeepHit, no DFCI",
    `Dynamic DeepHit, no DFCI`="Dynamic DeepHit, no DFCI")
  no_dfci <- no_dfci_performance %>% filter(config=="both",as.character(name) %in% names(renamed)) %>%
    mutate(name=unname(renamed[as.character(name)]))
  colors <- c(FEDERATED_MODEL_COLORS,FEDERATED_NO_DFCI_MODEL_COLORS)[FEDERATED_FIG4_LEVELS]
  manuscript_federated_multivariable(bind_rows(mutate(performance,name=as.character(name)),no_dfci),
    coefficients,importance,key="federated_04_multivariable",colors=colors,legend_nrow=2,
    note=paste("Darker bars (\"no DFCI\"): full models refit without Dana-Farber and tested on held-out Fred Hutch, Johns Hopkins and MSK patients;",
      "the other bars are the pooled four-site models on their own test set. Panels b and c show the pooled models."))
}

# Federated Figure 5: platinum models trained without Dana-Farber, scored on the
# local Dana-Farber platinum, AVPC and NEPC endpoints (04). All endpoints are
# out of sample. a: discrimination in the full ADT cohort; then, for each
# subpopulation in `include`, the transferred full models vs local clinical Cox
# and combined (score + clinical) Cox models in patients with a Gleason score
# (ADT clock), sequenced patients and sequenced patients with a Gleason score
# (both on the sequencing-date clock). Facet strips give day-0 patients/events.
# Per-horizon AUC(t) stays on the standalone transfer page. With the KM tables
# (`km`), a row of top- vs bottom-10% score KM panels (one per endpoint) and the
# event-free gradient follow (manuscript_federated_transfer_km).
MANUSCRIPT_TRANSFER_ROW_HEIGHTS <- c(first=2.6,clinical=2.6,km=2.7,gradient=2.6)
manuscript_federated_transfer_no_dfci <- function(metrics,clinical=NULL,km=NULL,
    include=names(FEDERATED_TRANSFER_CLINICAL),key="federated_05_transfer_no_dfci") {
  # Panel a shows the full models only; the age-only rows stay on the standalone transfer page.
  metrics <- metrics[metrics$config!="baseline",,drop=FALSE]
  d <- prepare_federated_transfer(metrics,federated_transfer_strip_counts(metrics))
  discrimination <- plot_federated_transfer(d,NULL,NULL,NULL) +
    ggplot2::guides(shape="none") + ggplot2::labs(x="Landmark after ADT initiation")
  plots <- list(discrimination); titles <- "Discrimination"
  legend <- c(
    "a: mean AUC(t) of the full models by endpoint and landmark; bars are 95% bootstrap CIs over patients. Strips give day-0 patients/events.",
    paste0("Patients/events by landmark — ",federated_transfer_count_note(metrics),"."))
  panel_titles <- c(gleason="Patients with Gleason score",trio="Sequenced patients",
    trio_gleason="Sequenced patients with Gleason score")
  panel_text <- c(
    gleason="patients with a Gleason score before the landmark. Transferred full models vs. an unpenalized local Cox of Gleason score + age, and a combined Cox of the transferred score + Gleason score",
    trio="sequenced patients, landmarks measured from the sequencing date (labs before and outcomes after sequencing + landmark). Transferred full models vs. an unpenalized local Cox of TP53, RB1 and PTEN any-alteration indicators (anchoring specimen) + age at sequencing, and a combined Cox of the transferred score + the three indicators",
    trio_gleason="sequenced patients with a Gleason score before sequencing + landmark, on the sequencing-date clock. Transferred full models vs. an unpenalized local Cox of Gleason score + TP53, RB1 and PTEN any-alteration indicators + age at sequencing, and a combined Cox of the transferred score + Gleason score + the three indicators")
  available <- if(is.null(clinical)) character() else
    intersect(intersect(names(FEDERATED_TRANSFER_CLINICAL),include),unique(clinical$subpopulation))
  if(!length(available)) message("Federated Figure 5: clinical comparison unavailable; drawing panel a only")
  for(sub in available) {
    transferred <- clinical[clinical$subpopulation==sub & clinical$model=="transferred_labs",,drop=FALSE]
    plots[[length(plots)+1]] <- plot_federated_transfer_clinical(prepare_federated_transfer_clinical(clinical,sub,
      federated_transfer_strip_counts(transferred)),NULL) +
      ggplot2::labs(x=paste("Landmark after",FEDERATED_TRANSFER_CLINICAL_ORIGIN[[sub]]))
    titles <- c(titles,panel_titles[[sub]])
    legend <- c(legend,paste0(letters[length(plots)],": ",panel_text[[sub]],
      "; mean AUC(t). Local and combined Cox scores are out-of-fold."),
      paste0("Patients/events by landmark — ",federated_transfer_clinical_count_note(clinical,sub),"."))
  }
  n <- length(plots)
  # Six columns: discrimination rows span the width, the KM panels share one
  # row and the gradient spans the width.
  rows <- lapply(seq_len(n),function(i) rep(i,6L))
  row_heights <- MANUSCRIPT_TRANSFER_ROW_HEIGHTS[c("first",rep("clinical",n-1L))]
  extra <- if(is.null(km)) NULL else manuscript_federated_transfer_km(km,first=n+1L)
  if(!is.null(km) && !length(extra$plots)) message("Federated Figure 5: no KM panels estimable; omitting them")
  k <- length(extra$km)
  if(k) {
    rows <- c(rows,list(n+rep(seq_len(k),each=6L%/%k)))
    row_heights <- c(row_heights,MANUSCRIPT_TRANSFER_ROW_HEIGHTS["km"])
  }
  if(!is.null(extra$gradient)) {
    rows <- c(rows,list(rep(n+k+1L,6L)))
    row_heights <- c(row_heights,MANUSCRIPT_TRANSFER_ROW_HEIGHTS["gradient"])
  }
  plots <- c(plots,extra$plots); titles <- c(titles,extra$titles); legend <- c(legend,extra$legend)
  row_heights <- unname(row_heights)
  # 10 in (not the usual 7.2) keeps the one-row b-d legends unclipped; d's is ~8.5 in.
  spec <- list(key=key,width=10,height=sum(row_heights),layout=do.call(rbind,rows),titles=titles,
    row_heights=row_heights,shared_legend_panels=integer())
  list(plot=manuscript_combine(plots,spec),spec=spec,data=metrics,legend=paste(
    c(legend,FEDERATED_TRANSFER_SPEC$source),collapse="\n"))
}

# Federated Figure 5 time-to-event panels from the captured KM tables
# (render_federated_transfer_km), for one bundle and landmark: event-free
# survival by the top vs bottom 10% of the transferred score for platinum, AVPC
# and NEPC (ADT clock; `km`), then the event-free probability at
# FEDERATED_TRANSFER_KM_HORIZON by score quartile and Gleason group (ADT clock)
# and by mutually exclusive TP53/PTEN/RB1 alteration groups (sequencing clock;
# `gradient`). Panels that are not estimable are dropped.
manuscript_federated_transfer_km <- function(km, first=1L, figure=FEDERATED_TRANSFER_KM_FIGURE) {
  family <- figure$model_family; landmark <- figure$landmark_days
  model <- c(elastic_net_cox="Elastic-Net",xgboost_cox="XGBoost")[family]
  if(is.na(model)) model <- federated_transfer_model_label(family)
  horizon <- federated_transfer_horizon_label()
  plots <- list(); titles <- character(); legend <- character()
  add <- function(p, title, text) {
    if(is.null(p)) return(FALSE)
    plots[[length(plots)+1]] <<- p; titles <<- c(titles,title)
    legend <<- c(legend,paste0(letters[first+length(plots)-1L],": ",text))
    TRUE
  }
  counts <- function(endpoint) {
    d <- federated_transfer_km_pick(km$summary,endpoint,family,landmark,"score_decile")
    paste0(paste(sprintf("%s %s/%s",d$stratum,scales::comma(d$n),scales::comma(d$events)),collapse="; "),
      sprintf("; log-rank P = %.2g",d$logrank_p[[1]]))
  }
  short <- c(platinum="Platinum",avpc="AVPC",nepc="NEPC")
  for(endpoint in names(FEDERATED_TRANSFER_ENDPOINTS)) {
    p <- plot_federated_transfer_km(km,"score_decile",endpoint,family,landmark,title=NULL)
    if(!is.null(p)) {
      logrank <- p$labels$subtitle
      p <- p + ggplot2::labs(subtitle=NULL,y=sub(" ","\n",FEDERATED_TRANSFER_KM_YLAB[[endpoint]]),
          x=sprintf("Days from day-%d landmark",as.integer(landmark))) +
        ggplot2::annotate("text",x=Inf,y=-Inf,label=logrank,hjust=1.08,vjust=-.6,size=2.3,color="#52514e")
    }
    add(p,paste0(short[[endpoint]],": score extremes"),paste0(
      sprintf("Kaplan-Meier %s-free survival from the %d-day landmark after ADT initiation, ",
        if(endpoint=="platinum") "platinum" else short[[endpoint]],as.integer(landmark)),
      sprintf("top vs bottom 10%% of the transferred %s score. Patients/events: ",model),
      if(is.null(p)) "" else counts(endpoint),"."))
  }
  n_km <- length(plots)
  gradient <- plot_federated_transfer_gradient(km,family,landmark)
  if(!is.null(gradient)) gradient <- gradient + ggplot2::labs(y=paste(horizon,"event-free\nprobability"))
  has_gradient <- add(gradient,paste(horizon,"event-free probability"),paste0(
    sprintf("Kaplan-Meier %s event-free probability (95%% CI) from the %d-day landmark for each endpoint, ",horizon,as.integer(landmark)),
    sprintf("by transferred %s score quartile and Gleason group (ADT clock), then 0 altered, TP53 only, PTEN only, RB1 only, any two, or all three altered (sequencing clock).",model)))
  if(length(plots)) legend <- c(legend,
    "Score cutpoints were computed in the Dana-Farber patients at risk for each endpoint at each landmark.")
  list(plots=plots,titles=titles,legend=legend,km=plots[seq_len(n_km)],
    gradient=if(has_gradient) plots[[length(plots)]] else NULL)
}

# Figure 3 manuscript styling, shared by the local (DFCI) and federated versions:
# common volcano axes, centered landmark headers, trajectory/KM axis labels and
# an in-panel log-rank annotation. The local volcanoes are displayed through
# -log10(p) = 20; pooled federated p values are far smaller, so the federated
# panels share the range their own (capped) data need.
manuscript_univariable_restyle <- function(plots, federated=FALSE) {
  axes <- lapply(plots[1:3],function(p) p$coordinates$limits)
  if(all(vapply(axes,function(x) length(x$x)==2 && length(x$y)==2,logical(1)))) {
    ylim <- if(federated) range(unlist(lapply(axes,`[[`,"y"))) else manuscript_dfci_volcano_ylim
    for(i in 1:3) plots[[i]] <- plots[[i]] + ggplot2::coord_cartesian(
      xlim=range(unlist(lapply(axes,`[[`,"x"))),ylim=ylim)
  }
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
      plots[[i]] <- plots[[i]] + ggplot2::annotate("text",x=Inf,y=-Inf,label=logrank,
        hjust=1.08,vjust=-.6,size=2.3,color="#52514e")
  }
  list(plots=plots,legend=if(federated)
    "Panels a-c share one -log10(p) display range (values above 30 drawn as triangles at 30) and centered landmark headers." else
    "Panels a-c share a -log10(p) display range through 20 and centered landmark headers.")
}

# "Cohort: <cohort> with LLM label (n = <labeled>)", the count read from the
# panel caption written by render_figure2v3().
manuscript_figure2_header <- function(variant,caption) {
  caption <- gsub("\\s+"," ",paste(caption,collapse=" "))
  n <- regmatches(caption,regexec("([0-9][0-9,]*) with an LLM label",caption))[[1]]
  paste0("Cohort: ",FIGURE2_COHORTS[[variant]]," with LLM label",
    if(length(n)==2) paste0(" (n = ",n[2],")") else "")
}

manuscript_build <- function(items,tables,root) {
  result <- list()
  for(spec in manuscript_specs()) {
    # Specs sharing a key are fallbacks, tried in order.
    if(spec$key %in% names(result) || !all(spec$members %in% names(items))) next
    plots <- lapply(items[spec$members],`[[`,"plot")
    legend <- unlist(lapply(seq_along(plots),function(i) {
      p <- plots[[i]]
      paste0(letters[i],". ",paste(c(p$labels$title,p$labels$subtitle,p$labels$caption),collapse="\n"))
    }))
    if(spec$key=="04_multivariable_labs") {
      legend <- c(
        paste0("a. ",paste(c(plots[[1]]$labels$title,plots[[1]]$labels$subtitle),collapse="\n")),
        "b. Elastic-Net model coefficients at landmark days 0, 90, and 180.",
        "c. XGBoost feature importances at landmark days 0, 90, and 180. Gains are unsigned split gains.")
      result[[spec$key]] <- list(plot=manuscript_multivariable_labs(plots,spec),spec=spec,
        legend=paste(legend,collapse="\n\n"),data=NULL)
      next
    }
    if(manuscript_base_key(spec$key)=="02_llm_nepc_avpc") {
      plots[[3]] <- plots[[3]] + ggplot2::labs(y="Fraction within\nplatinum group")
      plots[[4]] <- plots[[4]] + ggplot2::labs(y="Platinum-positive fraction")
      spec$header <- manuscript_figure2_header(sub("^.*__","",spec$key),plots[[3]]$labels$caption)
    }
    if(spec$key %in% MANUSCRIPT_UNIVARIABLE_KEYS) {
      restyled <- manuscript_univariable_restyle(plots,federated=spec$key!="03_dfci_univariable")
      plots <- restyled$plots
      legend <- c(legend,restyled$legend)
    }
    if(spec$key=="01_cohort_overview") {
      # Patient counts and notes stay in the external legend; omit panel subtitles.
      for(i in seq_along(plots)[-1]) plots[[i]] <- plots[[i]] + ggplot2::labs(subtitle=NULL)
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
      shared_legend=startsWith(spec$key,"04_")),spec=spec,
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
  federated <- lapply(c(elasticnet_performance="elasticnet_performance",xgboost_performance="xgboost_performance",
    coefficients="elasticnet_coefficients",importance="xgboost_importance"),
    function(stem) read(paste0("^",stem,"__platinum[.]csv$")))
  available <- !vapply(federated,is.null,logical(1))
  if(all(available)) {
    # An all-empty audit column reads back as logical; align before binding.
    performance <- dplyr::bind_rows(lapply(federated[1:2],function(d)
      mutate(d,input_audit_note=as.character(input_audit_note))))
    deephit <- read("^deephit_performance__platinum[.]csv$")
    if(!is.null(deephit) && nrow(deephit))
      performance <- dplyr::bind_rows(performance,deephit)
    result[["08"]] <- manuscript_federated_multivariable(performance,federated$coefficients,federated$importance)
  } else if(any(available))
    message("Skipping manuscript Figure 8: missing captured federated ",
      paste(names(federated)[!available],collapse=", "))
  no_dfci <- lapply(c(elasticnet="elasticnet_performance_no_dfci",xgboost="xgboost_performance_no_dfci"),
    function(stem) read(paste0("^",stem,"__platinum[.]csv$")))
  if(all(available) && all(!vapply(no_dfci,is.null,logical(1)))) {
    no_dfci_performance <- dplyr::bind_rows(lapply(no_dfci,function(d)
      mutate(d,input_audit_note=as.character(input_audit_note))))
    deephit_no_dfci <- read("^deephit_performance_no_dfci__platinum[.]csv$")
    if(!is.null(deephit_no_dfci) && nrow(deephit_no_dfci))
      no_dfci_performance <- dplyr::bind_rows(no_dfci_performance,deephit_no_dfci)
    result[["federated_04"]] <- manuscript_federated_multivariable_no_dfci(performance,no_dfci_performance,
      federated$coefficients,federated$importance)
  } else if(all(available)) message("Skipping federated manuscript Figure 4: missing no-DFCI performance tables")
  transfer <- read("^transfer_discrimination_no_dfci__platinum[.]csv$")
  if(!is.null(transfer) && nrow(transfer)) {
    km <- lapply(FEDERATED_TRANSFER_KM_TABLES,function(stem) read(paste0("^",stem,"__platinum[.]csv$")))
    km <- if(all(!vapply(km,is.null,logical(1)))) km else NULL
    clinical <- read("^transfer_clinical_comparison_no_dfci__platinum[.]csv$")
    # Full figure, and the same figure without the sequenced-with-Gleason panel (d).
    result[["federated_05"]] <- manuscript_federated_transfer_no_dfci(transfer,clinical,km)
    result[["federated_05__no_d"]] <- manuscript_federated_transfer_no_dfci(transfer,clinical,km,
      include=c("gleason","trio"),key="federated_05_transfer_no_dfci__no_d")
  }
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
  publish <- function(plot, destination, width, height, stem, title = NULL, landmarks = "", manuscript = FALSE) {
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
      width = width, height = height, stem = stem, title = title, landmark = landmarks)
    scenes[[length(scenes)]]$manuscript <<- manuscript
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
        target <- figure_sidecar_path(target)
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
  # A compiled figure whose panels all appear in a built manuscript figure is a
  # duplicate; only the manuscript version is exported for that identity.
  manuscript_members <- if(is.null(manuscript_root)) character() else unique(unlist(lapply(
    Filter(function(spec) all(spec$members %in% names(manuscript_items)),manuscript_specs()),`[[`,"members")))
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
      if(basename(selected[[1]]$destination)=="platinum__all__incl" &&
         all(vapply(selected,`[[`,character(1),"stem") %in% manuscript_members)) next
      # Variable-length bundles size their height to the rows actually used.
      if(!is.null(spec[["page_row_height"]])) spec$height <- spec[["page_row_height"]]*ceiling(length(selected)/spec$cols)
      compiled <- figure_combine(selected,spec,generation)
      destination <- figure_public_path(selected[[1]]$destination)
      identity <- basename(selected[[1]]$destination)
      key <- paste0(spec$key,if(length(pages)>1) paste0("_page",page))
      # Lab name is already the folder; omit it from the compiled filename.
      if(startsWith(key,"km_quintile_")) key <- "km_extremes"
      folder <- dirname(destination)
      # Multi-lab pages sit beside, not inside, the per-lab km/ and trajectories/ folders.
      if(isTRUE(spec$multi_lab)) folder <- sub("/(km|trajectories)/[^/]+$","",folder)
      destination <- file.path(folder,paste0(key,"__",identity))
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
      destination <- file.path(manuscript_folder(manuscript_root,item$spec$key),item$spec$key)
      dir.create(dirname(destination),recursive=TRUE,showWarnings=FALSE)
      legend_path <- figure_sidecar_path(paste0(destination,".md"))
      dir.create(dirname(legend_path),recursive=TRUE,showWarnings=FALSE)
      writeLines(c(paste0("# ",gsub("_"," ",item$spec$key)),"",
        sprintf("Source-rendered manuscript figure: %.1f-inch width; 600-dpi PNG and vector PDF.",
          item$spec$width),"",item$legend),legend_path)
      table_capture(legend_path)
      if(is.data.frame(item$data)) {
        data_path <- figure_sidecar_path(paste0(destination,".csv"))
        readr::write_csv(item$data,data_path); table_capture(data_path)
      }
      publish(item$plot,destination,item$spec$width,item$spec$height,item$spec$key,manuscript=TRUE)
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
    # Notebook 05 previously wrote standalone cohort-comparison panels here.
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
  guidance <- "Run 06_prep_figure_data.ipynb with the requested settings, then rerun 07_figures/all.R."
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
  if (isTRUE(config$checklist_supplement) && !federated_only) {
    # Absent runs are a reported skip, but a stale preparation must not be read.
    for (key in c("checklist_labs_run", "checklist_trio_run"))
      if (!isTRUE(prepared$checklist_supplement) || !same_path(prepared[[key]], config[[key]]))
        fail(paste("Prepared checklist supplement does not match:", key))
    if (!same(prepared$checklist_horizon, config$checklist_horizon))
      fail("Prepared checklist horizon does not match.")
  }
  if (isTRUE(config$federated)) {
    if (!isTRUE(prepared$federated) || !same_path(prepared$federated_path, config$federated_path))
      fail("Requested federated results were not prepared.")
    registered_inputs <- c(vapply(c("cox_within_site_all_sites_cohort.csv", "cox_within_site_all_sites_results.csv"),
      function(filename) federated_within_site_file(config$federated_path, filename), character(1)),
      unlist(lapply(names(FEDERATED_MULTIVARIABLE_VARIANTS), function(variant)
        lapply(names(FEDERATED_MULTIVARIABLE), function(family)
          federated_multivariable_files(config$federated_path, family, variant))), use.names = FALSE))
    registered_inputs <- c(registered_inputs,
      unlist(lapply(names(FEDERATED_MULTIVARIABLE_VARIANTS),function(variant)
        federated_deephit_metric_paths(config$federated_path,variant)),use.names=FALSE),
      unlist(federated_checklist_files(config$federated_path),use.names=FALSE),
      file.path(federated_transfer_no_dfci_dir(config), FEDERATED_TRANSFER_NO_DFCI_FILES),
      file.path(federated_transfer_no_dfci_dir(config),"all_models",
        c("transfer_metrics.csv","transfer_auc_t.csv")),
      file.path(federated_figure_data_dir(config), FEDERATED_FIGURE_DATA_FILES))
    if (!same_path(federated_transfer_no_dfci_dir(prepared), federated_transfer_no_dfci_dir(config)))
      fail("Prepared no-Dana-Farber federated transfer path does not match.")
    if (!same_path(federated_figure_data_dir(prepared), federated_figure_data_dir(config)))
      fail("Prepared federated KM/trajectory data path does not match.")
    for (site_path in registered_inputs) {
      filename <- basename(site_path)
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
    metastatic_config = NULL, forest_config = NULL, federated_config = NULL,
    figure_key = NULL) {
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
  if (is.null(figure_key)) {
    figure_archive_legacy_receipts(config$fig_root, config$cache_root)
    figure_flatten_export_layout(config$fig_root)
  }
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
  checklist_enabled <- isTRUE(config$checklist_supplement) && !identical(config$scope, "federated") &&
    "adt" %in% config$cohorts
  if (checklist_enabled) {
    if (is.null(manifest$checklist)) {
      message("Checklist supplement skipped; no predictive-checklist runs were prepared by 06.")
    } else {
      signature <- figure_object_hash(list(manifest$checklist$key, runtime))
      jobs$checklist_supplement <- list(name = "checklist_supplement", signature = signature,
        directory = file.path(config$cache_root, "scenes", "checklist_supplement", signature))
    }
  }
  if (!is.null(figure_key)) {
    figure_jobs <- list(
      `01_cohort_overview` = c("adt__platinum", "cohort_overview", "federated"),
      `02_llm_nepc_avpc` = "adt__platinum",
      `03_dfci_univariable` = "adt__platinum",
      `04_multivariable_labs` = c("adt__platinum", "checklist_supplement"),
      `06_dfci_cohort_sensitivity` = "cohort_overview",
      `07_federated_incidence_associations` = "federated",
      `08_federated_multivariable` = "federated",
      federated_03_univariable = "federated",
      federated_04_multivariable = "federated",
      federated_05_transfer_no_dfci = "federated")
    if (!figure_key %in% names(figure_jobs))
      stop("Unknown manuscript figure: ", figure_key)
    jobs <- jobs[intersect(names(jobs), figure_jobs[[figure_key]])]
    if (!length(jobs)) stop("No prepared figure jobs are enabled for ", figure_key)
  }
  build_job <- function(job) {
    if (!is.null(manifest$errors[[job$name]])) stop(manifest$errors[[job$name]])
    # Optional missing federation inputs produce a reported skip, not an empty
    # completion marker. A later delivery is picked up by source fingerprints.
    if (job$name == "cohort_overview") {
      source(file.path(dirname(pipeline_path), "figure_supplements.R"), local = TRUE)
      build <- function() render_cohort_overview(manifest, config, forest_config)
    } else if (job$name == "checklist_supplement") {
      data_dir <- manifest$checklist$directory
      if (stage != "render" && !all(file.exists(file.path(data_dir, CHECKLIST_TABLES)))) {
        warning("Checklist supplement skipped; missing prepared tables in ", data_dir)
        return(list(skipped = TRUE))
      }
      build <- function() render_checklist_supplement(data_dir,
        file.path(config$fig_root, "ADT", FIGURE_SECTIONS[["multivariate_internal"]]),
        dpi, TRUE, save_federated_panel, figure_sidecar_path,
        getOption("compass.figure_table_capture"))
    } else if (job$name == "federated") {
      inputs <- federated_config$results
      if (stage != "render" && any(!file.exists(inputs))) {
        warning("Federated supplement skipped; missing: ", paste(inputs[!file.exists(inputs)], collapse = ", "))
        return(list(skipped = TRUE))
      }
      build <- function() render_federated_supplement(config$data_root, config$fig_root,
        federated_config$results, dpi = dpi, overwrite = TRUE,
        transfer_no_dfci_dir = federated_transfer_no_dfci_dir(config),
        figure_data_dir = federated_figure_data_dir(config))
    } else {
      build <- function() {
        message("Preparing figure set: ", job$name)
        generate_figures(job$cohort, config$data_root, config$fig_root,
          endpoint = job$endpoint, llm_annotations_path = config$classifier_path,
          lab_figure_labs = unlist(config$labs),
          plot_gam_trajectories = config$gam, plot_adt_intent_supplement = config$adt_intent,
          overwrite = TRUE, save_pdf = FALSE, show = FALSE,
          metastatic_supplement = if (config$metastatic) metastatic_config else NULL,
          cohort_forest_config = forest_config,
          federated_path = if (job$name == "adt__platinum") config$federated_path else NULL,
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
        file.path(config$fig_root,"ADT") else NULL)
  }
  # Only pending jobs warm tables. A resumed all-mode run reads no patient
  # Parquet at all when every scene snapshot is current.
  if (stage != "render") {
    pending <- Filter(function(job) !job$name %in% c("federated", "cohort_overview", "checklist_supplement") &&
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
    if (!is.null(figure_key)) {
      figure_section <- MANUSCRIPT_SECTIONS[[figure_key]]
      section_path <- paste0("/", FIGURE_SECTIONS[[figure_section]], "/")
      tasks <- Filter(function(task) {
        destination <- task$scene$destination
        stem <- task$scene$stem
        # Each script includes panels and supplements in its output section,
        # while manuscript variants sharing a section remain independently selectable.
        if (figure_key == "07_federated_incidence_associations" &&
            stem == "site_incidence_lm000") return(TRUE)
        if (!grepl(section_path, destination, fixed = TRUE)) return(FALSE)
        if (grepl("/manuscript/", destination, fixed = TRUE))
          return(startsWith(stem, figure_key))
        if (figure_key == "07_federated_incidence_associations")
          return(grepl("^(psa_forest|testosterone_forest|federated_|site_incidence)", stem))
        if (figure_key == "federated_03_univariable")
          return(grepl("^(federated_|fed3_)", stem))
        TRUE
      }, tasks)
    }
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
    if (is.null(figure_key)) figure_archive_old_exports(config,prepared)
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
  pattern <- paste0("^(figure1_cohort|(figure2v3_llm|classifier_validation|subtype_platinum)(_(icd|adt|lm0))?|",
    "metastatic_label_agreement|labs_univariate_all_landmarks|somatic_(snv|any)_carrier_km_page[0-9]+|",
    "discrimination|sensitivity_(gleason|somatic)|importance_(cox|xgb)|km_extremes|",
    "xgboost_(performance|importance)|elasticnet_(performance|coefficients)|",
    "event_incidence_lm[0-9]+|metastatic_label_overlap)__")
  selected <- files[grepl(pattern,basename(files)) & !startsWith(files,"compiled/")]
  for(relative in selected) {
    p <- download_read_panel(file.path(input,relative))
    stem <- sub("__.*$","",basename(relative))
    count_line <- startsWith(stem,"sensitivity_")
    incidence <- startsWith(stem,"event_incidence_")
    overlap <- stem=="metastatic_label_overlap"
    p <- download_remove_header(p,lines=if(count_line || incidence) 3L else 2L,
      retain=if(incidence) 2:3 else if(count_line || overlap ||
        grepl("^(xgboost|elasticnet)_",stem)) 2L else integer())
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
    extremes <- sprintf("bottom vs top %d%%",round(100*LAB_EXTREME_FRACTION))
    headings <- c("d   PSA trajectory (log1p)",paste0("e   PSA ",extremes," — day ",landmark),
      "f   Testosterone trajectory (log1p)",paste0("g   Testosterone ",extremes," — day ",landmark))
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
  manifest <- file.path(output,FIGURE_SIDECAR_DIR,"manifest.csv")
  # Downloads made before tables moved into data/ keep the manifest at the root.
  if(!file.exists(manifest)) manifest <- file.path(output,"manifest.csv")
  metadata <- readr::read_csv(manifest,show_col_types=FALSE,
    col_types=readr::cols(.default="c"))
  moved <- metadata$path %in% selected
  metadata$path[moved] <- paste0("compiled/",metadata$path[moved])
  metadata$topic[moved] <- "compiled"
  for(format in c("png","pdf")) {
    row <- metadata[1,,drop=FALSE]
    row$path <- paste0("compiled/associations/",stem,".",format)
    row$topic <- "compiled"; row$figure_id <- sub("__.*$","",stem)
    row$endpoint <- "platinum"; row$cohort <- "all__incl"; row$landmark <- "0,90,180"
    row$title <- paste0("DFCI lab associations, trajectories and extreme-decile survival (KM day ",landmark,")")
    row$format <- format
    metadata <- rbind(metadata,row)
  }
  dir.create(file.path(output,FIGURE_SIDECAR_DIR),showWarnings=FALSE)
  readr::write_csv(metadata,file.path(output,FIGURE_SIDECAR_DIR,"manifest.csv"))
  unlink(file.path(output,c("manifest.csv","index.html")))
  figure_write_catalog(list(fig_root=dirname(output)),list())
  invisible(list(selected=selected,new_figure=target))
}

if(sys.nframe()==0L) {
  args <- commandArgs(trailingOnly=TRUE)
  if(length(args)<2L) stop("Supply input arm directory and a new output directory; optional KM landmark 0/90/180.")
  reformat_downloaded_figures(args[1],args[2],if(length(args)>2L) as.integer(args[3]) else 0L)
}
