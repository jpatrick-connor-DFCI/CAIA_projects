# Manuscript exports are additional views of the original statistics/plots.
# Never refit, recompute q-values, or serialize patient-bearing ggplot objects.
.manuscript_sources <- Filter(function(x) is.character(x) && length(x)==1 &&
  basename(x)=="manuscript_figures.R", lapply(sys.frames(),function(frame) frame$ofile))
source(file.path(dirname(tail(.manuscript_sources,1)[[1]]), "federated_no_msk_figures.R"), local=TRUE)
rm(.manuscript_sources)
manuscript_specs <- function() {
  spec <- function(key, members, height, layout, titles) list(key=key, members=members,
    width=7.2, height=height, layout=layout, titles=titles)
  list(
    spec("01_cohort_overview", c("figure1a_consort","figure1b_km","figure1c_span",
      "figure1c_dx_to_tx","figure1c_time_to_platinum"), 6.4,
      rbind(c(1,1,2,2),c(1,1,3,3),c(4,4,5,5)),
      c("Cohort selection","Platinum-free survival","Lab record span","Diagnosis to ADT","ADT to platinum")),
    spec("02_llm_nepc_avpc", c("figure2v3_confusion_matrix","figure2v3_metric_bar",
      "figure2v3_subtype_landscape","figure2v3_enrichment"), 6.2, rbind(1:2,3:4),
      c("NEPC: chart-review agreement","NEPC classification metrics","NEPC/AVPC proxy subtypes","Platinum enrichment")),
    spec("03_dfci_univariable", figure_dfci_overview_spec()$members, 8.6,
      rbind(c(1,1,2,2,3,3),c(4,4,4,5,5,5),c(6,6,6,7,7,7)),
      c("Day 0","Day 90","Day 180","PSA trajectory","PSA: extreme quintiles",
        "Testosterone trajectory","Testosterone: extreme quintiles")),
    spec("04_multivariable_labs", paste0("figure4a_discrimination_",c("auc","cindex"),"_platinum"),
      5.6, matrix(1:2,2), c("Mean AUC(t)","C-index")),
    spec("05_gleason_sensitivity", paste0("figure4c_sensitivity_gleason_",c("auc","cindex"),"_platinum"),
      5.6, matrix(1:2,2), c("Mean AUC(t)","C-index")),
    spec("07_cohort_sensitivity_associations", "cohort_forest_platinum_landmark180",
      5.0, matrix(1), "PSA and testosterone: day 180"))
}

manuscript_theme <- function() ggplot2::theme(
  text=ggplot2::element_text(family="sans",size=8,color="#222222"),
  plot.title=ggplot2::element_text(size=9,face="bold",margin=ggplot2::margin(b=5)),
  plot.subtitle=ggplot2::element_text(size=7,margin=ggplot2::margin(b=4)),
  plot.tag=ggplot2::element_text(size=11,face="bold"),
  axis.text=ggplot2::element_text(size=7,color="#222222"),
  axis.text.x=ggplot2::element_text(size=7,color="#222222"),
  axis.text.y=ggplot2::element_text(size=7,color="#222222"),
  axis.title=ggplot2::element_text(size=8),
  axis.title.x=ggplot2::element_text(size=8),
  axis.title.y=ggplot2::element_text(size=8),
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
  result <- p + ggplot2::labs(title=NULL,tag=tag,caption=NULL) + manuscript_theme()
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

manuscript_combine <- function(plots,spec,shared_legend=FALSE) {
  legends <- list(); grobs <- list()
  for(i in seq_along(plots)) {
    p <- manuscript_style(plots[[i]],spec$titles[i],if(length(plots)>1) LETTERS[i] else NULL)
    if(inherits(p,"ggplot")) {
      if(shared_legend) {
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
  footer <- if(length(legends)) legends[[1]] else NULL
  gridExtra::arrangeGrob(grobs=grobs,layout_matrix=spec$layout,heights=row_heights,bottom=footer,
    padding=grid::unit(3,"pt"))
}

manuscript_cohort_labels <- c(adt="All ADT",adt_noprecastrate="All ADT; no prior castrate",
  adt_metastatic_adt="Metastatic (ADT intent)",
  adt_metastatic_adt_noprecastrate="Metastatic (ADT intent);\nno prior castrate",
  adt_metastatic_llm="Metastatic (LLM)",
  adt_metastatic_llm_noprecastrate="Metastatic (LLM);\nno prior castrate")

manuscript_incidence <- function(d) {
  d <- d[d$endpoint=="platinum",,drop=FALSE]
  stopifnot(nrow(d)>0,!anyDuplicated(d$cohort))
  d$label <- unname(manuscript_cohort_labels[d$cohort])
  d$label[is.na(d$label)] <- d$cohort[is.na(d$label)]
  d$label <- factor(d$label,levels=rev(unique(d$label)))
  d$available <- d$status=="ok" & !is.na(d$n_patients) & d$n_patients>0
  stopifnot(all(d$n_events[d$available]>=0),all(d$n_events[d$available]<=d$n_patients[d$available]))
  good <- d[d$available,,drop=FALSE]
  z <- 1.96; p <- good$n_events/good$n_patients; den <- 1+z^2/good$n_patients
  centre <- (p+z^2/(2*good$n_patients))/den
  half <- z*sqrt(p*(1-p)/good$n_patients+z^2/(4*good$n_patients^2))/den
  good$rate <- 100*p; good$lo <- 100*pmax(0,centre-half);good$hi <- 100*pmin(1,centre+half)
  a <- ggplot2::ggplot(good,ggplot2::aes(n_patients,label)) +
    ggplot2::geom_col(fill="#bbd6ea",width=.6) +
    ggplot2::geom_col(ggplot2::aes(x=n_events),fill="#0072B2",width=.6) +
    ggplot2::geom_text(ggplot2::aes(label=paste0(scales::comma(n_events),"/",scales::comma(n_patients))),
      hjust=-.1,size=2.5) + ggplot2::scale_x_continuous(expand=ggplot2::expansion(mult=c(0,.36)),labels=scales::comma) +
    ggplot2::scale_y_discrete(drop=FALSE) + ggplot2::labs(x="Patients",y=NULL) + ggplot2::theme_classic()
  b <- ggplot2::ggplot(good,ggplot2::aes(rate,label)) +
    ggplot2::geom_errorbar(ggplot2::aes(xmin=lo,xmax=hi),orientation="y",width=.15,linewidth=.4) +
    ggplot2::geom_point(color="#0072B2",size=1.8) +
    ggplot2::geom_text(ggplot2::aes(x=hi,label=sprintf("%.1f%%%s",rate,ifelse(n_events<25,"*",""))),hjust=-.15,size=2.5) +
    ggplot2::scale_x_continuous(limits=c(0,max(good$hi)*1.35)) +
    ggplot2::scale_y_discrete(drop=FALSE) + ggplot2::labs(x="Observed platinum events (%)",y=NULL) +
    ggplot2::theme_classic() + ggplot2::theme(axis.text.y=ggplot2::element_blank(),axis.ticks.y=ggplot2::element_blank())
  # Explicit missing rows, never substituted with zero events.
  missing <- d[!d$available,,drop=FALSE]
  if(nrow(missing)) for(which in c("a","b")) {
    p <- get(which)+ggplot2::geom_text(data=missing,ggplot2::aes(x=0,label="Unavailable"),hjust=0,size=2.5)
    assign(which,p)
  }
  spec <- list(key="06_cohort_sensitivity_incidence",width=7.2,height=3.5,
    layout=matrix(1:2,1),titles=c("Cohort size and platinum events","Platinum event fraction"))
  # The shared labels occupy space only in A; both panels still share row order.
  list(plot=manuscript_combine(list(a,b),spec),spec=spec,data=d,
    legend=paste("Landmark day",paste(unique(d$landmark_days),collapse=", "),
      ". A: pale bars show eligible patients; dark bars show platinum events; labels are events/patients.",
      "B: observed event fraction during follow-up with 95% Wilson confidence intervals; * marks fewer than 25 events.",
      "These are not censoring-adjusted cumulative incidences or risks by the landmark day. Cohorts overlap.",
      "'No prior castrate' denotes the source noprecastrate cohort restriction."))
}

manuscript_federated_labs <- function(d) {
  d <- d[d$feature_stat %in% c("mean","min","max","last") &
    d$landmark_days %in% c(0,90,180),,drop=FALSE]
  stopifnot(!anyDuplicated(d[c("lab_name","landmark_days","source","feature_stat")]))
  note <- paste(unique(d$population_note[d$source=="Federated*"]),collapse="\n")
  # Reuse the standalone forests: source colors, landmark/statistic facets,
  # log scales and significance symbols all retain their original meaning.
  plots <- lapply(c("PSA","Testosterone"),function(analyte)
    plot_federated_comparison(d,analyte,note) + ggplot2::labs(subtitle=NULL))
  spec <- list(key="08_federated_psa_testosterone",width=7.2,height=8.4,
    layout=matrix(1:2,2),titles=c("PSA","Testosterone"))
  list(plot=manuscript_combine(plots,spec,shared_legend=TRUE),spec=spec,data=d,
    legend=paste("A: PSA. B: testosterone. Platinum endpoint; original site colors and landmark/statistic facets.",
      "Columns: mean, minimum, maximum and last value. Rows: days 0, 90 and 180.",
      "Open circles: not significant; filled circles: nominal p < 0.05 only; diamonds: supplied FDR q < 0.05.",
      paste(vapply(plots,function(p) p$labels$caption,character(1)),collapse="\n"),sep="\n"))
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
  spec <- list(key="09_federated_xgboost",width=7.2,height=9.2,
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
  if(length(audit)) p <- gridExtra::arrangeGrob(p,bottom=grid::textGrob(
    paste(strwrap(paste(audit,collapse=" "),width=115),collapse="\n"),
    gp=grid::gpar(fontsize=7,col="#a33b19")))
  counts <- paste(sprintf("Day %d, %s: test n=%s; events=%s",metrics$landmark_days,
    metrics$config,metrics$n_test,metrics$n_events_test),collapse="\n")
  list(plot=p,spec=spec,data=metrics,legend=paste(
    "A: test mean AUC(t). B: test C-index. C–E: feature importance at days 0, 90 and 180.",
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
      paste0(LETTERS[i],". ",paste(c(p$labels$title,p$labels$subtitle,p$labels$caption),collapse="\n"))
    }))
    if(spec$key=="02_llm_nepc_avpc") {
      plots[[3]] <- plots[[3]] + ggplot2::labs(y="Fraction within\nplatinum group")
      plots[[4]] <- plots[[4]] + ggplot2::labs(y="Platinum-positive fraction")
    }
    if(spec$key=="03_dfci_univariable") {
      axes <- lapply(plots[1:3],function(p) p$coordinates$limits)
      if(all(vapply(axes,function(x) length(x$x)==2 && length(x$y)==2,logical(1))))
        for(i in 1:3) plots[[i]] <- plots[[i]] + ggplot2::coord_cartesian(
          xlim=range(unlist(lapply(axes,`[[`,"x"))),ylim=range(unlist(lapply(axes,`[[`,"y"))))
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
      for(i in 1:3) plots[[i]] <- plots[[i]] + ggplot2::labs(subtitle=NULL) +
        ggplot2::guides(color="none",fill=ggplot2::guide_legend(ncol=2,
          override.aes=list(shape=21,size=1.8,alpha=1)))
      for(i in c(4,6)) plots[[i]] <- plots[[i]] + ggplot2::labs(
        x="Years from ADT initiation",y=paste0("Mean log1p(",if(i==4) "PSA" else "testosterone", ")\n(95% CI)"))
      for(i in c(5,7)) plots[[i]] <- plots[[i]] + ggplot2::labs(
        x=paste0("Days from day-",figure_overview_km_landmark()," landmark"),y="Platinum-free probability")
    }
    if(spec$key=="07_cohort_sensitivity_associations") plots[[1]] <- plots[[1]] +
      ggplot2::scale_y_discrete(labels=function(x) unname(manuscript_cohort_labels[x]))
    if(spec$key=="01_cohort_overview") {
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
      data=if(startsWith(spec$key,"07_")) plots[[1]]$data else NULL)
  }
  read <- function(pattern) {
    paths <- tables[grepl(pattern,basename(tables))]
    if(length(paths)!=1) return(NULL)
    readr::read_csv(paths,show_col_types=FALSE)
  }
  incidence <- read("^event_incidence_lm180__platinum__all__incl[.]csv$")
  if(!is.null(incidence)) result[["06"]] <- manuscript_incidence(incidence)
  psa <- read("^psa_forest__platinum[.]csv$"); testosterone <- read("^testosterone_forest__platinum[.]csv$")
  if(!is.null(psa)&&!is.null(testosterone)) result[["08"]] <- manuscript_federated_labs(dplyr::bind_rows(psa,testosterone))
  metrics <- read("^xgboost_performance__platinum[.]csv$"); importance <- read("^xgboost_importance__platinum[.]csv$")
  if(!is.null(metrics)&&!is.null(importance)) result[["09"]] <- manuscript_federated_xgb(metrics,importance)
  result
}
