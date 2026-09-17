# Manuscript exports are additional views of the original statistics/plots.
# Never refit, recompute q-values, or serialize patient-bearing ggplot objects.
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
  result <- p + ggplot2::labs(title=title,tag=tag,caption=NULL) + manuscript_theme()
  for(field in c("axis.text","axis.text.x","axis.text.y","axis.title.x","axis.title.y"))
    if(inherits(p$theme[[field]],"element_blank")) result$theme[[field]] <- p$theme[[field]]
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
    grobs[[i]] <- manuscript_grob(p,spec$width*length(unique(cells[,2]))/ncol(spec$layout),
      spec$height*length(unique(cells[,1]))/nrow(spec$layout))
  }
  footer <- if(length(legends)) legends[[1]] else NULL
  gridExtra::arrangeGrob(grobs=grobs,layout_matrix=spec$layout,bottom=footer,
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
  stats <- c("mean","min","max","last")
  d <- d[d$feature_stat %in% stats & d$landmark_days %in% c(0,90,180),,drop=FALSE]
  stopifnot(!anyDuplicated(d[c("lab_name","landmark_days","source","feature_stat")]))
  sources <- c("Dana-Farber","Fred Hutch","Johns Hopkins","Federated*")
  sources <- c(sources[sources %in% d$source],setdiff(unique(d$source),sources))
  short <- c("Dana-Farber"="DFCI","Fred Hutch"="Fred Hutch","Johns Hopkins"="Johns Hopkins","Federated*"="Federated*")
  rows <- as.vector(t(outer(c(0,90,180),sources,paste,sep=" | ")))
  d$row <- factor(paste(d$landmark_days,d$source,sep=" | "),levels=rev(rows))
  labels <- setNames(gsub("Dana-Farber","DFCI",paste0(sub(" \\| ","d · ",rows))),rows)
  d$feature_stat <- factor(d$feature_stat,levels=stats,labels=c("Mean","Minimum","Maximum","Last"))
  d$lab_name <- factor(d$lab_name,levels=c("PSA","Testosterone"))
  valid <- d$valid_estimate & !is.na(d$valid_estimate)
  ci <- valid & d$valid_ci & !is.na(d$valid_ci)
  p <- ggplot2::ggplot(d,ggplot2::aes(hazard_ratio_per_sd,row,color=lab_name)) +
    ggplot2::geom_vline(xintercept=1,linetype=2,linewidth=.3,color="grey65") +
    ggplot2::geom_blank(ggplot2::aes(x=1)) +
    ggplot2::geom_errorbar(data=d[ci,],ggplot2::aes(xmin=ci_lower,xmax=ci_upper),orientation="y",width=.18,linewidth=.4) +
    ggplot2::geom_point(data=d[valid,],ggplot2::aes(shape=significance),size=1.7,stroke=.5) +
    ggplot2::scale_shape_manual(values=c("Not significant"=1,"Nominal p < 0.05 only"=16,
      "FDR q < 0.05"=18,"Significance unavailable"=4),name=NULL) +
    ggplot2::scale_color_manual(values=c(PSA="#0072B2",Testosterone="#D55E00"),guide="none") +
    ggplot2::scale_x_log10(breaks=c(.25,.5,1,2,4,8),labels=scales::label_number()) +
    ggplot2::scale_y_discrete(labels=labels,drop=FALSE) +
    ggplot2::facet_grid(lab_name~feature_stat) + ggplot2::theme_classic() +
    ggplot2::labs(x="Hazard ratio per SD (95% CI; log scale)",y=NULL) + manuscript_theme()
  spec <- list(key="08_federated_psa_testosterone",width=7.2,height=7.2)
  list(plot=p,spec=spec,data=d,legend=paste(
    "Platinum endpoint; within-site and supplied federated estimates at days 0, 90 and 180.",
    "Columns show mean, minimum, maximum and last lab value. Supplied p/q values are unchanged.",
    "Open circles: not significant; filled circles: nominal p < 0.05 only; diamonds: FDR q < 0.05.",
    "Missing estimates/CIs remain unavailable; see the aggregate CSV.",
    paste(unique(d$population_note),collapse="\n")))
}

manuscript_federated_xgb <- function(metrics,importance) {
  metrics$name <- factor(ifelse(metrics$config=="both","Labs","Age baseline"),levels=c("Labs","Age baseline"))
  metrics$landmark <- factor(metrics$landmark_days,levels=c(0,90,180))
  plots <- lapply(c("test_mean_auc_t","test_c_index"),function(column) {
    ggplot2::ggplot(metrics,ggplot2::aes(landmark,.data[[column]],color=name,group=name)) +
      ggplot2::geom_hline(yintercept=.5,linetype=3,color="grey65") +
      ggplot2::geom_line(linewidth=.45) + ggplot2::geom_point(size=1.7) +
      ggplot2::scale_color_manual(values=c(Labs="#0072B2","Age baseline"="#D55E00"),name=NULL) +
      ggplot2::coord_cartesian(ylim=c(.45,1)) + ggplot2::labs(x="Landmark (days)",y=NULL) + ggplot2::theme_classic()
  })
  for(lm in c(0,90,180)) {
    d <- importance[importance$landmark_days==lm & importance$displayed %in% TRUE,,drop=FALSE]
    d <- d[order(d$gain),,drop=FALSE]
    label <- paste(d$lab_name,d$feature_stat)
    label <- gsub("Alkaline phosphatase","ALP",label,fixed=TRUE)
    label <- gsub("n_observations","n obs.",label,fixed=TRUE)
    label <- gsub("Neutrophils absolute","Neutrophils (abs.)",label,fixed=TRUE)
    label <- gsub("person id","person_id",label,fixed=TRUE)
    d$label <- factor(label,levels=unique(label))
    p <- ggplot2::ggplot(d,ggplot2::aes(gain,label,fill=identifier_feature)) +
      ggplot2::geom_col(width=.7,show.legend=FALSE) +
      ggplot2::scale_fill_manual(values=c(`FALSE`="#0072B2",`TRUE`="#D55E00")) +
      ggplot2::scale_y_discrete() +
      ggplot2::labs(x="Split gain",y=NULL) + ggplot2::theme_classic()
    plots[[length(plots)+1L]] <- p
  }
  spec <- list(key="09_federated_xgboost",width=7.2,height=9.2,
    layout=rbind(c(1,1,2,2),c(3,3,3,3),c(4,4,4,4),c(5,5,5,5)),
    titles=c("Test mean AUC(t)","Test C-index","Feature importance: day 0",
      "Feature importance: day 90","Feature importance: day 180"))
  # Full-width importance rows keep all 15 feature labels readable.
  p <- manuscript_combine(plots,spec)
  audit <- unique(c(metrics$input_audit_note,importance$input_audit_note))
  audit <- audit[!is.na(audit)&nzchar(audit)]
  if(length(audit)) p <- gridExtra::arrangeGrob(p,bottom=grid::textGrob(
    paste(strwrap(paste(audit,collapse=" "),width=115),collapse="\n"),
    gp=grid::gpar(fontsize=7,col="#a33b19")))
  counts <- paste(sprintf("Day %d, %s: test n=%s; events=%s",metrics$landmark_days,
    metrics$name,metrics$n_test,metrics$n_events_test),collapse="\n")
  list(plot=p,spec=spec,data=metrics,legend=paste(
    "Held-out test performance, not training or tuning CV. Missing results are not interpolated.",
    "Importance displays up to 15 positive-gain features per landmark, with the source display exclusions.",
    "ALP: alkaline phosphatase; n obs.: observation count; abs.: absolute count.",
    "Split gains are not signed effects or SHAP values. Orange flags identifier features.",counts,paste(audit,collapse="\n"),sep="\n"))
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
