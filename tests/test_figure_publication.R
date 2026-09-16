source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_data_cache.R")
source("COMPASS/survival_analysis/federated_no_msk_figures.R")
local({
  root <- tempfile("publication-",tmpdir="/private/tmp"); dir.create(root)
  if(Sys.getenv("COMPASS_PUBLICATION_REVIEW")!="1") on.exit(unlink(root,recursive=TRUE))
  else message("Publication review: ",root)
  output <- file.path(root,"figures"); cache <- file.path(root,"data")
  plot <- ggplot(tibble(x=c("A fairly long label","Another label"),y=c(.3,.8)),aes(x,y,fill=x))+
    geom_col()+theme_fig()+labs(title="Synthetic comparison",caption="Synthetic test data; no clinical interpretation.")
  plot <- plot + guides(fill=guide_legend(ncol=1))
  stems <- c("figure2v3_confusion_matrix","figure2v3_metric_bar","figure2v3_subtype_landscape","figure2v3_enrichment",
    as.vector(outer(c("figure4a_discrimination","figure4c_sensitivity_gleason","figure4d_sensitivity_somatic"),
      c("auc_platinum","cindex_platinum"),paste,sep="_")),
    as.vector(outer(c("figure4b_importance_platinum_cox","figure4b_importance_platinum_xgb"),
      c("landmark0","landmark90","landmark180"),paste,sep="_")),
    paste0("figure3b_sequencing_platinum_km_gene",1:7,"_snv_landmark0"),
    as.vector(outer(c("km_quintile_psa","km_quintile_testosterone"),c("landmark0","landmark90","landmark180"),paste,sep="_")))
  old_paths <- character()
  build <- function() for(stem in stems) {
    family <- if(startsWith(stem,"figure2")) "figure2v3_llm" else if(startsWith(stem,"figure4")) "figure4" else if(startsWith(stem,"figure3")) "figure3b" else
      paste0("labs/Androgen axis/",if(grepl("testosterone",stem)) "Testosterone" else "PSA","/km_quintile")
    destination <- file.path(output,"ADT","by_figure",family,stem,"platinum__all__incl")
    old_paths <<- c(old_paths,paste0(destination,".png"))
    p <- plot
    if(grepl("sensitivity",stem)) p <- p+labs(title="Sensitivity\n0d: n=40; +90d: n=39; +180d: n=38")
    getOption("compass.figure_capture")(p,destination,8,6,stem)
  }
  m <- prepare_figure_scenes(file.path(cache,"scenes"),"test",build)
  stopifnot(length(stems)==29, length(m$scenes)==11,
    !anyDuplicated(vapply(m$scenes,`[[`,character(1),"destination")),
    !any(grepl("/by_figure/",vapply(m$scenes,`[[`,character(1),"destination"))))
  stopifnot(sum(grepl("somatic_carrier_km_page",vapply(m$scenes,`[[`,character(1),"stem")))==2)
  for(scene in m$scenes) render_figure_scene(scene,"test",120,FALSE)
  prepared <- list(adt__platinum=m)
  config <- list(fig_root=output,cache_root=cache)
  figure_write_catalog(config,prepared)
  registry <- read_csv(file.path(output,"ADT","manifest.csv"),show_col_types=FALSE)
  stopifnot(nrow(registry)==11,all(file.exists(file.path(output,"ADT",registry$path))),
    !any(grepl(root,registry$path,fixed=TRUE)),file.exists(file.path(output,"ADT","index.html")))
  html <- paste(readLines(file.path(output,"ADT","index.html")),collapse="\n")
  urls <- regmatches(html,gregexpr('(?:href|src)="[^"]+"',html,perl=TRUE))[[1]]
  urls <- sub('^[^=]+="(.*)"$',"\\1",urls)
  stopifnot(all(file.exists(file.path(output,"ADT",utils::URLdecode(urls[!startsWith(urls,"#")])))),
    identical(figure_href("trajectories/PSA test/a.png"),"trajectories/PSA%20test/a.png"))
  # Known regenerated identities archive without losing bytes; unselected
  # endpoints stay untouched. Index generation does not copy PNG thumbnails.
  dir.create(dirname(old_paths[1]),recursive=TRUE)
  writeLines("old export",old_paths[1])
  unknown <- sub("platinum__","nepc__",old_paths[1]); writeLines("other cohort",unknown)
  figure_archive_old_exports(config,prepared)
  stopifnot(!file.exists(old_paths[1]),file.exists(unknown),
    length(list.files(file.path(cache,"previous_exports"),recursive=TRUE,pattern="png$"))==1,
    length(list.files(output,recursive=TRUE,pattern="rds$"))==0)
  # Every federated estimate survives composition with its original p/q values.
  estimates <- tidyr::expand_grid(source=c("Dana-Farber","Fred Hutch","Johns Hopkins","Federated*"),
    landmark_days=c(0,90,180),lab_name=c("PSA","Testosterone"),feature_stat=c("mean","min","max","last")) %>%
    mutate(hazard_ratio_per_sd=1.2,ci_lower=.9,ci_upper=1.5,p_value=.04,q_value=.2,
      source_kind=if_else(source=="Federated*","across_sites","within_site"),
      valid_estimate=TRUE,valid_ci=TRUE,significance="Nominal p < 0.05 only")
  for(lab in c("PSA","Testosterone")) {
    p <- plot_federated_comparison(estimates,lab,"Unverified population")
    stopifnot(nrow(p$data)==48,all(p$data$p_value==.04),all(p$data$q_value==.2))
    b <- ggplot_build(p)
    stopifnot(nrow(b$data[[4]])==48,length(b$layout$panel_params)==12,
      max(b$layout$layout$ROW)==3,max(b$layout$layout$COL)==4,
      identical(as.character(b$layout$layout$feature_stat[1:4]),c("Mean","Minimum","Maximum","Last")))
  }
  stopifnot(isTRUE(all.equal(unname(FEDERATED_FOREST_SLIDE_SIZE[["width"]] /
    FEDERATED_FOREST_SLIDE_SIZE[["height"]]),16/9)))
  counts <- tibble(site_name=rep(c("fred_hutch_caia_1_1","jhu_caia_1_1","dana_farber_caia_1_1"),each=3),
    landmark_days=rep(c(0,90,180),3),n_patients=rep(c(100,50,200),each=3),n_events=rep(c(10,5,30),each=3))
  across <- tibble(landmark_days=c(0,90,180),n_patients_used=150,n_events_used=15)
  note <- federated_population_note(across,counts)
  stopifnot(grepl("Fred Hutch + Johns Hopkins",note,fixed=TRUE),grepl("not verified",note,fixed=TRUE),
    grepl("unverified",federated_population_note(mutate(across,n_events_used=16),counts)))
})
cat("Publication compilation, paths, catalog, recoverable archives and federated data checks passed.\n")
