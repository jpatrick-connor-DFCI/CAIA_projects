# Federated-only PSA/testosterone forests and site cohort counts. Significance uses the supplied
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

federated_site_label <- function(site_name) {
  labels <- c(dana_farber_caia_1_1 = "Dana-Farber", fred_hutch_caia_1_1 = "Fred Hutch",
              jhu_caia_1_1 = "Johns Hopkins")
  coalesce(unname(labels[site_name]), site_name)
}

validate_federated_sites <- function(site_name) {
  if (any(is.na(site_name) | !nzchar(trimws(site_name)))) stop("Missing federated site name")
  if (any(grepl("msk|sloan", site_name, ignore.case = TRUE))) stop("MSK site found in no-MSK inputs")
}

load_federated_no_msk_forest <- function(path, within_site = FALSE) {
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

plot_federated_no_msk_forest <- function(d, landmark, site_name = NULL) {
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
      title = sprintf("Federated no-MSK | %s | ADT platinum | +%d days", scope, landmark),
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

load_federated_no_msk_sites <- function(federated_path) {
  path <- file.path(dirname(federated_path), "nvflare_within_site_cox_univariate",
                    "cox_within_site_all_sites_cohort.csv")
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

plot_federated_no_msk_sites <- function(d) {
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
    "Federated no-MSK | ADT platinum incidence by site | landmark 0d",
    sprintf("Hatched/* = <25 events (%d of %d available sites); read as power, not biology", sum(good$thin), nrow(good)),
    sep = "\n"), gp = grid::gpar(fontsize = 14)),
    bottom = grid::textGrob("Observed events / analyzed patients during follow-up; not fixed-horizon cumulative incidence.",
      gp = grid::gpar(fontsize = 10)), padding = grid::unit(2.3, "lines"))
}

save_federated_no_msk_panel <- function(plot, path, width, height, dpi, overwrite) {
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
    geom_point(data=filter(d,valid_estimate),aes(shape=significance),size=2.8,stroke=.8) +
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
    facet_grid(feature_stat ~ landmark,drop=FALSE) +
    labs(title=paste(analyte,"associations across sites"),
      subtitle="ADT · platinum endpoint · no MSK · hazard ratios per SD with 95% confidence intervals",
      x="Hazard ratio per SD (log scale)",y=NULL,
      caption=paste(population_note,count_note,
        "Open: not significant; filled: nominal p < 0.05 only; diamond: supplied FDR q < 0.05. No FDR recalculation.",
        "Exact HR/CI/p/q, modeled and observed counts, and fitting metadata are in the accompanying CSV; no delta features.",sep="\n")) +
    theme_classic(base_size=11) +
    theme(strip.background=element_blank(),strip.text=element_text(face="bold"),
      panel.spacing=grid::unit(.9,"lines"),legend.position="bottom",
      legend.text=element_text(margin=margin(l=5,r=10)),
      plot.caption=element_text(hjust=0,size=8),plot.title.position="plot",plot.caption.position="plot",
      plot.margin=margin(12,18,12,12))
}

render_federated_no_msk_supplement <- function(data_root, fig_root, federated_path,
                                                dpi = 200, overwrite = FALSE) {
  d <- load_federated_no_msk_forest(federated_path)
  root <- file.path(fig_root, "ADT", "federated_no_msk")
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  capture <- getOption("compass.figure_table_capture")
  within <- NULL
  within_path <- file.path(dirname(federated_path), "nvflare_within_site_cox_univariate",
                           "cox_within_site_all_sites_results.csv")
  if (file.exists(within_path)) {
    within <- load_federated_no_msk_forest(within_path, within_site = TRUE)
  } else warning("Within-site federated forests unavailable; missing: ", within_path)
  sites <- load_federated_no_msk_sites(federated_path)
  estimates <- prepare_federated_comparison(d,within)
  note <- federated_population_note(d,sites)
  estimates$population_note <- ifelse(estimates$source_kind=="across_sites",note,"Within-site model")
  for(analyte in c("PSA","Testosterone")) {
    base <- file.path(root,paste0(tolower(analyte),"_forest__platinum"))
    readr::write_csv(filter(estimates,lab_name==analyte),paste0(base,".csv"))
    if(is.function(capture)) capture(paste0(base,".csv"))
    save_federated_no_msk_panel(plot_federated_comparison(estimates,analyte,note),paste0(base,".png"),12,10, dpi,overwrite)
  }
  if (!is.null(sites)) {
    site_path <- file.path(root, "site_incidence_lm000__platinum.csv")
    dir.create(dirname(site_path), recursive = TRUE, showWarnings = FALSE)
    readr::write_csv(prepare_federated_site_incidence(sites), site_path)
    if (is.function(capture)) capture(site_path)
    save_federated_no_msk_panel(plot_federated_no_msk_sites(sites),
      file.path(root, "site_incidence_lm000__platinum.png"), 14, 5.5, dpi, overwrite)
  }
  invisible(d)
}
