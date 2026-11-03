#!/usr/bin/env Rscript
# From the repository root:
# Rscript COMPASS/figures/preview_03_05.R <exported-ADT> [output-ADT]
# Defaults to manuscript/ inside each input figure section.
# Aggregate-export previews only. Production figures use the raw testing family.
args <- commandArgs(TRUE)
if(!length(args) || length(args)>2L) stop("Expected exported ADT directory and optional output ADT root")
input <- normalizePath(args[1],mustWork=TRUE)
output <- if(length(args)==2L) args[2] else input
source("COMPASS/survival_analysis/figure_pipeline/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_pipeline/figure_workflow.R")
figure3_output <- manuscript_folder(output,"federated_03_univariable")
figure5_output <- manuscript_folder(output,"federated_05_transfer_no_dfci")
for(folder in c(figure3_output,figure5_output))
  dir.create(file.path(folder,"data"),recursive=TRUE,showWarnings=FALSE)
save_review <- function(plot,stem,width,height,folder) {
  figure_require_font()
  plot <- figure_style_plot(plot)
  ggplot2::ggsave(file.path(folder,paste0(stem,".pdf")),plot,width=width,height=height,
    device=figure_pdf_device,bg="white")
}

associations <- bind_rows(lapply(c(0,90,180),function(lm) readr::read_csv(file.path(input,
  "03_univariate/federated/data",paste0("federated_volcano_landmark",lm,"__platinum.csv")),
  show_col_types=FALSE)))
means <- prepare_mean_univariable(associations)
readr::write_csv(means,file.path(figure3_output,"data","figure3_mean_associations.csv"))
counts <- means %>% group_by(landmark_days) %>% summarise(
  mean_tests=first(n_mean_tests),displayed=sum(!lab_name %in% DROP),
  significant=sum(q_value<.05 & !lab_name %in% DROP,na.rm=TRUE),.groups="drop")
readr::write_csv(counts,file.path(figure3_output,"data","figure3_counts.csv"))
plots <- lapply(c(0,90,180),function(lm) plot_volcano_panel(filter(means,landmark_days==lm),
  paste0("Day ",lm)) + labs(subtitle=paste0("Day ",lm)))
for(i in 1:3) plots[[i]] <- plots[[i]] +
  coord_cartesian(xlim=c(1,exp(max(.75,max(means$risk_coef,na.rm=TRUE)*1.15))),ylim=c(-.2,36)) +
  scale_size_manual(values=c(`TRUE`=1.8,`FALSE`=1.2),guide="none") +
  guides(fill=guide_legend(nrow=1,override.aes=list(shape=21,size=1.8,alpha=1))) +
  labs(y=if(i==1) "−log10(q)" else NULL)
spec <- list(key="federated_03_univariable",width=10,height=4.5,
  layout=rbind(1:3,rep(4,3)),row_heights=c(1,.15),titles=rep("",3),shared_legend_panels=1:3)
save_review(manuscript_combine(plots,spec),"figure3_risk_direction",10,4.5,figure3_output)

# A deliberately selected, explicitly labeled alternative; q values still
# reflect the entire mean-feature family, not these ten displayed examples.
selected <- c("Testosterone","PSA","Hemoglobin","Hematocrit","RBC","RDW",
  "Albumin","Alkaline phosphatase","LDH","Platelets")
forest <- means %>% filter(lab_name %in% selected) %>%
  mutate(lab_name=factor(lab_name,levels=rev(selected)),
    landmark=factor(paste0("Day ",landmark_days),levels=paste0("Day ",c(0,90,180))),
    significant=q_value<.05)
forest_plot <- ggplot(forest,aes(risk_hr,lab_name)) +
  geom_vline(xintercept=1,color="grey65",linetype="dashed") +
  geom_linerange(aes(xmin=risk_ci_lower,xmax=risk_ci_upper),linewidth=.5) +
  geom_point(aes(shape=significant),size=2) +
  geom_text(aes(label=risk_direction,x=2.2),hjust=0,size=2.6) +
  scale_x_log10(breaks=c(1,1.25,1.5,2),limits=c(.95,2.8)) +
  scale_shape_manual(values=c(`TRUE`=16,`FALSE`=1),limits=c("FALSE","TRUE"),
    labels=c("q >= 0.05","q < 0.05"),drop=FALSE,name=NULL) +
  facet_grid(.~landmark) + labs(x="HR per SD in the labeled direction (95% CI)",y=NULL,
    title="Alternative: ten selected labs, with uncertainty",
    caption="Higher/Lower denotes the observed direction; BH correction includes all available mean tests.") +
  theme_fig() + theme(strip.background=element_blank(),strip.text=element_text(face="bold"))
save_review(forest_plot,"figure3_selected_forest",10,4.8,figure3_output)

metrics <- readr::read_csv(file.path(input,"05_transfer_survival/data/transfer_discrimination_no_dfci__platinum.csv"),
  show_col_types=FALSE)
clinical <- readr::read_csv(file.path(input,"05_transfer_survival/data/transfer_clinical_comparison_no_dfci__platinum.csv"),
  show_col_types=FALSE)
km_paths <- file.path(input,"05_transfer_survival/km/data",paste0(FEDERATED_TRANSFER_KM_TABLES,"__platinum.csv"))
km <- if(all(file.exists(km_paths))) setNames(lapply(km_paths,readr::read_csv,show_col_types=FALSE),
  names(FEDERATED_TRANSFER_KM_TABLES)) else NULL
for(landmarks in list(c(0,90,180),180)) {
  result <- manuscript_federated_transfer_no_dfci(metrics,clinical,include=c("gleason","trio"),
    key="federated_05_transfer_no_dfci__no_d",settings=list(layout="interval",landmarks=landmarks))
  stem <- if(length(landmarks)==1) "figure5_day180" else "figure5_all_landmarks"
  save_review(result$plot,stem,result$spec$width,result$spec$height,figure5_output)
  if(!is.null(km)) {
    full <- manuscript_federated_transfer_no_dfci(metrics,clinical,km,include=c("gleason","trio"),
      key="federated_05_transfer_no_dfci__no_d",settings=list(layout="interval",landmarks=landmarks))
    save_review(full$plot,paste0(stem,"_full"),full$spec$width,full$spec$height,figure5_output)
  }
}

# Use the supplied paired bootstrap differences, never differences of marginal CIs.
combined <- clinical %>% filter(model=="combined_cox",landmark_days==180,
  subpopulation %in% c("gleason","trio"))
delta <- bind_rows(
  mutate(combined,comparison="Add score to clinical",value=delta_auc_vs_clinical,
    lo=delta_auc_vs_clinical_lo,hi=delta_auc_vs_clinical_hi),
  mutate(combined,comparison="Add clinical to score",value=delta_auc_vs_transferred,
    lo=delta_auc_lo,hi=delta_auc_hi)) %>%
  mutate(endpoint=factor(endpoint,levels=c("platinum","avpc","nepc"),labels=c("Platinum","AVPC","NEPC")),
    population=factor(subpopulation,levels=c("gleason","trio"),
      labels=c("Gleason\nADT day 180","TP53/RB1/PTEN\nSequencing day 180")),
    model=factor(model_family,levels=c("xgboost_cox","elastic_net_cox"),labels=c("XGBoost","Elastic-net Cox")))
delta_plot <- ggplot(delta,aes(comparison,value,color=model,group=model)) +
  geom_hline(yintercept=0,color="grey65",linetype="dashed") +
  geom_linerange(aes(ymin=lo,ymax=hi),position=position_dodge(.5),linewidth=.5) +
  geom_point(position=position_dodge(.5),size=2) + coord_flip() +
  scale_color_manual(values=c(XGBoost="#B58900",`Elastic-net Cox`="#2a78d6"),name=NULL) +
  facet_grid(population~endpoint) + labs(x=NULL,y="Change in mean AUC(t) (paired 95% CI)",
    title="Alternative: incremental value at day 180",
    caption="Differences and intervals are the paired bootstrap results supplied in the clinical-comparison export.") +
  theme_fig() + theme(strip.background=element_blank(),strip.text=element_text(face="bold"),
    strip.text.y=element_text(angle=0,size=9))
save_review(delta_plot,"figure5_paired_gain",11,5,figure5_output)
writeLines(c("Design previews from aggregate Desktop exports; original images are unchanged.",
  "Figure 3: BH uses every available mean-feature P value in each exported landmark table, including hidden height.",
  "These exports have already passed the old stability filters. Rebuild production from raw Cox results for final FDR values.",
  "The forest alternative shows ten selected labs; q values are not recomputed on that selection."),
  file.path(figure3_output,"data","preview_notes.txt"))
writeLines(c("Design previews from aggregate transfer exports.",
  "All-landmark and day-180 previews show panels a-c; variants ending in _full include KM and 5-year probability panels.",
  "Mean-landmark set 1: full cohort, Gleason-only cohort and somatic cohort. Set 2: full cohort and combined Gleason/somatic cohort.",
  "Mean-landmark variants require all three day-0/90/180 AUCs and show their equal-weight mean without a confidence interval.",
  "The paired-gain alternative uses the supplied paired bootstrap differences and confidence intervals."),
  file.path(figure5_output,"data","preview_notes.txt"))
print(counts)
