# Rscript tests/test_metastatic_figure_labels.R
suppressPackageStartupMessages({library(dplyr); library(tidyr)})
source("COMPASS/survival_analysis/prepare_metastatic_figure_labels.R")
intent <- tibble(DFCI_MRN=1:6, ADT_FIRST_DATE="2020-01-01",
  ADT_INTENT=c("METASTATIC", "LOCALIZED_ADJUVANT", "METASTATIC",
               "LOCALIZED_ADJUVANT", "INDETERMINATE", "LOCALIZED_ADJUVANT"))
notes <- tibble(DFCI_MRN=c(1,2,3,3,4,4,5,5,6),
  EVENT_DATE=c(rep("2019-12-01",3), "2020-02-01", rep("2019-12-01",4), "2017-01-01"),
  DERIVED_STAGE_MERGED=c(4,1,3,4,2,4,2,5,4))
llm <- tibble(DFCI_MRN=c(1,2,2,3,4,5), has_metastatic_disease=c(TRUE,FALSE,TRUE,FALSE,NA,FALSE))
labels <- build_metastatic_labels(intent, notes, llm)
stopifnot(identical(labels$REGEX_LABEL, c("Metastatic","Local","Local","Metastatic","Local",NA)),
          identical(labels$LLM_LABEL, c("Metastatic","Metastatic","Local",NA,"Local",NA)),
          identical(labels$REGEX_MAX_AFTER, c(NA,NA,"Metastatic",NA,NA,NA)),
          labels$REGEX_MAX_BEFORE[6] == "Metastatic", is.na(labels$ADT_LABEL[5]))
empty <- build_metastatic_labels(intent, notes[FALSE,], llm[FALSE,])
stopifnot(nrow(empty)==6, all(is.na(empty$REGEX_LABEL)), all(is.na(empty$LLM_LABEL)))
failure <- tryCatch(build_metastatic_labels(bind_rows(intent,intent[1,]), notes,llm), error=identity)
stopifnot(inherits(failure,"error"), grepl("one row per patient", conditionMessage(failure)))
stage <- c("I","II","III","IV","1","2","3","4",NA,"0","V","unknown","4.0")
stopifnot(identical(metastatic_collapse_stage(stage),
  c(rep("Local",3),"Metastatic",rep("Local",3),"Metastatic",rep(NA_character_,4),"Metastatic")))
dates <- metastatic_datetime(c("2020-01-01", "2020-01-01T00:00:00Z", "01/01/2020",
                                "01/01/2020 00:00:00", "invalid", NA))
stopifnot(length(unique(na.omit(dates))) == 1, sum(is.na(dates)) == 2)
ids <- metastatic_normalize_id(tibble(DFCI_MRN=c("1", "2.0", NA, "invalid")))
stopifnot(identical(ids$DFCI_MRN,c("1","2")))

# Distinct organs, not code counts. Post-ADT codes and unspecified sites excluded.
icds <- tibble(DFCI_MRN=c(1,1,1,1,2,2,3,4),
  DIAGNOSIS_ICD10_CD=c("C79.51","C7952","c78.7","C78.01","C79.9","C7B.8","C79.31","C77.1"),
  START_DT=c("2019-12-01","12/02/2019","2019-12-01","2020-02-01",
             "2019-12-01","2019-12-01","2020-01-01","2019-12-01"))
burden <- metastatic_burden_at_adt(icds, labels)
stopifnot(identical(as.numeric(burden$N_MET_SITES), c(2,0,1,1,0,0)),
          burden$MET_SITE_bone[1]==1, burden$MET_SITE_liver[1]==1, burden$MET_SITE_lung[1]==0)
stopifnot(identical(metastatic_icd_site(c("C7931","C7949","C782","C786","C7972","C78","C799","C61")),
                    c("brain","other","lung","peritoneal","adrenal",NA,NA,NA)))

# Intent history can predate the model's post-diagnosis treatment anchor.
# Recompute every dated label/ICD window by ID, even with shuffled anchor rows.
anchors <- tibble(DFCI_MRN=c(7,3,1,2,4,5,6),
  TREATMENT_ANCHOR_DATE=as.Date(c(rep("2020-03-01",3),rep("2020-01-01",3),NA)))
more_notes <- bind_rows(notes, tibble(DFCI_MRN=7, EVENT_DATE="2020-02-01", DERIVED_STAGE_MERGED=4))
more_llm <- bind_rows(llm, tibble(DFCI_MRN=7, has_metastatic_disease=TRUE))
aligned <- build_metastatic_labels(intent, more_notes, more_llm, anchors)
patient3 <- aligned[aligned$DFCI_MRN=="3",]
patient7 <- aligned[aligned$DFCI_MRN=="7",]
stopifnot(identical(aligned$DFCI_MRN,as.character(anchors$DFCI_MRN)),
          patient3$REGEX_LABEL=="Metastatic", is.na(patient3$REGEX_MAX_AFTER),
          patient3$ANCHOR_DELTA_DAYS==60,
          as.Date(patient3$ADT_FIRST_DATE)==as.Date("2020-01-01"),
          as.Date(patient3$ANALYSIS_ANCHOR_DATE)==as.Date("2020-03-01"),
          patient3$ADT_LABEL=="Metastatic", patient3$LLM_LABEL=="Local",
          is.na(patient7$ADT_LABEL), patient7$LLM_LABEL=="Metastatic",
          patient7$REGEX_LABEL=="Metastatic",
          is.na(aligned$REGEX_MAX_BEFORE[aligned$DFCI_MRN=="6"]))
aligned_burden <- metastatic_burden_at_adt(icds,aligned)
stopifnot(aligned_burden$N_MET_SITES[aligned_burden$DFCI_MRN=="1"]==3,
          is.na(aligned_burden$N_MET_SITES[aligned_burden$DFCI_MRN=="6"]))
failure <- tryCatch(build_metastatic_labels(intent,notes,llm,bind_rows(anchors,anchors[1,])),error=identity)
stopifnot(inherits(failure,"error"), grepl("one row per patient",conditionMessage(failure)))

# Prove that the active R workflow has no external Python execution/configuration.
active <- paste(readLines("COMPASS/survival_analysis/05_figures.Rmd"),
                collapse="\n")
renderer <- paste(readLines("COMPASS/survival_analysis/metastatic_figure_supplements.R"),collapse="\n")
stopifnot(!grepl("COMPASS_FIGURE_PYTHON|prepare_metastatic_figure_labels.py|system2\\(", active),
          !grepl("system2\\(|config\\$python", renderer))
cat("R metastatic labels: stage collapse, windows/ties, missingness, IDs, dates, ICD burden, and R-only execution passed.\n")
