# R-only preparation for 05_figures.Rmd. Uses the existing intent labels,
# metastatic-diagnosis LLM task and dated regex stages; no model refits.
metastatic_parquet_backend <- function() {
  if (requireNamespace("arrow", quietly = TRUE)) return("arrow")
  if (requireNamespace("nanoparquet", quietly = TRUE)) return("nanoparquet")
  stop('Metastatic figures need an R Parquet reader. Run install.packages("nanoparquet") ',
       'in this R environment, then re-knit. The R package arrow is also supported.', call. = FALSE)
}

read_metastatic_parquet <- function(path, columns) {
  if (metastatic_parquet_backend() == "arrow")
    return(as.data.frame(arrow::read_parquet(path, col_select = columns)))
  nanoparquet::read_parquet(path, col_select = columns)
}

metastatic_normalize_id <- function(frame) {
  id <- suppressWarnings(as.numeric(as.character(frame$DFCI_MRN)))
  keep <- is.finite(id)
  frame <- frame[keep, , drop = FALSE]
  frame$DFCI_MRN <- format(trunc(id[keep]), scientific = FALSE, trim = TRUE)
  frame
}

metastatic_datetime <- function(values) {
  if (inherits(values, "POSIXt")) return(as.POSIXct(values, tz = "UTC"))
  if (inherits(values, "Date")) return(as.POSIXct(values, tz = "UTC"))
  values <- trimws(as.character(values))
  parsed <- suppressWarnings(readr::parse_datetime(values))
  for (format in c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y")) {
    missing <- is.na(parsed) & !is.na(values) & nzchar(values)
    if (!any(missing)) break
    parsed[missing] <- suppressWarnings(readr::parse_datetime(values[missing], format = format))
  }
  parsed
}

metastatic_stage_number <- function(values) {
  key <- toupper(trimws(as.character(values)))
  lookup <- c(`1`=1L, `2`=2L, `3`=3L, `4`=4L, I=1L, II=2L, III=3L, IV=4L,
              `1.0`=1L, `2.0`=2L, `3.0`=3L, `4.0`=4L)
  unname(lookup[key])
}

metastatic_collapse_stage <- function(values) {
  stage <- metastatic_stage_number(values)
  ifelse(is.na(stage), NA_character_, ifelse(stage == 4L, "Metastatic", "Local"))
}

build_metastatic_labels <- function(intent, notes, llm, analysis_anchors = NULL) {
  intent <- metastatic_normalize_id(intent) %>%
    transmute(DFCI_MRN, ADT_FIRST_DATE = metastatic_datetime(ADT_FIRST_DATE),
      ADT_LABEL = unname(c(LOCALIZED_ADJUVANT = "Local", METASTATIC = "Metastatic")[as.character(ADT_INTENT)]))
  if (anyDuplicated(intent$DFCI_MRN)) stop("ADT intent input must contain exactly one row per patient")
  if (is.null(analysis_anchors)) {
    # Standalone use without an analysis cohort retains the intent-file clock.
    intent <- intent %>% mutate(ANALYSIS_ANCHOR_DATE = ADT_FIRST_DATE)
  } else {
    if (!all(c("DFCI_MRN", "TREATMENT_ANCHOR_DATE") %in% names(analysis_anchors)))
      stop("Metastatic supplement requires patient DFCI_MRN and TREATMENT_ANCHOR_DATE")
    anchors <- metastatic_normalize_id(analysis_anchors) %>%
      transmute(DFCI_MRN, ANALYSIS_ANCHOR_DATE = metastatic_datetime(TREATMENT_ANCHOR_DATE))
    if (anyDuplicated(anchors$DFCI_MRN)) stop("Analysis anchors must contain exactly one row per patient")
    # Match by patient ID, never row order. Preserve patients without an intent
    # label so their independent LLM and regex evidence is still represented.
    intent <- anchors %>% left_join(intent, by = "DFCI_MRN")
  }
  intent <- intent %>% mutate(ANCHOR_DELTA_DAYS = as.numeric(
    as.Date(ANALYSIS_ANCHOR_DATE, tz = "UTC") - as.Date(ADT_FIRST_DATE, tz = "UTC")))
  llm <- metastatic_normalize_id(llm) %>%
    mutate(.verdict = tolower(as.character(has_metastatic_disease))) %>%
    group_by(DFCI_MRN) %>% summarise(LLM_LABEL = case_when(
      any(.verdict %in% c("true", "1", "1.0")) ~ "Metastatic",
      any(.verdict %in% c("false", "0", "0.0")) ~ "Local",
      TRUE ~ NA_character_), .groups = "drop")
  notes <- metastatic_normalize_id(notes) %>%
    transmute(DFCI_MRN, .date = metastatic_datetime(EVENT_DATE),
              .stage = metastatic_stage_number(DERIVED_STAGE_MERGED)) %>%
    filter(!is.na(.date), !is.na(.stage)) %>%
    inner_join(intent %>% select(DFCI_MRN, ANALYSIS_ANCHOR_DATE), by = "DFCI_MRN") %>%
    mutate(.days = trunc(as.numeric(difftime(.date, ANALYSIS_ANCHOR_DATE, units = "days"))))
  before <- notes %>% filter(.days <= 0)
  nearest <- before %>% filter(.days >= -365) %>% arrange(desc(.days), desc(.stage)) %>%
    distinct(DFCI_MRN, .keep_all = TRUE) %>% transmute(DFCI_MRN, REGEX_STAGE = .stage)
  max_before <- before %>% group_by(DFCI_MRN) %>%
    summarise(REGEX_MAX_BEFORE_STAGE = if (length(.stage)) max(.stage) else NA_integer_, .groups = "drop")
  max_after <- notes %>% filter(.days > 0) %>% group_by(DFCI_MRN) %>%
    summarise(REGEX_MAX_AFTER_STAGE = if (length(.stage)) max(.stage) else NA_integer_, .groups = "drop")
  # Maximum across the entire record: pre- and post-ADT staging pooled, with no
  # anchor window. Post-ADT progression therefore counts toward the label, so
  # this is not a point-in-time status at ADT the way REGEX_LABEL is.
  max_any <- notes %>% group_by(DFCI_MRN) %>%
    summarise(REGEX_MAX_ANY_STAGE = if (length(.stage)) max(.stage) else NA_integer_, .groups = "drop")
  intent %>% left_join(llm, by = "DFCI_MRN") %>% left_join(nearest, by = "DFCI_MRN") %>%
    left_join(max_before, by = "DFCI_MRN") %>% left_join(max_after, by = "DFCI_MRN") %>%
    left_join(max_any, by = "DFCI_MRN") %>%
    mutate(REGEX_LABEL = metastatic_collapse_stage(REGEX_STAGE),
           REGEX_MAX_BEFORE = metastatic_collapse_stage(REGEX_MAX_BEFORE_STAGE),
           REGEX_MAX_AFTER = metastatic_collapse_stage(REGEX_MAX_AFTER_STAGE),
           REGEX_MAX_ANY = metastatic_collapse_stage(REGEX_MAX_ANY_STAGE)) %>%
    select(DFCI_MRN, ADT_FIRST_DATE, ANALYSIS_ANCHOR_DATE, ANCHOR_DELTA_DAYS,
           ADT_LABEL, LLM_LABEL, REGEX_LABEL, REGEX_MAX_BEFORE, REGEX_MAX_AFTER,
           REGEX_MAX_ANY)
}

metastatic_icd_site <- function(values) {
  code <- gsub("[^A-Z0-9]", "", toupper(trimws(as.character(values))))
  valid <- !is.na(code) & nchar(code) >= 4L & grepl("^C7[789]", code) & code != "C799"
  site <- rep(NA_character_, length(code))
  site[valid] <- "other"
  # Mirrors validate_adt_intent.py: spinal/cranial-nerve codes remain "other";
  # unspecified C79.9 and C7B neuroendocrine codes do not count as organ sites.
  prefixes <- c(C795="bone", C7931="brain", C7932="brain", C797="adrenal",
                C780="lung", C782="lung", C787="liver", C786="peritoneal", C77="node")
  for (prefix in names(prefixes)) site[valid & startsWith(code, prefix)] <- prefixes[[prefix]]
  site
}

metastatic_burden_at_adt <- function(icds, labels) {
  groups <- c("brain", "bone", "liver", "lung", "node", "adrenal", "peritoneal", "other")
  anchors <- labels %>% select(DFCI_MRN, ANALYSIS_ANCHOR_DATE) %>% filter(!is.na(ANALYSIS_ANCHOR_DATE))
  coded <- metastatic_normalize_id(icds) %>% inner_join(anchors, by = "DFCI_MRN") %>%
    mutate(.date = metastatic_datetime(START_DT), .site = metastatic_icd_site(DIAGNOSIS_ICD10_CD)) %>%
    filter(!is.na(.date), !is.na(.site), .date <= ANALYSIS_ANCHOR_DATE) %>% distinct(DFCI_MRN, .site)
  base <- labels %>% distinct(DFCI_MRN)
  for (group in groups)
    base[[paste0("MET_SITE_", group)]] <- as.integer(base$DFCI_MRN %in% coded$DFCI_MRN[coded$.site == group])
  # With no analysis date, "before ADT" is unknown rather than zero burden.
  unavailable <- !base$DFCI_MRN %in% anchors$DFCI_MRN
  for (column in paste0("MET_SITE_", groups)) base[[column]][unavailable] <- NA_integer_
  base$N_MET_SITES <- rowSums(base[paste0("MET_SITE_", groups)])
  base
}

prepare_metastatic_figure_labels <- function(config, analysis_anchors = NULL) {
  intent <- readr::read_csv(config$intent, show_col_types = FALSE,
    col_select = all_of(c("DFCI_MRN", "ADT_FIRST_DATE", "ADT_INTENT")),
    col_types = readr::cols(.default = readr::col_character()))
  notes <- read_metastatic_parquet(config$stage, c("DFCI_MRN", "EVENT_DATE", "DERIVED_STAGE_MERGED"))
  llm <- read_metastatic_parquet(config$llm, c("DFCI_MRN", "has_metastatic_disease"))
  labels <- build_metastatic_labels(intent, notes, llm, analysis_anchors)
  if (!is.null(config$icd) && file.exists(config$icd)) {
    icds <- readr::read_csv(config$icd, show_col_types = FALSE,
      col_select = all_of(c("DFCI_MRN", "DIAGNOSIS_ICD10_CD", "START_DT")),
      col_types = readr::cols(.default = readr::col_character()))
    labels <- labels %>% left_join(metastatic_burden_at_adt(icds, labels), by = "DFCI_MRN")
  }
  labels
}
