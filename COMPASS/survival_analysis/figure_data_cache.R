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
    if (anyDuplicated(stratum_values$DFCI_MRN)) stop("Duplicate patient IDs in trajectory label lookup")
    d <- inner_join(d, stratum_values %>% select(DFCI_MRN, stratum), by = "DFCI_MRN")
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

prepare_figure_scenes <- function(directory, signature, build, force = FALSE) {
  m <- if (!force) figure_scene_manifest(directory, signature) else NULL
  if (!is.null(m)) { message("Reused prepared figures: ", basename(dirname(directory))); return(m) }
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  generation <- tempfile("generation-", tmpdir = directory)
  dir.create(generation)
  scenes <- list(); tables <- character()
  capture <- function(plot, destination, width, height, stem) {
    # Resolve all aesthetics/stats now; renderer needs neither raw data nor fits.
    if (is.null(plot)) return(invisible(NULL))
    path <- file.path(generation, sprintf("panel-%04d.rds", length(scenes) + 1L))
    plot <- prepare_figure_text(plot, width)
    grob <- if (inherits(plot, "ggplot")) ggplot2::ggplotGrob(plot) else plot
    figure_atomic_rds(grob, path)
    scenes[[length(scenes) + 1L]] <<- list(path = path, destination = destination,
                                          width = width, height = height, stem = stem)
    invisible(NULL)
  }
  table_capture <- function(paths) tables <<- unique(c(tables, paths))
  old <- options(compass.figure_capture = capture, compass.figure_table_capture = table_capture,
                 device = function(...) grDevices::pdf(file = NULL, ...))
  on.exit(options(old), add = TRUE)
  build()
  paths <- vapply(scenes, `[[`, character(1), "path")
  m <- list(signature = signature, scenes = scenes, files = figure_file_identity(paths),
            tables = tables, table_files = figure_file_identity(tables))
  figure_atomic_rds(m, file.path(directory, "complete.rds"))
  m
}

render_figure_scene <- function(scene, signature, dpi, pdf, overwrite = FALSE) {
  grob <- NULL
  rendered <- 0L
  for (format in c("png", if (pdf) "pdf")) {
    path <- paste0(scene$destination, ".", format)
    receipt <- paste0(path, ".cache.rds")
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
  all(vapply(c("png", if (pdf) "pdf"), function(format) {
    path <- paste0(scene$destination, ".", format)
    receipt <- paste0(path, ".cache.rds")
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

run_cached_figure_workflow <- function(config, pipeline_path, stage = "all",
    prepare_workers = 1L, render_workers = 2L, dpi = 200, pdf = FALSE,
    prepare_overwrite = FALSE, render_overwrite = FALSE,
    metastatic_config = NULL, forest_config = NULL, federated_config = NULL) {
  stage <- match.arg(stage, c("all", "prepare", "render"))
  started <- proc.time()[["elapsed"]]
  dir.create(config$cache_root, recursive = TRUE, showWarnings = FALSE)
  for (name in c("data_root", "cache_root", "fig_root"))
    config[[name]] <- figure_absolute_path(config[[name]])
  request <- figure_object_hash(config)
  request_dir <- file.path(config$cache_root, "requests", request)
  dir.create(request_dir, recursive = TRUE, showWarnings = FALSE)
  manifest_path <- file.path(request_dir, "manifest.json")
  if (stage != "render") {
    python_config <- config
    python_config$manifest_path <- manifest_path
    python_config$force <- prepare_overwrite
    config_path <- tempfile("config-", tmpdir = request_dir, fileext = ".json")
    jsonlite::write_json(python_config, config_path, auto_unbox = TRUE, pretty = TRUE)
    python <- Sys.getenv("COMPASS_FIGURE_PYTHON", "python3")
    threads <- Sys.getenv("POLARS_MAX_THREADS", "")
    if (!nzchar(threads)) {
      allocation <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "4")))
      if (is.na(allocation) || allocation < 1L) allocation <- 4L
      threads <- as.character(min(4L, allocation))
    }
    if (!grepl("^[1-9][0-9]*$", threads)) stop("POLARS_MAX_THREADS must be a positive integer")
    message("Preparing/validating Polars figure data (threads=", threads, ") ...")
    status <- system2(python, c(shQuote(file.path(dirname(pipeline_path), "prepare_figure_data.py")),
                               "--config", shQuote(config_path)), env = paste0("POLARS_MAX_THREADS=", threads))
    if (status != 0L) stop("Figure preparation failed; see Python output above. Config: ", config_path)
  }
  if (!file.exists(manifest_path)) stop("No prepared snapshot for these settings; run COMPASS_FIGURE_STAGE=all or prepare first.")
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  old <- options(compass.figure_data_manifest = manifest)
  on.exit(options(old), add = TRUE)
  runtime <- list(R = as.character(getRversion()), packages = vapply(
    c("ggplot2", "survival", "survminer", "mgcv", "dplyr", "tidyr", "readr", "ggrepel", "scales", "stringr"),
    function(x) as.character(utils::packageVersion(x)), character(1)),
    code = unname(tools::md5sum(c(pipeline_path, file.path(dirname(pipeline_path), "figure_data_cache.R")))),
    fig_root = config$fig_root)
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
  build_job <- function(job) {
    if (!is.null(manifest$errors[[job$name]])) stop(manifest$errors[[job$name]])
    # Optional missing federation inputs produce a reported skip, not an empty
    # completion marker. A later delivery is picked up by source fingerprints.
    if (job$name == "federated") {
      source(federated_config$script, local = TRUE)
      inputs <- c(federated_config$results, vapply(c(0L, 90L, 180L),
        federated_no_msk_local_result_path, character(1), data_root = config$data_root))
      if (stage != "render" && any(!file.exists(inputs))) {
        warning("Federated supplement skipped; missing: ", paste(inputs[!file.exists(inputs)], collapse = ", "))
        return(list(skipped = TRUE))
      }
      build <- function() render_federated_no_msk_supplement(config$data_root, config$fig_root,
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
    prepare_figure_scenes(job$directory, job$signature, build, force = prepare_overwrite)
  }
  # Only pending jobs warm tables. A resumed all-mode knit reads no patient
  # Parquet at all when every scene snapshot is current.
  if (stage != "render") {
    pending <- Filter(function(job) job$name != "federated" &&
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
  invisible(list(prepared = prepared, rendered = render_results))
}
