# Startup regressions need only base R, independently of plotting packages/data.
path <- normalizePath("COMPASS/figures/ESMO_AI_Figures.R", mustWork = TRUE)
local({
  old <- options(compass.esmo.autorun = NULL)
  on.exit(options(old))

  # Exercise actual source(), as used by RStudio. Stop at the dependency-load
  # boundary so a successful startup never reads patient data or writes figures.
  env <- new.env(parent = globalenv())
  env$source <- function(...) {
    env$started <- TRUE
    stop(structure(list(message = "Reached dependency loading", call = NULL),
      class = c("esmo_startup_test", "error", "condition")))
  }
  messages <- character()
  error <- tryCatch(withCallingHandlers(base::source(path, local = env),
    message = function(m) {
      messages <<- c(messages, conditionMessage(m))
      invokeRestart("muffleMessage")
    }), error = identity)
  stopifnot(isTRUE(env$started), inherits(error, "esmo_startup_test"),
    any(grepl("ESMO AI figures: Results 1, 2, 3, 4", messages, fixed = TRUE)),
    any(grepl("Output directory:", messages, fixed = TRUE)))

  # The documented opt-out must still support function-only loading in tests.
  options(compass.esmo.autorun = FALSE)
  env <- new.env(parent = globalenv())
  env$source <- function(...) stop("Unexpected autorun")
  base::source(path, local = env)
  stopifnot(is.function(env$esmo_main), is.function(env$esmo_export))

  # --help must enter main through Rscript without loading plot dependencies.
  help <- system2(file.path(R.home("bin"), "Rscript"),
    args = c(shQuote(path), "--help"), stdout = TRUE, stderr = TRUE)
  stopifnot(is.null(attr(help, "status")),
    any(grepl("Generate Results 1-4", help, fixed = TRUE)),
    any(grepl("COMPASS_ESMO_OUTPUT_ROOT", help, fixed = TRUE)))
})
cat("ESMO AI entry point: source, function-only loading, and Rscript --help passed.\n")
