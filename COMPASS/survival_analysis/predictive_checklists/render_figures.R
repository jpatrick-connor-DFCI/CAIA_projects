#!/usr/bin/env Rscript
# Standalone R figures from held-out horizon metrics; no Python plotting.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  stop("Usage: Rscript render_figures.R <run-output-directory> [figure-directory]")
}
root <- normalizePath(args[[1L]], mustWork = TRUE)
out <- if (length(args) >= 2L) args[[2L]] else file.path(root, "figures")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
metrics <- read.csv(file.path(root, "metrics.csv"), check.names = FALSE,
                    stringsAsFactors = FALSE)
metrics <- metrics[metrics$status == "ok" & metrics$dataset == "test", , drop = FALSE]
if (nrow(metrics) == 0L) stop("No supported held-out metrics to plot")
models <- intersect(c("checklist", "age_cox", "elastic_net", "xgboost"), unique(metrics$model))
colors <- c(checklist = "#0072B2", age_cox = "#666666", elastic_net = "#D55E00", xgboost = "#009E73")[models]
plot_panel <- function(rows, metric, landmark, endpoint, population) {
  rows <- rows[rows$landmark_days == landmark, , drop = FALSE]
  upper_column <- paste0("bootstrap.intervals.", metric, ".upper")
  upper_values <- if (upper_column %in% names(rows)) rows[[upper_column]] else numeric()
  limits <- if (metric == "auc") c(0, 1) else c(0, max(c(.3, rows[[metric]], upper_values), na.rm = TRUE))
  x <- range(metrics$horizon_days / 365)
  if (diff(x) == 0) x <- x + c(-.1, .1)
  plot(NA, xlim = x, ylim = limits, xlab = "Years after prediction landmark",
       ylab = if (metric == "auc") "Time-dependent AUC" else "IPCW Brier score",
       main = sprintf("%s | ADT +%dd | %s", endpoint, landmark, population), las = 1)
  if (metric == "auc") abline(h = .5, col = "grey70", lty = 3)
  for (name in models) {
    d <- rows[rows$model == name, , drop = FALSE]
    d <- d[order(d$horizon_days), , drop = FALSE]
    if (!nrow(d)) next
    xx <- d$horizon_days / 365
    yy <- d[[metric]]
    lines(xx, yy, type = "b", pch = 16, col = colors[[name]], lwd = 2)
    lower <- paste0("bootstrap.intervals.", metric, ".lower")
    upper <- paste0("bootstrap.intervals.", metric, ".upper")
    if (all(c(lower, upper) %in% names(d))) {
      good <- is.finite(d[[lower]]) & is.finite(d[[upper]])
      segments(xx[good], d[[lower]][good], xx[good], d[[upper]][good], col = colors[[name]])
    }
  }
  legend(if (metric == "auc") "bottomright" else "topright", legend = models,
         col = colors, lty = 1, pch = 16, bty = "n", cex = .8)
}
for (population in unique(metrics$population)) {
  for (endpoint in unique(metrics$endpoint)) {
    rows <- metrics[metrics$population == population & metrics$endpoint == endpoint, , drop = FALSE]
    if (!nrow(rows)) next
    landmarks <- sort(unique(rows$landmark_days))
    draw <- function() {
      par(mfrow = c(length(landmarks), 2L), mar = c(4.2, 4.3, 3, 1))
      for (landmark in landmarks) {
        plot_panel(rows, "auc", landmark, endpoint, population)
        plot_panel(rows, "brier", landmark, endpoint, population)
      }
    }
    stem <- file.path(out, paste(endpoint, population, "held_out", sep = "_"))
    png(paste0(stem, ".png"), width = 2200, height = 720 * length(landmarks), res = 200,
        type = if (capabilities("cairo")) "cairo" else getOption("bitmapType"))
    draw(); dev.off()
    svg(paste0(stem, ".svg"), width = 11, height = 3.6 * length(landmarks))
    draw(); dev.off()
  }
}
calibration_path <- file.path(root, "calibration.csv")
if (file.exists(calibration_path)) {
  calibration <- read.csv(calibration_path, stringsAsFactors = FALSE)
  calibration <- calibration[calibration$model == "checklist", , drop = FALSE]
  draw_calibration <- function() {
  all_keys <- as.character(interaction(calibration$endpoint, calibration$landmark_days,
                                       calibration$population, calibration$horizon_days, drop = TRUE))
  for (key in unique(all_keys)) {
    d <- calibration[all_keys == key, , drop = FALSE]
    plot(d$mean_predicted_risk, d$weighted_observed_risk, xlim = c(0, 1), ylim = c(0, 1),
         xlab = "Mean predicted horizon risk", ylab = "IPCW observed horizon risk",
         main = as.character(key), pch = 16, col = "#0072B2", las = 1)
    abline(0, 1, lty = 2, col = "grey60")
  }
  }
  png(file.path(out, "checklist_calibration.png"), width = 1200, height = 1200, res = 200,
      type = if (capabilities("cairo")) "cairo" else getOption("bitmapType"))
  draw_calibration(); dev.off()
  svg(file.path(out, "checklist_calibration.svg"), width = 6, height = 6)
  draw_calibration(); dev.off()
}
message("Figures written to ", out)
