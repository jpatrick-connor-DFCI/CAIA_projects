# Run from the repository root: Rscript tests/test_figure2_subtypes.R
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(stringr)
})
pipeline <- new.env(parent = globalenv())
needed <- c("COLOR_PLATINUM_POS", "COLOR_PLATINUM_NEG", "COLOR_NEUTRAL_INK",
            "theme_fig", "prepare_figure_text", "wilson_ci", "CLASS_ORDER", "CLASS_LABELS", "count_labels",
            "compute_enrichment", "render_landscape_panel", "render_enrichment_panel", "render_confusion_panel")
collect <- function(expr) {
  if (missing(expr) || !is.call(expr)) return(invisible(NULL))
  if (identical(expr[[1]], as.name("<-")) && is.symbol(expr[[2]]) &&
      as.character(expr[[2]]) %in% needed) {
    eval(expr, pipeline)
  } else {
    for (child in as.list(expr)[-1]) collect(child)
  }
}
for (expr in parse("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R"))
  collect(expr)

# Columns are manual truth, rows are LLM predictions. Unequal FP/FN expose
# the transpose that previously contradicted the reported precision/recall.
cm <- pipeline$render_confusion_panel(list(TN=28,FP=1,FN=2,TP=10))$data
stopifnot(cm$n[cm$truth=="NEPC" & cm$pred=="Non-NEPC"]==2,
          cm$n[cm$truth=="Non-NEPC" & cm$pred=="NEPC"]==1,
          sum(cm$n[cm$truth=="NEPC"])==12)

labels <- tibble(
  primary_label = factor(c(rep("avpc", 4), rep("nepc", 2),
                           rep("conventional", 6), rep("biomarker", 2), NA),
                         levels = pipeline$CLASS_ORDER),
  is_platinum = c(rep(TRUE, 4), rep(FALSE, 2), TRUE, rep(FALSE, 5), TRUE, TRUE, TRUE)
)
enrichment <- pipeline$compute_enrichment(labels)
expected <- matrix(c(4L, 2L, 1L, 5L), nrow = 2, byrow = TRUE)
ft <- fisher.test(expected, alternative = "greater")
stopifnot(identical(unname(enrichment$ct), expected),
          enrichment$n_excluded == 3L,
          enrichment$n_aggressive == 6L, enrichment$n_conventional == 6L,
          isTRUE(all.equal(enrichment$OR, unname(ft$estimate))),
          isTRUE(all.equal(enrichment$p_value, ft$p.value)),
          isTRUE(all.equal(unname(enrichment$p_agg), 4 / 6)),
          isTRUE(all.equal(unname(enrichment$p_conv), 1 / 6)),
          isTRUE(all.equal(unname(c(enrichment$lo_agg, enrichment$hi_agg)),
                           unname(pipeline$wilson_ci(4, 6)[2:3]))))

classified <- labels %>% filter(!is.na(primary_label))
positive <- pipeline$count_labels(filter(classified, is_platinum)) %>%
  mutate(platinum_status = "positive")
negative <- pipeline$count_labels(filter(classified, !is_platinum)) %>%
  mutate(platinum_status = "negative")
stopifnot(sum(positive$count) == 7L, sum(negative$count) == 7L,
          abs(sum(positive$frac) - 1) < 1e-12,
          abs(sum(negative$frac) - 1) < 1e-12)
landscape <- pipeline$render_landscape_panel(bind_rows(positive, negative), 7, 7)
contrast <- pipeline$render_enrichment_panel(enrichment)
invisible(ggplot_build(landscape))
invisible(ggplot_build(contrast))
stopifnot(nrow(contrast$data) == 2L,
          identical(as.character(contrast$data$group),
                    c("Aggressive\n(AVPC + NEPC)", "Conventional")))

# Reproduce the cached-scene path: size text on a PDF device, then render the
# resulting grob on a raster device. Stacked legend labels must occupy different
# rows even with large cohort counts. This only changes layout, not the bars.
cached_landscapes <- list()
grDevices::pdf(file = NULL)
for (counts in list(c(213L, 3541L), c(123456L, 987654L))) {
  panel <- pipeline$render_landscape_panel(bind_rows(positive, negative), counts[1], counts[2],
    title = "Subtype landscape by platinum status") +
    labs(caption = str_wrap(paste("Synthetic layout fixture; subtype fractions are test data.",
                                 "Long cohort labels exercise the cached figure export."), 85)) +
    theme(plot.caption = element_text(size = 8, color = pipeline$COLOR_NEUTRAL_INK, hjust = .5))
  stopifnot(identical(ggplot_build(panel)$data[[1]], ggplot_build(landscape)$data[[1]]),
            as.numeric(panel$theme$axis.title.y$margin[2]) >= 14,
            as.numeric(panel$theme$axis.text.y$margin[2]) >= 6)
  for (width in c(9.5, 6.5)) {
    grob <- ggplotGrob(pipeline$prepare_figure_text(panel, width))
    boxes <- grob$grobs[grepl("^guide-box", grob$layout$name)]
    box <- Filter(function(x) inherits(x, "gtable"), boxes)[[1]]
    guide <- box$grobs[[which(box$layout$name == "guides")]]
    label_cells <- guide$layout[grepl("^label", guide$layout$name), ]
    key_cells <- guide$layout[grepl("^key", guide$layout$name), ]
    stopifnot(nrow(label_cells) == 2L, length(unique(label_cells$t)) == 2L,
              length(unique(label_cells$l)) == 1L,
              all(key_cells$r < min(label_cells$l)))
    key <- sprintf("panel_b_cached_%s_%s", counts[1], width)
    cached_landscapes[[key]] <- list(grob = grob, width = width)
  }
}
invisible(grDevices::dev.off())

# Missing comparison groups should produce an explanatory panel, not crash the cell.
for (subset in list(filter(labels, primary_label == "conventional"), labels[0, ])) {
  result <- pipeline$compute_enrichment(subset)
  stopifnot(is.na(result$OR), is.na(result$p_value))
  invisible(ggplot_build(pipeline$render_enrichment_panel(result)))
}
invisible(ggplot_build(pipeline$render_landscape_panel(positive[0, ], 0, 0)))

# Optional local preview of the actual restored plotting functions.
review_dir <- Sys.getenv("COMPASS_FIGURE2_REVIEW_DIR", "")
if (nzchar(review_dir)) {
  dir.create(review_dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(review_dir, "panel_b.png"), landscape, width = 6.5, height = 8,
         dpi = 120, device = ragg::agg_png)
  ggsave(file.path(review_dir, "panel_c.png"), contrast, width = 4.5, height = 5.5,
         dpi = 120, device = ragg::agg_png)
  for (key in names(cached_landscapes)) {
    scene <- cached_landscapes[[key]]
    ggsave(file.path(review_dir, paste0(key, ".png")), scene$grob,
           width = scene$width, height = 8, dpi = 200, device = ragg::agg_png)
  }
}
cat("Figure 2 subtype/enrichment checks passed.\n")
