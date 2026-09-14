# Run from the repository root: Rscript tests/test_figure2_subtypes.R
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(stringr)
})
pipeline <- new.env(parent = globalenv())
needed <- c("COLOR_PLATINUM_POS", "COLOR_PLATINUM_NEG", "COLOR_NEUTRAL_INK",
            "theme_fig", "wilson_ci", "CLASS_ORDER", "CLASS_LABELS", "count_labels",
            "compute_enrichment", "render_landscape_panel", "render_enrichment_panel")
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
}
cat("Figure 2 subtype/enrichment checks passed.\n")
