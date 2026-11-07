# Rendering normalization must preserve plot data, blank axes and source layers.
source("COMPASS/figures/COMPASS_generate_figures_pipeline.R")
source("COMPASS/figures/figure_workflow.R")
source("COMPASS/figures/checklist_supplement.R")

stopifnot(figure_resolve_font(c("DejaVu Sans","Arial","Liberation Sans"))=="Arial",
  figure_resolve_font(c("DejaVu Sans","Liberation Sans"))=="Liberation Sans",
  figure_resolve_font("DejaVu Sans")=="DejaVu Sans",
  figure_resolve_font(character())=="sans")

p <- ggplot(tibble(x=1:2,y=2:3,label=c("A","B")),aes(x,y)) +
  geom_point(size=3) + geom_text(aes(label=label),size=6,fontface="bold") +
  ggrepel::geom_text_repel(aes(label=label),size=5) +
  labs(title="Title",subtitle="Subtitle",tag="a") + theme_classic() +
  theme(axis.text=element_blank(),axis.title.y=element_text(size=20,angle=90),
    axis.title.x.top=element_text(size=24),plot.subtitle=element_text(face="bold"))
q <- figure_style_plot(p)
stopifnot(identical(p$data,q$data),p$layers[[2]]$aes_params$size==6,
  p$layers[[3]]$aes_params$size==5,q$layers[[1]]$aes_params$size==3,
  q$layers[[2]]$aes_params$fontface=="bold",
  all(vapply(q$layers[2:3],function(x) x$aes_params$size==10/ggplot2::.pt,logical(1))),
  all(vapply(q$layers[2:3],function(x) x$aes_params$family==FIGURE_FONT,logical(1))),
  inherits(q$theme$axis.text.x,"element_blank"),inherits(q$theme$axis.text.y,"element_blank"),
  q$theme$axis.title.y$angle==90,q$theme$axis.title.y$size==11,
  q$theme$axis.title.x.top$size==11,q$theme$plot.subtitle$face=="bold",
  q$theme$plot.subtitle$size==11,q$theme$plot.tag$size==14,q$theme$plot.title$size==13)

# Both composed ggplot text and manually placed grid labels use the same font.
# Explicit checklist type survives both styling passes in manuscript export.
card_lines <- data.frame(text=c("Header"," ","[ ] Example"," "),face="plain",
  landmark_label="Day 0")
card_layout <- checklist_card_layout(card_lines,12)
card <- plot_checklist_cards(card_layout$lines,NULL,font_size=12)
styled_card <- figure_style_plot(manuscript_style(card,"Checklist","a"))
stopifnot(nrow(card_layout$lines)==3,
  styled_card$layers[[1]]$aes_params$size==12/ggplot2::.pt,
  (card_layout$lines$y[2]-card_layout$lines$y[3]) <
    (card_layout$lines$y[1]-card_layout$lines$y[2]))
g <- grid::grobTree(figure_plot_grob(q),
  grid::textGrob("Footer",gp=figure_text_gpar("caption")))
text_grobs <- function(g) c(if(inherits(g,"text")) list(g) else list(),
  unlist(lapply(c(g$grobs,as.list(g$children)),text_grobs),recursive=FALSE))
labels <- text_grobs(g)
stopifnot(length(labels)>4,all(vapply(labels,function(x) all(x$gp$fontfamily==FIGURE_FONT),logical(1))))
legacy <- grid::grobTree(grid::textGrob("Legacy",gp=grid::gpar(fontfamily="serif",fontsize=8)))
stopifnot(figure_style_grob(legacy)$children[[1]]$gp$fontfamily==FIGURE_FONT,
  legacy$children[[1]]$gp$fontfamily=="serif")

# Exercise Cairo's separate plotmath font default as well as ordinary labels.
local({
  path <- tempfile(fileext=".pdf")
  on.exit(unlink(path))
  for(label in list(expression(-log[10](q)),"−log10(q)")) {
    ggsave(path,figure_style_plot(p+labs(y=label)),device=figure_pdf_device,width=4,height=3)
    pdf <- readLines(path,warn=FALSE,skipNul=TRUE)
    font_lines <- grep("/BaseFont",pdf,value=TRUE,useBytes=TRUE)
    stopifnot(length(font_lines)>0)
    if(FIGURE_FONT!="sans") stopifnot(all(grepl(gsub("[ -]","",FIGURE_FONT),
      font_lines,useBytes=TRUE,fixed=TRUE)))
  }
})

# A missing preferred font reports the chosen replacement once and still exports.
local({
  source("COMPASS/figures/figure_typography.R",local=TRUE)
  FIGURE_FONT <- "sans"
  notices <- capture.output({figure_require_font();figure_require_font()},type="message")
  stopifnot(length(notices)==1L,grepl("using sans",notices,fixed=TRUE))
  path <- tempfile(fileext=".pdf")
  on.exit(unlink(path))
  ggsave(path,figure_style_plot(p),device=figure_pdf_device,width=4,height=3)
  stopifnot(file.info(path)$size>0)
})
cat("Figure typography checks passed\n")
