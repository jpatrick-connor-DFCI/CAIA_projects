# Rendering normalization must preserve plot data, blank axes and source layers.
source("COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R")
source("COMPASS/survival_analysis/figure_workflow.R")

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
  all(vapply(q$layers[2:3],function(x) x$aes_params$size==8/ggplot2::.pt,logical(1))),
  all(vapply(q$layers[2:3],function(x) x$aes_params$family=="Arial",logical(1))),
  inherits(q$theme$axis.text.x,"element_blank"),inherits(q$theme$axis.text.y,"element_blank"),
  q$theme$axis.title.y$angle==90,q$theme$axis.title.y$size==9,
  q$theme$axis.title.x.top$size==9,q$theme$plot.subtitle$face=="bold",
  q$theme$plot.subtitle$size==9,q$theme$plot.tag$size==12,q$theme$plot.title$size==11)

# Both composed ggplot text and manually placed grid labels use Arial.
g <- grid::grobTree(figure_plot_grob(q),
  grid::textGrob("Footer",gp=figure_text_gpar("caption")))
text_grobs <- function(g) c(if(inherits(g,"text")) list(g) else list(),
  unlist(lapply(c(g$grobs,as.list(g$children)),text_grobs),recursive=FALSE))
labels <- text_grobs(g)
stopifnot(length(labels)>4,all(vapply(labels,function(x) all(x$gp$fontfamily=="Arial"),logical(1))))
legacy <- grid::grobTree(grid::textGrob("Legacy",gp=grid::gpar(fontfamily="serif",fontsize=8)))
stopifnot(figure_style_grob(legacy)$children[[1]]$gp$fontfamily=="Arial",
  legacy$children[[1]]$gp$fontfamily=="serif")

# Exercise Cairo's separate plotmath font default as well as ordinary labels.
local({
  path <- tempfile(fileext=".pdf")
  on.exit(unlink(path))
  for(label in list(expression(-log[10](q)),"−log10(q)")) {
    ggsave(path,figure_style_plot(p+labs(y=label)),device=figure_pdf_device,width=4,height=3)
    pdf <- readLines(path,warn=FALSE,skipNul=TRUE)
    font_lines <- grep("/BaseFont",pdf,value=TRUE,useBytes=TRUE)
    stopifnot(length(font_lines)>0,all(grepl("Arial",font_lines,useBytes=TRUE)))
  }
})
cat("Figure typography checks passed\n")
