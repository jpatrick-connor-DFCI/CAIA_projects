# Shared typography for every COMPASS figure writer. Sizes are points and also
# feed the Python survival plots through the same JSON file.
FIGURE_TYPOGRAPHY <- jsonlite::fromJSON(file.path(dirname(tail(Filter(Negate(is.null), lapply(sys.frames(), function(frame) frame$ofile)), 1)[[1]]),
  "..", "..", "survival_common", "figure_typography.json"))
FIGURE_FONT_SIZES <- unlist(FIGURE_TYPOGRAPHY$sizes)
figure_font_face <- function(role) if(role %in% c("axis_title","strip","title","tag")) "bold" else "plain"

# Resolve before constructing plots so measurement, text and devices agree.
figure_resolve_font <- function(installed=if(requireNamespace("systemfonts",quietly=TRUE))
    systemfonts::system_fonts()$family else character()) {
  candidates <- c(FIGURE_TYPOGRAPHY$family,FIGURE_TYPOGRAPHY$fallback_families)
  available <- candidates[tolower(candidates) %in% tolower(installed)]
  if(length(available)) available[[1]] else "sans"
}
FIGURE_FONT <- figure_resolve_font()

figure_require_font <- local({
  reported <- FALSE
  function() {
    if(!reported && FIGURE_FONT!=FIGURE_TYPOGRAPHY$family)
      message(FIGURE_TYPOGRAPHY$family," is unavailable or could not be verified on this rendering host; using ",
        FIGURE_FONT,". Figure sizes and styling are unchanged. For Arial output, install Arial and ",
        "the systemfonts R package on this host, then restart R.")
    reported <<- TRUE
    invisible(FIGURE_FONT)
  }
})

# Cairo otherwise uses a separate default family for plotmath glyphs, even
# when the surrounding grid text explicitly requests the resolved font.
figure_pdf_device <- function(filename, width=7, height=7, ...) {
  figure_require_font()
  grDevices::cairo_pdf(filename, width=width, height=height, family=FIGURE_FONT,
    symbolfamily=grDevices::cairoSymbolFont(FIGURE_FONT,usePUA=FALSE), ...)
}

figure_text_gpar <- function(role="body", ...) do.call(grid::gpar,
  utils::modifyList(list(fontfamily=FIGURE_FONT,fontsize=unname(FIGURE_FONT_SIZES[[role]]),
    fontface=figure_font_face(role),lineheight=1),list(...)))

figure_font_theme <- function() ggplot2::theme(
  text=ggplot2::element_text(family=FIGURE_FONT,size=FIGURE_FONT_SIZES[["body"]],face="plain",lineheight=1),
  axis.title=ggplot2::element_text(size=FIGURE_FONT_SIZES[["axis_title"]],face="bold"),
  axis.text=ggplot2::element_text(size=FIGURE_FONT_SIZES[["body"]],face="plain"),
  legend.text=ggplot2::element_text(size=FIGURE_FONT_SIZES[["legend"]],face="plain"),
  legend.title=ggplot2::element_text(size=FIGURE_FONT_SIZES[["legend"]],face="plain"),
  strip.text=ggplot2::element_text(size=FIGURE_FONT_SIZES[["strip"]],face="bold"),
  plot.title=ggplot2::element_text(size=FIGURE_FONT_SIZES[["title"]],face="bold"),
  plot.subtitle=ggplot2::element_text(size=FIGURE_FONT_SIZES[["subtitle"]]),
  plot.caption=ggplot2::element_text(size=FIGURE_FONT_SIZES[["caption"]],face="plain"),
  plot.tag=ggplot2::element_text(size=FIGURE_FONT_SIZES[["tag"]],face="bold"))

# Normalize before layout measurement, including plot-specific overrides. Keep
# blank axes, label colors, rotations, margins and deliberate bold annotations.
figure_style_plot <- function(p) {
  if(!inherits(p,"ggplot")) return(figure_style_grob(p))
  complete <- ggplot2::theme_get() + p$theme
  roles <- c(text="body",plot.title="title",plot.subtitle="subtitle",plot.caption="caption",plot.tag="tag",
    axis.title="axis_title",axis.title.x="axis_title",axis.title.y="axis_title",
    axis.text="body",axis.text.x="body",axis.text.y="body",
    legend.text="legend",legend.title="legend",strip.text="strip",strip.text.x="strip",strip.text.y="strip")
  extra <- names(complete)[vapply(complete,function(x) inherits(x,"element_text"),logical(1))]
  for(field in setdiff(extra,names(roles))) roles[[field]] <- if(startsWith(field,"axis.title")) "axis_title" else
    if(startsWith(field,"strip.text")) "strip" else if(startsWith(field,"legend")) "legend" else "body"
  overrides <- lapply(names(roles),function(field) {
    element <- ggplot2::calc_element(field,complete)
    if(!inherits(element,"element_text")) return(element)
    role <- roles[[field]]
    element$family <- FIGURE_FONT
    element$size <- unname(FIGURE_FONT_SIZES[[role]])
    if(field!="plot.subtitle") element$face <- figure_font_face(role)
    element$lineheight <- 1
    element
  })
  p <- p + do.call(ggplot2::theme,stats::setNames(overrides,names(roles)))
  p$layers <- lapply(p$layers,function(layer) {
    if(!any(vapply(c("GeomText","GeomLabel","GeomTextRepel","GeomLabelRepel"),
      function(geom) inherits(layer$geom,geom),logical(1)))) return(layer)
    copy <- ggplot2::ggproto(NULL,layer)
    copy$aes_params$family <- FIGURE_FONT
    # Checklist cards explicitly size their text and spacing together.
    font_size <- layer$figure_font_size
    if(is.null(font_size)) font_size <- FIGURE_FONT_SIZES[["body"]]
    copy$aes_params$size <- font_size / ggplot2::.pt
    if(!is.null(copy$geom_params$size.unit)) copy$geom_params$size.unit <- "mm"
    copy
  })
  p
}

# Nested grid objects (panel letters, footnotes and assembled legends) do not
# inherit ggplot themes. Their role sizes are set at construction; enforce the
# same family throughout the tree without changing measured layout geometry.
figure_style_grob <- function(g) {
  if(!grid::is.grob(g)) return(g)
  if(inherits(g,"text")) {
    if(is.null(g$gp)) g$gp <- figure_text_gpar()
    g$gp$fontfamily <- FIGURE_FONT
  }
  if(length(g$grobs)) g$grobs <- lapply(g$grobs,figure_style_grob)
  if(length(g$children)) g$children <- do.call(grid::gList,lapply(g$children,figure_style_grob))
  g
}

figure_plot_grob <- function(p) {
  figure_require_font()
  # grid measures labels while building the gtable. Base PDF/PostScript devices
  # cannot resolve system fonts, and R otherwise opens one from a null device.
  if(names(grDevices::dev.cur()) %in% c("null device","pdf","postscript")) {
    size <- if(grDevices::dev.cur()==1L) c(7,7) else grDevices::dev.size("in")
    path <- tempfile(fileext=".pdf")
    figure_pdf_device(path,width=size[1],height=size[2])
    on.exit({grDevices::dev.off();unlink(path)},add=TRUE)
  }
  figure_style_grob(ggplot2::ggplotGrob(figure_style_plot(p)))
}

# The standalone checklist metrics use base graphics instead of ggplot.
figure_base_par <- function() graphics::par(family=FIGURE_FONT,ps=FIGURE_FONT_SIZES[["body"]],
  cex=1,cex.axis=1,cex.lab=FIGURE_FONT_SIZES[["axis_title"]]/FIGURE_FONT_SIZES[["body"]],
  cex.main=FIGURE_FONT_SIZES[["title"]]/FIGURE_FONT_SIZES[["body"]],font=1,font.axis=1,font.lab=2,font.main=2)
