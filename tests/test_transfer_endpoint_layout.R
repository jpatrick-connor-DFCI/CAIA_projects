# Base-R regression checks; no plotting packages or clinical data required.
definitions <- parse("COMPASS/figures/figure_workflow.R")
needed <- c("FEDERATED_TRANSFER_ENDPOINTS_SHORT", "FEDERATED_TRANSFER_ENDPOINTS",
  "FEDERATED_TRANSFER_DEFAULT_ENDPOINTS", "FEDERATED_TRANSFER_KM_YLAB",
  "FEDERATED_TRANSFER_ENDPOINT_COLORS", "federated_transfer_grid_endpoints",
  "manuscript_transfer_panel_rows", "federated_transfer_free_phrase")
for(expr in definitions) {
  if(!is.call(expr) || !identical(expr[[1]],as.name("<-"))) next
  lhs <- expr[[2]]
  name <- if(is.symbol(lhs)) as.character(lhs) else if(is.call(lhs)) as.character(lhs[[2]]) else ""
  if(name %in% needed) eval(expr)
}
defaults <- c("platinum","nepc_t1","nepc_t1_t2","nepc_t1_t2_t3","avpc","avpc_or_adjacent")
stopifnot(identical(FEDERATED_TRANSFER_DEFAULT_ENDPOINTS, defaults),
  all(c(defaults,"nepc") %in% names(FEDERATED_TRANSFER_ENDPOINTS)),
  !anyNA(FEDERATED_TRANSFER_KM_YLAB[c(defaults,"nepc")]),
  identical(unname(FEDERATED_TRANSFER_KM_YLAB[c("platinum","avpc","avpc_or_adjacent")]),
    c("Platinum-free probability","AVPC-free probability","AVPC + adjacent event-free probability")),
  identical(vapply(c("platinum","nepc_t1"),federated_transfer_free_phrase,character(1),USE.NAMES=FALSE),
    c("platinum-free","NEPC (t1) event-free")),
  !anyNA(FEDERATED_TRANSFER_ENDPOINT_COLORS[FEDERATED_TRANSFER_ENDPOINTS]),
  identical(federated_transfer_grid_endpoints(defaults),defaults),
  identical(federated_transfer_grid_endpoints("nepc_t1"),defaults),
  identical(federated_transfer_grid_endpoints(c("platinum","avpc","nepc")),c("platinum","avpc","nepc")))
# All endpoint counts, including suppressed KM curves and historical exports:
# each panel appears exactly once as a contiguous block, in a six-column row.
for(n in 0:7) {
  indices <- 2L+seq_len(n)
  rows <- manuscript_transfer_panel_rows(indices)
  stopifnot(length(rows)==ceiling(n/3),all(lengths(rows)==6L),
    identical(unname(unlist(lapply(rows,unique))),if(n) indices else NULL))
}
cat("Transfer endpoint labels, grid selection and 0–7-panel layouts passed.\n")
