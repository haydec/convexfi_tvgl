library(spectralGraphTopology)
library(igraph)
library(Matrix)


#' @export
  plot_graph <- function(Laplacian,true_labels) {
  # number of nodes
  r <- nrow(Laplacian)
  # number of time stamps
  
  # w <- spectralGraphTopology:::Ainv(adjacency)
  # Laplacian <- L(w)
  w <- spectralGraphTopology:::Linv(Laplacian)
  
  adjacency <- spectralGraphTopology:::A(w)
  
  q = r - rankMatrix(Laplacian)
  
  net_Fin <- graph_from_adjacency_matrix(adjacency, mode = "undirected", weighted = TRUE)
  
  
  # where do our predictions differ from GICS?
  metric <- evaluate_clustering(net_Fin, true_labels, r, q)
  labels_pred_adj <-  metric$labels_pred_adj
  
  
  
  colors <- c("#55efc4", "#ff7675", "#0984e3", "#a29bfe", "#B33771", "#48dbfb", "#FDA7DF", "#C4E538")
  colors <- colors[1:q]
  
  
  
  # ground truth coloring 
  V(net_Fin)$color <- c(colors[true_labels])
  V(net_Fin)$type <- c(rep(FALSE, r))
  V(net_Fin)$cluster <- c(true_labels)
  E(net_Fin)$color <- apply(
    as.data.frame(get.edgelist(net_Fin)), 1,
    function(x) {
      ifelse(V(net_Fin)$cluster[x[1]] == V(net_Fin)$cluster[x[2]],
             colors[V(net_Fin)$cluster[x[1]]], "grey"
      )
    }
  )
  
  mask <- labels_pred_adj != true_labels
  # node_labels <- colnames(stock_prices)[1:r]
  # node_labels[!mask] <- NA
  node_labels <- rep(NA, r)
  label_colors <- rep("black",r)
  # label_colors[mask] <- "red"
  
  # plot network
  plot(net_Fin,
       vertex.size = c(rep(4, r)),
       vertex.label = c(node_labels),
       vertex.label.color = label_colors,
       vertex.label.cex = 0.8, vertex.label.dist = 0.5,
       vertex.frame.color = c(colors[true_labels]),
       layout = layout_nicely(net_Fin),
       vertex.label.family = "Helvetica", vertex.label.color = "black",
       vertex.shape = c(rep("circle", r)),
       edge.width = 3 * E(net_Fin)$weight
  )
  
  
  # implied clusters
  true_labels_res <- true_labels
  true_labels <- igraph::components(net_Fin)$membership
  V(net_Fin)$color <- c(colors[true_labels])
  V(net_Fin)$type <- c(rep(FALSE, r))
  V(net_Fin)$cluster <- c(true_labels)
  E(net_Fin)$color <- apply(
    as.data.frame(get.edgelist(net_Fin)), 1,
    function(x) {
      ifelse(V(net_Fin)$cluster[x[1]] == V(net_Fin)$cluster[x[2]],
             colors[V(net_Fin)$cluster[x[1]]], "grey"
      )
    }
  )
  
  
  # node_labels <- colnames(stock_prices)[1:r]
  node_labels <- rep(NA, r)
  # plot network
  plot(net_Fin,
       vertex.size = c(rep(4, r)),
       vertex.label = c(node_labels),
       vertex.label.cex = 0.8, vertex.label.dist = 0.5,
       vertex.frame.color = c(colors[true_labels]),
       layout = layout_nicely(net_Fin),
       vertex.label.family = "Helvetica", vertex.label.color = "black",
       vertex.shape = c(rep("circle", r)),
       edge.width = 3 * E(net_Fin)$weight
  )
  
  
  
  print("metric:\n")
  print(metric$accuracy)
  print(metric$accuracy_adj)
  print(metric$purity)
  print(metric$mod_g)
  print(metric$mod_gt)
  print(metric$balanced)
  print(metric$balanced_norm)
  print(metric$GINI)
  
  return(net_Fin)
}
