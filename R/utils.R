from_B_to_laplacian <- function(B) {
  A <- from_B_to_adjacency(B)
  return(diag(rowSums(A)) - A)
}

from_B_to_adjacency <- function(B) {
  r <- nrow(B)
  q <- ncol(B)
  zeros_rxr <- matrix(0, r, r)
  zeros_qxq <- matrix(0, q, q)
  return(rbind(cbind(zeros_rxr, B), cbind(t(B), zeros_qxq)))
}



Mode <- function(v) {
  uniqv <- unique(v)
  return ( uniqv[which.max(tabulate(match(v, uniqv)))] )
}


Balancedness <- function(netobj, r, q) {
  clust_size <- igraph::components(netobj)$csize
  k <- length(clust_size)
  if (k<q)
  {
    clust_size <- c(rep(0,q-k), clust_size)
  }
  return( sum(abs(clust_size - r/q))/q )
}



Balancedness_norm <- function(netobj, r, q) {
  clust_size <- igraph::components(netobj)$csize
  k <- length(clust_size)
  if (k<q)
  {
    clust_size <- c(rep(0,q-k), clust_size)
  }
  num <- sum(abs(clust_size - r/q))
  clust_size <- c(rep(1,(q-1)), r-(q-1))
  dennum <- sum(abs(clust_size - r/q))
  return( 1 - num/dennum )
}


GINI <- function(netobj) {
  clust_size <- igraph::components(netobj)$csize
  k <- length(clust_size)
  clust_size_avg <- mean(clust_size)
  num <- 0 
  for (i in 1:k)
  {
    for (j in 1:k) {
      num <- num + abs(clust_size[i]-clust_size[j]) }
  }
  return(num/ (2*k^2*clust_size_avg))
}

simplex_project <- function(x0, lb) {
  n <- length(x0)
  x0_pos <- x0
  x0_pos[x0_pos<0] <- 0
  
  if (sum(x0_pos) >= lb) {
    alpha <- 0
  }
  
  else if (sum(x0>=0) == n) {
    alpha <- (lb - sum(x0)) / n  
  }
  
  else{
    
    x0_sorted <- sort(x0, decreasing = TRUE)
    id_neg <- which(x0_sorted<0)
    if (id_neg[1]>1) {
      id_neg <- c(id_neg[1]-1,id_neg)
    }
    
    x0_csum <- cumsum(x0_sorted)
    
    for (j in 1:length(id_neg)) {
      id <- id_neg[j]
      alpha <- (lb - x0_csum[id])/id
      
      if (id<n) {
        if  (alpha>=0 & -alpha < x0_sorted[id] & -alpha >= x0_sorted[id+1]) {
          break
        }    
      }
      
    }
    
  }
  
  x <- x0  + alpha
  x[x<0] = 0
  
  return(x)
}



rho_gamma <- function(x, gamma, u) {
  return( log(1+abs(x)/u) / log(1+gamma/u) )
}
  



project_circle <- function(z0, d, u) {
  beta <- min(z0)
  itermax <- 1000
  reltol <- 1e-6
  p <- length(z0)
  c <- (sum(z0)-d)/p
  pu <- p*u
  
  for (j in 1:itermax) {
    z0_sub <- z0 - rep(beta,p)
    z0_sub_pos <- z0_sub
    z0_sub_pos[z0_sub_pos<0] <- 0;
    z0_sub_neg <- -z0_sub;
    z0_sub_neg[z0_sub_neg<0] <- 0;
    
    beta_new <- ( u*sum(z0_sub_neg) - d*max(0,norm(z0_sub_pos,'2')-u)  )/pu + c;
    if ( norm(beta - beta_new,'2')/norm(beta,'2') < reltol ) {
      break
    }
    
    beta <- beta_new
  }
  
  z <- z0_sub_pos
  z_norm <- norm(z,'2');
  if (z_norm>u) {
    z <- z/z_norm * u;
  }
  
  return( z )
}



evaluate_clustering <- function(net_Fin, stock_sectors_index, r, q) {
  labels_pred <- rep(0, r)
  for (j in 1:q){
    idx <- igraph::components(net_Fin)$membership %in% c(j)
    labels_pred[idx] <- Mode(stock_sectors_index[idx])
  }
  
  mask <- labels_pred != stock_sectors_index
  purity_Fin <- 1- sum(mask)/length(mask)
  
  
  labels_pred = igraph::components(net_Fin)$membership
  mask <- labels_pred != stock_sectors_index
  accuracy_metric <- 1- sum(mask)/length(mask)
  
  
  
  labels_pred_adj <- labels_pred 
  perms <- permn(c(1:q))
  acc_max <- 0
  for (k in 1:length(perms)){
    perm <- perms[[k]]
    for (j in 1:q){
      idx <- labels_pred %in% j
      labels_pred_adj[idx] <- perm[j]
      
    }
    mask <- labels_pred_adj != stock_sectors_index
    acc <- 1- sum(mask)/length(mask)
    if (acc>= acc_max) {
      acc_max <- acc
      ind_max <- k
    }
  }
  perm <- perms[[ind_max]]
  for (j in 1:q){
    idx <- labels_pred %in% j
    labels_pred_adj[idx] <- perm[j]
  }
  
  NMI_val <- randnet::NMI(labels_pred_adj, stock_sectors_index)
  
  ARI <- mclust::adjustedRandIndex(labels_pred_adj, stock_sectors_index)
  mask <- labels_pred_adj != stock_sectors_index
  accuracy_adj_metric <- 1- sum(mask)/length(mask)
  
  
  
  mod_metric <- modularity(cluster_walktrap(net_Fin))
  mod_g_metric <- modularity(net_Fin, membership(cluster_walktrap(net_Fin)))
  mod_gt_metric <- modularity(net_Fin, stock_sectors_index)
  
  balanced_metric <- Balancedness(net_Fin, r, q)
  balanced_norm_metric <- Balancedness_norm(net_Fin, r, q)
  GINI_metric <- GINI(net_Fin)
  
  metrics <- list( labels_pred = labels_pred,  labels_pred_adj = labels_pred_adj , accuracy = accuracy_metric, 
                   accuracy_adj = accuracy_adj_metric, purity = purity_Fin, 
                   mod_g = mod_g_metric, mod_gt = mod_gt_metric, balanced = balanced_metric, 
                   balanced_norm = balanced_norm_metric, GINI = GINI_metric,
                   NMI = NMI_val, ARI = ARI)
  return( metrics )
}




spectral_clustering <- function(X, # matrix of data points
                                nn = 10, # the k nearest neighbors to consider
                                n_eig = 2) # m number of eignenvectors to keep
{
  mutual_knn_graph <- function(X, nn = 10)
  {
    D <- as.matrix( dist(X) ) # matrix of euclidean distances between data points in X
    
    # intialize the knn matrix
    knn_mat <- matrix(0,
                      nrow = nrow(X),
                      ncol = nrow(X))
    
    # find the 10 nearest neighbors for each point
    for (i in 1: nrow(X)) {
      neighbor_index <- order(D[i,])[2:(nn + 1)]
      knn_mat[i,][neighbor_index] <- 1 
    }
    
    # Now we note that i,j are neighbors iff K[i,j] = 1 or K[j,i] = 1 
    knn_mat <- knn_mat + t(knn_mat) # find mutual knn
    
    knn_mat[ knn_mat == 2 ] = 1
    
    return(knn_mat)
  }
  
  graph_laplacian <- function(W, normalized = TRUE)
  {
    stopifnot(nrow(W) == ncol(W)) 
    
    g = colSums(W) # degrees of vertices
    n = nrow(W)
    
    if(normalized)
    {
      D_half = diag(1 / sqrt(g) )
      return( diag(n) - D_half %*% W %*% D_half )
    }
    else
    {
      return( diag(g) - W )
    }
  }
  
  W = mutual_knn_graph(X) # 1. matrix of similarities
  L = graph_laplacian(W) # 2. compute graph laplacian
  ei = eigen(L, symmetric = TRUE) # 3. Compute the eigenvectors and values of L
  n = nrow(L)
  return(list(Eigv = ei$vectors[,(n - n_eig):(n - 1)], W=W, L= L)) # return the eigenvectors of the n_eig smallest eigenvalues
  
}
