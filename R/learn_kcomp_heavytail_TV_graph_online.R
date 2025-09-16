library(spectralGraphTopology)
library(assertthat)

#' @title Theta matrix of a k-component graph with heavy-tailed data
#'
#' Computes the Theta matrix of a graph on the basis of an observed data matrix,
#' where we assume the data to be Student-t distributed.
#'
#' @param X an T_n x p data matrix, where T_n is the number of observations and p is
#'        the number of nodes in the graph.
#' @param k the number of components of the graph.
#' @param heavy_type a string which selects the statistical distribution of the data    .
#'        Valid values are "gaussian" or "student".
#' @param nu the degrees of freedom of the Student-t distribution.
#'        Must be a real number greater than 2.
#' @param sigma_e hyperparameter that controls graph weight sparsity and time-consistency
#' @param gamma hyperparameter that controls the sparsity of VAR coefficients
#' @param w0 initial vector of graph weights. Either a vector of length p(p-1)/2 or
#'        a string indicating the method to compute an initial value.
#' @param w0 initial value of the VAR coefficient
#' @param eta hyperparameter that controls the regularization to obtain a
#'        k-component graph
#' @param update_eta whether to update eta during the optimization.
#' @param d the nodes' degrees. Either a vector or a single value.
#' @param rho ADMM hyperparameter.
#' @param update_rho whether or not to update rho during the optimization.
#' @param maxiter maximum number of iterations.
#' @param reltol relative tolerance as a convergence criteria.
#' @param verbose whether or not to show a progress bar during the iterations.
#' @export
#' @import spectralGraphTopology
learn_kcomp_heavytail_TV_graph_online <- function(X, w_lagged = 0,
                                                  sigma_e = exp(0.1),
                                                  k = 1,
                                                  heavy_type = "gaussian",
                                                  nu = NULL,
                                                  w0 = "naive",
                                                  a0 = 1,
                                                  d = 1,
                                                  gamma = 10,
                                                  eta = 1e-8,
                                                  update_eta = TRUE,
                                                  early_stopping = FALSE,
                                                  rho = 1,
                                                  update_rho = FALSE,
                                                  maxiter = 10000,
                                                  reltol = 1e-5,
                                                  verbose = TRUE,
                                                  record_objective = FALSE) {
  
  
  X <- scale(as.matrix(X))
  
  # number of nodes
  p <- ncol(X)
  

  # number of observations
  T_n <- nrow(X)

  print(cat("T_n",T_n))
  print(cat("sigma_e",sigma_e))
  alpha <- 2/(T_n*sigma_e)
  print(cat("alpha",alpha))
  beta <- 2*log(sigma_e)/T_n
  print(cat("beta",beta))
  
  
  LstarSq <- vector(mode = "list", length = T_n)
  for (i in 1:T_n)
    LstarSq[[i]] <- Lstar(X[i, ] %*% t(X[i, ])) 
  #print(LstarSq)
  # w-initialization
  if (assertthat::is.string(w0)) {
    w <- spectralGraphTopology:::w_init(w0, MASS::ginv(cor(X)))
    A0 <- A(w)
    A0 <- A0 / rowSums(A0)
    w <- spectralGraphTopology:::Ainv(A0)
    #print(cat("w from A0",w))
  }
  else {
    w <-w0
    #print(cat("w from w0",w))
  }
  Lw <- L(w)
  #print(cat("Lw",Lw))
  #print(Lw)
  Aw <- A(w)
  #print(cat("Aw",Aw))
  U <- eigen(Lw, symmetric = TRUE)$vectors[, (p - k + 1):p]
  #print(cat("U",U))
  #print(eigen(Lw, symmetric = TRUE)$values)
  
  if (!is.null(a0)){ 
    a <- rep(a0, p*(p-1)/2)
  } else {
    a <-  rep(1, p*(p-1)/2)
  }
  #print(cat("a",a))
  if (length(w_lagged)==1){ 
    w_lagged <-  rep(w_lagged, p*(p-1)/2)
  }
  #print(cat("w_lagged",w_lagged))
  
  # Theta-initilization
  Theta <- Lw
  Phi <- matrix(0, p, p)
  #print(cat("Theta",Theta))
  #print(cat("Phi",Phi))
  
  
  # u-initilization
  u <- w - a*w_lagged
  mu_vec <- rep(0, p*(p-1)/2)
  #print(cat("u",u))
  #print(cat("mu_vec",mu_vec))
  
  # degree dual initilization
  z <- rep(0, p)
  #print(cat("z",z))
  #print(cat("d",d))

  
  # ADMM constants
  mu <- 2
  tau <- 2
  # residual vectors
  primal_lap_residual <- c()
  primal_deg_residual <- c()
  dual_residual <- c()
  # augmented lagrangian vector
  lagrangian <- c()
  eta_seq <- c()
  if (verbose)
    pb <- progress::progress_bar$new(format = "<:bar> :current/:total  eta: :eta",
                                     total = maxiter, clear = FALSE, width = 80)
  elapsed_time <- c()
  start_time <- proc.time()[3]






  for (i in 1:maxiter) {
    
    print(cat("iter: ", i))
    for (j in 1:1){
        # update w
        LstarLw <- Lstar(Lw)
        #print("LstarLw")
        #print(LstarLw)
        DstarDw <- Dstar(diag(Lw))
        #print("DstarDw")
        #print(DstarDw)
        LstarSweighted <- rep(0, .5*p*(p-1))
        if (heavy_type == "student") {
          for (q in 1:T_n)
            LstarSweighted <- LstarSweighted + LstarSq[[q]] * compute_student_weights(w, LstarSq[[q]], p, nu)
        } else if (heavy_type == "gaussian") {
          for (q in 1:T_n)
            LstarSweighted <- LstarSweighted + LstarSq[[q]]
        }
        #print("LstarSweighted")
        #print(LstarSweighted)
        grad <- LstarSweighted/T_n + Lstar( eta * crossprod(t(U)) + Phi - rho * Theta ) + rho * (LstarLw )
        #print("aw")
        #print(grad)
        bw <- - mu_vec - rho*(u+a*w_lagged) +  Dstar(z - rho * d) + rho *  DstarDw
        #print("bw")
        #print(bw)
        grad <- grad - mu_vec - rho*(u+a*w_lagged) +  Dstar(z - rho * d) + rho *  DstarDw
        #print("aw + bw")
        #print(grad)
        ratio <- 1 / (rho*(4*p-1))
        #print("ratio")
        #print(ratio)
        wi <- (1-rho*ratio)*w - ratio *  grad
        #print("wi")
        #print(wi)
        thr <- sqrt( 2*beta *ratio )
        wi[wi< thr] <- 0
        #print("wi")
        #print(wi)
        Lwi <- L(wi)
        #print("Lwi")
        #print(Lwi)
        Awi <- A(wi)   
        #print("Awi")
        #print(Awi)
    }
    #print(cat("w Update",wi))

    #print("Lwi")
    #print(Lwi)
    #print("Awi")
    #print(Awi)


    # Update u
    u <- wi - a*w_lagged - mu_vec/rho
    #print(cat("u Temp",u))
    thr <- alpha/(rho)
    #print(cat("thr",thr))
    u <- softThresh(u, thr)
    #print(cat("u Update",u))
    
    
    # update a
    f_temp <- wi -u - mu_vec/rho
    f_temp[f_temp<0] <- 0
    idx <- w_lagged>0
    thr <- gamma/(rho* w_lagged[idx]^2)
    a[idx] <- softThresh(f_temp[idx]/w_lagged[idx], thr)
    a[!idx] <- 0
    #print(cat("a Update",a))
    

    # update U
    U <- eigen(Lwi, symmetric = TRUE)$vectors[, (p - k + 1):p]
    #print(cat("U Update from Lwi",U))
    
    # update Theta
    eig <- eigen( Lwi + Phi/rho, symmetric = TRUE)
    V <- eig$vectors[,1:(p-k)]
    Gamma_U <- eig$values[1:(p-k)]
    Thetai <- V %*% diag((Gamma_U + sqrt(Gamma_U^2 + 4/rho)) / 2) %*% t(V)
    #print("Theta Update")
    #print(Thetai)
    
    
    
    # update Phi
    R1 <-  Lwi - Thetai 
    Phi <- Phi + rho * R1
    #print("Phi")
    #print(Phi)
    #print(typeof(Phi))
    
    
    # update mu
    R0 <- u - wi + a*w_lagged 
    mu_vec <- mu_vec + rho * R0
    #print("mu_vec")
    #print(mu_vec)
    
    # update z
    R2 <- diag(Lwi) - d
    #print("diag(Lwi)")
    #print(diag(Lwi))
    z <- z + rho * R2
    #print("z")
    #print(z)
    

    # compute primal, dual residuals, & lagrangian
    primal_lap_residual <- c(primal_lap_residual, norm(R1, "F"))
    primal_deg_residual <- c(primal_deg_residual, norm(R2, "2"))
    dual_residual <- c(dual_residual, rho*norm(Lstar(Theta - Thetai), "2"))
    #print(c("primal_lap_residual",primal_lap_residual))
    #print(c("primal_deg_residual",primal_deg_residual))
    #print(c("dual_residual",dual_residual))



    lagrangian <- c(lagrangian, compute_augmented_lagrangian_kcomp_mine(wi, LstarSq, Thetai, U, Phi, z, d, heavy_type, T_n, p, k, rho, eta, nu, w_lagged, u, mu_vec, alpha, beta, a, gamma))
    
    # update rho
    if (update_rho) {
      # s <- rho * norm(Lstar(Theta - Thetai), "2")
      # r <- norm(R1, "F")# + norm(R2, "2")
      # if (r > mu * s)
      #   rho <- rho * tau
      # else if (s > mu * r)
      #   rho <- rho / tau
      eig_vals <- spectralGraphTopology:::eigval_sym(Theta)
      n_zero_eigenvalues <- sum(eig_vals < 1e-9)
      if (k < n_zero_eigenvalues)
        rho <- .5 * rho
      else if (k > n_zero_eigenvalues)
        rho <- 2 * rho
      else {
        if (early_stopping) {
          has_converged <- TRUE
          break
        }
      }
    }
    if (update_eta) {
      eig_vals <- spectralGraphTopology:::eigval_sym(L(wi))
      n_zero_eigenvalues <- sum(eig_vals < 1e-9)
      if (k < n_zero_eigenvalues)
        eta <- .5 * eta
      else if (k > n_zero_eigenvalues)
        eta <- 2 * eta
      else {
        if (early_stopping) {
          has_converged <- TRUE
          break
        }
      }
      eta_seq <- c(eta_seq, eta)
    }


    if (verbose)
      pb$tick()
    
    elapsed_time <- c(elapsed_time, proc.time()[3] - start_time)
    has_converged <- (norm(Lwi - Lw, 'F') / norm(Lw, 'F') < reltol) && (i > 1)
    # if (has_converged)
    #   break
    w <- wi
    Lw <- Lwi
    Aw <- Awi
    
    Theta <- Thetai
  }
  print(c("maxiter",maxiter))
  results <- list(laplacian = L(wi), adjacency = A(wi), weights = wi, theta = Thetai, maxiter = i,
                  convergence = has_converged, eta_seq = eta_seq,
                  primal_lap_residual = primal_lap_residual,
                  primal_deg_residual = primal_deg_residual,
                  dual_residual = dual_residual,
                  lagrangian = lagrangian,
                  elapsed_time = elapsed_time)
  return(results)
}

compute_augmented_lagrangian_kcomp_mine <- function(w, LstarSq, Theta, U, Phi, z, d, heavy_type, T_n, p, k, rho, eta, nu, w_lagged, u, mu_vec, alpha, beta, a, gamma ) {
  eig <- eigen(Theta, symmetric = TRUE, only.values = TRUE)$values[1:(p-k)]
  Lw <- L(w)
  Dw <- diag(Lw)
  u_func <- 0
  if (heavy_type == "student") {
    for (q in 1:T_n)
      u_func <- u_func + (p + nu) * log(1 + sum(w * LstarSq[[q]]) / nu)
  } else if (heavy_type == "gaussian"){
    for (q in 1:T_n)
      u_func <- u_func + sum( w * LstarSq[[q]])
  }
  u_func <- u_func/T_n
  return(u_func - sum(log(eig))
         + eta * sum(w * Lstar(crossprod(t(U))))
         + beta* sum(w>0)
         + alpha* sum(abs(u))
         + gamma *sum(a)
         + sum(z * (Dw - d)) + 0.5 * rho * (norm(Dw - d, "2")^2
         + sum(mu_vec * ( u - w + a*w_lagged )) + .5 * rho * (norm(u - w + a*w_lagged, "2"))^2
         + sum(Phi * (Lw - Theta))  + 0.5* rho * norm(Lw - Theta, "F")^2)    
  )
}

hardThresh <- function(v, thr){
  return( v * (abs(v) > thr) )
}



softThresh <- function(v, thr){
  temp <- abs(v) - thr
  temp <- temp * (temp>0)
  return( sign(v) * temp )
  
}

# Added by Me 
compute_student_weights <- function(w, LstarSq, p, nu) {
  return((p + nu) / (sum(w * LstarSq) + nu))
}