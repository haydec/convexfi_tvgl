import numpy as np
from sklearn.preprocessing import scale
from numpy.linalg import slogdet
from scipy.special import digamma, polygamma, gammaln

# ===========================================================================
# ===========================================================================
# A Unified Framework for Structured Graph Learning via Spectral Constraints

# https://jmlr.org/papers/volume21/19-276/19-276.pdf
# ===========================================================================
# ===========================================================================

def w_from_L(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")

    n = M.shape[0]
    w = np.empty(n * (n - 1) // 2, dtype=M.dtype)
    k = 0
    for i in range(n - 1):            # i = 0 .. n-2
        for j in range(i + 1, n):     # j = i+1 .. n-1 (strict upper)
            w[k] = -M[i, j]
            k += 1
    return w

def L_from_w(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w).ravel()
    k = w.size
    n = int((1 + np.sqrt(1 + 8 * k)) / 2)
    if n * (n - 1) // 2 != k:
        raise ValueError("Invalid length of w: must be n(n-1)/2 for some integer n.")

    Lw = np.zeros((n, n), dtype=float)

    # Fill the strict upper triangle row by row, backwards like in C++
    for i in range(n - 2, -1, -1):
        j = n - i - 1
        # Equivalent of Eigen: Lw.row(i).tail(j) = -w.head(k).tail(j)
        Lw[i, -j:] = -w[:k][-j:]
        k -= j

    # Make symmetric: copy upper triangle into lower
    Lw = Lw + Lw.T

    # Adjust diagonal: subtract column sums
    colsum = Lw.sum(axis=0)
    np.fill_diagonal(Lw, Lw.diagonal() - colsum)

    return Lw

def W_init(M):
    W0 = w_from_L(M)
    W0[W0 < 0] = 0
    return W0

def L_star(Y):

    Y = np.asarray(Y)
    
    if Y.ndim != 2 or Y.shape[0] != Y.shape[1]:
        raise ValueError("Y must be a square matrix.")
    p = Y.shape[0]
    Lstar = np.zeros(int(p*(p-1)/2))

    for i in range(1,p+1):

        for j in range(1,p+1):
           
            if (i > j):
                k = int( (i - j) + ((j - 1)/2)*(2*p - j) ) 
                Lstar[k-1] = Y[i-1,i-1] - Y[i-1, j-1] - Y[ j-1, i-1] + Y[ j-1, j-1]


    return Lstar

def A_from_w(w):

    w = np.asarray(w, dtype=float).ravel()
    k = w.size

    # Infer n from k = n*(n-1)/2
    n_float = (1 + np.sqrt(1 + 8*k)) / 2
    n = int(n_float)
    if n*(n-1)//2 != k:
        raise ValueError(f"len(w)={k} is not a triangular number; expected k=n*(n-1)/2.")

    A = np.zeros((n, n), dtype=float)

    # Indices of strict upper triangle in row-major order
    iu = np.triu_indices(n, k=1)
    A[iu] = w            # fill upper triangle
    A[(iu[1], iu[0])] = w  # mirror to lower triangle
    return A

def w_from_A(M):
    
    N = M.shape[1]
    k = N * (N - 1) // 2
    w = np.zeros(k)
    l = 0

    for i in range(N - 1):
        for j in range(i + 1, N):
            w[l] = M[i, j]
            l += 1
    return w


def A_star(Y):

    Y = np.asarray(Y)
    
    if Y.ndim != 2 or Y.shape[0] != Y.shape[1]:
        raise ValueError("Y must be a square matrix.")
    p = Y.shape[0]
    Astar = np.zeros(int(p*(p-1)/2))

    for i in range(1,p+1):

        for j in range(1,p+1):
           
            if (i > j):
                k = int( (i - j) + ((j - 1)/2)*(2*p - j) ) 
                Astar[k-1] = Y[i-1, j-1] + Y[ j-1, i-1]


    return Astar

def D_star(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w).ravel()
    dStar = L_star(np.diag(w))
    return  dStar

def D_from_w(w: np.ndarray) -> np.ndarray:
    """
    Return the column-wise sums of A(w).
    Equivalent to Eigen: A(w).colwise().sum()
    """
    M = A_from_w(w)               # shape (m, p)
    return M.sum(axis=0)   # shape (p,)

# ===========================================================================
# ===========================================================================


# ===========================================================================
# ===========================================================================
# Reference
# https://zouyuxin.github.io/Note/EMtDistribution.pdf
# https://shoichimidorikawa.github.io/Lec/ProbDistr/t-e.pdf
def fit_multivariate_t(
    X,
    nu_init=10.0,
    max_iter=200,
    tol=1e-6,
    fix_nu=None,
    jitter=1e-6, 
    verbose=False,
):
    """
    
    Estimate (mu, Sigma, nu) for a multivariate Student's t via EM.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Data.
    nu_init : float
        Initial degrees of freedom (ignored if fix_nu is not None).
    max_iter : int
        Maximum EM iterations.
    tol : float
        Relative tolerance on log-likelihood for convergence.
    fix_nu : float or None
        If set, keep nu fixed at this value.
    jitter : float
        Diagonal jitter multiplier for numerical stability.
    verbose : bool
        Print progress if True.

    Returns
    -------
    mu : array, shape (p,)
    Sigma : array, shape (p, p)
    nu : float
    history : dict with 'loglik'
    """
    X = np.asarray(X)
    n, p = X.shape

    # Initialize
    mu = X.mean(axis=0)
    # sample covariance (unbiased=False), add tiny jitter
    centered0 = X - mu
    Sigma = centered0.T @ centered0 / n
    Sigma += np.eye(p) * (jitter * np.trace(Sigma) / p + 1e-12)
    nu = float(nu_init if fix_nu is None else fix_nu)

    def mahal_sq(Sigma, X, mu):
        # δ_i = (x_i - mu)^T Sigma^{-1} (x_i - mu)
        L = np.linalg.cholesky(Sigma)
        Y = np.linalg.solve(L, (X - mu).T)   # shape (p, n)
        return np.sum(Y * Y, axis=0)         # shape (n,)

    def loglik(mu, Sigma, nu):
        # log-likelihood of multivariate t
        sgn, logdet = slogdet(Sigma)
        if sgn <= 0:
            return -np.inf
        delta = mahal_sq(Sigma, X, mu)
        term1 = gammaln((nu + p) / 2) - gammaln(nu / 2)
        term2 = - (p / 2) * np.log(nu * np.pi) - 0.5 * logdet
        term3 = - ((nu + p) / 2) * np.log1p(delta / nu)
        return np.sum(term1 + term2 + term3)

    ll_old = -np.inf
    history = {'loglik': []}

    for it in range(1, max_iter + 1):
        # E-step
        delta = mahal_sq(Sigma, X, mu)
        w = (nu + p) / (nu + delta)  # E[lambda_i | x_i]
        eloglam = digamma((nu + p) / 2) - np.log((nu + delta) / 2)  # E[log lambda_i | x_i]

        # M-step: mu (weighted mean)
        sumw = w.sum()
        mu = (w[:, None] * X).sum(axis=0) / sumw

        # M-step: Sigma
        Xc = X - mu
        Sigma = (Xc.T * w) @ Xc / n
        # stabilize
        Sigma += np.eye(p) * (jitter * np.trace(Sigma) / p + 1e-12)

        # M-step: nu (solve for root of dQ/dnu = 0) unless fixed
        if fix_nu is None:
            # f(nu) = log(nu/2) - psi(nu/2) + 1 + (1/n) * sum(E[log lambda_i] - E[lambda_i]) = 0
            c = (eloglam.mean() - w.mean())

            def f(nu_):
                return np.log(nu_ / 2.0) - digamma(nu_ / 2.0) + 1.0 + c

            def fprime(nu_):
                return 1.0 / nu_ - 0.5 * polygamma(1, nu_ / 2.0)

            # Newton update with simple guarding
            nu_new = max(nu, 2.01)  # keep > 2 so covariance exists
            for _ in range(30):
                val = f(nu_new)
                der = fprime(nu_new)
                step = val / der
                nu_new = nu_new - step
                if nu_new < 2.01:
                    nu_new = 2.01
                if abs(step) / nu_new < 1e-6:
                    break
            nu = float(nu_new)
        else:
            nu = float(fix_nu)

        # Evaluate log-likelihood and check convergence
        ll = loglik(mu, Sigma, nu)
        history['loglik'].append(ll)
        if verbose:
            print(f"iter {it:3d}: ll={ll:.6f}, nu={nu:.4f}")

        if np.isfinite(ll_old):
            if abs(ll - ll_old) / (abs(ll_old) + 1e-12) < tol:
                break
        ll_old = ll

    return mu, Sigma, nu, history

# ===========================================================================
# ===========================================================================


def init_parameters(X,k):

    '''
    X: ( n_samples, n_features ) Data Array For Window (Frame) Length
    k: ( n_clusters )            Clusters  
    
    '''

    
    p = X.shape[1] # n_features
    S_corr = np.corrcoef(X.T) # Estimate Correlation Matrix
    S_prec = np.linalg.pinv(S_corr)

    w = W_init(S_prec) # Init Weights

    A0 = A_from_w(w)
    A0 = A0/np.sum(A0,axis=1,keepdims=True) # Normalize Rows 
    w = w_from_A(A0)
    Aw = A_from_w(w) 



    a0 = 1 # Maybe Add Init Options
    a = np.repeat(a0, p*(p-1)/2) 
    w_lagged = np.repeat(0, p*(p-1)/2) # Maybe Add Init Options

    L_n = L_from_w(w) # Laplacian Constrained Graph Matrix
    Phi_n = np.zeros((p,p)) # Ln Dual
   

    u_n = w - np.multiply( a, w_lagged) # Temporal Consistency Parameter
    mu_vec = np.repeat(0,p*(p-1)/2) # u_n ADMM dual

    d = 1 # degree Constraint
    z_n = np.repeat(0,p) # Degree Dual 

    eigVals,eigVecs = np.linalg.eigh(L_n)
    V = eigVecs[:,p-k:] # Penalty to control rank of Ln

    return w, Aw, a, w_lagged, L_n, Phi_n, u_n, mu_vec, d, z_n, V



def L_update( L_n, Phi_n, rho, k):

    ''' 
    L_n:   (n_features, n_features) Laplace Matrix (Ln = Diag(Wn1) - Wn = Lwn) Laplacian operator “A Unified Framework for Structured Graph Learning via Spectral Constraints”
    Phi_n: (n_features, n_features) ADMM Dual Variable 
    rho:   (1)                      ADMM Hyperparameter   
    k:     (n_clusters)             Number of Clusters
    '''
    # Reference: Equation 16

    p = L_n.shape[0] #  p: Number of features
    Y = L_n + Phi_n/rho

    eigVals,eigVecs = np.linalg.eigh(Y)
    eigenVecs = eigVecs[:, ::-1]       # take first (p-k) eigenvectors
    eigenVals = eigVals[::-1]  
    U = eigenVecs[:, :p-k]         # take first (p-k) eigenvectors
    R = np.diag(eigenVals[:p-k]  )
    I = np.eye(R.shape[0]) 
    L_nUpdate = 0.5 * U @ ( R + np.sqrt(  R**2 + 4/rho * I  ) ) @ U.T



    return L_nUpdate


def w_update( X, w, w_lagged,a,  nu, L_n, L_nUpdate,  u_n, z_n, Phi_n, V, rho, d, eta, beta):

    '''
    X:           (n_samples, n_features)        Data Array For Window (Frame) Length

    w:           (n_features*(n_features-1)/2)  Graph Weights for current Frame
    w_lagged:    (n_features*(n_features-1)/2)  Graph Weights for previous Frame
    a            (n_features*(n_features-1)/2)  VAR Weights for Previous Frame
    u_n:         (n_features*(n_features-1)/2)  Temporal Consistency Parameter

    L_n:         (n_features, n_features)       Laplacian Constrained Graph Matrix
    L_nUpdate:   (n_features, n_features)       Updated Laplacian Constrained Graph Matrix
    Phi_n:       (n_features, n_features)       Laplacian Constrained Graph Matrix Dual
    
    
    z_n:         (n_features)                   Degree Dual 
    d:           (1)                            Degree Constraint
    

    V:           (n_features, n_clusters)       Penalty to control rank of Ln

    nu           (1)                            Student T Degrees Of Freedom Parameter
    rho:         (1)                            ADMM Hyperparameter    
    eta          (1)                            Hyperparameter that controls the regularization to obtain a k-component graph
    beta         (1)                            Weight Sparsity Hyperparameter
    '''

    T_n = X.shape[0] # Window (Frame) Length
    p = X.shape[1]   # Number Of Features

    # can be replaced by an einsum
    S_sum = 0
    for t in range(0,T_n):
        S_sum += ( np.outer(X[t,:],X[t,:]) )/( X[t,:].T @ L_n @ X[t,:] + nu )
    C = (p + nu)/T_n
    S_tilde = C * S_sum


    aw = L_star(S_tilde + Phi_n + rho*(L_n - L_nUpdate) + eta*np.outer(V,V.T) )
    bw = -u_n - rho*( u_n + np.multiply(a,w_lagged) ) + D_star(z_n - rho*(d - D_from_w(w) ))
    cw = (1 - rho/(rho*(4*rho -1) ) ) * w - 1/( rho*(4*rho - 1) ) * (aw + bw)
    cth = np.sqrt((2*beta)/(rho*(4*rho - 1))) * np.repeat(1,p*(p-1)/2)

    wUpdate = np.multiply((cw > cth),cw)
    return wUpdate


def softThresh(v,thr):

    '''
    v:   ( )  Value(s) to be Threshed  
    thr: (1) Threshold Value
    '''

    temp = abs(v) - thr
    temp = temp*(temp>0)
    return(np.sign(v)*temp)


def u_update(wUpdate,a,w_lagged,mu_vec,rho,alpha):

    '''
    wUpdate:  ( n_features*(n_features-1)/2 )  Updated Graph Weights for current Frame
    a:        ( n_features*(n_features-1)/2 )  VAR Weights for Previous Frame
    w_lagged: ( n_features*(n_features-1)/2 )  Graph Weights for previous Frame

    mu_vec:   ( n_features*(n_features-1)/2 )  Temporal Consistency Parameter Dual
    alpha:    (1)                              Temporal Weight Sparsity Hyperparameter
    rho:      (1)                              ADMM Hyperparameter
    '''

    # Update u
    u_nTemp  = wUpdate - a*w_lagged - mu_vec/rho
    thr = alpha/(rho)
    u_nUpdate = softThresh(u_nTemp, thr)
    return u_nUpdate

def a_update( a, w_lagged, wUpdate, u_nUpdate, mu_vec, gamma, rho):
    '''
    a:        ( n_features*(n_features-1)/2 )  VAR Weights for Previous Frame
    wUpdate:  ( n_features*(n_features-1)/2 )  Updated Graph Weights for current Frame
    u_nUpdate:( n_features*(n_features-1)/2 )  Updated Temporal Consistency Parameter
    mu_vec:   ( n_features*(n_features-1)/2 )  Temporal Consistency Parameter Dual 
    gamma:    (1)                              Hyperparameter that controls the sparsity of VAR coefficients
    rho:      (1)                              ADMM Hyperparameter    
    '''
    
    # update a
    f_temp = wUpdate - u_nUpdate - mu_vec/rho
    f_temp[f_temp<0] = 0
    idx = w_lagged > 0
    thr = gamma/(rho* w_lagged[idx]**2)
    a[idx] = softThresh(f_temp[idx]/w_lagged[idx], thr)
    a[~idx] = 0
    aUpdate = a.copy()
    return aUpdate

def V_update(wUpdate,k):

    '''
    wUpdate:  ( n_features*(n_features-1)/2 )  Updated Graph Weights for current Frame
    k:        ( n_clusters )                   Number of Clusters   
    '''
    LwUpdate = L_from_w(wUpdate)
    # update V
    eigenVals, eigenVectors = np.linalg.eigh(LwUpdate) # Returns in Ascending Order
    vUpdate = eigenVectors[:, :k]   # smallest k eigenvalues
    return vUpdate


def dual_update( rho, wUpdate, L_nUpdate, Phi_n, aUpdate, w_lagged, mu_vec, u_nUpdate, d, z_n):

    '''
    rho:      (1)                              ADMM Hyperparameter  

    wUpdate:   ( n_features*(n_features-1)/2 )  Updated Graph Weights for current Frame
    L_nUpdate: ( n_features, n_features )       Updated Laplacian Constrained Graph Matrix
    Phi_n:     ( n_features, n_features )       Laplacian Constrained Graph Matrix Dual

    aUpdate:   ( n_features*(n_features-1)/2 )  Updated VAR Weights for Previous Frame
    w_lagged:  ( n_features*(n_features-1)/2 )  Graph Weights for previous Frame  
    u_nUpdate: ( n_features*(n_features-1)/2 )  Updated Temporal Consistency Parameter
    mu_vec:    ( n_features*(n_features-1)/2 )  Temporal Consistency Parameter Dual 

    z_n:         (n_features)                   Degree Dual 
    d:           (1)                            Degree Constraint
    
    '''

    # update Phi
    Phi_n_res =  L_from_w(wUpdate) - L_nUpdate
    Phi_nUpdate = Phi_n + rho * Phi_n_res

    # update mu
    u_n_res = u_nUpdate - wUpdate + np.multiply(aUpdate,w_lagged) 
    mu_vecUpdate = mu_vec + rho * u_n_res

    # update z
    z_n_res = D_from_w(wUpdate) - d
    z_nUpdate = z_n + rho * z_n_res

    return Phi_nUpdate, mu_vecUpdate, z_nUpdate, Phi_n_res, u_n_res, z_n_res


def residual_Update( rho, L_n, L_nUpdate, Phi_n_res, z_n_res, primal_lap_residual, primal_deg_residual, dual_residual):

    '''
    rho:         (1)                             ADMM Hyperparameter 

    L_n:         ( n_features, n_features )      Laplacian Constrained Graph Matrix
    L_nUpdate:   ( n_features, n_features )      Updated Laplacian Constrained Graph Matrix

    Phi_n_res:   ( n_features, n_features )      Residual of Laplacian Constrained Graph Matrix Dual 
    z_n_res      (n_features)                    Residual of Degree Dual 
    '''

    primal_lap_residual = np.hstack( ( primal_lap_residual,     np.linalg.norm(Phi_n_res, "f") ) )
    primal_deg_residual = np.hstack( ( primal_deg_residual,     np.linalg.norm(z_n_res,   2) ) )
    dual_residual       = np.hstack( ( dual_residual,       rho*np.linalg.norm( L_star(L_n - L_nUpdate), 2) ) )

    return primal_lap_residual, primal_deg_residual, dual_residual


def updateADMMpenalties( update_rho, rho, rho_seq, L_n, k, wUpdate, update_eta, eta, eta_seq, early_stopping):

    has_converged = False 
    # update rho
    if update_rho: 

        eig_vals = np.linalg.eigvalsh(L_n)
        
        n_zero_eigenvalues = sum(eig_vals < 1e-9)
        if k < n_zero_eigenvalues:

            rho = .5 * rho

        elif k > n_zero_eigenvalues:

            rho = 2 * rho

        else:

            if early_stopping: 
                has_converged = True
                
            
        rho_seq = np.hstack((rho_seq, rho))
        



    if update_eta:


        eig_vals = np.linalg.eigvalsh(L_from_w(wUpdate))
        n_zero_eigenvalues = sum(eig_vals < 1e-9)
        if k < n_zero_eigenvalues:
            eta = .5 * eta
        elif k > n_zero_eigenvalues:
            eta = 2 * eta
        else: 

            if early_stopping:
                has_converged = True
                
        
        
        eta_seq = np.hstack((eta_seq, eta))

    return rho, eta, rho_seq, eta_seq, has_converged




def update_parameters( L_nUpdate, wUpdate, u_nUpdate, aUpdate, vUpdate, Phi_nUpdate, mu_vecUpdate, z_nUpdate):



    L_n = L_nUpdate
    w = wUpdate

    u_n = u_nUpdate

    # x Pass

    a = aUpdate
    V = vUpdate

    Phi_n = Phi_nUpdate
    mu_vec = mu_vecUpdate
    z_n = z_nUpdate

    return  L_n, w, u_n, a, V, Phi_n, mu_vec, z_n
