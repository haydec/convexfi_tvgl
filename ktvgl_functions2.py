import math
import numpy as np
from sklearn.preprocessing import scale
from numpy.linalg import slogdet
from scipy.special import digamma, polygamma, gammaln
from functools import lru_cache


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

'''
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
'''



@lru_cache(maxsize=None)
def _lower_pairs_columnwise(p: int):
    """
    Return (i_idx, j_idx) for all i>j, ordered by increasing j, then i.
    This matches k = (i-j) + ((j-1)/2)*(2p - j) in your original code.
    """
    i_all, j_all = np.tril_indices(p, k=-1)     # default order: row-major (by i)
    order = np.lexsort((i_all, j_all))          # sort by j first, then i
    return i_all[order], j_all[order]

def L_star(Y: np.ndarray, *, assume_symmetric: bool = False) -> np.ndarray:
    """
    Vectorized L* operator:
        for each edge (i>j): L*[Y]_{ij} = Y_ii - Y_ij - Y_ji + Y_jj
    If Y is symmetric, this simplifies to: diag_i + diag_j - 2*Y_ij.

    Returns a length m = p*(p-1)//2 vector ordered column-wise by j (to match your k).
    """
    Y = np.asarray(Y)
    if Y.ndim != 2 or Y.shape[0] != Y.shape[1]:
        raise ValueError("Y must be a square matrix.")
    p = Y.shape[0]
    m = p * (p - 1) // 2

    # indices of all (i,j) with i>j, ordered like your original k
    i_idx, j_idx = _lower_pairs_columnwise(p)

    d = np.diag(Y)
    if assume_symmetric:
        # Y_ji == Y_ij, so formula reduces and avoids extra gather:
        v = d[i_idx] + d[j_idx] - 2.0 * Y[i_idx, j_idx]
    else:
        v = d[i_idx] - Y[i_idx, j_idx] - Y[j_idx, i_idx] + d[j_idx]

    # already in desired order (column-wise by j)
    # shape is (m,)
    return v


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
# https://github.com/convexfi/fitHeavyTail/blob/master/R/fit_mvt.R


def nu_mle_diag_resampled(X, *, fT_resampling=4, N_resampling_factor=1.2,
                          nu_min=2.5, nu_max=100.0, random_state=None):
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    rng = np.random.default_rng(random_state)
    mu = X.mean(axis=0)
    Xc = X - mu
    var = (Xc**2).sum(axis=0) / (T - 1)
    if np.any(var <= 0):
        raise ValueError("Non-positive variance encountered.")

    delta2_var = Xc**2 / var
    Tf = T * fT_resampling
    delta_rep = np.tile(delta2_var, (fT_resampling, 1))
    N_resampling = int(round(N_resampling_factor * N))
    idx = rng.integers(0, N, size=(Tf, N_resampling))
    delta2_cov = delta_rep[np.arange(Tf)[:, None], idx].sum(axis=1)

    def negLL(nu: float) -> float:
        if nu <= 2.0: return np.inf
        term1 = (N * Tf) * 0.5 * (math.log((nu - 2.0) / nu))
        scaled = nu + (nu / (nu - 2.0)) * delta2_cov
        if np.any(scaled <= 0): return np.inf
        term2 = ((N + nu) * 0.5) * np.log(scaled).sum()
        term3 = -Tf * math.lgamma((N + nu) * 0.5)
        term4 =  Tf * math.lgamma(nu * 0.5)
        term5 = -(nu * Tf * 0.5) * math.log(nu)
        return term1 + term2 + term3 + term4 + term5

    def golden_section_minimize(f, a, b, tol=1e-5, max_iter=200):
        invphi = (math.sqrt(5) - 1) / 2
        invphi2 = (3 - math.sqrt(5)) / 2
        c = a + invphi2 * (b - a); d = a + invphi * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(max_iter):
            if abs(b - a) < tol * (abs(a) + abs(b)): break
            if fc < fd:
                b, d, fd = d, c, fc
                c = a + invphi2 * (b - a); fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + invphi * (b - a); fd = f(d)
        return c if fc < fd else d

    return float(golden_section_minimize(negLL, nu_min, nu_max))

# --- EM fit of mu and full Sigma with fixed nu (no missing data, no factor model) ---
def fit_multivariate_t(
    X,
    nu: float | None = None,
    *,
    estimate_nu: bool = True,
    nu_min: float = 2.5,
    nu_max: float = 100.0,
    max_iter: int = 200,
    ptol: float = 1e-3,
    px_em: bool = True,
    jitter: float = 1e-9,
    random_state: int | None = None
):
    """
    EM updates for mu and full scatter Sigma of a multivariate t with fixed nu.
    If estimate_nu=True and nu is None, nu is obtained with diag-resampled MLE (one-shot).

    Returns:
        dict(mu, Sigma(scatter), cov, nu, converged, num_iterations)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (T x N).")
    T, N = X.shape
    if T <= N:
        raise ValueError("Need T > N.")

    # 1) One-shot nu (if requested)
    if estimate_nu and (nu is None):
        nu = nu_mle_diag_resampled(X, nu_min=nu_min, nu_max=nu_max, random_state=random_state)
    if nu is None:
        raise ValueError("nu must be provided or estimate_nu=True.")

    # 2) Initialize mu, Sigma like in R (scatter from sample cov)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov_sample = (Xc.T @ Xc) / (T - 1)                # sample covariance
    Sigma = ((max(nu, 2.1) - 2.0) / max(nu, 2.1)) * cov_sample  # scatter init

    # 3) EM loop
    def fnu(nu):  # used in R stopping crit
        return nu / (nu - 2.0)

    for it in range(1, max_iter + 1):
        mu_old = mu.copy()
        Sigma_old = Sigma.copy()

        # E-step
        # r2_t = x_c^T Sigma^{-1} x_c   for each row
        # add tiny jitter to ensure SPD in solve
        Sigma_j = Sigma + jitter * np.eye(N)
        Sinv = np.linalg.inv(Sigma_j)
        Xc = X - mu  # current centering
        r2 = np.einsum("ti,ij,tj->t", Xc, Sinv, Xc)   # shape (T,)
        E_tau = (N + nu) / (nu + r2)                  # shape (T,)

        ave_E_tau = E_tau.mean()
        ave_E_tau_X = (E_tau[:, None] * X).mean(axis=0)  # (1/T) * sum_t E_tau_t * x_t

        # M-step
        mu = ave_E_tau_X / ave_E_tau
        Xc = X - mu
        # (1/T) * Xc^T diag(E_tau) Xc
        Sigma = (Xc.T * E_tau) @ Xc / T
        if px_em:
            Sigma = Sigma / ave_E_tau  # PX-EM acceleration (alpha = ave_E_tau)

        # stopping: relative change of mu and Sigma
        mu_denom = np.maximum(1.0, np.abs(mu_old) + np.abs(mu))
        mu_conv = np.all(np.abs(mu - mu_old) <= 0.5 * ptol * mu_denom)

        Sig_denom = np.maximum(1.0, np.abs(Sigma_old) + np.abs(Sigma))
        Sig_conv = np.all(np.abs(Sigma - Sigma_old) <= 0.5 * ptol * Sig_denom)

        if mu_conv and Sig_conv:
            converged = True
            break
    else:
        converged = False
        it = max_iter

    # covariance from scatter
    cov = (nu / (nu - 2.0)) * Sigma if nu > 2 else np.full_like(Sigma, np.nan)

    return {
        "mu": mu,
        "Sigma": Sigma,   # scatter
        "cov": cov,       # covariance
        "nu": nu,
        "converged": converged,
        "num_iterations": it,
    }

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
    w = w/np.sum(w,axis=0,keepdims=True) 
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
    V = eigVecs[:,0:k] # Penalty to control rank of Ln 

    return w, Aw, a, w_lagged, L_n, Phi_n, u_n, mu_vec, d, z_n, V



def L_update( wUpdate, Phi_n, rho, k):

    ''' 
    L_n:   (n_features, n_features) Laplace Matrix (Ln = Diag(Wn1) - Wn = Lwn) Laplacian operator “A Unified Framework for Structured Graph Learning via Spectral Constraints”
    Phi_n: (n_features, n_features) ADMM Dual Variable 
    rho:   (1)                      ADMM Hyperparameter   
    k:     (n_clusters)             Number of Clusters
    '''
    # Reference: Equation 16
    LwUpdate = L_from_w(wUpdate)
    p = LwUpdate.shape[0] #  p: Number of features
    Y = LwUpdate + Phi_n/rho

    eigVals,eigVecs = np.linalg.eigh(Y)
    eigenVecs = eigVecs[:, ::-1]       # take first (p-k) eigenvectors
    eigenVals = eigVals[::-1]  
    U = eigenVecs[:, :p-k]         # take first (p-k) eigenvectors
    R = np.diag(eigenVals[:p-k]  )
    I = np.eye(R.shape[0]) 
    L_nUpdate = 0.5 * U @ ( R + np.sqrt(  R**2 + 4/rho * I  ) ) @ U.T



    return L_nUpdate


def w_update(X, w, w_lagged, a, nu, LstarXsq, L_n, u_n, mu_vec, z_n, Phi_n, V, rho, d, eta, beta):
    '''
    X:           (n_samples, n_features)        Data Array For Window (Frame) Length

    w:           (n_features*(n_features-1)/2)  Graph Weights for current Frame
    w_lagged:    (n_features*(n_features-1)/2)  Graph Weights for previous Frame
    a            (n_features*(n_features-1)/2)  VAR Weights for Previous Frame
    u_n:         (n_features*(n_features-1)/2)  Temporal Consistency Parameter
    mu_vec:      (n_features*(n_features-1)/2)  Temporal Consistency Parameter Dual

    LstarXsq:    (n_features, n_features)       Current weight Converted to Laplacian Matrix
    L_n:         (n_features, n_features)       Updated Laplacian Constrained Graph Matrix
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

    Lw = L_from_w(w) # (n_features, n_features) Laplacian with current weights
    LstarLw = L_star(Lw)
    DstarDw = D_star(D_from_w(w))
    #print("LstarLw: ",LstarLw)
    #print("DstarDw: ",DstarDw)
    S_tilde = np.repeat(0, .5*p*(p-1))

    for t in range(T_n):
        S_tilde = S_tilde + LstarXsq[t] * ( (p + nu) / ( ( w @ LstarXsq[t] ) + nu ) )
    #print("S_tilde",S_tilde)

    a_w = S_tilde/T_n + L_star(eta * (V @ V.T) + Phi_n - rho*L_n ) + rho*LstarLw
    b_w = -mu_vec - rho*(u_n + a * w_lagged) + D_star(z_n - rho*d) + rho*DstarDw
 

    ratio = 1 / (rho*(4*p-1))
    c_w = (1-rho*ratio)*w - ratio *  (a_w + b_w)

    thr = np.sqrt( 2*beta *ratio )
    w_new = np.multiply(c_w > thr, c_w)     # (n_features*(n_features-1)/2)  Updated Graph Weights for current Frame
    Lw_new = L_from_w(w_new)                # (n_features, n_features)       Updated weights Converted to Laplacian Matrix
    Aw_new = A_from_w(w_new)                # (n_features, n_features)       Updated weights Converted to Adjacency Matrix
    return w_new, Lw_new, Aw_new




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
    #print("u_nTemp", u_nTemp)
    thr = alpha/(rho)
    #print("thr", thr)
    u_nUpdate = softThresh(u_nTemp, thr)
    return u_nUpdate

def a_update( w_lagged, wUpdate, u_nUpdate, mu_vec, gamma, rho):
    '''
    wUpdate:  ( n_features*(n_features-1)/2 )  Updated Graph Weights for current Frame
    u_nUpdate:( n_features*(n_features-1)/2 )  Updated Temporal Consistency Parameter
    mu_vec:   ( n_features*(n_features-1)/2 )  Temporal Consistency Parameter Dual 
    gamma:    (1)                              Hyperparameter that controls the sparsity of VAR coefficients
    rho:      (1)                              ADMM Hyperparameter    
    '''
    
    aUpdate = np.zeros_like(w_lagged) # ( n_features*(n_features-1)/2 )  Updated VAR Weights for current Frame
    
    f_temp = wUpdate - u_nUpdate - mu_vec/rho
    f_temp[f_temp<0] = 0
    idx = w_lagged > 0
    thr = gamma/(rho* w_lagged[idx]**2)
    aUpdate[idx] = softThresh(f_temp[idx]/w_lagged[idx], thr)
    aUpdate [~idx] = 0
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
        #print("eig_vals ", eig_vals)
        n_zero_eigenvalues = sum(eig_vals < 1e-9)
        #print("n_zero_eigenvalues ", n_zero_eigenvalues)
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
        #print("eig_vals ", eig_vals)
        n_zero_eigenvalues = sum(eig_vals < 1e-9)
        #print("n_zero_eigenvalues ", n_zero_eigenvalues)
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
