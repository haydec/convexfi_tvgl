
import numpy as np

def L_from_w(w, p):
    L = np.zeros((p, p))
    idx = 0
    for i in range(p):
        for j in range(i+1, p):
            L[i, i] += w[idx]
            L[j, j] += w[idx]
            L[i, j] -= w[idx]
            L[j, i] -= w[idx]
            idx += 1
    return L

def A_from_w(w, p):
    A = np.zeros((p, p))
    idx = 0
    for i in range(p):
        for j in range(i+1, p):
            A[i, j] = w[idx]
            A[j, i] = w[idx]
            idx += 1
    return A

def Lstar(M):
    p = M.shape[0]
    vals = []
    for i in range(p):
        for j in range(i+1, p):
            vals.append(M[i, i] + M[j, j] - 2*M[i, j])
    return np.array(vals)

def Dstar(v):
    p = v.shape[0]
    vals = []
    for i in range(p):
        for j in range(i+1, p):
            vals.append(v[i] + v[j])
    return np.array(vals)

def soft_thresh(v, thr):
    return np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)

def compute_student_weights(w, Lstar_xxt, p, nu):
    return (p + nu) / (np.dot(w, Lstar_xxt) + nu)

def w_init_naive(X):
    p = X.shape[1]
    corr = np.corrcoef(X, rowvar=False)
    prec = np.linalg.pinv(corr)
    A_init = np.maximum(-prec, 0)
    np.fill_diagonal(A_init, 0)
    A_init = (A_init + A_init.T) / 2
    row_sums = A_init.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_init /= row_sums
    w = []
    for i in range(p):
        for j in range(i+1, p):
            w.append(A_init[i, j])
    return np.array(w)

def learn_kcomp_heavytail_TV_graph_online(X, w_lagged=0.0, sigma_e=np.exp(0.1),
                                          k=1, heavy_type="gaussian", nu=None,
                                          w0="naive", a0=1.0, d=1.0,
                                          gamma=10.0, eta=1e-8,
                                          update_eta=True, early_stopping=False,
                                          rho=1.0, update_rho=False,
                                          maxiter=100, reltol=1e-5,
                                          verbose=True):
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    p = X.shape[1]
    T_n = X.shape[0]
    alpha = 2/(T_n * sigma_e)
    beta = 2*np.log(sigma_e)/T_n
    LstarSq = [Lstar(np.outer(X[t], X[t])) for t in range(T_n)]
    if isinstance(w0, str):
        w = w_init_naive(X)
    else:
        w = w0.copy()
    Lw = L_from_w(w, p)
    eigvals, eigvecs = np.linalg.eigh(Lw)
    U = eigvecs[:, (p-k):]
    a = np.full_like(w, a0)
    if np.isscalar(w_lagged):
        w_lagged = np.full_like(w, w_lagged)
    Theta = Lw.copy()
    Phi = np.zeros((p, p))
    u = w - a*w_lagged
    mu_vec = np.zeros_like(w)
    z = np.zeros(p)
    has_converged = False
    for i in range(maxiter):
        LstarLw = Lstar(Lw)
        DstarDw = Dstar(np.diag(Lw))
        LstarSweighted = np.zeros_like(w)
        if heavy_type == "student":
            for q in range(T_n):
                LstarSweighted += LstarSq[q] * compute_student_weights(w, LstarSq[q], p, nu)
        else:
            for q in range(T_n):
                LstarSweighted += LstarSq[q]
        grad = (LstarSweighted/T_n + Lstar(eta * (U @ U.T) + Phi - rho * Theta) +
                rho * LstarLw - mu_vec - rho*(u + a*w_lagged) +
                Dstar(z - rho*d) + rho * DstarDw)
        ratio = 1 / (rho*(4*p - 1))
        wi = (1 - rho*ratio)*w - ratio * grad
        thr = np.sqrt(2*beta*ratio)
        wi[wi < thr] = 0
        u = soft_thresh(wi - a*w_lagged - mu_vec/rho, alpha/rho)
        f_temp = wi - u - mu_vec/rho
        f_temp[f_temp < 0] = 0
        idx = w_lagged > 0
        thr_a = gamma/(rho * w_lagged[idx]**2)
        a[idx] = soft_thresh(f_temp[idx]/w_lagged[idx], thr_a)
        a[~idx] = 0
        eigvals, eigvecs = np.linalg.eigh(L_from_w(wi, p))
        U = eigvecs[:, (p-k):]
        eigvals_t, eigvecs_t = np.linalg.eigh(L_from_w(wi, p) + Phi/rho)
        V = eigvecs_t[:, :(p-k)]
        Gamma_U = eigvals_t[:(p-k)]
        Thetai = V @ np.diag((Gamma_U + np.sqrt(Gamma_U**2 + 4/rho))/2) @ V.T
        Phi += rho * (L_from_w(wi, p) - Thetai)
        mu_vec += rho * (u - wi + a*w_lagged)
        z += rho * (np.diag(L_from_w(wi, p)) - d)
        if np.linalg.norm(L_from_w(wi, p) - Lw, 'fro') / np.linalg.norm(Lw, 'fro') < reltol and i > 0:
            has_converged = True
            break
        w = wi
        Lw = L_from_w(w, p)
        Theta = Thetai
    return {
        "laplacian": L_from_w(wi, p),
        "adjacency": A_from_w(wi, p),
        "weights": wi,
        "theta": Thetai,
        "maxiter": i+1,
        "convergence": has_converged
    }






