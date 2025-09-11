import numpy as np
from numpy.linalg import inv, pinv, slogdet, LinAlgError
from scipy.special import digamma, polygamma, gammaln

# ---------- helpers ----------
def _quadforms(X, mu, Lambda):
    """q_i = (x_i - mu)^T Lambda (x_i - mu) for all rows."""
    diff = X - mu[None, :]
    y = diff @ Lambda  # (N,D)
    return np.einsum("ni,ni->n", diff, y)

def _loglik_student_t(X, mu, Lambda, nu):
    """Total log-likelihood under multivariate Student-t (precision form)."""
    N, D = X.shape
    q = _quadforms(X, mu, Lambda)
    sign, logdet_L = slogdet(Lambda)
    if sign <= 0:
        return -np.inf
    c = (
        gammaln((nu + D) / 2.0)
        - gammaln(nu / 2.0)
        + 0.5 * logdet_L
        - (D / 2.0) * np.log(nu * np.pi)
    )
    return np.sum(c - 0.5 * (nu + D) * np.log1p(q / nu))

def _update_nu(E_eta, E_log_eta, nu0, max_iter=50, tol=1e-8):
    """
    Newton–Raphson for: psi(nu/2) - log(nu/2) = 1 + mean(E[log eta]) - mean(E[eta]).
    """
    target = 1.0 + float(np.mean(E_log_eta)) - float(np.mean(E_eta))
    nu = max(float(nu0), 1e-6)
    for _ in range(max_iter):
        f  = digamma(nu / 2.0) - np.log(nu / 2.0) - target
        df = 0.5 * polygamma(1, nu / 2.0) - 1.0 / nu
        step = f / df
        nu_new = max(nu - step, 1e-8)
        if abs(nu_new - nu) < tol * (1.0 + abs(nu)):
            return nu_new
        nu = nu_new
    return nu

def _safe_inv(S):
    """Try inv, fall back to pinv if S is near-singular."""
    try:
        return inv(S)
    except LinAlgError:
        return pinv(S)

# ---------- EM main ----------
def em_multivariate_student_t(
    X, mu=None, Lambda=None, nu=10.0,
    max_iter=200, tol=1e-6, jitter=1e-6, verbose=False
):
    """
    Fit a multivariate Student's t via EM (precision parameterization).
    
    Parameters
    ----------
    X : (N,D) array
        Data.
    mu : (D,) or None
        Initial mean (defaults to sample mean).
    Lambda : (D,D) or None
        Initial precision (defaults to inverse sample covariance).
    nu : float
        Initial degrees of freedom (>0).
    max_iter : int
        Max EM iterations.
    tol : float
        Relative LL tolerance for convergence.
    jitter : float
        Base jitter added to the scatter before inversion (stability).
    verbose : bool
        Print progress if True.
        
    Returns
    -------
    mu, Lambda, nu, loglik_trace
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape

    # --- init ---
    if mu is None:
        mu = X.mean(axis=0)
    if Lambda is None:
        S0 = np.cov(X.T, bias=False)
        S0 = 0.5 * (S0 + S0.T)
        S0 += jitter * np.trace(S0) / D * np.eye(D)
        Lambda = _safe_inv(S0)
    nu = float(nu)

    loglik_trace = []
    prev_ll = -np.inf

    for it in range(1, max_iter + 1):
        mu_old, Lambda_old, nu_old = mu.copy(), Lambda.copy(), nu

        # ----- E-step -----
        q = _quadforms(X, mu, Lambda)              # (N,)
        E_eta = (nu + D) / (nu + q)                # weights E[eta_i]
        E_log_eta = digamma((nu + D) / 2.0) - np.log((nu + q) / 2.0)

        # ----- M-step -----
        # mean (weighted)
        wsum = E_eta.sum()
        mu = (X * E_eta[:, None]).sum(axis=0) / wsum

        # weighted scatter -> precision
        diff = X - mu[None, :]
        S = (diff * E_eta[:, None]).T @ diff / N
        # symmetrize and stabilize
        S = 0.5 * (S + S.T)
        S += jitter * np.trace(S) / D * np.eye(D)
        Lambda = _safe_inv(S)
        # symmetrize Lambda too (tiny numerical asymmetries)
        Lambda = 0.5 * (Lambda + Lambda.T)

        # update nu (scalar)
        nu = _update_nu(E_eta, E_log_eta, nu)

        # ----- log-likelihood with updated params -----
        ll = _loglik_student_t(X, mu, Lambda, nu)
        loglik_trace.append(ll)
        if verbose:
            print(f"[{it:03d}] ll={ll:.6f}  nu={nu:.4f}")

        # convergence: ΔLL and parameter movement
        param_delta = max(
            np.linalg.norm(mu - mu_old, ord=np.inf),
            np.linalg.norm(Lambda - Lambda_old, ord='fro') / (1e-12 + np.linalg.norm(Lambda_old, ord='fro')),
            abs(nu - nu_old) / (1e-12 + abs(nu_old))
        )
        if (np.isfinite(prev_ll) and ll - prev_ll < tol * (1.0 + abs(prev_ll))) or (param_delta < 1e-7):
            break
        prev_ll = ll

    return mu, Lambda, nu, np.array(loglik_trace)


# -------------------- demo / usage --------------------
if __name__ == "__main__":
    from scipy.stats import multivariate_t

    # mean (mu)
    loc = np.zeros(5)

    # "shape" matrix (Σ) used by scipy's multivariate_t (this is NOT the covariance)
    shape = np.array([
        [1.5,     0.75,    0.375,   0.1875,  0.09375],
        [0.75,    1.5,     0.75,    0.375,   0.1875 ],
        [0.375,   0.75,    1.5,     0.75,    0.375  ],
        [0.1875,  0.375,   0.75,    1.5,     0.75   ],
        [0.09375, 0.1875,  0.375,   0.75,    1.5    ],
    ])
    Lambda_true = inv(shape)   # precision in the EM model
    df = 6.0

    print("\nTrue mu:\n", loc)
    print("\nTrue precision Lambda:\n", Lambda_true)
    print("\nTrue nu:\n", df)

    # Generate samples
    rng = np.random.default_rng(0)
    dist = multivariate_t(loc=loc, shape=shape, df=df, seed=rng)
    num_samples = 200000  # more data → better ν
    samples = dist.rvs(size=num_samples)

    # Fit
    mu_hat, Lambda_hat, nu_hat, ll = em_multivariate_student_t(
        samples, mu=None, Lambda=None, nu=8.0,
        max_iter=300, tol=1e-10, jitter=1e-6, verbose=False
    )

    print("\nEstimated mu:\n", mu_hat)
    print("\nEstimated precision Lambda:\n", Lambda_hat)
    print("\nEstimated nu:\n", nu_hat)

    # simple diagnostics
    rel_err = np.linalg.norm(Lambda_hat - Lambda_true, 'fro') / np.linalg.norm(Lambda_true, 'fro')
    print("\nRelative Frobenius error vs true Λ:", rel_err)
    print("Max |mu_hat - mu_true|:", np.max(np.abs(mu_hat - loc)))


    import numpy as np
from scipy.special import digamma, polygamma, gammaln

def _update_nu_univariate(E_eta, E_log_eta, nu0, max_iter=50, tol=1e-8):
    target = 1.0 + np.mean(E_log_eta) - np.mean(E_eta)
    nu = max(float(nu0), 1e-6)
    for _ in range(max_iter):
        f  = digamma(nu/2.0) - np.log(nu/2.0) - target
        df = 0.5 * polygamma(1, nu/2.0) - 1.0/nu
        step = f / df
        nu_new = max(nu - step, 1e-8)
        if abs(nu_new - nu) < tol * (1.0 + abs(nu)):
            return nu_new
        nu = nu_new
    return nu

def em_student_t_univariate(x, mu=None, lam=None, nu=10.0,
                            max_iter=200, tol=1e-6, verbose=False):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if mu  is None: mu  = x.mean()
    if lam is None: lam = 1.0 / np.var(x)   # precision of t's scale s^2 (not variance)
    nu = float(nu)

    loglik_trace = []
    prev_ll = -np.inf

    for it in range(1, max_iter+1):
        # Save old params for param-change stop
        mu_prev, lam_prev, nu_prev = mu, lam, nu

        # -------- E-step --------
        q = lam * (x - mu)**2
        E_eta     = (nu + 1.0) / (nu + q)
        E_log_eta = digamma((nu + 1.0)/2.0) - np.log((nu + q)/2.0)

        # -------- M-step --------
        mu  = np.sum(E_eta * x) / np.sum(E_eta)
        lam = N / np.sum(E_eta * (x - mu)**2)
        nu  = _update_nu_univariate(E_eta, E_log_eta, nu)

        # -------- Recompute q with UPDATED params --------
        q = lam * (x - mu)**2

        # -------- Log-likelihood with UPDATED params --------
        ll = (
            gammaln((nu + 1.0)/2.0) - gammaln(nu/2.0)
            + 0.5*np.log(lam / (np.pi * nu))
            - ((nu + 1.0)/2.0) * np.log1p(q / nu)
        ).sum()
        loglik_trace.append(ll)

        if verbose:
            print(f"[{it:03d}] ll={ll:.6f}, mu={mu:.3f}, lam={lam:.3f}, nu={nu:.3f}")

        # Convergence: ΔLL and parameter change
        param_change = max(abs(mu - mu_prev), abs(lam - lam_prev), abs(nu - nu_prev))
        if (np.isfinite(prev_ll) and ll - prev_ll < tol * (1.0 + abs(prev_ll))) or (param_change < 1e-7):
            break
        prev_ll = ll

    return mu, lam, nu, np.array(loglik_trace)


import numpy as np
from scipy.stats import t

# --- Generate synthetic univariate Student-t data ---
rng = np.random.default_rng(0)

N = 5000
true_mu = 0.0     # location
true_sigma = 2.0  # scale (std dev-like, not precision)
true_nu = 5.0     # degrees of freedom

# scipy's t takes df=nu, loc=mu, scale=sigma
x = t.rvs(df=true_nu, loc=true_mu, scale=true_sigma, size=N, random_state=rng)


# Fit with EM
mu_hat, lam_hat, nu_hat, ll = em_student_t_univariate(x, mu=None, lam=0.5, nu=8,
                            max_iter=200, tol=1e-6, verbose=True)
print("\nEstimated parameters:")
print("mu =", mu_hat)
print("sigma^2 ≈", np.sqrt(1/lam_hat))
print("nu =", nu_hat)


from scipy.stats import multivariate_t

# mean (mu)
mu = [0., 0., 0., 0., 0.]

# covariance (Sigma)
Sigma = [
    [1.5,     0.75,    0.375,   0.1875,  0.09375],
    [0.75,    1.5,     0.75,    0.375,   0.1875 ],
    [0.375,   0.75,    1.5,     0.75,    0.375  ],
    [0.1875,  0.375,   0.75,    1.5,     0.75   ],
    [0.09375, 0.1875,  0.375,   0.75,    1.5    ],
]

# degrees of freedom
df = 6.0

# Create a frozen multivariate_t object
# This allows fixing the parameters and then calling methods like rvs()
dist = multivariate_t(loc=mu, shape=Sigma, df=df)

# Generate samples
num_samples = 1000
samples = dist.rvs(size=num_samples)
samples.shape 


Prec = np.round(np.linalg.pinv(Sigma),4)
print(Prec)