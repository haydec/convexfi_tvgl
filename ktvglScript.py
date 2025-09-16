import cProfile
import pstats
import io

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from ktvgl_functions2 import init_parameters, fit_multivariate_t, L_update, w_update,  u_update, a_update, V_update, dual_update, residual_Update, update_parameters, updateADMMpenalties, L_star,L_from_w,D_star,D_from_w

def main():




    winLen = 200

    #log_returns = pd.read_csv("log_returns.csv")
    #r1 = min(100,log_returns.shape[1])
    #Xraw = log_returns.iloc[0:winLen, 0:r1] 
    log_rw_returns = pd.read_csv("log_rw_returns.csv")
    r2 = min(5,log_rw_returns.shape[1])
    Xraw = log_rw_returns.iloc[0:winLen, 0:r2] 

    maxIter = 200
    update_rho = False
    update_eta = True
    early_stopping = False


    X = scale(Xraw.to_numpy())

    # number of nodes
    p = X.shape[1] # n_features
    k = 1 # n_clusters

    # number of observations
    T_n = X.shape[0]      # n_observations in Frame
    sigma_e = np.exp(10) # Student T Distribution Standard Dev. 
    alpha = 2/(T_n*sigma_e)
    beta = 2*np.log(sigma_e)/T_n

    print("sigma_e",sigma_e)
    print("alpha",alpha)
    print("beta",beta)


    rho = 2    # ADMM hyperparameter. 
    gamma = 10 # hyperparameter that controls the sparsity of VAR coefficients
    eta = 1e-8 # hyperparameter that controls the regularization to obtain a k-component graph
    #tau = 2 # Not used
    #mu = 2 # Not used


    # BOOK KEEPING
    # residual vectors
    primal_lap_residual = np.array([], dtype=float)
    primal_deg_residual = np.array([], dtype=float)
    dual_residual       = np.array([], dtype=float)

    lagrangian = np.array([], dtype=float)
    rho_seq = np.array([], dtype=float)
    eta_seq = np.array([], dtype=float)

    elapsed_time = np.array([], dtype=float)


    #mu, Sigma, nu, history = fit_multivariate_t( X, nu_init=30.0, max_iter=200, tol=1e-6, fix_nu=None, jitter=1e-6, verbose=False)
    w, Aw, a, w_lagged, L_n, Phi_n, u_n, mu_vec, d, z_n, V = init_parameters(X,k)
    nu = 10.02571
    #nu = 5.16522 

    LstarXsq = [None]*(T_n)
    for t in range(T_n):
        x = X[t, :]
        LstarXsq[t] = L_star(np.outer(x, x))
    

    for iter in range(0,maxIter):


        wUpdate, LwUpdate, AwUpdate   = w_update( X, w, w_lagged, a, nu, LstarXsq, L_n, u_n, mu_vec, z_n, Phi_n, V, rho, d, eta, beta)              
        u_nUpdate = u_update(wUpdate,a,w_lagged,mu_vec,rho,alpha)
        # x_update()
        aUpdate = a_update( w_lagged, wUpdate, u_nUpdate, mu_vec, gamma, rho)
        vUpdate = V_update(wUpdate,k)
        L_nUpdate = L_update(wUpdate, Phi_n, rho, k)
        Phi_nUpdate, mu_vecUpdate, z_nUpdate, Phi_n_res, u_n_res, z_n_res = dual_update(rho, wUpdate, L_nUpdate, Phi_n, aUpdate, w_lagged, mu_vec, u_nUpdate, d, z_n)
        primal_lap_residual, primal_deg_residual, dual_residual = residual_Update(rho, L_n, L_nUpdate, Phi_n_res, z_n_res, primal_lap_residual, primal_deg_residual, dual_residual) 
        rho, eta, rho_seq, eta_seq, has_converged = updateADMMpenalties( update_rho, rho, rho_seq,L_n, k, wUpdate, update_eta, eta, eta_seq, early_stopping)    
        L_n, w, u_n, a, V, Phi_n, mu_vec, z_n = update_parameters( L_nUpdate, wUpdate, u_nUpdate, aUpdate, vUpdate, Phi_nUpdate, mu_vecUpdate, z_nUpdate)

        if has_converged:
            break

    np.set_printoptions(threshold=np.inf)  # no truncation
    print("w: ", w)
    print("Done")


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    s = io.StringIO()
    # Sort by cumulative time spent in each function (includes subcalls)
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)   # top 30 entries
    print(s.getvalue())