# =====================================================================
# Description: contains all differentiable objective functions used for
# training the parameters of the state abstraction
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import synthetic_system as ss
import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import verification_tools as vt

A_GLOBAL = np.array([[0.8, -0.3],
                     [0.3,  0.8]])

# =====================================================================
# Optimization
# =====================================================================

def optimize_grid_widths(
        L: np.array,
        volume: float
    ):
    """
    Convex optimization algorithm to solve for minimizing aspect ratio
    """

    n = np.shape(L)[0]
    A = np.eye(n) + L
    gamma = np.log(volume)

    def phi(x): # minimize
        eta = np.exp(x)
        r = A @ eta
        if np.any(r <= 0):
            return np.inf
        return np.sum(np.log(r))

    def grad_phi(x): # derivative
        eta = np.exp(x)
        r = A @ eta
        return eta * (A.T @ (1.0 / r))
    
    x0 = np.full(n, gamma / n) # initial guess
    # eta_init = np.exp(x0)
    # E_init = np.prod((A @ eta_init) / eta_init)
    # print(f"Initial avg. succ/state: {E_init}")
    
    constraints = { # linear constraints
        "type": "eq",
        "fun": lambda x: np.sum(x) - gamma,
        "jac": lambda x: np.ones(n),
    }

    res = minimize(
        phi,
        x0,
        jac=grad_phi,
        constraints=[constraints],
        bounds=None,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 5000},
    )

    eta = np.exp(res.x)
    E_pred = np.prod((A @ eta) / eta)

    return eta, E_pred, res
    

# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    # Cache
    gt_reach_fname = "synthetic-v3/synthetic_reach_regions.pkl"

    # Abstraction settings
    domain_lb = np.array([-10.0, -10.0])
    domain_ub = np.array([10.0, 10.0])

    # Define the initial state subset domain
    init_domain_lb = np.array([-10.0, -10.0])
    init_domain_ub = np.array([10.0, 10.0])

    # Evaluate Lipschitz constant array
    L = np.abs(A_GLOBAL)

    # Optimize and determine abstraction shape
    eta, E_pred, res = optimize_grid_widths(L, 0.004444)
    shape = [int(np.round((ub - lb)/e)) for ub, lb, e in zip(domain_ub, domain_lb, eta)]
    print(f"Predicted avg. succ/state: {E_pred}")
    print(f"Optimal shape: {shape}")
    
    # Initialize abstraction parameters
    u1 = jnp.zeros((shape[0],))  # uniform spacing
    u2 = jnp.zeros((shape[1],))
    params = jnp.concatenate([u1, u2])

    # Build and verify
    recall = vt.build_and_verify_from_params(params,
                                             shape,
                                             domain_lb,
                                             domain_ub,
                                             init_domain_lb,
                                             init_domain_ub,
                                             gt_reach_fname=gt_reach_fname,
                                             verbose=True,
                                             log_time=True)
    print(recall)
