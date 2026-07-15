# =====================================================================
# Description: contains all differentiable objective functions used for
# training the parameters of the state abstraction
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import unicycle_system_jax as usj
import unicycle_simulation_analysis as usa
import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import verification_tools as vt


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
    gt_reach_fname = "unicycle-taylor/unicycle_gt_reach_regions_100.pkl"

    # Abstraction settings
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])

    # Define the initial state subset domain
    init_domain_lb = np.array([0.0, 0.0, -np.pi/4])
    init_domain_ub = np.array([50.0, 50.0, np.pi/4])

    # Evaluate Lipschitz constant array
    L = usj.quantile_lipschitz_array(
        domain_lb=domain_lb,
        domain_ub=domain_ub
    )

    # Optimize and determine abstraction shape
    eta, E_pred, res = optimize_grid_widths(L, 0.126)
    shape = [int(np.round((ub - lb)/e)) for ub, lb, e in zip(domain_ub, domain_lb, eta)]
    print(f"Predicted avg. succ/state: {E_pred}")
    print(f"Optimal shape: {shape}")
    
    # Initialize abstraction parameters
    u1 = jnp.zeros((shape[0],))  # uniform spacing
    u2 = jnp.zeros((shape[1],))
    u3 = jnp.zeros((shape[2],))
    params = jnp.concatenate([u1, u2, u3])

    # Evaluate the optimized system
    recall, kripke_components = vt.build_and_verify_from_params(
                                            params,
                                            shape,
                                            domain_lb,
                                            domain_ub,
                                            init_domain_lb,
                                            init_domain_ub,
                                            gt_reach_fname=gt_reach_fname,
                                            verbose=True,
                                            log_time=True)
    print(f"    > Recall = {recall}")
    result = usa.evaluate_simulation_metric(
        params,
        kripke_components,
        shape,
        domain_lb,
        domain_ub,
        horizon=3,
        num_samples=64,
        batch_size=256,
        refine=False,
        verbose=False,
    )
    print(f"    > Epsilon = {result.epsilon}")
    print(f"    > Mean epsilon = {result.epsilon_mean}")
    print(f"    > Median epsilon = {result.epsilon_median}")
    print(f"    > Q3 epsilon = {result.epsilon_q3}")
