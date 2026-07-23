# =====================================================================
# Description: contains all differentiable objective functions used for
# training the parameters of the state abstraction
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import mountain_car_system as mcs
import mountain_car_simulation_analysis as mcsa
import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from pathlib import Path
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

    reference_shape = [200, 200]
    
    gt_reach_fname = str(Path(__file__).with_name("mc_reach_regions.pkl"))

    # Abstraction settings
    domain_lb = np.array([-1.2, -0.07])
    domain_ub = np.array([0.6, 0.07])

    # Define the initial state subset domain
    init_domain_lb = np.array([-1.2, -0.07])
    init_domain_ub = np.array([0.6, 0.07])

    fixed_volume = (domain_ub[0]-domain_lb[0])*(domain_ub[1]-domain_lb[1])/(reference_shape[0]*reference_shape[1])

    # Evaluate Lipschitz constant array
    L = mcs.lipschitz_array(
        domain_lb=domain_lb,
        domain_ub=domain_ub
    )
    print(L)

    # Optimize and determine abstraction shape
    eta, E_pred, res = optimize_grid_widths(L, fixed_volume)
    shape = [int(np.round((ub - lb)/e)) for ub, lb, e in zip(domain_ub, domain_lb, eta)]
    print(f"Predicted avg. succ/state: {E_pred}")
    print(f"Optimal shape: {shape}")
    
    # Initialize abstraction parameters
    u1 = jnp.zeros((shape[0],))  # uniform spacing
    u2 = jnp.zeros((shape[1],))
    params = jnp.concatenate([u1, u2])

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
    result = mcsa.evaluate_simulation_metric(
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

    # result = usa.evaluate_simulation_metric(
    #     params,
    #     kripke_components,
    #     shape,
    #     domain_lb,
    #     domain_ub,
    #     horizon=3,
    #     num_samples=64,
    #     batch_size=256,
    #     refine=False,
    #     verbose=False,
    # )
    # print(f"    > Epsilon = {result.epsilon}")
    # print(f"    > Mean epsilon = {result.epsilon_mean}")
    # print(f"    > Median epsilon = {result.epsilon_median}")
    # print(f"    > Q3 epsilon = {result.epsilon_q3}")


    # result = usa.evaluate_simulation_metric(
    #         params,
    #         kripke_components,
    #         shape,
    #         domain_lb,
    #         domain_ub,
    #         horizon=4,
    #         num_samples=64,
    #         batch_size=256,
    #         refine=False,
    #         verbose=False,
    #     )
    # print(f"    > Epsilon = {result.epsilon}")
    # print(f"    > Mean epsilon = {result.epsilon_mean}")
    # print(f"    > Median epsilon = {result.epsilon_median}")
    # print(f"    > Q3 epsilon = {result.epsilon_q3}")

    # result = usa.evaluate_simulation_metric(
    #         params,
    #         kripke_components,
    #         shape,
    #         domain_lb,
    #         domain_ub,
    #         horizon=5,
    #         num_samples=64,
    #         batch_size=256,
    #         refine=False,
    #         verbose=False,
    #     )
    # print(f"    > Epsilon = {result.epsilon}")
    # print(f"    > Mean epsilon = {result.epsilon_mean}")
    # print(f"    > Median epsilon = {result.epsilon_median}")
    # print(f"    > Q3 epsilon = {result.epsilon_q3}")

