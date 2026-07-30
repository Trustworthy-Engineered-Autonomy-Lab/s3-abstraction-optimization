# =====================================================================
# Description: contains all differentiable objective functions used for
# training the parameters of the state abstraction
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import numpy as np
import synthetic_simulation_analysis as ssa
from scipy.optimize import minimize
import jax.numpy as jnp
import verification_tools as vt
import argparse
from pathlib import Path

# =====================================================================
# User-defined settings
# =====================================================================

SHAPE = 100
HORIZON = 3
A_GLOBAL = np.array([[0.8, -0.3],
                     [0.3,  0.8]])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Weber optimization of the spiral abstraction."
    )
    parser.add_argument("--shape", type=int, default=SHAPE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    return parser.parse_args()

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

    cli_args = parse_args()
        
    gt_reach_fname = Path(__file__).parent / 'artifacts' / 'cache' / 'reach.pkl'

    # Abstraction settings
    domain_lb = np.array([-10.0, -10.0])
    domain_ub = np.array([10.0, 10.0])
    init_domain_lb = np.array([-10.0, -10.0])
    init_domain_ub = np.array([10.0, 10.0])

    # Evaluate Lipschitz constant array
    L = np.abs(A_GLOBAL)

    # Optimize and determine abstraction shape
    print("\n=== Optimizing Initial Resolution ===")
    reference_volume = (20.0)*(20.0)/(cli_args.shape**2)
    eta, E_pred, res = optimize_grid_widths(L, reference_volume)
    shape = [int(np.round((ub - lb)/e)) for ub, lb, e in zip(domain_ub, domain_lb, eta)]
    print(f"  Optimal resolution:   {shape[0]}x{shape[1]}")
    
    # Initialize abstraction parameters
    u1 = jnp.zeros((shape[0],))  # uniform spacing
    u2 = jnp.zeros((shape[1],))
    params = jnp.concatenate([u1, u2])

    # Evaluate the optimized system
    print("\n=== Building Final Model ===")
    recall, kripke_components = vt.build_and_verify_from_params(
        params,
        shape,
        domain_lb,
        domain_ub,
        init_domain_lb,
        init_domain_ub,
        gt_reach_fname=gt_reach_fname,
        verbose=True,
        log_time=False
    )
    result = ssa.evaluate_simulation_metric(
        params,
        kripke_components,
        shape,
        domain_lb,
        domain_ub,
        horizon=cli_args.horizon
    )
    print("Final Model Performance")
    print(f"  Recall:       {recall:.4f}")
    print(f"  Sim. metric:  {result.epsilon:.4f}")
    print(f"  Mean delta:   {result.epsilon_mean:.4f}")
    print(f"  Q1 delta:     {result.epsilon_q1:.4f}")
    print(f"  Median delta: {result.epsilon_median:.4f}")
    print(f"  Q3 delta:     {result.epsilon_q3:.4f}")
