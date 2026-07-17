# =====================================================================
# Description: contains all differentiable objective functions used for
# training the parameters of the state abstraction
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import os
import unicycle_system_jax as usj
import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import verification_tools as vt

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - fallback for environments without matplotlib
    plt = None


# =====================================================================
# Optimization
# =====================================================================

def plot_history(cost_history, cost: bool, grad: bool, volume, shape, output_path=None, show=False):
    aspect_ratio = [np.round(i/shape[0], 3) for i in shape]
    """Plot and save the optimization cost cost_history."""
    if plt is None:
        print("matplotlib is not installed; skipping cost cost_history plot.")
        return None

    if not cost_history:
        print("Cost cost_history is empty; nothing to plot.")
        return None

    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_history.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    iters = np.arange(1, len(cost_history) + 1)
    ax.plot(iters, cost_history, marker="o", linewidth=1.5, alpha=0.85)
    if cost:
        ax.set_title(f"Weber baseline cost history for volume = {volume} \ncell shape = {shape}, aspect ratio = {aspect_ratio}")
        ax.set_ylabel("Cost")
    if grad:
        ax.set_title(f"Weber baseline gradient history for volume = {volume} \ncell shape = {shape}, aspect ratio = {aspect_ratio}")
        ax.set_ylabel("Gradient Norm")
    # ax.set_title(f"Weber baseline cost hist, volume = {volume}, gamma = {np.log(volume):.4f}")
    # ax.set_xlabel("Iteration")
    # ax.set_ylabel("Cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved cost_history plot to {output_path}")
    return output_path


def optimize_grid_widths(
        L: np.ndarray,
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
    
    #x0 = np.full(n, gamma/n) # initial guess
    x0 =  np.random.randn(n) # initial guess
    # eta_init = np.exp(x0)
    # E_init = np.prod((A @ eta_init) / eta_init)
    # print(f"Initial avg. succ/state: {E_init}")
    
    constraints = { # linear constraints
        "type": "eq",
        "fun": lambda x: np.sum(x) - gamma,
        "jac": lambda x: np.ones(n),
    }

    grad_history = []
    cost_history = []

    def callback(xk):
        eta = np.exp(xk)
        E = np.prod((A @ eta) / eta)
        cost_history.append(float(E))
        print("callback", xk)
        g = grad_phi(xk)
        kkt_residual = np.linalg.norm(g - np.mean(g))
        grad_history.append(kkt_residual)
        print(f"g {len(grad_history)}: {g}, E: {E}")

    res = minimize(
        phi,
        x0,
        jac=grad_phi,
        constraints=[constraints],
        bounds=None,
        method="SLSQP",
        callback=callback,
        options={"ftol": 1e-12, "maxiter": 5000},
    )

    print("Gradient cost_history:", grad_history)
    print("Cost cost_history:", cost_history)
    
    eta = np.exp(res.x)
    E_pred = np.prod((A @ eta) / eta)

    return eta, E_pred, res, cost_history, grad_history
    

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
    volume = ((50/50)**2) * (2*np.pi/50)
    eta, E_pred, res, cost_history, grad_history = optimize_grid_widths(L, volume)
    print(f"Optimized eta : {eta}")
    shape = [int(np.round((ub - lb)/e)) for ub, lb, e in zip(domain_ub, domain_lb, eta)]
    
    plot_history(cost_history, cost=True, grad=False, volume=volume, shape = shape, show=True, output_path="Cost_Hist_02.png")
    plot_history(grad_history, cost=False, grad=True, volume=volume, shape=shape, show=True, output_path="Cost_Hist_02.png")

    print(f"Predicted avg. succ/state: {E_pred}")
    print(f"Optimal shape: {shape}")
    
    # Initialize abstraction parameters
    u1 = jnp.zeros((shape[0],))  # uniform spacing
    u2 = jnp.zeros((shape[1],))
    u3 = jnp.zeros((shape[2],))
    params = jnp.concatenate([u1, u2, u3])

    # # Build and verify
    # recall,_ = vt.build_and_verify_from_params(params,
    #                                          shape,
    #                                          domain_lb,
    #                                          domain_ub,
    #                                          init_domain_lb,
    #                                          init_domain_ub,
    #                                          gt_reach_fname=gt_reach_fname,
    #                                          verbose=True,
    #                                          log_time=True)
    # print(recall)
