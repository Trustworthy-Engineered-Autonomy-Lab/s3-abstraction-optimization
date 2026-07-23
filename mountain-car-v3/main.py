# =====================================================================
# Description: main script for training abstraction parameters
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import jax
import jax.numpy as jnp
import numpy as np
import pyModelChecking as pmc
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
import mountain_car_abstraction as mca
import verification_tools as vt
import mountain_car_objectives as mco
import mountain_car_optimizers as mc_opt
import mountain_car_simulation_analysis as mcsa


# =====================================================================
# Main training and evaluation program
# =====================================================================

if __name__ == "__main__":

    gt_reach_fname = Path(__file__).with_name("mc_reach_regions.pkl")

    # Fixed abstraction and environment settings
    abstraction_shape = [70, 70]
    domain_lb = np.array([-1.2, -0.07])
    domain_ub = np.array([0.6, 0.07])

    # Define the initial state subset domain
    init_domain_lb = np.array([-1.2, -0.07])
    init_domain_ub = np.array([0.6, 0.07])

    # Initialize abstraction parameters
    key = jax.random.PRNGKey(0)
    sigma_u = 0.1
    # u1 = jnp.zeros((abstraction_shape[0],))  # initial uniform spacing
    # u2 = jnp.zeros((abstraction_shape[1],))
    key, k_u1, k_u2 = jax.random.split(key, 3)
    u1 = sigma_u * jax.random.normal(k_u1, (abstraction_shape[0],))
    u2 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[1],))
    params = jnp.concatenate([u1, u2])

    args = {}
    args['shape'] = abstraction_shape
    args['domain_lb'] = domain_lb
    args['domain_ub'] = domain_ub
    args['horizon'] = 5
    args['temp_in'] = 0.01
    args['temp_out'] = 0.03
    args['norm_order'] = 2.0
    args['propagation'] = 'interval'
    args['inflation_coefs'] = np.zeros(2)
    args['snap_temperatures'] = (
        (domain_ub - domain_lb) / (2.0 * np.asarray(abstraction_shape))
    )

    # Evaluate initial abstraction
    recall, kripke_components = vt.build_and_verify_from_params(params,
                                                                abstraction_shape,
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
        abstraction_shape,
        domain_lb,
        domain_ub,
        horizon=args['horizon'],
        num_samples=64,
        batch_size=256,
        refine=False,
        verbose=False,
    )
    print(f"    > Epsilon = {result.epsilon}")
    print(f"    > Mean epsilon = {result.epsilon_mean}")
    print(f"    > Min epsilon = {result.epsilon_min}")
    print(f"    > Q1 epsilon = {result.epsilon_q1}")
    print(f"    > Median epsilon = {result.epsilon_median}")
    print(f"    > Q3 epsilon = {result.epsilon_q3}")

    # J_bruteforce = mco.upward_proxy_bruteforce(params, args=args)
    # print(f"Brute-force objective value: {J_bruteforce:.4f}")
    val, grad = jax.value_and_grad(mco.upward_proxy)(
        params,
        args=args)
    print(f"JAX objective value: {val:.4f}")
    print(f"Initial objective grad norm: {jnp.linalg.norm(grad):.4f}")

    # Employ gradient descent to optimize the grid
    params_opt, cost_history, grad_norm_history = mc_opt.gradient_descent(
        params,
        mco.upward_proxy,
        args=args,
        steps=500,
        lr=0.02,
        grad_clip=1.0,
        print_every=10,
        record_every=10,
        return_best=True)

    # Evaluate final abstraction
    recall, kripke_components = vt.build_and_verify_from_params(params_opt,
                                                                abstraction_shape,
                                                                domain_lb,
                                                                domain_ub,
                                                                init_domain_lb,
                                                                init_domain_ub,
                                                                gt_reach_fname=gt_reach_fname,
                                                                verbose=True,
                                                                log_time=True)
    print(f"Recall = {recall}")
    result = mcsa.evaluate_simulation_metric(
        params_opt,
        kripke_components,
        abstraction_shape,
        domain_lb,
        domain_ub,
        horizon=args['horizon'],
        num_samples=64,
        batch_size=256,
        refine=False,
        verbose=False,
    )
    print(f"    > Epsilon = {result.epsilon}")
    print(f"    > Mean epsilon = {result.epsilon_mean}")
    print(f"    > Min epsilon = {result.epsilon_min}")
    print(f"    > Q1 epsilon = {result.epsilon_q1}")
    print(f"    > Median epsilon = {result.epsilon_median}")
    print(f"    > Q3 epsilon = {result.epsilon_q3}")


    x_edges, y_edges = mco.extract_grid_params(
        params_opt,
        abstraction_shape,
        domain_lb,
        domain_ub,
    )
    x_gaps = np.diff(x_edges)
    y_gaps = np.diff(y_edges)

    fig, ax = plt.subplots(figsize=(8, 6))
    for x_edge in x_edges:
        ax.axvline(x_edge, color="tab:blue", linewidth=0.6, alpha=0.5)
    for y_edge in y_edges:
        ax.axhline(y_edge, color="tab:orange", linewidth=0.6, alpha=0.5)

    ax.set_xlim(domain_lb[0], domain_ub[0])
    ax.set_ylim(domain_lb[1], domain_ub[1])
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(
        "Final optimized abstraction grid\n"
        f"x-gap range: [{x_gaps.min():.4f}, {x_gaps.max():.4f}], "
        f"y-gap range: [{y_gaps.min():.4f}, {y_gaps.max():.4f}]"
    )
    ax.set_aspect("auto")
    ax.grid(False)
    fig.tight_layout()
    plt.show()

    
