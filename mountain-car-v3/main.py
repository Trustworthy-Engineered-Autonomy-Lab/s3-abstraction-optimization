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
import mountain_car_abstraction as mca
import verification_tools as vt
import mountain_car_objectives as mco
import mountain_car_optimizers as mc_opt
import mountain_car_simulation_analysis as mcsa


# =====================================================================
# Main training and evaluation program
# =====================================================================

if __name__ == "__main__":

    gt_reach_fname = "mountain-car-v3/mc_reach_regions.pkl"

    # Fixed abstraction and environment settings
    abstraction_shape = [200, 200]
    domain_lb = np.array([-1.2, -0.07])
    domain_ub = np.array([0.6, 0.07])

    # Define the initial state subset domain
    init_domain_lb = np.array([-1.2, -0.07])
    init_domain_ub = np.array([0.6, 0.07])

    # Initialize abstraction parameters
    key = jax.random.PRNGKey(0)
    sigma_u = 2.0
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
    args['horizon'] = 3
    args['temp_in'] = 0.01
    args['temp_out'] = 0.01
    args['inflation_coefs'] = [0.005, 0.0004]

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
        steps=101,
        lr=100,
        grad_clip=1e3,
        print_every=1,
        record_every=100)

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
    print(f"    > Median epsilon = {result.epsilon_median}")
    print(f"    > Q3 epsilon = {result.epsilon_q3}")
