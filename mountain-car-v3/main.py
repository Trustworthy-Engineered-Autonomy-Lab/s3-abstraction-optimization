# =====================================================================
# Description: main script for training abstraction parameters
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
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


# =====================================================================
# Main training and evaluation program
# =====================================================================

if __name__ == "__main__":

    gt_reach_fname = "mountain-car-v3/mc_reach_regions.pkl"

    # Fixed abstraction and environment settings
    abstraction_shape = [100, 100]
    domain_lb = np.array([-1.2, -0.07])
    domain_ub = np.array([0.6, 0.07])

    # Define the initial state subset domain
    init_domain_lb = np.array([-1.2, -0.07])
    init_domain_ub = np.array([0.6, 0.07])

    # Initialize abstraction parameters
    key = jax.random.PRNGKey(0)
    sigma_u = 1.0
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
    args['temp'] = 0.5
    args['inflation_coefs'] = [0.005, 0.0005]

    # recall, kripke_components = vt.build_and_verify_from_params(params,
    #                                                             abstraction_shape,
    #                                                             domain_lb,
    #                                                             domain_ub,
    #                                                             init_domain_lb,
    #                                                             init_domain_ub,
    #                                                             gt_reach_fname=gt_reach_fname,
    #                                                             verbose=True,
    #                                                             log_time=True)
    # print(f"    > Recall = {recall}")

    val, grad = jax.value_and_grad(mco.epsilon_H_bound)(
        params,
        args=args)
    print(f"Initial objective value: {val:.4f}")
    print(f"Initial objective grad norm: {jnp.linalg.norm(grad):.4f}")

    # # Employ gradient descent to optimize the grid
    # params_opt, cost_history, grad_norm_history = mc_opt.gradient_descent(
    #     params,
    #     mco.image_area,
    #     args=args,
    #     steps=20,
    #     lr=10,
    #     grad_clip=1e3,
    #     print_every=1,
    #     record_every=100)

    # # Evaluate final abstraction
    # recall, kripke_components = vt.build_and_verify_from_params(params_opt,
    #                                                             abstraction_shape,
    #                                                             domain_lb,
    #                                                             domain_ub,
    #                                                             init_domain_lb,
    #                                                             init_domain_ub,
    #                                                             gt_reach_fname=gt_reach_fname,
    #                                                             verbose=True,
    #                                                             log_time=True)
    # print(f"Recall = {recall}")
