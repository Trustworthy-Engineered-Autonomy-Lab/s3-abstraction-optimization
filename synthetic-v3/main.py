# =====================================================================
# Description: contains the necessary tools for modeling the unicycle
# system as a finite transition system with robust Taylor reachability.
# Utilizes PyModelChecking for abstraction as a Kripke structure.
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import numpy as np
import synthetic_abstraction as sa
import synthetic_optimizers as s_opt
import time
import jax
import jax.numpy as jnp
import verification_tools as vt
import synthetic_objectives as so


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    gt_reach_fname = "synthetic-v3/synthetic_reach_regions.pkl"

    # Fixed abstraction and environment settings
    abstraction_shape = [100, 100]
    domain_lb = np.array([-10.0, -10.0])
    domain_ub = np.array([10.0, 10.0])

    # Define the initial state subset domain
    init_domain_lb = np.array([-10.0, -10.0])
    init_domain_ub = np.array([10.0, 10.0])

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

    # J = so.image_area(params, args=args)
    # print(J)
    # val, grad = jax.value_and_grad(so.image_area)(
    #     params,
    #     args=args)
    # print(f"Initial objective value: {val:.4f}")
    # print(f"Initial objective grad norm: {jnp.linalg.norm(grad):.4f}")

    # recall = vt.build_and_verify_from_params(params,
    #                                         abstraction_shape,
    #                                         domain_lb,
    #                                         domain_ub,
    #                                         init_domain_lb,
    #                                         init_domain_ub,
    #                                         gt_reach_fname=gt_reach_fname,
    #                                         verbose=True,
    #                                         log_time=True)
    # print(recall)

    # Employ gradient descent to optimize the grid
    params_opt, cost_history, grad_norm_history = s_opt.gradient_descent(
        params,
        so.image_area,
        args=args,
        steps=3_000,
        lr=3e-2,
        grad_clip=1e3,
        print_every=500,
        record_every=100)

    recall = vt.build_and_verify_from_params(params_opt,
                                            abstraction_shape,
                                            domain_lb,
                                            domain_ub,
                                            init_domain_lb,
                                            init_domain_ub,
                                            gt_reach_fname=gt_reach_fname,
                                            verbose=True,
                                            log_time=True)
    print(recall)
