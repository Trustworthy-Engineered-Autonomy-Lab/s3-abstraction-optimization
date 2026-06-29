# =====================================================================
# Description: main script for training abstraction parameters
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import unicycle_abstraction as ua
import simulation_analysis as sa
import unicycle_objectives as uo
import verification_tools as vt
import unicycle_optimizers as u_opt
import unicycle_system_jax as usj
import jax
import jax.numpy as jnp
import numpy as np
import pyModelChecking as pmc
import pickle as pkl
import matplotlib.pyplot as plt


# =====================================================================
# Main training and evaluation program
# =====================================================================

if __name__ == "__main__":

    # Fixed abstraction and environment settings
    abstraction_shape = [100, 100, 100]
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])

    # Define the initial state subset domain
    init_domain_lb = np.array([0.0, 0.0, -np.pi/4])
    init_domain_ub = np.array([50.0, 50.0, np.pi/4])

    gt_reach_fname = "unicycle-taylor/unicycle_gt_reach_regions_100.pkl"

    # Initialize abstraction parameters
    key = jax.random.PRNGKey(0)
    sigma_u = 1.0
    # u1 = jnp.zeros((abstraction_shape[0],))  # initial uniform spacing
    # u2 = jnp.zeros((abstraction_shape[1],))
    # u3 = jnp.zeros((abstraction_shape[2],))
    key, k_u1, k_u2 = jax.random.split(key, 3)
    u1 = sigma_u * jax.random.normal(k_u1, (abstraction_shape[0],))
    u2 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[1],))
    u3 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[2],))
    params = jnp.concatenate([u1, u2, u3])

    # L = usj.estimate_lipschitz_array(domain_lb, domain_ub)
    # J = uo.succ_bound(params,
    #               shape=abstraction_shape,
    #               domain_lb=domain_lb,
    #               domain_ub=domain_ub,
    #               L=L,
    #               p=20.0)
    # print(J)

    # # Verify the initial system
    # recall = vt.build_and_verify_from_params(params,
    #                                            abstraction_shape,
    #                                            domain_lb,
    #                                            domain_ub,
    #                                            init_domain_lb,
    #                                            init_domain_ub,
    #                                            gt_reach_fname=gt_reach_fname,
    #                                            verbose=True,
    #                                            log_time=True)
    # print(recall)

    # Compute initial objective and gradient
    args = {}
    args['shape'] = abstraction_shape
    args['domain_lb'] = domain_lb
    args['domain_ub'] = domain_ub
    # args['L'] = L
    # val, grad = jax.value_and_grad(uo.succ_bound)(
    #     params,
    #     args=args)
    # print(f"Initial objective value: {val:.4f}")
    # print(f"Initial objective grad norm: {jnp.linalg.norm(grad):.4f}")

    # Employ gradient descent to optimize the grid
    params_opt, cost_history, grad_norm_history = u_opt.gradient_descent(
        params,
        uo.image_volume_over_parent,
        args=args,
        steps=300,
        lr=1e-1,
        grad_clip=1e3,
        print_every=10,
        record_every=50)
    
    # Verify the final system
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

























    
    # recalls = []
    # costs = []

    # for key_int in range(20):

    #     # Initialize abstraction parameters
    #     key = jax.random.PRNGKey(key_int)
    #     sigma_u = 1.0
    #     key, k_u1, k_u2 = jax.random.split(key, 3)
    #     u1 = sigma_u * jax.random.normal(k_u1, (abstraction_shape[0],))
    #     u2 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[1],))
    #     u3 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[2],))
    #     params = jnp.concatenate([u1, u2, u3])

    #     # Verify the initial system
    #     gt_reach_fname = "unicycle-taylor/unicycle_gt_reach_regions_100.pkl"
    #     recall = vt.build_and_verify_from_params(params,
    #                                             abstraction_shape,
    #                                             domain_lb,
    #                                             domain_ub,
    #                                             init_domain_lb,
    #                                             init_domain_ub,
    #                                             gt_reach_fname=gt_reach_fname,
    #                                             verbose=False)
    #     recalls.append(recall)

    #     # Compute initial objective and gradient
    #     val, grad = jax.value_and_grad(uo.image_volume)(
    #         params,
    #         shape=abstraction_shape,
    #         domain_lb=domain_lb,
    #         domain_ub=domain_ub)
    #     costs.append(val)

    #     print(f"Cost: {val:.4f}")
    #     print(f"Recall: {recall}")

    # uni_corr_data = {}
    # uni_corr_data['recalls'] = recalls
    # uni_corr_data['costs'] = costs
    # with open("uni_corr_data.pkl", "wb") as f:
    #     pkl.dump(uni_corr_data, f)

    # with open("uni_corr_data.pkl", "rb") as f:
    #     data = pkl.load(f)

    # sat_props = np.array(data['sat_props'])
    # costs = np.array(data['costs'])

    # print(np.corrcoef(sat_props, costs))

    # plt.scatter(sat_props, costs)
    # plt.show()


    # # Extract the initial grid parameters (edges)
    # x_edges, y_edges, theta_edges = uo.extract_grid_params(params, abstraction_shape, domain_lb, domain_ub)
    # edges = [x_edges, y_edges, theta_edges]

    # # Build the initial Kripke components
    # kripke_components = ua.build_abstraction(x_edges, y_edges, theta_edges, verbose=True)

    # # Evaluate the upward simulation metric
    # upward_delta = sa.approx_upward_metric(kripke_components,
    #                                        abstraction_shape,
    #                                        edges,
    #                                        delta_iterations=50,
    #                                        num_samples=50,
    #                                        tol=1e-1,
    #                                        verbose=True)

    # # Build the full kripke structure
    # kripke_structure = pmc.Kripke(S=kripke_components['kripke_states'],
    #                               S0=init_states,
    #                               R=list(kripke_components['kripke_transitions']),
    #                               L=kripke_components['kripke_labels'])
    
    # # Run verification
    # sat_init_states, sat_prop = vt.model_check_kripke(kripke_structure)
    # print(sat_prop)
