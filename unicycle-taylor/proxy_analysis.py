# =====================================================================
# Description: main script for training abstraction parameters
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import unicycle_abstraction as ua
import unicycle_simulation_analysis as usa
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
    abstraction_shape = [20, 20, 20]
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])

    # Define the initial state subset domain
    init_domain_lb = np.array([0.0, 0.0, -np.pi/4])
    init_domain_ub = np.array([50.0, 50.0, np.pi/4])

    gt_reach_fname = "unicycle-taylor/unicycle_gt_reach_regions_100.pkl"

    args = {}
    args['shape'] = abstraction_shape
    args['domain_lb'] = domain_lb
    args['domain_ub'] = domain_ub
    args['horizon'] = 3
    args['temp_in'] = 0.5
    args['temp_out'] = 0.5
    args['inflation_coefs'] = np.array([1.0, 1.0, 0.3])

    keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    proxies = []
    epsilons = []
    mean_epsilons = []
    for key_no in keys:

        # Initialize abstraction parameters
        key = jax.random.PRNGKey(key_no)
        sigma_u = 2.0
        key, k_u1, k_u2 = jax.random.split(key, 3)
        u1 = sigma_u * jax.random.normal(k_u1, (abstraction_shape[0],))
        u2 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[1],))
        u3 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[2],))
        params = jnp.concatenate([u1, u2, u3])

        # Evaluate the proxy
        J, _ = jax.value_and_grad(uo.upward_proxy)(
        params,
        args=args)
        print(f"    > Proxy: {J:.4f}")
        proxies.append(J)

        # Build the abstract model
        x_edges, y_edges, theta_edges = uo.extract_grid_params(params,
                                                           abstraction_shape,
                                                           domain_lb,
                                                           domain_ub)
        kripke_components = ua.build_abstraction(x_edges,
                                             y_edges,
                                             theta_edges,
                                             verbose=False)
        
        # Evaluate the real simulation metric
        result = usa.evaluate_simulation_metric(
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
        epsilons.append(result.epsilon)
        mean_epsilons.append(result.epsilon_mean)


    with open("unicycle-taylor/proxy_analysis_data.pkl", "wb") as f:
        pkl.dump(
            {
                "proxies": proxies,
                "epsilons": epsilons,
                "mean_epsilons": mean_epsilons,
            },
            f,
        )

    plt.plot(keys, proxies, label="proxy")
    plt.plot(keys, epsilons, label="epsilon")
    plt.plot(keys, mean_epsilons, label="mean_epsilon")
    plt.xlabel("random test")
    plt.ylabel("value")
    plt.legend()
    plt.show()

    plt.scatter(proxies, epsilons)
    plt.xlabel("proxy")
    plt.ylabel("epsilon")
    plt.show()

    plt.scatter(proxies, mean_epsilons)
    plt.xlabel("proxy")
    plt.ylabel("mean epsilon")
    plt.show()

