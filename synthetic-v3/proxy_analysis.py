# =====================================================================
# Description: main script for training abstraction parameters
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import synthetic_abstraction as sa
import synthetic_simulation_analysis as ssa
import synthetic_objectives as so
import verification_tools as vt
import synthetic_optimizers as s_opt
import synthetic_system as ss
import jax
import jax.numpy as jnp
import numpy as np
import pyModelChecking as pmc
import pickle as pkl
import matplotlib.pyplot as plt
from scipy import stats


# =====================================================================
# Main training and evaluation program
# =====================================================================

if __name__ == "__main__":

    # Fixed abstraction and environment settings
    abstraction_shape = [50, 50]
    domain_lb = np.array([-10.0, -10.0])
    domain_ub = np.array([10.0, 10.0])

    # Define the initial state subset domain
    init_domain_lb = np.array([-10.0, -10.0])
    init_domain_ub = np.array([10.0, 10.0])

    gt_reach_fname = "synthetic-v3/synthetic_reach_regions.pkl"

    args = {}
    args['shape'] = abstraction_shape
    args['domain_lb'] = domain_lb
    args['domain_ub'] = domain_ub
    args['horizon'] = 3
    args['inflation_coefs'] = np.array([0.05, 0.05])
    args['temp_in'] = 1.0
    args['temp_out'] = 0.3

    keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
        params = jnp.concatenate([u1, u2])

        # Evaluate the proxy
        J, _ = jax.value_and_grad(so.upward_proxy)(
        params,
        args=args)
        print(f"    > Proxy: {J:.4f}")
        proxies.append(J)

        # Build the abstract model
        x_edges, y_edges = so.extract_grid_params(params,
                                                abstraction_shape,
                                                domain_lb,
                                                domain_ub)
        kripke_components = sa.build_abstraction(x_edges,
                                             y_edges,
                                             verbose=False)
        
        # Evaluate the real simulation metric
        result = ssa.evaluate_simulation_metric(params,
                                                kripke_components,
                                                abstraction_shape,
                                                domain_lb,
                                                domain_ub,
                                                horizon=args['horizon'])
        print(f"    > Epsilon = {result.epsilon}")
        print(f"    > Mean epsilon = {result.epsilon_mean}")
        epsilons.append(result.epsilon)
        mean_epsilons.append(result.epsilon_mean)


    # with open("synthetic-v3/proxy_analysis_data.pkl", "wb") as f:
    #     pkl.dump(
    #         {
    #             "proxies": proxies,
    #             "epsilons": epsilons,
    #             "mean_epsilons": mean_epsilons,
    #         },
    #         f,
    #     )

    # with open("unicycle-taylor/proxy_analysis_data.pkl", "rb") as f:
    #     data = pkl.load(f)
    
    # proxies = np.array(data['proxies'])
    # epsilons = np.array(data['epsilons'])
    # mean_epsilons = np.array(data['mean_epsilons'])

    pearson_corr, p_value_p = stats.pearsonr(epsilons, proxies)
    print(f"Pearson r: {pearson_corr:.3f}, p-value: {p_value_p:.3f}")

    spearman_corr, p_value_s = stats.spearmanr(epsilons, proxies)
    print(f"Spearman r: {spearman_corr:.3f}, p-value: {p_value_s:.3f}")

    pearson_corr, p_value_p = stats.pearsonr(mean_epsilons, proxies)
    print(f"Pearson r: {pearson_corr:.3f}, p-value: {p_value_p:.3f}")

    spearman_corr, p_value_s = stats.spearmanr(mean_epsilons, proxies)
    print(f"Spearman r: {spearman_corr:.3f}, p-value: {p_value_s:.3f}")

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

