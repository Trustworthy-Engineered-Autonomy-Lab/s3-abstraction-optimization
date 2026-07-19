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
import pandas as pd
import time
from scipy import stats

def proxy_by_horizon(h, rows):
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
    args['horizon'] = h
    args['temp_in'] = .5
    args['temp_out'] = 0.6
    args['inflation_coefs'] = np.array([1.25, 1.25, np.pi/20])

    print(f"temp_in: {args['temp_in']}, temp_out: {args['temp_out']}, inflation_coefs: {args['inflation_coefs']}")

    keys = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
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
        proxy_start = time.process_time()
        J, _ = jax.value_and_grad(uo.upward_proxy)(
        params,
        args=args)
        proxy_time = time.process_time() - proxy_start
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
        simulate_start = time.process_time()
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
        simulate_time = time.process_time() - simulate_start
        print(f"    > Epsilon = {result.epsilon}")
        print(f"    > Mean epsilon = {result.epsilon_mean}")
        epsilons.append(result.epsilon)
        mean_epsilons.append(result.epsilon_mean)

        rows.append(
            {
                "horizon": h,
                "key": key_no,
                "proxy": float(J),
                "epsilon": float(result.epsilon),
                "mean_epsilon": float(result.epsilon_mean),
                "min_epsilon": float(result.epsilon_min),
                "median_epsilon": float(result.epsilon_median),
                "q1_epsilon": float(result.epsilon_q1),
                "q3_epsilon": float(result.epsilon_q3),
                "proxy_time": proxy_time,
                "simulate_time": simulate_time,
            }
        )


    with open("unicycle-taylor/proxy_analysis_data.pkl", "wb") as f:
        pkl.dump(
            {
                "proxies": proxies,
                "epsilons": epsilons,
                "mean_epsilons": mean_epsilons,
            },
            f,
        )

    # with open("unicycle-taylor/proxy_analysis_data.pkl", "rb") as f:
    #     data = pkl.load(f)
    
    # proxies = np.array(data['proxies'])
    # epsilons = np.array(data['epsilons'])
    # mean_epsilons = np.array(data['mean_epsilons'])

    pearson_corr, p_value_p = stats.pearsonr(mean_epsilons, proxies)
    print(f"Pearson r: {pearson_corr:.3f}, p-value: {p_value_p:.3f}")

    spearman_corr, p_value_s = stats.spearmanr(mean_epsilons, proxies)
    print(f"Spearman r: {spearman_corr:.3f}, p-value: {p_value_s:.3f}")

    # plt.plot(keys, proxies, label="proxy")
    # plt.plot(keys, epsilons, label="epsilon")
    # plt.plot(keys, mean_epsilons, label="mean_epsilon")
    # plt.xlabel("random test")
    # plt.ylabel("value")
    # plt.legend()
    # plt.show()

    # plt.scatter(proxies, epsilons)
    # plt.xlabel("proxy")
    # plt.ylabel("epsilon")
    # plt.show()

    # plt.scatter(proxies, mean_epsilons)
    # plt.xlabel("proxy")
    # plt.ylabel("mean epsilon")
    # plt.show()



# =====================================================================
# Main training and evaluation program
# =====================================================================

if __name__ == "__main__":
    rows = []
    for horizon in [1]:
        print(f"Running proxy analysis for horizon {horizon}")
        proxy_by_horizon(horizon, rows)

    df = pd.DataFrame(rows)
    csv_path = f"unicycle-taylor/proxy_analysis_results_draft(2).csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved proxy analysis results to {csv_path}")

   