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
import pandas as pd
import time


# =====================================================================
# Main training and evaluation program
# =====================================================================
def proxy_by_horizon(h, rows):
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
    args['horizon'] = h
    args['inflation_coefs'] = np.array([0.1, 0.1])
    args['temp_in'] = .5
    args['temp_out'] = .5
    print(f"horizon {args['horizon']}: temp_in = {args['temp_in']}, temp_out = {args['temp_out']}, inflation_coefs = {args['inflation_coefs']}\n")

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
        proxy_start = time.process_time()
        J, _ = jax.value_and_grad(so.upward_proxy)(
        params,
        args=args)
        proxy_time = time.process_time() - proxy_start
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
        simulate_start = time.process_time()
        result = ssa.evaluate_simulation_metric(params,
                                                kripke_components,
                                                abstraction_shape,
                                                domain_lb,
                                                domain_ub,
                                                horizon=args['horizon'])
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

    pearson_corr, p_value_p = stats.pearsonr(epsilons, proxies)
    print(f"Pearson r: {pearson_corr:.3f}, p-value: {p_value_p:.3f}")

    spearman_corr, p_value_s = stats.spearmanr(epsilons, proxies)
    print(f"Spearman r: {spearman_corr:.3f}, p-value: {p_value_s:.3f}")

    pearson_corr_mean, p_value_p = stats.pearsonr(mean_epsilons, proxies)
    print(f"Pearson r (mean): {pearson_corr_mean:.3f}, p-value: {p_value_p:.3f}")

    spearman_corr_mean, p_value_s = stats.spearmanr(mean_epsilons, proxies)
    print(f"Spearman r (mean): {spearman_corr_mean:.3f}, p-value: {p_value_s:.3f}")

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

    return rows, pearson_corr, spearman_corr, pearson_corr_mean, spearman_corr_mean


if __name__ == "__main__":
    rows = []
    p1,p2,s1,s2 = 0,0,0,0
    pearsons, spearmans, pearson_means, spearman_means = [],[],[],[]
    for horizon in [1,2,3,4,5,6,7,8,9,10]:
        print(f"Running proxy analysis for horizon {horizon}")
        rows, p1, s1, p2, s2 = proxy_by_horizon(horizon, rows)
        pearsons.append(p1)
        spearmans.append(s1)
        pearson_means.append(p2)
        spearman_means.append(s2)
    print(f"Pearson correlations: {pearsons}")
    print(f"Spearman correlations: {spearmans}")
    print(f"Pearson mean correlations: {pearson_means}")
    print(f"Spearman mean correlations: {spearman_means}")

    df = pd.DataFrame(rows)
    csv_path = f"synthetic-v3/proxy_analysis_results_1-10.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved proxy analysis results to {csv_path}")

