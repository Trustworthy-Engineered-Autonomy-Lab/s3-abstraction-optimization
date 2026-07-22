# =====================================================================
# Description: main script for training abstraction parameters
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import mountain_car_abstraction as mca
import mountain_car_simulation_analysis as mcsa
import mountain_car_objectives as mco
import verification_tools as vt
import mountain_car_optimizers as mc_opt
import mountain_car_system as mcs
import jax
import jax.numpy as jnp
import numpy as np
import pyModelChecking as pmc
import pickle as pkl
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import time


# =====================================================================
# Data collection helpers
# =====================================================================

def make_eval_objective(args):
    @jax.jit
    def eval_objective(params):
        value, _ = jax.value_and_grad(mco.upward_proxy)(params, args=args)
        return value

    return eval_objective

def collect_data(fname):

    # Fixed abstraction and environment settings
    abstraction_shape = [50, 50]
    domain_lb = np.array([-1.2, -0.07])
    domain_ub = np.array([0.6, 0.07])

    # Hyperparameters of the proxy
    args = {}
    args['shape'] = abstraction_shape
    args['domain_lb'] = domain_lb
    args['domain_ub'] = domain_ub
    args['temp_in'] = 0.01
    args['temp_out'] = 0.03
    args['norm_order'] = 2.0
    args['propagation'] = 'interval'
    args['inflation_coefs'] = np.zeros(2)
    args['snap_temperatures'] = ((domain_ub - domain_lb) / (2.0 * np.asarray(abstraction_shape)))

    horizons = [1, 2, 3, 4, 5]

    # Instantiate random parameters
    num_samples = 100
    key = jax.random.PRNGKey(0)
    n_cols = abstraction_shape[0] + abstraction_shape[1]
    param_array = 3.0 * jax.random.normal(key, shape=(num_samples, n_cols))
    objective_evaluators = {
        horizon: make_eval_objective({**args, 'horizon': horizon})
        for horizon in horizons
    }

    proxies = np.zeros((len(horizons), num_samples), dtype=float)
    max_epsilons = np.zeros((len(horizons), num_samples), dtype=float)
    mean_epsilons = np.zeros((len(horizons), num_samples), dtype=float)
    transitions = np.zeros((len(horizons), num_samples), dtype=float)
    proxy_times = np.zeros((len(horizons), num_samples), dtype=float)
    sim_times = np.zeros((len(horizons), num_samples), dtype=float)

    for i in range(num_samples):

        if i % 10 == 0:
            print(f"Evaluating metrics at sample {i}...")

        params = param_array[i, :]

        # Build the abstract model
        x_edges, y_edges = mco.extract_grid_params(params,
                                                   abstraction_shape,
                                                   domain_lb,
                                                   domain_ub)
        build_start_time = time.process_time()
        kripke_components = mca.build_abstraction(x_edges,
                                                y_edges,
                                                verbose=False)
        build_time = time.process_time() - build_start_time
        
        # Model metrics
        avg_transitions = len(kripke_components['kripke_transitions'])/len(kripke_components['kripke_states'])
        
        for j in range(len(horizons)):

            args['horizon'] = horizons[j]

            # Evaluate the proxy
            proxy_eval_start_time = time.process_time()
            J = objective_evaluators[args['horizon']](params)
            proxy_eval_time = time.process_time() - proxy_eval_start_time

            # Evaluate the simulation metric
            sim_eval_start_time = time.process_time()
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
            sim_eval_time = time.process_time() - sim_eval_start_time + build_time

            # Append data
            proxies[j, i] = J
            max_epsilons[j, i] = result.epsilon
            mean_epsilons[j, i] = result.epsilon_mean
            transitions[j, i] = avg_transitions
            proxy_times[j, i] = proxy_eval_time
            sim_times[j, i] = sim_eval_time

    with (fname).open("wb") as f:
        pkl.dump(
            {
                "proxies": proxies,
                "max_epsilons": max_epsilons,
                "mean_epsilons": mean_epsilons,
                "transitions": transitions,
                "proxy_times": proxy_times,
                "sim_times": sim_times
            },
            f,
        )


# =====================================================================
# Correlation helpers
# =====================================================================

def pearson_stat(x, y, axis=-1):
    return stats.pearsonr(x, y, axis=axis).statistic

def spearman_stat(x, y, axis=-1):
    # Spearman correlation is Pearson correlation applied to ranks.
    x_rank = stats.rankdata(x, axis=axis)
    y_rank = stats.rankdata(y, axis=axis)
    return stats.pearsonr(x_rank, y_rank, axis=axis).statistic

def mean_stat(x, axis=-1):
    return np.mean(x, axis=axis)

def bootstrap_mn(x):

    rng = np.random.default_rng(12345)

    mean_x = mean_stat(x)

    res = stats.bootstrap(
    (x,),
    mean_stat,
    paired=True,
    vectorized=True,
    method="BCa",
    confidence_level=0.95,
    n_resamples=20_000,
    rng=rng,)

    ci = np.array([
            res.confidence_interval.low,
            res.confidence_interval.high
    ])

    diff = ci - mean_x
    
    return mean_x, diff, ci

def bootstrap_corr(x, y):

    rng = np.random.default_rng(12345)

    pearson_r = stats.pearsonr(x, y).statistic
    spearman_r = stats.spearmanr(x, y).statistic

    pearson_res = stats.bootstrap(
        (x, y),
        pearson_stat,
        paired=True,
        vectorized=True,
        method="BCa",
        confidence_level=0.95,
        n_resamples=20_000,
        rng=rng,)

    spearman_res = stats.bootstrap(
        (x, y),
        spearman_stat,
        paired=True,
        vectorized=True,
        method="BCa",
        confidence_level=0.95,
        n_resamples=20_000,
        rng=rng,)
    
    pearson_ci = np.array([
                pearson_res.confidence_interval.low,
                pearson_res.confidence_interval.high
    ])
    spearman_ci = np.array([
                spearman_res.confidence_interval.low,
                spearman_res.confidence_interval.high
    ])
    
    pearson_diff = pearson_ci - pearson_r
    spearman_diff = spearman_ci - spearman_r

    return pearson_r, pearson_diff, pearson_ci, spearman_r, spearman_diff, spearman_ci


def mixture_ci(x, y):

    rng = np.random.default_rng(12345)

    pearson_r = stats.pearsonr(x, y).statistic
    spearman_r = stats.spearmanr(x, y).statistic

    pearson_res = stats.pearsonr(x, y).confidence_interval(0.95)
    spearman_res = stats.bootstrap(
        (x, y),
        spearman_stat,
        paired=True,
        vectorized=True,
        method="BCa",
        confidence_level=0.95,
        n_resamples=20_000,
        rng=rng,)
    
    pearson_ci = np.array([
                pearson_res.low,
                pearson_res.high
    ])
    spearman_ci = np.array([
                spearman_res.confidence_interval.low,
                spearman_res.confidence_interval.high
    ])
    
    pearson_diff = pearson_ci - pearson_r
    spearman_diff = spearman_ci - spearman_r

    return pearson_r, pearson_diff, pearson_ci, spearman_r, spearman_diff, spearman_ci

# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    case_study_dir = Path(__file__).resolve().parent

    fname = case_study_dir / "proxy_analysis_data.pkl"

    collect_data(fname)

    with open(fname, "rb") as f:
        data = pkl.load(f)
    
    proxies = data["proxies"]
    max_epsilons = data["max_epsilons"]
    mean_epsilons = data["mean_epsilons"]
    transitions = data["transitions"]
    proxy_times = data["proxy_times"]
    sim_times = data["sim_times"]

    num_rows = proxies.shape[0]

    for i in range(num_rows):
        correlations = (
            ("Max epsilon vs. proxy", max_epsilons[i, :]),
            ("Mean epsilon vs. proxy", mean_epsilons[i, :]),
            ("Transitions vs. proxy", transitions[i, :]),
        )

        print(f"\nHorizon {i + 1}")
        print(f"  {'Metric':<24} {'Pearson r (95% CI)':>24} {'Spearman r (95% CI)':>24}")
        print(f"  {'-' * 24} {'-' * 24} {'-' * 24}")
        for metric_name, values in correlations:
            pearson_corr, pearson_diff, _, spearman_corr, spearman_diff, _ = bootstrap_corr(
                values, proxies[i, :]
            )
            pearson_ci = f"{pearson_corr:.2f} {pearson_diff[1]:+.2f}/{pearson_diff[0]:+.2f}"
            spearman_ci = f"{spearman_corr:.2f} {spearman_diff[1]:+.2f}/{spearman_diff[0]:+.2f}"
            print(f"  {metric_name:<24} {pearson_ci:>24} {spearman_ci:>24}")

        proxy_time, proxy_diff, _ = bootstrap_mn(proxy_times[i, :])
        sim_time, sim_diff, _ = bootstrap_mn(sim_times[i, :])
        print(f"  Average proxy time (95% CI): {proxy_time:.2f} s {proxy_diff[1]:+.2f}/{proxy_diff[0]:+.2f}")
        print(f"  Average simulation time (95% CI): {sim_time:.2f} s {sim_diff[1]:+.2f}/{sim_diff[0]:+.2f}")



    # sim_time, diff, ci = bootstrap_mn(sim_times[0, :])
    # print(sim_time)
    # print(diff)

    # pr, pd, pci, sr, sd, sci = bootstrap_corr(proxies[0, :], mean_epsilons[0, :])
    # print(pr)
    # print(pd)
    # print(sr)
    # print(sd)

    # proxies = np.array(data['proxies'])
    # epsilons = np.array(data['epsilons'])
    # mean_epsilons = np.array(data['mean_epsilons'])

    # pearson_corr, p_value_p = stats.pearsonr(epsilons, proxies)
    # print(f"Pearson r: {pearson_corr:.3f}, p-value: {p_value_p:.3f}")

    # spearman_corr, p_value_s = stats.spearmanr(epsilons, proxies)
    # print(f"Spearman r: {spearman_corr:.3f}, p-value: {p_value_s:.3f}")

    # pearson_corr, p_value_p = stats.pearsonr(mean_epsilons, proxies)
    # print(f"Mean-epsilon Pearson r: {pearson_corr:.3f}, p-value: {p_value_p:.3f}")

    # spearman_corr, p_value_s = stats.spearmanr(mean_epsilons, proxies)
    # print(f"Mean-epsilon Spearman r: {spearman_corr:.3f}, p-value: {p_value_s:.3f}")

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

