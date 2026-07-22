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


def mean_error_stat(x, y, axis=-1):
    return np.mean(x - y, axis=axis)


def mean_abs_error_stat(x, y, axis=-1):
    return np.mean(np.abs(x - y), axis=axis)


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


def bootstrap_errors(x, y):
    rng = np.random.default_rng(12345)

    # Build bootstrap intervals for mean error
    mean_error = mean_error_stat(x, y)
    res_error = stats.bootstrap(
        (x, y),
        mean_error_stat,
        paired=True,
        vectorized=True,
        method="BCa",
        confidence_level=0.95,
        n_resamples=20_000,
        rng=rng,
    )
    ci_error = np.array([
        res_error.confidence_interval.low,
        res_error.confidence_interval.high,
    ])
    diff_error = ci_error - mean_error

    # Build bootstrap intervals for mean absolute error
    mean_abs_error = mean_abs_error_stat(x, y)
    res_abs_error = stats.bootstrap(
        (x, y),
        mean_abs_error_stat,
        paired=True,
        vectorized=True,
        method="BCa",
        confidence_level=0.95,
        n_resamples=20_000,
        rng=rng,
    )
    ci_abs_error = np.array([
        res_abs_error.confidence_interval.low,
        res_abs_error.confidence_interval.high,
    ])
    diff_abs_error = ci_abs_error - mean_abs_error

    return (
        mean_error,
        diff_error,
        ci_error,
        mean_abs_error,
        diff_abs_error,
        ci_abs_error,
    )


def format_estimate(value, diff, *, unit=""):
    """Format an estimate followed by its lower and upper 95% CI offsets."""
    suffix = f" {unit}" if unit else ""
    return f"{value:.3f} {diff[1]:+.3f}/{diff[0]:+.3f}{suffix}"


def print_report(data):
    proxies = data["proxies"]
    max_epsilons = data["max_epsilons"]
    mean_epsilons = data["mean_epsilons"]
    transitions = data["transitions"]
    proxy_times = data["proxy_times"]
    sim_times = data["sim_times"]

    for i in range(proxies.shape[0]):
        correlations = (
            ("Proxy vs. max epsilon", max_epsilons[i]),
            ("Proxy vs. mean epsilon", mean_epsilons[i]),
            ("Proxy vs. transitions", transitions[i]),
        )
        errors = (
            ("Max epsilon", max_epsilons[i]),
            ("Mean epsilon", mean_epsilons[i]),
        )

        print(f"\nHorizon {i + 1}")
        print("  Correlations")
        print(
            f"    {'Comparison':<26} {'Pearson r (95%)':>29} "
            f"{'Spearman r (95%)':>31}"
        )
        print(f"    {'-' * 26} {'-' * 29} {'-' * 31}")
        for metric_name, values in correlations:
            pearson, pearson_diff, _, spearman, spearman_diff, _ = bootstrap_corr(
                proxies[i], values
            )
            print(
                f"    {metric_name:<26} "
                f"{format_estimate(pearson, pearson_diff):>29} "
                f"{format_estimate(spearman, spearman_diff):>31}"
            )

        print("\n  Errors (proxy - target)")
        print(
            f"    {'Target':<26} {'Mean error (95%)':>29} "
            f"{'Mean abs. error (95%)':>35}"
        )
        print(f"    {'-' * 26} {'-' * 29} {'-' * 35}")
        for target_name, target_values in errors:
            mean_error, error_diff, _, mean_abs_error, abs_error_diff, _ = bootstrap_errors(
                proxies[i], target_values
            )
            print(
                f"    {target_name:<26} "
                f"{format_estimate(mean_error, error_diff):>29} "
                f"{format_estimate(mean_abs_error, abs_error_diff):>35}"
            )

        proxy_time, proxy_time_diff, _ = bootstrap_mn(proxy_times[i])
        sim_time, sim_time_diff, _ = bootstrap_mn(sim_times[i])
        print("\n  Latency")
        print(f"    {'Process':<26} {'Mean time (95%)':>29}")
        print(f"    {'-' * 26} {'-' * 29}")
        print(
            f"    {'Proxy evaluation':<26} "
            f"{format_estimate(proxy_time, proxy_time_diff, unit='s'):>29}"
        )
        print(
            f"    {'Simulation evaluation':<26} "
            f"{format_estimate(sim_time, sim_time_diff, unit='s'):>29}"
        )


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

    # collect_data(fname)

    with open(fname, "rb") as f:
        data = pkl.load(f)

    print_report(data)
