# =====================================================================
# Description: user-friendy wrapper to run S3-based optimization 
# of unicycle quantization parameters.
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import unicycle_simulation_analysis as usa
import unicycle_objectives as uo
import verification_tools as vt
import unicycle_optimizers as u_opt
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import argparse

# =====================================================================
# User-defined settings
# =====================================================================

SHAPE = 50
HORIZON = 3
TEMP_1 = 0.1
TEMP_2 = 0.1
SIGMA = 0.1
EVAL_INIT = True
LR = 1e-3
STEPS = 100

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run S3-based optimization of the unicycle abstraction."
    )
    parser.add_argument("--shape", type=int, default=SHAPE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--temp-1", type=float, default=TEMP_1)
    parser.add_argument("--temp-2", type=float, default=TEMP_2)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument(
        "--eval-init",
        action=argparse.BooleanOptionalAction,
        default=EVAL_INIT,
        help="Evaluate the initial abstraction before optimization.",
    )
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--steps", type=int, default=STEPS)
    return parser.parse_args()


# =====================================================================
# Main 
# =====================================================================

if __name__ == "__main__":

    cli_args = parse_args()

    gt_reach_fname = Path(__file__).parent / 'artifacts' / 'cache' / 'reach.pkl'

    # Abstraction settings
    abstraction_shape = [cli_args.shape, cli_args.shape, cli_args.shape]
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])
    init_domain_lb = np.array([0.0, 0.0, -np.pi/4])
    init_domain_ub = np.array([50.0, 50.0, np.pi/4])

    # S3 hyperparameters
    args = {}
    args['shape'] = abstraction_shape
    args['domain_lb'] = domain_lb
    args['domain_ub'] = domain_ub
    args['horizon'] = cli_args.horizon
    args['temp_in'] = cli_args.temp_1
    args['temp_out'] = cli_args.temp_2
    args['inflation_coefs'] = np.array([
        50.0/cli_args.shape/2, 50.0/cli_args.shape/2, 2*np.pi/cli_args.shape/2
    ])

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    key, k_u1, k_u2 = jax.random.split(key, 3)
    u1 = cli_args.sigma * jax.random.normal(k_u1, (abstraction_shape[0],))
    u2 = cli_args.sigma * jax.random.normal(k_u2, (abstraction_shape[1],))
    u3 = cli_args.sigma * jax.random.normal(k_u2, (abstraction_shape[2],))
    params = jnp.concatenate([u1, u2, u3])

    # Evaluate the initial system
    if cli_args.eval_init:
        print("\n=== Building Initial Model ===")
        recall, kripke_components = vt.build_and_verify_from_params(params,
                                                abstraction_shape,
                                                domain_lb,
                                                domain_ub,
                                                init_domain_lb,
                                                init_domain_ub,
                                                gt_reach_fname=gt_reach_fname,
                                                verbose=True,
                                                log_time=False)
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
        print("Initial Model Performance")
        print(f"  Recall:       {recall:.4f}")
        print(f"  Sim. metric:  {result.epsilon:.4f}")
        print(f"  Mean delta:   {result.epsilon_mean:.4f}")
        print(f"  Q1 delta:     {result.epsilon_q1:.4f}")
        print(f"  Median delta: {result.epsilon_median:.4f}")
        print(f"  Q3 delta:     {result.epsilon_q3:.4f}")


    # Employ gradient descent to optimize the grid
    print("\n=== Optimizing Initial Quantization ===")
    params_opt = u_opt.gradient_descent(
        params,
        uo.upward_proxy,
        args=args,
        steps=cli_args.steps,
        lr=cli_args.lr,
        grad_clip=1.0,
        print_every=10,
        return_best=True
    )
    
    # Evaluate the final system
    print("\n=== Building Final Model ===")
    recall, kripke_components = vt.build_and_verify_from_params(
        params_opt,
        abstraction_shape,
        domain_lb,
        domain_ub,
        init_domain_lb,
        init_domain_ub,
        gt_reach_fname=gt_reach_fname,
        verbose=True,
        log_time=False
    )
    result = usa.evaluate_simulation_metric(
        params_opt,
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
    print("Final Model Performance")
    print(f"  Recall:       {recall:.4f}")
    print(f"  Sim. metric:  {result.epsilon:.4f}")
    print(f"  Mean delta:   {result.epsilon_mean:.4f}")
    print(f"  Q1 delta:     {result.epsilon_q1:.4f}")
    print(f"  Median delta: {result.epsilon_median:.4f}")
    print(f"  Q3 delta:     {result.epsilon_q3:.4f}")
