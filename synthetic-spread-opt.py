# Libraries
import numpy as np
import grid_plot_tools as gpt
from spread_objective_tools import (
    trans_matrix,
    diff_objective,
    diff_successor_count_objective,
    extract_grid_params,
)
from model_checking_tools import make_kripke_from_params, model_check_kripke, fixed_check_ground_truth, false_negative_rate
import jax
import jax.numpy as jnp
from typing import Optional, Union
import time

CENTER = np.array([5.0, 5.0])


def gradient_descent_optimize(
    params_init: jnp.ndarray,
    objective_fn,
    objective_kwargs: dict,
    steps: int,
    *,
    lr: Union[float, jnp.ndarray] = 1e-2,
    clip: Optional[float] = None,
    print_every: int = 100,
    project_theta: bool = True,
):
    """Basic gradient descent loop for a flat params vector.

    - Uses JAX `value_and_grad` for gradients.
    - Optional elementwise gradient clipping.
    - Optional projection of the last 4 params (theta,a1,a2,h) to keep theta wrapped.
    """

    value_and_grad_fn = jax.value_and_grad(objective_fn)

    @jax.jit
    def step(params: jnp.ndarray, step_lr: jnp.ndarray):
        val, grad = value_and_grad_fn(params, **objective_kwargs)
        if clip is not None:
            grad = jnp.clip(grad, -clip, clip)
        # `step_lr` can be a scalar () or a vector (params.shape)
        new_params = params - step_lr * grad

        if project_theta:
            # params layout: [u1 (n1), u2 (n2), theta, a1, a2, h]
            theta = new_params[-4]
            theta = (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
            new_params = new_params.at[-4].set(theta)

        return new_params, val, grad

    params = params_init

    # Allow scalar lr or per-parameter lr vector.
    if isinstance(lr, (int, float)):
        lr_arr = jnp.asarray(lr, dtype=params.dtype)
    else:
        lr_arr = jnp.asarray(lr, dtype=params.dtype)
        if lr_arr.shape != params.shape:
            raise ValueError(f"per-parameter lr must have shape {params.shape}, got {lr_arr.shape}")
    last_val = None

    for k in range(steps):
        params, val, grad = step(params, lr_arr)
        last_val = val

        if (k % print_every) == 0 or k == steps - 1:
            grad_norm = jnp.linalg.norm(grad)
            if lr_arr.shape == ():
                lr_msg = f"lr={float(lr_arr):.2e}"
            else:
                # Handy diagnostics: typical step sizes by block
                n_last = 4
                step_u_max = float(jnp.max(jnp.abs(lr_arr[:-n_last] * grad[:-n_last])))
                step_last_max = float(jnp.max(jnp.abs(lr_arr[-n_last:] * grad[-n_last:])))
                lr_msg = f"lr_u~{float(jnp.max(lr_arr[:-n_last])):.2e} lr_last4={list(map(float, lr_arr[-n_last:]))}  step_u_max={step_u_max:.2e} step_last4_max={step_last_max:.2e}"

            print(f"step={k:5d}  obj={float(val): .6e}  grad_norm={float(grad_norm): .6e}  {lr_msg}")

    return params, last_val



if __name__ == "__main__":

    start_cpu = time.process_time()

    # Define the system transition matrix
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])
    
    # Define the X-domain
    x1_min, x1_max = -10.0, 10.0
    x2_min, x2_max = -10.0, 10.0
    x_domain = (x1_min, x1_max, x2_min, x2_max)
    x_star = np.array([5.0, 5.0])
    radius = 2.0

    # Initial tessellation parameters
    n1_internal = 100
    n2_internal = 100
    key = jax.random.PRNGKey(0)

    # Random initialization
    # u1/u2 are unconstrained; actual grid gaps come from softplus(u) then normalization.
    use_random_init = False
    sigma_u = 0.05      # spacing log-gap noise; smaller => closer to uniform spacing
    sigma_a = 0.05      # scale log-params (a1/a2); s1,s2 = exp(a)
    sigma_h = 0.02      # shear

    if use_random_init:
        key, k_u1, k_u2, k_th, k_a, k_h = jax.random.split(key, 6)
        u1 = sigma_u * jax.random.normal(k_u1, (n1_internal,))
        u2 = sigma_u * jax.random.normal(k_u2, (n2_internal,))

        # Either a small normal perturbation around 0, or swap to uniform if you prefer:
        # theta = jax.random.uniform(k_th, (), minval=-jnp.pi, maxval=jnp.pi)
        theta = 0.25 * jax.random.normal(k_th, ())

        a1 = sigma_a * jax.random.normal(k_a, ())
        a2 = sigma_a * jax.random.normal(k_a, ())
        h = sigma_h * jax.random.normal(k_h, ())
    else:
        u1 = jnp.zeros((n1_internal,))  # initial uniform spacing
        u2 = jnp.zeros((n2_internal,))
        theta = -jnp.pi/4
        a1, a2 = 0.0, 0.0
        h = 0.0


    # Compute y-domain bounds from x-domain and transformation
    M = np.array(trans_matrix(theta, a1, a2, h))
    bounds_y, verts_y = gpt.get_yspace_bounds(
        M,
        x1_min, x1_max,
        x2_min, x2_max
    )
    y1_lo, y1_hi = bounds_y["y1"][0], bounds_y["y1"][1]
    y2_lo, y2_hi = bounds_y["y2"][0], bounds_y["y2"][1]


    # Pack initial params; check initial objective + grad
    params = jnp.concatenate([u1, u2, jnp.array([theta, a1, a2, h])])

    # # Compute initial FNR
    # M, y1_params, y2_params = extract_grid_params(
    # params,
    # n1_internal=n1_internal, n2_internal=n2_internal,
    # x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max,
    # min_gap=0.0)
    # # gpt.grid_plotter(M, y1_params, y2_params, x1_min, x1_max, x2_min, x2_max)

    # # run model checking on this tessellation
    # y1_params_ends = np.concatenate(([y1_lo], np.array(y1_params), [y1_hi]))
    # y2_params_ends = np.concatenate(([y2_lo], np.array(y2_params), [y2_hi]))
    # ground_truth_check = fixed_check_ground_truth(x_domain, x_star, radius,
    #                                         y1_params_ends, y2_params_ends, M)
    # kripke_structure = make_kripke_from_params(x_domain, x_star, radius,
    #                                            y1_params_ends, y2_params_ends, M)
    # sat_init_states = model_check_kripke(kripke_structure)
    # checked_safe_states = set(sat_init_states)
    # true_safe_states = {s for s, v in ground_truth_check.items() if v == 'goal'}
    # fnr, false_negative_states = false_negative_rate(true_safe_states, checked_safe_states)
    # print(f"False Negative Rate (FNR): {fnr:.4f}")

    # gpt.plot_x_grid_false_negative_map(
    #     M,
    #     y1_params_ends,
    #     y2_params_ends,
    #     sat=sat_init_states,
    #     false_negative_states=false_negative_states,
    #     x_domain=x_domain,
    #     x_star=x_star,
    #     radius=radius,
    # )






    val, grad = jax.value_and_grad(diff_successor_count_objective)(
        params,
        A=A, center=CENTER,
        y1_lo=y1_lo, y1_hi=y1_hi, y2_lo=y2_lo, y2_hi=y2_hi,
        x_domain=x_domain,
        n1_internal=n1_internal, n2_internal=n2_internal,
        horizon=1,
        tau_bbox=0.02,
        beta_overlap=500.0,
        lam_oob=10.0,
        beta_oob=50.0,
        min_gap=0.0,
        weight_mode="cell_area_y",
        weight_power=1.0,
    )

    # Basic gradient descent training loop
    obj_kwargs = dict(
        A=jnp.asarray(A),
        center=jnp.asarray(CENTER),
        y1_lo=y1_lo, y1_hi=y1_hi,
        y2_lo=y2_lo, y2_hi=y2_hi,
        x_domain=x_domain,
        n1_internal=n1_internal, n2_internal=n2_internal,
        horizon=1,
        tau_bbox=0.02,
        beta_overlap=200.0,
        lam_oob=10.0,
        beta_oob=50.0,
        min_gap=0.0,
        weight_mode="cell_area_y",
        weight_power=1.0)
    params_opt, final_val = gradient_descent_optimize(
        params,
        diff_successor_count_objective,
        obj_kwargs,
        steps=400,
        # Per-parameter learning rates: spacings (u1,u2) vs (theta,a1,a2,h)
        lr=jnp.concatenate(
            [
                1e-3 * jnp.ones_like(u1),
                1e-3 * jnp.ones_like(u2),
                jnp.array([1e-3, 1e-3, 1e-3, 0.0]),
            ]
        ),
        clip=50.0,
        print_every=100,
    )
    print("Final objective:", float(final_val))

    # params_opt = params
    
    M, y1_params, y2_params = extract_grid_params(
    params_opt,
    n1_internal=n1_internal, n2_internal=n2_internal,
    x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max,
    min_gap=0.0)
    # gpt.grid_plotter(M, y1_params, y2_psarams, x1_min, x1_max, x2_min, x2_max)

    # run model checking on this tessellation
    y1_params_ends = np.concatenate(([y1_lo], np.array(y1_params), [y1_hi]))
    y2_params_ends = np.concatenate(([y2_lo], np.array(y2_params), [y2_hi]))
    ground_truth_check = fixed_check_ground_truth(x_domain, x_star, radius,
                                            y1_params_ends, y2_params_ends, M)
    kripke_structure = make_kripke_from_params(x_domain, x_star, radius,
                                               y1_params_ends, y2_params_ends, M)
    
    end_cpu = time.process_time()
    cpu_time = end_cpu - start_cpu
    print(f"Building CPU time (s): {cpu_time:.2f}")

    sat_init_states = model_check_kripke(kripke_structure)
    checked_safe_states = set(sat_init_states)
    true_safe_states = {s for s, v in ground_truth_check.items() if v == 'goal'}
    fnr, false_negative_states = false_negative_rate(true_safe_states, checked_safe_states)
    print(f"False Negative Rate (FNR): {fnr:.4f}")

    true_negative_states = {s for s, v in ground_truth_check.items() if v == 'unk' or v == 'fail'}
    num_true_negatives = len(true_negative_states)
    true_negative_prop = num_true_negatives / len(ground_truth_check)
    print(f"Proportion of True Negative States: {true_negative_prop:.4f}")


    gpt.plot_x_grid_false_negative_map(
        M,
        y1_params_ends,
        y2_params_ends,
        sat=sat_init_states,
        false_negative_states=false_negative_states,
        x_domain=x_domain,
        x_star=x_star,
        radius=radius,
    )

    
    # # Optimization
    # obj_kwargs = dict(
    # A=jnp.asarray(A),
    # center=jnp.asarray(CENTER),
    # y1_lo=y1_lo, y1_hi=y1_hi,
    # y2_lo=y2_lo, y2_hi=y2_hi,
    # n1_internal=n1_internal, n2_internal=n2_internal,
    # tau=0.05,
    # min_gap=0.0,          # you can also try min_gap > 0
    # lam_gap=50.0,         # start moderately strong
    # lam_det=10.0,
    # lam_cond=1e-3,
    # lam_shear=1e-2)
    # params_opt = adam_optimize(params, diff_objective_reg, obj_kwargs, steps=2000, lr=1e-2, clip=50.0)




    # # Compute false negative rate
    # checked_safe_states = set(sat_init_states)
    # true_safe_states = {s for s, v in ground_truth_check.items() if v == 'goal'}
    # fnr, false_negative_states = false_negative_rate(true_safe_states, checked_safe_states)
    # print(f"False Negative Rate (FNR): {fnr:.4f}")



    # # Run model checking and compute FNR with these new tessellation parameters
    # M, y1_params, y2_params = extract_grid_params(
    # params_opt,
    # n1_internal=n1_internal, n2_internal=n2_internal,
    # x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max,
    # min_gap=0.0)
    
    # gpt.grid_plotter(M, y1_params, y2_params, x1_min, x1_max, x2_min, x2_max)

    # gpt.plot_x_grid_false_negative_map(
    #     M,
    #     y1_params_ends,
    #     y2_params_ends,
    #     sat=sat_init_states,
    #     false_negative_states=false_negative_states,
    #     x_domain=x_domain,
    #     x_star=x_star,
    #     radius=radius,
    # )


    # print("Objective:", float(val))
    # print("Grad norm:", float(jnp.linalg.norm(grad)))

    # # Compute cost for this tessellation
    # J = quick_objective(
    #     y1_vals, y2_vals,
    #     theta, s1, s2, h,
    #     A=A, center=CENTER,
    #     spread_mode="bbox_area",     # try: "bbox_diag", "mean_pairwise", "trace_cov"
    #     weight_mode="uniform"        # try: "cell_area_y"
    # )
    # print("Objective:", J)

    # # Plot the grids
    # gpt.grid_plotter(M, y1_vals, y2_vals, x1_min, x1_max, x2_min, x2_max)



    # # Define the X-domain
    # x1_min, x1_max = -10.0, 10.0
    # x2_min, x2_max = -10.0, 10.0
    # verts_x = np.array([
    #     [x1_min, x2_min],
    #     [x1_min, x2_max],
    #     [x1_max, x2_max],
    #     [x1_max, x2_min],
    # ], dtype=float)

    # # Define abstraction mapping parameters
    # theta = -np.pi / 4  # 45 degree rotation
    # s1, s2 = 1.0, 1.0  # scaling factors
    # h = 0.0 # shear factor
    # R = np.array([[np.cos(theta), -np.sin(theta)],
    #               [np.sin(theta),  np.cos(theta)]])
    # S = np.array([[s1, 0],
    #               [0, s2]])
    # H = np.array([[1, h],
    #               [0, 1]])
    # M = H @ S @ R  # combined transformation matrix

    # # Define the induced Y-domain
    # bounds_y, verts_y = gpt.get_yspace_bounds(
    #     M,
    #     x1_min, x1_max,
    #     x2_min, x2_max
    # )
    # print("Y-space bounds:", bounds_y)

    # # Example grid
    # n_lines = 10
    # y1_vals = np.linspace(bounds_y["y1"][0], bounds_y["y1"][1], n_lines)
    # y2_vals = np.linspace(bounds_y["y2"][0], bounds_y["y2"][1], n_lines)
    # gpt.grid_plotter(M, y1_vals, y2_vals, x1_min, x1_max, x2_min, x2_max)

