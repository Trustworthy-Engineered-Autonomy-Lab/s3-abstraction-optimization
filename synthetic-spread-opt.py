# Libraries
import numpy as np
import grid_plot_tools as gpt
from spread_objective_tools import trans_matrix, diff_objective_reg, extract_grid_params
from model_checking_tools import make_kripke_from_params, model_check_kripke, check_ground_truth, false_negative_rate
import jax
import jax.numpy as jnp

CENTER = np.array([5.0, 5.0])

def adam_optimize(params, obj_fn, obj_kwargs, steps=2000, lr=1e-2,
                  b1=0.9, b2=0.999, eps=1e-8, clip=10.0, print_every=100):
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)

    value_and_grad = jax.jit(jax.value_and_grad(lambda p: obj_fn(p, **obj_kwargs)))

    for t in range(1, steps + 1):
        val, g = value_and_grad(params)

        # optional grad clip (helps a lot early on)
        gnorm = jnp.linalg.norm(g)
        g = jnp.where(gnorm > clip, g * (clip / (gnorm + 1e-12)), g)

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)

        # params = params - lr * mhat / (jnp.sqrt(vhat) + eps)
        # Update only the last 4 parameters
        update = lr * mhat[-4:] / (jnp.sqrt(vhat[-4:]) + eps)
        params = params.at[-4:].add(-update)

        if (t % print_every) == 0 or t == 1:
            print(f"step {t:5d} | obj {float(val):.6f} | ||g|| {float(gnorm):.6f}")

    return params


if __name__ == "__main__":

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
    u1 = jnp.zeros((n1_internal,))
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
    val, grad = jax.value_and_grad(diff_objective_reg)(
        params,
        A=A, center=CENTER,
        y1_lo=y1_lo, y1_hi=y1_hi, y2_lo=y2_lo, y2_hi=y2_hi,
        n1_internal=n1_internal, n2_internal=n2_internal,
        tau=0.05, min_gap=0.0
    )



    # Run model checking and compute FNR with these initial tessellation parameters
    M, y1_params, y2_params = extract_grid_params(
    params,
    n1_internal=n1_internal, n2_internal=n2_internal,
    x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max,
    min_gap=0.0)
    y1_params_ends = np.concatenate(([y1_lo], np.array(y1_params), [y1_hi]))
    y2_params_ends = np.concatenate(([y2_lo], np.array(y2_params), [y2_hi]))
    ground_truth_check = check_ground_truth(x_domain, x_star, radius,
                                            y1_params_ends, y2_params_ends, M)
    kripke_structure = make_kripke_from_params(x_domain, x_star, radius,
                                               y1_params_ends, y2_params_ends, M)
    sat_init_states = model_check_kripke(kripke_structure)
    checked_safe_states = set(sat_init_states)
    true_safe_states = {s for s, v in ground_truth_check.items() if v == 'goal'}
    fnr, false_negative_states = false_negative_rate(true_safe_states, checked_safe_states)
    print(f"False Negative Rate (FNR): {fnr:.4f}")





    # Compute false negative rate
    checked_safe_states = set(sat_init_states)
    true_safe_states = {s for s, v in ground_truth_check.items() if v == 'goal'}
    fnr, false_negative_states = false_negative_rate(true_safe_states, checked_safe_states)
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    
    # Optimization
    obj_kwargs = dict(
    A=jnp.asarray(A),
    center=jnp.asarray(CENTER),
    y1_lo=y1_lo, y1_hi=y1_hi,
    y2_lo=y2_lo, y2_hi=y2_hi,
    n1_internal=n1_internal, n2_internal=n2_internal,
    tau=0.05,
    min_gap=0.0,          # you can also try min_gap > 0
    lam_gap=50.0,         # start moderately strong
    lam_det=10.0,
    lam_cond=1e-3,
    lam_shear=1e-2)
    params_opt = adam_optimize(params, diff_objective_reg, obj_kwargs, steps=2000, lr=1e-2, clip=50.0)

    # Run model checking and compute FNR with these new tessellation parameters
    M, y1_params, y2_params = extract_grid_params(
    params_opt,
    n1_internal=n1_internal, n2_internal=n2_internal,
    x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max,
    min_gap=0.0)
    y1_params_ends = np.concatenate(([y1_lo], np.array(y1_params), [y1_hi]))
    y2_params_ends = np.concatenate(([y2_lo], np.array(y2_params), [y2_hi]))
    ground_truth_check = check_ground_truth(x_domain, x_star, radius,
                                            y1_params_ends, y2_params_ends, M)
    kripke_structure = make_kripke_from_params(x_domain, x_star, radius,
                                               y1_params_ends, y2_params_ends, M)
    sat_init_states = model_check_kripke(kripke_structure)
    checked_safe_states = set(sat_init_states)
    true_safe_states = {s for s, v in ground_truth_check.items() if v == 'goal'}
    fnr, false_negative_states = false_negative_rate(true_safe_states, checked_safe_states)
    print(f"False Negative Rate (FNR): {fnr:.4f}")
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

