# Libraries
import numpy as np
import mc_model_checking_tools as mct
import mc_objective_tools as mot
import matplotlib.pyplot as plt
# import mc_grid_plot_tools as gpt
# from mc_spread_objective_tools import extract_grid_params, trans_matrix, diff_successor_count_objective
# from mc_model_checking_tools import make_kripke_from_params, model_check_kripke, fixed_check_ground_truth, false_negative_rate
import jax
import jax.numpy as jnp


def plot_grid_sat_map(
    x1_params: np.ndarray,
    x2_params: np.ndarray,
    sat_states,
    *,
    x_domain,
    out_path: str = "mountaincar_grid_sat.png",
    show: bool = True,
    alpha: float = 0.65,
):
    """Plot axis-aligned grid in (x1,x2) and color each cell by satisfaction."""

    from matplotlib.collections import PolyCollection

    x1_params = np.asarray(x1_params, dtype=float).ravel()
    x2_params = np.asarray(x2_params, dtype=float).ravel()
    sat_states = set(sat_states)

    n1 = len(x1_params) - 1
    n2 = len(x2_params) - 1

    def state_id(i: int, j: int) -> int:
        return i * n2 + j

    polys = []
    facecolors = []
    for i in range(n1):
        x1_lo, x1_hi = x1_params[i], x1_params[i + 1]
        for j in range(n2):
            x2_lo, x2_hi = x2_params[j], x2_params[j + 1]
            polys.append(
                np.array(
                    [
                        [x1_lo, x2_lo],
                        [x1_lo, x2_hi],
                        [x1_hi, x2_hi],
                        [x1_hi, x2_lo],
                    ],
                    dtype=float,
                )
            )
            is_sat = state_id(i, j) in sat_states
            facecolors.append((0.2, 0.75, 0.2, alpha) if is_sat else (0.85, 0.2, 0.2, alpha))

    fig, ax = plt.subplots(figsize=(7, 5))
    coll = PolyCollection(
        polys,
        facecolors=facecolors,
        edgecolors=(0.0, 0.0, 0.0, 0.15),
        linewidths=0.3,
    )
    ax.add_collection(coll)

    x1_min, x1_max, x2_min, x2_max = map(float, x_domain)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("MC grid: sat (green) vs unsat (red)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    return fig, ax






def gradient_descent(
    params_init,
    objective_fn,
    *,
    x_domain,
    n1_internal,
    n2_internal,
    steps,
    lr,
    grad_clip,
    do_verify=False,
    record_every=10,
    print_every=10,
):

    @jax.jit
    def gd_step(p, lr_value):
        value, g = jax.value_and_grad(objective_fn)(
            p,
            x_domain=x_domain,
            n1_internal=n1_internal,
            n2_internal=n2_internal,
        )
        g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        g_norm = jnp.linalg.norm(g)
        scale = jnp.minimum(1.0, grad_clip / (g_norm + 1e-12))
        g = g * scale
        return p - jnp.asarray(lr_value, dtype=p.dtype) * g, value, g_norm
    
    if do_verify:
        gt_reach_regions = mct.get_gt_reach_regions(x_domain, grid_resolution=10)
    
    params_gd = params_init
    cost_history = []
    grad_norm_history = []
    sat_history = []
    fnr_history = []
    for k in range(steps):
        params_gd, value, g_norm = gd_step(params_gd, lr)

        if k % record_every == 0:
            cost_history.append(float(value))
            grad_norm_history.append(float(g_norm))

        if do_verify and k % record_every == 0:

            # Make Kripke from parameters and model check
            x1_params, x2_params = mot.extract_grid_params(
            params_gd,
            n1_internal=n1_internal,
            n2_internal=n2_internal,
            x1_min=x1_min,
            x1_max=x1_max,
            x2_min=x2_min,
            x2_max=x2_max,
            )
            kripke_structure = mct.make_kripke(x1_params, x2_params, allow_self_loops=False, advanced_metrics=True)
            sat_states = mct.model_check_kripke(kripke_structure)
            sat_rate = len(sat_states)/((n1_internal+1) * (n2_internal+1))
            sat_history.append(sat_rate)

            # Compare cells to ground truth and compute FNR
            ground_truth_reference = mct.check_ground_truth_fast(x1_params, x2_params, x_domain, gt_reach_regions)
            checked_sat_states = set(sat_states)
            true_sat_states = {s for s, v in ground_truth_reference.items() if v == 'goal'}
            fnr, _ = mct.false_negative_rate(true_sat_states, checked_sat_states)
            fnr_history.append(fnr)
        
        if k % print_every == 0:
            if do_verify and k % record_every == 0:
                print(f"[{k}] J(p)={float(value):.3f}, |∇J(p)|={float(g_norm):.3f}, Sat rate: {sat_rate*100.0:.2f}%, FNR: {fnr:.4f}")
            else:
                print(f"[{k}] J(p)={float(value):.3f}, |∇J(p)|={float(g_norm):.3f}")

    return params_gd, np.array(cost_history), np.array(grad_norm_history), np.array(sat_history), np.array(fnr_history)


if __name__ == "__main__":
    
    # Define the X-domain
    x1_min, x1_max = -1.2, 0.6
    x2_min, x2_max = -0.07, 0.07
    x_domain = (x1_min, x1_max, x2_min, x2_max)

    # # Mountain Car Simulation
    # state = np.array([-0.5, 0.0])
    # for _ in range(100):
    #     state = mct.mc_cl_dynamics(state)
    #     print(state)

    # Initial tessellation parameters
    n1_internal = 80
    n2_internal = 80
    sigma_u = 0.5
    key = jax.random.PRNGKey(0)
    u1 = jnp.zeros((n1_internal,))  # initial uniform spacing
    u2 = jnp.zeros((n2_internal,))
    # key, k_u1, k_u2 = jax.random.split(key, 3)
    # u1 = sigma_u * jax.random.normal(k_u1, (n1_internal,))
    # u2 = sigma_u * jax.random.normal(k_u2, (n2_internal,))
    # theta = 0 # -jnp.pi / 4
    # a1, a2 = 0.0, 0.0
    # h = 0.0

    params = jnp.concatenate([u1, u2])

    # Compute initial objective and gradient
    val, grad = jax.value_and_grad(mot.image_area)(
        params,
        x_domain=x_domain,
        n1_internal=n1_internal,
        n2_internal=n2_internal,
    )
    print(f"Initial objective value: {val:.4f}")
    print(f"Initial objective grad norm: {jnp.linalg.norm(grad):.4f}")


    # Make Kripke structure and model check
    x1_params, x2_params = mot.extract_grid_params(
        params,
        n1_internal=n1_internal,
        n2_internal=n2_internal,
        x1_min=x1_min,
        x1_max=x1_max,
        x2_min=x2_min,
        x2_max=x2_max,
    )
    kripke_structure = mct.make_kripke(x1_params, x2_params, allow_self_loops=False, advanced_metrics=True, verbose=True)
    sat_states = mct.model_check_kripke(kripke_structure)
    sat_rate = len(sat_states)/((n1_internal+1) * (n2_internal+1))
    print(f"Sat rate: {sat_rate*100.0:.2f}%")

    # gt_reach_regions = mct.get_gt_reach_regions(x_domain, grid_resolution=10)
    # ground_truth_reference = mct.check_ground_truth_fast(x1_params, x2_params, x_domain, gt_reach_regions)
    # checked_sat_states = set(sat_states)
    # true_sat_states = {s for s, v in ground_truth_reference.items() if v == 'goal'}
    # fnr, _ = mct.false_negative_rate(true_sat_states, checked_sat_states)
    # print(fnr)


    plot_grid_sat_map(
        x1_params,
        x2_params,
        sat_states,
        x_domain=x_domain,
        out_path="mountaincar_init.png",
        show=True,
    )



    # Optimize params
    params_opt, cost_history, grad_norm_history, sat_history, fnr_history = gradient_descent(
        params,
        mot.image_area,
        x_domain=x_domain,
        n1_internal=n1_internal,
        n2_internal=n2_internal,
        steps=1_000,
        lr=4e-1,
        grad_clip=1e3,
        print_every=10,
        record_every=10,
        do_verify=False,)
    




    # Make Kripke structure and model check
    x1_params, x2_params = mot.extract_grid_params(
        params_opt,
        n1_internal=n1_internal,
        n2_internal=n2_internal,
        x1_min=x1_min,
        x1_max=x1_max,
        x2_min=x2_min,
        x2_max=x2_max,
    )
    kripke_structure = mct.make_kripke(x1_params, x2_params, allow_self_loops=False, advanced_metrics=True, verbose=True)
    sat_states = mct.model_check_kripke(kripke_structure)
    sat_rate = len(sat_states)/((n1_internal+1) * (n2_internal+1))
    print(f"Sat rate: {sat_rate*100.0:.2f}%")

    # Compare cells to ground truth and compute FNR
    gt_reach_regions = mct.get_gt_reach_regions(x_domain, grid_resolution=10)
    ground_truth_reference = mct.check_ground_truth_fast(x1_params, x2_params, x_domain, gt_reach_regions)
    checked_sat_states = set(sat_states)
    true_sat_states = {s for s, v in ground_truth_reference.items() if v == 'goal'}
    fnr, _ = mct.false_negative_rate(true_sat_states, checked_sat_states)


    plot_grid_sat_map(
        x1_params,
        x2_params,
        sat_states,
        x_domain=x_domain,
        out_path="mountaincar_final.png",
        show=True,
    )



    # # Simple visualization: vertical lines at x1_params, horizontal lines at x2_params
    # x1_lines = np.asarray(x1_params, dtype=float).ravel()
    # x2_lines = np.asarray(x2_params, dtype=float).ravel()

    # fig, ax = plt.subplots(figsize=(6, 6))
    # for x in x1_lines:
    #     ax.axvline(x, color="C0", lw=1.0, alpha=0.9)
    # for y in x2_lines:
    #     ax.axhline(y, color="C1", lw=1.0, alpha=0.9)

    # ax.set_xlim(x1_min, x1_max)
    # ax.set_ylim(x2_min, x2_max)
    # ax.set_aspect("auto")
    # ax.set_xlabel("x1")
    # ax.set_ylabel("x2")
    # ax.set_title("Optimized grid")
    # ax.grid(False)
    # fig.tight_layout()
    # fig.savefig("synthetic_v4_grid.png", dpi=200)
    # plt.show()


    # # Initial model checking results
    # M, y1_params, y2_params = extract_grid_params(
    #     params,
    #     n1_internal=n1_internal,
    #     n2_internal=n2_internal,
    #     x1_min=x1_min,
    #     x1_max=x1_max,
    #     x2_min=x2_min,
    #     x2_max=x2_max,
    #     min_gap=0.0,
    # )
    # y1_params_ends = np.array(y1_params)
    # y2_params_ends = np.array(y2_params)
    # gpt.grid_plotter(M, y1_params_ends, y2_params_ends, x1_min, x1_max, x2_min, x2_max)
    # kripke_structure = make_kripke_from_params(y1_params_ends, y2_params_ends, M)
    # ground_truth_check = fixed_check_ground_truth(x_domain, y1_params_ends, y2_params_ends, M, grid_resolution=10)
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
    #     x_domain=x_domain)




    
    # # Compute y-domain bounds from x-domain and initial transformation
    # M = np.array(trans_matrix(theta, a1, a2, h))
    # bounds_y, verts_y = gpt.get_yspace_bounds(
    #     M,
    #     x1_min, x1_max,
    #     x2_min, x2_max
    # )
    # y1_lo, y1_hi = bounds_y["y1"][0], bounds_y["y1"][1]
    # y2_lo, y2_hi = bounds_y["y2"][0], bounds_y["y2"][1]

    # # -----------------
    # # Optimize grid
    # # -----------------
    # obj_kwargs = dict(
    #     y1_lo=y1_lo,
    #     y1_hi=y1_hi,
    #     y2_lo=y2_lo,
    #     y2_hi=y2_hi,
    #     n1_internal=n1_internal,
    #     n2_internal=n2_internal,
    #     min_gap=0.0,
    #     temperature=0.10,          # closer to greedy while still smooth
    #     horizon=1,
    #     tau_bbox=0.02,
    #     beta_overlap=200.0,
    #     weight_mode="uniform",
    # )

    # val0, grad0 = jax.value_and_grad(diff_successor_count_objective)(params, **obj_kwargs)
    # print("init obj:", float(val0))
    # print("init grad norm:", float(jnp.linalg.norm(grad0)))

    # lr_vec = jnp.concatenate(
    #     [
    #         1e-4 * jnp.ones_like(u1),
    #         1e-4 * jnp.ones_like(u2),
    #         jnp.array([0.0, 0.0, 0.0, 0.0]),
    #     ]
    # )

    # params_opt, final_val = gradient_descent_optimize(
    #     params,
    #     diff_successor_count_objective,
    #     obj_kwargs,
    #     steps=10000,
    #     lr=lr_vec,
    #     clip=50.0,
    #     print_every=500,
    # )
    # print("final obj:", float(final_val))

    # # params_opt = params

    # # Extract optimized grid for plotting + model checking
    # M, y1_params, y2_params = extract_grid_params(
    #     params_opt,
    #     n1_internal=n1_internal,
    #     n2_internal=n2_internal,
    #     x1_min=x1_min,
    #     x1_max=x1_max,
    #     x2_min=x2_min,
    #     x2_max=x2_max,
    #     min_gap=0.0,
    # )


    # # y1_params/y2_params from extract_grid_params already include endpoints
    # y1_params_ends = np.array(y1_params)
    # y2_params_ends = np.array(y2_params)

    # # gpt.grid_plotter(M, y1_params_ends, y2_params_ends, x1_min, x1_max, x2_min, x2_max)


    # kripke_structure = make_kripke_from_params(y1_params_ends, y2_params_ends, M)
    # # print(kripke_structure)

    # end_cpu = time.process_time()
    # cpu_time = end_cpu - start_cpu
    # print(f"Building CPU time (s): {cpu_time:.2f}")

    # ground_truth_check = fixed_check_ground_truth(x_domain, y1_params_ends, y2_params_ends, M, grid_resolution=10)
    
    # sat_init_states = model_check_kripke(kripke_structure)
    # # print(sat_init_states)

    # checked_safe_states = set(sat_init_states)
    # true_safe_states = {s for s, v in ground_truth_check.items() if v == 'goal'}
    # fnr, false_negative_states = false_negative_rate(true_safe_states, checked_safe_states)
    # print(f"False Negative Rate (FNR): {fnr:.4f}")


    # true_negative_states = {s for s, v in ground_truth_check.items() if v == 'unk' or v == 'fail'}
    # num_true_negatives = len(true_negative_states)
    # true_negative_prop = num_true_negatives / len(ground_truth_check)
    # print(f"Proportion of True Negative States: {true_negative_prop:.4f}")


    # gpt.plot_x_grid_false_negative_map(
    #     M,
    #     y1_params_ends,
    #     y2_params_ends,
    #     sat=sat_init_states,
    #     false_negative_states=false_negative_states,
    #     x_domain=x_domain)
    




    # # Random initialization
    # # u1/u2 are unconstrained; actual grid gaps come from softplus(u) then normalization.
    # use_random_init = False
    # sigma_u = 0.05      # spacing log-gap noise; smaller => closer to uniform spacing
    # sigma_a = 0.05      # scale log-params (a1/a2); s1,s2 = exp(a)
    # sigma_h = 0.02      # shear

    # if use_random_init:
    #     key, k_u1, k_u2, k_th, k_a, k_h = jax.random.split(key, 6)
    #     u1 = sigma_u * jax.random.normal(k_u1, (n1_internal,))
    #     u2 = sigma_u * jax.random.normal(k_u2, (n2_internal,))

    #     # Either a small normal perturbation around 0, or swap to uniform if you prefer:
    #     # theta = jax.random.uniform(k_th, (), minval=-jnp.pi, maxval=jnp.pi)
    #     theta = 0.25 * jax.random.normal(k_th, ())

    #     a1 = sigma_a * jax.random.normal(k_a, ())
    #     a2 = sigma_a * jax.random.normal(k_a, ())
    #     h = sigma_h * jax.random.normal(k_h, ())
    # else:
    #     u1 = jnp.zeros((n1_internal,))  # initial uniform spacing
    #     u2 = jnp.zeros((n2_internal,))
    #     theta = 0  # -jnp.pi/4
    #     a1, a2 = 0.0, 0.0
    #     h = 0.0


    # # Compute y-domain bounds from x-domain and transformation
    # M = np.array(trans_matrix(theta, a1, a2, h))
    # bounds_y, verts_y = get_yspace_bounds(
    #     M,
    #     x1_min, x1_max,
    #     x2_min, x2_max
    # )
    # y1_lo, y1_hi = bounds_y["y1"][0], bounds_y["y1"][1]
    # y2_lo, y2_hi = bounds_y["y2"][0], bounds_y["y2"][1]


    # # Pack initial params; check initial objective + grad
    # params = jnp.concatenate([u1, u2, jnp.array([theta, a1, a2, h])])
    # val, grad = jax.value_and_grad(diff_objective)(
    #     params,
    #     A=A, center=CENTER,
    #     y1_lo=y1_lo, y1_hi=y1_hi, y2_lo=y2_lo, y2_hi=y2_hi,
    #     n1_internal=n1_internal, n2_internal=n2_internal,
    #     tau=0.05,
    #     min_gap=0.0,
    #     weight_mode="cell_area_y",
    #     weight_power=1.0,
    # )
    # print(grad)

    # # Basic gradient descent training loop
    # obj_kwargs = dict(
    #     A=jnp.asarray(A),
    #     center=jnp.asarray(CENTER),
    #     y1_lo=y1_lo, y1_hi=y1_hi,
    #     y2_lo=y2_lo, y2_hi=y2_hi,
    #     n1_internal=n1_internal, n2_internal=n2_internal,
    #     tau=0.05,
    #     min_gap=0.0,
    #     weight_mode="cell_area_y",
    #     weight_power=1.0,
    # )
    # params_opt, final_val = gradient_descent_optimize(
    #     params,
    #     diff_objective,
    #     obj_kwargs,
    #     steps=10000,
    #     # Per-parameter learning rates: spacings (u1,u2) vs (theta,a1,a2,h)
    #     lr=jnp.concatenate(
    #         [
    #             1e-1 * jnp.ones_like(u1),
    #             1e-1 * jnp.ones_like(u2),
    #             jnp.array([1e-2, 1e-3, 1e-3, 0]),
    #         ]
    #     ),
    #     clip=50.0,
    #     print_every=500,
    # )
    # print("Final objective:", float(final_val))
    
    # M, y1_params, y2_params = extract_grid_params(
    # params_opt,
    # n1_internal=n1_internal, n2_internal=n2_internal,
    # x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max,
    # min_gap=0.0)
    # grid_plotter(M, y1_params, y2_params, x1_min, x1_max, x2_min, x2_max)

    
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
    # y1_params_ends = np.concatenate(([y1_lo], np.array(y1_params), [y1_hi]))
    # y2_params_ends = np.concatenate(([y2_lo], np.array(y2_params), [y2_hi]))
    # ground_truth_check = check_ground_truth(x_domain, x_star, radius,
    #                                         y1_params_ends, y2_params_ends, M)
    # kripke_structure = make_kripke_from_params(x_domain, x_star, radius,
    #                                            y1_params_ends, y2_params_ends, M)
    # sat_init_states = model_check_kripke(kripke_structure)
    # checked_safe_states = set(sat_init_states)
    # true_safe_states = {s for s, v in ground_truth_check.items() if v == 'goal'}
    # fnr, false_negative_states = false_negative_rate(true_safe_states, checked_safe_states)
    # print(f"False Negative Rate (FNR): {fnr:.4f}")
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

