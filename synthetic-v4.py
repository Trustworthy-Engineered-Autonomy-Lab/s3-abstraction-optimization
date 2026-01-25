# Libraries
import numpy as np
import grid_plot_tools as gpt
import jax
import jax.numpy as jnp
import synthetic_objective_tools as sot
import matplotlib.pyplot as plt
import model_checking_tools as mct
import pickle as pkl
import time


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
        gt_reach_regions = mct.get_gt_reach_regions(x_domain, grid_resolution=100)
    
    params_gd = params_init
    cost_history = []
    grad_norm_history = []
    sat_history = []
    fnr_history = []
    tnp_history = []
    sr_history = []
    for k in range(steps):
        params_gd, value, g_norm = gd_step(params_gd, lr)

        if k % record_every == 0:
            cost_history.append(float(value))
            grad_norm_history.append(float(g_norm))

        if do_verify and k % record_every == 0:

            # Make Kripke from parameters and model check
            x1_params, x2_params = sot.extract_grid_params(
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

            # Compare cells to ground truth validation
            ground_truth_reference = mct.check_ground_truth_fast(x1_params, x2_params, x_domain, gt_reach_regions)
            checked_sat_states = set(sat_states)
            true_sat_states = {s for s, v in ground_truth_reference.items() if v == 'goal'}

            # Compute sat coverage in ground truth
            gt_sat_states = {s for s, v in gt_reach_regions.items() if v in ['goal']}
            gt_coverage = len(gt_sat_states) / len(gt_reach_regions)

            # Compute sat coverage of abstract states (% of area covered by sat states); compare to gt
            sat_coverage = mct.compute_sat_coverage(sat_states, x1_params, x2_params)
            sr = sat_coverage / gt_coverage
            sr_history.append(sr)

            # Compute the proportion of true negative states
            true_negative_states = {s for s, v in gt_reach_regions.items() if v == 'unk' or v == 'fail'}
            num_true_negatives = len(true_negative_states)
            tnp = num_true_negatives / len(gt_reach_regions)
            tnp_history.append(tnp)

            # Compute FNR
            fnr, _ = mct.false_negative_rate(true_sat_states, checked_sat_states)
            fnr_history.append(fnr)
        
        if k % print_every == 0:
            if do_verify and k % record_every == 0:
                print(f"[{k}] J(p)={float(value):.3f}, |∇J(p)|={float(g_norm):.3f}")
                print(f"      Sat rate: {sat_rate*100.0:.2f}%, SR: {sr*100.0:.2f}")
                print(f"      FNR: {fnr*100.0:.2f}%, TNP: {tnp*100.0:.2f}")
            else:
                print(f"[{k}] J(p)={float(value):.3f}, |∇J(p)|={float(g_norm):.3f}")

    return params_gd, np.array(cost_history), np.array(grad_norm_history), np.array(sat_history), np.array(sr_history), np.array(tnp_history), np.array(fnr_history)



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
    sigma_u = 2.0
    key = jax.random.PRNGKey(0)
    key, k_u1, k_u2 = jax.random.split(key, 3)
    u1 = sigma_u * jax.random.normal(k_u1, (n1_internal,))
    u2 = sigma_u * jax.random.normal(k_u2, (n2_internal,))
    # u1 = jnp.zeros((n1_internal,))  # initial uniform spacing
    # u2 = jnp.zeros((n2_internal,))
    # theta = 0 # -jnp.pi/4
    # a1, a2 = 0.0, 0.0
    # h = 0.0
    
    # Pack initial params; check initial objective + grad
    params = jnp.concatenate([u1, u2])

    # # Make Kripke of initial grid for reference
    # x1_params, x2_params = sot.extract_grid_params(
    #     params,
    #     n1_internal=n1_internal,
    #     n2_internal=n2_internal,
    #     x1_min=x1_min,
    #     x1_max=x1_max,
    #     x2_min=x2_min,
    #     x2_max=x2_max,
    # )
    # kripke_structure = mct.make_kripke(x1_params, x2_params, allow_self_loops=True, advanced_metrics=True)
    # sat_states = mct.model_check_kripke(kripke_structure)
    # sat_rate = len(sat_states)/((n1_internal+1) * (n2_internal+1))
    # print(f"Sat rate: {sat_rate*100.0:.2f}%")

    # # Compute initial objective and gradient
    # val, grad = jax.value_and_grad(sot.image_area_over_parent)(
    #     params,
    #     x_domain=x_domain,
    #     n1_internal=n1_internal,
    #     n2_internal=n2_internal,
    # )
    # print(f"Initial objective value: {val:.4f}")
    # print(f"Initial objective grad norm: {jnp.linalg.norm(grad):.4f}")

    # Optimize parameters via gradient descent
    params_opt, cost_history, grad_norm_history, sat_history, sr_history, tnp_history, fnr_history = gradient_descent(
        params,
        sot.image_area_over_parent,
        x_domain=x_domain,
        n1_internal=n1_internal,
        n2_internal=n2_internal,
        steps=150,
        lr=5e-4,
        grad_clip=1e3,
        print_every=1,
        record_every=1,
        do_verify=False,)
    
    # # Save the training history
    # results = {
    #     "params": params_opt,
    #     "cost_history": cost_history,
    #     "grad_norm_history": grad_norm_history,
    #     "sat_history": sat_history,
    #     "sr_history": sr_history,
    #     "tnp_history": tnp_history,
    #     "fnr_history": fnr_history,
    # }
    # with open("synthetic_training_history.pkl", "wb") as f:
    #     pkl.dump(results, f)

    # plot_history_curves(
    #     cost_history,
    #     grad_norm_history,
    #     sat_history,
    #     out_path="synthetic_v4_history.png",
    #     show=True,
    # )

    # Make Kripke of optimized grid for reference
    x1_params, x2_params = sot.extract_grid_params(
        params_opt,
        n1_internal=n1_internal,
        n2_internal=n2_internal,
        x1_min=x1_min,
        x1_max=x1_max,
        x2_min=x2_min,
        x2_max=x2_max,
    )
    kripke_structure = mct.make_kripke(x1_params, x2_params, allow_self_loops=True, advanced_metrics=True, verbose=True)

    end_cpu = time.process_time()
    cpu_time = end_cpu - start_cpu
    print(f"Model build CPU time (s): {cpu_time:.2f}")

    sat_states = mct.model_check_kripke(kripke_structure, verbose=True)
    sat_rate = len(sat_states)/((n1_internal+1) * (n2_internal+1))
    print(f"Sat rate: {sat_rate*100.0:.2f}%")

    # Compare against ground truth safety
    gt_reach_regions = mct.get_gt_reach_regions(x_domain, grid_resolution=100)
    ground_truth_reference = mct.check_ground_truth_fast(x1_params, x2_params, x_domain, gt_reach_regions)
    checked_sat_states = set(sat_states)
    true_sat_states = {s for s, v in ground_truth_reference.items() if v == 'goal'}

    # Compute sat coverage in ground truth
    gt_sat_states = {s for s, v in gt_reach_regions.items() if v in ['goal']}
    gt_coverage = len(gt_sat_states) / len(gt_reach_regions)

    # Compute sat coverage of abstract states (% of area covered by sat states); compare to gt
    sat_coverage = mct.compute_sat_coverage(sat_states, x1_params, x2_params)
    coverage_proportion = sat_coverage / gt_coverage
    print(f"Coverage proportion: {coverage_proportion:.4f}%")

    # Compute the proportion of true negative states
    true_negative_states = {s for s, v in gt_reach_regions.items() if v == 'unk' or v == 'fail'}
    num_true_negatives = len(true_negative_states)
    tnp = num_true_negatives / len(gt_reach_regions)
    print(f"TNP: {tnp:.4f}")

    # Compute FNR
    fnr, _ = mct.false_negative_rate(true_sat_states, checked_sat_states)
    print(f"False Negative Rate (FNR): {fnr:.4f}")


    # # print(x1_params)
    # # print(x2_params)

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
    # ax.set_aspect("equal", adjustable="box")
    # ax.set_xlabel("x1")
    # ax.set_ylabel("x2")
    # ax.set_title("Optimized grid")
    # ax.grid(False)
    # fig.tight_layout()
    # fig.savefig("synthetic_v4_grid.png", dpi=200)
    # plt.show()





    # # Compute initial FNR
    # # Compute y-domain bounds from x-domain and transformation
    # M = np.array(sot.trans_matrix(theta, a1, a2, h))
    # bounds_y, verts_y = gpt.get_yspace_bounds(M, x1_min, x1_max, x2_min, x2_max)
    # M, y1_params, y2_params = sot.extract_grid_params(
    # params,
    # n1_internal=n1_internal, n2_internal=n2_internal,
    # x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max,
    # min_gap=0.0)
    # # gpt.grid_plotter(M, y1_params, y2_params, x1_min, x1_max, x2_min, x2_max)

    # # run model checking on this tessellation
    # y1_lo, y1_hi = bounds_y["y1"][0], bounds_y["y1"][1]
    # y2_lo, y2_hi = bounds_y["y2"][0], bounds_y["y2"][1]
    # y1_params_ends = np.concatenate(([y1_lo], np.array(y1_params), [y1_hi]))
    # y2_params_ends = np.concatenate(([y2_lo], np.array(y2_params), [y2_hi]))
    # # ground_truth_check = fixed_check_ground_truth(x_domain, x_star, radius,
    # #                                         y1_params_ends, y2_params_ends, M)
    # kripke_structure = mct.make_kripke_from_params(y1_params_ends, y2_params_ends, M, allow_self_loops=True, advanced_metrics=True)
    # sat_states = mct.model_check_kripke(kripke_structure)
    # sat_rate = len(sat_states)/((n1_internal+1) * (n2_internal+1))
    # print(f"Sat rate: {sat_rate*100.0:.2f}%")

    # Old plot for reference:
    # gpt.plot_x_grid_satisfaction(M, y1_params_ends, y2_params_ends,
    #                              kripke_structure=kripke_structure, sat=sat_init_states,
    #                              x_domain=x_domain, goal_ap="g",
    #                              x_star=x_star, radius=radius)

    # # Plot the grids
    # gpt.grid_plotter(M, y1_params_ends, y2_params_ends, x1_min, x1_max, x2_min, x2_max)

