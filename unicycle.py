# Libraries
import numpy as np
import matplotlib.pyplot as plt
import uni_model_checking_tools as umct
import uni_objective_tools as uot
import jax
import jax.numpy as jnp
import pickle as pkl
import time


def gradient_descent(
    params_init,
    objective_fn,
    *,
    domain,
    n1_internal,
    n2_internal,
    n3_internal,
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
            domain=domain,
            n1_internal=n1_internal,
            n2_internal=n2_internal,
            n3_internal=n3_internal,
        )
        g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        g_norm = jnp.linalg.norm(g)
        scale = jnp.minimum(1.0, grad_clip / (g_norm + 1e-12))
        g = g * scale
        return p - jnp.asarray(lr_value, dtype=p.dtype) * g, value, g_norm
    
    params_gd = params_init
    cost_history = []
    grad_norm_history = []
    for k in range(steps):
        params_gd, value, g_norm = gd_step(params_gd, lr)

        if k % record_every == 0:
            cost_history.append(float(value))
            grad_norm_history.append(float(g_norm))
        
        if k % print_every == 0:
            print(f"[{k}] J(p)={float(value):.3f}, |∇J(p)|={float(g_norm):.3f}")

    return params_gd, np.array(cost_history), np.array(grad_norm_history)


def plot_optimized_grid_params(
    x_params: np.ndarray,
    y_params: np.ndarray,
    theta_params: np.ndarray,
    *,
    domain,
    grid_alpha: float = 0.35,
):
    """Simple visualization of optimized grid parameters.

    - XY: draws vertical/horizontal grid lines.
    - Cell widths: bar plots for dx, dy, dtheta.
    """

    x_min, x_max, y_min, y_max, theta_min, theta_max = domain

    dx = np.diff(x_params)
    dy = np.diff(y_params)
    dtheta = np.diff(theta_params)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    # 1) XY grid lines
    ax = axs[0]
    for xv in x_params:
        ax.axvline(xv, color="k", linewidth=0.7, alpha=grid_alpha)
    for yv in y_params:
        ax.axhline(yv, color="k", linewidth=0.7, alpha=grid_alpha)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Optimized XY grid")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # 2) dx bar plot
    ax = axs[1]
    ax.bar(np.arange(len(dx)), dx, width=1.0)
    ax.set_title("Cell widths: dx")
    ax.set_xlabel("x-cell index")
    ax.set_ylabel("dx")

    # 3) dy bar plot
    ax = axs[2]
    ax.bar(np.arange(len(dy)), dy, width=1.0)
    ax.set_title("Cell widths: dy")
    ax.set_xlabel("y-cell index")
    ax.set_ylabel("dy")

    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 3), constrained_layout=True)
    ax = axs2[0]
    ax.vlines(theta_params, 0.0, 1.0, color="k", linewidth=0.8)
    ax.set_xlim(theta_min, theta_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_title("Theta partition lines")
    ax.set_xlabel("theta")

    ax = axs2[1]
    ax.bar(np.arange(len(dtheta)), dtheta, width=1.0)
    ax.set_title("Cell widths: dtheta")
    ax.set_xlabel("theta-cell index")
    ax.set_ylabel("dtheta")

    return fig, fig2


if __name__ == "__main__":

    start_cpu = time.process_time()

    # Environment details
    obs_center = np.array([25.0, 25.0])
    obs_radius = 5.0
    goal_center = np.array([40.0, 20.0])
    goal_radius = 8.0

    obstacle_centers = np.array([obs_center])
    obstacle_radii = np.array([obs_radius])

    # # Batch simulate
    # N = 10
    # steps = 250
    # theta0 = 0.0 # fixed initial heading

    # rng = np.random.default_rng(0)
    # x0s = rng.uniform([0.0, 0.0], [20.0, 40.0], size=(N, 2))

    # trajectories = []
    # for i in range(N):
    #     x0 = np.array([x0s[i, 0], x0s[i, 1], theta0], dtype=float)
    #     traj = uot.simulate_trajectory(
    #         x0,
    #         steps=steps,
    #         goal_center=goal_center,
    #         goal_radius=goal_radius,
    #     )
    #     trajectories.append(traj)

    # # Plot trajectories and environment
    # plt.figure()
    # for i, traj in enumerate(trajectories):
    #     plt.plot(traj[:, 0], traj[:, 1], linewidth=1.0, alpha=0.35)
    #     plt.plot(traj[0, 0], traj[0, 1], marker='.', color='k', markersize=3, alpha=0.6)

    # obs_circle = plt.Circle(obs_center, obs_radius, color='r', alpha=0.5, label='Obstacle')
    # goal_circle = plt.Circle(goal_center, goal_radius, color='g', alpha=0.35, label='Goal')
    # plt.gca().add_artist(obs_circle)
    # plt.gca().add_artist(goal_circle)

    # plt.title(f"Dubins trajectories w/obstacle and goal")
    # plt.xlabel("X position")
    # plt.ylabel("Y position")
    # plt.axis('equal')
    # plt.grid(True)
    # plt.show()

    # State space domain
    x_min, x_max = 0.0, 50.0
    y_min, y_max = 0.0, 40.0
    theta_min, theta_max = -np.pi, np.pi
    domain = (x_min, x_max, y_min, y_max, theta_min, theta_max)

    # Example grid parameters
    num_x_params = 60
    num_y_params = 60
    num_theta_params = 60
    key = jax.random.PRNGKey(0)
    sigma_u = 1.0
    # u1 = jnp.zeros((num_x_params,))  # initial uniform spacing
    # u2 = jnp.zeros((num_y_params,))
    # u3 = jnp.zeros((num_theta_params,))
    key, k_u1, k_u2 = jax.random.split(key, 3)
    u1 = sigma_u * jax.random.normal(k_u1, (num_x_params,))
    u2 = sigma_u * jax.random.normal(k_u2, (num_y_params,))
    u3 = sigma_u * jax.random.normal(k_u2, (num_theta_params,))

    # # Pack initial params; check initial objective + grad
    params = jnp.concatenate([u1, u2, u3])

    # # # Compute initial objective and gradient
    # # val, grad = jax.value_and_grad(uot.image_volume)(
    # #     params,
    # #     domain=domain,
    # #     n1_internal=num_x_params,
    # #     n2_internal=num_y_params,
    # #     n3_internal=num_theta_params,
    # # )
    # # print(f"Initial objective value: {val:.4f}")
    # # print(f"Initial objective grad norm: {jnp.linalg.norm(grad):.4f}")

    # # Make the kripke structure; check LTL property
    # x_params, y_params, theta_params = uot.extract_grid_params(
    #     params,
    #     n1_internal=num_x_params,
    #     n2_internal=num_y_params,
    #     n3_internal=num_theta_params,
    #     domain=domain,
    # )
    # kripke_structure = umct.make_kripke_from_params(x_params, y_params, theta_params, allow_self_loops=False, advanced_metrics=True)
    # sat_states = umct.model_check_kripke(kripke_structure)
    # sat_rate = len(sat_states)/((num_x_params+1) * (num_y_params+1) * (num_theta_params+1))
    # print(f"Sat rate: {sat_rate*100.0:.2f}%")

    # # Compare against ground truth safety
    # # gt_reach_regions = umct.get_gt_reach_regions(domain, grid_resolution=100, verbose=True)
    # # with open("unicycle_gt_reach_regions_100.pkl", "wb") as f:
    # #     pkl.dump(gt_reach_regions, f)
    # with open("unicycle_gt_reach_regions_100.pkl", "rb") as f:
    #     gt_reach_regions = pkl.load(f)
    # ground_truth_reference = umct.check_ground_truth_fast(x_params, y_params, theta_params, domain, gt_reach_regions)
    # checked_sat_states = set(sat_states)
    # true_sat_states = {s for s, v in ground_truth_reference.items() if v == 'goal'}

    # # Compute sat coverage in ground truth
    # gt_sat_states = {s for s, v in gt_reach_regions.items() if v in ['goal']}
    # gt_coverage = len(gt_sat_states) / len(gt_reach_regions)

    # # Compute sat coverage of abstract states (% of area covered by sat states); compare to gt
    # sat_coverage = umct.compute_sat_coverage(sat_states, x_params, y_params, theta_params)
    # coverage_proportion = sat_coverage / gt_coverage
    # print(f"Coverage proportion: {coverage_proportion:.4f}%")

    # # Compute FNR
    # fnr, _ = umct.false_negative_rate(true_sat_states, checked_sat_states)
    # print(f"False Negative Rate (FNR): {fnr:.4f}")


    # Gradient descent to optimize grid
    params_opt, cost_history, grad_norm_history = gradient_descent(
        params,
        uot.image_volume_over_parent,
        domain=domain,
        n1_internal=num_x_params,
        n2_internal=num_y_params,
        n3_internal=num_theta_params,
        steps=200,
        lr=1e-4,
        grad_clip=1e3,
        print_every=1,
        record_every=10,
        do_verify=False,
    )


    # Extract grid parameters from line gaps
    x_params, y_params, theta_params = uot.extract_grid_params(
        params_opt,
        n1_internal=num_x_params,
        n2_internal=num_y_params,
        n3_internal=num_theta_params,
        domain=domain,
    )

    plot_optimized_grid_params(x_params, y_params, theta_params, domain=domain)
    plt.show()

    # Make the kripke structure; check LTL property
    x_params, y_params, theta_params = uot.extract_grid_params(
        params_opt,
        n1_internal=num_x_params,
        n2_internal=num_y_params,
        n3_internal=num_theta_params,
        domain=domain,
    )
    kripke_structure = umct.make_kripke_from_params(x_params, y_params, theta_params, allow_self_loops=False, advanced_metrics=True)

    end_cpu = time.process_time()
    cpu_time = end_cpu - start_cpu
    print(f"Model build CPU time (s): {cpu_time:.2f}")

    sat_states = umct.model_check_kripke(kripke_structure)
    sat_rate = len(sat_states)/((num_x_params+1) * (num_y_params+1) * (num_theta_params+1))
    print(f"Sat rate: {sat_rate*100.0:.2f}%")

    # Compare against ground truth safety
    # gt_reach_regions = umct.get_gt_reach_regions(domain, grid_resolution=100, verbose=True)
    # with open("unicycle_gt_reach_regions_100.pkl", "wb") as f:
    #     pkl.dump(gt_reach_regions, f)
    with open("unicycle_gt_reach_regions_100.pkl", "rb") as f:
        gt_reach_regions = pkl.load(f)
    ground_truth_reference = umct.check_ground_truth_fast(x_params, y_params, theta_params, domain, gt_reach_regions)
    checked_sat_states = set(sat_states)
    true_sat_states = {s for s, v in ground_truth_reference.items() if v == 'goal'}

    # Compute sat coverage in ground truth
    gt_sat_states = {s for s, v in gt_reach_regions.items() if v in ['goal']}
    gt_coverage = len(gt_sat_states) / len(gt_reach_regions)

    # Compute sat coverage of abstract states (% of area covered by sat states); compare to gt
    sat_coverage = umct.compute_sat_coverage(sat_states, x_params, y_params, theta_params)
    coverage_proportion = sat_coverage / gt_coverage
    print(f"Coverage proportion: {coverage_proportion:.4f}%")

    # Compute FNR
    fnr, _ = umct.false_negative_rate(true_sat_states, checked_sat_states)
    print(f"False Negative Rate (FNR): {fnr:.4f}")



    # # Make the kripke structure; check LTL property
    # x_params, y_params, theta_params = uot.extract_grid_params(
    #     params_opt,
    #     n1_internal=num_x_params,
    #     n2_internal=num_y_params,
    #     n3_internal=num_theta_params,
    #     domain=domain,
    # )
    # kripke_structure = umct.make_kripke_from_params(x_params, y_params, theta_params,
    #                                                 allow_self_loops=True, advanced_metrics=True)
    # sat_states = umct.model_check_kripke(kripke_structure)
    # sat_rate = len(sat_states)/((num_x_params+1) * (num_y_params+1) * (num_theta_params+1))
    # print(f"Sat rate: {sat_rate*100.0:.2f}%")

    # # Compare against ground truth safety
    # gt_reach_regions = umct.get_gt_reach_regions(x_domain, grid_resolution=100)
    # ground_truth_reference = umct.check_ground_truth_fast(x1_params, x2_params, x_domain, gt_reach_regions)
    # checked_sat_states = set(sat_states)
    # true_sat_states = {s for s, v in ground_truth_reference.items() if v == 'goal'}

    # # Compute sat coverage in ground truth
    # gt_sat_states = {s for s, v in gt_reach_regions.items() if v in ['goal']}
    # gt_coverage = len(gt_sat_states) / len(gt_reach_regions)

    # Compute sat coverage of abstract states (% of area covered by sat states); compare to gt
    # sat_coverage = umct.compute_sat_coverage(sat_states, x_params, y_params, theta_params)
    # print(f"Sat coverage: {sat_coverage*100:.2f}")
    # coverage_proportion = sat_coverage / gt_coverage
    # print(f"Coverage proportion: {coverage_proportion*100:.2f}%")

    # # Compute FNR
    # fnr, _ = mct.false_negative_rate(true_sat_states, checked_sat_states)
    # print(f"False Negative Rate (FNR): {fnr:.4f}")






    # # Visualization (3D voxel plot of state cubes)
    # # Tip: set plot_unsat=False if rendering is too slow.
    # umct.plot_sat_voxels(
    #     x_params_ends,
    #     y_params_ends,
    #     theta_params_ends,
    #     sat_states,
    #     plot_unsat=False,
    # )


    # # Single trajectory example
    # # Environment details (you can add more obstacles by extending these arrays)
    # obs_center = np.array([20.0, 30.0])
    # obs_radius = 5.0
    # goal_center = np.array([40.0, 20.0])
    # goal_radius = 5.0  # only used for termination check

    # obstacle_centers = np.array([obs_center])
    # obstacle_radii = np.array([obs_radius])

    # # Initial state: x, y, theta
    # state = np.array([15.0, 40.0, 0.0])

    # trajectory = simulate_trajectory(
    #     state,
    #     steps=250,
    #     goal_center=goal_center,
    #     goal_radius=goal_radius,
    # )

    # trajectory = np.array(trajectory)

    # # Plotting the trajectory and environment
    # plt.figure()
    # plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=2, label='Trajectory')
    # obs_circle = plt.Circle(obs_center, obs_radius, color='r', alpha=0.5, label='Obstacle')
    # goal_circle = plt.Circle(goal_center, goal_radius, color='g', alpha=0.35, label='Goal')
    # plt.gca().add_artist(obs_circle)
    # plt.gca().add_artist(goal_circle)
    # plt.title("Unicycle Model Trajectory")
    # plt.xlabel("X position")
    # plt.ylabel("Y position")
    # plt.axis('equal')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

# def wrap_to_pi(angle):
#     return (angle + np.pi) % (2 * np.pi) - np.pi

# def unicycle_dynamics(state, control, control_bound = np.pi/4):

#     # Unicycle model parameters
#     delta_t = 0.1
#     velocity = 5
#     pose_x, pose_y, theta = state

#     # Apply control bounds
#     control = np.clip(control, -control_bound, control_bound) # heading rate of change

#     # Update the state
#     next_pose_x = pose_x + (delta_t * velocity * np.cos(theta))
#     next_pose_y = pose_y + (delta_t * velocity * np.sin(theta))
#     next_theta = theta + (delta_t * control)
#     next_theta = wrap_to_pi(next_theta) # normalize angle to [-pi, pi]

#     return np.array([next_pose_x, next_pose_y, next_theta])


# def state_controller(
#     state,
#     *,
#     goal_center,
#     obstacle_centers,
#     obstacle_radii,
#     # gains / shaping
#     k_goal=1.0,
#     k_rep=8.0,
#     alpha=0.6,
#     # heading control
#     k_theta=2.5,
#     omega_max=np.pi/4,
#     # numerical smoothing
#     eps=1e-6,
# ):
#     """
#     Deterministic smooth controller for Dubins/unicycle:
#       1) Build desired planar direction v(p) = v_att + v_rep
#       2) Convert to desired heading theta_d
#       3) Apply smooth saturated turn rate: omega = omega_max * tanh(k_theta * e_theta)
#     """

#     px, py, theta = state
#     p = np.array([px, py], dtype=float)

#     # Attractive component (toward goal)
#     v_att = k_goal * (goal_center - p)

#     # Repulsive component (sum over discs)
#     v_rep = np.zeros(2, dtype=float)
#     for c, r in zip(obstacle_centers, obstacle_radii):
#         diff = p - c
#         dist = np.sqrt(diff[0]**2 + diff[1]**2 + eps)  # smoothed distance to center
#         clearance = dist - r  # signed clearance (positive outside)

#         # Smooth activation: stronger when near obstacle, decays with clearance
#         # exp(-alpha * clearance) grows as you get closer (clearance -> 0+)
#         w = np.exp(-alpha * clearance)

#         # Direction away from obstacle (unit-ish) with additional smoothing in denom
#         # Using dist^3 in denom makes it fall off with distance and avoids singularity.
#         v_rep += k_rep * w * diff / (dist**3 + eps)

#     v = v_att + v_rep

#     # If v is (near) zero, define a deterministic fallback
#     v_norm = np.linalg.norm(v)
#     if v_norm < 1e-9:
#         return 0.0

#     # Desired heading from vector field
#     theta_d = np.arctan2(v[1], v[0])

#     # Heading error (wrapped)
#     e_theta = wrap_to_pi(theta_d - theta)

#     # Smooth saturated turn-rate command
#     omega = omega_max * np.tanh(k_theta * e_theta)
#     return float(omega)


# # Unicylce system with controller in the loop
# def cl_unicycle_dynamics(state):

#     obs_center = np.array([25.0, 25.0])
#     obs_radius = 5.0
#     goal_center = np.array([40.0, 20.0])
#     goal_radius = 8.0

#     control_input = state_controller(
#             state,
#             goal_center=goal_center,
#             obstacle_centers=np.array([obs_center]),
#             obstacle_radii=np.array([obs_radius]),
#             k_goal=0.3,
#             k_rep=300.0,
#             alpha=0.1,
#             k_theta=2.0,
#             omega_max=np.pi/4,
#         )
#     next_state = unicycle_dynamics(state, control_input)
#     return next_state


# def simulate_trajectory(x0, *, steps=250, goal_center=None, goal_radius=None):
#     """Roll out the closed-loop unicycle from x0. Returns array [T, 3]."""
#     state = np.array(x0, dtype=float)
#     traj = [state.copy()]
#     for _ in range(steps):
#         state = cl_unicycle_dynamics(state)
#         traj.append(state.copy())

#         if (goal_center is not None) and (goal_radius is not None):
#             if np.linalg.norm(state[:2] - goal_center) <= goal_radius:
#                 break
#     return np.array(traj)s