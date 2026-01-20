# Libraries
import numpy as np
import matplotlib.pyplot as plt
import uni_model_checking_tools as umct

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
#     return np.array(traj)


if __name__ == "__main__":

    # Multiple trajectory example

    # Environment details
    obs_center = np.array([25.0, 25.0])
    obs_radius = 5.0
    goal_center = np.array([40.0, 20.0])
    goal_radius = 8.0

    obstacle_centers = np.array([obs_center])
    obstacle_radii = np.array([obs_radius])

    # Batch simulate
    N = 100
    steps = 250
    theta0 = np.pi # fixed initial heading

    rng = np.random.default_rng(0)
    x0s = rng.uniform([0.0, 0.0], [20.0, 40.0], size=(N, 2))

    trajectories = []
    for i in range(N):
        x0 = np.array([x0s[i, 0], x0s[i, 1], theta0], dtype=float)
        traj = umct.simulate_trajectory(
            x0,
            steps=steps,
            goal_center=goal_center,
            goal_radius=goal_radius,
        )
        trajectories.append(traj)

    # Plot trajectories and environment
    plt.figure()
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], linewidth=1.0, alpha=0.35)
        plt.plot(traj[0, 0], traj[0, 1], marker='.', color='k', markersize=3, alpha=0.6)

    obs_circle = plt.Circle(obs_center, obs_radius, color='r', alpha=0.5, label='Obstacle')
    goal_circle = plt.Circle(goal_center, goal_radius, color='g', alpha=0.35, label='Goal')
    plt.gca().add_artist(obs_circle)
    plt.gca().add_artist(goal_circle)

    plt.title(f"Dubins trajectories w/obstacle and goal")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # State space domain
    x_min, x_max = 0.0, 50.0
    y_min, y_max = 0.0, 40.0
    theta_min, theta_max = -np.pi, np.pi

    # Example grid parameters
    num_x_params = 80
    num_y_params = 80
    num_theta_params = 40
    x_params_ends = np.linspace(x_min, x_max, num_x_params + 2)  # include boundaries
    y_params_ends = np.linspace(y_min, y_max, num_y_params + 2)
    theta_params_ends = np.linspace(theta_min, theta_max, num_theta_params + 2)

    # Make the kripke structure; check LTL property
    kripke_structure = umct.make_kripke_from_params(x_params_ends, y_params_ends, theta_params_ends,
                                                    allow_self_loops=True, advanced_metrics=True)
    sat_states = umct.model_check_kripke(kripke_structure)
    sat_rate = len(sat_states)/((num_x_params+1) * (num_y_params+1) * (num_theta_params+1))
    print(f"Sat rate: {sat_rate*100.0:.2f}%")

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

