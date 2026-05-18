# =====================================================================
# Description: closed-loop unicycle pipeline
# =====================================================================

# =====================================================================
# Libraries 
# =====================================================================

import numpy as np


# =====================================================================
# End-to-end unicycle system
# =====================================================================

def unicycle_plant(
        state,
        control,
        *,
        max_control=np.pi/4):
    """
    Description: Open-loop discrete equations of motion of Unicycle
    """

    state = np.asarray(state)
    control = np.asarray(control)

    # Unicycle model parameters
    delta_t = np.asarray(0.5, dtype=state.dtype)
    velocity = np.asarray(5.0, dtype=state.dtype)
    max_control = np.asarray(max_control, dtype=state.dtype)
    pose_x, pose_y, theta = state

    # Apply control bounds (heading rate of change)
    control = np.clip(control, -max_control, max_control)

    # Update the state
    next_pose_x = pose_x + (delta_t * velocity * np.cos(theta))
    next_pose_y = pose_y + (delta_t * velocity * np.sin(theta))
    next_theta = wrap_to_pi(theta + (delta_t * control))  # normalize angle to [-pi, pi]

    return np.stack([next_pose_x, next_pose_y, next_theta])


def state_controller(
    state,
    *,
    goal_center,
    obs_center,
    obs_radius,
    k_goal=1.0,
    k_rep=8.0,
    alpha=0.6,
    k_theta=2.5,
    omega_max=np.pi/4,
    eps=1e-6,
):
    """
    Deterministic repulsion-attraction smooth controller for Dubins/unicycle
    """

    # Format all inputs as JAX arrays
    state = np.asarray(state)
    goal_center = np.asarray(goal_center, dtype=state.dtype)
    obs_center = np.asarray(obs_center, dtype=state.dtype)
    obs_radius = np.asarray(obs_radius, dtype=state.dtype)
    k_goal = np.asarray(k_goal, dtype=state.dtype)
    k_rep = np.asarray(k_rep, dtype=state.dtype)
    alpha = np.asarray(alpha, dtype=state.dtype)
    k_theta = np.asarray(k_theta, dtype=state.dtype)
    omega_max = np.asarray(omega_max, dtype=state.dtype)
    eps = np.asarray(eps, dtype=state.dtype)
    px, py, theta = state
    p = np.stack([px, py])

    # Attractive component
    v_att = k_goal * (goal_center - p)

    # Repulsive component
    diff = p - obs_center
    dist = np.sqrt(np.sum(diff**2, axis=-1) + eps)
    clearance = dist - obs_radius
    w = np.exp(-alpha * clearance)
    denom = (dist**3 + eps)
    v_rep = np.sum(k_rep * w * diff / denom, axis=0)

    # Compute desire heading angle and smooth control input
    v = v_att + v_rep
    theta_d = np.arctan2(v[1], v[0])
    e_theta = wrap_to_pi(theta_d - theta)
    omega = omega_max * np.tanh(k_theta * e_theta)

    return omega


def cl_system(
        state,
        *,
        obs_center = np.array([25.0, 25.0], dtype=np.float32),
        obs_radius = np.array(5.0, dtype=np.float32),
        goal_center = np.array([40.0, 20.0], dtype=np.float32)):
    """
    Closed-loop wrapper for the plant with the state controller
    """

    state = np.asarray(state)

    control_input = state_controller(
        state,
        goal_center=goal_center,
        obs_center=obs_center,
        obs_radius=obs_radius,
        k_goal=0.3,
        k_rep=300.0,
        alpha=0.1,
        k_theta=2.0,
        omega_max=np.pi / 4,
    )
    return unicycle_plant(state, control_input)


# =====================================================================
# Helper methods
# =====================================================================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi