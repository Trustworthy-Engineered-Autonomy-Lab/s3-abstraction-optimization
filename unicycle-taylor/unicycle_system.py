# =====================================================================
# Description: closed-loop unicycle pipeline
# =====================================================================

# =====================================================================
# Libraries 
# =====================================================================

import numpy as np
import jax
import jax.numpy as jnp


# =====================================================================
# End-to-end unicycle system (numpy)
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
    pose_x = state[..., 0]
    pose_y = state[..., 1]
    theta = state[..., 2]

    # Apply control bounds (heading rate of change)
    control = np.clip(control, -max_control, max_control)

    # Update the state
    next_pose_x = pose_x + (delta_t * velocity * np.cos(theta))
    next_pose_y = pose_y + (delta_t * velocity * np.sin(theta))
    next_theta = wrap_to_pi(theta + (delta_t * control))  # normalize angle to [-pi, pi]

    return np.stack([next_pose_x, next_pose_y, next_theta], axis=-1)


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
    px = state[..., 0]
    py = state[..., 1]
    theta = state[..., 2]
    p = np.stack([px, py], axis=-1)

    # Attractive component
    v_att = k_goal * (goal_center - p)

    # Repulsive component
    diff = p - obs_center
    dist = np.sqrt(np.sum(diff**2, axis=-1) + eps)
    clearance = dist - obs_radius
    w = np.exp(-alpha * clearance)
    denom = (dist**3 + eps)
    v_rep = k_rep * w[..., None] * diff / denom[..., None]

    # Compute desire heading angle and smooth control input
    v = v_att + v_rep
    theta_d = np.arctan2(v[..., 1], v[..., 0])
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
# Helper methods (numpy)
# =====================================================================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# =====================================================================
# End-to-end unicycle system (jax.numpy)
# =====================================================================

def unicycle_plant_jax(state, control, control_bound = np.pi/4):
    """
    JAX-configured open-loop unicycle dynamics.
    """

    state = jnp.asarray(state)
    control = jnp.asarray(control)

    # Unicycle model parameters
    delta_t = jnp.asarray(0.5, dtype=state.dtype)
    velocity = jnp.asarray(5.0, dtype=state.dtype)
    control_bound = jnp.asarray(control_bound, dtype=state.dtype)
    pose_x, pose_y, theta = state

    # Apply control bounds (heading rate of change)
    control = jnp.clip(control, -control_bound, control_bound)

    # Update the state
    next_pose_x = pose_x + (delta_t * velocity * jnp.cos(theta))
    next_pose_y = pose_y + (delta_t * velocity * jnp.sin(theta))
    next_theta = wrap_to_pi_jax(theta + (delta_t * control))  # normalize angle to [-pi, pi]

    return jnp.stack([next_pose_x, next_pose_y, next_theta])

def state_controller_jax(
    state,
    *,
    goal_center,
    obstacle_centers,
    obstacle_radii,
    k_goal=1.0,
    k_rep=8.0,
    alpha=0.6,
    k_theta=2.5,
    omega_max=np.pi/4,
    eps=1e-6,
    ):
    """
    JAX-configured deterministic & smooth controller for unicycle.
    """

    state = jnp.asarray(state)
    goal_center = jnp.asarray(goal_center, dtype=state.dtype)
    obstacle_centers = jnp.asarray(obstacle_centers, dtype=state.dtype)
    obstacle_radii = jnp.asarray(obstacle_radii, dtype=state.dtype)

    k_goal = jnp.asarray(k_goal, dtype=state.dtype)
    k_rep = jnp.asarray(k_rep, dtype=state.dtype)
    alpha = jnp.asarray(alpha, dtype=state.dtype)
    k_theta = jnp.asarray(k_theta, dtype=state.dtype)
    omega_max = jnp.asarray(omega_max, dtype=state.dtype)
    eps = jnp.asarray(eps, dtype=state.dtype)

    px, py, theta = state
    p = jnp.stack([px, py])

    # Attractive component (toward goal)
    v_att = k_goal * (goal_center - p)

    # Repulsive component (sum over discs)
    # obstacle_centers: [N, 2], obstacle_radii: [N]
    diff = p[None, :] - obstacle_centers
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + eps)  # smoothed distance to center
    clearance = dist - obstacle_radii

    # Smooth activation: stronger when near obstacle, decays with clearance
    w = jnp.exp(-alpha * clearance)

    # Direction away from obstacle with smoothing in denom
    denom = (dist**3 + eps)[:, None]
    v_rep = jnp.sum((k_rep * w)[:, None] * diff / denom, axis=0)

    v = v_att + v_rep
    v_norm = jnp.linalg.norm(v)

    def _omega_when_ok(_):
        theta_d = jnp.arctan2(v[1], v[0])
        e_theta = wrap_to_pi_jax(theta_d - theta)
        return omega_max * jnp.tanh(k_theta * e_theta)

    omega = jax.lax.cond(v_norm < jnp.asarray(1e-9, dtype=state.dtype), lambda _: jnp.asarray(0.0, dtype=state.dtype), _omega_when_ok, operand=None)
    return omega

def cl_system_jax(state):
    """
    JAX-configured closed-loop unicycle dynamics.
    """

    state = jnp.asarray(state)
    obs_center = jnp.array([25.0, 25.0], dtype=state.dtype)
    obs_radius = jnp.asarray(5.0, dtype=state.dtype)
    goal_center = jnp.array([40.0, 20.0], dtype=state.dtype)

    control_input = state_controller_jax(
        state,
        goal_center=goal_center,
        obstacle_centers=jnp.stack([obs_center], axis=0),
        obstacle_radii=jnp.stack([obs_radius], axis=0),
        k_goal=0.3,
        k_rep=300.0,
        alpha=0.1,
        k_theta=2.0,
        omega_max=jnp.pi / 4,
    )
    return unicycle_plant_jax(state, control_input)


# =====================================================================
# Helper methods (jax.numpy)
# =====================================================================

def wrap_to_pi_jax(angle):
    angle = jnp.asarray(angle)
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi